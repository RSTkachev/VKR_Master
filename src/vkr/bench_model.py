"""Two-stage retrieval benchmark: dual-encoder stage followed by optional VLM reranking.

Writes JSON metrics; per-query ranks can be saved as ``.npz`` via ``--save-per-query-ranks``.
"""

import time
import argparse
import json
import gc
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from tqdm.auto import tqdm

from vkr.utils import set_seed, choose_device, load_model_bundle
from vkr.preprocessing import Example
from vkr.splits import load_split
from vkr.encode import encode_texts, encode_images
from vkr.ann import ann_ranks_for_aligned_pairs
from vkr.scoring import retrieval_metrics_from_ranks
from vkr.reranking import (
    build_listwise_prompt,
    build_pairs_for_rerank,
    build_prompt,
    parse_listwise_ranking,
)


def score_pair_logits(model, processor, image_path, text, device):
    """Score image-text pairs by the log-odds of ``yes`` vs ``no`` next-token logits.

    Args:
        model: causal VLM used for scoring.
        processor: processor matching ``model``.
        image_path: image paths, one per pair.
        text: texts, one per pair.
        device: torch device for inputs.

    Return: per-pair scores; higher means stronger match.
    """
    messages = [build_prompt(i, t) for i, t in zip(image_path, text)]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding_side": "left",
            "padding": True,
            "truncation": True,
        },
        enable_thinking=False,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    tokenizer = processor.tokenizer

    yes_strings = [" yes", "yes", " Yes", "Yes", " YES", "YES"]
    no_strings = [" no", "no", " No", "No", " NO", "NO"]

    yes_tokens = set()
    for s in yes_strings:
        yes_tokens.update(tokenizer.encode(s, add_special_tokens=False))

    no_tokens = set()
    for s in no_strings:
        no_tokens.update(tokenizer.encode(s, add_special_tokens=False))

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[:, -1, :]

    scores = []

    for b in range(last_logits.shape[0]):
        logits_b = last_logits[b]

        yes_logit = torch.logsumexp(logits_b[list(yes_tokens)], dim=0)
        no_logit = torch.logsumexp(logits_b[list(no_tokens)], dim=0)

        score = (yes_logit - no_logit).item()

        scores.append(score)

    return scores


def load_qwen(model_name, device, lora_path: Optional[str] = None):
    """Load a Qwen-family VLM with optional LoRA adapter merged into the base weights.

    Args:
        model_name: Hugging Face model id or local path.
        device: torch device.
        lora_path: directory containing a trained LoRA adapter; ``None`` skips it.

    Return: tuple ``(model, processor)`` with the adapter merged when supplied.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    if lora_path is not None:
        print(f"Loading reranker LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model.to(device)

    return model, processor


def rerank_listwise(
    model,
    processor,
    image_path: str,
    texts: List[str],
    device: str,
    max_new_tokens: int = 64,
) -> List[float]:
    """Score K candidates with a single listwise generation call.

    Args:
        model: VLM used to produce a ranking.
        processor: processor matching ``model``.
        image_path: path to the query image.
        texts: candidate descriptions in original (input) order.
        device: torch device.
        max_new_tokens: generation cap.

    Return: score per candidate aligned with ``texts``; higher is better.
        ``argsort(scores)[::-1]`` recovers the model's ranking.
    """
    k = len(texts)
    messages = build_listwise_prompt(image_path, texts)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {key: v.to(device) for key, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id,
        )

    generated = output_ids[0, input_len:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)
    parsed = parse_listwise_ranking(response, k)

    scores = [0.0] * k
    for rank_position, candidate_idx_1b in enumerate(parsed):
        scores[candidate_idx_1b - 1] = float(k - rank_position)
    return scores


def evaluate_model(
    model_name: str,
    examples: Sequence[Example],
    image_batch_size: int,
    text_batch_size: int,
    device: str,
    top_k: int = 50,
    second_level_model: Optional[str] = None,
    lora_checkpoint: Optional[str] = None,
    second_lora_checkpoint: Optional[str] = None,
    rerank_chunk_size: int = 10,
    rerank_mode: str = "pointwise",
):
    """Evaluate the retrieval pipeline and collect per-query ranks for downstream analysis.

    Args:
        model_name: base retriever model id.
        examples: queries and target pool (one ground-truth pair per index).
        image_batch_size: batch size for image encoding.
        text_batch_size: batch size for text encoding.
        device: torch device.
        top_k: candidates retrieved per query by stage 1.
        second_level_model: Hugging Face id of the VLM reranker; ``None`` disables stage 2.
        lora_checkpoint: LoRA adapter directory for the retriever (merged before inference).
        second_lora_checkpoint: LoRA adapter directory for the reranker.
        rerank_chunk_size: pairs scored per VLM forward pass in pointwise mode.
        rerank_mode: ``"pointwise"`` or ``"listwise"``.

    Return: dict with metrics, latencies, memory usage and a ``__per_query__`` entry
        holding per-query ranks and item ids for optional ``.npz`` persistence.
    """
    if "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model, processor = load_model_bundle(model_name, device=device)

    if lora_checkpoint is not None:
        print(f"Loading LoRA adapter from {lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint)
        model = model.merge_and_unload()
        model.to(device)

    model.eval()

    texts = [example.text for example in examples]
    image_paths = [example.image_file for example in examples]

    if "cuda" in device:
        torch.cuda.synchronize()
    start_text = time.perf_counter()

    text_embeds = encode_texts(
        model,
        processor,
        texts,
        batch_size=text_batch_size,
        device=device,
        model_name=model_name,
    )

    if "cuda" in device:
        torch.cuda.synchronize()
    text_time = time.perf_counter() - start_text

    if "cuda" in device:
        torch.cuda.synchronize()
    start_image = time.perf_counter()

    image_embeds = encode_images(
        model,
        processor,
        image_paths,
        batch_size=image_batch_size,
        device=device,
        model_name=model_name,
    )

    if "cuda" in device:
        torch.cuda.synchronize()
    image_time = time.perf_counter() - start_image

    i2t_ranks, i2t_retrieved = ann_ranks_for_aligned_pairs(
        image_embeds, text_embeds, top_k
    )

    mem_mb = 0
    if "cuda" in device:
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    results = {
        "model_name": model_name,
        "lora_checkpoint": lora_checkpoint,
        "top_k_first_stage": top_k,
        "num_examples": len(examples),
        "image_to_text": retrieval_metrics_from_ranks(i2t_ranks, [top_k]),
        "text_to_embedding_time": text_time,
        "image_to_embedding_time": image_time,
        "avg_text_latency_ms": (text_time / len(examples)) * 1000,
        "avg_image_latency_ms": (image_time / len(examples)) * 1000,
        "memory_consumption_mb": round(mem_mb, 2),
        "__per_query__": {
            "item_ids": [ex.item_id for ex in examples],
            "retriever_ranks": np.array(i2t_ranks, dtype=np.int32),
        },
    }

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    if second_level_model is not None:
        if rerank_mode == "listwise" and top_k > 10:
            print(
                f"Listwise reranking with top_k={top_k}: prompts will be "
                f"long and may degrade quality."
            )
        if rerank_mode not in ("pointwise", "listwise"):
            raise ValueError(
                f"Unknown rerank_mode: {rerank_mode!r}. "
                f"Expected 'pointwise' or 'listwise'."
            )

        second_model, second_processor = load_qwen(
            second_level_model,
            device=device,
            lora_path=second_lora_checkpoint,
        )
        second_model.eval()

        i2t_pairs = build_pairs_for_rerank(image_paths, texts, "i2t", i2t_retrieved)

        i2t_ranks_second = []
        desc = f"Second level i2t ({rerank_mode})"
        for pair in tqdm(i2t_pairs, leave=True, desc=desc):
            correct_index = None

            if rerank_mode == "pointwise":
                second_level_scores = []
                for start_index in range(0, len(pair), rerank_chunk_size):
                    end_index = min(start_index + rerank_chunk_size, len(pair))
                    chunk_images = []
                    chunk_texts = []
                    for index in range(start_index, end_index):
                        chunk_images.append(pair[index]["image"])
                        chunk_texts.append(pair[index]["text"])
                        if pair[index]["label"] == True:
                            correct_index = index
                    second_level_scores.extend(
                        score_pair_logits(
                            second_model,
                            second_processor,
                            chunk_images,
                            chunk_texts,
                            device,
                        )
                    )
            else:
                pair_image = pair[0]["image"]
                pair_texts = [p["text"] for p in pair]
                for index, p in enumerate(pair):
                    if p["label"] == True:
                        correct_index = index
                second_level_scores = rerank_listwise(
                    second_model,
                    second_processor,
                    pair_image,
                    pair_texts,
                    device,
                )

            if correct_index is not None:
                ranking = np.argsort(second_level_scores)[::-1]
                rank_position = np.where(ranking == correct_index)[0][0] + 1
            else:
                rank_position = top_k + 1
            i2t_ranks_second.append(rank_position)
        results["image_to_text_generative_vlm"] = retrieval_metrics_from_ranks(
            np.array(i2t_ranks_second), [top_k]
        )
        results["__per_query__"]["reranker_ranks"] = np.array(
            i2t_ranks_second,
            dtype=np.int32,
        )

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(description="Stage 1 zero-shot evaluation on ABO")
    parser.add_argument(
        "--project-path", type=str, default="./", help="Project root path"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory for outputs"
    )
    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        help="Split identifier created via `python -m vkr.splits create`.",
    )
    parser.add_argument(
        "--split-part",
        type=str,
        default="test",
        choices=["test", "val", "train_siglip", "train_reranker"],
        help="Which part of the split to evaluate on. Default: test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", help="auto, cuda, cpu, cuda:0, ..."
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument(
        "--model",
        type=str,
        default="siglip",
        help="Short identifier used in the output filename.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/metaclip-b16-400m",
        help="Hugging Face model id of the base retriever.",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help="LoRA adapter directory for the retriever.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of candidates returned by stage-1 retrieval.",
    )
    parser.add_argument(
        "--second_model_id",
        type=str,
        default=None,
        help="VLM reranker id; ``None`` disables stage 2.",
    )
    parser.add_argument(
        "--second-lora-checkpoint",
        type=str,
        default=None,
        help="LoRA adapter directory for the reranker VLM.",
    )
    parser.add_argument(
        "--rerank-chunk-size",
        type=int,
        default=10,
        help="Pairs scored per VLM forward pass (pointwise mode).",
    )
    parser.add_argument(
        "--rerank-mode",
        type=str,
        default="pointwise",
        choices=["pointwise", "listwise"],
        help="Reranker scoring mode.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional tag appended to the output filename.",
    )
    parser.add_argument(
        "--save-per-query-ranks",
        action="store_true",
        help="Persist per-query ranks as a NumPy .npz next to the metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: run the benchmark, write JSON metrics and (optionally) per-query ranks."""
    args = parse_args()
    set_seed(args.seed)

    project_path = Path(args.project_path)
    output_dir = (
        Path(args.output_dir) if args.output_dir else project_path / "eval_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    if args.split_part == "test":
        print(
            "\nRunning on the test split. Use this for final numbers only; "
            "do not iterate on hyperparameters based on these results.\n"
        )

    split_data = load_split(
        project_path=project_path,
        split_name=args.split_name,
        parts=[args.split_part],
    )
    examples = split_data[args.split_part]

    examples = [ex for ex in examples if ex.is_main_image]

    print(
        f"Loaded {len(examples)} examples from split '{args.split_name}' "
        f"part '{args.split_part}'."
    )

    all_results: Dict[str, Any] = {
        "config": {
            "model": args.model_id,
            "lora_checkpoint": args.lora_checkpoint,
            "split_name": args.split_name,
            "split_part": args.split_part,
            "top_k": args.top_k,
            "seed": args.seed,
            "device": device,
            "image_batch_size": args.image_batch_size,
            "text_batch_size": args.text_batch_size,
            "second_model_id": args.second_model_id,
            "second_lora_checkpoint": args.second_lora_checkpoint,
            "rerank_chunk_size": args.rerank_chunk_size,
            "rerank_mode": args.rerank_mode,
        },
    }

    print(f"Evaluating {args.model}")
    model_result = evaluate_model(
        model_name=args.model_id,
        examples=examples,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        device=device,
        top_k=args.top_k,
        second_level_model=args.second_model_id,
        lora_checkpoint=args.lora_checkpoint,
        second_lora_checkpoint=args.second_lora_checkpoint,
        rerank_chunk_size=args.rerank_chunk_size,
        rerank_mode=args.rerank_mode,
    )
    all_results["results"] = model_result

    name_parts = [args.model, args.split_name, args.split_part]
    if args.lora_checkpoint:
        name_parts.append("lora")
    if args.second_model_id:
        rerank_tag = f"rerank-{args.rerank_mode}"
        if args.second_lora_checkpoint:
            rerank_tag += "-ft"
        name_parts.append(rerank_tag)
    if args.run_name:
        name_parts.append(args.run_name)
    base_filename = "_".join(name_parts)
    out_path = output_dir / (base_filename + ".json")

    per_query = model_result.pop("__per_query__", None)
    if args.save_per_query_ranks and per_query is not None:
        npz_path = output_dir / (base_filename + ".npz")
        ranks = per_query.get("reranker_ranks", per_query["retriever_ranks"])
        np.savez(
            npz_path,
            ranks=ranks,
            item_ids=np.array(per_query["item_ids"]),
            retriever_ranks=per_query["retriever_ranks"],
        )
        print(f"Saved per-query ranks to: {npz_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
