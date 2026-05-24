"""LoRA fine-tuning of a VLM as a listwise reranker over candidates mined by a SigLIP retriever."""

import argparse
import gc
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv(".env", override=True)

import mlflow
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from vkr.ann import ann_ranks_for_aligned_pairs
from vkr.encode import encode_images, encode_texts
from vkr.reranking import build_listwise_prompt, parse_listwise_ranking
from vkr.splits import load_split
from vkr.utils import choose_device, load_model_bundle, set_seed


@dataclass
class ListwiseExample:
    """Training example consisting of an image and retriever-ordered candidate texts.

    Attributes:
        image_path: path to the query image on disk.
        candidate_texts_retriever_order: candidate descriptions ordered by retriever
            similarity; the ground-truth candidate is at index 0 and the remaining
            items follow in decreasing similarity.
    """

    image_path: str
    candidate_texts_retriever_order: List[str]


def build_listwise_pairs(
    image_paths: List[str],
    texts: List[str],
    topk_indices: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
) -> List[ListwiseExample]:
    """Build one :class:`ListwiseExample` per query from stage-1 top-K indices.

    Args:
        image_paths: image path per query.
        texts: text per query; the query's own text is its positive candidate.
        topk_indices: stage-1 top-K candidate indices per query.
        top_k: total candidates per example (including the positive).
        rng: generator used to draw fillers when the positive was missed by stage 1.

    Return: list with one :class:`ListwiseExample` per query.
    """
    examples: List[ListwiseExample] = []
    n = len(image_paths)
    for i in range(n):
        retrieved = topk_indices[i].tolist()
        non_self = [j for j in retrieved if j != i][: top_k - 1]
        if len(non_self) < top_k - 1:
            extras_pool = [j for j in range(n) if j != i and j not in non_self]
            rng.shuffle(extras_pool)
            non_self.extend(extras_pool[: (top_k - 1) - len(non_self)])

        ordered_indices = [i] + non_self
        examples.append(
            ListwiseExample(
                image_path=image_paths[i],
                candidate_texts_retriever_order=[texts[j] for j in ordered_indices],
            )
        )
    return examples


def encode_split_for_retrieval(
    siglip_model_id: str,
    siglip_lora_path: Optional[str],
    examples,
    image_batch_size: int,
    text_batch_size: int,
    device: str,
):
    """Encode the given examples with SigLIP and an optional LoRA adapter.

    Args:
        siglip_model_id: Hugging Face model id of the base SigLIP encoder.
        siglip_lora_path: optional LoRA adapter directory; merged before encoding.
        examples: examples to encode (only main-image entries are expected).
        image_batch_size: batch size for image encoding.
        text_batch_size: batch size for text encoding.
        device: torch device.

    Return: tuple ``(image_paths, texts, text_embeds, image_embeds)``.
    """
    model, processor = load_model_bundle(siglip_model_id, device=device)
    if siglip_lora_path:
        print(f"Loading SigLIP LoRA from {siglip_lora_path}")
        model = PeftModel.from_pretrained(model, siglip_lora_path)
        model = model.merge_and_unload()
        model.to(device)
    model.eval()

    texts = [ex.text for ex in examples]
    image_paths = [ex.image_file for ex in examples]

    text_embeds = encode_texts(
        model,
        processor,
        texts,
        batch_size=text_batch_size,
        device=device,
        model_name=siglip_model_id,
    )
    image_embeds = encode_images(
        model,
        processor,
        image_paths,
        batch_size=image_batch_size,
        device=device,
        model_name=siglip_model_id,
    )

    del model, processor
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return image_paths, texts, text_embeds, image_embeds


def generate_or_load_pairs(
    split_part_examples,
    cache_path: Path,
    args,
    device: str,
) -> List[ListwiseExample]:
    """Read the cached listwise pairs or build them from scratch.

    Args:
        split_part_examples: source examples (main-image only).
        cache_path: location of the JSON cache file.
        args: parsed CLI arguments providing retriever id, top-K, batch sizes and
            the ``--regenerate-pairs`` flag.
        device: torch device for retriever encoding.

    Return: cached or freshly generated pairs.
    """
    if cache_path.exists() and not args.regenerate_pairs:
        print(f"Loading cached pairs from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        return [
            ListwiseExample(
                image_path=item["image_path"],
                candidate_texts_retriever_order=item["candidate_texts_retriever_order"],
            )
            for item in data
        ]

    print(f"Generating pairs (cache miss or regenerate flag)...")
    image_paths, texts, text_embeds, image_embeds = encode_split_for_retrieval(
        args.siglip_model_id,
        args.siglip_lora,
        split_part_examples,
        args.image_batch_size,
        args.text_batch_size,
        device,
    )

    print(f"Running top-{args.top_k} retrieval over {len(texts)} candidates...")
    _, i2t_retrieved = ann_ranks_for_aligned_pairs(
        image_embeds,
        text_embeds,
        args.top_k,
    )

    rng = np.random.default_rng(args.seed)
    pairs = build_listwise_pairs(
        image_paths,
        texts,
        i2t_retrieved,
        args.top_k,
        rng,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            [
                {
                    "image_path": ex.image_path,
                    "candidate_texts_retriever_order": ex.candidate_texts_retriever_order,
                }
                for ex in pairs
            ],
            f,
        )
    print(f"Cached {len(pairs)} pairs to {cache_path}")
    return pairs


class ListwiseDataset(Dataset):
    """Dataset producing prompt/target pairs with a per-epoch prompt-order shuffle.

    Each ``__getitem__`` call deterministically reshuffles the candidate order based on
    ``(base_seed, epoch, idx)`` and emits the target ranking string that lists prompt
    positions in retriever order (positive first).
    """

    def __init__(self, examples: List[ListwiseExample], base_seed: int = 0):
        """Store examples and initialise the per-epoch seed component.

        Args:
            examples: source examples produced by :func:`build_listwise_pairs`.
            base_seed: seed component combined with the current epoch for shuffles.
        """
        self.examples = examples
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Set the epoch index used for per-(epoch, idx) shuffling.

        Args:
            epoch: current epoch (zero-based).

        Return: None.
        """
        self.epoch = epoch

    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int):
        """Return ``(messages, target_str, positive_index_in_prompt)`` for one example.

        Args:
            idx: example index.

        Return: tuple ``(messages, target_str, positive_index_in_prompt)``.
        """
        ex = self.examples[idx]
        ordered = ex.candidate_texts_retriever_order
        k = len(ordered)

        rng = random.Random(f"{self.base_seed}:{self.epoch}:{idx}")
        perm = list(range(k))
        rng.shuffle(perm)

        candidate_texts_in_prompt = [ordered[perm[p]] for p in range(k)]

        inv = [0] * k
        for p, t in enumerate(perm):
            inv[t] = p
        target_positions_1b = [inv[t] + 1 for t in range(k)]
        target_str = ", ".join(str(p) for p in target_positions_1b)

        positive_index_in_prompt = inv[0]
        messages = build_listwise_prompt(ex.image_path, candidate_texts_in_prompt)
        return messages, target_str, positive_index_in_prompt


def make_collate_fn(processor, device):
    """Build a collate function that tokenises prompts and appends teacher-forced targets.

    The collator left-pads ``input_ids``, ``attention_mask`` and ``labels`` so that
    target tokens align across the batch and the loss covers the full target sequence.

    Args:
        processor: VLM processor providing the tokenizer.
        device: unused placeholder for symmetry with downstream APIs.

    Return: collate function producing a batch dict ready to move to the device.
    """
    tokenizer = processor.tokenizer

    def _collate(batch):
        all_input_ids: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_attn: List[torch.Tensor] = []
        all_mm_tt: List[torch.Tensor] = []
        all_pixel_values = []
        all_image_grids = []

        for messages, target_str, _positive_index_in_prompt in batch:
            prompt_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            prompt_ids = prompt_inputs["input_ids"][0]
            prompt_attn = prompt_inputs["attention_mask"][0]

            target_ids = tokenizer(
                target_str,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]

            full_ids = torch.cat([prompt_ids, target_ids], dim=0)
            full_attn = torch.cat([prompt_attn, torch.ones_like(target_ids)], dim=0)

            labels = torch.full_like(full_ids, fill_value=-100)
            target_start = len(prompt_ids)
            labels[target_start:] = full_ids[target_start:]

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attn.append(full_attn)

            if "mm_token_type_ids" in prompt_inputs:
                prompt_mm_tt = prompt_inputs["mm_token_type_ids"][0]
                target_mm_tt = torch.zeros_like(target_ids)
                all_mm_tt.append(torch.cat([prompt_mm_tt, target_mm_tt], dim=0))

            if "pixel_values" in prompt_inputs:
                all_pixel_values.append(prompt_inputs["pixel_values"])
            if "image_grid_thw" in prompt_inputs:
                all_image_grids.append(prompt_inputs["image_grid_thw"])

        max_len = max(t.shape[0] for t in all_input_ids)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        padded_ids = torch.full(
            (len(batch), max_len),
            pad_id,
            dtype=all_input_ids[0].dtype,
        )
        padded_labels = torch.full(
            (len(batch), max_len),
            -100,
            dtype=all_labels[0].dtype,
        )
        padded_attn = torch.zeros(
            (len(batch), max_len),
            dtype=all_attn[0].dtype,
        )
        padded_mm_tt = None
        if all_mm_tt:
            padded_mm_tt = torch.zeros(
                (len(batch), max_len),
                dtype=all_mm_tt[0].dtype,
            )

        for i, (ids, lbl, attn) in enumerate(zip(all_input_ids, all_labels, all_attn)):
            n = ids.shape[0]
            padded_ids[i, max_len - n :] = ids
            padded_labels[i, max_len - n :] = lbl
            padded_attn[i, max_len - n :] = attn
            if padded_mm_tt is not None:
                padded_mm_tt[i, max_len - n :] = all_mm_tt[i]

        out: Dict[str, torch.Tensor] = {
            "input_ids": padded_ids,
            "attention_mask": padded_attn,
            "labels": padded_labels,
        }
        if padded_mm_tt is not None:
            out["mm_token_type_ids"] = padded_mm_tt
        if all_pixel_values:
            out["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grids:
            out["image_grid_thw"] = torch.cat(all_image_grids, dim=0)
        return out

    return _collate


@torch.no_grad()
def evaluate_listwise(
    model,
    processor,
    val_examples: List[ListwiseExample],
    device: str,
    eval_seed: int = 0,
    max_new_tokens: int = 64,
    max_eval_examples: Optional[int] = None,
) -> Dict[str, float]:
    """Run greedy listwise generation on the validation set and collect metrics.

    Args:
        model: VLM under evaluation.
        processor: matching processor.
        val_examples: validation examples.
        device: torch device.
        eval_seed: seed for the per-example prompt-order shuffle.
        max_new_tokens: generation cap.
        max_eval_examples: optional cap on the number of evaluated examples.

    Return: dict with ``r_at_1`` and ``full_ranking_rate``.
    """
    model.eval()
    if max_eval_examples is not None:
        val_examples = val_examples[:max_eval_examples]

    correct = 0
    total = 0
    n_full_rankings = 0

    for ex_idx, ex in enumerate(tqdm(val_examples, desc="Validating", leave=False)):
        ordered = ex.candidate_texts_retriever_order
        k = len(ordered)

        rng = random.Random(f"{eval_seed}:{ex_idx}")
        perm = list(range(k))
        rng.shuffle(perm)
        candidate_texts_in_prompt = [ordered[perm[p]] for p in range(k)]
        positive_prompt_position = perm.index(0)

        messages = build_listwise_prompt(ex.image_path, candidate_texts_in_prompt)
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

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id,
        )
        generated = output_ids[0, input_len:]
        response = processor.tokenizer.decode(generated, skip_special_tokens=True)

        raw_numbers = [int(x) for x in re.findall(r"\d+", response)]
        seen: set = set()
        valid_numbers: List[int] = []
        for n in raw_numbers:
            if 1 <= n <= k and n not in seen:
                seen.add(n)
                valid_numbers.append(n)
        if len(valid_numbers) == k:
            n_full_rankings += 1

        ranking = parse_listwise_ranking(response, k)
        pred_prompt_pos_0b = ranking[0] - 1
        if pred_prompt_pos_0b == positive_prompt_position:
            correct += 1
        total += 1

    return {
        "r_at_1": correct / max(total, 1),
        "full_ranking_rate": n_full_rankings / max(total, 1),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the listwise fine-tuning script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-path", type=Path, default=Path("./"))
    p.add_argument("--split-name", type=str, default="abo_v4")
    p.add_argument("--train-part", type=str, default="train_reranker")
    p.add_argument("--val-part", type=str, default="val")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--siglip-model-id", type=str, default="google/siglip2-so400m-patch14-384"
    )
    p.add_argument(
        "--siglip-lora",
        type=str,
        default=None,
        help="SigLIP LoRA checkpoint.",
    )
    p.add_argument("--vlm-model-id", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=7e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-delta", type=float, default=0.005)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Per-device batch size.",
    )
    p.add_argument(
        "--image-batch-size",
        type=int,
        default=32,
        help="Batch size for SigLIP image encoding.",
    )
    p.add_argument("--text-batch-size", type=int, default=128)
    p.add_argument(
        "--max-eval-examples",
        type=int,
        default=500,
        help="Cap on validation examples per epoch.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("./checkpoints/qwen_listwise_lora")
    )
    p.add_argument(
        "--pairs-cache",
        type=Path,
        default=None,
        help="Cache directory for listwise pairs; defaults to <output-dir>/pairs.",
    )
    p.add_argument(
        "--regenerate-pairs",
        action="store_true",
        help="Regenerate pairs even if a cache exists.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--run-name", type=str, default="qwen_listwise")
    return p.parse_args()


def main():
    """Entry point for the listwise reranker fine-tuning script."""
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_cache_dir = args.pairs_cache or (args.output_dir / "pairs")
    pairs_cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading split '{args.split_name}', parts: "
        f"{args.train_part}, {args.val_part}"
    )
    data = load_split(
        args.project_path,
        args.split_name,
        parts=[args.train_part, args.val_part],
    )
    train_examples = [ex for ex in data[args.train_part] if ex.is_main_image]
    val_examples = [ex for ex in data[args.val_part] if ex.is_main_image]
    print(
        f"Train: {len(train_examples)} products, " f"Val: {len(val_examples)} products"
    )

    train_pairs_path = pairs_cache_dir / f"train_k{args.top_k}.json"
    val_pairs_path = pairs_cache_dir / f"val_k{args.top_k}.json"

    train_pairs = generate_or_load_pairs(
        train_examples,
        train_pairs_path,
        args,
        device,
    )
    val_pairs = generate_or_load_pairs(
        val_examples,
        val_pairs_path,
        args,
        device,
    )

    print(f"\nLoading VLM: {args.vlm_model_id}")
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        args.vlm_model_id,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_model_id)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    vlm_model = get_peft_model(vlm_model, lora_cfg)
    vlm_model.print_trainable_parameters()

    train_ds = ListwiseDataset(train_pairs, base_seed=args.seed)
    val_ds = ListwiseDataset(val_pairs, base_seed=args.seed + 1)

    collate_fn = make_collate_fn(vlm_processor, device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = AdamW(
        [p for p in vlm_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    total_optim_steps = max(len(train_loader) // args.grad_accum_steps, 1) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optim_steps)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("vkr")

    best_r_at_1 = -1.0
    epochs_since_improvement = 0

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "vlm_model_id": args.vlm_model_id,
                "siglip_model_id": args.siglip_model_id,
                "siglip_lora": args.siglip_lora,
                "split_name": args.split_name,
                "top_k": args.top_k,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "grad_accum_steps": args.grad_accum_steps,
                "train_batch_size": args.train_batch_size,
                "n_train_pairs": len(train_pairs),
                "n_val_pairs": len(val_pairs),
            }
        )

        global_step = 0
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            train_ds.set_epoch(epoch)

            vlm_model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            running_count = 0

            pbar = tqdm(train_loader, desc=f"Train epoch {epoch + 1}")
            for step, batch in enumerate(pbar):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                outputs = vlm_model(**batch)
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()

                running_loss += outputs.loss.item()
                running_count += 1

                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(
                    train_loader
                ):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in vlm_model.parameters() if p.requires_grad],
                        max_norm=args.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    pbar.set_postfix(
                        {
                            "loss": running_loss / max(running_count, 1),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    mlflow.log_metric(
                        "train_loss",
                        running_loss / max(running_count, 1),
                        step=global_step,
                    )

            eval_metrics = evaluate_listwise(
                vlm_model,
                vlm_processor,
                val_pairs,
                device,
                eval_seed=12345,
                max_eval_examples=args.max_eval_examples,
            )
            r_at_1 = eval_metrics["r_at_1"]
            full_rate = eval_metrics["full_ranking_rate"]
            print(
                f"Epoch {epoch + 1} val R@1: {r_at_1:.4f}, "
                f"full_ranking_rate: {full_rate:.3f}"
            )
            mlflow.log_metric("val_r_at_1", r_at_1, step=epoch + 1)
            mlflow.log_metric("val_full_ranking_rate", full_rate, step=epoch + 1)
            mlflow.log_metric(
                "epoch_train_loss",
                running_loss / max(running_count, 1),
                step=epoch + 1,
            )

            if r_at_1 > best_r_at_1 + args.min_delta:
                best_r_at_1 = r_at_1
                epochs_since_improvement = 0
                print(f"New best R@1 = {r_at_1:.4f}, saving adapter...")
                vlm_model.save_pretrained(args.output_dir)
                vlm_processor.save_pretrained(args.output_dir)
                with open(args.output_dir / "best_metric.json", "w") as f:
                    json.dump(
                        {
                            "epoch": epoch + 1,
                            "val_r_at_1": r_at_1,
                            "val_full_ranking_rate": full_rate,
                        },
                        f,
                        indent=2,
                    )
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= args.patience:
                    print(
                        f"Early stopping after {epoch + 1} epochs "
                        f"(no improvement for {epochs_since_improvement})."
                    )
                    break

        mlflow.log_metric("best_val_r_at_1", best_r_at_1)
        print(f"\nDone. Best val R@1: {best_r_at_1:.4f}")
        print(f"Best adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
