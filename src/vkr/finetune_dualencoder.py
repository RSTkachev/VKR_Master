"""Contrastive fine-tuning of a SigLIP-style dual encoder on ABO with optional LoRA adapters."""

import os
import gc
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import argparse
import math

load_dotenv(".env", override=True)

import numpy as np
import torch
import torch.nn.functional as F
import mlflow
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from peft import LoraConfig, get_peft_model
from PIL import Image

from vkr.splits import load_split
from vkr.encode import encode_texts, encode_images
from vkr.scoring import retrieval_metrics_from_ranks, compute_ranks
from vkr.utils import load_model_bundle, choose_device, set_seed


class ContrastiveDataset(torch.utils.data.Dataset):
    """Dataset yielding ``(image, text, product_index)`` records for contrastive training."""

    def __init__(self, examples):
        """Store the list of :class:`vkr.preprocessing.Example` records.

        Args:
            examples: examples to iterate over.
        """
        self.examples = examples

    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx):
        """Return a single record with the image opened lazily.

        Args:
            idx: index into the example list.

        Return: dict with ``image`` (RGB ``PIL.Image``), ``text`` and ``product_index``.
        """
        ex = self.examples[idx]
        image = Image.open(ex.image_file).convert("RGB")
        return {
            "image": image,
            "text": ex.text,
            "product_index": ex.product_index,
        }


def collate_fn(batch, processor):
    """Collate a batch of records into model inputs using the SigLIP processor.

    Args:
        batch: records produced by :class:`ContrastiveDataset`.
        processor: SigLIP processor.

    Return: batched ``pixel_values`` / ``input_ids`` / ``attention_mask`` plus a ``product_index`` tensor.
    """
    encoded = processor(
        images=[b["image"] for b in batch],
        text=[b["text"].lower() for b in batch],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    encoded["product_index"] = torch.tensor(
        [b["product_index"] for b in batch], dtype=torch.long
    )
    return encoded


def evaluate_trained(model, processor, examples, batch_size, device):
    """Compute image-to-text and text-to-image retrieval metrics.

    Args:
        model: dual encoder model.
        processor: matching processor.
        examples: queries and target pool.
        batch_size: batch size for encoding.
        device: torch device.

    Return: ``(i2t_metrics, t2i_metrics)`` from :func:`retrieval_metrics_from_ranks`.
    """
    model.eval()

    texts = [e.text.lower() for e in examples]
    image_paths = [e.image_file for e in examples]

    text_embeds = encode_texts(
        model,
        processor,
        texts,
        batch_size=batch_size,
        device=device,
        model_name=model.name_or_path,
    )

    image_embeds = encode_images(
        model,
        processor,
        image_paths,
        batch_size=batch_size,
        device=device,
    )

    similarity = (image_embeds @ text_embeds.T).numpy()

    i2t = retrieval_metrics_from_ranks(compute_ranks(similarity))
    t2i = retrieval_metrics_from_ranks(compute_ranks(similarity.T))

    return i2t, t2i


def calculate_infonce(image_embeds, text_embeds, logit_scale, product_index=None):
    """Symmetric InfoNCE loss with optional multi-positive (SupCon) variant.

    Args:
        image_embeds: image embeddings of shape ``(B, D)``.
        text_embeds: text embeddings of shape ``(B, D)``.
        logit_scale: log-temperature parameter; exponentiated to scale logits.
        product_index: optional; pairs sharing the same value count as positive.

    Return: scalar loss tensor.
    """
    zimg = F.normalize(image_embeds, dim=-1)
    ztxt = F.normalize(text_embeds, dim=-1)
    t = torch.exp(logit_scale)

    logits = torch.matmul(zimg, ztxt.T) * t

    if product_index is None:
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    positive_mask = (product_index.unsqueeze(0) == product_index.unsqueeze(1)).float()

    log_prob_i2t = F.log_softmax(logits, dim=1)
    log_prob_t2i = F.log_softmax(logits.T, dim=1)

    n_positives = positive_mask.sum(dim=1).clamp(min=1)
    loss_i = -(positive_mask * log_prob_i2t).sum(dim=1) / n_positives
    loss_t = -(positive_mask * log_prob_t2i).sum(dim=1) / n_positives

    return (loss_i.mean() + loss_t.mean()) / 2


def calculate_sigmoid(
    image_embeds, text_embeds, logit_scale, logit_bias, product_index=None
):
    """Sigmoid contrastive loss with optional multi-positive labels.

    Args:
        image_embeds: image embeddings of shape ``(B, D)``.
        text_embeds: text embeddings of shape ``(B, D)``.
        logit_scale: log-temperature parameter; exponentiated to scale logits.
        logit_bias: additive logit bias.
        product_index: optional; pairs sharing the same value are labelled positive.

    Return: scalar loss tensor.
    """
    n = image_embeds.shape[0]
    device = image_embeds.device

    t = torch.exp(logit_scale)
    zimg = F.normalize(image_embeds, dim=-1)
    ztxt = F.normalize(text_embeds, dim=-1)

    logits = torch.matmul(zimg, ztxt.T) * t + logit_bias

    if product_index is None:
        labels = 2 * torch.eye(n, device=device) - torch.ones(n, n, device=device)
    else:
        positive_mask = (
            product_index.unsqueeze(0) == product_index.unsqueeze(1)
        ).float()
        labels = 2 * positive_mask - 1

    loss = -torch.sum(F.logsigmoid(labels * logits)) / n
    return loss


def apply_lora(model, rank, alpha, dropout, target_modules):
    """Attach LoRA adapters to the SigLIP model and keep contrastive scalars trainable.

    Args:
        model: dual encoder model to adapt.
        rank: LoRA rank ``r``.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout probability.
        target_modules: module name patterns LoRA is injected into.

    Return: adapted model with ``logit_scale`` and ``logit_bias`` trainable.
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "logit_scale" in name or "logit_bias" in name:
            param.requires_grad = True

    return model


def count_trainable_params(model):
    """Count trainable and total parameters.

    Args:
        model: module to inspect.

    Return: tuple ``(trainable, total)``.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def finetune_contrastive(
    model,
    processor,
    train_dataloader,
    val_examples,
    infer_batch_size,
    optimizer,
    loss_fn,
    device,
    n_epochs=10,
    use_multi_positive=False,
    patience=None,
    min_delta=0.0,
    accumulation_steps=1,
    scheduler=None,
):
    """Fine-tune the dual encoder with the chosen contrastive loss.

    Args:
        model: dual encoder model (optionally LoRA-adapted).
        processor: matching processor.
        train_dataloader: loader yielding batches produced by :func:`collate_fn`.
        val_examples: validation examples used for retrieval metrics each epoch.
        infer_batch_size: batch size used during validation encoding.
        optimizer: optimiser over trainable parameters.
        loss_fn: ``"infonce"`` or ``"sigmoid"``.
        device: torch device.
        n_epochs: maximum number of epochs.
        use_multi_positive: enable multi-positive labels via ``product_index``.
        patience: early-stopping patience in epochs.
        min_delta: minimum improvement in mean R@1 counted as progress.
        accumulation_steps: micro-batches per optimiser step.
        scheduler: optional LR scheduler stepped after each optimiser step.

    Return: dict with epoch-level ``metrics``, ``best_epoch`` and ``best_val_mrr``.
    """
    metrics = {"loss": [], "i2t": [], "t2i": []}
    best_metric = -float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    print("Training")
    if accumulation_steps > 1:
        print(
            f"Gradient accumulation: {accumulation_steps} micro-batches per step "
            f"(effective batch ~= micro_batch * {accumulation_steps})."
        )

    for epoch in range(n_epochs):
        total_loss = 0
        n_micro_batches = 0
        model.train()
        optimizer.zero_grad()

        progress_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            leave=False,
            desc=f"Train {epoch+1}/{n_epochs} epoch",
        )
        for step, batch in enumerate(progress_bar):
            product_index = batch.pop("product_index", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if product_index is not None:
                product_index = product_index.to(device)

            image_embeds = model.get_image_features(
                pixel_values=batch["pixel_values"]
            ).pooler_output
            text_embeds = model.get_text_features(
                input_ids=batch["input_ids"]
            ).pooler_output

            pi = product_index if use_multi_positive else None

            if loss_fn == "infonce":
                loss = calculate_infonce(
                    image_embeds, text_embeds, model.logit_scale, pi
                )
            elif loss_fn == "sigmoid":
                loss = calculate_sigmoid(
                    image_embeds, text_embeds, model.logit_scale, model.logit_bias, pi
                )
            else:
                return

            (loss / accumulation_steps).backward()

            is_accum_boundary = (step + 1) % accumulation_steps == 0
            is_last_step = (step + 1) == len(train_dataloader)
            if is_accum_boundary or is_last_step:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            cur_loss = loss.item()
            total_loss += cur_loss
            n_micro_batches += 1

            progress_bar.set_postfix_str(f"Loss: {cur_loss:.3}")

        gc.collect()
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            i2t, t2i = evaluate_trained(
                model, processor, val_examples, infer_batch_size, device
            )

        avg_loss = total_loss / max(n_micro_batches, 1)
        metrics["loss"].append(avg_loss)
        metrics["i2t"].append(i2t)
        metrics["t2i"].append(t2i)

        val_mrr = (i2t["mrr"] + t2i["mrr"]) / 2
        val_r1 = (i2t["r_at_1"] + t2i["r_at_1"]) / 2
        val_r10 = (i2t["r_at_10"] + t2i["r_at_10"]) / 2

        gc.collect()
        torch.cuda.empty_cache()

        improved = val_r1 > best_metric + min_delta
        if improved:
            best_metric = val_r1
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
            improvement_marker = " [best]"
        else:
            epochs_without_improvement += 1
            if patience is not None:
                improvement_marker = (
                    f" (no improve {epochs_without_improvement}/{patience})"
                )
            else:
                improvement_marker = ""

        print(
            f"Epoch {epoch+1}: Train loss = {avg_loss:.4f}; "
            f"Val MRR: {val_mrr:.4f}, R@1: {val_r1:.4f}, R@10: {val_r10:.4f}"
            f"{improvement_marker}"
        )

        mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
        mlflow.log_metric("val_mrr_mean", val_mrr, step=epoch + 1)
        mlflow.log_metric("val_r_at_1_mean", val_r1, step=epoch + 1)
        mlflow.log_metric("val_r_at_10_mean", val_r10, step=epoch + 1)
        mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch + 1)
        mlflow.log_metrics(
            metrics={f"i2t_{k}": v for k, v in i2t.items()}, step=epoch + 1
        )
        mlflow.log_metrics(
            metrics={f"t2i_{k}": v for k, v in t2i.items()}, step=epoch + 1
        )

        if patience is not None and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch+1}. "
                f"Best epoch: {best_epoch} (Val MRR={best_metric:.4f})"
            )
            break

    if best_state is not None:
        print(f"Restoring best weights from epoch {best_epoch}")
        model.load_state_dict(best_state)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_mrr", best_metric)

    return {
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_val_mrr": best_metric if best_state is not None else None,
    }


def save_lora_checkpoint(model, checkpoint_path, metadata):
    """Persist a LoRA adapter together with a JSON metadata file.

    Args:
        model: adapted model exposing ``save_pretrained``.
        checkpoint_path: destination directory (created if missing).
        metadata: JSON-serialisable metadata stored as ``run_info.json``.

    Return: None.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "save_pretrained"):
        raise RuntimeError(
            "Model has no save_pretrained method; was it wrapped with PEFT?"
        )
    model.save_pretrained(str(checkpoint_path))

    with open(checkpoint_path / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the dual-encoder training script."""
    parser = argparse.ArgumentParser(description="Stage 2 contrastive training")
    parser.add_argument(
        "--project-path", type=str, default="./", help="Project root path"
    )
    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        help="Split identifier; training uses 'train_siglip', validation 'val'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", help="auto, cuda, cpu, cuda:0, ..."
    )
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/siglip-base-patch16-256",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate (peak LR after warmup if scheduler is enabled).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="AdamW beta1.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.95,
        help="AdamW beta2.",
    )
    parser.add_argument(
        "--main-image-only",
        action="store_true",
        help="Restrict training data to main images only.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "none"],
        help="LR scheduler type.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Fraction of total optimizer steps for linear LR warmup.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Root directory for saving LoRA adapters; empty disables saving.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional tag for the checkpoint subdirectory name.",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="sigmoid",
        help="Loss function",
    )
    parser.add_argument(
        "--use-multi-positive",
        action="store_true",
        help="Treat batch items with the same product_index as positive pairs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs; 0 or omit disables.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum MRR improvement counted as progress.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning instead of full fine-tuning",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (r).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,out_proj",
        help="Comma-separated list of module names to inject LoRA into.",
    )
    return parser.parse_args()


def main():
    """Entry point for the dual-encoder fine-tuning script."""
    args = parse_args()
    set_seed(args.seed)

    project_path = Path(args.project_path)

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    device = choose_device(args.device)
    model, processor = load_model_bundle(args.model_id, device=device)

    if args.use_lora:
        target_modules = [
            m.strip() for m in args.lora_target_modules.split(",") if m.strip()
        ]
        model = apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        trainable, total = count_trainable_params(model)
        print(
            f"LoRA enabled. Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.4f}%)"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    split_data = load_split(
        project_path=project_path,
        split_name=args.split_name,
        parts=["train_siglip", "val"],
    )
    train_examples = split_data["train_siglip"]
    val_examples = split_data["val"]

    if args.main_image_only:
        n_before = len(train_examples)
        train_examples = [ex for ex in train_examples if ex.is_main_image]
        print(
            f"--main-image-only: filtered training data from {n_before} to "
            f"{len(train_examples)} examples (one view per product)."
        )

    val_examples = [ex for ex in val_examples if ex.is_main_image]

    n_train_products = len({ex.item_id for ex in train_examples})
    n_val_products = len({ex.item_id for ex in val_examples})
    print(
        f"Split '{args.split_name}': "
        f"Train: {len(train_examples)} examples ({n_train_products} products); "
        f"Val: {len(val_examples)} examples ({n_val_products} products, main-image only)"
    )

    train_dataset = ContrastiveDataset(train_examples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor),
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_steps = steps_per_epoch * args.epochs

    scheduler = None
    warmup_steps = 0
    if args.scheduler == "cosine":
        warmup_steps = int(total_steps * args.warmup_ratio)
        decay_steps = max(total_steps - warmup_steps, 1)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=0.0,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[max(warmup_steps, 1)],
        )
        print(
            f"Scheduler: linear warmup + cosine decay (PyTorch native). "
            f"Total optimizer steps: {total_steps}, warmup steps: {warmup_steps} "
            f"({args.warmup_ratio * 100:.1f}%)."
        )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("vkr")

    run_suffix = "lora" if args.use_lora else "full_ft"

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "model_id": args.model_id,
                "split_name": args.split_name,
                "n_train_products": n_train_products,
                "n_val_products": n_val_products,
                "n_train_examples": len(train_examples),
                "main_image_only": args.main_image_only,
                "loss_function": args.loss_function,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "train_batch_size": args.train_batch_size,
                "accumulation_steps": args.accumulation_steps,
                "effective_batch_size": args.train_batch_size * args.accumulation_steps,
                "epochs": args.epochs,
                "use_lora": args.use_lora,
                "use_multi_positive": args.use_multi_positive,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "scheduler": args.scheduler,
                "warmup_ratio": args.warmup_ratio if scheduler is not None else None,
                "warmup_steps": warmup_steps if scheduler is not None else None,
                "total_optimizer_steps": total_steps if scheduler is not None else None,
                "seed": args.seed,
            }
        )
        if args.use_lora:
            mlflow.log_params(
                {
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                }
            )

        results = finetune_contrastive(
            model=model,
            processor=processor,
            train_dataloader=train_loader,
            val_examples=val_examples,
            infer_batch_size=args.infer_batch_size,
            optimizer=optimizer,
            loss_fn=args.loss_function,
            device=device,
            n_epochs=args.epochs,
            use_multi_positive=args.use_multi_positive,
            patience=args.patience if args.patience and args.patience > 0 else None,
            min_delta=args.min_delta,
            accumulation_steps=args.accumulation_steps,
            scheduler=scheduler,
        )

        if not args.checkpoint_dir:
            print("Checkpoint saving disabled (--checkpoint-dir is empty).")
        elif not args.use_lora:
            print(
                "Skipping checkpoint save: full fine-tuning chosen, but only "
                "LoRA adapter saving is implemented. Pass --use-lora to enable saving."
            )
        elif results["best_val_mrr"] is None:
            print("Skipping checkpoint save: training never produced an improvement.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_tag = args.run_name or timestamp
            ckpt_subdir = f"{args.split_name}_{args.loss_function}_lora_r{args.lora_rank}_{run_tag}"
            ckpt_path = Path(args.checkpoint_dir) / ckpt_subdir

            metadata = {
                "base_model_id": args.model_id,
                "split_name": args.split_name,
                "n_train_products": n_train_products,
                "n_train_examples": len(train_examples),
                "main_image_only": args.main_image_only,
                "loss_function": args.loss_function,
                "use_multi_positive": args.use_multi_positive,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "train_batch_size": args.train_batch_size,
                "accumulation_steps": args.accumulation_steps,
                "effective_batch_size": args.train_batch_size * args.accumulation_steps,
                "scheduler": args.scheduler,
                "warmup_ratio": args.warmup_ratio if scheduler is not None else None,
                "lora": {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": args.lora_target_modules,
                },
                "best_epoch": results["best_epoch"],
                "best_val_mrr": results["best_val_mrr"],
                "seed": args.seed,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "mlflow_run_name": args.run_name,
            }
            save_lora_checkpoint(model, ckpt_path, metadata)
            print(f"Saved LoRA adapter to: {ckpt_path}")

            mlflow.log_param("checkpoint_path", str(ckpt_path))


if __name__ == "__main__":
    main()
