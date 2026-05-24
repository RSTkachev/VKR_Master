"""Analyse the nearest-neighbour similarity distribution of SigLIP text embeddings on ABO."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.neighbors import NearestNeighbors

from vkr.splits import load_split
from vkr.encode import encode_texts
from vkr.utils import choose_device, load_model_bundle


def tokenize(text: str) -> set:
    """Tokenise text into a set of lowercase alphanumeric tokens.

    Args:
        text: input string.

    Return: set of token strings.
    """
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def jaccard(a: set, b: set) -> float:
    """Compute the Jaccard similarity between two token sets.

    Args:
        a: first token set.
        b: second token set.

    Return: ``len(a & b) / len(a | b)``, or ``0.0`` when both sets are empty.
    """
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_args():
    """Parse command-line arguments for the analysis script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-path", type=Path, default=Path("../"))
    p.add_argument("--split-name", type=str, default="abo_v3")
    p.add_argument(
        "--model-id",
        type=str,
        default="google/siglip2-base-patch16-256",
        help="Base model id used to encode texts.",
    )
    p.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help="Optional LoRA adapter merged into the base model before encoding.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    """Entry point: encode every unique product, find nearest neighbours and save arrays."""
    args = parse_args()
    device = choose_device(args.device)
    print(f"Device: {device}")

    parts = ["train_siglip", "train_reranker", "val", "test"]
    print(f"Loading split '{args.split_name}'...")
    data = load_split(args.project_path, args.split_name, parts=parts)

    all_texts = []
    seen_items = set()
    for part in parts:
        for ex in data[part]:
            if ex.is_main_image and ex.item_id not in seen_items:
                all_texts.append(ex.text)
                seen_items.add(ex.item_id)

    n = len(all_texts)
    print(f"Total unique products: {n}")

    print(f"Loading model: {args.model_id}")
    model, processor = load_model_bundle(args.model_id, device=device)

    if args.lora_checkpoint:
        print(f"Applying LoRA adapter: {args.lora_checkpoint}")
        model = PeftModel.from_pretrained(model, args.lora_checkpoint)
        model = model.merge_and_unload()
        model.to(device)

    model.eval()

    print(f"Encoding {n} texts (batch_size={args.batch_size})...")
    text_embeds = encode_texts(
        model,
        processor,
        all_texts,
        batch_size=args.batch_size,
        device=device,
        model_name=args.model_id,
    )
    text_embeds_np = text_embeds.numpy().astype(np.float32)
    print(f"Embeddings shape: {text_embeds_np.shape}")

    del model, processor
    torch.cuda.empty_cache() if "cuda" in device else None

    print("Finding nearest neighbor for each product...")
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)
    nn.fit(text_embeds_np)
    distances, indices = nn.kneighbors(text_embeds_np)

    nn_similarities = 1.0 - distances[:, 1]
    nn_indices = indices[:, 1]

    self_mismatch = (indices[:, 0] != np.arange(n)).sum()
    if self_mismatch > 0:
        print(
            f"Warning: {self_mismatch} products had a non-self entry at "
            f"position 0 (probably duplicate texts). Their nn-similarity "
            f"may be slightly off, but the overall distribution is unaffected."
        )

    print("\n=== SigLIP text-embedding nearest-neighbor similarity ===")
    print("Cumulative (products with NN similarity >= T):")
    for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        n_above = (nn_similarities >= t).sum()
        pct = 100 * n_above / n
        print(f"  >= {t}:  {n_above:>6}  ({pct:>5.1f}%)")

    print("\nComputing Jaccard on the same nearest pairs for comparison...")
    token_sets = [tokenize(t) for t in all_texts]
    jaccard_for_nn_pair = np.array(
        [jaccard(token_sets[i], token_sets[nn_indices[i]]) for i in range(n)]
    )

    print("\n=== Cross-tab: SigLIP-NN-similarity vs Jaccard on same pairs ===")
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
    print(
        f"{'sim_range':>14}  {'count':>8}  {'mean_jaccard':>14}  {'%pairs_jacc>=0.8':>18}"
    )
    for lo, hi in bins:
        mask = (nn_similarities >= lo) & (nn_similarities < hi)
        c = mask.sum()
        if c == 0:
            continue
        mean_j = jaccard_for_nn_pair[mask].mean()
        pct_high_j = 100 * (jaccard_for_nn_pair[mask] >= 0.8).mean()
        print(f"  [{lo:.2f}, {hi:.2f})  {c:>8}  {mean_j:>14.3f}  {pct_high_j:>17.1f}%")

    print("\n=== Examples: high SigLIP similarity (>= 0.95) ===")
    high_sim_idx = np.where(nn_similarities >= 0.95)[0]
    for i in high_sim_idx[:5]:
        j = nn_indices[i]
        print(
            f"\n  sim = {nn_similarities[i]:.4f}, jaccard = {jaccard_for_nn_pair[i]:.3f}"
        )
        print(f"  A: {all_texts[i][:120]}")
        print(f"  B: {all_texts[j][:120]}")

    print(
        "\n=== Examples: high SigLIP sim but low Jaccard "
        "(semantic close, lexically different) ==="
    )
    interesting_mask = (nn_similarities >= 0.85) & (jaccard_for_nn_pair < 0.5)
    interesting_idx = np.where(interesting_mask)[0]
    print(f"Total: {interesting_mask.sum()}")
    for i in interesting_idx[:5]:
        j = nn_indices[i]
        print(
            f"\n  sim = {nn_similarities[i]:.4f}, jaccard = {jaccard_for_nn_pair[i]:.3f}"
        )
        print(f"  A: {all_texts[i][:120]}")
        print(f"  B: {all_texts[j][:120]}")

    out_path = args.project_path / "siglip_nn_similarity.npz"
    np.savez(
        out_path,
        nn_similarities=nn_similarities,
        nn_indices=nn_indices,
        jaccard_for_nn_pair=jaccard_for_nn_pair,
    )
    print(f"\nSaved arrays to: {out_path}")


if __name__ == "__main__":
    main()
