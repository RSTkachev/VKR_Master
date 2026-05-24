"""Compare retriever-only and retriever+reranker per-query ranks and report where the reranker helps or hurts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args():
    """Parse command-line arguments for the analysis script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--retriever-result",
        type=Path,
        required=True,
        help=".npz with per-query ranks from retriever-only run.",
    )
    p.add_argument(
        "--reranker-result",
        type=Path,
        required=True,
        help=".npz with per-query ranks from retriever+reranker run.",
    )
    p.add_argument("--project-path", type=Path, default=Path("./"))
    p.add_argument("--split-name", type=str, required=True)
    p.add_argument("--split-part", type=str, default="test")
    p.add_argument("--out-dir", type=Path, default=Path("./error_analysis"))
    p.add_argument(
        "--sample-size",
        type=int,
        default=15,
        help="Sample queries to show per confusion cell.",
    )
    p.add_argument(
        "--min-category-size",
        type=int,
        default=30,
        help="Skip categories with fewer queries (too noisy).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def confusion_table(ret_ranks: np.ndarray, rer_ranks: np.ndarray) -> Dict:
    """Build a four-cell ``R@1`` confusion table for retriever vs. reranker.

    Args:
        ret_ranks: retriever-only ranks per query.
        rer_ranks: reranker ranks per query.

    Return: dict with counts ``both_correct``, ``both_wrong``, ``reranker_fixed``,
        ``reranker_broke``, ``net_change`` and ``R@1`` for each system.
    """
    n = len(ret_ranks)
    ret_at1 = ret_ranks == 1
    rer_at1 = rer_ranks == 1
    both_correct = int((ret_at1 & rer_at1).sum())
    both_wrong = int((~ret_at1 & ~rer_at1).sum())
    fixed = int((~ret_at1 & rer_at1).sum())
    broke = int((ret_at1 & ~rer_at1).sum())
    return {
        "n_queries": n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "reranker_fixed": fixed,
        "reranker_broke": broke,
        "net_change": fixed - broke,
        "retriever_r_at_1": float(ret_at1.mean()),
        "reranker_r_at_1": float(rer_at1.mean()),
    }


def rank_shift_distribution(ret_ranks: np.ndarray, rer_ranks: np.ndarray) -> Dict:
    """Compute the distribution of ``reranker_rank - retriever_rank``.

    Args:
        ret_ranks: retriever-only ranks per query.
        rer_ranks: reranker ranks per query.

    Return: summary statistics together with the raw per-query shifts.
    """
    shifts = rer_ranks - ret_ranks
    return {
        "mean_shift": float(shifts.mean()),
        "median_shift": float(np.median(shifts)),
        "shift_neg2_or_better": int((shifts <= -2).sum()),
        "shift_neg1": int((shifts == -1).sum()),
        "shift_zero": int((shifts == 0).sum()),
        "shift_pos1": int((shifts == 1).sum()),
        "shift_pos2_or_worse": int((shifts >= 2).sum()),
        "raw_shifts": shifts.tolist(),
    }


def per_category_breakdown(
    item_ids: List[str],
    ret_ranks: np.ndarray,
    rer_ranks: np.ndarray,
    item_id_to_ptype: Dict[str, str],
    min_size: int,
) -> List[Dict]:
    """Compute per ``product_type`` R@1 deltas between retriever and reranker.

    Args:
        item_ids: item id per query.
        ret_ranks: retriever-only ranks per query.
        rer_ranks: reranker ranks per query.
        item_id_to_ptype: mapping from item id to ABO ``product_type``.
        min_size: minimum queries required to include a category.

    Return: one dict per kept category with counts and ``R@1`` values.
    """
    by_cat = defaultdict(lambda: {"ret_correct": 0, "rer_correct": 0, "n": 0})
    for iid, r_ret, r_rer in zip(item_ids, ret_ranks, rer_ranks):
        ptype = item_id_to_ptype.get(iid, "UNKNOWN")
        by_cat[ptype]["n"] += 1
        if r_ret == 1:
            by_cat[ptype]["ret_correct"] += 1
        if r_rer == 1:
            by_cat[ptype]["rer_correct"] += 1
    rows = []
    for ptype, stats in by_cat.items():
        if stats["n"] < min_size:
            continue
        ret_r1 = stats["ret_correct"] / stats["n"]
        rer_r1 = stats["rer_correct"] / stats["n"]
        rows.append(
            {
                "product_type": ptype,
                "n_queries": stats["n"],
                "retriever_r_at_1": ret_r1,
                "reranker_r_at_1": rer_r1,
                "delta": rer_r1 - ret_r1,
            }
        )
    return rows


def sample_queries(
    mask: np.ndarray,
    item_ids: List[str],
    texts: List[str],
    ret_ranks: np.ndarray,
    rer_ranks: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Sample queries from a boolean mask for qualitative inspection.

    Args:
        mask: boolean mask selecting eligible queries.
        item_ids: item id per query.
        texts: text per query.
        ret_ranks: retriever ranks per query.
        rer_ranks: reranker ranks per query.
        n_samples: maximum number of samples returned.
        rng: random generator used for sampling.

    Return: up to ``n_samples`` dicts describing the sampled queries.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    chosen = rng.choice(idx, size=min(n_samples, len(idx)), replace=False)
    return [
        {
            "item_id": item_ids[i],
            "text": texts[i][:200],
            "retriever_rank": int(ret_ranks[i]),
            "reranker_rank": int(rer_ranks[i]),
        }
        for i in sorted(chosen)
    ]


def load_product_types(project_path: Path, needed_ids: set) -> Dict[str, str]:
    """Read ABO metadata and build a mapping from ``item_id`` to ``product_type``.

    Args:
        project_path: project root containing ``data/raw_data/abo/metadata``.
        needed_ids: item ids to resolve; other ids are skipped.

    Return: mapping from item id to product type (English preferred).
    """
    import polars as pl

    metadata_dir = project_path / "data/raw_data/abo/metadata"
    item_id_to_ptype = {}
    for ndjson_path in sorted(metadata_dir.glob("*.json*")):
        try:
            df = pl.read_ndjson(str(ndjson_path))
        except Exception:
            continue
        for record in df.iter_rows(named=True):
            iid = record.get("item_id")
            if not iid or iid not in needed_ids or iid in item_id_to_ptype:
                continue
            ptype_field = record.get("product_type")
            if not ptype_field:
                continue
            chosen = None
            if isinstance(ptype_field, list) and ptype_field:
                for entry in ptype_field:
                    if isinstance(entry, dict) and entry.get(
                        "language_tag", ""
                    ).lower().startswith("en"):
                        chosen = entry.get("value")
                        break
                if chosen is None and isinstance(ptype_field[0], dict):
                    chosen = ptype_field[0].get("value")
            elif isinstance(ptype_field, str):
                chosen = ptype_field
            if chosen:
                item_id_to_ptype[iid] = chosen
    return item_id_to_ptype


def main():
    """Entry point: load two ranking arrays, compute diagnostics and persist outputs."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading retriever ranks: {args.retriever_result}")
    ret = np.load(args.retriever_result, allow_pickle=True)
    print(f"Loading reranker ranks:  {args.reranker_result}")
    rer = np.load(args.reranker_result, allow_pickle=True)

    ret_ranks = ret["ranks"].astype(int)
    rer_ranks = rer["ranks"].astype(int)
    item_ids = ret["item_ids"]
    if not (item_ids == rer["item_ids"]).all():
        raise ValueError(
            "item_id arrays differ between runs: they evaluated different examples."
        )
    item_ids = item_ids.tolist()

    from vkr.splits import load_split

    data = load_split(args.project_path, args.split_name, parts=[args.split_part])
    examples = [ex for ex in data[args.split_part] if ex.is_main_image]
    iid_to_text = {ex.item_id: ex.text for ex in examples}
    texts = [iid_to_text.get(iid, "<missing>") for iid in item_ids]

    print("\n=== Confusion table ===")
    conf = confusion_table(ret_ranks, rer_ranks)
    pct = lambda x: 100 * x / conf["n_queries"]
    print(f"  Total queries:            {conf['n_queries']}")
    print(
        f"  Both correct (R@1=1):     {conf['both_correct']:>6} ({pct(conf['both_correct']):5.1f}%)"
    )
    print(
        f"  Reranker fixed retriever: {conf['reranker_fixed']:>6} ({pct(conf['reranker_fixed']):5.1f}%)"
    )
    print(
        f"  Reranker broke retriever: {conf['reranker_broke']:>6} ({pct(conf['reranker_broke']):5.1f}%)"
    )
    print(
        f"  Both wrong:               {conf['both_wrong']:>6} ({pct(conf['both_wrong']):5.1f}%)"
    )
    print(f"  Net change: {conf['net_change']:+d} ({pct(conf['net_change']):+.1f} pp)")
    print(f"  Retriever R@1: {conf['retriever_r_at_1']:.4f}")
    print(f"  Reranker  R@1: {conf['reranker_r_at_1']:.4f}")

    print("\n=== Rank-shift distribution ===")
    shift = rank_shift_distribution(ret_ranks, rer_ranks)
    print(f"  Mean shift:   {shift['mean_shift']:+.3f} (negative = improvement)")
    print(f"  Median shift: {shift['median_shift']:+.0f}")
    print(
        f"  Improved by 2+: {shift['shift_neg2_or_better']:>5} ({pct(shift['shift_neg2_or_better']):5.1f}%)"
    )
    print(
        f"  Improved by 1:  {shift['shift_neg1']:>5} ({pct(shift['shift_neg1']):5.1f}%)"
    )
    print(
        f"  Unchanged:      {shift['shift_zero']:>5} ({pct(shift['shift_zero']):5.1f}%)"
    )
    print(
        f"  Worsened by 1:  {shift['shift_pos1']:>5} ({pct(shift['shift_pos1']):5.1f}%)"
    )
    print(
        f"  Worsened by 2+: {shift['shift_pos2_or_worse']:>5} ({pct(shift['shift_pos2_or_worse']):5.1f}%)"
    )

    print("\nLoading product_type metadata...")
    iid_to_ptype = load_product_types(args.project_path, set(item_ids))
    print(f"  Resolved product_type for {len(iid_to_ptype)}/{len(item_ids)} items")
    cat_rows = per_category_breakdown(
        item_ids,
        ret_ranks,
        rer_ranks,
        iid_to_ptype,
        args.min_category_size,
    )

    print(f"\n=== Top categories where reranker HELPS most (R@1 delta) ===")
    print(
        f"{'product_type':>40}  {'N':>5}  {'ret_R@1':>8}  {'rer_R@1':>8}  {'delta':>7}"
    )
    for row in sorted(cat_rows, key=lambda r: -r["delta"])[:10]:
        print(
            f"{row['product_type']:>40}  {row['n_queries']:>5}  "
            f"{row['retriever_r_at_1']:>8.3f}  {row['reranker_r_at_1']:>8.3f}  "
            f"{row['delta']:>+7.3f}"
        )
    print(f"\n=== Top categories where reranker HURTS most (R@1 delta) ===")
    print(
        f"{'product_type':>40}  {'N':>5}  {'ret_R@1':>8}  {'rer_R@1':>8}  {'delta':>7}"
    )
    for row in sorted(cat_rows, key=lambda r: r["delta"])[:10]:
        print(
            f"{row['product_type']:>40}  {row['n_queries']:>5}  "
            f"{row['retriever_r_at_1']:>8.3f}  {row['reranker_r_at_1']:>8.3f}  "
            f"{row['delta']:>+7.3f}"
        )

    rng = np.random.default_rng(args.seed)
    ret_at1 = ret_ranks == 1
    rer_at1 = rer_ranks == 1
    samples = {
        "reranker_fixed": sample_queries(
            ~ret_at1 & rer_at1,
            item_ids,
            texts,
            ret_ranks,
            rer_ranks,
            args.sample_size,
            rng,
        ),
        "reranker_broke": sample_queries(
            ret_at1 & ~rer_at1,
            item_ids,
            texts,
            ret_ranks,
            rer_ranks,
            args.sample_size,
            rng,
        ),
        "both_wrong": sample_queries(
            ~ret_at1 & ~rer_at1,
            item_ids,
            texts,
            ret_ranks,
            rer_ranks,
            args.sample_size,
            rng,
        ),
    }

    out = {
        "config": {
            "retriever_result": str(args.retriever_result),
            "reranker_result": str(args.reranker_result),
            "split_name": args.split_name,
            "split_part": args.split_part,
            "min_category_size": args.min_category_size,
        },
        "confusion": conf,
        "rank_shift": {k: v for k, v in shift.items() if k != "raw_shifts"},
        "category_breakdown": sorted(cat_rows, key=lambda r: -r["n_queries"]),
        "sample_queries": samples,
    }
    out_json = args.out_dir / "rerank_error_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote analysis to: {out_json}")

    np.savez(
        args.out_dir / "rank_shifts.npz",
        shifts=np.array(shift["raw_shifts"]),
        item_ids=np.array(item_ids),
    )
    print(f"Wrote raw shifts to: {args.out_dir / 'rank_shifts.npz'}")


if __name__ == "__main__":
    main()
