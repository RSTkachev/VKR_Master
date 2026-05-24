"""Train/val/test split manifests for ABO experiments.

Splits are persisted to disk as item-id lists with a metadata file capturing
the exact preprocessing parameters required for deterministic reconstruction.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from vkr.preprocessing import Example, prepare_examples

SPLIT_PARTS = ["train_siglip", "train_reranker", "val", "test"]


def _splits_root(project_path: Path, splits_root: Optional[Path] = None) -> Path:
    """Return the directory that holds split manifests."""
    return splits_root if splits_root is not None else project_path / "data" / "splits"


def create_split(
    project_path: Path,
    split_name: str,
    sizes: Dict[str, int],
    seed: int = 42,
    preprocessing_kwargs: Optional[Dict] = None,
    splits_root: Optional[Path] = None,
    overwrite: bool = False,
    dedupe_by_main_image: bool = False,
) -> Path:
    """Build a fixed split and persist it to disk.

    Args:
        project_path: project root forwarded to :func:`prepare_examples`.
        split_name: identifier of the split.
        sizes: products per split part; keys must be a subset of :data:`SPLIT_PARTS`.
        seed: seed forwarded to :func:`prepare_examples` and stored in the manifest.
        preprocessing_kwargs: extra kwargs forwarded to :func:`prepare_examples` and persisted.
        splits_root: override location of the splits directory.
        overwrite: replace an existing split when True; otherwise raise.
        dedupe_by_main_image: drop products sharing ``main_image_id``, keeping the
            lexicographically smallest ``item_id``.

    Return: directory containing the written manifest files.
    """
    unknown_parts = set(sizes) - set(SPLIT_PARTS)
    if unknown_parts:
        raise ValueError(
            f"Unknown split parts in `sizes`: {unknown_parts}. "
            f"Allowed: {SPLIT_PARTS}"
        )
    for part, n in sizes.items():
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"Size for '{part}' must be a positive int, got {n}")

    split_dir = _splits_root(project_path, splits_root) / split_name
    if split_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Split '{split_name}' already exists at {split_dir}. "
            "Pass overwrite=True or use a new name."
        )

    pp_kwargs = dict(preprocessing_kwargs or {})
    examples = prepare_examples(project_path=project_path, seed=seed, **pp_kwargs)

    n_dropped = 0
    if dedupe_by_main_image:
        main_image_to_items: Dict[str, List[str]] = {}
        for ex in examples:
            if ex.is_main_image and ex.item_id is not None:
                main_image_to_items.setdefault(ex.image_id, []).append(ex.item_id)

        items_to_drop: set = set()
        for img_id, item_ids_for_img in main_image_to_items.items():
            unique_items = set(item_ids_for_img)
            if len(unique_items) > 1:
                keeper = min(unique_items)
                items_to_drop.update(unique_items - {keeper})

        if items_to_drop:
            examples = [ex for ex in examples if ex.item_id not in items_to_drop]
            n_dropped = len(items_to_drop)
            print(
                f"Dedupe by main_image_id: dropped {n_dropped} products "
                f"that share main image with another product."
            )

    seen = set()
    item_ids: List[str] = []
    for ex in examples:
        if ex.item_id is None:
            continue
        if ex.item_id in seen:
            continue
        seen.add(ex.item_id)
        item_ids.append(ex.item_id)

    total_required = sum(sizes.values())
    if len(item_ids) < total_required:
        raise ValueError(
            f"Not enough products to build the split: have {len(item_ids)}, "
            f"need {total_required}. Adjust sizes or preprocessing filters."
        )

    split_to_ids: Dict[str, List[str]] = {}
    cursor = 0
    for part in SPLIT_PARTS:
        if part not in sizes:
            continue
        n = sizes[part]
        split_to_ids[part] = item_ids[cursor : cursor + n]
        cursor += n

    all_assigned = [iid for ids in split_to_ids.values() for iid in ids]
    if len(all_assigned) != len(set(all_assigned)):
        raise RuntimeError("Internal error: overlapping ids across split parts")

    split_dir.mkdir(parents=True, exist_ok=overwrite)

    for part, ids in split_to_ids.items():
        with open(split_dir / f"{part}.json", "w") as f:
            json.dump(ids, f)

    metadata = {
        "name": split_name,
        "seed": seed,
        "sizes": {part: len(ids) for part, ids in split_to_ids.items()},
        "total_products_available": len(item_ids),
        "total_products_used": sum(len(ids) for ids in split_to_ids.values()),
        "preprocessing_kwargs": pp_kwargs,
        "dedupe_by_main_image": dedupe_by_main_image,
        "n_products_dropped_by_dedupe": n_dropped,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(split_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Created split '{split_name}' at {split_dir}")
    print(f"  Products available: {metadata['total_products_available']}")
    for part, n in metadata["sizes"].items():
        print(f"  {part}: {n} products")

    return split_dir


def load_split(
    project_path: Path,
    split_name: str,
    parts: Optional[List[str]] = None,
    splits_root: Optional[Path] = None,
) -> Dict[str, List[Example]]:
    """Load a previously created split and reconstruct its examples.

    Args:
        project_path: project root.
        split_name: split identifier (matches a directory under ``data/splits/``).
        parts: which parts to load; ``None`` loads every part present on disk.
        splits_root: override location of the splits directory.

    Return: mapping from part name to the list of :class:`Example` objects.
    """
    split_dir = _splits_root(project_path, splits_root) / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split '{split_name}' not found at {split_dir}")

    with open(split_dir / "metadata.json") as f:
        metadata = json.load(f)

    if parts is None:
        parts = sorted(f.stem for f in split_dir.glob("*.json") if f.stem != "metadata")
    else:
        missing = [p for p in parts if not (split_dir / f"{p}.json").exists()]
        if missing:
            raise FileNotFoundError(
                f"Split '{split_name}' has no parts: {missing}. "
                f"Available: {sorted(metadata.get('sizes', {}))}"
            )

    pp_kwargs = metadata.get("preprocessing_kwargs", {})
    examples = prepare_examples(
        project_path=project_path,
        seed=metadata["seed"],
        **pp_kwargs,
    )

    by_item_id: Dict[str, List[Example]] = {}
    for ex in examples:
        if ex.item_id is None:
            continue
        by_item_id.setdefault(ex.item_id, []).append(ex)

    result: Dict[str, List[Example]] = {}
    for part in parts:
        with open(split_dir / f"{part}.json") as f:
            item_ids = json.load(f)

        part_examples: List[Example] = []
        missing_ids = []
        for iid in item_ids:
            if iid in by_item_id:
                part_examples.extend(by_item_id[iid])
            else:
                missing_ids.append(iid)

        if missing_ids:
            raise RuntimeError(
                f"Split '{split_name}' part '{part}' references "
                f"{len(missing_ids)} item_ids not produced by current "
                f"preprocessing (e.g. {missing_ids[:3]}). The dataset or "
                f"preprocessing logic has changed since the split was built."
            )

        result[part] = part_examples

    return result


def describe_split(
    project_path: Path,
    split_name: str,
    splits_root: Optional[Path] = None,
) -> Dict:
    """Return a split's metadata without materialising its examples.

    Args:
        project_path: project root.
        split_name: split identifier.
        splits_root: override location of the splits directory.

    Return: parsed contents of the split's ``metadata.json``.
    """
    split_dir = _splits_root(project_path, splits_root) / split_name
    with open(split_dir / "metadata.json") as f:
        return json.load(f)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ``vkr.splits`` CLI."""
    parser = argparse.ArgumentParser(description="Manage train/val/test splits")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new split")
    p_create.add_argument("--name", required=True, help="Split identifier, e.g. abo_v1")
    p_create.add_argument("--project-path", default=".", type=Path)
    p_create.add_argument("--seed", type=int, default=42)
    p_create.add_argument("--train-siglip", type=int, default=45000)
    p_create.add_argument("--train-reranker", type=int, default=15000)
    p_create.add_argument("--val", type=int, default=10000)
    p_create.add_argument("--test", type=int, default=25000)
    p_create.add_argument(
        "--preferred-lang",
        type=str,
        default="en",
        help="Preferred metadata language; empty string disables.",
    )
    p_create.add_argument(
        "--strict-preferred-lang",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop products without metadata in preferred_lang.",
    )
    p_create.add_argument("--max-images-per-product", type=int, default=None)
    p_create.add_argument(
        "--main-image-only",
        action="store_true",
        help="Use only main_image_id for each product.",
    )
    p_create.add_argument(
        "--unique-other-images-only",
        action="store_true",
        help="Drop other_image_ids reused elsewhere in ABO.",
    )
    p_create.add_argument(
        "--subsample-category",
        type=str,
        nargs="*",
        default=None,
        help="Subsample categories to a fixed count, e.g. 'CELLULAR_PHONE_CASE:5000'.",
    )
    p_create.add_argument(
        "--dedupe-by-main-image",
        action="store_true",
        help="Drop products sharing main_image_id; keep the smallest item_id.",
    )
    p_create.add_argument("--overwrite", action="store_true")

    p_info = sub.add_parser("info", help="Show metadata of an existing split")
    p_info.add_argument("--name", required=True)
    p_info.add_argument("--project-path", default=".", type=Path)

    return parser


def main():
    """Entry point for the ``vkr.splits`` CLI."""
    args = _build_parser().parse_args()

    if args.cmd == "create":
        sizes = {
            "train_siglip": args.train_siglip,
            "train_reranker": args.train_reranker,
            "val": args.val,
            "test": args.test,
        }
        sizes = {k: v for k, v in sizes.items() if v > 0}

        subsample_per_category: Optional[Dict[str, int]] = None
        if args.subsample_category:
            subsample_per_category = {}
            for spec in args.subsample_category:
                if ":" not in spec:
                    raise ValueError(
                        f"--subsample-category expects 'CATEGORY:COUNT', got '{spec}'"
                    )
                cat, count_str = spec.rsplit(":", 1)
                subsample_per_category[cat] = int(count_str)

        pp_kwargs = {
            "max_images_per_product": args.max_images_per_product,
            "include_other_images": not args.main_image_only,
            "unique_other_images_only": args.unique_other_images_only,
            "subsample_per_category": subsample_per_category,
        }
        if args.preferred_lang:
            pp_kwargs["preferred_lang"] = args.preferred_lang
            pp_kwargs["strict_preferred_lang"] = args.strict_preferred_lang

        create_split(
            project_path=args.project_path,
            split_name=args.name,
            sizes=sizes,
            seed=args.seed,
            preprocessing_kwargs=pp_kwargs,
            overwrite=args.overwrite,
            dedupe_by_main_image=args.dedupe_by_main_image,
        )

    elif args.cmd == "info":
        meta = describe_split(args.project_path, args.name)
        print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
