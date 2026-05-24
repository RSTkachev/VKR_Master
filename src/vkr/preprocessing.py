"""ABO metadata parsing and example construction for retrieval training and evaluation."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import polars as pl
from PIL import Image
from tqdm import tqdm

from vkr.utils import clean_whitespace, normalize_text_value


@dataclass
class Example:
    """A single (image, text) training/evaluation record.

    Attributes:
        item_index: position of the example in the flat output list.
        item_id: original ABO ``item_id`` the example belongs to.
        image_id: ABO ``image_id`` of the view.
        image_file: absolute path to the image on disk.
        text: cleaned product description.
        is_main_image: whether this image is the product's main view.
        product_index: position of the parent product after product-level shuffling.
    """

    item_index: int
    item_id: Optional[str]
    image_id: str
    image_file: str
    text: str
    is_main_image: bool = False
    product_index: int = -1


def load_abo_metadata(metadata_path: Path) -> pl.DataFrame:
    """Load ABO metadata from a single ND-JSON file or a directory of files.

    Args:
        metadata_path: ND-JSON file or directory with ``*.json`` ND-JSON parts.

    Return: concatenated metadata as a Polars dataframe.
    """
    if metadata_path.is_dir():
        ndjson_files = sorted(list(metadata_path.glob("*.json")))
        if not ndjson_files:
            raise FileNotFoundError(f"No metadata files found under: {metadata_path}")
        frames = [
            pl.read_ndjson(str(path))
            for path in tqdm(ndjson_files, desc="Loading metadata files")
        ]
        return pl.concat(frames, how="diagonal_relaxed")

    return pl.read_ndjson(str(metadata_path))


def load_images(paths: Sequence[str]):
    """Load and RGB-convert images from disk.

    Args:
        paths: paths to image files.

    Return: list of RGB ``PIL.Image`` in the same order as ``paths``.
    """
    images: List[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def get_product_data(
    record: Dict[str, Any],
    preferred_lang: Optional[str] = None,
    strict_preferred_lang: bool = False,
):
    """Pick the best textual description for an ABO product.

    Each candidate language is scored by attributes contributing non-redundant
    tokens (brand, product type, color, material, style); a preferred language
    wins ties.

    Args:
        record: raw ABO metadata row.
        preferred_lang: language tag fragment (e.g. ``"en"``) preferred when present.
        strict_preferred_lang: if True and ``preferred_lang`` is missing, return ``None``.

    Return: dict with ``text``, ``score`` and ``is_preferred``, or ``None`` if no
        usable description exists.
    """

    def get_val(field_name: str, lang: str):
        entries = record.get(field_name)
        if not entries or not isinstance(entries, list):
            return None
        entry = next((e for e in entries if e.get("language_tag") == lang), None)
        if not entry:
            return None

        val = entry.get("value")
        if not val and "standardized_values" in entry:
            sv = entry.get("standardized_values")
            if isinstance(sv, list):
                val = ", ".join([str(v) for v in sv if v is not None])
            elif sv is not None:
                val = str(sv)

        return normalize_text_value(val) if val else None

    item_name_entries = record.get("item_name")
    if not item_name_entries or not isinstance(item_name_entries, list):
        return None

    langs = {e.get("language_tag") for e in item_name_entries if e.get("language_tag")}
    if not langs:
        return None

    lang_data = {}

    for lang in langs:
        name = get_val("item_name", lang)
        if not name:
            continue

        score = 10
        brand = get_val("brand", lang)

        if brand and brand.lower() not in name.lower():
            score += 1
            title = f"{brand} {name}"
        else:
            title = name

        parts = []
        for attr in ["product_type", "color", "material", "style"]:
            val = get_val(attr, lang)
            if val and val.lower() not in name.lower():
                score += 1
                parts.append(val)

        final_text = title
        if parts:
            final_text += ". " + ", ".join(parts)

        lang_data[lang] = {"score": score, "text": clean_whitespace(final_text)}

    if not lang_data:
        return None

    if preferred_lang is not None:
        potential_winners = [
            l for l in lang_data if preferred_lang.lower() in l.lower()
        ]
        if potential_winners:
            winner_lang = max(potential_winners, key=lambda l: lang_data[l]["score"])
            is_preferred = True
        elif strict_preferred_lang:
            return None
        else:
            winner_lang = max(lang_data, key=lambda l: lang_data[l]["score"])
            is_preferred = False
    else:
        winner_lang = max(lang_data, key=lambda l: lang_data[l]["score"])
        is_preferred = True

    winner_data = lang_data[winner_lang]

    return {
        "text": winner_data["text"],
        "score": winner_data["score"],
        "is_preferred": is_preferred,
    }


def prepare_examples(
    project_path: Path,
    preferred_lang: Optional[str] = None,
    strict_preferred_lang: bool = False,
    max_items: Optional[int] = None,
    max_images_per_product: Optional[int] = None,
    include_other_images: bool = True,
    unique_other_images_only: bool = False,
    subsample_per_category: Optional[Dict[str, int]] = None,
    seed: int = 42,
):
    """Build the flat list of :class:`Example` records from the ABO dataset.

    Args:
        project_path: project root used to locate ``data/raw_data/abo``.
        preferred_lang: preferred metadata language; ``None`` picks the richest one per product.
        strict_preferred_lang: drop products without metadata in ``preferred_lang``.
        max_items: cap on the number of products.
        max_images_per_product: cap on image views per product; ``None`` keeps all.
        include_other_images: if True, additionally use ``other_image_id`` views.
        unique_other_images_only: drop additional images reused as main or other by any other product.
        subsample_per_category: maximum products retained per ABO ``product_type``.
        seed: seed for product-level shuffling and per-category subsampling.

    Return: examples grouped by product (views contiguous) in product-shuffled order,
        with ``product_index`` and ``item_index`` populated.
    """
    image_root = project_path / "data/raw_data/abo/images/small"
    metadata_df = load_abo_metadata(project_path / "data/raw_data/abo/metadata")
    images_df = pl.read_csv(
        str(project_path / "data/raw_data/abo/images/metadata/images.csv")
    )

    image_path_lookup: Dict[str, str] = dict(
        zip(images_df["image_id"].to_list(), images_df["path"].to_list())
    )

    image_usage_count: Dict[str, int] = {}
    if include_other_images and unique_other_images_only:
        for record in tqdm(
            metadata_df.iter_rows(named=True),
            total=metadata_df.height,
            desc="Counting image usage",
        ):
            main_img = record.get("main_image_id")
            if main_img:
                image_usage_count[main_img] = image_usage_count.get(main_img, 0) + 1
            other_imgs = record.get("other_image_id") or []
            if isinstance(other_imgs, list):
                for oid in other_imgs:
                    if oid:
                        image_usage_count[oid] = image_usage_count.get(oid, 0) + 1

    products: Dict[str, Dict[str, Any]] = {}

    for record in tqdm(
        metadata_df.iter_rows(named=True),
        total=metadata_df.height,
        desc="Building examples",
    ):
        res = get_product_data(
            record,
            preferred_lang=preferred_lang,
            strict_preferred_lang=strict_preferred_lang,
        )
        if not res:
            continue

        item_id = record.get("item_id")
        if not item_id:
            continue

        main_image_id = record.get("main_image_id")
        other_image_ids = record.get("other_image_id") or []
        if not isinstance(other_image_ids, list):
            other_image_ids = []

        candidate_ids: List[tuple] = []
        seen_ids = set()
        if main_image_id:
            candidate_ids.append((main_image_id, True))
            seen_ids.add(main_image_id)
        if include_other_images:
            for oid in other_image_ids:
                if not oid or oid in seen_ids:
                    continue
                if unique_other_images_only and image_usage_count.get(oid, 0) > 1:
                    continue
                candidate_ids.append((oid, False))
                seen_ids.add(oid)

        product_examples: List[Example] = []
        for image_id, is_main in candidate_ids:
            rel_path = image_path_lookup.get(image_id)
            if not rel_path:
                continue
            img_path = image_root / rel_path
            if not img_path.exists():
                continue

            product_examples.append(
                Example(
                    item_index=0,
                    item_id=item_id,
                    image_id=image_id,
                    image_file=str(img_path),
                    text=res["text"],
                    is_main_image=is_main,
                )
            )

        if not product_examples:
            continue

        ptype_field = record.get("product_type")
        product_type = None
        if isinstance(ptype_field, list) and ptype_field:
            for entry in ptype_field:
                tag = entry.get("language_tag", "") or ""
                if tag.lower().startswith("en"):
                    product_type = entry.get("value")
                    break
            if product_type is None:
                product_type = ptype_field[0].get("value")
        elif isinstance(ptype_field, str):
            product_type = ptype_field

        if max_images_per_product is not None:
            product_examples = product_examples[:max_images_per_product]

        new_score = res["score"]
        new_is_pref = res["is_preferred"]
        if item_id in products:
            old = products[item_id]
            replace = new_score > old["score"] or (
                new_score == old["score"] and new_is_pref and not old["is_preferred"]
            )
            if not replace:
                continue

        products[item_id] = {
            "score": new_score,
            "is_preferred": new_is_pref,
            "product_type": product_type,
            "examples": product_examples,
        }

    if subsample_per_category:
        rng = random.Random(seed)
        items_by_category: Dict[str, List[str]] = {}
        for iid, pdata in products.items():
            ptype = pdata.get("product_type")
            if ptype is not None:
                items_by_category.setdefault(ptype, []).append(iid)

        items_to_drop: set = set()
        for cat, max_count in subsample_per_category.items():
            ids_in_cat = items_by_category.get(cat, [])
            if len(ids_in_cat) <= max_count:
                continue
            ids_in_cat = sorted(ids_in_cat)
            rng_cat = random.Random(f"{seed}:{cat}")
            rng_cat.shuffle(ids_in_cat)
            items_to_drop.update(ids_in_cat[max_count:])
            print(
                f"Subsample category '{cat}': {len(ids_in_cat)} -> {max_count} "
                f"(dropping {len(ids_in_cat) - max_count})"
            )

        for iid in items_to_drop:
            del products[iid]

    product_ids = list(products.keys())
    random.Random(seed).shuffle(product_ids)

    if max_items:
        product_ids = product_ids[:max_items]

    prepared: List[Example] = []
    for product_index, product_id in enumerate(product_ids):
        for ex in products[product_id]["examples"]:
            ex.product_index = product_index
            prepared.append(ex)

    for i, ex in enumerate(prepared):
        ex.item_index = i

    return prepared


def split_by_product(
    examples: List[Example], n_train_products: int, n_val_products: Optional[int] = None
):
    """Split a flat list of examples on product boundaries.

    Args:
        examples: output of :func:`prepare_examples` with ``product_index`` populated.
        n_train_products: products assigned to the training split.
        n_val_products: products assigned to validation; ``None`` puts all remaining into validation.

    Return: tuple ``(train_examples, val_examples)``.
    """
    train, val = [], []
    val_upper = (
        n_train_products + n_val_products if n_val_products is not None else None
    )
    for ex in examples:
        if ex.product_index < n_train_products:
            train.append(ex)
        elif val_upper is None or ex.product_index < val_upper:
            val.append(ex)
    return train, val
