"""Shared prompts and helpers for VLM-based reranking (pointwise and listwise)."""

import re
from typing import Any, Dict, List, Sequence

POINTWISE_SYSTEM_PROMPT = (
    "You are a strict evaluator for e-commerce product search. Your task is "
    "to determine whether a product description correctly identifies the main "
    "product in an image. Be strict: even similar but different products "
    "should be marked as NO."
)


LISTWISE_SYSTEM_PROMPT = (
    "You are an expert evaluator for an e-commerce product search system. "
    "Your task is to rank candidate product descriptions by how well they "
    "match a given product image. Be precise: prefer the description that "
    "specifically identifies the main product, not just any plausible match."
)


def build_pairs_for_rerank(
    image_paths: Sequence[str],
    texts: Sequence[str],
    type_retrieve: str,
    topk_indices,
) -> List[List[Dict[str, Any]]]:
    """Build candidate (image, text) pairs from stage-1 top-K indices.

    Args:
        image_paths: query image paths ordered by query index.
        texts: query texts ordered by query index.
        type_retrieve: ``"i2t"`` for image-to-text, otherwise text-to-image.
        topk_indices: indices of top-K candidates per query.

    Return: list of per-query candidate lists; each candidate is a dict with
        ``image``, ``text`` and ``label`` (True for the ground-truth pair).
    """
    pairs: List[List[Dict[str, Any]]] = []
    for i, indices in enumerate(topk_indices):
        cur_pairs: List[Dict[str, Any]] = []
        for j in indices:
            image_index, text_index = (i, j) if type_retrieve == "i2t" else (j, i)
            cur_pairs.append(
                {
                    "image": image_paths[image_index],
                    "text": texts[text_index],
                    "label": i == j,
                }
            )
        pairs.append(cur_pairs)
    return pairs


def build_prompt(image_path: str, text: str) -> List[Dict[str, Any]]:
    """Build the chat-style messages payload for pointwise yes/no scoring.

    Args:
        image_path: path to the image displayed to the model.
        text: candidate product description.

    Return: messages payload for ``processor.apply_chat_template``.
    """
    text_prompt = f"""Look carefully at the image and the description
Does the following product description match the image?

Description:
{text}

Answer with only one word: YES or NO.
"""

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": POINTWISE_SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        },
    ]
    return messages


def build_listwise_prompt(image_path: str, texts: List[str]) -> List[Dict[str, Any]]:
    """Build a single listwise-ranking messages payload for K candidates.

    Args:
        image_path: path to the query image.
        texts: candidate descriptions in the order shown to the model.

    Return: messages payload for ``processor.apply_chat_template``.
    """
    candidates_block = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    k = len(texts)
    user_text = (
        "Look at the product image. Below are candidate descriptions, numbered "
        f"1 to {k}.\n\n"
        f"Candidates:\n{candidates_block}\n\n"
        f"Rank the candidates from best to worst match for the image. "
        f"Respond with only the {k} numbers separated by commas, in ranking "
        f'order from best to worst, like: "3, 7, 1, 5, 9, 2, 4, 8, 6, 10". '
        f"Do not include explanations."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": LISTWISE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def parse_listwise_ranking(response: str, k: int) -> List[int]:
    """Extract a permutation of ``[1..K]`` from the model's free-text response.

    Args:
        response: raw text returned by the VLM.
        k: number of candidates ranked.

    Return: length-``k`` list of 1-based indices in best-to-worst order;
        missing indices are appended in natural order.
    """
    raw_numbers = [int(x) for x in re.findall(r"\d+", response)]
    seen: set = set()
    valid_ranking: List[int] = []
    for n in raw_numbers:
        if 1 <= n <= k and n not in seen:
            seen.add(n)
            valid_ranking.append(n)
    for n in range(1, k + 1):
        if n not in seen:
            valid_ranking.append(n)
    return valid_ranking[:k]
