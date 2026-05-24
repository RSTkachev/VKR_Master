"""General-purpose utilities: random seeding, text normalisation, device selection, model loading."""

import re
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: value applied to all RNGs.

    Return: None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim stray punctuation.

    Args:
        text: input string.

    Return: single-spaced string without trailing whitespace or punctuation.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip(" \n\t,.;:")


def normalize_text_value(text: Optional[str]) -> Optional[str]:
    """Normalise an optional text value to a clean string or ``None``.

    Args:
        text: raw value that may be ``None``, empty, or contain stray whitespace.

    Return: cleaned string, or ``None`` when the input is empty after cleaning.
    """
    if text is None:
        return None
    text = str(text)
    text = clean_whitespace(text)
    if not text:
        return None
    return text


def choose_device(device_arg: str) -> str:
    """Resolve a device string, mapping ``"auto"`` to the best available device.

    Args:
        device_arg: ``"auto"`` or an explicit torch device specifier.

    Return: concrete device string usable with ``torch.device``.
    """
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move tensor values in a batch dictionary to the target device.

    Args:
        batch: mapping whose tensor values are relocated; other values pass through.
        device: target torch device.

    Return: new dictionary with tensors on ``device``.
    """
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_model_bundle(model_name: str, device: str):
    """Load a Hugging Face model and its processor and move them to ``device``.

    Args:
        model_name: Hugging Face model id or local path.
        device: target torch device; CUDA devices switch to bfloat16.

    Return: tuple ``(model, processor)``.
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True}
    if device.startswith("cuda"):
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(
        model_name, attn_implementation="flash_attention_2", **model_kwargs
    )
    model.to(device)
    return model, processor


def _to_numpy_f32(x) -> np.ndarray:
    """Convert a torch tensor or array-like to a contiguous float32 NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)
