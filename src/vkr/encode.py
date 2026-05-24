"""Batched text and image encoding helpers for dual-encoder retrieval."""

from typing import List, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from vkr.utils import move_to_device
from vkr.preprocessing import load_images


def encode_texts(model, processor, texts, batch_size, device, model_name: str):
    """Encode strings into L2-normalised text embeddings.

    Args:
        model: model exposing ``get_text_features``.
        processor: tokenizer/processor matching ``model``.
        texts: input strings.
        batch_size: number of texts per forward pass.
        device: torch device for inputs.
        model_name: model id; SigLIP variants use lowercased fixed-length padding (64).

    Return: float32 CPU tensor of shape ``(len(texts), embedding_dim)``, L2-normalised.
    """
    embeddings = []

    for start in tqdm(
        range(0, len(texts), batch_size), desc="Encoding texts", leave=False
    ):
        batch_texts = list(texts[start : start + batch_size])

        if "siglip" in model_name:
            inputs = processor(
                text=[text.lower() for text in batch_texts],
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            )

        else:
            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        inputs = move_to_device(inputs, device)

        with torch.no_grad():
            feats = model.get_text_features(**inputs).pooler_output

        if feats.ndim == 3:
            feats = feats[:, 0, :]

        feats = F.normalize(feats.float(), dim=-1)
        embeddings.append(feats.cpu())

    return torch.cat(embeddings, dim=0)


def encode_images(
    model,
    processor,
    image_paths: Sequence[str],
    batch_size: int,
    device: str,
    model_name: str = "",
):
    """Encode image files into L2-normalised image embeddings.

    Args:
        model: model exposing ``get_image_features`` or an image-embedding output.
        processor: processor matching ``model``.
        image_paths: paths to image files.
        batch_size: number of images per forward pass.
        device: torch device for inputs.
        model_name: kept for symmetry with :func:`encode_texts`.

    Return: float32 CPU tensor of shape ``(len(image_paths), embedding_dim)``, L2-normalised.
    """
    embeddings: List[torch.Tensor] = []
    model_dtype = next(model.parameters()).dtype

    for start in tqdm(
        range(0, len(image_paths), batch_size), desc="Encoding images", leave=False
    ):
        batch_paths = list(image_paths[start : start + batch_size])
        images = load_images(batch_paths)

        inputs = processor(images=images, return_tensors="pt")
        inputs = move_to_device(inputs, device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(**inputs).pooler_output
            else:
                outputs = model(**inputs).pooler_output
                feats = getattr(outputs, "image_embeds", outputs[0])

        if feats.ndim == 3:
            feats = feats[:, 0, :]

        feats = F.normalize(feats.float(), dim=-1)
        embeddings.append(feats.cpu())

    return torch.cat(embeddings, dim=0)
