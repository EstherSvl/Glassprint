"""Soft-mask helpers.

A mask is a float32 array in 0..1 with the same height and width as the image
it belongs to. Soft edges matter for print: a hard 1-bit cutout shows stair-step
artefacts on curves, which a UV printer reproduces faithfully.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def zeros(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def ones(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)


def clean(mask: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(mask.astype(np.float32), nan=0.0), 0.0, 1.0)


def union(*masks: np.ndarray) -> np.ndarray:
    out = clean(masks[0])
    for m in masks[1:]:
        out = np.maximum(out, clean(m))
    return out


def intersect(*masks: np.ndarray) -> np.ndarray:
    out = clean(masks[0])
    for m in masks[1:]:
        out = out * clean(m)
    return out


def subtract(mask: np.ndarray, other: np.ndarray) -> np.ndarray:
    return clean(clean(mask) * (1.0 - clean(other)))


def invert(mask: np.ndarray) -> np.ndarray:
    return 1.0 - clean(mask)


def feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian-soften the mask edge by ``radius`` pixels."""
    if radius <= 0:
        return clean(mask)
    img = Image.fromarray((clean(mask) * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(img, dtype=np.float32) / 255.0


def grow(mask: np.ndarray, pixels: float) -> np.ndarray:
    """Spread (positive) or choke (negative) the mask by ``pixels``."""
    pixels = float(pixels)
    if abs(pixels) < 0.5:
        return clean(mask)
    size = int(abs(round(pixels))) * 2 + 1
    data = clean(mask)
    if pixels > 0:
        return clean(ndimage.grey_dilation(data, size=(size, size)))
    return clean(ndimage.grey_erosion(data, size=(size, size)))


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (clean(mask) >= threshold).astype(bool)


def fill_holes(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    filled = ndimage.binary_fill_holes(binarize(mask, threshold))
    return union(mask, filled.astype(np.float32))


def despeckle(mask: np.ndarray, min_area_fraction: float = 0.0008, threshold: float = 0.5) -> np.ndarray:
    """Drop connected blobs smaller than ``min_area_fraction`` of the image."""
    binary = binarize(mask, threshold)
    if not binary.any():
        return clean(mask)
    labels, count = ndimage.label(binary)
    if count == 0:
        return clean(mask)
    min_area = max(1, int(min_area_fraction * binary.size))
    sizes = ndimage.sum(binary, labels, index=np.arange(1, count + 1))
    keep = np.concatenate([[False], sizes >= min_area])
    return clean(mask) * keep[labels].astype(np.float32)


def largest_component(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    binary = binarize(mask, threshold)
    if not binary.any():
        return clean(mask)
    labels, count = ndimage.label(binary)
    if count <= 1:
        return clean(mask)
    sizes = ndimage.sum(binary, labels, index=np.arange(1, count + 1))
    winner = int(np.argmax(sizes)) + 1
    return clean(mask) * (labels == winner).astype(np.float32)


def component_count(mask: np.ndarray, threshold: float = 0.5) -> int:
    binary = binarize(mask, threshold)
    if not binary.any():
        return 0
    _, count = ndimage.label(binary)
    return int(count)


def coverage(mask: np.ndarray) -> float:
    return float(clean(mask).mean())


def bbox(mask: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    """Bounding box of the mask as ``(left, top, right, bottom)``, exclusive."""
    binary = binarize(mask, threshold)
    if not binary.any():
        return None
    rows = np.flatnonzero(binary.any(axis=1))
    cols = np.flatnonzero(binary.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def touching_border(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Keep only components that touch the image border."""
    binary = binarize(mask, threshold)
    if not binary.any():
        return clean(mask)
    labels, count = ndimage.label(binary)
    if count == 0:
        return clean(mask)
    edge_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    edge_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    edge_labels.discard(0)
    if not edge_labels:
        return zeros(mask.shape)
    keep = np.isin(labels, list(edge_labels))
    return clean(mask) * keep.astype(np.float32)


def resize(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return clean(mask)
    img = Image.fromarray((clean(mask) * 255).astype(np.uint8), mode="L")
    img = img.resize((max(1, width), max(1, height)), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
