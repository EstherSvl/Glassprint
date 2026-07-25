"""Turning a selector into a mask.

A :class:`Selector` names *what* to select ("the background", "gold", "the
flowers"); this module works out *which pixels*. Selectors that need semantic
understanding try the optional local models first and fall back to colour and
tone heuristics, so the tool is useful with nothing downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import colors, masks
from .raster import Raster

SelectorKind = str  # all | alpha | background | subject | color | tone | semantic | rest


@dataclass
class Selector:
    kind: SelectorKind
    value: str | None = None
    tolerance: float = 1.0
    color_hint: str | None = None

    def describe(self) -> str:
        if self.kind in {"all", "rest"}:
            return self.kind
        if self.value:
            return f"{self.kind}:{self.value}"
        return self.kind


class Backends:
    """Lazily-loaded optional models, with graceful degradation."""

    def __init__(self, *, allow_models: bool = True) -> None:
        self.allow_models = allow_models
        self.notes: list[str] = []
        self._rembg = None
        self._rembg_failed = False
        self._clipseg = None
        self._clipseg_failed = False

    # -- availability -------------------------------------------------------

    @staticmethod
    def probe() -> dict[str, bool]:
        import importlib.util

        def has(name: str) -> bool:
            try:
                return importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                return False

        return {
            "rembg": has("rembg"),
            "clipseg": has("transformers") and has("torch"),
            "anthropic": has("anthropic"),
        }

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    # -- rembg (subject cutout) --------------------------------------------

    def subject_mask(self, raster: Raster) -> np.ndarray | None:
        if not self.allow_models or self._rembg_failed:
            return None
        try:
            if self._rembg is None:
                from rembg import new_session, remove  # type: ignore

                self._rembg = (remove, new_session("u2net"))
            remove, session = self._rembg
            cut = remove(raster.to_pil().convert("RGB"), session=session, only_mask=True)
            return masks.clean(np.asarray(cut, dtype=np.float32) / 255.0)
        except Exception as exc:  # pragma: no cover - depends on optional install
            self._rembg_failed = True
            self.note(f"rembg unavailable ({exc.__class__.__name__}); using colour-based background removal")
            return None

    # -- CLIPSeg (text-driven regions) -------------------------------------

    def semantic_mask(self, raster: Raster, prompt: str) -> np.ndarray | None:
        if not self.allow_models or self._clipseg_failed:
            return None
        try:
            if self._clipseg is None:
                import torch  # type: ignore
                from transformers import (  # type: ignore
                    CLIPSegForImageSegmentation,
                    CLIPSegProcessor,
                )

                name = "CIDAS/clipseg-rd64-refined"
                processor = CLIPSegProcessor.from_pretrained(name)
                model = CLIPSegForImageSegmentation.from_pretrained(name)
                model.eval()
                self._clipseg = (torch, processor, model)
            torch, processor, model = self._clipseg

            image = raster.scaled_to_fit(768).to_pil().convert("RGB")
            inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
            return masks.resize(probs, raster.width, raster.height)
        except Exception as exc:  # pragma: no cover - depends on optional install
            self._clipseg_failed = True
            self.note(
                f"CLIPSeg unavailable ({exc.__class__.__name__}); "
                "falling back to colour/tone matching for described regions"
            )
            return None


def background_mask(raster: Raster, tolerance: float = 1.0) -> np.ndarray:
    """Detect a flat-ish background by growing inwards from the image border.

    This is the no-model path, and it is the right one for most exported
    artwork: patterns and motifs from Procreate or Affinity usually sit on a
    flat white, transparent or single-colour ground.
    """
    alpha = raster.alpha_f
    if (alpha < 0.5).mean() > 0.02:
        # The file already carries transparency -- trust the artist's cutout.
        return masks.clean(1.0 - alpha)

    rgb = raster.rgb_f
    h, w = rgb.shape[:2]
    ring = max(1, int(round(min(h, w) * 0.01)))
    border = np.concatenate(
        [
            rgb[:ring, :, :].reshape(-1, 3),
            rgb[-ring:, :, :].reshape(-1, 3),
            rgb[:, :ring, :].reshape(-1, 3),
            rgb[:, -ring:, :].reshape(-1, 3),
        ]
    )
    reference = np.median(border, axis=0)

    weights = np.array([0.30, 0.59, 0.11], dtype=np.float32)
    dist = np.sqrt((((rgb - reference[None, None, :]) ** 2) * weights[None, None, :]).sum(axis=-1))
    radius = 0.13 * max(tolerance, 0.05)
    near = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)

    # Only count regions connected to the edge, so a white flower centre in the
    # middle of the artwork is not mistaken for background.
    connected = masks.touching_border(near, threshold=0.5)
    connected = masks.despeckle(connected, min_area_fraction=0.0005)
    return masks.clean(np.minimum(near, masks.feather(connected, 1.0) * 1.2))


def resolve(
    selector: Selector,
    raster: Raster,
    backends: Backends | None = None,
) -> np.ndarray:
    """Produce the mask a selector refers to."""
    backends = backends or Backends()
    shape = (raster.height, raster.width)
    kind = selector.kind

    if kind in {"all", "rest"}:
        return masks.ones(shape)

    if kind == "alpha":
        return masks.clean(raster.alpha_f)

    if kind == "background":
        return background_mask(raster, tolerance=selector.tolerance)

    if kind == "subject":
        found = backends.subject_mask(raster)
        if found is not None:
            return found
        return masks.invert(background_mask(raster, tolerance=selector.tolerance))

    if kind == "color":
        if not selector.value:
            raise ValueError("colour selector needs a value")
        return colors.color_mask(raster.rgb_f, selector.value, tolerance=selector.tolerance)

    if kind == "tone":
        if not selector.value:
            raise ValueError("tone selector needs a value")
        return colors.tone_mask(raster.rgb_f, selector.value, tolerance=selector.tolerance)

    if kind == "semantic":
        prompt = (selector.value or "").strip()
        if not prompt:
            return masks.ones(shape)
        found = backends.semantic_mask(raster, prompt)
        if found is not None:
            # CLIPSeg output is a soft heat map; lift the mid-tones so the
            # result reads as a selection rather than a blur.
            return masks.clean(np.clip((found - 0.35) / 0.35, 0.0, 1.0))
        return _semantic_fallback(selector, raster, backends)

    raise ValueError(f"unknown selector kind {kind!r}")


def _semantic_fallback(selector: Selector, raster: Raster, backends: Backends) -> np.ndarray:
    """Best effort when no segmentation model is installed."""
    if selector.color_hint:
        backends.note(
            f"matched '{selector.value}' by its colour ({selector.color_hint}) "
            "— install the 'smart' extra for true object selection"
        )
        return colors.color_mask(raster.rgb_f, selector.color_hint, tolerance=selector.tolerance)

    backends.note(
        f"could not identify '{selector.value}' without a segmentation model; "
        "treated it as the foreground subject"
    )
    found = backends.subject_mask(raster)
    if found is not None:
        return found
    return masks.invert(background_mask(raster, tolerance=selector.tolerance))


@dataclass
class MaskOp:
    action: str  # "keep" | "remove"
    selector: Selector
    feather: float = 0.0


@dataclass
class MaskPlan:
    ops: list[MaskOp] = field(default_factory=list)
    source: str = "rules"  # rules | claude | default
    explanation: str = ""

    def describe(self) -> str:
        if not self.ops:
            return "keep everything"
        return "; ".join(f"{op.action} {op.selector.describe()}" for op in self.ops)


def evaluate(plan: MaskPlan, raster: Raster, backends: Backends | None = None) -> np.ndarray:
    """Run a mask plan against an image.

    Semantics: if the plan contains any ``keep``, we start from nothing and add
    them; otherwise we start from everything (or the file's own alpha). Then
    every ``remove`` is subtracted.
    """
    backends = backends or Backends()
    shape = (raster.height, raster.width)
    keeps = [op for op in plan.ops if op.action == "keep"]
    removes = [op for op in plan.ops if op.action == "remove" and op.selector.kind != "rest"]

    if keeps:
        result = masks.zeros(shape)
        for op in keeps:
            layer = resolve(op.selector, raster, backends)
            if op.feather:
                layer = masks.feather(layer, op.feather)
            result = masks.union(result, layer)
    else:
        result = masks.clean(raster.alpha_f) if raster.has_alpha else masks.ones(shape)

    for op in removes:
        layer = resolve(op.selector, raster, backends)
        if op.feather:
            layer = masks.feather(layer, op.feather)
        result = masks.subtract(result, layer)

    # An image with real transparency should never gain opaque pixels.
    if raster.has_alpha:
        result = masks.intersect(result, raster.alpha_f)
    return masks.clean(result)
