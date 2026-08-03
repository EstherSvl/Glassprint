"""Turning a selector into a mask.

A :class:`Selector` names *what* to select ("the background", "gold", "the
flowers"); this module works out *which pixels*. Selectors that need semantic
understanding try the optional local models first and fall back to colour and
tone heuristics, so the tool is useful with nothing downloaded.
"""

from __future__ import annotations

import sys
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


#: Pyodide reports this platform, and it is the one place where "install the
#: extra" is not advice — a browser tab cannot install torch, and never will be
#: able to. Telling someone to do the impossible reads as a broken tool.
IN_BROWSER = sys.platform == "emscripten"


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

    def fallback(self, what: str, instead: str, extra: str) -> None:
        """Record that an optional model was missing and something else ran.

        Deliberately not phrased as a failure. Nothing went wrong: the model is
        optional, the fallback is the documented path, and on a tablet it is the
        *only* path. The first version of this said "CLIPSeg unavailable
        (ModuleNotFoundError)" in the same warning colour the real problems use,
        which made a working tool look broken and offered a remedy that half the
        installs cannot perform.
        """
        if IN_BROWSER:
            self.note(f"{what} by {instead} — object models need a desktop install, not a tablet.")
        else:
            self.note(f"{what} by {instead}. For true object selection: pip install -e \".[{extra}]\"")

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
            self.fallback("Cut the subject out", "colour, not by shape", "smart")
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
            self.fallback("Matched what you described", "colour and tone", "smart")
            return None


#: How the eye weighs the channels when judging "is that the same colour".
_CHANNEL_WEIGHTS = np.array([0.30, 0.59, 0.11], dtype=np.float32)


def _border_samples(rgb: np.ndarray, ring: int) -> tuple[np.ndarray, np.ndarray]:
    """Border pixels and where on the canvas each one came from."""
    h, w = rgb.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    edge = np.zeros((h, w), dtype=bool)
    edge[:ring, :] = edge[-ring:, :] = True
    edge[:, :ring] = edge[:, -ring:] = True
    return rgb[edge], np.stack([ys[edge], xs[edge]], axis=1).astype(np.float32)


def _candidate_grounds(border: np.ndarray, limit: int = 4) -> list[np.ndarray]:
    """The handful of distinct colours the border is actually made of.

    The median used to stand in for this, and it is only the background when
    the background is most of the border. Artwork whose motifs run off the edge
    of the canvas — which is what a repeat *is* — can put more red along the
    border than white, and then the median is red and the tool confidently
    removes the flowers and keeps the paper.
    """
    quantised = np.round(border * 12.0).astype(np.int16)
    keys, index, counts = np.unique(quantised, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [border[index == bucket].mean(axis=0) for bucket in order]


def _ground_field(
    reference: np.ndarray,
    border: np.ndarray,
    where: np.ndarray,
    shape: tuple[int, int],
    radius: float,
) -> np.ndarray:
    """Predict the ground's colour at every pixel, not just its average.

    A studio backdrop falls off toward one corner and a scanned sheet is
    brighter under the lamp. Measuring every pixel against one colour then
    fails at both ends of the sweep: the tool keeps the top and bottom of the
    backdrop as though they were the subject.

    So a plane is fitted through the border pixels that match the reference —
    the lowest-order thing that can express "gets darker downwards". On a truly
    flat ground the fit comes back flat and this is exactly what it always was.
    """
    h, w = shape
    dist = np.sqrt((((border - reference[None, :]) ** 2) * _CHANNEL_WEIGHTS[None, :]).sum(axis=-1))
    belongs = dist < radius
    if belongs.sum() < 24:
        return np.broadcast_to(reference.astype(np.float32), (h, w, 3))

    ys, xs = where[belongs, 0] / max(h - 1, 1), where[belongs, 1] / max(w - 1, 1)
    design = np.stack([np.ones_like(ys), ys, xs], axis=1)
    try:
        coeffs, *_ = np.linalg.lstsq(design, border[belongs], rcond=None)
    except np.linalg.LinAlgError:
        return np.broadcast_to(reference.astype(np.float32), (h, w, 3))

    grid_y, grid_x = np.mgrid[0:h, 0:w]
    full = np.stack(
        [np.ones((h, w)), grid_y / max(h - 1, 1), grid_x / max(w - 1, 1)], axis=-1
    ).astype(np.float32)
    return np.clip(full @ coeffs.astype(np.float32), 0.0, 1.0)


def _perimeter_reach(mask: np.ndarray) -> float:
    """How much of the canvas edge this region runs along.

    The test that tells a ground from a motif. Background touches most of the
    way round; a flower bleeding off one corner does not, however much of the
    border it happens to occupy there.
    """
    edge = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float((edge > 0.5).mean())


def background_mask(
    raster: Raster, tolerance: float = 1.0, backends: "Backends | None" = None
) -> np.ndarray:
    """Detect the ground the artwork sits on, working inwards from the border.

    This is the no-model path, and it is the right one for most exported
    artwork: patterns and motifs from Procreate or Affinity usually sit on a
    flat white, transparent or single-colour ground.

    Two things it does *not* assume, both learned from artwork that broke it:
    that the border is mostly background, and that the ground is one colour.
    """
    alpha = raster.alpha_f
    if (alpha < 0.5).mean() > 0.02:
        # The file already carries transparency -- trust the artist's cutout.
        return masks.clean(1.0 - alpha)

    rgb = raster.rgb_f
    h, w = rgb.shape[:2]
    ring = max(1, int(round(min(h, w) * 0.01)))
    border, where = _border_samples(rgb, ring)
    radius = 0.13 * max(tolerance, 0.05)

    # Try each colour the border is made of and keep whichever behaves like a
    # ground. Two properties together, because neither settles it alone: a
    # ground runs round the edge of the canvas, *and* it covers ground. Reach
    # alone picks the red dots out of a repeat, where motifs legitimately own
    # most of the border; area alone picks a subject that fills the frame.
    best_score = -1.0
    best: np.ndarray | None = None
    best_dist: np.ndarray | None = None
    for reference in _candidate_grounds(border):
        field = _ground_field(reference, border, where, (h, w), radius)
        dist = np.sqrt((((rgb - field) ** 2) * _CHANNEL_WEIGHTS[None, None, :]).sum(axis=-1))
        near = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)

        # Only count regions connected to the edge, so a white flower centre in
        # the middle of the artwork is not mistaken for background.
        connected = masks.despeckle(
            masks.touching_border(near, threshold=0.5), min_area_fraction=0.0005
        )
        score = _perimeter_reach(connected) * float(connected.mean())
        if score > best_score:
            best_score, best, best_dist = score, masks.clean(
                np.minimum(near, masks.feather(connected, 1.0) * 1.2)
            ), dist

    if best is None:
        return masks.zeros((h, w))

    if backends is not None:
        if best_score < 0.25:
            backends.note(
                "Could not find a clear ground behind this artwork — nothing in it both "
                "runs round the edge and covers much area. Removing the background by "
                "colour will be rough: cut it out before importing, or install the "
                "'smart' extra for a subject cutout."
            )
        elif _too_close_to_call(best, best_dist, radius):
            backends.note(
                "What was kept is barely a different colour from what was removed — a pale "
                "subject on pale ground, or a cast shadow on the paper. No tolerance "
                "setting separates those, so expect halos and holes: cut the artwork out "
                "before importing, or install the 'smart' extra for a subject cutout."
            )
    return best


def _too_close_to_call(background: np.ndarray, dist: np.ndarray, radius: float) -> bool:
    """Whether the subject is actually a different colour from its ground.

    Worth measuring rather than assuming, because when it is not there is no
    setting that helps. White orchids photographed on white paper were the case
    that showed it: at every tolerance the flowers and the shadow they cast
    move together, and the best the tool ever manages is to keep two fifths of
    the flower and a fifth of the paper. Tuning the number produces a different
    bad answer, not a good one.

    So the tool says which situation it is in instead of quietly picking one.
    """
    kept = (1.0 - background) > 0.5
    if not kept.any():
        return False
    return float(np.median(dist[kept])) < radius * 1.5


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
        return background_mask(raster, tolerance=selector.tolerance, backends=backends)

    if kind == "subject":
        found = backends.subject_mask(raster)
        if found is not None:
            return found
        return masks.invert(background_mask(raster, tolerance=selector.tolerance, backends=backends))

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
