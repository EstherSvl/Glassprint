"""The composition pipeline.

Base image + overlay artwork + a sentence about what to keep, in; a composite
and a standalone overlay layer, out.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace

import numpy as np

from . import masks, nl, pattern, recolor, segment
from .pattern import Box, PatternInfo, Placement
from .raster import Raster
from .recolor import ColorSpec
from .segment import Backends, MaskPlan

TARGET_MODES = ("alpha", "describe", "largest", "full", "rect")
BLEND_MODES = ("normal", "multiply", "screen", "overlay")


@dataclass
class ComposeSpec:
    # What of the overlay to keep.
    keep: str = ""
    tolerance: float = 1.0
    use_claude: bool = False

    # Where on the base it goes.
    target: str = "alpha"
    target_describe: str = ""
    target_rect: tuple[float, float, float, float] | None = None  # fractions of the base

    # How it sits there.
    clip_to_shape: bool = True
    shape_grow: float = 0.0        # pixels; negative chokes the shape inwards
    shape_feather: float = 0.0     # pixels
    edge_feather: float = 0.0      # pixels, on the overlay cutout
    opacity: float = 1.0
    blend: str = "normal"

    placement: Placement = field(default_factory=Placement)
    color: ColorSpec = field(default_factory=ColorSpec)

    def validated(self) -> "ComposeSpec":
        if self.target not in TARGET_MODES:
            raise ValueError(f"unknown target mode {self.target!r}; choose from {', '.join(TARGET_MODES)}")
        if self.blend not in BLEND_MODES:
            raise ValueError(f"unknown blend mode {self.blend!r}; choose from {', '.join(BLEND_MODES)}")
        return dataclass_replace(self, placement=self.placement.normalised())


@dataclass
class ComposeResult:
    composite: Raster
    overlay_layer: Raster
    base: Raster
    shape_mask: np.ndarray
    cutout_mask: np.ndarray
    plan: MaskPlan
    info: PatternInfo
    box: Box
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        left, top, right, bottom = self.box
        dpi_x, dpi_y = self.base.effective_dpi
        return {
            "base_size": list(self.base.size),
            "base_dpi": [round(dpi_x, 2), round(dpi_y, 2)],
            "base_size_mm": [round(v, 2) for v in self.base.size_mm],
            "shape_box": [left, top, right, bottom],
            "shape_coverage": round(masks.coverage(self.shape_mask), 4),
            "cutout_coverage": round(masks.coverage(self.cutout_mask), 4),
            "plan": self.plan.describe(),
            "plan_source": self.plan.source,
            "plan_explanation": self.plan.explanation,
            "pattern": {
                "is_pattern": self.info.is_pattern,
                "seamless": self.info.seamless,
                "components": self.info.components,
                "coverage": round(self.info.coverage, 4),
                "suggested_fit": self.info.suggested_fit,
                "suggested_repeats": self.info.suggested_repeats,
                "reason": self.info.reason,
            },
            "notes": self.notes,
        }


def resolve_shape(
    base: Raster,
    spec: ComposeSpec,
    backends: Backends,
) -> tuple[np.ndarray, list[str]]:
    """Work out which part of the base image the overlay should land on."""
    notes: list[str] = []
    shape = (base.height, base.width)
    mode = spec.target

    if mode == "alpha" and not base.has_alpha:
        notes.append(
            "The base image has no transparency, so there is no cut-out shape to fill — "
            "used the largest solid region instead. Export from Procreate or Affinity as "
            "PNG with a transparent background to target an exact shape."
        )
        mode = "largest"

    if mode == "full":
        return masks.ones(shape), notes

    if mode == "alpha":
        return masks.clean(base.alpha_f), notes

    if mode == "rect":
        if not spec.target_rect:
            notes.append("No rectangle given for the target area — used the whole canvas.")
            return masks.ones(shape), notes
        left, top, right, bottom = spec.target_rect
        mask = masks.zeros(shape)
        x0 = int(round(np.clip(left, 0, 1) * base.width))
        x1 = int(round(np.clip(right, 0, 1) * base.width))
        y0 = int(round(np.clip(top, 0, 1) * base.height))
        y1 = int(round(np.clip(bottom, 0, 1) * base.height))
        mask[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)] = 1.0
        return mask, notes

    if mode == "describe":
        instruction = spec.target_describe.strip()
        if not instruction:
            notes.append("No description given for the target area — used the whole canvas.")
            return masks.ones(shape), notes
        plan = nl.build_plan(
            instruction, base, use_claude=spec.use_claude, tolerance=spec.tolerance
        )
        mask = segment.evaluate(plan, base, backends)
        mask = masks.fill_holes(masks.despeckle(mask, 0.0008))
        if masks.coverage(mask) < 0.001:
            notes.append(
                f"Could not find '{instruction}' on the base image — used the whole canvas."
            )
            return masks.ones(shape), notes
        return mask, notes

    # largest solid region
    background = segment.background_mask(base, tolerance=spec.tolerance)
    mask = masks.invert(background)
    mask = masks.fill_holes(masks.despeckle(mask, 0.001))
    mask = masks.largest_component(mask)
    if masks.coverage(mask) < 0.001:
        notes.append("Could not find a distinct shape on the base image — used the whole canvas.")
        return masks.ones(shape), notes
    return mask, notes


def compose(
    base: Raster,
    overlay: Raster,
    spec: ComposeSpec | None = None,
    backends: Backends | None = None,
) -> ComposeResult:
    spec = (spec or ComposeSpec()).validated()
    backends = backends or Backends()
    notes: list[str] = []

    # 1. Which part of the base are we filling?
    shape_mask, shape_notes = resolve_shape(base, spec, backends)
    notes.extend(shape_notes)

    shaped = shape_mask
    if spec.shape_grow:
        shaped = masks.grow(shaped, spec.shape_grow)
    if spec.shape_feather:
        shaped = masks.feather(shaped, spec.shape_feather)

    box = masks.bbox(shaped, threshold=0.35) or (0, 0, base.width, base.height)

    # 2. Which part of the overlay are we using?
    plan = nl.build_plan(spec.keep, overlay, use_claude=spec.use_claude, tolerance=spec.tolerance)
    cutout = segment.evaluate(plan, overlay, backends)
    if spec.edge_feather:
        cutout = masks.feather(cutout, spec.edge_feather)

    # 3. Read the artwork's language and place it.
    info = pattern.analyse(
        overlay,
        cutout,
        target_width_px=box[2] - box[0],
        target_dpi=base.effective_dpi[0],
    )
    art = pattern.apply_cutout(overlay, cutout)
    art = recolor.apply(art, spec.color)

    placed = pattern.place(
        art,
        base.size,
        box,
        spec.placement,
        info,
        target_dpi=base.effective_dpi[0],
    )

    # 4. Clip to the shape and apply opacity.
    layer_alpha = placed[:, :, 3].astype(np.float32) / 255.0
    if spec.clip_to_shape:
        layer_alpha = layer_alpha * shaped
    layer_alpha = layer_alpha * float(np.clip(spec.opacity, 0.0, 1.0))

    overlay_layer = placed.copy()
    overlay_layer[:, :, 3] = np.clip(layer_alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # 5. Blend over the base.
    composite = _blend_over(base.rgba, overlay_layer, spec.blend)

    notes.extend(backends.notes)
    return ComposeResult(
        composite=Raster(composite, dpi=base.dpi, source_format=base.source_format, name=base.name),
        overlay_layer=Raster(overlay_layer, dpi=base.dpi, name=(overlay.name or "overlay")),
        base=base,
        shape_mask=shaped,
        cutout_mask=cutout,
        plan=plan,
        info=info,
        box=box,
        notes=notes,
    )


def _blend_over(base_rgba: np.ndarray, layer_rgba: np.ndarray, mode: str) -> np.ndarray:
    base_rgb = base_rgba[:, :, :3].astype(np.float32) / 255.0
    base_a = base_rgba[:, :, 3].astype(np.float32) / 255.0
    layer_rgb = layer_rgba[:, :, :3].astype(np.float32) / 255.0
    layer_a = layer_rgba[:, :, 3].astype(np.float32) / 255.0

    if mode == "multiply":
        blended = base_rgb * layer_rgb
    elif mode == "screen":
        blended = 1.0 - (1.0 - base_rgb) * (1.0 - layer_rgb)
    elif mode == "overlay":
        blended = np.where(
            base_rgb <= 0.5,
            2.0 * base_rgb * layer_rgb,
            1.0 - 2.0 * (1.0 - base_rgb) * (1.0 - layer_rgb),
        )
    else:
        blended = layer_rgb

    # Blend modes only apply where the base is actually opaque; over empty
    # canvas the overlay keeps its own colour.
    src_rgb = base_a[:, :, None] * blended + (1.0 - base_a[:, :, None]) * layer_rgb

    out_a = layer_a + base_a * (1.0 - layer_a)
    safe = np.maximum(out_a, 1e-6)[:, :, None]
    out_rgb = (
        src_rgb * layer_a[:, :, None] + base_rgb * base_a[:, :, None] * (1.0 - layer_a[:, :, None])
    ) / safe

    out = np.empty_like(base_rgba)
    out[:, :, :3] = np.clip(out_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out
