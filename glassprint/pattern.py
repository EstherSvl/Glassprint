"""Reading an overlay's "language" and placing it onto the target shape.

The interesting part here is deciding *how* a piece of art wants to be applied.
A repeating pattern should tile at a sensible physical repeat size; a single
motif should be scaled to fit once. :func:`analyse` makes that call, and
:func:`place` carries it out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import masks
from .raster import MM_PER_INCH, Raster, mm_to_px

Box = tuple[int, int, int, int]  # left, top, right, bottom (right/bottom exclusive)

FIT_MODES = ("auto", "shape", "contain", "cover", "tile", "stretch")


@dataclass
class Placement:
    fit: str = "auto"
    repeat_across: float | None = None   # how many repeats across the shape width
    repeat_mm: float | None = None       # physical size of one repeat
    scale: float = 1.0
    rotation: float = 0.0                # degrees, clockwise
    offset_x: float = 0.0                # fraction of the target box width
    offset_y: float = 0.0
    mirror: str = "auto"                 # auto | on | off
    flip_h: bool = False
    flip_v: bool = False

    def normalised(self) -> "Placement":
        if self.fit not in FIT_MODES:
            raise ValueError(f"unknown fit mode {self.fit!r}; choose from {', '.join(FIT_MODES)}")
        return self


@dataclass
class PatternInfo:
    """What we worked out about the overlay artwork."""

    is_pattern: bool
    seamless: bool
    components: int
    coverage: float
    suggested_fit: str
    suggested_repeats: float
    reason: str = ""
    notes: list[str] = field(default_factory=list)


def analyse(
    overlay: Raster,
    cutout: np.ndarray,
    *,
    target_width_px: int | None = None,
    target_dpi: float | None = None,
) -> PatternInfo:
    """Decide whether the artwork reads as a repeating pattern or a single motif."""
    small = masks.resize(cutout, min(overlay.width, 512), min(overlay.height, 512))
    cover = masks.coverage(small)
    components = masks.component_count(masks.despeckle(small, 0.0005))

    is_pattern = components >= 6 or (components >= 3 and cover > 0.45) or cover > 0.75
    seamless = _looks_seamless(overlay, cutout)

    if is_pattern:
        reason = f"{components} separate marks covering {cover:.0%} of the canvas"
    else:
        reason = f"{components} shape{'s' if components != 1 else ''} covering {cover:.0%} of the canvas"

    repeats = _suggest_repeats(
        overlay,
        is_pattern=is_pattern,
        target_width_px=target_width_px,
        target_dpi=target_dpi,
    )
    return PatternInfo(
        is_pattern=is_pattern,
        seamless=seamless,
        components=components,
        coverage=cover,
        suggested_fit="tile" if is_pattern else "shape",
        suggested_repeats=repeats,
        reason=reason,
    )


def _looks_seamless(overlay: Raster, cutout: np.ndarray) -> bool:
    """Do opposite edges match closely enough to tile without mirroring?"""
    rgb = overlay.rgb_f
    alpha = cutout[:, :, None]
    left, right = rgb[:, :2] * alpha[:, :2], rgb[:, -2:] * alpha[:, -2:]
    top, bottom = rgb[:2, :] * alpha[:2, :], rgb[-2:, :] * alpha[-2:, :]
    horizontal = float(np.abs(left - right).mean())
    vertical = float(np.abs(top - bottom).mean())
    return horizontal < 0.06 and vertical < 0.06


def _suggest_repeats(
    overlay: Raster,
    *,
    is_pattern: bool,
    target_width_px: int | None,
    target_dpi: float | None,
) -> float:
    if not is_pattern:
        return 1.0
    if target_width_px and target_dpi and overlay.dpi:
        # Both files know their physical size: keep the artwork at 1:1 scale.
        overlay_mm = overlay.width / overlay.dpi[0] * MM_PER_INCH
        target_mm = target_width_px / target_dpi * MM_PER_INCH
        if overlay_mm > 0:
            return max(0.5, round(target_mm / overlay_mm, 2))
    return 4.0


def apply_cutout(overlay: Raster, cutout: np.ndarray) -> np.ndarray:
    """RGBA array of the overlay with the cutout mask as its alpha."""
    rgba = overlay.rgba.astype(np.float32).copy()
    rgba[:, :, 3] = np.clip(masks.clean(cutout) * 255.0, 0, 255)
    return rgba.astype(np.uint8)


def place(
    art: np.ndarray,
    canvas_size: tuple[int, int],
    box: Box,
    placement: Placement,
    info: PatternInfo | None = None,
    *,
    target_dpi: float | None = None,
) -> np.ndarray:
    """Render ``art`` (RGBA uint8) into a canvas of ``canvas_size``, inside ``box``.

    Returns an RGBA uint8 array the size of the canvas, transparent outside the
    placed artwork.
    """
    placement = placement.normalised()
    canvas_w, canvas_h = canvas_size
    left, top, right, bottom = box
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)

    source = Image.fromarray(art, mode="RGBA")
    if placement.flip_h:
        source = source.transpose(Image.FLIP_LEFT_RIGHT)
    if placement.flip_v:
        source = source.transpose(Image.FLIP_TOP_BOTTOM)

    fit = placement.fit
    if fit == "auto":
        fit = info.suggested_fit if info else "shape"

    if fit == "tile":
        patch = _render_tiled(source, (box_w, box_h), placement, info, target_dpi)
    else:
        patch = _render_single(source, (box_w, box_h), placement, fit)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    offset_px = (
        int(round(placement.offset_x * box_w)),
        int(round(placement.offset_y * box_h)),
    )
    if fit == "tile":
        # Tiling already accounts for the offset by shifting the phase.
        canvas.paste(patch, (left, top), patch)
    else:
        px = left + (box_w - patch.width) // 2 + offset_px[0]
        py = top + (box_h - patch.height) // 2 + offset_px[1]
        canvas.paste(patch, (px, py), patch)
    return np.array(canvas, dtype=np.uint8)


def _render_single(source: Image.Image, box: tuple[int, int], placement: Placement, fit: str) -> Image.Image:
    box_w, box_h = box
    if placement.rotation:
        source = source.rotate(
            -placement.rotation, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
        )

    if fit == "stretch":
        width = max(1, int(round(box_w * placement.scale)))
        height = max(1, int(round(box_h * placement.scale)))
        return source.resize((width, height), Image.LANCZOS)

    ratio_w = box_w / source.width
    ratio_h = box_h / source.height
    ratio = max(ratio_w, ratio_h) if fit == "cover" else min(ratio_w, ratio_h)
    ratio *= max(placement.scale, 0.01)

    width = max(1, int(round(source.width * ratio)))
    height = max(1, int(round(source.height * ratio)))
    resized = source.resize((width, height), Image.LANCZOS)

    if fit == "cover" and (width > box_w or height > box_h):
        cl = max(0, (width - box_w) // 2)
        ct = max(0, (height - box_h) // 2)
        resized = resized.crop((cl, ct, cl + min(width, box_w), ct + min(height, box_h)))
    return resized


def _render_tiled(
    source: Image.Image,
    box: tuple[int, int],
    placement: Placement,
    info: PatternInfo | None,
    target_dpi: float | None,
) -> Image.Image:
    box_w, box_h = box

    tile_w = _tile_width(source, box_w, placement, info, target_dpi)
    tile_w = max(2, int(round(tile_w * max(placement.scale, 0.01))))
    tile_h = max(2, int(round(tile_w * source.height / source.width)))
    tile = source.resize((tile_w, tile_h), Image.LANCZOS)

    mirror = placement.mirror
    if mirror == "auto":
        mirror = "off" if (info and info.seamless) else "on"
    if mirror == "on":
        tile = _mirror_tile(tile)
        tile_w, tile_h = tile.size

    # Tile onto an oversized canvas so rotation cannot expose bare corners.
    if placement.rotation:
        span = int(math.ceil(math.hypot(box_w, box_h))) + 2 * max(tile_w, tile_h)
    else:
        span = max(box_w, box_h)
    field_w = span if placement.rotation else box_w
    field_h = span if placement.rotation else box_h

    cols = int(math.ceil(field_w / tile_w)) + 2
    rows = int(math.ceil(field_h / tile_h)) + 2
    field = Image.new("RGBA", (cols * tile_w, rows * tile_h), (0, 0, 0, 0))
    for row in range(rows):
        for col in range(cols):
            field.paste(tile, (col * tile_w, row * tile_h))

    phase_x = int(round(placement.offset_x * box_w)) % tile_w
    phase_y = int(round(placement.offset_y * box_h)) % tile_h
    field = field.crop((tile_w - phase_x, tile_h - phase_y,
                        tile_w - phase_x + field_w, tile_h - phase_y + field_h))

    if placement.rotation:
        field = field.rotate(-placement.rotation, resample=Image.BICUBIC, expand=False)
        cl = (field.width - box_w) // 2
        ct = (field.height - box_h) // 2
        field = field.crop((cl, ct, cl + box_w, ct + box_h))
    return field


def _tile_width(
    source: Image.Image,
    box_w: int,
    placement: Placement,
    info: PatternInfo | None,
    target_dpi: float | None,
) -> float:
    if placement.repeat_mm:
        dpi = target_dpi or 300.0
        return float(mm_to_px(placement.repeat_mm, dpi))
    repeats = placement.repeat_across
    if repeats is None:
        repeats = info.suggested_repeats if info else 4.0
    repeats = max(0.05, float(repeats))
    return box_w / repeats


def _mirror_tile(tile: Image.Image) -> Image.Image:
    """Build a 2x2 mirrored block so edges always meet."""
    w, h = tile.size
    block = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    block.paste(tile, (0, 0))
    block.paste(tile.transpose(Image.FLIP_LEFT_RIGHT), (w, 0))
    block.paste(tile.transpose(Image.FLIP_TOP_BOTTOM), (0, h))
    block.paste(tile.transpose(Image.ROTATE_180), (w, h))
    return block
