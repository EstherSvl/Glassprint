"""Writing the result out at the right size, in the right formats.

Two things matter for the printer: the file must carry a DPI tag, and the pixel
grid must correspond to the physical size you want on the glass. Both layers
are resampled together so the overlay stays registered to the composite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import masks
from .colors import parse_color
from .compose import ComposeResult
from .raster import (
    ALPHA_FORMATS,
    WRITE_FORMATS,
    Raster,
    mm_to_px,
    normalise_format,
    px_to_mm,
)

TARGETS = ("composite", "overlay", "shape-mask", "cutout-mask")


@dataclass
class ExportSpec:
    formats: list[str] = field(default_factory=lambda: ["png"])
    targets: list[str] = field(default_factory=lambda: ["composite", "overlay"])
    include_base_format: bool = True
    dpi: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    quality: int = 95
    background: str = "#ffffff"
    basename: str | None = None

    def validated(self) -> "ExportSpec":
        for fmt in self.formats:
            normalise_format(fmt)
        for target in self.targets:
            if target not in TARGETS:
                raise ValueError(
                    f"unknown export target {target!r}; choose from {', '.join(TARGETS)}"
                )
        if self.width_mm and self.height_mm:
            raise ValueError("set width_mm or height_mm, not both — aspect ratio is preserved")
        return self


def export(result: ComposeResult, out_dir: str | Path, spec: ExportSpec | None = None) -> list[dict]:
    """Write the requested files and return a manifest describing each one."""
    spec = (spec or ExportSpec()).validated()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basename = spec.basename or result.base.name or "glassprint"
    dpi = spec.dpi or result.base.effective_dpi[0]
    background = parse_color(spec.background) or (255, 255, 255)

    layers = _collect_layers(result, spec.targets)
    layers = {name: _resize_for_output(layer, spec, dpi) for name, layer in layers.items()}

    formats = _formats_for(spec, result)
    written: list[dict] = []

    for name, layer in layers.items():
        wants_alpha = name != "composite" or result.base.has_alpha
        for fmt in formats[name]:
            suffix = "jpg" if fmt in ("jpg", "jpeg") else fmt
            path = out_dir / f"{basename}_{name}.{suffix}"
            layer.save(
                path,
                fmt=fmt,
                dpi=(dpi, dpi),
                quality=spec.quality,
                background=background,
                keep_alpha=wants_alpha,
            )
            written.append(
                {
                    "path": str(path),
                    "file": path.name,
                    "target": name,
                    "format": fmt,
                    "alpha": bool(wants_alpha and fmt in ALPHA_FORMATS),
                    "pixels": [layer.width, layer.height],
                    "dpi": round(dpi, 2),
                    "size_mm": [
                        round(px_to_mm(layer.width, dpi), 2),
                        round(px_to_mm(layer.height, dpi), 2),
                    ],
                }
            )
    return written


def _collect_layers(result: ComposeResult, targets: list[str]) -> dict[str, Raster]:
    layers: dict[str, Raster] = {}
    for target in targets:
        if target == "composite":
            layers["composite"] = result.composite
        elif target == "overlay":
            layers["overlay"] = result.overlay_layer
        elif target == "shape-mask":
            layers["shape-mask"] = _mask_to_raster(result.shape_mask, result.composite)
        elif target == "cutout-mask":
            layers["cutout-mask"] = _mask_to_raster(result.cutout_mask, result.composite)
    return layers


def _mask_to_raster(mask: np.ndarray, like: Raster) -> Raster:
    grey = np.clip(masks.clean(mask) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([grey, grey, grey, np.full_like(grey, 255)])
    return Raster(rgba, dpi=like.dpi)


def _formats_for(spec: ExportSpec, result: ComposeResult) -> dict[str, list[str]]:
    requested = [normalise_format(f) for f in spec.formats] or ["png"]
    composite = list(requested)

    base_fmt = (result.base.source_format or "").lower()
    if spec.include_base_format and base_fmt in WRITE_FORMATS:
        if base_fmt not in composite:
            composite.insert(0, base_fmt)

    # The overlay-only file has to hold transparency; if every requested format
    # is opaque, add PNG so the layer is actually usable.
    overlay = [f for f in requested if f in ALPHA_FORMATS]
    if not overlay:
        overlay = ["png"]

    return {
        "composite": composite,
        "overlay": overlay,
        "shape-mask": requested,
        "cutout-mask": requested,
    }


def _resize_for_output(layer: Raster, spec: ExportSpec, dpi: float) -> Raster:
    if spec.width_mm:
        return layer.with_physical_width(spec.width_mm, dpi)
    if spec.height_mm:
        return layer.with_physical_height(spec.height_mm, dpi)
    from dataclasses import replace

    return replace(layer, dpi=(dpi, dpi))


def plan_output_size(
    layer: Raster, *, dpi: float, width_mm: float | None = None, height_mm: float | None = None
) -> tuple[int, int]:
    """What pixel dimensions an export would produce, without doing the work."""
    if width_mm:
        w = mm_to_px(width_mm, dpi)
        return w, max(1, int(round(w * layer.height / layer.width)))
    if height_mm:
        h = mm_to_px(height_mm, dpi)
        return max(1, int(round(h * layer.width / layer.height))), h
    return layer.size
