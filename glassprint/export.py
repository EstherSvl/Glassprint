"""Writing the result out at the right size, in the right formats.

Two things matter for the printer: the file must carry a DPI tag, and the pixel
grid must correspond to the physical size you want on the glass. Both layers
are resampled together so the overlay stays registered to the composite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import masks
from .colors import parse_color, to_hex
from .compose import ComposeResult
from .raster import (
    ALPHA_FORMATS,
    WRITE_FORMATS,
    Raster,
    mm_to_px,
    normalise_format,
    px_to_mm,
)

TARGETS = (
    "composite", "overlay", "shape-mask", "cutout-mask",
    "layer-map", "layers", "glaze-layers", "print-order",
)


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
        for fmt in formats.get(name, spec.formats):
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

    if "print-order" in spec.targets:
        sheet = write_print_order(result, out_dir, basename, written)
        if sheet:
            written.append(sheet)
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
        elif target == "layer-map":
            layers.update(_layer_map(result))
        elif target == "layers":
            layers.update(_ink_layers(result))
        elif target == "glaze-layers":
            layers.update(_glaze_layers(result))
    return layers


def _glaze_layers(result: ComposeResult) -> dict[str, Raster]:
    """One file per glaze pass: which regions get this ink, this many times.

    Pass *k* of an ink covers every region whose recipe asks for at least *k*
    of it. Where a fade is running, the counts are scaled along its ramp, so
    layers come off toward the transparent edge.
    """
    plan = result.glaze_plan
    if plan is None or result.coverage is None:
        return {}

    ramp = result.fade_field
    out: dict[str, Raster] = {}
    for order, (ink, index) in enumerate(plan.stack, start=1):
        counts = plan.counts_for(ink)
        if ramp is not None:
            counts = np.round(counts * ramp)
        alpha = result.coverage * (counts >= index).astype(np.float32)
        if alpha.max() <= 0.001:
            continue
        rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.array(ink.rgb, dtype=np.uint8)[None, None, :]
        rgba[:, :, 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # Zero-padded and numbered globally, so sorting the folder by name
        # gives you the order to feed them to the printer.
        out[glaze_pass_name(order, ink.name)] = Raster(rgba, dpi=result.composite.dpi)
    return out


def glaze_pass_name(order: int, ink_name: str) -> str:
    return f"pass{order:02d}-{ink_name.lstrip('#')}"


def _layer_map(result: ComposeResult) -> dict[str, Raster]:
    """A greyscale map of how many ink layers each region gets.

    White is the full stack, black is bare glass. This is the shape a relief or
    height pass wants.
    """
    if result.layer_map is None:
        return {}
    count = max(1, result.fade.layers)
    return {"layer-map": _mask_to_raster(result.layer_map / count, result.composite)}


def _ink_layers(result: ComposeResult) -> dict[str, Raster]:
    """One file per printed pass, each at full strength.

    Pass *k* covers everywhere that gets at least *k* layers, so printing them
    in order builds the gradient out of solid ink — no dithering anywhere.
    """
    if result.layer_map is None or result.coverage is None:
        return {}

    count = max(1, result.fade.layers)
    out: dict[str, Raster] = {}
    for index in range(1, count + 1):
        alpha = result.coverage * (result.layer_map >= index).astype(np.float32)
        rgba = result.overlay_layer.rgba.copy()
        rgba[:, :, 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out[f"layer{index}of{count}"] = Raster(rgba, dpi=result.composite.dpi)
    return out


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

    formats = {
        "composite": composite,
        "overlay": overlay,
        "shape-mask": requested,
        "cutout-mask": requested,
        "layer-map": requested,
    }
    # Each printed pass is a cut-out, so it needs a format that holds alpha.
    for index in range(1, max(1, result.fade.layers) + 1):
        formats[f"layer{index}of{result.fade.layers}"] = overlay
    if result.glaze_plan is not None:
        for order, (ink, _) in enumerate(result.glaze_plan.stack, start=1):
            formats[glaze_pass_name(order, ink.name)] = overlay
    return formats


def write_print_order(result: ComposeResult, out_dir: Path, basename: str, written: list[dict]) -> dict | None:
    """A sheet telling you what to print, in what order, and what to watch for.

    Printing a glaze means feeding the machine one pass at a time, so the thing
    that actually goes wrong is a human one: passes out of order, or the white
    underbase left switched on and burying the glaze underneath it.
    """
    # Manifest entries are labelled by layer name, so pick the pass files out
    # by their naming: "pass01-cyan" for glazes, "layer2of4" for one repeated ink.
    passes = [
        entry
        for entry in written
        if entry["target"].startswith("pass") or re.fullmatch(r"layer\d+of\d+", entry["target"])
    ]
    if not passes:
        return None

    passes.sort(key=lambda entry: entry["file"])
    lines = [
        f"# Print order — {basename}",
        "",
        f"{len(passes)} passes, printed one at a time in this order.",
        "",
        "**Turn the white underbase off on every pass.** White is opaque; laid under or",
        "over a glaze it blocks the stack and you lose both the colour and the",
        "transparency. Each pass goes straight onto the glass or onto the cured pass",
        "below it.",
        "",
        "Every file is the full canvas at the same size and DPI, so the passes register",
        "with each other as long as the piece does not move between them.",
        "",
    ]

    if result.glaze_plan is not None:
        lines += [f"Glass: `{to_hex(result.glaze_plan.glass)}`", ""]

    lines.append("| # | File | Ink |")
    lines.append("| --- | --- | --- |")
    for index, entry in enumerate(passes, start=1):
        ink = entry["file"].rsplit("-", 1)[-1].rsplit(".", 1)[0]
        lines.append(f"| {index} | `{entry['file']}` | {ink} |")

    if result.glaze_plan is not None:
        lines += ["", "## Recipes", ""]
        for recipe in result.glaze_plan.recipes:
            mark = "" if recipe.reachable else "  ⚠"
            lines.append(
                f"- `{to_hex(recipe.target)}` → `{to_hex(recipe.achieved)}` "
                f"— {recipe.describe()}{mark}"
            )
        unreachable = [r for r in result.glaze_plan.recipes if not r.reachable]
        if unreachable:
            lines += [
                "",
                "⚠ marks colours brighter than the glass. They print darker than asked; "
                "only a white base could give them the brightness.",
            ]

    path = out_dir / f"{basename}_print-order.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "file": path.name,
        "target": "print-order",
        "format": "md",
        "alpha": False,
        "pixels": [0, 0],
        "dpi": 0,
        "size_mm": [0, 0],
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
