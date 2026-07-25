"""Previewing what the print will look like on coloured glass.

Without a white underbase the ink is a glaze, so what reaches your eye is light
that has passed through the glass *and* through however many layers of ink sit
on it. That is multiplicative — transmittances multiply, they do not composite —
which is why an ordinary alpha preview over a colour swatch is misleading, and
why stacking layers deepens colour so quickly.

The model treats each ink's RGB as its transmittance, which is crude but gets
the two things that matter right: the direction the hue shifts, and how fast
stacking pulls the colour away from the glass and toward the ink.
"""

from __future__ import annotations

import numpy as np

from .colors import RGB
from .compose import ComposeResult
from .raster import Raster


def glaze(
    result: ComposeResult,
    glass: RGB,
    *,
    layers: int = 1,
    layer_map: np.ndarray | None = None,
) -> Raster:
    """Render the composite as ink on coloured glass, with no white behind it.

    ``layers`` is how many passes a fully-inked region gets. ``layer_map`` gives
    a per-pixel count for a stacked fade; without one every inked pixel gets
    ``layers`` passes.
    """
    glass_rgb = np.array(glass, dtype=np.float32)[None, None, :] / 255.0

    # The base artwork is ink too, and gets one pass.
    transmitted = glass_rgb * _film(result.base.rgb_f, result.base.alpha_f, None)

    overlay = result.overlay_layer
    coverage = result.coverage if result.coverage is not None else overlay.alpha_f
    if layer_map is None:
        counts = np.full(coverage.shape, float(max(1, layers)), dtype=np.float32)
    else:
        counts = layer_map.astype(np.float32)

    transmitted = transmitted * _film(overlay.rgb_f, coverage, counts)

    rgba = np.empty_like(result.composite.rgba)
    rgba[:, :, :3] = np.clip(transmitted * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    return Raster(rgba, dpi=result.composite.dpi)


def _film(ink: np.ndarray, coverage: np.ndarray, counts: np.ndarray | None) -> np.ndarray:
    """Transmittance of one inked film: ``ink ** layers`` where it covers."""
    if counts is None:
        stacked = ink
    else:
        # ink ** 0 is 1 (bare glass), so uncovered areas fall out on their own.
        stacked = np.power(np.clip(ink, 1e-4, 1.0), counts[:, :, None])
    alpha = np.clip(coverage, 0.0, 1.0)[:, :, None]
    return (1.0 - alpha) + alpha * stacked


def stack_preview(ink: RGB, glass: RGB, up_to: int = 6) -> list[dict]:
    """What one ink looks like over one glass at 1..``up_to`` layers.

    Answers the question stacking is really for: how many passes does this
    colour need before the glass stops dictating its hue?
    """
    ink_t = np.array(ink, dtype=np.float32) / 255.0
    glass_t = np.array(glass, dtype=np.float32) / 255.0

    rows: list[dict] = []
    for n in range(1, max(1, up_to) + 1):
        transmitted = glass_t * np.power(np.clip(ink_t, 1e-4, 1.0), n)
        spread = float(transmitted.max() / max(float(transmitted.min()), 1e-6))
        rows.append(
            {
                "layers": n,
                "rgb": [int(round(c * 255)) for c in transmitted],
                # How strongly the ink's own colour dominates the glass tint.
                "ink_dominance": round(spread, 1),
                "light": round(float(transmitted.mean()), 3),
            }
        )
    return rows
