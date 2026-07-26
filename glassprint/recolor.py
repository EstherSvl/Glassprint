"""Recolouring the overlay.

For glass printing the usual need is "same pattern, different ink" — so the
default recolour keeps the artwork's own light and shade and only replaces its
hue, rather than flooding it with flat colour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colors import RGB, hsv_to_rgb, luminance_of, parse_color, rgb_to_hsv

MODES = ("none", "tint", "duotone", "replace", "mono")


@dataclass
class ColorSpec:
    mode: str = "none"
    color: str | None = None        # tint / duotone highlight / replacement target
    color2: str | None = None       # duotone shadow
    from_color: str | None = None   # for mode="replace"
    strength: float = 1.0
    tolerance: float = 1.0
    hue_shift: float = 0.0          # degrees
    saturation: float = 1.0         # multiplier
    brightness: float = 1.0         # multiplier
    contrast: float = 1.0           # multiplier around mid grey
    invert: bool = False

    #: Snap anything this dark, or darker, to exactly RGB(0,0,0). 0 turns it off.
    #:
    #: Worth having because the printer treats pure black as a different colour
    #: from near-black — RGB(0,0,0) prints dense and warm, RGB(20,20,24) thin and
    #: blue-grey. Artwork rarely contains exact zeros: a photograph, a scan, a
    #: brightness tweak or a JPEG round-trip all leave the darkest pixels a
    #: little above it, and every one of those lands on the weak side of the
    #: discontinuity.
    black_point: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.mode in ("none", "")
            and not self.hue_shift
            and self.saturation == 1.0
            and self.brightness == 1.0
            and self.contrast == 1.0
            and not self.invert
            and not self.black_point
        )


def apply(rgba: np.ndarray, spec: ColorSpec) -> np.ndarray:
    """Apply a colour treatment to an RGBA uint8 array, leaving alpha alone."""
    if spec.is_identity:
        return rgba

    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = rgba[:, :, 3]

    mode = (spec.mode or "none").lower()
    if mode not in MODES:
        raise ValueError(f"unknown colour mode {spec.mode!r}; choose from {', '.join(MODES)}")

    if mode == "tint":
        target = parse_color(spec.color)
        if target is None:
            raise ValueError("tint mode needs a colour")
        rgb = _blend(rgb, colorize(rgb, target), spec.strength)
    elif mode == "mono":
        grey = luminance_of(rgb)[:, :, None].repeat(3, axis=2)
        rgb = _blend(rgb, grey, spec.strength)
    elif mode == "duotone":
        shadow = parse_color(spec.color2) or (0, 0, 0)
        highlight = parse_color(spec.color) or (255, 255, 255)
        rgb = _blend(rgb, duotone(rgb, shadow, highlight), spec.strength)
    elif mode == "replace":
        source = parse_color(spec.from_color)
        target = parse_color(spec.color)
        if source is None or target is None:
            raise ValueError("replace mode needs both from_color and color")
        rgb = replace_color(rgb, source, target, tolerance=spec.tolerance, strength=spec.strength)

    if spec.hue_shift or spec.saturation != 1.0:
        hsv = rgb_to_hsv(rgb)
        hsv[..., 0] = (hsv[..., 0] + spec.hue_shift / 360.0) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * spec.saturation, 0.0, 1.0)
        rgb = hsv_to_rgb(hsv)

    if spec.brightness != 1.0:
        rgb = np.clip(rgb * spec.brightness, 0.0, 1.0)
    if spec.contrast != 1.0:
        rgb = np.clip((rgb - 0.5) * spec.contrast + 0.5, 0.0, 1.0)
    if spec.invert:
        rgb = 1.0 - rgb
    if spec.black_point > 0.0:
        # Last, so it is not undone by a later brightness or contrast tweak.
        rgb = np.where(
            luminance_of(rgb)[:, :, None] <= spec.black_point, np.float32(0.0), rgb
        )

    out = np.empty_like(rgba)
    out[:, :, :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[:, :, 3] = alpha
    return out


def colorize(rgb: np.ndarray, target: RGB) -> np.ndarray:
    """Map luminance onto a black -> colour -> white ramp.

    This keeps the artwork's internal contrast, which a flat hue replacement
    would flatten out.
    """
    lum = luminance_of(rgb)[:, :, None]
    colour = np.array(target, dtype=np.float32)[None, None, :] / 255.0
    mid = float(luminance_of(colour.reshape(1, 1, 3))[0, 0])
    mid = min(max(mid, 0.05), 0.95)

    lower = lum / mid
    upper = (lum - mid) / (1.0 - mid)
    dark = colour * np.clip(lower, 0.0, 1.0)
    light = colour + (1.0 - colour) * np.clip(upper, 0.0, 1.0)
    return np.where(lum <= mid, dark, light).astype(np.float32)


def duotone(rgb: np.ndarray, shadow: RGB, highlight: RGB) -> np.ndarray:
    lum = luminance_of(rgb)[:, :, None]
    low = np.array(shadow, dtype=np.float32)[None, None, :] / 255.0
    high = np.array(highlight, dtype=np.float32)[None, None, :] / 255.0
    return (low + (high - low) * lum).astype(np.float32)


def replace_color(
    rgb: np.ndarray,
    source: RGB,
    target: RGB,
    *,
    tolerance: float = 1.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Swap one colour for another, keeping shading intact."""
    ref = np.array(source, dtype=np.float32) / 255.0
    weights = np.array([0.30, 0.59, 0.11], dtype=np.float32)
    dist = np.sqrt((((rgb - ref[None, None, :]) ** 2) * weights[None, None, :]).sum(axis=-1))
    radius = 0.20 * max(tolerance, 0.05)
    match = np.clip(1.0 - dist / radius, 0.0, 1.0)[:, :, None] * float(strength)
    return (rgb * (1.0 - match) + colorize(rgb, target) * match).astype(np.float32)


def _blend(base: np.ndarray, other: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength >= 1.0:
        return other
    return (base * (1.0 - strength) + other * strength).astype(np.float32)
