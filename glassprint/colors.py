"""Colour parsing and colour-name based selection.

This is the part that makes "keep the gold bits, drop the white background"
work with no model downloaded: colour words map to regions of HSV space, and
membership is scored softly so edges stay anti-aliased.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RGB = tuple[int, int, int]

NAMED_COLORS: dict[str, RGB] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "red": (220, 38, 38),
    "crimson": (190, 24, 60),
    "maroon": (127, 29, 29),
    "orange": (234, 88, 12),
    "amber": (245, 158, 11),
    "yellow": (250, 204, 21),
    "gold": (201, 162, 39),
    "olive": (132, 132, 32),
    "lime": (132, 204, 22),
    "green": (34, 139, 60),
    "emerald": (16, 185, 129),
    "mint": (167, 243, 208),
    "teal": (13, 148, 136),
    "turquoise": (45, 212, 191),
    "cyan": (34, 211, 238),
    "sky": (56, 189, 248),
    "blue": (37, 99, 235),
    "navy": (30, 58, 138),
    "indigo": (67, 56, 202),
    "violet": (124, 58, 237),
    "purple": (147, 51, 234),
    "magenta": (219, 39, 119),
    "pink": (244, 114, 182),
    "rose": (244, 63, 94),
    "brown": (120, 72, 40),
    "tan": (180, 140, 100),
    "beige": (232, 217, 190),
    "cream": (245, 236, 217),
    "ivory": (250, 245, 232),
    "charcoal": (54, 57, 62),
}


@dataclass(frozen=True)
class ColorBand:
    """A region of HSV space that a colour word refers to."""

    hue: tuple[float, float] | None  # degrees, may wrap past 360
    sat: tuple[float, float] = (0.18, 1.0)
    val: tuple[float, float] = (0.12, 1.0)


# Hue ranges are deliberately generous: people say "blue" about a lot of blues.
COLOR_BANDS: dict[str, ColorBand] = {
    "red": ColorBand((-18, 14), sat=(0.25, 1.0), val=(0.12, 1.0)),
    "crimson": ColorBand((-12, 10), sat=(0.35, 1.0), val=(0.10, 0.85)),
    "maroon": ColorBand((-14, 14), sat=(0.30, 1.0), val=(0.05, 0.45)),
    "orange": ColorBand((14, 40), sat=(0.35, 1.0), val=(0.30, 1.0)),
    "amber": ColorBand((30, 48), sat=(0.35, 1.0), val=(0.40, 1.0)),
    "gold": ColorBand((36, 56), sat=(0.28, 1.0), val=(0.35, 1.0)),
    "yellow": ColorBand((44, 68), sat=(0.25, 1.0), val=(0.35, 1.0)),
    "olive": ColorBand((55, 85), sat=(0.20, 1.0), val=(0.15, 0.6)),
    "lime": ColorBand((68, 95), sat=(0.30, 1.0), val=(0.35, 1.0)),
    "green": ColorBand((85, 155), sat=(0.20, 1.0), val=(0.12, 1.0)),
    "emerald": ColorBand((140, 165), sat=(0.30, 1.0)),
    "mint": ColorBand((140, 170), sat=(0.12, 0.6), val=(0.6, 1.0)),
    "teal": ColorBand((160, 190), sat=(0.20, 1.0)),
    "turquoise": ColorBand((165, 190), sat=(0.30, 1.0), val=(0.4, 1.0)),
    "cyan": ColorBand((180, 200), sat=(0.25, 1.0), val=(0.35, 1.0)),
    "sky": ColorBand((190, 215), sat=(0.20, 1.0), val=(0.45, 1.0)),
    "blue": ColorBand((200, 250), sat=(0.20, 1.0), val=(0.10, 1.0)),
    "navy": ColorBand((205, 250), sat=(0.30, 1.0), val=(0.05, 0.4)),
    "indigo": ColorBand((240, 265), sat=(0.25, 1.0)),
    "violet": ColorBand((255, 285), sat=(0.20, 1.0)),
    "purple": ColorBand((265, 300), sat=(0.20, 1.0)),
    "magenta": ColorBand((295, 325), sat=(0.25, 1.0)),
    "pink": ColorBand((300, 350), sat=(0.10, 0.75), val=(0.55, 1.0)),
    "rose": ColorBand((330, 360), sat=(0.25, 1.0), val=(0.35, 1.0)),
    "brown": ColorBand((10, 45), sat=(0.20, 1.0), val=(0.10, 0.55)),
    "tan": ColorBand((20, 45), sat=(0.15, 0.55), val=(0.5, 0.9)),
    "beige": ColorBand((25, 55), sat=(0.05, 0.35), val=(0.7, 1.0)),
    "cream": ColorBand((30, 60), sat=(0.03, 0.25), val=(0.82, 1.0)),
    "ivory": ColorBand((30, 65), sat=(0.0, 0.18), val=(0.88, 1.0)),
    # Achromatic terms: the hue is irrelevant, saturation and value decide.
    "black": ColorBand(None, sat=(0.0, 1.0), val=(0.0, 0.22)),
    "charcoal": ColorBand(None, sat=(0.0, 0.4), val=(0.08, 0.35)),
    "grey": ColorBand(None, sat=(0.0, 0.16), val=(0.22, 0.82)),
    "gray": ColorBand(None, sat=(0.0, 0.16), val=(0.22, 0.82)),
    "silver": ColorBand(None, sat=(0.0, 0.14), val=(0.6, 0.92)),
    "white": ColorBand(None, sat=(0.0, 0.14), val=(0.84, 1.0)),
}

#: Words that describe brightness rather than hue.
TONE_BANDS: dict[str, tuple[float, float]] = {
    "dark": (0.0, 0.38),
    "darks": (0.0, 0.38),
    "shadow": (0.0, 0.35),
    "shadows": (0.0, 0.35),
    "deep": (0.0, 0.40),
    "light": (0.62, 1.0),
    "lights": (0.62, 1.0),
    "bright": (0.66, 1.0),
    "pale": (0.70, 1.0),
    "highlight": (0.75, 1.0),
    "highlights": (0.75, 1.0),
    "midtone": (0.35, 0.68),
    "midtones": (0.35, 0.68),
}

COLOR_WORDS = set(COLOR_BANDS) | set(NAMED_COLORS)


def parse_color(value: str | tuple[int, int, int] | None) -> RGB | None:
    """Accept ``#rgb``, ``#rrggbb``, ``r,g,b`` or a colour name."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) < 3:
            return None
        return tuple(int(max(0, min(255, c))) for c in value[:3])  # type: ignore[return-value]

    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("#"):
        text = text[1:]
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) == 8:
            text = text[:6]
        if len(text) != 6:
            raise ValueError(f"cannot parse colour {value!r}")
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) >= 3:
            return tuple(int(max(0, min(255, float(p)))) for p in parts[:3])  # type: ignore[return-value]
    if text in NAMED_COLORS:
        return NAMED_COLORS[text]
    raise ValueError(f"cannot parse colour {value!r}")


def to_hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(max(0, min(255, c))) for c in rgb))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV. Input and output are float arrays in 0..1 (H in 0..1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-8
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.where(nonzero, (maxc - r) / np.where(nonzero, delta, 1), 0)
        gc = np.where(nonzero, (maxc - g) / np.where(nonzero, delta, 1), 0)
        bc = np.where(nonzero, (maxc - b) / np.where(nonzero, delta, 1), 0)
    hue = np.where(maxc == r, bc - gc, hue)
    hue = np.where((maxc == g) & (maxc != r), 2.0 + rc - bc, hue)
    hue = np.where((maxc == b) & (maxc != r) & (maxc != g), 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0
    hue = np.where(nonzero, hue, 0.0)

    sat = np.where(maxc > 1e-8, delta / np.where(maxc > 1e-8, maxc, 1), 0.0)
    return np.stack([hue, sat, maxc], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0] % 1.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.astype(np.int32) % 6

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _soft_range(values: np.ndarray, low: float, high: float, softness: float) -> np.ndarray:
    """1 inside [low, high], ramping to 0 over ``softness`` outside it."""
    softness = max(softness, 1e-4)
    below = np.clip((values - (low - softness)) / softness, 0.0, 1.0)
    above = np.clip(((high + softness) - values) / softness, 0.0, 1.0)
    return np.minimum(below, above).astype(np.float32)


def color_mask(rgb: np.ndarray, name: str, tolerance: float = 1.0) -> np.ndarray:
    """Soft mask of pixels matching a colour word.

    ``rgb`` is a float array in 0..1 shaped (H, W, 3). ``tolerance`` widens the
    accepted band; 1.0 is the tuned default.
    """
    key = name.strip().lower()
    band = COLOR_BANDS.get(key)
    if band is None:
        target = parse_color(key)
        if target is None:
            raise ValueError(f"unknown colour {name!r}")
        return color_distance_mask(rgb, target, tolerance=tolerance)

    hsv = rgb_to_hsv(rgb)
    hue_deg = hsv[..., 0] * 360.0
    sat, val = hsv[..., 1], hsv[..., 2]

    mask = _soft_range(sat, band.sat[0], band.sat[1], 0.10 * tolerance)
    mask = mask * _soft_range(val, band.val[0], band.val[1], 0.10 * tolerance)

    if band.hue is not None:
        low, high = band.hue
        # Compare on the hue circle so red (which wraps) behaves.
        centre = ((low + high) / 2.0) % 360.0
        half_width = (high - low) / 2.0
        delta = np.abs(((hue_deg - centre + 180.0) % 360.0) - 180.0)
        mask = mask * _soft_range(delta, 0.0, half_width, 8.0 * tolerance)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def color_distance_mask(rgb: np.ndarray, target: RGB, tolerance: float = 1.0) -> np.ndarray:
    """Soft mask of pixels near an exact colour."""
    ref = np.array(target, dtype=np.float32) / 255.0
    # Rough perceptual weighting: eyes care most about green, least about blue.
    weights = np.array([0.30, 0.59, 0.11], dtype=np.float32)
    dist = np.sqrt((((rgb - ref[None, None, :]) ** 2) * weights[None, None, :]).sum(axis=-1))
    radius = 0.16 * max(tolerance, 0.05)
    return np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)


def tone_mask(rgb: np.ndarray, word: str, tolerance: float = 1.0) -> np.ndarray:
    band = TONE_BANDS.get(word.strip().lower())
    if band is None:
        raise ValueError(f"unknown tone {word!r}")
    luminance = luminance_of(rgb)
    return _soft_range(luminance, band[0], band[1], 0.10 * tolerance)


def luminance_of(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def dominant_colors(rgb: np.ndarray, alpha: np.ndarray | None = None, count: int = 5) -> list[RGB]:
    """Cheap palette read-out: coarse histogram over a 16-level RGB cube."""
    flat = rgb.reshape(-1, 3)
    if alpha is not None:
        keep = alpha.reshape(-1) > 0.35
        if keep.any():
            flat = flat[keep]
    if flat.size == 0:
        return []
    quant = np.clip((flat * 15.0 + 0.5).astype(np.int32), 0, 15)
    keys = quant[:, 0] * 256 + quant[:, 1] * 16 + quant[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1][:count]
    out: list[RGB] = []
    for key in values[order]:
        r = (key // 256) / 15.0
        g = ((key // 16) % 16) / 15.0
        b = (key % 16) / 15.0
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out
