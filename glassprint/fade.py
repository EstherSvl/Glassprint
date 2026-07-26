"""Fading the overlay into the glass.

The whole point of a fade on a UV print is that the ink stops and the glass
takes over. That is a fade of *alpha*, not a fade toward a colour: Studio builds
the white underbase from the alpha channel, so lowering alpha thins the white
and the glass genuinely shows through. Blending the artwork toward the glass
colour instead would keep the underbase at full density and print a solid white
patch with pale ink on it — a sticker fading, not ink dissolving.

Two ways to express the fade, and they mix:

* **Tonal** — every element gets more transparent. Smooth on screen, and on a
  EufyMake E1 it stops dead: see ``ALPHA_CLIFF`` below.
* **Dissolve** — whole elements drop out at an increasing rate while the
  survivors stay fully opaque. Tone comes from how *much* full-strength ink
  there is, which is the only kind of tone this printer renders properly.

``Fade.dissolve`` is the blend between them.

Measured on a EufyMake E1, printing on green glass (see README):

* Partial alpha is **not** partial ink. Below about 50% it prints nothing at
  all — a cliff, not a gradient. A tonal fade therefore does not fade; it runs
  at roughly half strength and then vanishes mid-ramp.
* Coverage — dot size, dropped elements, dropped passes — works smoothly down
  to about 12%, four times further than alpha manages.

So tone has to be built from coverage, and the tonal mode is only safe over the
top half of its range. ``check()`` says so when a plan crosses the line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

MODES = ("none", "linear", "radial", "shape")

#: Alpha below which a EufyMake E1 prints nothing whatsoever. Measured twice on
#: one tile and agreeing to within a couple of percent: a stepped alpha ramp
#: stopped after the 45% patch, and an independent continuous ramp stopped at
#: half its length. Not a soft floor where quality degrades — a cliff.
ALPHA_CLIFF = 0.5

#: Coverage below which a dot screen starts to break up. Same tile: the screened
#: row was still printing at 12% coverage, where the flat row had been blank for
#: two thirds of its length.
COVERAGE_FLOOR = 0.12

#: Finest usable dot pitch in millimetres. At 0.25mm the dots bridged into
#: blotches — overspray closing the gaps — and 0.4mm printed but weakly. 0.6mm
#: and coarser came out clean.
MIN_HALFTONE_MM = 0.6


def check(fade: "Fade") -> list[str]:
    """What this fade will actually do on the printer, where that differs.

    The measurements behind these are in the module docstring. They are warnings
    rather than corrections because the right fix depends on the picture: a
    shallow fade may be exactly what someone wants.
    """
    notes: list[str] = []
    if not fade.active:
        return notes

    if not fade.screened and not fade.stacked and fade.dissolve < 1.0:
        # A tonal ramp is the one thing this printer cannot render.
        low = min(fade.min_alpha, fade.max_alpha)
        if low < ALPHA_CLIFF:
            notes.append(
                f"tonal fade runs down to {low:.0%} alpha, and the printer drops "
                f"everything under about {ALPHA_CLIFF:.0%} — the fade will stop "
                "dead partway instead of reaching the glass. Use a dot screen "
                "(halftone_mm), dissolve, or ink layers to carry the tail."
            )

    if fade.screened and fade.halftone_mm < MIN_HALFTONE_MM:
        notes.append(
            f"dot pitch {fade.halftone_mm:.2f}mm is finer than the {MIN_HALFTONE_MM}mm "
            "that printed cleanly; the dots bridge into blotches below that."
        )

    if fade.stacked and fade.layers > 4:
        notes.append(
            f"{fade.layers} layers means {fade.layers} passes fed by hand, and ink "
            "stops deepening well before that — three is usually the most that earns "
            "its keep."
        )
    return notes


@dataclass
class Fade:
    mode: str = "none"

    #: Which elements fade, in the same language as the keep instruction.
    #: Empty means everything.
    what: str = ""

    #: Direction for ``linear``, in degrees. 90 fades downward (opaque at the
    #: top), 0 fades to the right, 270 fades upward.
    angle: float = 90.0

    #: Centre for ``radial``, as fractions of the target box.
    center_x: float = 0.5
    center_y: float = 0.5

    #: Where along the axis the fade begins and completes (0..1). Narrow the
    #: gap to make the transition happen faster.
    start: float = 0.0
    end: float = 1.0

    #: Shape of the ramp. 1 is linear, >1 holds opacity then drops away late,
    #: <1 drops away immediately then trails off.
    curve: float = 1.0

    #: The two ends of the ramp. Raise ``min_alpha`` to fade to a ghost rather
    #: than to nothing.
    min_alpha: float = 0.0
    max_alpha: float = 1.0

    #: Give each element one opacity (taken from its own average) instead of
    #: letting the ramp cut through it.
    per_element: bool = False

    #: 0 = every element thins; 1 = elements drop out whole and survivors stay
    #: fully opaque. In between, some elements fade faster than others.
    dissolve: float = 0.0

    #: Keeps the dissolve reproducible between preview and export.
    seed: int = 0

    #: Build the fade out of this many printed ink layers instead of a smooth
    #: ramp, dropping layers toward the transparent end. 0 leaves it smooth.
    #: Every layer prints at full coverage, so there is no dither floor at all
    #: — the cost is that N layers can only make N+1 steps.
    layers: int = 0

    #: Render the ramp as a dot screen instead, at this pitch in millimetres.
    #: 0 turns it off. Keep it coarse — see :func:`halftone`.
    halftone_mm: float = 0.0

    #: Screen angle. 45 is the traditional choice; it is the least obtrusive
    #: to the eye and avoids lining the dots up with the artwork's own edges.
    halftone_angle: float = 45.0

    #: Flip the direction of the ramp.
    invert: bool = False

    #: Alpha below this is snapped to zero on the finished layer. UV dithering
    #: gets speckly under roughly 0.12; set this to drop an unprintable tail.
    cutoff: float = 0.0

    def validated(self) -> "Fade":
        if self.mode not in MODES:
            raise ValueError(f"unknown fade mode {self.mode!r}; choose from {', '.join(MODES)}")
        if self.curve <= 0:
            raise ValueError("fade curve must be greater than zero")
        return self

    @property
    def active(self) -> bool:
        return self.mode != "none"

    @property
    def screened(self) -> bool:
        return self.halftone_mm > 0

    @property
    def stacked(self) -> bool:
        return self.layers > 0

    def describe(self) -> str:
        if not self.active:
            return "none"
        parts = [self.mode]
        if self.mode == "linear":
            parts.append(f"{self.angle:g}°")
        if self.what.strip():
            parts.append(f"on '{self.what.strip()}'")
        parts.append(f"{self.start:g}–{self.end:g}")
        if self.curve != 1.0:
            parts.append(f"curve {self.curve:g}")
        if self.stacked:
            parts.append(f"{self.layers} ink layers")
        elif self.screened:
            parts.append(f"{self.halftone_mm:g}mm dots at {self.halftone_angle:g}°")
        elif self.dissolve > 0:
            parts.append(f"{self.dissolve:.0%} dissolve")
        elif self.per_element:
            parts.append("per element")
        return " · ".join(parts)


def ramp(
    fade: Fade,
    canvas_shape: tuple[int, int],
    box: tuple[int, int, int, int],
    shape_mask: np.ndarray | None = None,
) -> np.ndarray:
    """The opacity field: 1 where the artwork stays solid, 0 where it is gone."""
    fade = fade.validated()
    height, width = canvas_shape
    left, top, right, bottom = box
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)

    if fade.mode == "linear":
        travel = _linear_travel(fade, width, height, left, top, box_w, box_h)
    elif fade.mode == "radial":
        travel = _radial_travel(fade, width, height, left, top, box_w, box_h)
    elif fade.mode == "shape":
        travel = _shape_travel(shape_mask, width, height)
    else:
        return np.ones((height, width), dtype=np.float32)

    if fade.invert:
        travel = 1.0 - travel

    # How far into the fade each pixel is: 0 before it starts, 1 once complete.
    span = fade.end - fade.start
    if abs(span) < 1e-6:
        progress = (travel >= fade.start).astype(np.float32)
    else:
        progress = np.clip((travel - fade.start) / span, 0.0, 1.0)

    eased = np.power(progress, float(fade.curve), dtype=np.float32)
    opacity = fade.max_alpha + (fade.min_alpha - fade.max_alpha) * eased
    return np.clip(opacity, 0.0, 1.0).astype(np.float32)


def _linear_travel(
    fade: Fade, width: int, height: int, left: int, top: int, box_w: int, box_h: int
) -> np.ndarray:
    radians = math.radians(fade.angle)
    dx, dy = math.cos(radians), math.sin(radians)

    # Span the box inclusively, so the far edge really is the end of the ramp
    # and `end=1` means "completely gone by the edge of the shape".
    u = (np.arange(width, dtype=np.float32) - left) / max(1, box_w - 1)
    v = (np.arange(height, dtype=np.float32) - top) / max(1, box_h - 1)
    projected = u[None, :] * dx + v[:, None] * dy

    # Normalise against the range the projection covers over the box, so the
    # fade always spans the shape regardless of angle.
    low = min(0.0, dx) + min(0.0, dy)
    high = max(0.0, dx) + max(0.0, dy)
    if high - low < 1e-6:
        return np.zeros((height, width), dtype=np.float32)
    return np.clip((projected - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _radial_travel(
    fade: Fade, width: int, height: int, left: int, top: int, box_w: int, box_h: int
) -> np.ndarray:
    centre_x = left + fade.center_x * box_w
    centre_y = top + fade.center_y * box_h

    dx = np.arange(width, dtype=np.float32) - centre_x
    dy = np.arange(height, dtype=np.float32) - centre_y
    distance = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)

    corners = [
        (left, top), (left + box_w, top),
        (left, top + box_h), (left + box_w, top + box_h),
    ]
    reach = max(math.hypot(cx - centre_x, cy - centre_y) for cx, cy in corners)
    if reach < 1e-6:
        return np.zeros((height, width), dtype=np.float32)
    return np.clip(distance / reach, 0.0, 1.0).astype(np.float32)


def _shape_travel(shape_mask: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Distance from the edge of the shape: 0 deep inside, 1 at the rim."""
    if shape_mask is None:
        return np.zeros((height, width), dtype=np.float32)
    inside = shape_mask > 0.5
    if not inside.any():
        return np.zeros((height, width), dtype=np.float32)
    distance = ndimage.distance_transform_edt(inside).astype(np.float32)
    deepest = float(distance.max())
    if deepest < 1e-6:
        return np.ones((height, width), dtype=np.float32)
    return np.clip(1.0 - distance / deepest, 0.0, 1.0).astype(np.float32)


def quantise(opacity: np.ndarray, layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Step the ramp into ``layers`` printed ink layers.

    Returns the stepped opacity and the layer count at each pixel (0..layers).
    Each layer goes down at full coverage, so nothing is ever printed at a
    density the machine cannot hold — the gradient comes from *how many* passes
    a region gets, which is the same idea as dissolve turned on its side.
    """
    count = max(1, int(layers))
    counts = np.round(np.clip(opacity, 0.0, 1.0) * count).astype(np.float32)
    return (counts / count).astype(np.float32), counts


#: Coverage at which neighbouring dots first touch (a circle inscribed in its
#: cell covers pi/4 of it). Past this they overlap and the area maths changes.
_DOTS_TOUCH = math.pi / 4.0


def halftone(
    opacity: np.ndarray,
    pitch_px: float,
    angle: float = 45.0,
    softness: float = 0.8,
) -> np.ndarray:
    """Render an opacity ramp as a screen of round dots, manga style.

    Tone comes from dot *size*, so every dot is full-strength ink and nothing
    is laid down at a density the printer cannot hold — the same reason
    screentone worked with nothing but black ink.

    Keep the pitch coarse. The printer halftones too, at device resolution, and
    a fine screen here beats against that one and produces moiré. At 1-3mm a
    dot is hundreds of device pixels across, so there is nothing to interfere
    with and the dots read as a deliberate texture.
    """
    pitch = max(2.0, float(pitch_px))
    height, width = opacity.shape

    radians = math.radians(angle)
    cos, sin = math.cos(radians), math.sin(radians)
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]

    # Rotate into screen space, then find each pixel's offset from its own
    # cell centre.
    u = (xs * cos + ys * sin) / pitch
    v = (-xs * sin + ys * cos) / pitch
    du = (u % 1.0) - 0.5
    dv = (v % 1.0) - 0.5
    distance = np.sqrt(du * du + dv * dv) * pitch

    coverage = np.clip(opacity, 0.0, 1.0)
    radius = _dot_radius(coverage, pitch)
    screen = np.clip((radius - distance) / max(softness, 1e-3) + 0.5, 0.0, 1.0)

    # The anti-aliasing ramp straddles the dot edge, which at the extremes
    # leaves a half-lit pixel at each cell centre (empty) or corner (solid).
    # Pin the ends so "no ink" and "solid" are exactly that.
    screen = np.where(coverage <= 0.0, 0.0, screen)
    screen = np.where(coverage >= 1.0, 1.0, screen)
    return screen.astype(np.float32)


def _dot_radius(coverage: np.ndarray, pitch: float) -> np.ndarray:
    """Dot radius that puts ``coverage`` of each cell under ink.

    Area-faithful while the dots stand apart; past the point where they touch
    it ramps to the cell's half-diagonal so full coverage really is solid.
    """
    apart = pitch * np.sqrt(coverage / math.pi)
    merged_span = (coverage - _DOTS_TOUCH) / (1.0 - _DOTS_TOUCH)
    merged = pitch * (0.5 + np.clip(merged_span, 0.0, 1.0) * (math.sqrt(0.5) - 0.5))
    return np.where(coverage <= _DOTS_TOUCH, apart, merged).astype(np.float32)


def apply(
    layer_alpha: np.ndarray,
    opacity: np.ndarray,
    fade: Fade,
    scope: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Fade ``layer_alpha`` by ``opacity``, returning the alpha and element count."""
    keep = np.clip(opacity, 0.0, 1.0).astype(np.float32)
    elements = 0

    if fade.per_element or fade.dissolve > 0:
        keep, elements = _by_element(keep, layer_alpha, fade)

    if scope is not None:
        # Only the selected elements fade; everything else keeps its opacity.
        keep = 1.0 - np.clip(scope, 0.0, 1.0) * (1.0 - keep)

    return np.clip(layer_alpha * keep, 0.0, 1.0).astype(np.float32), elements


def _by_element(keep: np.ndarray, layer_alpha: np.ndarray, fade: Fade) -> tuple[np.ndarray, int]:
    binary = layer_alpha > 0.35
    if not binary.any():
        return keep, 0
    labels, count = ndimage.label(binary)
    if count == 0:
        return keep, 0

    index = np.arange(1, count + 1)
    means = np.atleast_1d(np.asarray(ndimage.mean(keep, labels, index=index), dtype=np.float32))

    # Index 0 is the gap between elements; leave it untouched.
    per_element = np.concatenate([[1.0], means]).astype(np.float32)[labels]
    tonal = per_element if fade.per_element else keep

    if fade.dissolve > 0:
        # Each element gets one draw. An element survives when its draw falls
        # under its own opacity, so elements thin out at exactly the rate the
        # ramp asks for — and the survivors print at full strength.
        draws = np.random.default_rng(fade.seed).random(count).astype(np.float32)
        survives = np.concatenate([[1.0], (draws < means).astype(np.float32)])[labels]
        blend = float(np.clip(fade.dissolve, 0.0, 1.0))
        result = tonal * (1.0 - blend) + survives * blend
    else:
        result = tonal

    return _spread_to_edges(result, labels).astype(np.float32), count


def _spread_to_edges(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Carry each element's opacity out across its own anti-aliased edge.

    Element detection works on solid cores, so the soft pixels around a shape
    fall outside every label. Left alone they keep the underlying ramp value,
    and a dissolved element leaves a faint outline of itself behind — which
    prints as exactly the speckly halo dissolve is meant to avoid. Filling from
    the nearest element makes the edge die with the shape it belongs to.
    """
    outside = labels == 0
    if not outside.any() or outside.all():
        return values
    nearest = ndimage.distance_transform_edt(
        outside, return_distances=False, return_indices=True
    )
    return np.where(outside, values[tuple(nearest)], values)


def apply_cutoff(layer_alpha: np.ndarray, cutoff: float) -> np.ndarray:
    """Drop alpha that is too faint to print cleanly."""
    if cutoff <= 0:
        return layer_alpha
    return np.where(layer_alpha < cutoff, 0.0, layer_alpha).astype(np.float32)
