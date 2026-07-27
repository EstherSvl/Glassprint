"""Replacing the tool's guess about ink with a measurement of your printer.

Everywhere else in this codebase an ink's RGB doubles as its transmittance:
ask for RGB(0,158,224) and the preview assumes 0%/62%/88% of the red, green and
blue light gets through. That is a guess, and a poor one. What actually reaches
the glass is whatever the RIP decides to lay down when handed that number, at
whatever density its heads run, and the relationship between the two is neither
linear nor the same in each channel.

This module measures it instead. Print :func:`chart`, photograph it, and
:func:`read` returns a :class:`Profile` — after which the preview is showing you
your printer rather than an idealisation of one.

    The split that matters
    ----------------------

Transmittances multiply, so a printed plate is::

    seen = white · T_glass · T_ink

``T_ink`` belongs to the printer. ``T_glass`` belongs to the glass. They are
independent, which is the whole reason one chart can serve many glasses: measure
the ink once, and any new glass costs three numbers you can read off a photo of
the bare offcut without printing anything at all.

That independence is exact for a full spectrum and approximate for three
channels, which is what a camera gives us. Two glasses that photograph the same
RGB but transmit different *spectra* will not respond to ink identically. Green
glass and cyan ink are the awkward case — both are narrow — so the profile
carries the glass it was measured on, and :func:`Profile.check` scores how well
it transferred when you try it on another.

    The model
    ---------

Absorbance ``a = -log₁₀ T`` adds where transmittance multiplies, so the fit is
done there. Per channel, the ink demanded is ``x = 1 - requested/255``; the
press's tone curve bends it, and each channel's ink absorbs a little in its
neighbours' bands::

    u = x ** gamma          (per channel, the tone curve)
    a = A · u               (3x3, the cross-talk)
    T = 10 ** -a

Twelve numbers, fitted from forty patches. It is deliberately small: a lookup
table with forty entries interpolates its own noise, extrapolates into nonsense
past the edges of the sample, and tells you nothing you can read. Twelve
parameters overfit far less, invert in closed form — which is what turns "what
will this look like" into "what should I ask for" — and can be read straight
off: ``gamma`` is how the press ramps, ``A`` off the diagonal is how muddy the
inks are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .colors import RGB, to_hex

#: Where transmittance is floored, so absorbance stays finite.
_FLOOR = 0.004

#: Photographs are sRGB-encoded. Light multiplies, encoded values do not, so
#: everything here happens after this.
def linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB in 0..1 to linear light."""
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def encode(rgb: np.ndarray) -> np.ndarray:
    """Linear light back to sRGB in 0..1."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


# -- the chart ---------------------------------------------------------------


def colours() -> list[RGB]:
    """The forty colours, in the order they are printed and read.

    Four groups, each earning its place:

    * **solid black**, the darkest the ink goes and the one asymmetric cell, so
      the reader can tell which way up the photograph is;
    * a **cube** at 0/128/255 per channel, which is what the fit needs — the
      model has cross-talk terms and they are only visible where channels mix;
    * a **grey ramp**, because the eye is least forgiving about neutrals and the
      cube passes through them only three times;
    * **eight held out**, never fitted. Their residual is the only honest score
      for the model, since a fit always looks good on its own training points.
      The last of them repeats an earlier patch, which turns the difference
      between the two into a direct reading of the measurement's own noise —
      the number every other error here has to be judged against.

    The held-out eight are **spread across the grid**, not left in a block at the
    end. Consecutive, they all landed in the bottom-right corner of the chart,
    which quietly turned the one number meant to measure *prediction* into a
    measure of how uneven the lamp was on one side of the plate. A real reading
    made that obvious: the fit was good and the held-out error was 40 levels.
    """
    cube = [
        (r, g, b)
        for r in (0, 128, 255)
        for g in (0, 128, 255)
        for b in (0, 128, 255)
        if (r, g, b) not in {(255, 255, 255), (0, 0, 0)}
    ]
    greys = [(v, v, v) for v in (32, 64, 96, 160, 192, 224)]
    held_out = [
        (200, 60, 40),
        (40, 120, 200),
        (120, 200, 60),
        (220, 180, 40),
        (90, 40, 140),
        (40, 160, 150),
        (180, 90, 140),
    ]
    fitted = [(0, 0, 0), *cube, *greys]
    keep_back = [*held_out, held_out[0]]

    out: list[RGB] = []
    fitted_iter, held_iter = iter(fitted), iter(keep_back)
    for index in range(len(fitted) + len(keep_back)):
        out.append(next(held_iter) if index in _HELD_OUT_AT else next(fitted_iter))
    return out


#: Where in the reading order the held-out patches sit — two per row, and no two
#: sharing a column.
#:
#: An even stride was the obvious way to spread them and does not work: the grid
#: is 11 wide but its first and last rows are 9 long, because the corners are
#: given over to bare substrate. Every fifth cell therefore beats against that
#: and lands on just four columns out of eleven, clustered in two pairs. These
#: are picked against the actual cell positions instead.
_HELD_OUT_AT = frozenset((1, 6, 13, 18, 20, 26, 33, 38))

#: Indices into :func:`colours` the fit never sees.
HELD_OUT = tuple(sorted(_HELD_OUT_AT))

#: The repeated patch, as ``(original, repeat)`` indices into :func:`colours`.
REPEAT = (HELD_OUT[0], HELD_OUT[7])

#: Index into :func:`colours` of the solid black cell.
BLACK = 0


def _legacy_colours() -> list[RGB]:
    """The order charts were printed in before the held-out patches were spread.

    Kept because plates already exist. Reading one with the current order does
    not produce a wrong answer — the repeated patch disagrees violently and the
    read is refused — but "reprint it" is a poor thing to tell someone holding a
    perfectly good plate, when the same checksum can just as easily say *which*
    chart they are holding.
    """
    cube = [
        (r, g, b)
        for r in (0, 128, 255)
        for g in (0, 128, 255)
        for b in (0, 128, 255)
        if (r, g, b) not in {(255, 255, 255), (0, 0, 0)}
    ]
    greys = [(v, v, v) for v in (32, 64, 96, 160, 192, 224)]
    held_out = [
        (200, 60, 40),
        (40, 120, 200),
        (120, 200, 60),
        (220, 180, 40),
        (90, 40, 140),
        (40, 160, 150),
        (180, 90, 140),
    ]
    return [(0, 0, 0), *cube, *greys, *held_out, held_out[0]]


#: Every chart layout the reader can recognise, newest first, as
#: ``(name, colours, held-out indices, repeated pair)``.
LAYOUTS = (
    ("current", colours, HELD_OUT, REPEAT),
    ("pre-spread", _legacy_colours, tuple(range(32, 40)), (32, 39)),
)


def glass_cells(layout: "Layout") -> tuple[int, int, int, int]:
    """The four grid corners, which carry the substrate rather than a colour.

    One reference in a corner was not enough. A phone's lamp falls off across
    the frame by a fifth or more, so dividing every patch by a single corner
    made the far side of the chart read up to 20 levels too bright — an error
    that landed squarely on the pale patches, where the eye is most alert. Four
    corners let the illumination be interpolated across the chart and divided
    out where each patch actually sits.
    """
    last = layout.columns * layout.rows - 1
    return (0, layout.columns - 1, last - layout.columns + 1, last)


def patches(layout: "Layout | None" = None, order=None) -> list[RGB | None]:
    """Every cell in reading order. ``None`` means bare glass — print nothing."""
    layout = layout or CHART
    corners = set(glass_cells(layout))
    queue = list((order or colours)())
    cells: list[RGB | None] = []
    for index in range(layout.columns * layout.rows):
        cells.append(None if index in corners else queue.pop(0))
    if queue:  # pragma: no cover - caught by a test, kept as a loud failure
        raise ValueError(f"{len(queue)} colours have nowhere to go on a {layout.columns}x{layout.rows} chart")
    return cells


@dataclass(frozen=True)
class Layout:
    """The chart's geometry, in millimetres.

    One definition, imported by both the generator and the reader. When these
    were separate the two drifted, and a reader looking in the wrong place
    reports colours rather than an error.
    """

    patch_mm: float = 5.5
    gap_mm: float = 1.3
    frame_mm: float = 1.6
    margin_mm: float = 2.5
    columns: int = 11
    rows: int = 4

    @property
    def pitch_mm(self) -> float:
        return self.patch_mm + self.gap_mm

    @property
    def grid_w_mm(self) -> float:
        return self.columns * self.pitch_mm + self.gap_mm

    @property
    def grid_h_mm(self) -> float:
        return self.rows * self.pitch_mm + self.gap_mm

    @property
    def frame_w_mm(self) -> float:
        """Outside of the black frame — the edge the reader looks for."""
        return self.grid_w_mm + 2 * self.frame_mm

    @property
    def frame_h_mm(self) -> float:
        return self.grid_h_mm + 2 * self.frame_mm

    @property
    def width_mm(self) -> float:
        return self.frame_w_mm + 2 * self.margin_mm

    @property
    def height_mm(self) -> float:
        return self.frame_h_mm + 2 * self.margin_mm + 4.0  # room for one caption

    def cell(self, index: int) -> tuple[float, float, float, float]:
        """``(x, y, w, h)`` of one patch, relative to the frame's outer corner."""
        row, column = divmod(index, self.columns)
        x = self.frame_mm + self.gap_mm + column * self.pitch_mm
        y = self.frame_mm + self.gap_mm + row * self.pitch_mm
        return x, y, self.patch_mm, self.patch_mm

    def centres(self) -> np.ndarray:
        """Patch centres in frame coordinates, as ``(n, 2)`` of ``(x, y)`` mm."""
        out = []
        for index in range(self.columns * self.rows):
            x, y, w, h = self.cell(index)
            out.append((x + w / 2.0, y + h / 2.0))
        return np.array(out, dtype=np.float64)


CHART = Layout()


# -- the profile -------------------------------------------------------------


@dataclass
class Profile:
    """What one printer does to one piece of glass, in twelve numbers."""

    gamma: np.ndarray                      # (3,) per-channel tone curve
    crosstalk: np.ndarray                  # (3, 3) absorbance mixing
    glass: RGB                             # the substrate it was measured on
    #: What the ink was measured on. Three cases, and the distinction that
    #: actually matters is not "glass or white" but **how light reaches your
    #: eye**, because that decides how many times it crosses the ink:
    #:
    #: * ``"transparent"`` — clear tinted glass, no white base. Light passes
    #:   through once. Measured backlit.
    #: * ``"opaque"`` — opaque or dark glass, no white base. Light goes in
    #:   through the ink, reflects off the glass and comes back out through it:
    #:   two crossings. Measured front-lit. The fit absorbs the doubling, so the
    #:   arithmetic is the same as ``"transparent"`` — only the lighting differs,
    #:   and getting *that* wrong is silent.
    #: * ``"white"`` — a white underbase, on any glass. Also two crossings, also
    #:   front-lit, but the base has hidden the glass, so the glass colour stops
    #:   being an input at all.
    substrate: str = "transparent"

    #: Substrates whose measurement is a reflection rather than a transmission,
    #: and so are photographed front-lit.
    REFLECTIVE = ("opaque", "white")

    #: Channels the substrate passes too little of to measure the ink in. Not a
    #: fault: you cannot find out what ink does to blue light through glass that
    #: passes no blue light. Their predictions come out near zero because the
    #: glass is near zero, which is the right answer for the wrong-sounding
    #: reason, and the profile says so rather than implying it measured them.
    unmeasured: tuple[str, ...] = ()

    #: Which patches were kept out of the fit. Carried on the profile because
    #: older charts put them elsewhere, and the residuals have to know.
    held_out: tuple[int, ...] = ()
    #: Everything measured, kept for reporting rather than for prediction.
    requested: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    measured: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    note: str = ""

    # -- using it ----------------------------------------------------------

    def transmittance(self, requested: np.ndarray) -> np.ndarray:
        """Transmittance of the ink film for a requested colour, glass aside.

        Accepts one colour or an image; the last axis must be RGB in 0..255.
        """
        demand = np.clip(1.0 - np.asarray(requested, dtype=np.float64) / 255.0, 0.0, 1.0)
        curved = demand ** self.gamma
        absorbance = curved @ self.crosstalk.T
        return np.clip(10.0 ** (-absorbance), _FLOOR, 1.0)

    def _base(self, glass: RGB | None) -> np.ndarray:
        """What the ink is sitting on.

        A white underbase hides the glass, so a glass colour handed to a
        white-base profile is ignored rather than obeyed — obeying it would tint
        a prediction the base has covered. Opaque glass is the opposite: the
        glass *is* the ground, so its colour matters as much as it does for a
        transparency, even though the light never gets through it.
        """
        if self.substrate == "white":
            return np.array(self.glass, dtype=np.float64) / 255.0
        return np.array(glass or self.glass, dtype=np.float64) / 255.0

    def predict(self, requested: RGB, glass: RGB | None = None) -> RGB:
        """What asking for ``requested`` actually looks like on the substrate."""
        glass_t = self._base(glass)
        seen = glass_t * self.transmittance(np.array(requested, dtype=np.float64))
        return tuple(int(round(float(c) * 255)) for c in np.clip(seen, 0.0, 1.0))  # type: ignore[return-value]

    def request_for(self, desired: RGB, glass: RGB | None = None) -> tuple[RGB, bool]:
        """What to ask for so it comes out ``desired``. Inverts the model exactly.

        Returns the colour and whether it was reachable. Ink only subtracts, so
        anything brighter than the substrate is not, and the answer is then the
        closest it allows rather than a number that would mislead.
        """
        glass_t = self._base(glass)
        want = np.array(desired, dtype=np.float64) / 255.0
        needed = np.clip(want / np.maximum(glass_t, 1e-6), _FLOOR, 1.0)
        reachable = bool(np.all(want <= glass_t + 0.02))

        absorbance = -np.log10(needed)
        curved = np.linalg.solve(self.crosstalk, absorbance)
        demand = np.clip(curved, 0.0, 1.0) ** (1.0 / self.gamma)
        rgb = np.clip(255.0 * (1.0 - demand), 0, 255)
        return tuple(int(round(float(c))) for c in rgb), reachable  # type: ignore[return-value]

    # -- how good is it ----------------------------------------------------

    @property
    def live(self) -> list[int]:
        """Channel indices the substrate actually passes enough light to measure."""
        dead = {name: index for index, name in enumerate(("red", "green", "blue"))}
        return [i for i in range(3) if i not in {dead[n] for n in self.unmeasured}]

    def residuals(self) -> dict:
        """Fit error in levels out of 255, and what it is an improvement on.

        ``held_out`` is the number that counts: those patches were kept out of
        the fit, so it is a prediction rather than a memory. ``naive`` is the
        error the rest of the codebase makes by reading an ink's RGB as its
        transmittance — the assumption this module exists to replace. If
        calibrating did not beat it, calibrating was not worth the glass.
        """
        if not len(self.requested):
            return {}
        live = self.live
        levels = lambda t: np.abs(t[:, live] - self.measured[:, live]).mean(axis=1) * 255.0  # noqa: E731
        error = levels(self.transmittance(self.requested))
        naive = levels(self.requested / 255.0)
        kept = self.held_out or HELD_OUT
        fitted = [i for i in range(len(error)) if i not in kept]
        held = [i for i in kept if i < len(error)]
        return {
            "fitted": round(float(error[fitted].mean()), 1),
            "held_out": round(float(error[held].mean()), 1) if held else None,
            "worst": round(float(error.max()), 1),
            "naive": round(float(naive.mean()), 1),
        }

    def check(self, other: "Profile") -> dict:
        """How well this profile predicts a chart measured on different glass.

        The question one chart per printer stands or falls on. Scored on the
        *ink*, with each profile's own glass divided out, so a difference here
        is the ink behaving differently rather than the glass being a different
        colour — which we already knew.
        """
        predicted = self.transmittance(other.requested)
        error = np.abs(predicted - other.measured).mean(axis=1) * 255.0
        return {
            "glass": to_hex(other.glass),
            "mean": round(float(error.mean()), 1),
            "worst": round(float(error.max()), 1),
            "transfers": bool(error.mean() < 8.0),
        }

    # -- storage -----------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "gamma": [round(float(g), 4) for g in self.gamma],
            "crosstalk": [[round(float(v), 4) for v in row] for row in self.crosstalk],
            "glass": to_hex(self.glass),
            "substrate": self.substrate,
            "unmeasured": list(self.unmeasured),
            "held_out": list(self.held_out),
            "requested": self.requested.astype(int).tolist(),
            "measured": [[round(float(v), 5) for v in row] for row in self.measured],
            "residuals": self.residuals(),
            "note": self.note,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        from .colors import parse_color

        glass = parse_color(data.get("glass", "#ffffff")) or (255, 255, 255)
        return cls(
            gamma=np.array(data["gamma"], dtype=np.float64),
            crosstalk=np.array(data["crosstalk"], dtype=np.float64),
            glass=glass,
            # "glass" was the old name for what is now "transparent".
            unmeasured=tuple(data.get("unmeasured") or ()),
            held_out=tuple(data.get("held_out") or ()),
            substrate={"glass": "transparent"}.get(
                str(data.get("substrate") or ""), str(data.get("substrate") or "transparent")
            ),
            requested=np.array(data.get("requested", []), dtype=np.float64).reshape(-1, 3),
            measured=np.array(data.get("measured", []), dtype=np.float64).reshape(-1, 3),
            note=data.get("note", ""),
        )


# -- fitting -----------------------------------------------------------------
#
# There is deliberately no "nominal profile" to fall back on. The assumption the
# rest of the codebase makes — that an ink's RGB is its transmittance — is a
# straight line in transmittance, so in absorbance it runs to infinity at full
# ink, and this model cannot represent it at any gamma. Trying produced a
# plausible-looking Profile that was wrong by 34 levels, which is worse than
# useless: it looks like a measurement. Where the naive assumption is wanted it
# stays where it is, in `simulate` and `glaze`, and `Profile.residuals` reports
# how far off it was rather than dressing it up as a calibration.


def fit(
    requested: np.ndarray,
    measured: np.ndarray,
    glass: RGB,
    *,
    note: str = "",
    substrate: str = "transparent",
    unmeasured: tuple[str, ...] = (),
    held_out: tuple[int, ...] = (),
) -> Profile:
    """Fit the twelve numbers to measured transmittances.

    ``measured`` is the ink's transmittance — the patch divided by bare glass,
    in linear light — so the glass has already dropped out.
    """
    requested = np.asarray(requested, dtype=np.float64).reshape(-1, 3)
    measured = np.clip(np.asarray(measured, dtype=np.float64).reshape(-1, 3), _FLOOR, 1.0)

    dead = [("red", "green", "blue").index(name) for name in unmeasured]
    live = [i for i in range(3) if i not in dead]

    keep = [i for i in range(len(requested)) if i not in (held_out or HELD_OUT)]
    demand = np.clip(1.0 - requested[keep] / 255.0, 0.0, 1.0)
    absorbance = -np.log10(measured[keep])[:, live]

    def solve(gamma: np.ndarray) -> tuple[np.ndarray, float]:
        curved = demand ** gamma
        matrix, *_ = np.linalg.lstsq(curved, absorbance, rcond=None)
        error = float(((curved @ matrix - absorbance) ** 2).mean())
        # A dead channel gets a row of zeros, so its transmittance comes out 1
        # and the prediction is the substrate alone — which, for a substrate
        # that passes none of it, is the truth.
        full = np.zeros((3, 3))
        full[live] = matrix.T
        return full, error

    # Coordinate descent on three gammas. The surface is smooth and shallow, so
    # a scan beats a solver here and cannot wander somewhere unphysical.
    gamma = np.ones(3)
    best = solve(gamma)[1]
    for _ in range(4):
        for channel in range(3):
            for candidate in np.linspace(0.3, 4.0, 75):
                trial = gamma.copy()
                trial[channel] = candidate
                _, error = solve(trial)
                if error < best - 1e-12:
                    best, gamma = error, trial
    crosstalk, _ = solve(gamma)

    return Profile(
        gamma=gamma,
        crosstalk=crosstalk,
        glass=glass,
        substrate=substrate,
        unmeasured=unmeasured,
        held_out=tuple(held_out or HELD_OUT),
        requested=requested,
        measured=measured,
        note=note,
    )


# -- reading a photograph ----------------------------------------------------


class ReadError(RuntimeError):
    """The photograph could not be read. The message says what to change."""


def _quad(mask: np.ndarray) -> np.ndarray:
    """The four corners of the largest dark region, as ``(x, y)`` in photo pixels.

    The frame and the gaps between patches are one connected mesh of ink, so the
    chart is a single blob whatever the patches themselves come out as. Corners
    come from the extremes along the two diagonals, which is exact for a
    rectangle under perspective and close enough for one photographed by hand.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        raise ReadError("no printed chart found — is the photograph of the right plate?")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labels == biggest)
    if len(xs) < 5000:
        raise ReadError("the chart is too small in frame — fill more of the photograph with it")

    total, difference = xs + ys, xs - ys
    corners = np.array(
        [
            [xs[np.argmin(total)], ys[np.argmin(total)]],          # top left
            [xs[np.argmax(difference)], ys[np.argmax(difference)]],  # top right
            [xs[np.argmax(total)], ys[np.argmax(total)]],          # bottom right
            [xs[np.argmin(difference)], ys[np.argmin(difference)]],  # bottom left
        ],
        dtype=np.float64,
    )
    return corners


def _homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The 3x3 taking ``source`` points to ``target`` points, from four pairs."""
    rows = []
    for (u, v), (x, y) in zip(source, target):
        rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v])
        rows.append([0, 0, 0, u, v, 1, -y * u, -y * v])
    solved = np.linalg.solve(np.array(rows, dtype=np.float64), target.reshape(-1))
    return np.append(solved, 1.0).reshape(3, 3)


def _patch(image: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """The pixels of a small square about ``centre``, or an error if it is off frame."""
    x, y = centre
    half = max(2, int(radius))
    x0, x1 = int(round(x)) - half, int(round(x)) + half + 1
    y0, y1 = int(round(y)) - half, int(round(y)) + half + 1
    height, width = image.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ReadError("the chart runs outside the photograph — include all four corners")
    return image[y0:y1, x0:x1]


def _sample(image: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """Median colour of a patch, which ignores dust and the odd specular fleck."""
    return np.median(_patch(image, centre, radius).reshape(-1, 3), axis=0)


def _sample_one(plane: np.ndarray, centre: np.ndarray, radius: float) -> float:
    """The same, for a single-channel image such as luminance."""
    return float(np.median(_patch(plane, centre, radius)))


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _locate(
    rgb: np.ndarray, luma: np.ndarray, layout: Layout, frame: np.ndarray
) -> np.ndarray:
    """Find the chart's black frame — and prove it is the frame before returning.

    The first version thresholded once, at the midpoint of the photograph's
    range, and took the largest dark blob. That works only when the chart is the
    only dark thing in shot. It is not: the chart gets printed on whatever
    offcut is to hand, so there is bare glass around it, and coloured glass is
    itself dark. On a wide plate of dark green the largest dark blob is the
    *plate*, and the reader then sampled a grid of colours that were not there
    and returned a profile wrong by 200 levels — without complaining, which is
    the worst way for a measurement to fail.

    So the threshold is swept rather than guessed, and each candidate has to
    earn it. The chart's own structure is the test: the four grid corners print
    nothing and must come out bright, the frame band is solid ink and must come
    out dark. A quad drawn round the whole plate has bare glass in both places
    and scores about 1. The real frame scores several times that.
    """
    corner_cells = list(glass_cells(layout))
    centres = layout.centres()
    # Midpoints of the four frame edges, halfway through the ink band.
    inset = layout.frame_mm / 2.0
    band = np.array(
        [
            [layout.frame_w_mm / 2, inset],
            [layout.frame_w_mm / 2, layout.frame_h_mm - inset],
            [inset, layout.frame_h_mm / 2],
            [layout.frame_w_mm - inset, layout.frame_h_mm / 2],
        ]
    )

    found_any = False
    best_score, best_corners = 0.0, None
    for percentile in (1, 2, 4, 7, 10, 14, 19, 25, 32, 40, 50, 62, 75):
        mask = luma <= np.percentile(luma, percentile)
        try:
            corners = _quad(mask)
        except ReadError:
            continue
        found_any = True

        # A chart wider than it is tall, photographed on its side, is still
        # legible; rotate the corner assignment rather than making that an error.
        wide = np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
        tall = np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])
        if (layout.frame_w_mm > layout.frame_h_mm) != (wide > tall):
            corners = np.roll(corners, 1, axis=0)

        matrix = _homography(frame, corners)
        try:
            spots = _project(matrix, centres)
            radius = max(2.0, 0.2 * layout.patch_mm * np.linalg.norm(spots[1] - spots[0]) / layout.pitch_mm)
            bare = min(_sample_one(luma, spots[i], radius) for i in corner_cells)
            inked = max(_sample_one(luma, spot, radius) for spot in _project(matrix, band))
        except ReadError:
            continue

        score = bare / max(inked, 1e-6)
        if score > best_score:
            best_score, best_corners = score, corners

    if not found_any:
        raise ReadError("no printed chart found — is the photograph of the right plate?")
    if best_corners is None or best_score < 1.8:
        raise ReadError(
            "found something dark but it is not the chart — check the whole chart is in "
            "frame, in focus, and lit from behind rather than in front"
        )
    return best_corners


def read(
    photo: np.ndarray,
    *,
    glass: RGB | None = None,
    layout: Layout = CHART,
    substrate: str = "transparent",
) -> Profile:
    """Measure a printed chart from a photograph of it.

    ``photo`` is an ordinary sRGB image — a phone picture of the plate held
    against a bright, even, white background. Every patch is measured against
    bare glass beside it, so the glass tint, the lamp and the camera's exposure
    all appear in both terms and all cancel. What does not cancel is a camera in
    HDR mode remapping the darks, so shoot it flat.
    """
    if photo.ndim != 3 or photo.shape[2] < 3:
        raise ReadError("expected a colour photograph")
    rgb = linear(photo[:, :, :3].astype(np.float64) / 255.0)

    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    frame = np.array(
        [
            [0.0, 0.0],
            [layout.frame_w_mm, 0.0],
            [layout.frame_w_mm, layout.frame_h_mm],
            [0.0, layout.frame_h_mm],
        ]
    )
    corners = _locate(rgb, luma, layout, frame)
    centres = layout.centres()
    cells = patches(layout)
    corner_cells = glass_cells(layout)
    black_cell = next(i for i, c in enumerate(cells) if c == (0, 0, 0))
    opposite = len(cells) - 1 - black_cell

    def sample_all(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = _homography(frame, quad)
        spots = _project(matrix, centres)
        scale = np.linalg.norm(spots[1] - spots[0]) / layout.pitch_mm
        radius = 0.3 * layout.patch_mm * scale
        return np.array([_sample(rgb, spot, radius) for spot in spots]), spots

    samples, _ = sample_all(corners)
    # The four grid corners print nothing, so the chart is symmetric under a
    # half turn except for the solid black cell. If the cell diagonally opposite
    # it is the darker of the two, the photograph is upside down.
    if samples[opposite].mean() < samples[black_cell].mean():
        corners = np.roll(corners, 2, axis=0)
        samples, _ = sample_all(corners)

    bare = samples[list(corner_cells)]

    # A channel the substrate barely passes cannot be measured, and that is not
    # a fault in the photograph. Deep green glass passes so little blue that a
    # well-exposed backlit shot reads 0, 1, 7, 0 out of 255 in its four bare
    # corners — there is no blue light arriving to find out what the ink does
    # to it. The first version called that underexposure and refused, which
    # sent a good photograph back to be taken again.
    level = bare.mean(axis=0)
    unmeasured = tuple(
        name for name, value in zip(("red", "green", "blue"), level) if value < _DEAD_CHANNEL
    )
    if len(unmeasured) == 3:
        raise ReadError(
            "every channel came back black — this really is underexposed, or the plate is "
            "not lit from behind at all"
        )

    # The illumination where each patch sits, interpolated bilinearly from the
    # four bare corners. This is what makes the reading immune to a lamp that
    # falls off across the plate, which a phone's is guaranteed to do.
    u = centres[:, 0] / layout.frame_w_mm
    v = centres[:, 1] / layout.frame_h_mm
    weights = np.stack(
        [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=1
    )
    reference = weights @ bare

    _refuse_if_veiled(samples, reference, bare, cells, black_cell, unmeasured)

    keep = [i for i in range(len(cells)) if i not in set(corner_cells)]
    ink = np.clip(samples[keep] / reference[keep], _FLOOR, 1.0)
    live = [i for i in range(3) if ("red", "green", "blue")[i] not in unmeasured]

    # The chart prints one colour twice. The two must agree, and how far they do
    # not is the noise floor of the whole reading — no fit can be more
    # consistent than the numbers it is fitted to. Checked before fitting,
    # because a fit to noise still returns a tone curve and a density and looks
    # for all the world like a measurement.
    #
    # It doubles as the way to tell which chart is on the plate. Charts printed
    # before the held-out patches were spread carry the colours in a different
    # order, and read against the wrong one those two patches are not the same
    # colour at all — so they disagree enormously, and the layout whose repeat
    # agrees is the layout that was printed.
    best = None
    for name, order, kept, (original, repeat) in LAYOUTS:
        table = patches(layout, order)
        apart = float(np.abs(ink[original][live] - ink[repeat][live]).mean() * 255.0)
        if best is None or apart < best[0]:
            best = (apart, name, table, kept)

    apart, layout_name, cells, kept = best
    requested = np.array([cells[i] for i in keep], dtype=np.float64)
    if layout_name != LAYOUTS[0][0]:
        note_layout = f"; read as the {layout_name} chart layout"
    else:
        note_layout = ""
    if apart > _MAX_REPEAT:
        raise ReadError(
            f"the same colour printed twice on this plate reads {apart:.0f} levels apart, so "
            "nothing fitted to it would mean much. Two causes. Usually the light: a lamp or a "
            "small window close to the plate falls off across it faster than the four corners "
            "can correct — use a big even source, an overcast window or a screen filling the "
            "frame, square on to the plate. Or the plate was printed from an older chart, whose "
            "colours are in a different order; reprint it from the current file."
        )

    ratio = float(bare[:, live].max() / max(bare[:, live].min(), 1e-6))

    measured_glass = glass
    if measured_glass is None:
        measured_glass = _glass_from_background(
            rgb, luma, corners.mean(axis=0), bare.mean(axis=0)
        )

    note = "measured" + note_layout
    if ratio > _MAX_CORNER_RATIO:
        note = f"measured{note_layout}, but the light fell off {ratio:.1f}x across the plate"
    if measured_glass is None:
        measured_glass = (255, 255, 255)
        note = "measured, but the glass colour could not be read — include some background around the plate"

    if unmeasured:
        which = " and ".join(unmeasured)
        note = f"{note}; {which} unmeasurable — the substrate passes almost none of it"
    return fit(
        requested,
        ink,
        measured_glass,
        note=note,
        substrate=substrate,
        unmeasured=unmeasured,
        held_out=kept,
    )


#: Below this share of the illumination, a channel carries no usable signal —
#: it is the substrate blocking it, not the camera missing it.
_DEAD_CHANNEL = 0.01

#: How much light the solid-black cell may still pass before the photograph is
#: judged to be measuring a reflection rather than the print. One pass of black
#: on glass is only dark grey, so this is deliberately generous; the reading it
#: was written for sat at 0.53.
_MAX_BLACK = 0.40

#: How far apart the two prints of the repeated colour may read, in levels out
#: of 255. This is the measurement measuring itself, so it is a hard ceiling on
#: what any fit can achieve — a model cannot be more consistent than its data.
_MAX_REPEAT = 12.0

#: How much brighter one corner may be than another before the lamp is too
#: uneven for a bilinear reference to straighten out.
_MAX_CORNER_RATIO = 1.6

#: How far the four bare-substrate corners may disagree in *colour*. Brightness
#: is allowed to vary — that is the lamp, and the bilinear reference removes it.
#: Hue is not: all four are the same substrate, so a difference means something
#: that is not the substrate is being measured.
_MAX_CORNER_CHROMA = 0.05


def _refuse_if_veiled(
    samples: np.ndarray,
    reference: np.ndarray,
    bare: np.ndarray,
    cells: list,
    black_cell: int,
    unmeasured: tuple[str, ...] = (),
) -> None:
    """Reject a photograph that is measuring light off the surface, not through it.

    Both checks come free from cells the chart already has, and both were
    written after a real plate came back unreadable in a way nothing else
    noticed. The profile that plate produced was not obviously wrong — it had a
    tone curve, a density, a glass colour. It was simply not a measurement of
    the ink.

    **The black cell.** Solid black is the most ink the printer can lay down, so
    it is the darkest thing on the plate. Photographed properly it passes very
    little. If it comes back bright, light is arriving at the camera without
    having gone through the ink — a reflection off the front face, or flare.
    That veil is *additive*, so it lands hardest on the darkest patches and
    flattens the whole scale.

    **The corners.** All four print the same substrate, so they may differ in
    brightness — that is the lamp — but not in hue. When they do, something
    coloured and not the substrate is in the light path. On the reading this was
    written for, the two top corners carried four times the blue of the two
    bottom ones: a window reflected in the upper half of the plate.
    """
    live = [i for i in range(3) if ("red", "green", "blue")[i] not in unmeasured]
    black = (samples[black_cell] / np.maximum(reference[black_cell], 1e-6))[live]
    if float(np.mean(black)) > _MAX_BLACK:
        raise ReadError(
            f"solid black came back at {float(np.mean(black)):.0%} of bare substrate, so most "
            "of what the camera saw never went through the ink. Light is reflecting off the "
            "front of the plate — shoot it square-on with the light behind it (or, for an "
            "opaque substrate, off to one side), and keep windows and lamps out of the reflection."
        )

    chroma = bare[:, live] / np.maximum(bare[:, live].sum(axis=1, keepdims=True), 1e-6)
    spread = float(np.ptp(chroma, axis=0).max())
    if spread > _MAX_CORNER_CHROMA:
        raise ReadError(
            f"the four bare-substrate corners differ in colour by {spread:.2f}, which they "
            "cannot do — they are the same glass. Something other than the substrate is in "
            "the light path, usually a window or lamp reflected in part of the plate. "
            "Move so the reflection is gone, then shoot again."
        )


def _glass_from_background(
    rgb: np.ndarray, luma: np.ndarray, centre: np.ndarray, bare: np.ndarray
) -> RGB | None:
    """The glass's own transmittance: bare glass against the paper behind it.

    Finding the *plate* was the obvious way in, and it does not work. On a white
    background a piece of dark glass separates by brightness, and a piece of
    pale green glass does not — so the outline came back as the chart, the ring
    around it was more bare glass, and dividing glass by glass declared every
    substrate colourless.

    What actually distinguishes the background is that it is **neutral**: white
    paper has no hue, coloured glass does, whatever its brightness.

    Which leaves the lamp. Taking the brightest neutral pixels biases the
    reference toward whichever side of the frame is better lit — measurably so,
    at about a quarter of the falloff, which on a 45% gradient made every glass
    read 11% too dark. So instead of picking pixels, the background's own
    gradient is fitted as a plane and read off at the middle of the chart. That
    is where the bare-glass corners are averaged from, so the two are finally
    being compared under the same light.

    Reported on the same convention the rest of the codebase uses — RGB over 255
    *is* the transmittance, linear — so it can be handed straight to
    :mod:`glassprint.simulate` without a conversion nobody would remember.
    """
    span = np.ptp(rgb, axis=2)
    peak = np.max(rgb, axis=2)
    neutral = span <= 0.16 * np.maximum(peak, 1e-6)
    # A weak floor: paper, not the shadow under the plate. Deliberately not a
    # high percentile, which is what caused the bias in the first place.
    neutral &= luma >= np.percentile(luma, 55)
    if neutral.sum() < 500:
        return None

    ys, xs = np.nonzero(neutral)
    if len(xs) > 40000:  # a plane needs no more than this, and fits faster
        step = len(xs) // 40000 + 1
        ys, xs = ys[::step], xs[::step]

    design = np.stack([np.ones(len(xs)), xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, rgb[ys, xs], rcond=None)
    white = np.array([1.0, centre[0], centre[1]]) @ coefficients
    if float(np.min(white)) <= 0:
        return None

    transmitted = np.clip(bare / white, 0.0, 1.0)
    if float(np.max(transmitted)) < 0.02:
        return None
    return tuple(int(round(float(c) * 255)) for c in transmitted)  # type: ignore[return-value]


# -- the printable -----------------------------------------------------------


def chart(
    layout: Layout = CHART,
    *,
    dpi: float = 600.0,
    label: str = "",
    substrate: str = "transparent",
):
    """Draw the chart to print. One pass, no registration to get wrong.

    Returns a :class:`~glassprint.raster.Raster`. Deliberately a single pass:
    this measures what one layer of ink does, which is the thing every other
    prediction is built out of, and asking for four passes would put the
    plate's registration between the question and the answer.

    ``substrate`` changes the corner cells, and that is not cosmetic.
    Everything is measured as a ratio against them, so those four cells have to
    *be* whatever the ink is sitting on. Straight onto glass — clear or opaque —
    that is the bare glass, so the corners are holes. Over a white underbase a
    hole is bare glass while every patch beside it sits on white ink, and the
    ratio is then two different substrates divided by each other, which measures
    nothing; so there the corners print solid white instead.

    Clear and opaque glass therefore produce an identical file. What differs
    between them is the light you photograph it under, which the caller has to
    get right — see :attr:`Profile.REFLECTIVE`.
    """
    from PIL import Image, ImageDraw

    from .raster import Raster, mm_to_px

    def px(value: float) -> int:
        return int(round(mm_to_px(value, dpi)))

    page = Image.new("RGBA", (px(layout.width_mm), px(layout.height_mm)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(page)
    origin = px(layout.margin_mm)

    # The frame and the gaps are one connected mesh of ink, which is what the
    # reader finds; drawing it as a filled block with the patches punched out
    # keeps the two definitions from ever disagreeing about a gap width.
    draw.rectangle(
        [origin, origin, origin + px(layout.frame_w_mm) - 1, origin + px(layout.frame_h_mm) - 1],
        fill=(0, 0, 0, 255),
    )
    reference = (255, 255, 255, 255) if substrate == "white" else (0, 0, 0, 0)
    for index, colour in enumerate(patches(layout)):
        x, y, w, h = layout.cell(index)
        box = [origin + px(x), origin + px(y), origin + px(x + w) - 1, origin + px(y + h) - 1]
        draw.rectangle(box, fill=(*colour, 255) if colour else reference)

    # The caption is the only part that can fail — in the browser build there
    # are no system fonts to load. A chart without its caption is still a
    # perfectly good chart, so it must not take the rest down with it.
    described = {
        "white": "WITH white base",
        "opaque": "NO white base · opaque glass",
    }.get(substrate, "NO white base · clear glass")
    caption = label or f"glassprint colour chart · one pass · {described} · 100%"
    try:
        draw.text(
            (origin, origin + px(layout.frame_h_mm) + px(0.8)),
            caption,
            font=_font(dpi, 1.4),
            fill=(0, 0, 0, 255),
        )
    except Exception:  # pragma: no cover - depends on which fonts exist
        pass

    return Raster(np.array(page, dtype=np.uint8), dpi=(dpi, dpi))


def _font(dpi: float, cap_mm: float):
    from PIL import ImageFont

    from .raster import mm_to_px

    size = int(round(mm_to_px(cap_mm, dpi) * 1.35))
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
