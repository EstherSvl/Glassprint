"""Building a colour by stacking layers of *different* ink.

Repeating one ink only amplifies its own spectral shape — ``ink ** n`` — so it
can deepen a colour but never move it far sideways. Stacking different inks
multiplies different shapes together, which reaches colours a single ink cannot.
That is glazing, and on tinted glass it is really a colour separation where the
paper happens to be green.

The maths is linear once you take logs. Transmittances multiply::

    result = glass · ink₁^n₁ · ink₂^n₂ · …

so in absorbance (``a = -log₁₀ T``) the passes simply add up::

    a_result = a_glass + n₁·a₁ + n₂·a₂ + …

Finding a recipe is then "which non-negative whole numbers of passes add up to
the absorbance I still need", which is small enough to solve exactly.

The one hard limit falls out of the same equation: every pass *adds* absorbance,
so you can only ever make things darker. A target lighter than the glass in any
channel is unreachable at any number of layers — that is what white ink is for.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .colors import RGB, dominant_colors, parse_color, to_hex

#: Perceptual-ish channel weights, matching the rest of the codebase.
_WEIGHTS = np.array([0.30, 0.59, 0.11], dtype=np.float32)

#: Floor on transmittance so absorbance stays finite.
_FLOOR = 0.02


@dataclass(frozen=True)
class Ink:
    """One printable ink, described by what it lets through."""

    name: str
    rgb: RGB

    @property
    def transmittance(self) -> np.ndarray:
        return np.clip(np.array(self.rgb, dtype=np.float32) / 255.0, _FLOOR, 1.0)

    @property
    def absorbance(self) -> np.ndarray:
        return -np.log10(self.transmittance)


#: Nominal process inks. These are transmittances, not screen colours — a UV
#: printer's actual inks will differ, so override them if you measure yours.
PROCESS_INKS: tuple[Ink, ...] = (
    Ink("cyan", (38, 191, 230)),
    Ink("magenta", (230, 38, 179)),
    Ink("yellow", (242, 217, 26)),
    Ink("black", (64, 64, 64)),
)


def palette_from(spec: str | None) -> tuple[Ink, ...]:
    """Parse ``"cyan,magenta,#7a2f8a"`` into inks, defaulting to process colours."""
    if not spec or not spec.strip():
        return PROCESS_INKS[:3]

    by_name = {ink.name: ink for ink in PROCESS_INKS}
    inks: list[Ink] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token.lower() in by_name:
            inks.append(by_name[token.lower()])
            continue
        rgb = parse_color(token)
        if rgb is None:
            raise ValueError(f"cannot read ink {token!r}")
        inks.append(Ink(to_hex(rgb), rgb))
    if not inks:
        raise ValueError("the glaze palette is empty")
    return tuple(inks)


@dataclass
class GlazePass:
    ink: Ink
    passes: int


@dataclass
class Recipe:
    """How to build one target colour on one glass."""

    target: RGB
    achieved: RGB
    passes: list[GlazePass] = field(default_factory=list)
    error: float = 0.0
    #: How far the target had to be dimmed to fit under the glass. 1.0 means it
    #: fitted as asked.
    dimming: float = 1.0
    reachable: bool = True
    note: str = ""

    @property
    def total_passes(self) -> int:
        return sum(p.passes for p in self.passes)

    def describe(self) -> str:
        if not self.passes:
            return "bare glass"
        return " + ".join(
            f"{p.passes}x {p.ink.name}" if p.passes > 1 else p.ink.name for p in self.passes
        )

    def as_dict(self) -> dict:
        return {
            "target": to_hex(self.target),
            "achieved": to_hex(self.achieved),
            "recipe": self.describe(),
            "passes": [{"ink": p.ink.name, "hex": to_hex(p.ink.rgb), "passes": p.passes} for p in self.passes],
            "total_passes": self.total_passes,
            "error": round(self.error, 4),
            "dimming": round(self.dimming, 3),
            "reachable": self.reachable,
            "note": self.note,
        }


def solve(
    target: RGB,
    glass: RGB,
    palette: tuple[Ink, ...] = PROCESS_INKS[:3],
    *,
    max_per_ink: int = 3,
    max_total: int = 5,
) -> Recipe:
    """Find the stack of ink layers that best reproduces ``target`` on ``glass``."""
    target_t = np.clip(np.array(target, dtype=np.float32) / 255.0, _FLOOR, 1.0)
    glass_t = np.clip(np.array(glass, dtype=np.float32) / 255.0, _FLOOR, 1.0)

    # Every pass only ever subtracts light, so anything brighter than the glass
    # is out of reach however you stack it.
    unreachable_channels = [
        name
        for name, want, have in zip("red green blue".split(), target_t, glass_t)
        if want - have > 0.06
    ]

    # Aim instead at the same colour dimmed until it fits under the glass. That
    # keeps the hue and gives up only the brightness that was never available —
    # far better advice than the literal nearest colour, which for a too-bright
    # target is "print nothing at all".
    dimming = float(min(1.0, float(np.min(glass_t / target_t))))
    aim = target_t * dimming if unreachable_channels else target_t

    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    for counts in itertools.product(range(max_per_ink + 1), repeat=len(palette)):
        total = sum(counts)
        if total > max_total:
            continue
        achieved = glass_t.copy()
        for count, ink in zip(counts, palette):
            if count:
                achieved = achieved * np.power(ink.transmittance, count)
        # A hair of preference for fewer passes, so we do not stack four layers
        # for a gain nobody could see.
        error = _distance(achieved, aim) + 0.0005 * total
        if best is None or error < best[0]:
            best = (error, counts, achieved)

    assert best is not None
    _, counts, achieved = best
    # Score against what was actually achievable. Reporting distance to a
    # colour the glass forbids would rank a wrong hue above a right one purely
    # because it happened to be lighter.
    error = _distance(achieved, aim)

    note = ""
    if unreachable_channels:
        channels = " and ".join(unreachable_channels)
        note = (
            f"Brighter than the glass in {channels}, so no stack reaches it — ink only "
            f"removes light. Aimed at the same colour at {dimming:.0%} brightness, which is "
            "as light as this glass goes. A white underbase is the only way to get the rest."
        )
    elif error > 0.10:
        note = "The palette cannot get close to this one; try adding an ink or raising the pass limit."

    return Recipe(
        target=target,
        achieved=tuple(int(round(c * 255)) for c in achieved),  # type: ignore[arg-type]
        passes=[GlazePass(ink, n) for n, ink in zip(counts, palette) if n],
        error=error,
        dimming=dimming,
        reachable=not unreachable_channels and error <= 0.10,
        note=note,
    )


def _distance(achieved: np.ndarray, target: np.ndarray) -> float:
    """How different two colours look, on the "redmean" approximation.

    Deliberately *not* the luminance weights used elsewhere in the codebase.
    Those answer "how visible is this edge", and weight blue at 0.11 — which
    when used to match colours lets a large blue error pass almost unnoticed,
    and happily calls teal a good match for green. This weights the channels
    much more evenly, shifting with how red the pair is.
    """
    mean_red = float(achieved[0] + target[0]) / 2.0
    weights = np.array(
        [2.0 + mean_red, 4.0, 2.0 + (1.0 - mean_red)], dtype=np.float32
    )
    return float(np.sqrt((((achieved - target) ** 2) * weights).sum() / weights.sum()))


@dataclass
class GlazePlan:
    """Recipes for each colour in the artwork, plus where each colour sits."""

    glass: RGB
    palette: tuple[Ink, ...]
    colours: list[RGB]
    labels: np.ndarray          # index into `colours` per pixel
    recipes: list[Recipe]

    @property
    def stack(self) -> list[tuple[Ink, int]]:
        """The printing plan: every (ink, pass number) any colour asks for."""
        deepest: dict[str, int] = {}
        for recipe in self.recipes:
            for item in recipe.passes:
                deepest[item.ink.name] = max(deepest.get(item.ink.name, 0), item.passes)
        order = {ink.name: index for index, ink in enumerate(self.palette)}
        return [
            (ink, index)
            for ink in sorted(self.palette, key=lambda i: order[i.name])
            for index in range(1, deepest.get(ink.name, 0) + 1)
        ]

    def counts_for(self, ink: Ink) -> np.ndarray:
        """How many passes of one ink each pixel wants."""
        per_colour = np.zeros(len(self.colours), dtype=np.float32)
        for index, recipe in enumerate(self.recipes):
            for item in recipe.passes:
                if item.ink.name == ink.name:
                    per_colour[index] = item.passes
        return per_colour[self.labels]

    def achieved_image(self) -> np.ndarray:
        """The colour each pixel actually comes out as, glass included."""
        table = np.array([recipe.achieved for recipe in self.recipes], dtype=np.float32) / 255.0
        return table[self.labels]

    def as_dict(self) -> dict:
        return {
            "glass": to_hex(self.glass),
            "palette": [ink.name for ink in self.palette],
            "stack": [f"{ink.name} #{index}" for ink, index in self.stack],
            "total_passes": len(self.stack),
            "recipes": [recipe.as_dict() for recipe in self.recipes],
            "unreachable": [r.as_dict() for r in self.recipes if not r.reachable],
        }


def plan(
    rgb: np.ndarray,
    coverage: np.ndarray,
    glass: RGB,
    palette: tuple[Ink, ...] = PROCESS_INKS[:3],
    *,
    colours: int = 5,
    max_per_ink: int = 3,
    max_total: int = 5,
) -> GlazePlan:
    """Work out a glaze recipe for each of the artwork's main colours.

    Solving per colour rather than per pixel keeps the result printable: flat
    artwork becomes a handful of passes with a recipe you can read, instead of
    a per-pixel separation nothing could register.
    """
    found = dominant_colors(rgb, coverage, count=max(1, colours))
    if not found:
        found = [(255, 255, 255)]

    # Snap every pixel to its nearest main colour, so the regions partition the
    # artwork instead of overlapping.
    table = np.array(found, dtype=np.float32) / 255.0
    diff = rgb[:, :, None, :] - table[None, None, :, :]
    labels = np.argmin((diff**2 * _WEIGHTS).sum(axis=-1), axis=-1).astype(np.int32)

    recipes = [
        solve(colour, glass, palette, max_per_ink=max_per_ink, max_total=max_total)
        for colour in found
    ]
    return GlazePlan(glass=glass, palette=palette, colours=found, labels=labels, recipes=recipes)


def render(plan_: GlazePlan, coverage: np.ndarray, glass: RGB) -> np.ndarray:
    """What the glazed print looks like: achieved colours where inked, glass elsewhere."""
    glass_rgb = np.array(glass, dtype=np.float32)[None, None, :] / 255.0
    alpha = np.clip(coverage, 0.0, 1.0)[:, :, None]
    return np.clip(glass_rgb * (1.0 - alpha) + plan_.achieved_image() * alpha, 0.0, 1.0)


def compare_single_ink(target: RGB, glass: RGB, max_passes: int = 5) -> Recipe:
    """Best a single repeated ink can do, for comparison against a glaze."""
    ink = Ink(to_hex(target), target)
    return solve(target, glass, (ink,), max_per_ink=max_passes, max_total=max_passes)
