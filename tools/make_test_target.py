"""Build a calibration tile to print on glass.

Everything glassprint decides about fades rests on numbers I picked from
reasoning rather than measurement: that dithering breaks up somewhere around
12% coverage, that a dot screen finer than about half a millimetre risks
beating against the printer's own screening, that alpha drives the white
underbase smoothly rather than as a threshold. This tile puts each of those on
one piece of glass so the guesses can be replaced with readings.

    python tools/make_test_target.py

The screened patches are rendered by ``glassprint.fade`` itself, not drawn to
look like it — so what comes out of the printer is the real output of the code
being calibrated.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

from glassprint.fade import halftone
from glassprint.raster import Raster, mm_to_px

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-target"

# Small on purpose: it has to fit any bed, and a calibration print should cost
# a scrap of glass and a thimble of ink, not a good sheet.
SIZE_MM = 80.0
DPI = 600.0
MARGIN_MM = 4.0

INK = (20, 20, 24)  # near-black: dither speckle shows against it soonest

#: Coverage steps down the tonal ramp. The interesting region is the bottom.
STEPS = [100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6, 4, 2]

#: Dot pitches in millimetres, spanning "certainly too fine" to "certainly safe".
PITCHES = [0.25, 0.4, 0.6, 0.8, 1.2, 1.8]

#: Line widths in millimetres, for the resolution limit.
LINES = [0.08, 0.12, 0.2, 0.3, 0.5]

SWATCHES = [
    ("C", (0, 158, 224)),
    ("M", (226, 0, 122)),
    ("Y", (255, 237, 0)),
    ("K", (26, 23, 27)),
    ("R", (200, 40, 40)),
    ("G", (40, 150, 80)),
    ("B", (40, 70, 180)),
    ("50K", (128, 128, 128)),
]


def mm(value: float) -> int:
    return mm_to_px(value, DPI)


def font(points: float) -> ImageFont.FreeTypeFont:
    """A size in millimetres of cap height, near enough."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", mm(points))
    except OSError:
        return ImageFont.load_default(size=mm(points))


def build(variant: str) -> Raster:
    side = mm(SIZE_MM)
    # Transparent: on this printer the alpha channel is what generates the white
    # underbase, so the tile has to carry its tone as alpha, not as pale ink.
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    small = font(1.4)
    tiny = font(1.1)
    left = mm(MARGIN_MM)
    width = side - 2 * left
    y = MARGIN_MM  # tracked in millimetres, so the budget below is readable

    def label(text: str, x: int, top_mm: float, chosen=None) -> None:
        draw.text((x, mm(top_mm)), text, font=chosen or small, fill=(*INK, 255))

    # -- title and scale bar -------------------------------------------------
    label(f"glassprint calibration · {SIZE_MM:.0f}mm · {DPI:.0f}dpi · {variant}", left, y)
    y += 2.4
    draw.rectangle([left, mm(y), left + mm(10), mm(y) + mm(0.7)], fill=(*INK, 255))
    label("10mm — measure this", left + mm(11), y - 0.3, tiny)
    y += 2.6

    # -- 1. tonal ramp, stepped ---------------------------------------------
    label("1  flat tone by alpha — where does it go gritty?", left, y)
    y += 2.0
    step_w = width // len(STEPS)
    for index, percent in enumerate(STEPS):
        x = left + index * step_w
        draw.rectangle(
            [x, mm(y), x + step_w - 1, mm(y + 6.0)], fill=(*INK, round(percent * 2.55))
        )
        label(str(percent), x + mm(0.2), y + 6.2, tiny)
    y += 9.4

    # -- 2. the same tones as a dot screen -----------------------------------
    label("2  the same tones as 0.8mm dots — compare with row 1", left, y)
    y += 2.0
    band = np.zeros((mm(6.0), width), dtype=np.float32)
    for index, percent in enumerate(STEPS):
        band[:, index * step_w : (index + 1) * step_w] = percent / 100.0
    paste_mask(canvas, halftone(band, mm(0.8), angle=45.0), left, mm(y))
    y += 7.5

    # -- 3. dot pitch, for moire against the printer's own screen -------------
    label("3  dot pitch at 50% — any interference patterns?", left, y)
    y += 2.0
    patch = mm(8.0)
    gap = (width - len(PITCHES) * patch) // max(1, len(PITCHES) - 1)
    for index, pitch in enumerate(PITCHES):
        x = left + index * (patch + gap)
        block = np.full((patch, patch), 0.5, dtype=np.float32)
        paste_mask(canvas, halftone(block, mm(pitch), angle=45.0), x, mm(y))
        label(f"{pitch}", x, y + 8.2, tiny)
    y += 11.4

    # -- 4. solid colour, for what the glass does to it -----------------------
    label("4  solid colour — photograph against light and against dark", left, y)
    y += 2.0
    swatch = width // len(SWATCHES)
    for index, (name, rgb) in enumerate(SWATCHES):
        x = left + index * swatch
        draw.rectangle([x, mm(y), x + swatch - 1, mm(y + 6.0)], fill=(*rgb, 255))
        label(name, x + mm(0.2), y + 6.2, tiny)
    y += 9.4

    # -- 5. fine detail -------------------------------------------------------
    label("5  line width in mm — the thinnest that survives", left, y)
    y += 2.2
    x = left
    for thickness in LINES:
        w = max(1, mm(thickness))
        draw.rectangle([x, mm(y), x + w - 1, mm(y + 5.0)], fill=(*INK, 255))
        label(f"{thickness}", x - mm(0.4), y + 5.2, tiny)
        x += w + mm(3.4)
    y += 8.4

    # -- 6. a continuous fade, which is the thing this is all for -------------
    label("6  smooth fade — banding? a hard stop? where does it vanish?", left, y)
    y += 2.0
    ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)[None, :].repeat(mm(6.0), axis=0)
    paste_mask(canvas, ramp, left, mm(y))
    label("100%", left, y + 6.2, tiny)
    label("0%", left + width - mm(3.4), y + 6.2, tiny)
    y += 8.4

    # Laying this out by hand is exactly the sort of thing that silently runs
    # off the edge, and a calibration tile with a row missing is worse than none.
    if y > SIZE_MM - MARGIN_MM:
        raise SystemExit(f"layout overflows: needs {y:.1f}mm of {SIZE_MM - MARGIN_MM:.1f}mm")

    # A hairline cross in each corner: print this twice and these say how far
    # the second pass landed from the first.
    for cx, cy in [
        (left, mm(MARGIN_MM)),
        (side - left, mm(MARGIN_MM)),
        (left, side - mm(MARGIN_MM)),
        (side - left, side - mm(MARGIN_MM)),
    ]:
        arm, w = mm(2.0), max(1, mm(0.1))
        draw.rectangle([cx - arm, cy - w // 2, cx + arm, cy + w // 2], fill=(*INK, 255))
        draw.rectangle([cx - w // 2, cy - arm, cx + w // 2, cy + arm], fill=(*INK, 255))

    return Raster(np.array(canvas, dtype=np.uint8), dpi=(DPI, DPI), name="glassprint-test")


def paste_mask(canvas: Image.Image, mask: np.ndarray, x: int, y: int) -> None:
    """Drop a 0..1 coverage mask onto the canvas as ink with matching alpha."""
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.array(INK, dtype=np.uint8)[None, None, :]
    rgba[:, :, 3] = np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(rgba, mode="RGBA"), (x, y))


def stack_test() -> list[Raster]:
    """Four identical files, for printing one on top of another.

    Glazing depends on repeated passes landing on each other and on ink
    genuinely deepening when it does. Printing these in order answers both:
    how far registration drifts, and whether pass four looks meaningfully
    different from pass two.
    """
    side = mm(40.0)
    out = []
    for index in range(1, 5):
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        # Each pass covers a shorter column, so one print shows all four depths
        # side by side and registration shows up along their shared edges.
        for column in range(4):
            if column + 1 >= index:
                x = mm(4 + column * 8)
                draw.rectangle([x, mm(6), x + mm(7), mm(30)], fill=(*INK, 255))
        # Staggered, because all four print onto the same glass: overlapping
        # text would cure into one black smudge instead of a legible tally.
        draw.text((mm(4), mm(30.4 + index * 1.9)), f"pass {index}", font=font(1.3), fill=(*INK, 255))
        if index == 1:
            # Drawn once only, for the same reason: this says how many layers
            # each column ends up with once all four have gone down.
            for column in range(4):
                draw.text(
                    (mm(6 + column * 8), mm(2)), f"{column + 1}", font=font(1.5), fill=(*INK, 255)
                )
        arm, w = mm(1.5), max(1, mm(0.1))
        for cx, cy in [(mm(3), mm(3)), (side - mm(3), mm(3))]:
            draw.rectangle([cx - arm, cy - w // 2, cx + arm, cy + w // 2], fill=(*INK, 255))
            draw.rectangle([cx - w // 2, cy - arm, cx + w // 2, cy + arm], fill=(*INK, 255))
        out.append(Raster(np.array(canvas, dtype=np.uint8), dpi=(DPI, DPI)))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Two files, identical but for the label. Printed and set aside, two pieces
    # of glass are otherwise impossible to tell apart a week later.
    for variant, name in [("WITH white base", "with-white"), ("NO white base", "no-white")]:
        tile = build(variant)
        tile.save(OUT / f"glassprint-test-{name}.png", fmt="png", dpi=(DPI, DPI))
        print(f"glassprint-test-{name}.png  {tile.width}×{tile.height}px  {SIZE_MM:.0f}×{SIZE_MM:.0f}mm")

    for index, raster in enumerate(stack_test(), start=1):
        name = f"glassprint-stack-pass{index}of4.png"
        raster.save(OUT / name, fmt="png", dpi=(DPI, DPI))
    print(f"glassprint-stack-pass1..4of4.png  40×40mm each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
