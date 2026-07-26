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
# a scrap of glass and a thimble of ink, not a good sheet. The wide variant is
# cut to the mini flatbed, where a 300mm ramp resolves banding that a 70mm one
# hides entirely.
SIZE_MM = 90.0
SHAPES = {"tile": (90.0, 90.0), "short": (160.0, 85.0), "strip": (320.0, 85.0)}
DPI = 600.0
MARGIN_MM = 4.0

INK = (20, 20, 24)  # near-black: dither speckle shows against it soonest

#: Coverage steps down the tonal ramp. The interesting region is the bottom.
STEPS = [100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6, 4, 2]

#: Dot pitches in millimetres, spanning "certainly too fine" to "certainly safe".
PITCHES = [0.25, 0.4, 0.6, 0.8, 1.2, 1.8]

#: Line widths in millimetres, for the resolution limit.
LINES = [0.08, 0.12, 0.2, 0.3, 0.5]

#: Cap heights in millimetres. The labels on this tile are already a legibility
#: test at 1.1 and 1.4mm; this makes that deliberate and gives it a range.
TEXT_SIZES = [0.8, 1.2, 1.8, 2.5]

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


def px_mm(pixels: int) -> float:
    return pixels / DPI * 25.4


def font(points: float) -> ImageFont.FreeTypeFont:
    """A size in millimetres of cap height, near enough."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", mm(points))
    except OSError:
        return ImageFont.load_default(size=mm(points))


def build(variant: str, width_mm: float = SIZE_MM, height_mm: float = SIZE_MM) -> Raster:
    canvas_w, canvas_h = mm(width_mm), mm(height_mm)
    # Transparent: on this printer the alpha channel is what generates the white
    # underbase, so the tile has to carry its tone as alpha, not as pale ink.
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    small = font(1.4)
    tiny = font(1.1)
    left = mm(MARGIN_MM)
    width = canvas_w - 2 * left
    y = MARGIN_MM  # tracked in millimetres, so the budget below is readable

    def label(text: str, x: int, top_mm: float, chosen=None) -> None:
        draw.text((x, mm(top_mm)), text, font=chosen or small, fill=(*INK, 255))

    # -- title and scale bar -------------------------------------------------
    label(f"glassprint calibration · {width_mm:.0f}×{height_mm:.0f}mm · {DPI:.0f}dpi · {variant}", left, y)
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
    y += 9.2

    # -- 2. the same tones as a dot screen -----------------------------------
    label("2  the same tones as 0.8mm dots — compare with row 1", left, y)
    y += 2.0
    band = np.zeros((mm(6.0), width), dtype=np.float32)
    for index, percent in enumerate(STEPS):
        band[:, index * step_w : (index + 1) * step_w] = percent / 100.0
    paste_mask(canvas, halftone(band, mm(0.8), angle=45.0), left, mm(y))
    y += 7.2

    # -- 3. dot pitch, for moire against the printer's own screen -------------
    label("3  dot pitch at 50% — any interference patterns?", left, y)
    y += 2.0
    patch = mm(7.0)
    gap = (width - len(PITCHES) * patch) // max(1, len(PITCHES) - 1)
    for index, pitch in enumerate(PITCHES):
        x = left + index * (patch + gap)
        block = np.full((patch, patch), 0.5, dtype=np.float32)
        paste_mask(canvas, halftone(block, mm(pitch), angle=45.0), x, mm(y))
        label(f"{pitch}", x, y + px_mm(patch) + 0.2, tiny)
    y += px_mm(patch) + 3.1

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
    # Spread rather than huddled: on the wide strip a fixed gap would leave the
    # lines in one corner with a hand's width of empty glass beside them.
    spacing = min(mm(8.0), width // len(LINES))
    for index, thickness in enumerate(LINES):
        x = left + index * spacing
        w = max(1, mm(thickness))
        draw.rectangle([x, mm(y), x + w - 1, mm(y + 5.0)], fill=(*INK, 255))
        label(f"{thickness}", x - mm(0.4), y + 5.2, tiny)
    y += 8.4

    # -- 6. a continuous fade, which is the thing this is all for -------------
    label("6  smooth fade — banding? a hard stop? where does it vanish?", left, y)
    y += 2.0
    ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)[None, :].repeat(mm(6.0), axis=0)
    paste_mask(canvas, ramp, left, mm(y))
    label("100%", left, y + 6.2, tiny)
    label("0%", left + width - mm(3.4), y + 6.2, tiny)
    y += 8.4

    # -- 7. type sizes --------------------------------------------------------
    label("7  text — the smallest that stays readable", left, y)
    y += 2.2
    x = left
    for size in TEXT_SIZES:
        draw.text((x, mm(y)), f"{size} Handgloves", font=font(size), fill=(*INK, 255))
        x += mm(9.0 + size * 9.0)
    y += max(TEXT_SIZES) + 1.6

    # Laying this out by hand is exactly the sort of thing that silently runs
    # off the edge, and a calibration tile with a row missing is worse than none.
    if y > height_mm - MARGIN_MM:
        raise SystemExit(f"layout overflows: needs {y:.1f}mm of {height_mm - MARGIN_MM:.1f}mm")

    # A hairline cross in each corner: print this twice and these say how far
    # the second pass landed from the first.
    for cx, cy in [
        (left, mm(MARGIN_MM)),
        (canvas_w - left, mm(MARGIN_MM)),
        (left, canvas_h - mm(MARGIN_MM)),
        (canvas_w - left, canvas_h - mm(MARGIN_MM)),
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


#: The glaze primaries, plus a grey. Black is in there deliberately: it should
#: saturate almost immediately, and knowing that stacking it is pointless is
#: worth one row.
GLAZE_INKS = [("C", (0, 158, 224)), ("M", (226, 0, 122)), ("Y", (255, 237, 0)), ("50K", (128, 128, 128))]

#: Pairs laid one over the other. Whether these overlaps land where the model
#: predicts is the question the whole glaze solver hangs on.
GLAZE_PAIRS = [
    (("C", (0, 158, 224)), ("M", (226, 0, 122))),
    (("C", (0, 158, 224)), ("Y", (255, 237, 0))),
    (("M", (226, 0, 122)), ("Y", (255, 237, 0))),
    (("C", (0, 158, 224)), ("C", (0, 158, 224))),
    (("M", (226, 0, 122)), ("50K", (128, 128, 128))),
    (("Y", (255, 237, 0)), ("50K", (128, 128, 128))),
]


def glaze_test() -> list[Raster]:
    """Four files to print one over another, without moving the glass.

    Two questions, both unanswerable from a single pass. Does repeating an ink
    deepen it the way the model says, and does one colour over another multiply
    the way the model says? The second is the load-bearing one: every recipe
    the glaze solver produces assumes it.

    Drawn in colour rather than near-black on purpose. Black is all but opaque
    after one pass, so a depth series in black shows four blocks that look
    identical and teaches nothing.
    """
    width_mm, height_mm = 160.0, 85.0
    canvas_w, canvas_h = mm(width_mm), mm(height_mm)
    left = mm(MARGIN_MM)

    pages = [Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0)) for _ in range(4)]
    draws = [ImageDraw.Draw(page) for page in pages]
    # Labels go on pass 1 alone: printed four times over they would cure into
    # an illegible smudge.
    first = draws[0]
    tiny, small = font(1.1), font(1.4)

    first.text((left + mm(4.0), mm(5.0)), "glassprint glaze test · 160x85mm · print all four, in order, without moving the glass", font=small, fill=(20, 20, 24, 255))

    # -- 1. depth: the same ink, one to four passes --------------------------
    first.text((left, mm(9.0)), "1  same ink, 1-4 passes — where does it stop deepening?", font=small, fill=(20, 20, 24, 255))
    block_w, block_h, gap = mm(14.0), mm(8.0), mm(2.0)
    top = mm(12.5)
    for depth in range(1, 5):
        first.text((left + (depth - 1) * (block_w + gap) + mm(5.5), mm(11.0)), str(depth), font=tiny, fill=(20, 20, 24, 255))
    for row, (name, rgb) in enumerate(GLAZE_INKS):
        y = top + row * (block_h + gap)
        for depth in range(1, 5):
            x = left + (depth - 1) * (block_w + gap)
            # Block *depth* takes ink on every pass up to and including depth.
            for index in range(depth):
                draws[index].rectangle([x, y, x + block_w, y + block_h], fill=(*rgb, 255))
        first.text((left + 4 * (block_w + gap) + mm(1.0), y + mm(2.5)), name, font=tiny, fill=(20, 20, 24, 255))

    # -- 2. pairs: one colour over another -----------------------------------
    y = top + 4 * (block_h + gap) + mm(4.0)
    first.text((left, y), "2  one colour over another — is the overlap what the tool predicts?", font=small, fill=(20, 20, 24, 255))
    y += mm(3.0)
    cell = (canvas_w - 2 * left) // len(GLAZE_PAIRS)
    for index, ((name_a, rgb_a), (name_b, rgb_b)) in enumerate(GLAZE_PAIRS):
        x = left + index * cell
        # Offset halves, so each cell reads: A alone, both, B alone.
        draws[0].rectangle([x, y, x + int(cell * 0.62), y + mm(11.0)], fill=(*rgb_a, 255))
        draws[1].rectangle([x + int(cell * 0.38), y, x + cell - mm(1.5), y + mm(11.0)], fill=(*rgb_b, 255))
        first.text((x, y + mm(11.4)), f"{name_a} then {name_b}", font=tiny, fill=(20, 20, 24, 255))

    # Every pass carries the same hairlines, so four prints on one piece of
    # glass measure exactly how far the registration wandered.
    for draw in draws:
        arm, w = mm(2.5), max(1, mm(0.1))
        for cx, cy in [
            (left, mm(MARGIN_MM)),
            (canvas_w - left, mm(MARGIN_MM)),
            (left, canvas_h - mm(MARGIN_MM)),
            (canvas_w - left, canvas_h - mm(MARGIN_MM)),
        ]:
            draw.rectangle([cx - arm, cy - w // 2, cx + arm, cy + w // 2], fill=(20, 20, 24, 255))
            draw.rectangle([cx - w // 2, cy - arm, cx + w // 2, cy + arm], fill=(20, 20, 24, 255))

    # Staggered down the corner, so all four printed on one piece of glass read
    # as a tally of what actually went down rather than one black blur.
    for index in range(1, 5):
        draws[index - 1].text(
            (canvas_w - left - mm(12.0), canvas_h - mm(14.0) + mm(index * 2.4)),
            f"pass {index}",
            font=tiny,
            fill=(20, 20, 24, 255),
        )

    return [Raster(np.array(page, dtype=np.uint8), dpi=(DPI, DPI)) for page in pages]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Two files, identical but for the label. Printed and set aside, two pieces
    # of glass are otherwise impossible to tell apart a week later.
    for shape, (width_mm, height_mm) in SHAPES.items():
        for variant, name in [("WITH white base", "with-white"), ("NO white base", "no-white")]:
            tile = build(variant, width_mm, height_mm)
            out = OUT / f"glassprint-{shape}-{name}.png"
            tile.save(out, fmt="png", dpi=(DPI, DPI))
            print(f"{out.name}  {tile.width}×{tile.height}px  {width_mm:.0f}×{height_mm:.0f}mm")

    for index, raster in enumerate(glaze_test(), start=1):
        name = f"glassprint-glaze-pass{index}of4.png"
        raster.save(OUT / name, fmt="png", dpi=(DPI, DPI))
    print("glassprint-glaze-pass1..4of4.png  160×85mm each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
