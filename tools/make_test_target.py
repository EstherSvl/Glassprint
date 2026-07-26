"""Build calibration tiles to print on glass.

Everything glassprint decides about fades rests on numbers I picked by
reasoning rather than measurement: that dithering breaks up somewhere around
12% coverage, that a dot screen finer than about half a millimetre risks
beating against the printer's own screening, that alpha drives the white
underbase smoothly rather than as a threshold. These tiles put each of those on
glass so the guesses can be replaced with readings.

    python tools/make_test_target.py

The screened rows are rendered by ``glassprint.fade`` itself, not drawn to look
like it — so what reaches the printer is the real output of the code being
calibrated.

Shapes are cut to real offcuts rather than to round numbers. A tile that does
not fit the glass you own is a tile that never gets printed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

from glassprint.fade import halftone
from glassprint.raster import Raster, mm_to_px

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-target"

DPI = 600.0
MARGIN_MM = 4.0

INK = (20, 20, 24)  # near-black: dither speckle shows against it soonest

ALL_ROWS = ("steps", "screen", "pitch", "colour", "lines", "fade", "text")

#: (width, height, rows). Sized to glass that exists.
SHAPES: dict[str, tuple[float, float, tuple[str, ...]]] = {
    # The 115x85mm offcuts, with a couple of millimetres to spare each side.
    "plate": (110.0, 80.0, ALL_ROWS),
    # The 145x50mm offcuts. Only about 40mm of usable height, so this carries
    # the tonal rows alone — which are the ones that answer the fade question.
    "band": (140.0, 46.0, ("steps", "screen", "fade")),
}

#: Coverage steps. The interesting region is the bottom, so the top end thins
#: out first when there is less width to spend.
STEPS_WIDE = [100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6, 4, 2]
STEPS_NARROW = [100, 80, 60, 45, 35, 25, 20, 15, 12, 10, 8, 5, 2]

#: Dot pitches in millimetres, spanning "certainly too fine" to "certainly safe".
PITCHES = [0.25, 0.4, 0.6, 0.8, 1.2, 1.8]

#: Line widths in millimetres, for the resolution limit.
LINES = [0.08, 0.12, 0.2, 0.3, 0.5]

#: Cap heights in millimetres. The labels are already a legibility test at 1.1
#: and 1.4mm; this makes that deliberate and gives it a range.
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


def font(cap_mm: float) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", mm(cap_mm))
    except OSError:
        return ImageFont.load_default(size=mm(cap_mm))


def paste_mask(canvas: Image.Image, mask: np.ndarray, x: int, y: int) -> None:
    """Drop a 0..1 coverage mask onto the canvas as ink with matching alpha."""
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.array(INK, dtype=np.uint8)[None, None, :]
    rgba[:, :, 3] = np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(rgba, mode="RGBA"), (x, y))


def corner_marks(draw: ImageDraw.ImageDraw, canvas_w: int, canvas_h: int) -> None:
    """Hairline crosses. Print a file twice and these measure the drift."""
    left = mm(MARGIN_MM)
    arm, weight = mm(2.0), max(1, mm(0.1))
    for cx, cy in [
        (left, mm(MARGIN_MM)),
        (canvas_w - left, mm(MARGIN_MM)),
        (left, canvas_h - mm(MARGIN_MM)),
        (canvas_w - left, canvas_h - mm(MARGIN_MM)),
    ]:
        draw.rectangle([cx - arm, cy - weight // 2, cx + arm, cy + weight // 2], fill=(*INK, 255))
        draw.rectangle([cx - weight // 2, cy - arm, cx + weight // 2, cy + arm], fill=(*INK, 255))


def build(variant: str, width_mm: float, height_mm: float, rows: tuple[str, ...]) -> Raster:
    canvas_w, canvas_h = mm(width_mm), mm(height_mm)
    # Transparent: on this printer the alpha channel generates the white
    # underbase, so the tile carries its tone as alpha, not as pale ink.
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    small, tiny = font(1.4), font(1.1)
    left = mm(MARGIN_MM)
    width = canvas_w - 2 * left
    y = MARGIN_MM  # in millimetres, so the budget below reads plainly

    steps = STEPS_WIDE if width_mm >= 130 else STEPS_NARROW
    step_w = width // len(steps)

    def label(text: str, x: int, top_mm: float, chosen=None) -> None:
        draw.text((x, mm(top_mm)), text, font=chosen or small, fill=(*INK, 255))

    # -- title and scale bar -------------------------------------------------
    label(
        f"glassprint · {width_mm:.0f}x{height_mm:.0f}mm · {DPI:.0f}dpi · {variant}",
        left + mm(3.5),
        y,
    )
    y += 2.4
    draw.rectangle([left, mm(y), left + mm(10), mm(y) + mm(0.7)], fill=(*INK, 255))
    label("10mm — measure this", left + mm(11), y - 0.3, tiny)
    y += 2.6

    if "steps" in rows:
        label("1  flat tone by alpha — where does it go gritty?", left, y)
        y += 2.0
        for index, percent in enumerate(steps):
            x = left + index * step_w
            draw.rectangle(
                [x, mm(y), x + step_w - 1, mm(y + 5.0)], fill=(*INK, round(percent * 2.55))
            )
            label(str(percent), x + mm(0.2), y + 5.2, tiny)
        y += 8.2

    if "screen" in rows:
        label("2  the same tones as 0.8mm dots — compare with row 1", left, y)
        y += 2.0
        band = np.zeros((mm(5.0), width), dtype=np.float32)
        for index, percent in enumerate(steps):
            band[:, index * step_w : (index + 1) * step_w] = percent / 100.0
        paste_mask(canvas, halftone(band, mm(0.8), angle=45.0), left, mm(y))
        y += 6.2

    if "pitch" in rows:
        label("3  dot pitch at 50% — any interference patterns?", left, y)
        y += 2.0
        patch = mm(6.0)
        gap = (width - len(PITCHES) * patch) // max(1, len(PITCHES) - 1)
        for index, pitch in enumerate(PITCHES):
            x = left + index * (patch + gap)
            block = np.full((patch, patch), 0.5, dtype=np.float32)
            paste_mask(canvas, halftone(block, mm(pitch), angle=45.0), x, mm(y))
            label(f"{pitch}", x, y + px_mm(patch) + 0.2, tiny)
        y += px_mm(patch) + 2.6

    if "colour" in rows:
        label("4  solid colour — photograph against light and against dark", left, y)
        y += 2.0
        swatch = width // len(SWATCHES)
        for index, (name, rgb) in enumerate(SWATCHES):
            x = left + index * swatch
            draw.rectangle([x, mm(y), x + swatch - 1, mm(y + 5.0)], fill=(*rgb, 255))
            label(name, x + mm(0.2), y + 5.2, tiny)
        y += 8.4

    if "lines" in rows:
        label("5  line width in mm — the thinnest that survives", left, y)
        y += 2.2
        spacing = min(mm(8.0), width // len(LINES))
        for index, thickness in enumerate(LINES):
            x = left + index * spacing
            w = max(1, mm(thickness))
            draw.rectangle([x, mm(y), x + w - 1, mm(y + 4.0)], fill=(*INK, 255))
            label(f"{thickness}", x - mm(0.4), y + 4.2, tiny)
        y += 7.4

    if "fade" in rows:
        label("6  smooth fade — banding? a hard stop? where does it vanish?", left, y)
        y += 2.0
        ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)[None, :].repeat(mm(5.5), axis=0)
        paste_mask(canvas, ramp, left, mm(y))
        label("100%", left, y + 5.7, tiny)
        label("0%", left + width - mm(3.4), y + 5.7, tiny)
        y += 7.9

    if "text" in rows:
        label("7  text — the smallest that stays readable", left, y)
        y += 2.2
        x = left
        for size in TEXT_SIZES:
            draw.text((x, mm(y)), f"{size} Handgloves", font=font(size), fill=(*INK, 255))
            x += mm(6.0 + size * 8.0)
        y += max(TEXT_SIZES) + 1.6

    # Laying this out by hand is exactly the sort of thing that silently runs
    # off the edge, and a calibration tile with a row missing is worse than none.
    if y > height_mm - MARGIN_MM:
        raise SystemExit(
            f"{width_mm:.0f}x{height_mm:.0f} overflows: needs {y:.1f}mm of "
            f"{height_mm - MARGIN_MM:.1f}mm"
        )

    corner_marks(draw, canvas_w, canvas_h)
    return Raster(np.array(canvas, dtype=np.uint8), dpi=(DPI, DPI), name="glassprint-test")


# -- the black and mechanism test -------------------------------------------

#: One set of tones, used by all three grey rows so they line up vertically and
#: can be read against each other by eye.
TONES = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5]


def black_test(width_mm: float = 134.0, height_mm: float = 70.0) -> Raster:
    """Is the black cartridge healthy, and which mechanism actually makes tone?

    Two jobs. The obvious one is density: on a white substrate, solid black
    should be black, and a suspect cartridge shows up as grey, banding or
    dropped nozzles.

    The one that matters more was nearly missed. Three rows carry the same
    twelve tones by three different means — RGB value, alpha, and dot coverage —
    aligned so a glance down a column compares them directly. On the glass tile
    a 50% *alpha* patch fell off the cliff while an RGB 128 swatch at full alpha
    printed perfectly well. If that holds here, tone by RGB value works where
    tone by alpha does not, and the fix for fades is far simpler than dots.
    """
    canvas_w, canvas_h = mm(width_mm), mm(height_mm)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    small, tiny = font(1.4), font(1.1)
    left = mm(MARGIN_MM)
    width = canvas_w - 2 * left
    step = width // len(TONES)
    y = MARGIN_MM

    def label(text: str, x: int, top_mm: float, chosen=None) -> None:
        draw.text((x, mm(top_mm)), text, font=chosen or small, fill=(*INK, 255))

    label(f"glassprint black · {width_mm:.0f}x{height_mm:.0f}mm · print on WHITE, no white base", left + mm(3.5), y)
    y += 2.4
    draw.rectangle([left, mm(y), left + mm(10), mm(y) + mm(0.7)], fill=(*INK, 255))
    label("10mm", left + mm(11), y - 0.3, tiny)
    y += 2.8

    # -- 1. is black black? --------------------------------------------------
    label("1  solid black — compare against something you know is black", left, y)
    y += 2.0
    draw.rectangle([left, mm(y), left + width // 2 - mm(1), mm(y + 7.0)], fill=(0, 0, 0, 255))
    # Beside it, the near-black used on the glass tile, so the two are
    # comparable and the earlier "dark grey" reading can be placed.
    draw.rectangle([left + width // 2 + mm(1), mm(y), left + width, mm(y + 7.0)], fill=(*INK, 255))
    label("RGB 0,0,0", left + mm(0.4), y + 7.2, tiny)
    label("RGB 20,20,24 (the glass tile's ink)", left + width // 2 + mm(1.4), y + 7.2, tiny)
    y += 10.4

    # -- 2, 3, 4. the same tones, three mechanisms, aligned ------------------
    for title, kind in [
        ("2  tone by RGB value, alpha 100% — does this one work?", "rgb"),
        ("3  tone by alpha, RGB near-black — the control, expect a cliff", "alpha"),
        ("4  tone by dot coverage at 0.8mm — known to work", "dots"),
    ]:
        label(title, left, y)
        y += 2.0
        if kind == "dots":
            band = np.zeros((mm(6.0), width), dtype=np.float32)
            for index, percent in enumerate(TONES):
                band[:, index * step : (index + 1) * step] = percent / 100.0
            paste_mask(canvas, halftone(band, mm(0.8), angle=45.0), left, mm(y))
        else:
            for index, percent in enumerate(TONES):
                x = left + index * step
                if kind == "rgb":
                    # Less ink asked for as a lighter grey, at full alpha.
                    level = round(255 * (1.0 - percent / 100.0))
                    fill = (level, level, level, 255)
                else:
                    fill = (*INK, round(percent * 2.55))
                draw.rectangle([x, mm(y), x + step - 1, mm(y + 6.0)], fill=fill)
        for index, percent in enumerate(TONES):
            label(str(percent), left + index * step + mm(0.2), y + 6.2, tiny)
        y += 9.4

    # -- 5. nozzles ----------------------------------------------------------
    label("5  fine lines — any gaps mean a blocked nozzle", left, y)
    y += 2.2
    for row in range(9):
        top = mm(y) + row * mm(0.6)
        draw.rectangle([left, top, left + width, top + max(1, mm(0.12))], fill=(0, 0, 0, 255))
    y += 6.4

    if y > height_mm - MARGIN_MM:
        raise SystemExit(f"black test overflows: needs {y:.1f}mm of {height_mm - MARGIN_MM:.1f}mm")

    corner_marks(draw, canvas_w, canvas_h)
    return Raster(np.array(canvas, dtype=np.uint8), dpi=(DPI, DPI), name="glassprint-black")


# -- the glaze test ---------------------------------------------------------

GLAZE_INKS = [
    ("C", (0, 158, 224)),
    ("M", (226, 0, 122)),
    ("Y", (255, 237, 0)),
    ("50K", (128, 128, 128)),
]

#: Pairs laid one over the other. Whether these overlaps land where the model
#: predicts is the question the whole glaze solver hangs on.
GLAZE_PAIRS = [
    (("C", (0, 158, 224)), ("M", (226, 0, 122))),
    (("C", (0, 158, 224)), ("Y", (255, 237, 0))),
    (("M", (226, 0, 122)), ("Y", (255, 237, 0))),
    (("M", (226, 0, 122)), ("50K", (128, 128, 128))),
]


def glaze_test(width_mm: float = 110.0, height_mm: float = 80.0) -> list[Raster]:
    """Four files to print one over another, without moving the glass.

    Two questions, neither answerable from a single pass. Does repeating an ink
    deepen it the way the model says, and does one colour over another multiply
    the way the model says? The second is load-bearing: every recipe the glaze
    solver emits assumes it.

    Drawn in colour rather than near-black on purpose. Black is all but opaque
    after one pass, so a depth series in black shows four identical blocks.
    """
    canvas_w, canvas_h = mm(width_mm), mm(height_mm)
    left = mm(MARGIN_MM)
    width = canvas_w - 2 * left

    pages = [Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0)) for _ in range(4)]
    draws = [ImageDraw.Draw(page) for page in pages]
    first = draws[0]  # labels go on pass 1 alone, or they cure into a smudge
    small, tiny = font(1.3), font(1.0)

    first.text(
        (left + mm(3.5), mm(4.0)),
        f"glassprint glaze · {width_mm:.0f}x{height_mm:.0f}mm · all four, in order, glass unmoved",
        font=small,
        fill=(*INK, 255),
    )

    # -- 1. depth: the same ink, one to four passes --------------------------
    first.text(
        (left, mm(8.0)),
        "1  same ink, 1-4 passes — where does it stop deepening?",
        font=small,
        fill=(*INK, 255),
    )
    block_w, block_h, gap = mm(15.0), mm(8.0), mm(2.0)
    top = mm(11.5)
    for depth in range(1, 5):
        first.text(
            (left + (depth - 1) * (block_w + gap) + mm(6.5), mm(10.2)),
            str(depth),
            font=tiny,
            fill=(*INK, 255),
        )
    for row, (name, rgb) in enumerate(GLAZE_INKS):
        y = top + row * (block_h + gap)
        for depth in range(1, 5):
            x = left + (depth - 1) * (block_w + gap)
            # Block *depth* takes ink on every pass up to and including depth.
            for index in range(depth):
                draws[index].rectangle([x, y, x + block_w, y + block_h], fill=(*rgb, 255))
        first.text(
            (left + 4 * (block_w + gap) + mm(1.0), y + mm(2.5)), name, font=tiny, fill=(*INK, 255)
        )

    # -- 2. pairs: one colour over another -----------------------------------
    y = top + 4 * (block_h + gap) + mm(3.0)
    first.text(
        (left, y), "2  one colour over another — is the overlap predicted?", font=small, fill=(*INK, 255)
    )
    y += mm(3.0)
    cell = width // len(GLAZE_PAIRS)
    for index, ((name_a, rgb_a), (name_b, rgb_b)) in enumerate(GLAZE_PAIRS):
        x = left + index * cell
        # Offset halves, so each cell reads: A alone, both, B alone.
        draws[0].rectangle([x, y, x + int(cell * 0.62), y + mm(10.0)], fill=(*rgb_a, 255))
        draws[1].rectangle(
            [x + int(cell * 0.38), y, x + cell - mm(1.5), y + mm(10.0)], fill=(*rgb_b, 255)
        )
        first.text((x, y + mm(10.4)), f"{name_a} then {name_b}", font=tiny, fill=(*INK, 255))

    for draw in draws:
        corner_marks(draw, canvas_w, canvas_h)

    # Staggered down the corner, so four prints on one piece of glass read as a
    # tally of what actually went down rather than one blur.
    for index in range(1, 5):
        draws[index - 1].text(
            (canvas_w - left - mm(10.0), canvas_h - mm(12.0) + mm(index * 2.2)),
            f"pass {index}",
            font=tiny,
            fill=(*INK, 255),
        )

    return [Raster(np.array(page, dtype=np.uint8), dpi=(DPI, DPI)) for page in pages]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for shape, (width_mm, height_mm, rows) in SHAPES.items():
        # Two files, identical but for the label. Printed and set aside, two
        # pieces of glass are otherwise impossible to tell apart a week later.
        for variant, name in [("WITH white base", "with-white"), ("NO white base", "no-white")]:
            tile = build(variant, width_mm, height_mm, rows)
            out = OUT / f"glassprint-{shape}-{name}.png"
            tile.save(out, fmt="png", dpi=(DPI, DPI))
            print(f"{out.name:38} {tile.width}x{tile.height}px  {width_mm:.0f}x{height_mm:.0f}mm")

    black = black_test()
    out = OUT / "glassprint-black.png"
    black.save(out, fmt="png", dpi=(DPI, DPI))
    print(f"{out.name:38} {black.width}x{black.height}px  134x70mm")

    for index, raster in enumerate(glaze_test(), start=1):
        out = OUT / f"glassprint-glaze-pass{index}of4.png"
        raster.save(out, fmt="png", dpi=(DPI, DPI))
    print(f"{'glassprint-glaze-pass1..4of4.png':38} 110x80mm each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
