from __future__ import annotations

import numpy as np
import pytest

from glassprint import colors, masks, nl, pattern, recolor, segment
from glassprint.raster import Raster, mm_to_px, px_to_mm


# --- raster -----------------------------------------------------------------

def test_round_trip_preserves_dpi_and_alpha(tmp_path, base_shape):
    path = base_shape.save(tmp_path / "panel.png")
    reloaded = Raster.open(path)
    assert reloaded.size == base_shape.size
    assert reloaded.dpi == (300.0, 300.0)
    assert reloaded.has_alpha


def test_jpeg_export_flattens_alpha(tmp_path, base_shape):
    path = base_shape.save(tmp_path / "panel.jpg", keep_alpha=False, background=(255, 255, 255))
    reloaded = Raster.open(path)
    assert not reloaded.has_alpha
    assert reloaded.dpi == (300.0, 300.0)


def test_physical_size_maths():
    assert mm_to_px(25.4, 300) == 300
    assert px_to_mm(300, 300) == pytest.approx(25.4)


def test_with_physical_width_resamples(base_shape):
    out = base_shape.with_physical_width(100.0, 300.0)
    assert out.width == mm_to_px(100.0, 300.0)
    assert out.size_mm[0] == pytest.approx(100.0, abs=0.2)


# --- colours ----------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("#ff0000", (255, 0, 0)),
        ("#f00", (255, 0, 0)),
        ("red", (220, 38, 38)),
        ("10,20,30", (10, 20, 30)),
    ],
)
def test_parse_color(value, expected):
    assert colors.parse_color(value) == expected


def test_parse_color_rejects_nonsense():
    with pytest.raises(ValueError):
        colors.parse_color("banana")


def test_hsv_round_trip():
    rgb = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    back = colors.hsv_to_rgb(colors.rgb_to_hsv(rgb))
    assert np.abs(rgb - back).max() < 1e-4


def test_color_mask_selects_the_named_hue(pattern_art):
    red = colors.color_mask(pattern_art.rgb_f, "red")
    green = colors.color_mask(pattern_art.rgb_f, "green")
    white = colors.color_mask(pattern_art.rgb_f, "white")

    assert red[20, 20] > 0.9        # centre of a red dot
    assert green[20, 20] < 0.1
    assert white[0, 0] > 0.9        # the background corner
    assert red[0, 0] < 0.1


# --- natural language -------------------------------------------------------

def test_empty_instruction_removes_background():
    plan = nl.parse("")
    assert plan.source == "default"
    assert [(op.action, op.selector.kind) for op in plan.ops] == [("remove", "background")]


def test_keep_and_remove_are_split():
    plan = nl.parse("keep the flowers, remove the white background")
    actions = [(op.action, op.selector.kind) for op in plan.ops]
    assert ("keep", "semantic") in actions
    assert ("remove", "background") in actions


def test_colour_only_phrase_becomes_a_colour_selector():
    plan = nl.parse("keep the gold parts")
    assert [(op.action, op.selector.kind, op.selector.value) for op in plan.ops] == [
        ("keep", "color", "gold")
    ]


def test_colour_qualified_noun_keeps_a_colour_hint():
    plan = nl.parse("keep the red roses")
    op = plan.ops[0]
    assert op.selector.kind == "semantic"
    assert op.selector.value == "red roses"
    assert op.selector.color_hint == "red"


def test_bare_noun_phrase_is_treated_as_keep():
    plan = nl.parse("the leaves")
    assert plan.ops[0].action == "keep"


def test_linework_maps_to_dark_tones():
    plan = nl.parse("keep only the linework")
    assert (plan.ops[0].selector.kind, plan.ops[0].selector.value) == ("tone", "dark")


def test_everything_else_is_not_a_literal_removal():
    plan = nl.parse("keep the roses and remove everything else")
    assert [op.action for op in plan.ops] == ["keep"]


def test_without_reads_as_removal():
    plan = nl.parse("the pattern without the background")
    actions = {(op.action, op.selector.kind) for op in plan.ops}
    assert ("remove", "background") in actions


# --- segmentation -----------------------------------------------------------

def test_background_detection_finds_the_white_ground(pattern_art):
    background = segment.background_mask(pattern_art)
    assert background[0, 0] > 0.8       # corner is background
    assert background[20, 20] < 0.2     # a dot is not


def test_background_detection_respects_existing_alpha(base_shape):
    background = segment.background_mask(base_shape)
    assert background[0, 0] > 0.9
    assert background[150, 200] < 0.1


def test_evaluate_keeps_only_the_named_colour(pattern_art):
    plan = nl.parse("keep the red")
    mask = segment.evaluate(plan, pattern_art)
    assert mask[20, 20] > 0.8      # red dot
    assert mask[0, 0] < 0.1        # white ground
    assert mask[31, 31] < 0.3      # green leaf


def test_evaluate_default_removes_background(pattern_art):
    mask = segment.evaluate(nl.parse(""), pattern_art)
    assert mask[0, 0] < 0.2
    assert mask[20, 20] > 0.8


def test_missing_model_is_reported_not_raised(motif_art):
    backends = segment.Backends()
    plan = nl.parse("keep the swirly ornaments")
    mask = segment.evaluate(plan, motif_art, backends)
    assert mask.shape == (motif_art.height, motif_art.width)
    assert backends.notes, "a fallback should explain itself"


# --- masks ------------------------------------------------------------------

def test_despeckle_drops_small_blobs():
    mask = masks.zeros((100, 100))
    mask[10:60, 10:60] = 1.0   # big
    mask[90, 90] = 1.0         # speck
    cleaned = masks.despeckle(mask, min_area_fraction=0.01)
    assert cleaned[30, 30] == 1.0
    assert cleaned[90, 90] == 0.0


def test_bbox_and_grow():
    mask = masks.zeros((50, 50))
    mask[10:20, 5:15] = 1.0
    assert masks.bbox(mask) == (5, 10, 15, 20)
    assert masks.coverage(masks.grow(mask, 3)) > masks.coverage(mask)
    assert masks.coverage(masks.grow(mask, -3)) < masks.coverage(mask)


def test_touching_border_keeps_only_edge_blobs():
    mask = masks.zeros((60, 60))
    mask[0:10, 0:10] = 1.0    # touches the corner
    mask[30:40, 30:40] = 1.0  # floats in the middle
    kept = masks.touching_border(mask)
    assert kept[5, 5] == 1.0
    assert kept[35, 35] == 0.0


# --- pattern analysis and placement ----------------------------------------

def test_repeating_art_is_recognised_as_a_pattern(pattern_art):
    cutout = segment.evaluate(nl.parse(""), pattern_art)
    info = pattern.analyse(pattern_art, cutout)
    assert info.is_pattern
    assert info.suggested_fit == "tile"


def test_single_motif_is_not_a_pattern(motif_art):
    cutout = segment.evaluate(nl.parse(""), motif_art)
    info = pattern.analyse(motif_art, cutout)
    assert not info.is_pattern
    assert info.suggested_fit == "shape"


def test_place_tiles_across_the_box(pattern_art):
    cutout = segment.evaluate(nl.parse(""), pattern_art)
    art = pattern.apply_cutout(pattern_art, cutout)
    placed = pattern.place(
        art,
        (400, 300),
        (50, 50, 350, 250),
        pattern.Placement(fit="tile", repeat_across=4),
        None,
    )
    assert placed.shape == (300, 400, 4)
    assert placed[:, :, 3][:50, :].max() == 0        # nothing above the box
    assert placed[50:250, 50:350, 3].max() > 200     # something inside it


def test_place_contain_keeps_the_motif_inside(motif_art):
    cutout = segment.evaluate(nl.parse(""), motif_art)
    art = pattern.apply_cutout(motif_art, cutout)
    placed = pattern.place(
        art, (400, 300), (100, 50, 300, 250), pattern.Placement(fit="contain"), None
    )
    alpha = placed[:, :, 3]
    box = masks.bbox(alpha.astype(np.float32) / 255.0)
    assert box is not None
    left, top, right, bottom = box
    assert left >= 99 and top >= 49 and right <= 301 and bottom <= 251


def test_repeat_mm_controls_physical_tile_size(pattern_art):
    cutout = segment.evaluate(nl.parse(""), pattern_art)
    art = pattern.apply_cutout(pattern_art, cutout)
    placement = pattern.Placement(fit="tile", repeat_mm=25.4, mirror="off")
    placed = pattern.place(art, (600, 600), (0, 0, 600, 600), placement, None, target_dpi=300.0)
    # One inch at 300dpi is 300px, so the pattern should repeat twice across 600px.
    assert placed.shape == (600, 600, 4)
    top_row = placed[10, :, :3].astype(int)
    assert np.abs(top_row[:300] - top_row[300:600]).mean() < 2


# --- recolour ---------------------------------------------------------------

def test_tint_changes_hue_and_keeps_alpha(pattern_art):
    cutout = segment.evaluate(nl.parse(""), pattern_art)
    art = pattern.apply_cutout(pattern_art, cutout)
    tinted = recolor.apply(art, recolor.ColorSpec(mode="tint", color="#0044ff"))

    assert np.array_equal(tinted[:, :, 3], art[:, :, 3])
    dot = tinted[20, 20, :3]
    assert dot[2] > dot[0]  # was red, now leans blue


def test_replace_swaps_one_colour_only(pattern_art):
    art = pattern_art.rgba.copy()
    out = recolor.apply(
        art, recolor.ColorSpec(mode="replace", from_color="#dc2626", color="#0000ff", tolerance=1.0)
    )
    assert out[20, 20, 2] > out[20, 20, 0]      # the red dot moved to blue
    assert tuple(out[31, 31, :3]) == pytest.approx(tuple(art[31, 31, :3]), abs=30)  # leaf untouched


def test_identity_spec_is_a_no_op(pattern_art):
    art = pattern_art.rgba
    assert recolor.apply(art, recolor.ColorSpec()) is art


# -- pure black is a different colour to this printer ------------------------


def test_the_black_point_snaps_near_blacks_to_exactly_zero():
    """RGB(0,0,0) prints dense; RGB(20,20,24) prints thin and blue-grey.

    Artwork almost never contains exact zeros — a scan, a brightness tweak or a
    JPEG round-trip all leave the darkest pixels just above it, on the weak side
    of the discontinuity.
    """
    art = np.zeros((1, 4, 4), dtype=np.uint8)
    art[0, :, 3] = 255
    art[0, 0, :3] = (20, 20, 24)   # the near-black that printed weakly
    art[0, 1, :3] = (60, 60, 60)   # a genuine dark grey, to be left alone
    art[0, 2, :3] = (0, 0, 0)      # already pure
    art[0, 3, :3] = (200, 40, 40)  # a colour, nowhere near the black point

    out = recolor.apply(art, recolor.ColorSpec(black_point=0.12))
    assert tuple(out[0, 0, :3]) == (0, 0, 0)
    assert tuple(out[0, 1, :3]) == (60, 60, 60)
    assert tuple(out[0, 2, :3]) == (0, 0, 0)
    assert tuple(out[0, 3, :3]) == (200, 40, 40)
    # Alpha is never touched by a colour treatment.
    assert list(out[0, :, 3]) == [255] * 4


def test_the_black_point_is_off_by_default():
    """Snapping darks is a decision, not a default — it costs shadow detail."""
    art = np.zeros((1, 1, 4), dtype=np.uint8)
    art[0, 0] = (20, 20, 24, 255)
    assert recolor.ColorSpec().is_identity
    assert tuple(recolor.apply(art, recolor.ColorSpec())[0, 0, :3]) == (20, 20, 24)


def test_the_black_point_survives_a_later_brightness_change():
    """Snapping happens last, or a brightness tweak would lift it back off zero."""
    art = np.zeros((1, 1, 4), dtype=np.uint8)
    art[0, 0] = (20, 20, 24, 255)
    out = recolor.apply(art, recolor.ColorSpec(black_point=0.12, brightness=1.4))
    assert tuple(out[0, 0, :3]) == (0, 0, 0)


# -- optional models that are not installed ---------------------------------


def test_a_missing_model_reads_as_a_choice_not_a_crash():
    """The fallback is the documented path, not a fault.

    The first wording was "CLIPSeg unavailable (ModuleNotFoundError); falling
    back to..." rendered in the same colour as real warnings. Nothing had gone
    wrong — the model is optional and the colour path is what the tool is built
    to do without it — but it read as a broken tool.
    """
    from glassprint import segment

    backends = segment.Backends()
    backends.fallback("Matched what you described", "colour and tone", "smart")
    note = backends.notes[0]
    for jargon in ("unavailable", "Error", "fall", "failed"):
        assert jargon not in note, f"{jargon!r} in {note!r}"
    assert note.startswith("Matched what you described")


def test_a_tablet_is_not_told_to_pip_install():
    """A browser tab cannot install torch, and never will be able to."""
    from glassprint import segment

    was = segment.IN_BROWSER
    try:
        segment.IN_BROWSER = True
        backends = segment.Backends()
        backends.fallback("Cut the subject out", "colour, not by shape", "smart")
        assert "pip" not in backends.notes[0]
        assert "desktop" in backends.notes[0]

        segment.IN_BROWSER = False
        desktop = segment.Backends()
        desktop.fallback("Cut the subject out", "colour, not by shape", "smart")
        assert "pip install" in desktop.notes[0]
    finally:
        segment.IN_BROWSER = was
