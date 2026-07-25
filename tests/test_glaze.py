from __future__ import annotations

import numpy as np
import pytest

from glassprint import ComposeSpec, ExportSpec, Fade, compose, export
from glassprint.compose import GlazeSpec
from glassprint.glaze import (
    PROCESS_INKS,
    Ink,
    compare_single_ink,
    palette_from,
    plan,
    render,
    solve,
)

GREEN_GLASS = (125, 155, 143)
CMY = PROCESS_INKS[:3]


# --- the palette ------------------------------------------------------------

def test_palette_defaults_to_process_inks():
    assert [ink.name for ink in palette_from("")] == ["cyan", "magenta", "yellow"]


def test_palette_accepts_names_and_hex():
    inks = palette_from("cyan, #7a2f8a, black")
    assert [ink.name for ink in inks] == ["cyan", "#7a2f8a", "black"]
    assert inks[1].rgb == (122, 47, 138)


def test_palette_rejects_nonsense():
    with pytest.raises(ValueError):
        palette_from("cyan,banana")


# --- solving ----------------------------------------------------------------

def test_a_reachable_colour_is_matched_closely():
    recipe = solve((30, 55, 130), GREEN_GLASS, CMY)
    assert recipe.reachable
    assert recipe.error < 0.1
    assert recipe.passes


def test_glazing_beats_repeating_one_ink_on_deep_colours():
    """The whole point: different spectra multiplied reach further than one squared."""
    for target in [(30, 55, 130), (30, 80, 45), (70, 30, 75)]:
        glazed = solve(target, GREEN_GLASS, CMY)
        single = compare_single_ink(target, GREEN_GLASS)
        assert glazed.error < single.error, f"glaze should win on {target}"


def test_a_colour_brighter_than_the_glass_is_flagged():
    recipe = solve((178, 45, 40), GREEN_GLASS, CMY)   # red brighter than the glass
    assert not recipe.reachable
    assert "red" in recipe.note and "white underbase" in recipe.note
    assert recipe.dimming < 1.0


def test_an_unreachable_colour_still_gets_the_right_hue():
    """Aiming at the dimmed target beats returning 'print nothing'."""
    recipe = solve((178, 45, 40), GREEN_GLASS, CMY)
    achieved = np.array(recipe.achieved, dtype=float)
    assert recipe.passes, "should still recommend ink"
    assert achieved[0] > achieved[1] and achieved[0] > achieved[2]  # still reads red


def test_white_on_white_glass_needs_no_ink():
    recipe = solve((255, 255, 255), (255, 255, 255), CMY)
    assert recipe.passes == []
    assert recipe.describe() == "bare glass"


def test_the_metric_does_not_mistake_teal_for_green():
    """Regression: luminance weights under-penalise blue and picked teal."""
    recipe = solve((51, 119, 68), GREEN_GLASS, CMY)
    achieved = np.array(recipe.achieved, dtype=float)
    assert achieved[1] > achieved[2] + 30, f"green should beat blue, got {recipe.achieved}"


def test_pass_limits_are_respected():
    recipe = solve((5, 5, 5), GREEN_GLASS, CMY, max_per_ink=2, max_total=3)
    assert recipe.total_passes <= 3
    assert all(p.passes <= 2 for p in recipe.passes)


def test_more_inks_can_only_help():
    target = (120, 60, 90)
    narrow = solve(target, GREEN_GLASS, (Ink("cyan", PROCESS_INKS[0].rgb),))
    wide = solve(target, GREEN_GLASS, CMY)
    assert wide.error <= narrow.error + 1e-6


# --- planning a whole artwork ----------------------------------------------

def _artwork():
    rgb = np.zeros((40, 40, 3), dtype=np.float32)
    rgb[:20, :, :] = np.array([0.12, 0.31, 0.18])   # forest green
    rgb[20:, :, :] = np.array([0.12, 0.22, 0.51])   # deep blue
    return rgb, np.ones((40, 40), dtype=np.float32)


def test_plan_solves_each_colour_and_partitions_the_artwork():
    rgb, coverage = _artwork()
    result = plan(rgb, coverage, GREEN_GLASS, CMY, colours=2)

    assert len(result.recipes) == len(result.colours) == 2
    assert set(np.unique(result.labels).tolist()) == {0, 1}
    # The two halves get different labels, so different recipes.
    assert result.labels[5, 5] != result.labels[35, 5]


def test_the_printing_plan_lists_every_pass_once():
    rgb, coverage = _artwork()
    result = plan(rgb, coverage, GREEN_GLASS, CMY, colours=2)

    deepest: dict[str, int] = {}
    for recipe in result.recipes:
        for item in recipe.passes:
            deepest[item.ink.name] = max(deepest.get(item.ink.name, 0), item.passes)

    assert len(result.stack) == sum(deepest.values())
    for ink, index in result.stack:
        assert 1 <= index <= deepest[ink.name]


def test_counts_map_matches_the_recipes():
    rgb, coverage = _artwork()
    result = plan(rgb, coverage, GREEN_GLASS, CMY, colours=2)
    for ink in result.palette:
        counts = result.counts_for(ink)
        for index, recipe in enumerate(result.recipes):
            wanted = sum(p.passes for p in recipe.passes if p.ink.name == ink.name)
            assert counts[result.labels == index].max(initial=0) == wanted


def test_render_shows_achieved_colours_over_glass():
    rgb, coverage = _artwork()
    result = plan(rgb, coverage, GREEN_GLASS, CMY, colours=2)
    image = render(result, coverage * 0, GREEN_GLASS)   # nothing covered
    assert np.allclose(image[0, 0], np.array(GREEN_GLASS) / 255.0, atol=0.01)


# --- through the pipeline ---------------------------------------------------

def test_glaze_reaches_the_summary(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
        ),
    )
    assert result.glaze_plan is not None
    glaze_summary = result.summary()["glaze"]
    assert glaze_summary["glass"] == "#7d9b8f"
    # Asking for more colours than the artwork has gives back what is there.
    assert 1 <= len(glaze_summary["recipes"]) <= 3
    assert len(glaze_summary["recipes"]) == len(result.glaze_plan.colours)
    assert glaze_summary["total_passes"] >= 1


def test_glaze_is_off_by_default(base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    assert result.glaze_plan is None
    assert result.summary()["glaze"] is None


def test_unreachable_colours_are_surfaced_as_notes(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#3d5148", colours=3),  # dark glass
        ),
    )
    assert any(note.startswith("Glaze —") for note in result.notes)


# --- exporting the passes ---------------------------------------------------

def test_export_writes_one_file_per_glaze_pass(tmp_path, base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
        ),
    )
    manifest = export(
        result, tmp_path, ExportSpec(formats=["png"], targets=["glaze-layers"], basename="panel")
    )
    assert manifest
    assert all(entry["file"].startswith("panel_pass") for entry in manifest)
    assert all(entry["alpha"] for entry in manifest)

    stack = result.glaze_plan.stack
    assert len(manifest) <= len(stack)


def test_glaze_passes_are_solid_ink(tmp_path, base_shape, pattern_art):
    from glassprint import Raster

    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
        ),
    )
    manifest = export(result, tmp_path, ExportSpec(formats=["png"], targets=["glaze-layers"]))
    alpha = Raster.open(manifest[0]["path"]).alpha_f
    assert alpha[alpha > 0.5].mean() > 0.98


def test_a_fade_drops_glaze_passes_toward_the_edge(tmp_path, base_shape, pattern_art):
    from glassprint import Raster

    spec = ComposeSpec(
        keep="remove the white background",
        glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
        fade=Fade(mode="linear"),
    )
    result = compose(base_shape, pattern_art, spec)
    manifest = export(result, tmp_path, ExportSpec(formats=["png"], targets=["glaze-layers"]))

    alpha = Raster.open(manifest[0]["path"]).alpha_f
    top = alpha[60:120, 100:300].sum()
    bottom = alpha[200:255, 100:300].sum()
    assert top > bottom, "the fade should take passes off toward the transparent edge"


def test_glaze_export_is_skipped_when_glazing_is_off(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    manifest = export(result, tmp_path, ExportSpec(formats=["png"], targets=["glaze-layers"]))
    assert manifest == []


# --- printing the passes by hand -------------------------------------------

def test_passes_are_numbered_so_sorting_gives_the_print_order(tmp_path, base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=4),
        ),
    )
    manifest = export(
        result, tmp_path, ExportSpec(formats=["png"], targets=["glaze-layers"], basename="panel")
    )
    names = sorted(entry["file"] for entry in manifest)
    numbers = [int(name.split("pass")[1][:2]) for name in names]
    assert numbers == sorted(numbers) == list(range(1, len(numbers) + 1))


def test_print_order_sheet_lists_the_passes_and_the_white_warning(tmp_path, base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
        ),
    )
    manifest = export(
        result, tmp_path,
        ExportSpec(formats=["png"], targets=["glaze-layers", "print-order"], basename="panel"),
    )
    sheet = next(e for e in manifest if e["target"] == "print-order")
    text = (tmp_path / sheet["file"]).read_text()

    assert "white underbase off" in text
    assert "#7d9b8f" in text
    for entry in manifest:
        if entry["target"] == "glaze-layers":
            assert entry["file"] in text


def test_print_order_sheet_is_skipped_when_there_are_no_passes(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    manifest = export(
        result, tmp_path, ExportSpec(formats=["png"], targets=["print-order"])
    )
    assert manifest == []


def test_a_smooth_fade_over_a_glaze_is_flagged(base_shape, pattern_art):
    """Dropping passes unwinds the correction — the tool should say so."""
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
            fade=Fade(mode="linear"),
        ),
    )
    assert any("Fade by coverage instead" in note for note in result.notes)


def test_a_screened_fade_over_a_glaze_is_not_flagged(base_shape, pattern_art):
    """A dot screen keeps the whole stack in every dot, so the colour holds."""
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
            fade=Fade(mode="linear", halftone_mm=2.0),
        ),
    )
    assert not any("Fade by coverage instead" in note for note in result.notes)


def test_a_screened_fade_keeps_every_dot_at_full_stack(base_shape, pattern_art):
    """The reason to prefer coverage fading: the recipe survives intact."""
    from glassprint import Raster

    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
            fade=Fade(mode="linear", halftone_mm=2.0),
        ),
    )
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmp:
        manifest = export(
            result, pathlib.Path(tmp), ExportSpec(formats=["png"], targets=["glaze-layers"])
        )
        # Every pass still prints somewhere, top and bottom of the ramp alike:
        # the dots thin out, the stack behind each dot does not.
        assert len(manifest) == len(result.glaze_plan.stack)
        for entry in manifest:
            alpha = Raster.open(entry["path"]).alpha_f
            assert alpha[alpha > 0.5].mean() > 0.95
