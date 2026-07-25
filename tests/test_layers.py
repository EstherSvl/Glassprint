from __future__ import annotations

import numpy as np
import pytest

from glassprint import ComposeSpec, ExportSpec, Fade, compose, export
from glassprint.fade import quantise
from glassprint.simulate import glaze, stack_preview


# --- quantising the ramp ----------------------------------------------------

def test_quantise_produces_discrete_steps():
    ramp = np.linspace(0.0, 1.0, 500, dtype=np.float32)
    stepped, counts = quantise(ramp, 4)

    assert sorted(np.unique(counts).tolist()) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert sorted(np.unique(np.round(stepped, 3)).tolist()) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_quantise_keeps_the_ends_intact():
    stepped, counts = quantise(np.array([0.0, 1.0], dtype=np.float32), 5)
    assert stepped.tolist() == [0.0, 1.0]
    assert counts.tolist() == [0.0, 5.0]


def test_more_layers_means_finer_steps():
    ramp = np.linspace(0.0, 1.0, 500, dtype=np.float32)
    assert len(np.unique(quantise(ramp, 2)[0])) == 3
    assert len(np.unique(quantise(ramp, 8)[0])) == 9


def test_quantise_is_monotonic():
    ramp = np.linspace(1.0, 0.0, 200, dtype=np.float32)
    stepped, _ = quantise(ramp, 5)
    assert np.all(np.diff(stepped) <= 1e-6)


# --- through the pipeline ---------------------------------------------------

def test_stacked_fade_bands_the_gradient(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(keep="remove the white background", fade=Fade(mode="linear", layers=4)),
    )
    assert result.layer_map is not None
    assert sorted(np.unique(result.layer_map).tolist()) == [0.0, 1.0, 2.0, 3.0, 4.0]

    # Wherever the artwork is solid, the alpha sits exactly on one of the five
    # plateaus. (Its own anti-aliased edges are still smooth, as they should
    # be — the stepping is the fade, not the artwork.)
    alpha = result.overlay_layer.alpha_f
    solid = result.coverage > 0.99
    assert np.abs(alpha[solid] - np.round(alpha[solid] * 4) / 4).max() < 0.02
    assert len(np.unique(np.round(alpha[solid], 2))) >= 4


def test_layers_take_precedence_over_the_other_expressions(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(fade=Fade(mode="linear", layers=3, halftone_mm=2.0, dissolve=1.0)),
    )
    assert any("layers were used" in note for note in result.notes)
    assert result.fade_elements == 0


def test_coverage_is_what_one_pass_lays_down(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(keep="remove the white background", fade=Fade(mode="linear", layers=4)),
    )
    # Coverage ignores the stepping, so it is solid wherever any layer prints.
    assert result.coverage is not None
    printing = result.layer_map > 0
    assert result.coverage[printing].max() > 0.99
    assert result.coverage[result.layer_map == 0].max() == pytest.approx(0.0)


def test_a_stacked_fade_never_prints_faint(base_shape, pattern_art):
    """Each pass is solid ink, so there is no dither floor to fall through."""
    smooth = compose(
        base_shape, pattern_art,
        ComposeSpec(keep="remove the white background", fade=Fade(mode="linear")),
    )
    stacked = compose(
        base_shape, pattern_art,
        ComposeSpec(keep="remove the white background", fade=Fade(mode="linear", layers=4)),
    )
    assert smooth.faintest_alpha() < 0.12          # the smooth tail is unprintable
    assert stacked.faintest_alpha() >= 0.24        # the thinnest plateau is 1 of 4


# --- exports ----------------------------------------------------------------

def test_export_writes_one_file_per_pass(tmp_path, base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(keep="remove the white background", fade=Fade(mode="linear", layers=3)),
    )
    manifest = export(
        result, tmp_path,
        ExportSpec(formats=["png"], targets=["layers", "layer-map"], basename="panel"),
    )
    names = {entry["file"] for entry in manifest}
    assert names == {
        "panel_layer1of3.png", "panel_layer2of3.png",
        "panel_layer3of3.png", "panel_layer-map.png",
    }
    assert all(entry["alpha"] for entry in manifest if "layer-map" not in entry["file"])


def test_each_pass_is_solid_ink_and_they_nest(tmp_path, base_shape, pattern_art):
    from glassprint import Raster

    result = compose(
        base_shape, pattern_art,
        ComposeSpec(keep="remove the white background", fade=Fade(mode="linear", layers=3)),
    )
    manifest = export(result, tmp_path, ExportSpec(formats=["png"], targets=["layers"]))
    passes = {
        entry["file"]: Raster.open(entry["path"]).alpha_f
        for entry in sorted(manifest, key=lambda e: e["file"])
    }
    first, second, third = (passes[k] for k in sorted(passes))

    # Later passes cover less, and always sit inside the earlier ones.
    assert first.sum() > second.sum() > third.sum()
    assert np.all(second <= first + 1e-6)
    assert np.all(third <= second + 1e-6)
    # And each is full-strength through its body — no dithering anywhere.
    assert first[first > 0.5].mean() > 0.98


def test_layer_targets_are_skipped_without_a_stacked_fade(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    manifest = export(
        result, tmp_path, ExportSpec(formats=["png"], targets=["layers", "layer-map"])
    )
    assert manifest == []


# --- glaze simulation -------------------------------------------------------

def test_glaze_multiplies_rather_than_composites(base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec(keep="remove the white background"))
    green = (125, 155, 143)
    simulated = glaze(result, green)

    # Bare glass outside the panel comes through untouched.
    assert tuple(simulated.rgba[5, 5, :3]) == pytest.approx(green, abs=2)
    # Ink over glass is darker than either — that is what multiplying means.
    inked = result.overlay_layer.alpha_f > 0.9
    assert simulated.rgb_f[inked].mean() < min(green) / 255.0
    assert simulated.rgba[:, :, 3].min() == 255  # glass is not see-through


def test_stacking_deepens_the_glaze(base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec(keep="remove the white background"))
    one = glaze(result, (125, 155, 143), layers=1)
    three = glaze(result, (125, 155, 143), layers=3)

    inked = result.overlay_layer.alpha_f > 0.9
    assert three.rgb_f[inked].mean() < one.rgb_f[inked].mean()


def test_glaze_follows_the_layer_map(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(keep="remove the white background", fade=Fade(mode="linear", layers=4)),
    )
    simulated = glaze(result, (125, 155, 143), layer_map=result.layer_map)

    # Only compare where there is actually ink — the layer map spans the whole
    # canvas, and bare glass looks the same at any layer count.
    inked = result.coverage > 0.9
    deep = inked & (result.layer_map >= 4)
    shallow = inked & (result.layer_map == 1)
    assert deep.any() and shallow.any()
    assert simulated.rgb_f[deep].mean() < simulated.rgb_f[shallow].mean()


# --- the question the stack is really answering -----------------------------

def test_stacking_pulls_a_saturated_ink_away_from_the_glass():
    rows = stack_preview((198, 52, 66), (125, 155, 143), up_to=4)
    dominance = [row["ink_dominance"] for row in rows]
    assert dominance == sorted(dominance)
    assert dominance[-1] > dominance[0] * 10   # the glass stops dictating the hue


def test_stacking_cannot_rescue_a_pale_ink():
    """A pale ink barely absorbs, so extra passes never win against the glass."""
    rows = stack_preview((247, 204, 216), (125, 155, 143), up_to=6)
    assert rows[-1]["ink_dominance"] < 3.0


def test_stacking_always_costs_brightness():
    rows = stack_preview((198, 52, 66), (125, 155, 143), up_to=5)
    light = [row["light"] for row in rows]
    assert light == sorted(light, reverse=True)
