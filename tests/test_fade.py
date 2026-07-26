from __future__ import annotations

import numpy as np
import pytest

from glassprint import ComposeSpec, Fade, Placement, compose
from glassprint import fade as fade_module
from glassprint.fade import apply, apply_cutoff, ramp

CANVAS = (100, 100)
BOX = (0, 0, 100, 100)


def _ramp(**kwargs) -> np.ndarray:
    return ramp(Fade(**kwargs), CANVAS, BOX)


# --- ramp geometry ----------------------------------------------------------

def test_linear_default_fades_downward():
    field = _ramp(mode="linear")
    assert field[0, 50] == pytest.approx(1.0, abs=0.02)
    assert field[99, 50] == pytest.approx(0.0, abs=0.02)
    assert field[50, 0] == pytest.approx(field[50, 99], abs=1e-6)  # no sideways change


def test_linear_angle_zero_fades_rightward():
    field = _ramp(mode="linear", angle=0)
    assert field[50, 0] > 0.95
    assert field[50, 99] < 0.05


def test_invert_reverses_the_direction():
    normal = _ramp(mode="linear")
    flipped = _ramp(mode="linear", invert=True)
    assert flipped[0, 50] < 0.05
    assert flipped[99, 50] > 0.95
    assert np.allclose(normal, flipped[::-1, :], atol=0.02)


def test_start_and_end_bound_the_transition():
    field = _ramp(mode="linear", start=0.4, end=0.6)
    assert field[:40, 50].min() > 0.99   # untouched before the fade starts
    assert field[62:, 50].max() < 0.01   # gone after it completes
    assert 0.1 < field[50, 50] < 0.9     # mid-transition


def test_curve_changes_the_rate_not_the_endpoints():
    linear = _ramp(mode="linear", curve=1.0)
    late = _ramp(mode="linear", curve=2.5)
    early = _ramp(mode="linear", curve=0.4)

    for field in (linear, late, early):
        assert field[0, 50] == pytest.approx(1.0, abs=0.02)
        assert field[99, 50] == pytest.approx(0.0, abs=0.02)

    # Halfway down: a high curve is still strong, a low curve has mostly gone.
    assert late[50, 50] > linear[50, 50] > early[50, 50]


def test_min_and_max_alpha_set_the_ends():
    field = _ramp(mode="linear", min_alpha=0.25, max_alpha=0.8)
    assert field[0, 50] == pytest.approx(0.8, abs=0.02)
    assert field[99, 50] == pytest.approx(0.25, abs=0.02)
    assert field.min() >= 0.25 - 1e-6


def test_ramp_is_monotonic_down_the_axis():
    column = _ramp(mode="linear", curve=1.8)[:, 50]
    assert np.all(np.diff(column) <= 1e-6)


def test_radial_is_solid_at_the_centre():
    field = _ramp(mode="radial")
    assert field[50, 50] > 0.98
    assert field[0, 0] < 0.05
    # Equidistant points match, so the ramp is circular rather than stretched.
    assert field[50, 20] == pytest.approx(field[20, 50], abs=0.02)


def test_radial_centre_can_be_moved():
    field = _ramp(mode="radial", center_x=0.0, center_y=0.0)
    assert field[0, 0] > 0.98
    assert field[99, 99] < 0.05


def test_shape_mode_fades_toward_the_rim():
    shape = np.zeros(CANVAS, dtype=np.float32)
    shape[20:80, 20:80] = 1.0
    field = ramp(Fade(mode="shape"), CANVAS, BOX, shape)
    assert field[50, 50] > 0.95      # deep inside
    assert field[21, 50] < 0.15      # at the rim


def test_shape_mode_without_a_mask_is_harmless():
    field = ramp(Fade(mode="shape"), CANVAS, BOX, None)
    assert np.allclose(field, 1.0)


def test_none_mode_is_fully_opaque():
    assert np.allclose(_ramp(mode="none"), 1.0)


def test_invalid_fade_is_rejected():
    with pytest.raises(ValueError):
        Fade(mode="sideways").validated()
    with pytest.raises(ValueError):
        Fade(mode="linear", curve=0).validated()


# --- element handling -------------------------------------------------------

def _three_dots() -> np.ndarray:
    alpha = np.zeros(CANVAS, dtype=np.float32)
    for cy in (15, 50, 85):
        alpha[cy - 6:cy + 6, 44:56] = 1.0
    return alpha


def test_per_element_gives_each_element_one_opacity():
    alpha = _three_dots()
    field = _ramp(mode="linear")
    out, count = apply(alpha, field, Fade(mode="linear", per_element=True))

    assert count == 3
    for cy in (15, 50, 85):
        patch = out[cy - 6:cy + 6, 44:56]
        assert patch.max() - patch.min() < 1e-5  # flat across the element
    # And the population still fades top to bottom.
    assert out[15, 50] > out[50, 50] > out[85, 50]


def test_continuous_fade_cuts_through_an_element():
    alpha = _three_dots()
    field = _ramp(mode="linear")
    out, _ = apply(alpha, field, Fade(mode="linear", per_element=False))
    patch = out[44:56, 44:56]
    assert patch.max() - patch.min() > 0.05


def test_dissolve_drops_whole_elements_and_leaves_survivors_solid():
    alpha = np.zeros((200, 200), dtype=np.float32)
    for row in range(10):
        for col in range(10):
            alpha[row * 20 + 6:row * 20 + 14, col * 20 + 6:col * 20 + 14] = 1.0

    field = ramp(Fade(mode="linear"), (200, 200), (0, 0, 200, 200))
    out, count = apply(alpha, field, Fade(mode="linear", dissolve=1.0, seed=7))

    assert count == 100
    present = out[alpha > 0.5]
    # Pure dissolve is binary: every element is either fully there or gone.
    assert set(np.unique(np.round(present, 5)).tolist()) <= {0.0, 1.0}
    assert 0.0 < present.mean() < 1.0          # some survived, some did not

    # Survival is one draw per element, so judge the population rather than any
    # single element or row. Across 50 elements a side the trend is decisive:
    # the top half sits around 75% survival, the bottom half around 25%.
    survivors = np.array([out[row * 20 + 10, 10::20].sum() for row in range(10)])
    assert survivors[:5].sum() > 30
    assert survivors[5:].sum() < 25


def test_a_dissolved_element_leaves_no_halo_behind():
    """The soft edge of a dropped element must go with it, not linger."""
    from glassprint import masks

    solid = np.zeros(CANVAS, dtype=np.float32)
    solid[40:60, 40:60] = 1.0
    alpha = masks.feather(solid, 3.0)  # an anti-aliased edge, as any artwork has

    # A flat ramp, so without the fix the halo would sit at 0.85 forever.
    field = _ramp(mode="linear", min_alpha=0.85, max_alpha=0.85)
    seed = next(s for s in range(200) if np.random.default_rng(s).random(1)[0] > 0.9)

    out, count = apply(alpha, field, Fade(mode="linear", dissolve=1.0, seed=seed))
    assert count == 1
    assert out.max() == 0.0, "the element was dropped but its soft edge survived"


def test_a_surviving_element_keeps_its_soft_edge():
    from glassprint import masks

    solid = np.zeros(CANVAS, dtype=np.float32)
    solid[40:60, 40:60] = 1.0
    alpha = masks.feather(solid, 3.0)

    field = _ramp(mode="linear", min_alpha=0.9, max_alpha=0.9)
    seed = next(s for s in range(200) if np.random.default_rng(s).random(1)[0] < 0.5)

    out, _ = apply(alpha, field, Fade(mode="linear", dissolve=1.0, seed=seed))
    assert out[50, 50] == pytest.approx(1.0, abs=1e-6)
    edge = out[(alpha > 0.05) & (alpha < 0.5)]
    assert edge.size > 0 and edge.max() > 0.0   # the anti-aliasing is still there


def test_dissolve_is_reproducible_for_a_seed():
    alpha = _three_dots()
    field = _ramp(mode="linear")
    spec = Fade(mode="linear", dissolve=1.0, seed=3)
    first, _ = apply(alpha, field, spec)
    second, _ = apply(alpha, field, spec)
    assert np.array_equal(first, second)

    other, _ = apply(alpha, field, Fade(mode="linear", dissolve=1.0, seed=4))
    assert not np.array_equal(first, other)


def test_scope_limits_the_fade_to_selected_elements():
    alpha = _three_dots()
    field = _ramp(mode="linear")
    scope = np.zeros(CANVAS, dtype=np.float32)
    scope[79:91, 44:56] = 1.0  # only the bottom dot is in scope

    out, _ = apply(alpha, field, Fade(mode="linear"), scope)
    assert out[15, 50] == pytest.approx(1.0, abs=1e-6)   # untouched
    assert out[50, 50] == pytest.approx(1.0, abs=1e-6)   # untouched
    assert out[85, 50] < 0.2                             # faded


def test_cutoff_drops_the_unprintable_tail():
    alpha = np.linspace(0.0, 1.0, 100, dtype=np.float32)[None, :].repeat(10, axis=0)
    out = apply_cutoff(alpha, 0.15)
    assert out[0, alpha[0] < 0.15].max() == 0.0
    assert out[0, -1] == pytest.approx(1.0)


def test_cutoff_of_zero_changes_nothing():
    alpha = np.linspace(0.0, 1.0, 20, dtype=np.float32)[None, :]
    assert np.array_equal(apply_cutoff(alpha, 0.0), alpha)


# --- through the pipeline ---------------------------------------------------

def test_fade_reaches_the_exported_overlay(base_shape, pattern_art):
    plain = compose(base_shape, pattern_art, ComposeSpec())
    faded = compose(
        base_shape,
        pattern_art,
        ComposeSpec(fade=Fade(mode="linear", angle=90)),
    )

    top = faded.overlay_layer.alpha_f[60:100, 150:250].max()
    bottom = faded.overlay_layer.alpha_f[200:255, 150:250].max()
    assert top > bottom
    assert plain.overlay_layer.alpha_f.sum() > faded.overlay_layer.alpha_f.sum()


def test_fade_summary_reports_what_happened(base_shape, pattern_art):
    result = compose(
        base_shape,
        pattern_art,
        ComposeSpec(fade=Fade(mode="linear", dissolve=1.0, seed=2, cutoff=0.1)),
    )
    summary = result.summary()["fade"]
    assert summary["mode"] == "linear"
    assert summary["elements"] > 0
    assert summary["faintest_alpha"] >= 0.1   # the cutoff removed anything fainter
    assert "dissolve" in summary["describe"]


def test_scoped_fade_through_the_pipeline(base_shape, pattern_art):
    """Fading only the green leaves must leave the red dots alone."""
    result = compose(
        base_shape,
        pattern_art,
        ComposeSpec(
            keep="remove the white background",
            placement=Placement(fit="tile", repeat_across=3),
            fade=Fade(mode="linear", what="green", min_alpha=0.0),
        ),
    )
    assert not any("Nothing in the overlay matched" in note for note in result.notes)
    # Plenty of the artwork is still fully opaque, because only part of it faded.
    assert (result.overlay_layer.alpha_f > 0.99).sum() > 1000


def test_unmatched_fade_scope_is_reported_and_leaves_the_art_alone(base_shape, pattern_art):
    plain = compose(base_shape, pattern_art, ComposeSpec())
    result = compose(
        base_shape,
        pattern_art,
        ComposeSpec(fade=Fade(mode="linear", what="#00ff00")),
    )
    assert any("Nothing in the overlay matched" in note for note in result.notes)
    assert np.array_equal(result.overlay_layer.rgba, plain.overlay_layer.rgba)


def test_cutoff_applies_without_a_fade(base_shape, pattern_art):
    result = compose(
        base_shape,
        pattern_art,
        ComposeSpec(edge_feather=4.0, fade=Fade(cutoff=0.3)),
    )
    alpha = result.overlay_layer.alpha_f
    present = alpha[alpha > 0]
    assert present.min() >= 0.3


# -- what the printer actually does -----------------------------------------
#
# These pin measurements taken from a real print on a EufyMake E1, not
# reasoning. A stepped alpha ramp stopped after its 45% patch and a continuous
# one stopped at half its length, while the screened row carrying the same tones
# was still printing at 12% coverage.


def test_a_tonal_fade_to_nothing_is_flagged_as_impossible():
    """It does not fade — it runs at half strength and then stops dead."""
    notes = fade_module.check(Fade(mode="linear", start=0.0, end=1.0))
    assert any("stop dead" in note for note in notes)
    assert any("dot screen" in note for note in notes)


def test_a_shallow_tonal_fade_is_left_alone():
    """Staying above the cliff is a legitimate, if subtle, fade."""
    assert fade_module.check(Fade(mode="linear", min_alpha=0.6)) == []


def test_carrying_the_tail_with_coverage_clears_the_warning():
    for carried in (
        Fade(mode="linear", halftone_mm=0.8),
        Fade(mode="linear", layers=3),
        Fade(mode="linear", dissolve=1.0),
    ):
        assert fade_module.check(carried) == [], carried


def test_too_fine_a_screen_is_flagged():
    notes = fade_module.check(Fade(mode="linear", halftone_mm=0.25))
    assert any("bridge" in note for note in notes)
    assert fade_module.check(Fade(mode="linear", halftone_mm=0.6)) == []


def test_the_measured_constants_are_what_the_print_showed():
    assert fade_module.ALPHA_CLIFF == 0.5
    assert fade_module.COVERAGE_FLOOR == 0.12
    assert fade_module.MIN_HALFTONE_MM == 0.6
