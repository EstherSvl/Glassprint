from __future__ import annotations

import numpy as np
import pytest

from glassprint import ComposeSpec, Fade, compose
from glassprint.fade import halftone

FLAT = (400, 400)


def _screen(coverage: float, pitch: float = 40.0, angle: float = 45.0) -> np.ndarray:
    return halftone(np.full(FLAT, coverage, dtype=np.float32), pitch, angle)


@pytest.mark.parametrize("coverage", [0.1, 0.25, 0.5, 0.75])
def test_dot_area_matches_the_requested_tone(coverage):
    """Tone comes from dot size, so the ink laid down must match the ramp."""
    assert _screen(coverage).mean() == pytest.approx(coverage, abs=0.02)


def test_full_coverage_is_completely_solid():
    assert _screen(1.0).min() == pytest.approx(1.0)


def test_zero_coverage_lays_down_nothing():
    assert _screen(0.0).max() == pytest.approx(0.0)


def test_dots_are_full_strength_ink():
    """The point of screening: no dilute ink, only more or less of it."""
    screen = _screen(0.4)
    assert screen.max() == pytest.approx(1.0)
    # Everything is either a dot, a gap, or the thin anti-aliased rim between.
    middling = ((screen > 0.15) & (screen < 0.85)).mean()
    assert middling < 0.12


def test_pitch_controls_how_many_dots_there_are():
    coarse = _screen(0.3, pitch=60.0)
    fine = _screen(0.3, pitch=20.0)
    assert coarse.mean() == pytest.approx(fine.mean(), abs=0.03)  # same tone

    from scipy import ndimage

    coarse_count = ndimage.label(coarse > 0.5)[1]
    fine_count = ndimage.label(fine > 0.5)[1]
    assert fine_count > coarse_count * 4


def test_angle_rotates_the_screen():
    assert not np.allclose(_screen(0.3, angle=0.0), _screen(0.3, angle=45.0))


def test_a_screened_ramp_still_fades():
    ramp = np.linspace(1.0, 0.0, 400, dtype=np.float32)[:, None].repeat(400, axis=1)
    screened = halftone(ramp, 30.0, 45.0)
    bands = [screened[i * 100:(i + 1) * 100].mean() for i in range(4)]
    assert bands == sorted(bands, reverse=True)
    assert bands[0] > 0.8 and bands[-1] < 0.2


# --- through the pipeline ---------------------------------------------------

def test_screen_gradates_a_solid_area_that_dissolve_cannot(base_shape, motif_art):
    """One big motif is a single element, so only a dot screen can grade it."""
    dissolved = compose(
        base_shape, motif_art,
        ComposeSpec(fade=Fade(mode="linear", dissolve=1.0, seed=1)),
    )
    screened = compose(
        base_shape, motif_art,
        ComposeSpec(fade=Fade(mode="linear", halftone_mm=2.0)),
    )

    # Dissolve sees one element and can only take it or leave it — whichever
    # way the draw fell, there is no gradation in the result.
    assert dissolved.fade_elements == 1
    dissolved_alpha = dissolved.overlay_layer.alpha_f
    tones = set(np.unique(np.round(dissolved_alpha[dissolved_alpha > 0.01], 3)).tolist())
    assert tones <= {1.0}

    # The screen grades within it. Measure against an unfaded compose so the
    # figure is "ink kept in this band", not "how much motif lands in it".
    plain = compose(base_shape, motif_art, ComposeSpec())
    kept = []
    for y in (100, 125, 150, 175):
        band = (slice(y, y + 25), slice(160, 240))
        reference = plain.overlay_layer.alpha_f[band].sum()
        assert reference > 100, "the sample band should sit on the motif"
        kept.append(screened.overlay_layer.alpha_f[band].sum() / reference)

    # The ramp spans the whole panel, so the top of the motif is already part
    # way into the fade — what matters is that it thins steadily through it.
    assert kept == sorted(kept, reverse=True)
    assert kept[0] > 0.6 and kept[-1] < 0.5


def test_screened_ink_is_reported_as_printable(base_shape, pattern_art):
    """Dots are full-strength, so the printability warning must not fire."""
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            fade=Fade(mode="linear", halftone_mm=2.0),
        ),
    )
    assert result.faintest_alpha() > 0.9


def test_a_fine_screen_is_flagged_as_a_moire_risk(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(fade=Fade(mode="linear", halftone_mm=0.3)),
    )
    assert any("moir" in note for note in result.notes)


def test_a_coarse_screen_is_not_flagged(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(fade=Fade(mode="linear", halftone_mm=2.0)),
    )
    assert not any("moir" in note for note in result.notes)


def test_screen_takes_precedence_over_dissolve_and_says_so(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(fade=Fade(mode="linear", halftone_mm=2.0, dissolve=1.0)),
    )
    assert any("dissolve left off" in note for note in result.notes)
    assert result.fade_elements == 0  # the element pass did not run


def test_screen_respects_the_fade_scope(base_shape, pattern_art):
    """A screened fade scoped to one colour leaves the rest of the art solid."""
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(
            keep="remove the white background",
            fade=Fade(mode="linear", halftone_mm=2.0, what="green"),
        ),
    )
    assert not any("Nothing in the overlay matched" in note for note in result.notes)
    assert (result.overlay_layer.alpha_f > 0.99).sum() > 1000


def test_screen_describes_itself(base_shape, pattern_art):
    result = compose(
        base_shape, pattern_art,
        ComposeSpec(fade=Fade(mode="linear", halftone_mm=1.5, halftone_angle=30)),
    )
    assert "1.5mm dots at 30°" in result.summary()["fade"]["describe"]
