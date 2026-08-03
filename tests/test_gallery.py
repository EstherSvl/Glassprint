"""End-to-end jobs, checked the way you would check a print.

``tools/make_gallery.py`` composes a spread of realistic pairings and lays them
out to be looked at. These are the same jobs with the judgements a person makes
in front of that contact sheet written down: is there ink, is it inside the
object, and did the background actually come off.

Every assertion here started as something visible in the picture and invisible
in the old tests, which used flat shapes on flat white and stayed green through
all of it:

* a motif exported with margins around it landed at seventy per cent of the
  size it was asked for, off-centre by however lopsided the padding was;
* a repeat whose motifs touch the edge of the tile had its background reading
  inverted — the tool removed the flowers and kept the paper, and on a white
  plate that is invisible until you print it on green glass;
* a graded studio backdrop came along with the subject, because the ground was
  measured against a single colour;
* a pale subject on pale ground came out full of holes and halos at every
  tolerance, with nothing said about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import gallery_art as art  # noqa: E402
import make_gallery  # noqa: E402

from glassprint import masks, segment  # noqa: E402
from glassprint.compose import ComposeSpec, compose  # noqa: E402
from glassprint.pattern import Placement  # noqa: E402
from glassprint.raster import Raster  # noqa: E402

JOBS = {job.name: job for job in make_gallery.jobs()}


@pytest.fixture(scope="module")
def ran() -> dict:
    """Every job, composed once and shared — the whole set takes a few seconds."""
    return {name: make_gallery.run(job)[1] for name, job in JOBS.items()}


# --- what has to be true of every job ---------------------------------------


@pytest.mark.parametrize("name", sorted(JOBS))
def test_ink_lands_on_the_object_and_nowhere_else(name, ran):
    """Clipping is on for all of these, so print off the edge is a scrap."""
    assert ran[name]["spill"] < 0.005, f"{name} prints outside the target shape"


@pytest.mark.parametrize("name", sorted(JOBS))
def test_something_actually_prints(name, ran):
    """A job that lays down nothing is a failure that writes a file happily."""
    assert ran[name]["filled"] > 0.02, f"{name} covers almost none of the shape"


@pytest.mark.parametrize("name", sorted(JOBS))
def test_a_background_that_was_removed_stays_removed(name, ran):
    """The whole shape solid with ink means the cut-out kept the paper.

    This is the one that hid. On a near-white plate a kept white ground looks
    exactly like a working result; it only shows up as a slab when you print it
    onto coloured glass, by which point you have used the glass.
    """
    if "remove" not in (JOBS[name].spec.keep or "") and not JOBS[name].spec.layers:
        pytest.skip("no background instruction in this job")
    assert ran[name]["filled"] < 0.90, f"{name} covers the whole shape — background kept?"


# --- the specific things that were broken -----------------------------------


def _content_box(rgba: np.ndarray) -> tuple[int, int, int, int]:
    return masks.bbox(rgba[:, :, 3].astype(np.float32) / 255.0, threshold=0.35)


def _padded(raster: Raster, factor: float = 2.2) -> Raster:
    """The same artwork on a much bigger canvas, off to one side."""
    height = int(raster.height * factor)
    width = int(raster.width * factor)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[10 : 10 + raster.height, 10 : 10 + raster.width] = raster.rgba
    return Raster(canvas, dpi=raster.dpi)


def test_padding_around_a_motif_does_not_shrink_it():
    """A flower on a big canvas must land the same size as one on a tight canvas.

    Procreate hands you whatever canvas you drew on. Scaling that to the target
    scales the empty space with it, and the artwork arrives small — which is
    what "it worked very poorly" looks like from the outside.
    """
    base = art.base_coaster()
    tight = art.overlay_gold_leaf()
    spec = ComposeSpec(placement=Placement(fit="contain", scale=0.9))

    snug = compose(base, tight, spec).overlay_layer.rgba
    roomy = compose(base, _padded(tight), spec).overlay_layer.rgba

    a, b = _content_box(snug), _content_box(roomy)
    snug_w, roomy_w = a[2] - a[0], b[2] - b[0]
    assert abs(snug_w - roomy_w) / snug_w < 0.06, (
        f"padding changed the printed size: {snug_w}px vs {roomy_w}px"
    )


def test_padding_does_not_push_a_motif_off_centre():
    base = art.base_coaster()
    tight = art.overlay_gold_leaf()
    spec = ComposeSpec(placement=Placement(fit="contain", scale=0.8))

    for overlay in (tight, _padded(tight)):
        box = _content_box(compose(base, overlay, spec).overlay_layer.rgba)
        centre_x = (box[0] + box[2]) / 2 / base.width
        centre_y = (box[1] + box[3]) / 2 / base.height
        assert abs(centre_x - 0.5) < 0.04, f"off-centre horizontally: {centre_x:.3f}"
        assert abs(centre_y - 0.5) < 0.04, f"off-centre vertically: {centre_y:.3f}"


def test_a_repeat_whose_motifs_touch_the_edge_is_not_read_inside_out():
    """The border of a tile is not a sample of its background.

    A repeat *is* motifs running off the canvas edge, so the border can easily
    be more flower than paper. Taking the median of it made the reference red,
    and the tool then removed the flowers and kept the ground, confidently.
    """
    tile = art.overlay_seamless_floral()
    background = segment.background_mask(tile, tolerance=1.0)
    assert background.mean() > 0.55, (
        f"only {background.mean():.0%} read as background — the ground is most of this tile"
    )


def test_a_graded_backdrop_is_still_a_backdrop():
    """A studio sweep falls off toward the corners; one colour cannot describe it."""
    photo = art.overlay_orchid_on_gradient()
    background = segment.background_mask(photo, tolerance=1.0)
    assert background.mean() > 0.6, (
        f"only {background.mean():.0%} of a graded backdrop was recognised"
    )


def test_a_pale_subject_on_pale_ground_is_flagged_rather_than_guessed_at():
    """White orchids on white paper: no tolerance separates them, so say so.

    Measured across the whole range, the flowers and the shadow they cast move
    together — the best the colour heuristic ever manages is two fifths of the
    flower and a fifth of the paper. Tuning the number gives a different bad
    answer, not a good one, so the tool has to name the situation.
    """
    backends = segment.Backends()
    segment.background_mask(art.overlay_orchid_on_white(), 1.0, backends=backends)
    assert any("barely a different colour" in note for note in backends.notes), backends.notes


def test_artwork_that_separates_cleanly_is_not_nagged_about():
    """The warning is worth nothing if it fires on work that came out right."""
    for name in ("seamless-floral", "scattered-motifs", "orchid-on-gradient"):
        backends = segment.Backends()
        segment.background_mask(art.OVERLAYS[name](), 1.0, backends=backends)
        assert not any("barely a different" in note for note in backends.notes), name


def test_a_lens_cutout_is_never_printed_over():
    """A hole in the alpha is a hole in the object, not an area to fill.

    Filling holes is right for a ragged silhouette and wrong here: ink over the
    camera is a scrapped case, and it costs the case to find out.
    """
    case = art.base_phone_case()
    result = compose(
        case, art.overlay_orchid_cutout(),
        ComposeSpec(placement=Placement(fit="contain", scale=0.95)),
    )
    lens = (case.alpha_f < 0.5) & _inside_top(case)
    ink = result.overlay_layer.rgba[:, :, 3].astype(np.float32) / 255.0
    assert float(ink[lens].max()) < 0.02, "printed over the lens cutout"


def _inside_top(raster: Raster) -> np.ndarray:
    """The upper region where the cutout lives, excluding the outer margin."""
    keep = np.zeros((raster.height, raster.width), dtype=bool)
    top, bottom = int(raster.height * 0.04), int(raster.height * 0.28)
    left, right = int(raster.width * 0.08), int(raster.width * 0.55)
    keep[top:bottom, left:right] = True
    return keep


def test_a_pattern_stays_inside_a_silhouette_that_is_not_its_bounding_box():
    """A vase is a fifth as wide at the neck as at the belly."""
    result = compose(
        art.base_vase(), art.overlay_seamless_floral(),
        ComposeSpec(
            keep="remove the white background",
            placement=Placement(fit="tile", repeat_across=4),
        ),
    )
    shape = result.shape_mask
    ink = result.overlay_layer.rgba[:, :, 3].astype(np.float32) / 255.0
    outside = float((ink * (1.0 - shape)).sum()) / max(float(ink.sum()), 1e-6)
    assert outside < 0.005, f"{outside:.1%} of the ink is off the vase"
