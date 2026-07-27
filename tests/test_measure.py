"""Calibration: does a photograph of the chart recover the printer that made it?

The tests plant a *known* printer inside a synthetic photograph and ask the
module to find it again. The planted model is deliberately not the model being
fitted — it is a Yule-Nielsen-ish power law with its own cross-talk, where the
fit assumes a gamma curve into a linear mixer. If the two agreed, these tests
would only prove that algebra is reversible.

Everything a real photograph does wrong is done here on purpose: perspective
from holding the plate at an angle, a lamp brighter on one side, sensor noise,
and a white balance that is not quite neutral.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pytest

from glassprint import measure

GREEN = (46, 168, 110)
AMBER = (196, 132, 44)
DARK_GREEN = (22, 74, 44)
PALE_GREEN = (150, 205, 175)


# -- the printer being impersonated ------------------------------------------


def true_transmittance(requested: np.ndarray) -> np.ndarray:
    """A plausible printer, and not the one :func:`measure.fit` assumes.

    Ink demand goes through a power law per channel, the channels bleed into one
    another, and the whole thing saturates — so full black transmits a little
    rather than nothing, which is what the real plates showed.
    """
    demand = np.clip(1.0 - np.asarray(requested, dtype=np.float64) / 255.0, 0.0, 1.0)
    bleed = np.array(
        [
            [1.00, 0.14, 0.09],
            [0.11, 1.00, 0.12],
            [0.07, 0.10, 1.00],
        ]
    )
    mixed = np.clip(demand ** np.array([1.35, 1.10, 0.85]) @ bleed.T, 0.0, 1.6)
    density = np.array([2.05, 1.85, 1.70]) * mixed
    return np.clip(10.0 ** (-density), 0.006, 1.0)


@lru_cache(maxsize=12)
def photograph(
    glass=GREEN,
    *,
    size=(1400, 2400),
    tilt=0.06,
    vignette=0.22,
    noise=0.004,
    seed=11,
    plate_margin_mm=2.5,
) -> np.ndarray:
    """A synthetic phone photo of the printed chart, warts included.

    ``plate_margin_mm`` is the bare glass around the chart. The default is the
    chart's own margin — a plate cut to the artwork. Raise it for the realistic
    case, where the chart is printed on whatever offcut came to hand and there
    is a wide border of bare glass that a naive detector will happily mistake
    for the chart.
    """
    layout = measure.CHART
    height, width = size
    rng = np.random.default_rng(seed)

    # Start from the backlight, then put glass and ink in front of parts of it.
    scene = np.ones((height, width, 3), dtype=np.float64)

    # Where the plate sits in the photo, as a quadrilateral with perspective.
    margin_x, margin_y = width * 0.10, height * 0.16
    quad = np.array(
        [
            [margin_x, margin_y + height * tilt],
            [width - margin_x, margin_y],
            [width - margin_x * 0.95, height - margin_y],
            [margin_x * 1.05, height - margin_y - height * tilt * 0.5],
        ]
    )
    frame = np.array(
        [
            [0.0, 0.0],
            [layout.frame_w_mm, 0.0],
            [layout.frame_w_mm, layout.frame_h_mm],
            [0.0, layout.frame_h_mm],
        ]
    )
    # The plate: the frame plus however much bare glass surrounds it. The photo
    # quad is the plate, so the chart's own coordinates are inset within it.
    bleed = plate_margin_mm
    plate = np.array(
        [
            [-bleed, -bleed],
            [layout.frame_w_mm + bleed, -bleed],
            [layout.frame_w_mm + bleed, layout.frame_h_mm + bleed],
            [-bleed, layout.frame_h_mm + bleed],
        ]
    )
    forward = measure._homography(plate, quad)
    back = np.linalg.inv(forward)

    ys, xs = np.mgrid[0:height, 0:width]
    points = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1) @ back.T
    chart_xy = (points[:, :2] / points[:, 2:3]).reshape(height, width, 2)

    on_plate = (
        (chart_xy[:, :, 0] >= -bleed)
        & (chart_xy[:, :, 0] <= layout.frame_w_mm + bleed)
        & (chart_xy[:, :, 1] >= -bleed)
        & (chart_xy[:, :, 1] <= layout.frame_h_mm + bleed)
    )
    inside = (
        (chart_xy[:, :, 0] >= 0)
        & (chart_xy[:, :, 0] <= layout.frame_w_mm)
        & (chart_xy[:, :, 1] >= 0)
        & (chart_xy[:, :, 1] <= layout.frame_h_mm)
    )
    glass_t = np.array(glass, dtype=np.float64) / 255.0
    scene[on_plate] *= glass_t

    # The mesh — frame and gaps — is solid black ink.
    ink = np.zeros((height, width, 3), dtype=np.float64)
    ink[...] = 1.0
    mesh = inside.copy()
    for index, colour in enumerate(measure.patches()):
        x, y, w, h = layout.cell(index)
        cell = (
            (chart_xy[:, :, 0] >= x)
            & (chart_xy[:, :, 0] <= x + w)
            & (chart_xy[:, :, 1] >= y)
            & (chart_xy[:, :, 1] <= y + h)
        )
        mesh &= ~cell
        if colour is not None:
            ink[cell] = true_transmittance(np.array(colour, dtype=np.float64))
    ink[mesh] = true_transmittance(np.array([0.0, 0.0, 0.0]))
    scene *= ink

    # A lamp that is not even, a camera that is not neutral, a sensor that is
    # not silent. None of these should change the answer.
    fall = 1.0 - vignette * (xs / width)
    scene *= fall[:, :, None]
    scene *= np.array([1.03, 1.00, 0.96])
    scene += rng.normal(0.0, noise, scene.shape)

    return np.clip(measure.encode(np.clip(scene, 0.0, 1.0)) * 255.0, 0, 255).astype(np.uint8)


# -- the chart itself --------------------------------------------------------


def test_chart_is_small_enough_for_an_offcut():
    layout = measure.CHART
    assert layout.width_mm <= 145.0 and layout.height_mm <= 50.0
    # The narrowest spare glass in the drawer is 50mm; a chart that only just
    # fits cannot be placed by hand.
    assert layout.height_mm <= 45.0


def test_every_cell_is_filled_and_the_corners_are_bare():
    layout = measure.CHART
    cells = measure.patches()
    assert len(cells) == layout.columns * layout.rows
    assert len(measure.colours()) + 4 == len(cells), "colours and cells must match exactly"
    for index in measure.glass_cells(layout):
        assert cells[index] is None
    assert all(c is not None for i, c in enumerate(cells) if i not in measure.glass_cells(layout))


def test_the_black_cell_is_not_symmetric():
    """The one feature that says which way up the plate was photographed."""
    cells = measure.patches()
    black = next(i for i, c in enumerate(cells) if c == (0, 0, 0))
    assert cells[len(cells) - 1 - black] != (0, 0, 0)


def test_chart_renders_bare_glass_as_a_hole():
    raster = measure.chart()
    layout = measure.CHART
    scale = raster.width / layout.width_mm
    for index in measure.glass_cells(layout):
        x, y, w, h = layout.cell(index)
        px = lambda v: int((layout.margin_mm + v) * scale)  # noqa: E731
        patch = raster.rgba[
            px(y + h / 2) - 4 : px(y + h / 2) + 4, px(x + w / 2) - 4 : px(x + w / 2) + 4
        ]
        assert patch[:, :, 3].max() == 0, "a bare-glass corner must print nothing at all"


def test_held_out_patches_are_not_in_the_cube():
    every = measure.colours()
    fitted = {every[i] for i in range(len(every)) if i not in measure.HELD_OUT}
    for index in measure.HELD_OUT:
        if index == measure.REPEAT[1]:
            continue
        assert every[index] not in fitted, "a held-out patch that is also fitted proves nothing"


def test_the_repeat_really_repeats():
    every = measure.colours()
    original, repeat = measure.REPEAT
    assert every[original] == every[repeat]
    assert repeat in measure.HELD_OUT and original in measure.HELD_OUT


# -- reading -----------------------------------------------------------------


def test_reads_a_tilted_unevenly_lit_photograph():
    profile = measure.read(photograph())
    residuals = profile.residuals()
    assert residuals["held_out"] < 3.0, residuals
    assert residuals["worst"] < 10.0, residuals


def test_recovers_the_tone_curve_that_was_planted():
    """The gamma is the headline number a user reads off the profile."""
    profile = measure.read(photograph())
    assert np.abs(profile.gamma - np.array([1.35, 1.10, 0.85])).max() < 0.15, profile.gamma


def test_the_repeated_patch_shows_the_noise_floor():
    """Every other error here is only meaningful against this one."""
    profile = measure.read(photograph())
    original, repeat = measure.REPEAT
    noise = float(np.abs(profile.measured[original] - profile.measured[repeat]).mean() * 255)
    assert noise < 2.0, f"the measurement is noisier than the thing being measured: {noise}"


def test_beats_the_assumption_it_replaces():
    residuals = measure.read(photograph()).residuals()
    assert residuals["naive"] > 10 * residuals["held_out"], (
        f"calibration is not worth the glass: {residuals}"
    )


def test_a_chart_on_a_bigger_plate_is_still_found():
    """The failure that mattered, because it was silent.

    A single threshold takes the largest dark blob, and on an offcut wider than
    the chart that blob is the *plate*. The reader then sampled a grid of
    colours that were not there and returned a profile wrong by 200 levels,
    with no complaint. Dark glass made it worse, because then the plate is
    unambiguously the darkest thing in shot.
    """
    for glass in (GREEN, DARK_GREEN, PALE_GREEN):
        for margin in (14.0, 30.0):
            residuals = measure.read(
                photograph(glass, plate_margin_mm=margin, seed=7)
            ).residuals()
            assert residuals["held_out"] < 3.0, (glass, margin, residuals)


def test_the_glass_colour_survives_a_wide_plate_and_a_steep_lamp():
    for glass, kwargs in [
        (GREEN, dict(plate_margin_mm=22.0)),
        (DARK_GREEN, dict(plate_margin_mm=18.0)),
        (PALE_GREEN, dict(plate_margin_mm=18.0)),
        (GREEN, dict(plate_margin_mm=14.0, vignette=0.45)),
    ]:
        read = measure.read(photograph(glass, **kwargs))
        assert np.abs(np.array(read.glass, float) - np.array(glass, float)).max() < 4, (
            glass,
            kwargs,
            read.glass,
        )


def test_the_four_corners_are_what_make_it_immune_to_an_uneven_lamp():
    """Halve the lamp across the plate; the answer must not move."""
    even = measure.read(photograph(vignette=0.02, seed=5)).residuals()
    steep = measure.read(photograph(vignette=0.45, seed=5)).residuals()
    assert steep["held_out"] < 3.0, steep
    assert steep["held_out"] < even["held_out"] + 2.0, (even, steep)


def test_predicts_a_colour_it_never_saw():
    profile = measure.read(photograph())
    for requested in [(70, 145, 30), (210, 40, 90), (150, 150, 150)]:
        expected = np.array(GREEN) / 255.0 * true_transmittance(np.array(requested, float))
        got = np.array(profile.predict(requested, GREEN)) / 255.0
        assert np.abs(got - expected).max() * 255 < 12, requested


def test_upside_down_is_still_read():
    upright = measure.read(photograph())
    flipped = measure.read(photograph()[::-1, ::-1])
    assert np.abs(flipped.measured - upright.measured).max() < 0.05


def test_survives_a_dim_photograph():
    photo = (photograph().astype(np.float64) * 0.55).astype(np.uint8)
    residuals = measure.read(photo).residuals()
    assert residuals["held_out"] < 12.0, residuals


# -- inverting ---------------------------------------------------------------


def test_request_for_round_trips():
    profile = measure.read(photograph())
    for desired in [(30, 90, 60), (20, 120, 80), (12, 60, 40)]:
        requested, reachable = profile.request_for(desired, GREEN)
        assert reachable, desired
        back = profile.predict(requested, GREEN)
        assert np.abs(np.array(back) - np.array(desired)).max() < 14, (desired, requested, back)


def test_asking_for_something_brighter_than_the_glass_says_so():
    profile = measure.read(photograph())
    _, reachable = profile.request_for((255, 255, 255), GREEN)
    assert not reachable
    requested, reachable = profile.request_for((200, 240, 220), GREEN)
    assert not reachable
    # Still returns the closest it can rather than nothing, so a caller that
    # ignores the flag is merely disappointed rather than broken.
    assert all(0 <= c <= 255 for c in requested)


# -- one printer, several glasses --------------------------------------------


def test_the_ink_profile_transfers_to_other_glass():
    """The claim the whole 'one chart' story rests on."""
    green = measure.read(photograph(GREEN))
    amber = measure.read(photograph(AMBER, seed=29))
    verdict = green.check(amber)
    assert verdict["transfers"], verdict
    assert verdict["mean"] < 3.0, verdict


def test_the_glass_colour_is_read_well_enough_to_use():
    for glass in (GREEN, AMBER):
        profile = measure.read(photograph(glass, seed=29 if glass is AMBER else 11))
        assert np.abs(np.array(profile.glass, float) - np.array(glass, float)).max() < 6, (
            glass,
            profile.glass,
        )


def test_predicting_on_glass_it_was_not_measured_on():
    green = measure.read(photograph(GREEN))
    for requested in [(0, 0, 0), (128, 128, 128), (255, 128, 0)]:
        expected = np.array(AMBER) / 255.0 * true_transmittance(np.array(requested, float))
        got = np.array(green.predict(requested, AMBER)) / 255.0
        assert np.abs(got - expected).max() * 255 < 14, requested


# -- failing usefully --------------------------------------------------------


def test_a_photograph_of_nothing_says_what_to_do():
    blank = np.full((600, 900, 3), 240, dtype=np.uint8)
    with pytest.raises(measure.ReadError) as caught:
        measure.read(blank)
    assert "chart" in str(caught.value)


def test_a_chart_too_small_in_frame_says_so():
    photo = np.full((1200, 1600, 3), 250, dtype=np.uint8)
    photo[590:610, 790:810] = 0
    with pytest.raises(measure.ReadError):
        measure.read(photo)


def test_a_dark_rectangle_that_is_not_the_chart_is_refused():
    """Better to say no than to sample forty colours out of a doormat."""
    photo = np.full((1400, 2000, 3), 246, dtype=np.uint8)
    photo[300:1100, 400:1600] = (30, 60, 40)
    with pytest.raises(measure.ReadError) as caught:
        measure.read(photo)
    assert "not the chart" in str(caught.value)


def test_a_greyscale_photograph_is_refused():
    with pytest.raises(measure.ReadError):
        measure.read(np.zeros((100, 100), dtype=np.uint8))


# -- storage -----------------------------------------------------------------


def test_a_profile_survives_a_round_trip_through_json():
    profile = measure.read(photograph())
    restored = measure.Profile.from_dict(json.loads(profile.to_json()))
    assert np.allclose(restored.gamma, profile.gamma, atol=1e-3)
    assert restored.predict((100, 100, 100)) == profile.predict((100, 100, 100))
    assert restored.residuals()["held_out"] == profile.residuals()["held_out"]


# -- with a white underbase --------------------------------------------------


def test_the_white_base_chart_prints_its_corners_instead_of_leaving_holes():
    """The corners are the reference, so they have to be the substrate.

    With a white base, a corner left as a hole is bare glass while every patch
    beside it sits on white ink. Dividing one by the other is not a measurement
    of the ink; it is two different substrates cancelling badly.
    """
    layout = measure.CHART
    raster = measure.chart(substrate="white")
    scale = raster.width / layout.width_mm
    for index in measure.glass_cells(layout):
        x, y, w, h = layout.cell(index)
        px = lambda v: int((layout.margin_mm + v) * scale)  # noqa: E731
        pixel = raster.rgba[px(y + h / 2), px(x + w / 2)]
        assert tuple(int(v) for v in pixel) == (255, 255, 255, 255)


def test_the_two_charts_differ_only_in_their_corners():
    plain = measure.chart().rgba.astype(int)
    based = measure.chart(substrate="white").rgba.astype(int)
    differ = np.any(plain != based, axis=2)
    layout = measure.CHART
    scale = plain.shape[1] / layout.width_mm
    for index in range(layout.columns * layout.rows):
        x, y, w, h = layout.cell(index)
        row = int((layout.margin_mm + y + h / 2) * scale)
        column = int((layout.margin_mm + x + w / 2) * scale)
        expected = index in measure.glass_cells(layout)
        assert bool(differ[row, column]) is expected, index


def test_clear_and_opaque_glass_measure_the_same_grid():
    """Same patches, different caption.

    The two want different photographs, and by the time a printed plate is in
    your hand there is nothing else to tell them apart — so the caption says
    which, and only the caption differs.
    """
    layout = measure.CHART
    clear = measure.chart(substrate="transparent").rgba
    opaque = measure.chart(substrate="opaque").rgba
    scale = clear.shape[1] / layout.width_mm
    grid = slice(0, int((layout.margin_mm + layout.frame_h_mm) * scale))
    assert np.array_equal(clear[grid], opaque[grid]), "the artwork must be identical"
    assert not np.array_equal(clear, opaque), "the caption must not be"

    assert "opaque" in measure.Profile.REFLECTIVE
    assert "transparent" not in measure.Profile.REFLECTIVE


def test_an_old_profile_still_loads():
    """"glass" was the earlier name for what is now "transparent"."""
    profile = measure.read(photograph())
    data = json.loads(profile.to_json())
    data["substrate"] = "glass"
    assert measure.Profile.from_dict(data).substrate == "transparent"


def test_a_white_base_profile_ignores_the_glass_colour():
    """An opaque base has covered the glass. Tinting the prediction anyway
    would answer a question about a substrate that is no longer there."""
    profile = measure.read(photograph(), substrate="white")
    assert profile.substrate == "white"
    assert profile.predict((128, 64, 32), GREEN) == profile.predict((128, 64, 32), AMBER)

    on_glass = measure.read(photograph())
    assert on_glass.predict((128, 64, 32), GREEN) != on_glass.predict((128, 64, 32), AMBER)


def test_the_substrate_survives_a_round_trip():
    profile = measure.read(photograph(), substrate="white")
    restored = measure.Profile.from_dict(json.loads(profile.to_json()))
    assert restored.substrate == "white"
    assert restored.predict((90, 90, 90)) == profile.predict((90, 90, 90))


# -- refusing a photograph that measures the wrong thing ---------------------


def veiled(fraction: float, **kwargs) -> np.ndarray:
    """A photograph with light reflecting off the front of the plate.

    An additive fraction of the illumination reaching the camera without having
    crossed the ink — which is what a window or lamp reflected in the glass
    does, and what a front-lit shot of a *transparency* does everywhere at once.
    """
    photo = photograph(**kwargs).astype(np.float64) / 255.0
    return np.clip(
        measure.encode(measure.linear(photo) * (1 - fraction) + fraction) * 255.0, 0, 255
    ).astype(np.uint8)


def test_a_veiled_photograph_is_refused_and_says_why():
    """The failure that produced a complete, plausible, meaningless profile.

    A real plate came back with solid black passing 58% of bare glass, which no
    ink can do. Nothing in the fit noticed: it returned a tone curve, a density
    and a glass colour, all of them fiction.
    """
    for fraction in (0.2, 0.35):
        with pytest.raises(measure.ReadError) as caught:
            measure.read(veiled(fraction))
        message = str(caught.value)
        assert "black" in message and "reflect" in message, (fraction, message)

    # Heavier than that and the chart cannot even be found — the veil washes out
    # the contrast the detector needs. Still refused, just for a blunter reason.
    with pytest.raises(measure.ReadError):
        measure.read(veiled(0.6))


def test_a_gentle_veil_still_reads():
    """The check has to clear a single pass of black on glass, which is only
    dark grey — refusing that would refuse every honest transparency."""
    assert measure.read(veiled(0.06)).residuals()["held_out"] < 12.0


def test_corners_disagreeing_in_colour_are_refused():
    """All four print the same substrate, so they cannot differ in hue."""
    photo = photograph().astype(np.float64) / 255.0
    linear = measure.linear(photo)
    linear[: linear.shape[0] // 2] *= np.array([1.0, 1.0, 3.5])  # a window in the top half
    photo = np.clip(measure.encode(linear) * 255.0, 0, 255).astype(np.uint8)
    with pytest.raises(measure.ReadError) as caught:
        measure.read(photo)
    assert "colour" in str(caught.value)


def test_a_smooth_falloff_is_survivable_and_reported():
    """A steep but smooth gradient is what the four corners are for.

    Worth pinning down, because it was the first guess at what wrecked a real
    plate and it turned out to be wrong: a 31x falloff across the frame still
    read to 4 levels. Whatever spoils a reading, it is not smoothness.
    """
    photo = photograph().astype(np.float64) / 255.0
    linear = measure.linear(photo)
    ys, xs = np.mgrid[0 : linear.shape[0], 0 : linear.shape[1]]
    radius = np.hypot(xs - linear.shape[1] * 0.15, ys - linear.shape[0] * 0.5)
    linear *= (1.0 / (1.0 + (radius / (linear.shape[1] * 0.25)) ** 2))[:, :, None]
    profile = measure.read(np.clip(measure.encode(linear) * 255.0, 0, 255).astype(np.uint8))
    assert profile.residuals()["held_out"] < 8.0
    assert "fell off" in profile.note, "it still has to say the light was bad"


def test_a_plate_whose_repeat_disagrees_is_refused():
    """The chart prints one colour twice; the two are a checksum on the read.

    Broken here with a *local* hot spot rather than a gradient, because that is
    what a bilinear reference cannot absorb — and, on the evidence, what a lamp
    close to a plate actually does.
    """
    layout = measure.CHART
    photo = photograph().astype(np.float64) / 255.0
    linear = measure.linear(photo)

    # Find where one of the two repeated patches landed, and shine on it.
    corners = set(measure.glass_cells(layout))
    order = [i for i in range(layout.columns * layout.rows) if i not in corners]
    cell = order[measure.REPEAT[1]]
    centre = layout.centres()[cell]
    frame = np.array(
        [[0.0, 0.0], [layout.frame_w_mm, 0.0],
         [layout.frame_w_mm, layout.frame_h_mm], [0.0, layout.frame_h_mm]]
    )
    luma = linear @ np.array([0.2126, 0.7152, 0.0722])
    quad = measure._locate(linear, luma, layout, frame)
    spot = measure._project(measure._homography(frame, quad), centre[None, :])[0]

    ys, xs = np.mgrid[0 : linear.shape[0], 0 : linear.shape[1]]
    hot = np.exp(-(((xs - spot[0]) ** 2 + (ys - spot[1]) ** 2) / (2 * 90.0**2)))
    linear *= (1.0 + 1.2 * hot)[:, :, None]

    with pytest.raises(measure.ReadError) as caught:
        measure.read(np.clip(measure.encode(linear) * 255.0, 0, 255).astype(np.uint8))
    assert "levels apart" in str(caught.value)


def test_the_held_out_patches_are_spread_over_the_whole_chart():
    """Clustered in one corner they measure the lamp, not the model.

    Left consecutive at the end of the list they all fell in the bottom-right
    of the grid, and on a real plate with an uneven lamp the fit was fine while
    the held-out error read 40 levels.
    """
    layout = measure.CHART
    corners = set(measure.glass_cells(layout))
    cells, colour = [], 0
    for index in range(layout.columns * layout.rows):
        if index in corners:
            continue
        if colour in measure.HELD_OUT:
            cells.append(index)
        colour += 1
    assert len(cells) == len(measure.HELD_OUT)
    assert len({index // layout.columns for index in cells}) == layout.rows, "every row"
    columns = {index % layout.columns for index in cells}
    assert len(columns) == len(cells), f"no two in the same column, got {sorted(columns)}"


def test_the_illumination_surface_passes_through_the_corner_cells():
    """The ring bends the surface; the corners still set its level.

    Fitted to the ring and the corners together, the surface came out 10-13%
    below the four corner cells it was meant to interpolate — because the ring
    sits outside the frame, further from the middle of the lens and of the
    lightbox, and therefore lower. Every patch then read that much too bright,
    which on a substrate with little tonal range is the difference between a
    grey and no ink at all.
    """
    layout = measure.CHART
    photo = photograph(plate_margin_mm=14.0)
    rgb = measure.linear(photo.astype(np.float64) / 255.0)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    frame = np.array(
        [[0.0, 0.0], [layout.frame_w_mm, 0.0],
         [layout.frame_w_mm, layout.frame_h_mm], [0.0, layout.frame_h_mm]]
    )
    corners = measure._locate(rgb, luma, layout, frame)
    centres = layout.centres()
    spots = measure._project(measure._homography(frame, corners), centres)
    radius = 0.3 * layout.patch_mm * np.linalg.norm(spots[1] - spots[0]) / layout.pitch_mm
    samples = np.array([measure._sample(rgb, spot, radius) for spot in spots])
    cells = list(measure.glass_cells(layout))

    surface = measure._illumination(
        rgb, layout, frame, corners, centres, samples[cells], radius
    )
    for cell in cells:
        off = np.abs(surface[cell] / np.maximum(samples[cell], 1e-9) - 1.0).max()
        assert off < 0.05, f"cell {cell} off by {off:.1%}"


def test_a_stray_sample_of_the_lightbox_cannot_bend_the_surface():
    """Beyond the plate's edge is the lamp, which is bright and neutral.

    Rejecting outliers by brightness catches the caption and the chipped edges,
    which are dark, and misses this entirely — so the filter goes by hue, which
    the substrate has and the lightbox does not.
    """
    layout = measure.CHART
    photo = photograph(plate_margin_mm=2.5)  # ring reaches past the plate
    rgb = measure.linear(photo.astype(np.float64) / 255.0)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    frame = np.array(
        [[0.0, 0.0], [layout.frame_w_mm, 0.0],
         [layout.frame_w_mm, layout.frame_h_mm], [0.0, layout.frame_h_mm]]
    )
    profile = measure.read(photo)
    assert profile.residuals()["held_out"] < 4.0, profile.residuals()
