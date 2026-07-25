from __future__ import annotations

import json

import numpy as np
import pytest

from glassprint import ComposeSpec, ExportSpec, Placement, Raster, compose, export
from glassprint.recolor import ColorSpec


def test_overlay_is_clipped_to_the_base_shape(base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec(keep="remove the white background"))
    layer_alpha = result.overlay_layer.alpha_f

    # Outside the panel there must be nothing at all.
    assert layer_alpha[0:20, 0:20].max() == 0.0
    assert layer_alpha[280:300, 380:400].max() == 0.0
    # Inside it there must be pattern.
    assert layer_alpha[100:200, 100:300].max() > 0.7
    # And the composite keeps the panel's own silhouette.
    assert result.composite.alpha_f[0, 0] == 0.0


def test_composite_matches_base_outside_the_shape(base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    outside = result.shape_mask < 0.01
    assert np.array_equal(
        result.composite.rgba[outside], base_shape.rgba[outside]
    )


def test_unclipped_overlay_spills_past_the_shape(base_shape, pattern_art):
    clipped = compose(base_shape, pattern_art, ComposeSpec())
    spec = ComposeSpec(clip_to_shape=False, target="full", placement=Placement(fit="tile"))
    loose = compose(base_shape, pattern_art, spec)

    outside = base_shape.alpha_f < 0.01
    assert clipped.overlay_layer.alpha_f[outside].max() == 0.0
    assert loose.overlay_layer.alpha_f[outside].max() > 0.5


def test_opaque_base_falls_back_to_largest_region(base_opaque, pattern_art):
    result = compose(base_opaque, pattern_art, ComposeSpec(target="alpha"))
    assert any("no transparency" in note for note in result.notes)
    # The blue ellipse is the target, not the white surround.
    assert result.shape_mask[150, 200] > 0.5
    assert result.shape_mask[5, 5] < 0.5


def test_target_rect_limits_the_area(base_shape, pattern_art):
    spec = ComposeSpec(target="rect", target_rect=(0.25, 0.25, 0.75, 0.75))
    result = compose(base_shape, pattern_art, spec)
    left, top, right, bottom = result.box
    assert left == pytest.approx(100, abs=2)
    assert right == pytest.approx(300, abs=2)


def test_pattern_language_is_detected_and_reported(base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    summary = result.summary()
    assert summary["pattern"]["is_pattern"] is True
    assert summary["pattern"]["suggested_fit"] == "tile"
    assert json.dumps(summary)  # the summary must be JSON-serialisable for the UI


def test_recolour_flows_through_the_pipeline(base_shape, pattern_art):
    plain = compose(base_shape, pattern_art, ComposeSpec())
    tinted = compose(
        base_shape,
        pattern_art,
        ComposeSpec(color=ColorSpec(mode="tint", color="#1166ff")),
    )
    assert not np.array_equal(plain.composite.rgba, tinted.composite.rgba)


def test_blend_modes_differ(base_shape, pattern_art):
    normal = compose(base_shape, pattern_art, ComposeSpec(blend="normal"))
    multiply = compose(base_shape, pattern_art, ComposeSpec(blend="multiply"))
    assert not np.array_equal(normal.composite.rgba, multiply.composite.rgba)


def test_invalid_modes_are_rejected(base_shape, pattern_art):
    with pytest.raises(ValueError):
        compose(base_shape, pattern_art, ComposeSpec(target="nope"))
    with pytest.raises(ValueError):
        compose(base_shape, pattern_art, ComposeSpec(blend="nope"))
    with pytest.raises(ValueError):
        compose(base_shape, pattern_art, ComposeSpec(placement=Placement(fit="nope")))


# --- export -----------------------------------------------------------------

def test_export_writes_composite_and_overlay(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    manifest = export(result, tmp_path, ExportSpec(formats=["png"], basename="panel"))

    names = {entry["file"] for entry in manifest}
    assert names == {"panel_composite.png", "panel_overlay.png"}
    for entry in manifest:
        assert (tmp_path / entry["file"]).exists()


def test_export_includes_the_base_format(tmp_path, base_shape, pattern_art):
    base = Raster(base_shape.rgba, dpi=base_shape.dpi, source_format="tiff", name="panel")
    result = compose(base, pattern_art, ComposeSpec())
    manifest = export(result, tmp_path, ExportSpec(formats=["jpg"], targets=["composite"]))

    formats = {entry["format"] for entry in manifest}
    assert formats == {"tiff", "jpg"}


def test_overlay_always_gets_an_alpha_capable_format(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    manifest = export(result, tmp_path, ExportSpec(formats=["jpg"], targets=["overlay"]))

    assert [entry["format"] for entry in manifest] == ["png"]
    assert Raster.open(manifest[0]["path"]).has_alpha


def test_export_honours_physical_width(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    manifest = export(
        result,
        tmp_path,
        ExportSpec(formats=["png"], targets=["composite"], dpi=600, width_mm=120.0),
    )
    entry = manifest[0]
    assert entry["pixels"][0] == 2835        # 120mm at 600dpi
    assert entry["dpi"] == 600
    assert entry["size_mm"][0] == pytest.approx(120.0, abs=0.2)
    assert Raster.open(entry["path"]).dpi == (600.0, 600.0)


def test_export_masks_on_request(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    manifest = export(
        result,
        tmp_path,
        ExportSpec(formats=["png"], targets=["shape-mask", "cutout-mask"]),
    )
    assert {entry["target"] for entry in manifest} == {"shape-mask", "cutout-mask"}


def test_export_rejects_both_dimensions(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    with pytest.raises(ValueError):
        export(result, tmp_path, ExportSpec(width_mm=100, height_mm=50))


def test_export_rejects_unknown_format(tmp_path, base_shape, pattern_art):
    result = compose(base_shape, pattern_art, ComposeSpec())
    with pytest.raises(ValueError):
        export(result, tmp_path, ExportSpec(formats=["heic"]))
