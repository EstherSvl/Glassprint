"""The transport-free interface.

This is the path the iPad build takes: no HTTP, no filesystem, just JSON in and
JSON out. It is worth testing directly, because in the browser there is no
server to blame when something goes wrong.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

from glassprint import bridge
from glassprint.raster import Raster


def _call(method: str, payload: dict | None = None) -> dict:
    return json.loads(bridge.handle(method, json.dumps(payload or {})))


def _b64(raster: Raster) -> str:
    return base64.b64encode(raster.to_png_bytes()).decode("ascii")


def _load(base_shape, pattern_art) -> None:
    bridge._BRIDGE = bridge.Bridge()
    _call("upload", {"role": "base", "filename": "panel.png", "data": _b64(base_shape)})
    _call("upload", {"role": "overlay", "filename": "dots.png", "data": _b64(pattern_art)})


def test_capabilities_reports_a_version_without_any_images_loaded():
    payload = _call("capabilities")["ok"]
    assert payload["version"]
    assert "png" in payload["read_formats"]


def test_an_upload_round_trips_through_base64(base_shape):
    bridge._BRIDGE = bridge.Bridge()
    payload = _call("upload", {"role": "base", "filename": "panel.png", "data": _b64(base_shape)})
    image = payload["ok"]["image"]
    assert (image["width"], image["height"]) == base_shape.size
    assert image["name"] == "panel"
    assert image["thumb"].startswith("data:image/png;base64,")


def test_preview_returns_pictures_and_measurements(base_shape, pattern_art):
    _load(base_shape, pattern_art)
    payload = _call("preview", {"keep": "remove the white background"})["ok"]

    assert payload["images"]["composite"].startswith("data:image/png;base64,")
    assert payload["images"]["overlay"].startswith("data:image/png;base64,")
    assert payload["summary"]["base_size"] == list(base_shape.size)


def test_export_hands_back_real_file_bytes(base_shape, pattern_art):
    _load(base_shape, pattern_art)
    payload = _call(
        "export",
        {
            "keep": "remove the white background",
            "export": {"formats": ["png"], "targets": ["composite", "overlay"], "basename": "panel"},
        },
    )["ok"]

    files = {entry["file"]: entry for entry in payload["files"]}
    assert set(files) == {"panel_composite.png", "panel_overlay.png"}

    # The bytes have to be a decodable PNG at the size the manifest claims, or
    # the download the browser saves is junk.
    entry = files["panel_composite.png"]
    decoded = Image.open(io.BytesIO(base64.b64decode(entry["data"])))
    assert list(decoded.size) == entry["pixels"] == list(base_shape.size)


def test_a_missing_overlay_is_an_error_message_not_a_traceback(base_shape):
    bridge._BRIDGE = bridge.Bridge()
    _call("upload", {"role": "base", "filename": "panel.png", "data": _b64(base_shape)})
    payload = _call("preview", {})
    assert "overlay" in payload["error"]
    assert payload["status"] == 400


def test_an_unreadable_file_names_the_formats_that_would_work():
    bridge._BRIDGE = bridge.Bridge()
    payload = _call(
        "upload", {"role": "base", "filename": "sketch.svg", "data": base64.b64encode(b"<svg/>").decode()}
    )
    assert ".svg" in payload["error"]
    assert "PNG" in payload["error"]


def test_an_unknown_method_does_not_raise():
    assert "error" in _call("nonsense")


def test_glaze_passes_come_out_as_separate_downloads(base_shape, pattern_art):
    _load(base_shape, pattern_art)
    payload = _call(
        "export",
        {
            "keep": "remove the white background",
            "glaze": {"enabled": True, "glass": "#bcd6c4", "colours": 3},
            "export": {"formats": ["png"], "targets": ["glaze-layers", "print-order"], "basename": "panel"},
        },
    )["ok"]

    names = [entry["file"] for entry in payload["files"]]
    assert any(name.startswith("panel_pass01-") for name in names)
    assert "panel_print-order.md" in names

    sheet = base64.b64decode(
        next(e["data"] for e in payload["files"] if e["file"] == "panel_print-order.md")
    ).decode("utf-8")
    assert "white underbase off" in sheet


def test_the_bridge_holds_no_state_between_sessions(base_shape, pattern_art):
    _load(base_shape, pattern_art)
    first = _call("preview", {})["ok"]["summary"]
    bridge._BRIDGE = bridge.Bridge()
    assert "error" in _call("preview", {})
    assert first["base_size"] == list(base_shape.size)


def test_a_direct_bridge_is_independent_of_the_module_level_one(base_shape, pattern_art):
    """The server keeps one Bridge per browser tab; they must not share images."""
    one, two = bridge.Bridge(), bridge.Bridge()
    one.load_image("base", base_shape.to_png_bytes(), "panel.png")
    assert "base" not in two.images

    two.load_image("base", pattern_art.to_png_bytes(), "dots.png")
    assert one.images["base"].size != two.images["base"].size


def test_preview_measurements_describe_the_full_size_file_not_the_preview(base_shape, pattern_art):
    """Previews run downscaled; the numbers on screen must be the real ones."""
    big = Raster(
        np.array(base_shape.to_pil().resize((2000, 1500), Image.LANCZOS), dtype=np.uint8),
        dpi=(300.0, 300.0),
    )
    bridge._BRIDGE = bridge.Bridge()
    _call("upload", {"role": "base", "filename": "big.png", "data": _b64(big)})
    _call("upload", {"role": "overlay", "filename": "dots.png", "data": _b64(pattern_art)})

    summary = _call("preview", {"preview_size": 400, "keep": "remove the white background"})["ok"]["summary"]
    assert summary["base_size"] == [2000, 1500]
    assert summary["preview_scale"] < 1.0
    # The shape box is reported against the full-size canvas, so it can exceed
    # the preview's own dimensions.
    assert max(summary["shape_box"]) > 400
