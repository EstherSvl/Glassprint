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


# -- calibration ------------------------------------------------------------


def _photo_b64() -> str:
    """A JPEG of the printed chart, base64'd, as the browser would send it."""
    from test_measure import photograph

    buffer = io.BytesIO()
    Image.fromarray(photograph()).save(buffer, "JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_chart_comes_back_printable_with_instructions():
    data = _call("chart", {})["ok"]
    assert data["file"].endswith(".png")
    assert data["size_mm"][0] < 145 and data["size_mm"][1] < 50
    assert data["patches"] == 44
    assert any("100%" in line for line in data["instructions"])
    png = base64.b64decode(data["data"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_calibrating_reports_something_a_person_can_read():
    report = _call("calibrate", {"data": _photo_b64()})["ok"]["report"]
    assert report["error_levels"] < 5
    # The whole justification for spending a plate of glass.
    assert report["uncalibrated_error_levels"] > 10 * report["error_levels"]
    assert len(report["gamma"]) == 3
    assert report["glass"].startswith("#")


def test_a_profile_makes_the_preview_say_it_is_calibrated(base_shape, pattern_art):
    _load(base_shape, pattern_art)
    _call("calibrate", {"data": _photo_b64()})["ok"]
    summary = _call("preview", {"simulate": {"glass": "#2ea76f"}})["ok"]["summary"]
    assert summary["calibrated"] is True


def test_the_glaze_preview_changes_once_calibrated(base_shape, pattern_art):
    """If calibration did not move the picture it was not doing anything."""
    _load(base_shape, pattern_art)
    payload = {"simulate": {"glass": "#2ea76f"}}
    before = _call("preview", payload)["ok"]["images"]["glaze"]
    _call("calibrate", {"data": _photo_b64()})["ok"]
    after = _call("preview", payload)["ok"]["images"]["glaze"]
    assert before != after


def test_asking_what_to_send_the_printer():
    _call("calibrate", {"data": _photo_b64()})["ok"]
    answer = _call("colour", {"wanted": "#1e6b4a", "glass": "#2ea76f"})["ok"]
    assert answer["reachable"]
    # On dark green glass the request has to be much lighter than the result,
    # because the glass has already done most of the darkening.
    asked = int(answer["ask_for"][1:], 16)
    wanted = int(answer["wanted"][1:], 16)
    assert asked > wanted


def test_asking_for_something_the_glass_forbids_explains_itself():
    _call("calibrate", {"data": _photo_b64()})["ok"]
    answer = _call("colour", {"wanted": "#ffffff", "glass": "#2ea76f"})["ok"]
    assert answer["reachable"] is False
    assert "subtracts" in answer["note"]


def test_colour_without_a_profile_says_so():
    bridge._BRIDGE = bridge.Bridge()
    error = _call("colour", {"wanted": "#123456"})
    assert "error" in error and "chart" in error["error"]


def test_a_profile_survives_being_saved_and_reloaded():
    first = _call("calibrate", {"data": _photo_b64()})["ok"]
    bridge._BRIDGE = bridge.Bridge()
    again = _call("load_profile", {"profile": first["profile"]})["ok"]
    assert again["report"]["gamma"] == first["report"]["gamma"]


def test_rubbish_instead_of_a_profile_is_refused():
    error = _call("load_profile", {"profile": {"gamma": "nonsense"}})
    assert "error" in error


def test_a_photograph_with_no_chart_in_it_says_what_to_do():
    buffer = io.BytesIO()
    Image.new("RGB", (900, 600), (238, 238, 238)).save(buffer, "PNG")
    error = _call("calibrate", {"data": base64.b64encode(buffer.getvalue()).decode()})
    assert "error" in error and "chart" in error["error"]


def test_each_substrate_comes_back_as_its_own_file():
    files = {
        s: _call("chart", {"substrate": s})["ok"]["file"]
        for s in ("transparent", "opaque", "white")
    }
    assert len(set(files.values())) == 3, files
    # Three plates that look alike in a drawer a week later, so each names
    # itself — in the filename and, more usefully, printed on the plate.
    for substrate, name in files.items():
        assert substrate in name


def test_the_lighting_advice_follows_how_light_reaches_the_eye():
    """Not "glass or white base" — how many times the light crosses the ink.

    Through clear glass, once. Off an opaque ground — dark glass or a white
    base — in, reflect, back out: twice. Swap the two and every colour reads as
    its own square or square root, and the fit absorbs it without complaining.
    """
    # Checked as properties rather than phrases: the wording is allowed to
    # improve, the advice is not allowed to invert.
    clear = " ".join(_call("chart", {"substrate": "transparent"})["ok"]["instructions"]).lower()
    assert "screen" in clear or "behind" in clear or "hold it up" in clear
    assert "twice" in clear, "name the double-pass trap, do not merely avoid it"
    assert "lit from the front" not in clear

    for reflective in ("opaque", "white"):
        lines = " ".join(_call("chart", {"substrate": reflective})["ok"]["instructions"]).lower()
        assert "front" in lines, reflective
        assert "hold it up" not in lines, reflective


def test_the_substrate_is_recorded_and_bad_ones_are_refused():
    for substrate in ("transparent", "opaque", "white"):
        report = _call("calibrate", {"data": _photo_b64(), "substrate": substrate})["ok"]["report"]
        assert report["substrate"] == substrate
    # A typo must not quietly become the default: the three want different
    # photographs, so guessing wrong yields a confident profile, not an error.
    assert "error" in _call("chart", {"substrate": "white-base"})


def test_opaque_glass_still_takes_its_colour_from_the_glass():
    """A white base hides the glass; opaque glass *is* the glass."""
    _call("calibrate", {"data": _photo_b64(), "substrate": "opaque"})
    on_green = _call("colour", {"wanted": "#1e6b4a", "glass": "#2ea76f"})["ok"]
    on_amber = _call("colour", {"wanted": "#1e6b4a", "glass": "#c4852c"})["ok"]
    assert on_green["ask_for"] != on_amber["ask_for"]

    _call("calibrate", {"data": _photo_b64(), "substrate": "white"})
    white_green = _call("colour", {"wanted": "#1e6b4a", "glass": "#2ea76f"})["ok"]
    white_amber = _call("colour", {"wanted": "#1e6b4a", "glass": "#c4852c"})["ok"]
    assert white_green["ask_for"] == white_amber["ask_for"]
