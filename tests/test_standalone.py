"""The single-file browser build.

``dist/glassprint.html`` is committed so it can be hosted without a build step,
which means it can quietly fall behind the sources it was built from. These
tests rebuild it and compare.
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_standalone  # noqa: E402

from glassprint import bridge  # noqa: E402
from glassprint.export import bundle  # noqa: E402


@pytest.fixture(scope="module")
def built() -> str:
    return build_standalone.build()


def test_the_committed_build_matches_the_sources(built):
    committed = build_standalone.OUTPUT
    assert committed.exists(), "run: python tools/build_standalone.py"
    assert committed.read_text(encoding="utf-8") == built, (
        "dist/glassprint.html is out of date — run python tools/build_standalone.py"
    )


def test_the_build_loads_nothing_from_a_server(built):
    """A tablet opening this file has no local server, so every asset must be inline."""
    external = re.findall(r'(?:src|href)="(/[^"]*)"', built)
    assert external == [], f"still loading {external} from a server that is not there"

    # HttpBackend comes along for the ride but must never be the one chosen.
    assert "window.GlassprintBackend = backend;" in built
    assert "PyodideBackend" in built.split("window.GlassprintBackend = backend;")[0][-400:]


def test_every_module_the_bridge_imports_is_embedded(built):
    sources = json.loads(re.search(r"window\.GLASSPRINT_PYTHON = (\{.*?\});\n", built, re.S).group(1))

    shipped = {name for name in sources}
    expected = {
        path.name for path in (ROOT / "glassprint").glob("*.py")
    } - build_standalone.SKIP
    assert shipped == expected

    # The two skipped modules need typer and fastapi, neither of which exists in
    # the browser; if they crept in, the import would fail on the tablet.
    assert "cli.py" not in shipped and "server.py" not in shipped
    for name in ("bridge.py", "__init__.py", "compose.py", "glaze.py"):
        assert name in shipped


def test_the_embedded_python_is_the_current_source(built):
    sources = json.loads(re.search(r"window\.GLASSPRINT_PYTHON = (\{.*?\});\n", built, re.S).group(1))
    on_disk = (ROOT / "glassprint" / "bridge.py").read_text(encoding="utf-8")
    assert sources["bridge.py"] == on_disk


def test_the_page_carries_its_own_styles_and_scripts(built):
    assert "<style>" in built
    assert "GlassprintBackends" in built  # backend.js
    assert "window.glassprintInit" in built  # app.js
    assert "PyodideBackend" in built


def test_the_export_bundle_is_a_readable_zip(base_shape, pattern_art):
    """On a tablet the whole export arrives as one file, so it has to open."""
    bridge._BRIDGE = bridge.Bridge()
    for role, art in (("base", base_shape), ("overlay", pattern_art)):
        bridge.handle(
            "upload",
            json.dumps(
                {
                    "role": role,
                    "filename": f"{role}.png",
                    "data": base64.b64encode(art.to_png_bytes()).decode("ascii"),
                }
            ),
        )

    payload = json.loads(
        bridge.handle(
            "export",
            json.dumps(
                {
                    "keep": "remove the white background",
                    "export": {
                        "formats": ["png"],
                        "targets": ["composite", "overlay"],
                        "basename": "panel",
                        "bundle": True,
                    },
                }
            ),
        )
    )["ok"]

    assert payload["bundle"]["file"] == "panel.zip"
    # The per-file bytes are dropped when bundling: sending both would double
    # the size of the reply for no reason.
    assert all("data" not in entry for entry in payload["files"])

    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload["bundle"]["data"])))
    assert archive.namelist() == ["panel/panel_composite.png", "panel/panel_overlay.png"]
    assert archive.testzip() is None
    assert len(archive.read("panel/panel_composite.png")) > 100


def test_bundling_keeps_files_in_their_own_folder():
    """Unzipping on a tablet drops the contents wherever you are standing."""
    data = bundle([{"file": "a.png", "data": b"one"}, {"file": "b.png", "data": b"two"}], "job")
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert names == ["job/a.png", "job/b.png"]
