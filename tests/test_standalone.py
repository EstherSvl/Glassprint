"""The single-file browser build.

``docs/index.html`` is committed so it can be hosted without a build step,
which means it can quietly fall behind the sources it was built from. These
tests rebuild it and compare.
"""

from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess
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


@pytest.fixture(scope="module")
def worker_source(built) -> str:
    """The worker is embedded as a JSON string, so decode before reading it."""
    return json.loads(re.search(r"window\.GLASSPRINT_WORKER = (\".*?\");\n", built, re.S).group(1))


def test_the_committed_build_matches_the_sources(built):
    committed = build_standalone.OUTPUT
    assert committed.exists(), "run: python tools/build_standalone.py"
    assert committed.read_text(encoding="utf-8") == built, (
        "docs/index.html is out of date — run python tools/build_standalone.py"
    )


def test_the_build_loads_nothing_from_a_server(built):
    """A tablet opening this file has no local server, so every asset must be inline."""
    external = re.findall(r'(?:src|href)="(/[^"]*)"', built)
    assert external == [], f"still loading {external} from a server that is not there"

    # HttpBackend comes along for the ride but must never be the one chosen.
    assert "window.GlassprintBackend = backend;" in built
    assert "PyodideBackend" in built.split("window.GlassprintBackend = backend;")[0][-400:]


def test_the_library_check_asks_python_not_the_package_names(built, worker_source):
    """Pyodide keys loaded packages by display name — "Pillow" for "pillow".

    Comparing those names invented a failure that had not happened and stopped
    the page from starting. Whether the libraries are usable is a question only
    an import can answer.
    """
    assert ".loadedPackages" not in built, "back to comparing package names"
    for module in ("numpy", "scipy.ndimage", "PIL.Image, PIL.ImageFilter"):
        assert f'"{module}"' in worker_source


def test_python_runs_in_a_worker_so_the_page_can_keep_painting(built, worker_source):
    """A synchronous render on the page's own thread freezes the whole tab.

    That is how a slow start became indistinguishable from a hung one: the
    clock meant to prove it was alive could not repaint while Python ran.
    """
    assert "window.GLASSPRINT_WORKER" in built
    assert "new Worker(" in built
    # Python is only ever touched inside the worker.
    assert "runPython" in worker_source
    page = built.replace(json.dumps(worker_source), "")
    assert "runPython" not in page, "the page still calls Python on its own thread"


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


def test_the_pages_javascript_actually_parses(built, tmp_path):
    """The build assembles JavaScript inside Python strings.

    Python reads escapes first given half a chance: a `\\n` meant for the
    browser becomes a real newline, `\\b` becomes a backspace, and the result is
    a page that fails to parse with nothing on screen to say why. Every test
    here passed while the build was doing exactly that.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    # Control characters have no business in source and are the fingerprint of
    # an escape Python ate on the way through.
    stray = {ch for ch in built if ord(ch) < 32 and ch not in "\n\r\t"}
    assert not stray, f"mangled escape produced {stray!r}"

    scripts = re.findall(r"<script>(.*?)</script>", built, re.S)
    assert scripts, "no scripts in the build"
    for index, source in enumerate(scripts):
        path = tmp_path / f"chunk{index}.js"
        path.write_text(source, encoding="utf-8")
        done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert done.returncode == 0, f"script {index} does not parse:\n{done.stderr}"


def test_the_worker_javascript_actually_parses(worker_source, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    path = tmp_path / "worker.js"
    path.write_text(worker_source, encoding="utf-8")
    done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_hidden_attribute_actually_hides(built):
    """The browser's own [hidden] rule loses to any author rule setting display.

    Several here set it, so hiding an element from script did nothing at all:
    the splash sat over a tool that had already finished starting, and the
    folder field showed in a build that has no folders to write to.
    """
    assert re.search(r"\[hidden\][^{]*\{[^}]*display:\s*none\s*!important", built), (
        "nothing in the stylesheet makes the hidden attribute stick"
    )


def test_the_readout_judges_its_own_noise():
    """A profile off a textured or badly lit plate must say so.

    The repeated patch is the one number that measures the measurement. Left as
    a bare figure it reads like any other statistic; a reader with no baseline
    cannot tell 0.4 from 9. So the built page has to carry the thresholds.
    """
    source = (ROOT / "glassprint" / "web" / "app.js").read_text()
    assert "noise_levels" in source
    for verdict in ("clean", "usable", "too noisy to trust"):
        assert verdict in source, verdict
    assert "louder than the ink" in source, "a noisy read must fail loudly, not pass quietly"


def test_notes_and_warnings_are_styled_apart():
    """A fallback that worked must not wear the colour of a fault."""
    app = (ROOT / "glassprint" / "web" / "app.js").read_text()
    css = (ROOT / "glassprint" / "web" / "style.css").read_text()
    assert 'class="warn-note"' in app and 'class="note"' in app
    assert ".readout .warn-note" in css and ".readout .note" in css
    accent = css[css.index(".readout .warn-note") : css.index(".readout .warn-note") + 90]
    plain = css[css.index(".readout .note {") : css.index(".readout .note {") + 90]
    assert "var(--accent)" in accent
    assert "var(--accent)" not in plain, "notes must not share the warning colour"
