from __future__ import annotations

import io

import pytest
from PIL import Image

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from glassprint.cli import app as cli_app  # noqa: E402
from glassprint.server import create_app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _png_bytes(raster) -> bytes:
    return raster.to_png_bytes()


def _upload(client, raster, role, session_id=""):
    return client.post(
        "/api/upload",
        files={"file": (f"{role}.png", io.BytesIO(_png_bytes(raster)), "image/png")},
        data={"role": role, "session_id": session_id},
    )


def test_index_and_capabilities(client):
    assert client.get("/").status_code == 200
    payload = client.get("/api/capabilities").json()
    assert "backends" in payload and "version" in payload
    assert "png" in payload["read_formats"]


def test_upload_reports_print_size(client, base_shape):
    response = _upload(client, base_shape, "base")
    assert response.status_code == 200
    body = response.json()
    assert body["image"]["width"] == 400
    assert body["image"]["has_alpha"] is True
    assert body["image"]["thumb"].startswith("data:image/png;base64,")
    assert body["session_id"]


def test_upload_rejects_a_bad_role(client, base_shape):
    response = client.post(
        "/api/upload",
        files={"file": ("x.png", io.BytesIO(_png_bytes(base_shape)), "image/png")},
        data={"role": "sideways"},
    )
    assert response.status_code == 400


def test_upload_rejects_unreadable_files(client):
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", io.BytesIO(b"not an image"), "text/plain")},
        data={"role": "base"},
    )
    assert response.status_code == 400


def test_preview_round_trip(client, base_shape, pattern_art):
    session_id = _upload(client, base_shape, "base").json()["session_id"]
    _upload(client, pattern_art, "overlay", session_id)

    response = client.post(
        "/api/preview",
        json={
            "session_id": session_id,
            "keep": "remove the white background",
            "placement": {"fit": "tile", "repeat_across": 6},
            "include_masks": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["images"]) == {"composite", "overlay", "shape_mask", "cutout_mask"}
    assert body["summary"]["pattern"]["is_pattern"] is True
    assert body["summary"]["base_size"] == [400, 300]


def test_preview_needs_both_images(client, base_shape):
    session_id = _upload(client, base_shape, "base").json()["session_id"]
    response = client.post("/api/preview", json={"session_id": session_id})
    assert response.status_code == 400


def test_preview_rejects_unknown_session(client):
    assert client.post("/api/preview", json={"session_id": "nope"}).status_code == 404


def test_export_writes_files_and_serves_them(client, base_shape, pattern_art, tmp_path):
    session_id = _upload(client, base_shape, "base").json()["session_id"]
    _upload(client, pattern_art, "overlay", session_id)

    response = client.post(
        "/api/export",
        json={
            "session_id": session_id,
            "keep": "remove the background",
            "export": {
                "formats": ["png", "jpg"],
                "targets": ["composite", "overlay"],
                "directory": str(tmp_path),
                "basename": "panel",
                "dpi": 300,
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["directory"] == str(tmp_path)

    names = {entry["file"] for entry in body["files"]}
    assert "panel_composite.png" in names
    assert "panel_composite.jpg" in names
    assert "panel_overlay.png" in names

    download = client.get(body["files"][0]["download"])
    assert download.status_code == 200
    assert Image.open(io.BytesIO(download.content)).size == (400, 300)


def test_export_surfaces_bad_options(client, base_shape, pattern_art, tmp_path):
    session_id = _upload(client, base_shape, "base").json()["session_id"]
    _upload(client, pattern_art, "overlay", session_id)

    response = client.post(
        "/api/export",
        json={
            "session_id": session_id,
            "export": {"formats": ["heic"], "directory": str(tmp_path)},
        },
    )
    assert response.status_code == 400
    assert "heic" in response.json()["detail"]


# --- CLI --------------------------------------------------------------------

def test_cli_version():
    result = CliRunner().invoke(cli_app, ["version"])
    assert result.exit_code == 0
    assert "glassprint" in result.stdout


def test_cli_inspect(tmp_path, pattern_art):
    path = pattern_art.save(tmp_path / "pattern.png")
    result = CliRunner().invoke(cli_app, ["inspect", str(path)])
    assert result.exit_code == 0
    assert "repeating pattern" in result.stdout
    assert "print size" in result.stdout


def test_cli_compose_writes_files(tmp_path, base_shape, pattern_art):
    base_path = base_shape.save(tmp_path / "panel.png")
    art_path = pattern_art.save(tmp_path / "pattern.png")
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        cli_app,
        [
            "compose", str(base_path), str(art_path),
            "--out", str(out_dir),
            "--keep", "remove the white background",
            "--fit", "tile", "--repeats", "5",
            "--color", "#0044ff",
            "--width-mm", "120",
            "--format", "png,jpg",
        ],
    )
    assert result.exit_code == 0, result.stdout
    written = {p.name for p in out_dir.iterdir()}
    assert "panel_composite.png" in written
    assert "panel_composite.jpg" in written
    assert "panel_overlay.png" in written


def test_cli_mask_writes_cutout(tmp_path, pattern_art):
    art_path = pattern_art.save(tmp_path / "pattern.png")
    out = tmp_path / "cutout.png"
    result = CliRunner().invoke(
        cli_app, ["mask", str(art_path), "--keep", "keep the red", "--out", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
