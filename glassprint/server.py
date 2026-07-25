"""Local web interface.

Runs on 127.0.0.1 and holds uploaded images in memory for the life of the
process. Previews are computed on downscaled copies (with the DPI scaled to
match, so physical repeat sizes stay honest) and returned as data URLs
alongside a JSON summary.
"""

from __future__ import annotations

import base64
import secrets
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .compose import ComposeSpec, compose
from .export import ExportSpec, export
from .fade import Fade
from .pattern import Placement
from .raster import READ_SUFFIXES, Raster
from .recolor import ColorSpec
from .segment import Backends

WEB_DIR = Path(__file__).parent / "web"
PREVIEW_MAX_SIDE = 1000
THUMB_MAX_SIDE = 320
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class Session:
    def __init__(self) -> None:
        self.images: dict[str, Raster] = {}
        self.exports: dict[str, Path] = {}


SESSIONS: dict[str, Session] = {}


def _session(session_id: str | None, *, create: bool = False) -> tuple[str, Session]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    if not create:
        raise HTTPException(status_code=404, detail="Session not found — reload the page.")
    new_id = secrets.token_urlsafe(12)
    SESSIONS[new_id] = Session()
    return new_id, SESSIONS[new_id]


def _preview_copy(raster: Raster, max_side: int) -> Raster:
    """Downscale for speed, scaling DPI so millimetre maths still holds."""
    longest = max(raster.width, raster.height)
    if longest <= max_side:
        return raster
    factor = max_side / longest
    smaller = raster.resized(int(round(raster.width * factor)), int(round(raster.height * factor)))
    dpi_x, dpi_y = raster.effective_dpi
    return replace(smaller, dpi=(dpi_x * factor, dpi_y * factor))


def _data_url(raster: Raster) -> str:
    return "data:image/png;base64," + base64.b64encode(raster.to_png_bytes()).decode("ascii")


def _describe(raster: Raster) -> dict[str, Any]:
    width_mm, height_mm = raster.size_mm
    return {
        "width": raster.width,
        "height": raster.height,
        "dpi": [round(v, 2) for v in raster.effective_dpi],
        "dpi_tagged": raster.dpi is not None,
        "size_mm": [round(width_mm, 1), round(height_mm, 1)],
        "has_alpha": raster.has_alpha,
        "format": raster.source_format,
        "name": raster.name,
        "thumb": _data_url(_preview_copy(raster, THUMB_MAX_SIDE)),
    }


def _float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _build_spec(payload: dict[str, Any]) -> ComposeSpec:
    placement_data = payload.get("placement") or {}
    color_data = payload.get("color") or {}
    fade_data = payload.get("fade") or {}

    rect = payload.get("target_rect")
    target_rect = None
    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        target_rect = tuple(float(v) for v in rect)  # type: ignore[assignment]

    return ComposeSpec(
        keep=str(payload.get("keep") or ""),
        tolerance=_float(payload.get("tolerance"), 1.0),
        use_claude=bool(payload.get("use_claude")),
        target=str(payload.get("target") or "alpha"),
        target_describe=str(payload.get("target_describe") or ""),
        target_rect=target_rect,
        clip_to_shape=bool(payload.get("clip_to_shape", True)),
        shape_grow=_float(payload.get("shape_grow"), 0.0),
        shape_feather=_float(payload.get("shape_feather"), 0.0),
        edge_feather=_float(payload.get("edge_feather"), 0.0),
        opacity=_float(payload.get("opacity"), 1.0),
        blend=str(payload.get("blend") or "normal"),
        placement=Placement(
            fit=str(placement_data.get("fit") or "auto"),
            repeat_across=_optional_float(placement_data.get("repeat_across")),
            repeat_mm=_optional_float(placement_data.get("repeat_mm")),
            scale=_float(placement_data.get("scale"), 1.0),
            rotation=_float(placement_data.get("rotation"), 0.0),
            offset_x=_float(placement_data.get("offset_x"), 0.0),
            offset_y=_float(placement_data.get("offset_y"), 0.0),
            mirror=str(placement_data.get("mirror") or "auto"),
            flip_h=bool(placement_data.get("flip_h")),
            flip_v=bool(placement_data.get("flip_v")),
        ),
        color=ColorSpec(
            mode=str(color_data.get("mode") or "none"),
            color=color_data.get("color") or None,
            color2=color_data.get("color2") or None,
            from_color=color_data.get("from_color") or None,
            strength=_float(color_data.get("strength"), 1.0),
            tolerance=_float(payload.get("tolerance"), 1.0),
            hue_shift=_float(color_data.get("hue_shift"), 0.0),
            saturation=_float(color_data.get("saturation"), 1.0),
            brightness=_float(color_data.get("brightness"), 1.0),
            contrast=_float(color_data.get("contrast"), 1.0),
            invert=bool(color_data.get("invert")),
        ),
        fade=Fade(
            mode=str(fade_data.get("mode") or "none"),
            what=str(fade_data.get("what") or ""),
            angle=_float(fade_data.get("angle"), 90.0),
            center_x=_float(fade_data.get("center_x"), 0.5),
            center_y=_float(fade_data.get("center_y"), 0.5),
            start=_float(fade_data.get("start"), 0.0),
            end=_float(fade_data.get("end"), 1.0),
            curve=_float(fade_data.get("curve"), 1.0),
            min_alpha=_float(fade_data.get("min_alpha"), 0.0),
            max_alpha=_float(fade_data.get("max_alpha"), 1.0),
            per_element=bool(fade_data.get("per_element")),
            dissolve=_float(fade_data.get("dissolve"), 0.0),
            seed=int(_float(fade_data.get("seed"), 0.0)),
            invert=bool(fade_data.get("invert")),
            cutoff=_float(fade_data.get("cutoff"), 0.0),
        ),
    )


def create_app() -> FastAPI:
    app = FastAPI(title="glassprint", version=__version__, docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/capabilities")
    def capabilities() -> JSONResponse:
        probe = Backends.probe()
        return JSONResponse(
            {
                "version": __version__,
                "backends": probe,
                "semantic_selection": probe["clipseg"],
                "subject_cutout": probe["rembg"],
                "claude": probe["anthropic"],
                "default_export_dir": str(Path.cwd() / "glassprint-exports"),
                "read_formats": sorted(s.lstrip(".") for s in READ_SUFFIXES),
            }
        )

    @app.post("/api/upload")
    async def upload(
        file: UploadFile = File(...),
        role: str = Form(...),
        session_id: str = Form(""),
    ) -> JSONResponse:
        if role not in {"base", "overlay"}:
            raise HTTPException(status_code=400, detail="role must be 'base' or 'overlay'")

        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File is larger than 200 MB.")

        suffix = Path(file.filename or "").suffix.lower()
        if suffix and suffix not in READ_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot read {suffix} files. Try PNG, TIFF, JPEG, WebP or PSD.",
            )
        try:
            raster = Raster.from_bytes(data, name=Path(file.filename or role).stem)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read that image ({exc}).")

        sid, session = _session(session_id or None, create=True)
        session.images[role] = raster
        return JSONResponse({"session_id": sid, "role": role, "image": _describe(raster)})

    @app.post("/api/preview")
    def preview(payload: dict[str, Any]) -> JSONResponse:
        sid, session = _session(payload.get("session_id"))
        base = session.images.get("base")
        overlay = session.images.get("overlay")
        if base is None or overlay is None:
            raise HTTPException(status_code=400, detail="Load both a base image and an overlay first.")

        max_side = int(payload.get("preview_size") or PREVIEW_MAX_SIDE)
        spec = _build_spec(payload)
        backends = Backends(allow_models=not payload.get("offline_only"))

        preview_base = _preview_copy(base, max_side)
        result = compose(
            preview_base,
            _preview_copy(overlay, max_side),
            spec,
            backends,
        )

        images = {
            "composite": _data_url(result.composite),
            "overlay": _data_url(result.overlay_layer),
        }
        if payload.get("include_masks"):
            from .export import _mask_to_raster  # local import: only needed for debugging views

            images["shape_mask"] = _data_url(_mask_to_raster(result.shape_mask, result.composite))
            images["cutout_mask"] = _data_url(_mask_to_raster(result.cutout_mask, result.composite))

        # The preview runs on a downscaled copy; report measurements against the
        # real file so the numbers on screen match what gets exported.
        scale = base.width / preview_base.width
        summary = result.summary()
        summary["base_size"] = list(base.size)
        summary["base_dpi"] = [round(v, 2) for v in base.effective_dpi]
        summary["base_size_mm"] = [round(v, 1) for v in base.size_mm]
        summary["shape_box"] = [int(round(v * scale)) for v in result.box]
        summary["preview_scale"] = round(1 / scale, 4)
        return JSONResponse({"session_id": sid, "images": images, "summary": summary})

    @app.post("/api/export")
    def export_files(payload: dict[str, Any]) -> JSONResponse:
        sid, session = _session(payload.get("session_id"))
        base = session.images.get("base")
        overlay = session.images.get("overlay")
        if base is None or overlay is None:
            raise HTTPException(status_code=400, detail="Load both a base image and an overlay first.")

        spec = _build_spec(payload)
        backends = Backends(allow_models=not payload.get("offline_only"))
        result = compose(base, overlay, spec, backends)

        export_data = payload.get("export") or {}
        out_dir = Path(str(export_data.get("directory") or (Path.cwd() / "glassprint-exports"))).expanduser()
        try:
            export_spec = ExportSpec(
                formats=[str(f) for f in (export_data.get("formats") or ["png"])],
                targets=[str(t) for t in (export_data.get("targets") or ["composite", "overlay"])],
                include_base_format=bool(export_data.get("include_base_format", True)),
                dpi=_optional_float(export_data.get("dpi")),
                width_mm=_optional_float(export_data.get("width_mm")),
                height_mm=_optional_float(export_data.get("height_mm")),
                quality=int(_float(export_data.get("quality"), 95)),
                background=str(export_data.get("background") or "#ffffff"),
                basename=str(export_data.get("basename") or base.name or "glassprint"),
            ).validated()
            manifest = export(result, out_dir, export_spec)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Could not write to {out_dir}: {exc}")

        for entry in manifest:
            token = secrets.token_urlsafe(8)
            session.exports[token] = Path(entry["path"])
            entry["download"] = f"/api/download/{sid}/{token}"

        return JSONResponse(
            {
                "session_id": sid,
                "directory": str(out_dir),
                "files": manifest,
                "summary": result.summary(),
            }
        )

    @app.get("/api/download/{session_id}/{token}")
    def download(session_id: str, token: str) -> FileResponse:
        _, session = _session(session_id)
        path = session.exports.get(token)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="File not found.")
        return FileResponse(path, filename=path.name)

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    return app


app = create_app()
