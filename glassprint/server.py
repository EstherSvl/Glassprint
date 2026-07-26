"""Local web interface.

Runs on 127.0.0.1 (or the whole local network with ``--lan``, so a tablet can
reach it) and holds uploaded images in memory for the life of the process.

All the actual work lives in :mod:`glassprint.bridge`; this module only maps it
onto HTTP and onto a folder of files. The browser build calls the same bridge
without any of this.
"""

from __future__ import annotations

import secrets

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .bridge import Bridge, BridgeError, capabilities

WEB_DIR = Path(__file__).parent / "web"


class Session(Bridge):
    """A bridge plus the files it has written, so they can be downloaded."""

    def __init__(self) -> None:
        super().__init__()
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


def _http(exc: BridgeError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.message)


def create_app() -> FastAPI:
    app = FastAPI(title="glassprint", version=__version__, docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/capabilities")
    def capabilities_route() -> JSONResponse:
        return JSONResponse(
            {
                **capabilities(),
                "writes_files": True,
                "default_export_dir": str(Path.cwd() / "glassprint-exports"),
            }
        )

    @app.post("/api/upload")
    async def upload(
        file: UploadFile = File(...),
        role: str = Form(...),
        session_id: str = Form(""),
    ) -> JSONResponse:
        data = await file.read()
        sid, session = _session(session_id or None, create=True)
        try:
            result = session.load_image(role, data, file.filename or "")
        except BridgeError as exc:
            raise _http(exc)
        return JSONResponse({"session_id": sid, **result})

    @app.post("/api/preview")
    def preview(payload: dict[str, Any]) -> JSONResponse:
        sid, session = _session(payload.get("session_id"))
        try:
            return JSONResponse({"session_id": sid, **session.preview(payload)})
        except BridgeError as exc:
            raise _http(exc)

    # Calibration is the same handful of Bridge methods whichever shell is in
    # front of it, so it gets one route rather than four near-identical ones.
    CALLABLE = {"chart", "calibrate", "load_profile", "colour"}

    @app.post("/api/call/{method}")
    def call(method: str, payload: dict[str, Any]) -> JSONResponse:
        if method not in CALLABLE:
            raise HTTPException(status_code=404, detail=f"unknown method {method!r}")
        sid, session = _session(payload.get("session_id"), create=True)
        try:
            return JSONResponse({"session_id": sid, **getattr(session, method)(payload)})
        except BridgeError as exc:
            raise _http(exc)

    @app.post("/api/export")
    def export_files(payload: dict[str, Any]) -> JSONResponse:
        sid, session = _session(payload.get("session_id"))
        export_data = payload.get("export") or {}
        out_dir = Path(
            str(export_data.get("directory") or (Path.cwd() / "glassprint-exports"))
        ).expanduser()

        try:
            rendered = session.render_export(payload)
        except BridgeError as exc:
            raise _http(exc)

        manifest = []
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            for entry in rendered["files"]:
                path = out_dir / entry["file"]
                path.write_bytes(entry.pop("data"))
                token = secrets.token_urlsafe(8)
                session.exports[token] = path
                manifest.append(
                    {"path": str(path), **entry, "download": f"/api/download/{sid}/{token}"}
                )
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Could not write to {out_dir}: {exc}")

        return JSONResponse(
            {
                "session_id": sid,
                "directory": str(out_dir),
                "files": manifest,
                "summary": rendered["summary"],
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
