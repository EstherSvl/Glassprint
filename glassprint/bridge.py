"""The work the interface asks for, with no opinion about how it was asked.

The web interface used to talk straight to FastAPI. That is fine on a laptop
and useless on an iPad, where there is no terminal to start a server from. So
everything between "here is an image and some settings" and "here are the
pictures and files" lives here instead, and there are two thin shells around
it: :mod:`glassprint.server` for the local HTTP server, and a browser build
that imports this module directly and calls :func:`handle`.

Nothing in here touches the disk or the network.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from . import __version__
from .colors import parse_color, to_hex
from .compose import ComposeResult, ComposeSpec, GlazeSpec, LayerSpec, compose
from .export import ExportSpec, bundle, render
from .fade import Fade
from .measure import Profile
from .pattern import Placement
from .raster import READ_SUFFIXES, Raster
from .recolor import ColorSpec
from .segment import Backends
from .simulate import glaze

PREVIEW_MAX_SIDE = 1000
THUMB_MAX_SIDE = 320
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class BridgeError(Exception):
    """Something the person at the keyboard can fix, phrased for them."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# -- helpers ----------------------------------------------------------------


def preview_copy(raster: Raster, max_side: int) -> Raster:
    """Downscale for speed, scaling DPI so millimetre maths still holds."""
    longest = max(raster.width, raster.height)
    if longest <= max_side:
        return raster
    factor = max_side / longest
    smaller = raster.resized(int(round(raster.width * factor)), int(round(raster.height * factor)))
    dpi_x, dpi_y = raster.effective_dpi
    return replace(smaller, dpi=(dpi_x * factor, dpi_y * factor))


def data_url(raster: Raster) -> str:
    return "data:image/png;base64," + base64.b64encode(raster.to_png_bytes()).decode("ascii")


def describe(raster: Raster) -> dict[str, Any]:
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
        "thumb": data_url(preview_copy(raster, THUMB_MAX_SIDE)),
    }


#: What the ink sits on, and therefore how the chart is drawn and lit.
SUBSTRATES = ("transparent", "opaque", "white")


def _substrate(payload: dict[str, Any]) -> str:
    """Read the substrate, refusing anything not on the list.

    Silently falling back to a default would be worse than an error: the three
    differ in how the plate must be *photographed*, and a wrong choice there
    produces a confident profile rather than a complaint.
    """
    value = str(payload.get("substrate") or "transparent")
    if value not in SUBSTRATES:
        raise BridgeError(f"substrate must be one of {', '.join(SUBSTRATES)} — got {value!r}")
    return value


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


def build_placement(data: dict[str, Any]) -> Placement:
    return Placement(
        fit=str(data.get("fit") or "auto"),
        repeat_across=_optional_float(data.get("repeat_across")),
        repeat_mm=_optional_float(data.get("repeat_mm")),
        scale=_float(data.get("scale"), 1.0),
        rotation=_float(data.get("rotation"), 0.0),
        offset_x=_float(data.get("offset_x"), 0.0),
        offset_y=_float(data.get("offset_y"), 0.0),
        mirror=str(data.get("mirror") or "auto"),
        flip_h=bool(data.get("flip_h")),
        flip_v=bool(data.get("flip_v")),
    )


def build_color(data: dict[str, Any], tolerance: float) -> ColorSpec:
    return ColorSpec(
        mode=str(data.get("mode") or "none"),
        color=data.get("color") or None,
        color2=data.get("color2") or None,
        from_color=data.get("from_color") or None,
        strength=_float(data.get("strength"), 1.0),
        tolerance=tolerance,
        hue_shift=_float(data.get("hue_shift"), 0.0),
        saturation=_float(data.get("saturation"), 1.0),
        brightness=_float(data.get("brightness"), 1.0),
        contrast=_float(data.get("contrast"), 1.0),
        invert=bool(data.get("invert")),
        black_point=_float(data.get("black_point"), 0.0),
    )


def build_fade(data: dict[str, Any]) -> Fade:
    return Fade(
        mode=str(data.get("mode") or "none"),
        what=str(data.get("what") or ""),
        angle=_float(data.get("angle"), 90.0),
        center_x=_float(data.get("center_x"), 0.5),
        center_y=_float(data.get("center_y"), 0.5),
        start=_float(data.get("start"), 0.0),
        end=_float(data.get("end"), 1.0),
        curve=_float(data.get("curve"), 1.0),
        min_alpha=_float(data.get("min_alpha"), 0.0),
        max_alpha=_float(data.get("max_alpha"), 1.0),
        per_element=bool(data.get("per_element")),
        dissolve=_float(data.get("dissolve"), 0.0),
        seed=int(_float(data.get("seed"), 0.0)),
        layers=int(_float(data.get("layers"), 0.0)),
        halftone_mm=_float(data.get("halftone_mm"), 0.0),
        halftone_angle=_float(data.get("halftone_angle"), 45.0),
        carrier=str(data.get("carrier") or "alpha"),
        invert=bool(data.get("invert")),
        cutoff=_float(data.get("cutoff"), 0.0),
    )


def build_layer(data: dict[str, Any], tolerance: float) -> LayerSpec:
    """One overlay's own settings.

    Anything the entry leaves out falls to the same default a single-overlay
    job would have used, so a caller can send ``{"keep": "the leaves"}`` and
    change only that.
    """
    return LayerSpec(
        keep=str(data.get("keep") or ""),
        edge_feather=_float(data.get("edge_feather"), 0.0),
        opacity=_float(data.get("opacity"), 1.0),
        blend=str(data.get("blend") or "normal"),
        placement=build_placement(data.get("placement") or {}),
        color=build_color(data.get("color") or {}, tolerance),
        fade=build_fade(data.get("fade") or {}),
    )


def build_spec(payload: dict[str, Any]) -> ComposeSpec:
    placement_data = payload.get("placement") or {}
    color_data = payload.get("color") or {}
    fade_data = payload.get("fade") or {}
    glaze_data = payload.get("glaze") or {}
    tolerance = _float(payload.get("tolerance"), 1.0)

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
        placement=build_placement(placement_data),
        color=build_color(color_data, tolerance),
        glaze=GlazeSpec(
            enabled=bool(glaze_data.get("enabled")),
            glass=str(glaze_data.get("glass") or "#ffffff"),
            palette=str(glaze_data.get("palette") or ""),
            colours=int(_float(glaze_data.get("colours"), 5.0)),
            max_per_ink=int(_float(glaze_data.get("max_per_ink"), 3.0)),
            max_total=int(_float(glaze_data.get("max_total"), 5.0)),
        ),
        fade=build_fade(fade_data),
        layers=[
            build_layer(entry or {}, tolerance)
            for entry in (payload.get("layers") or [])
        ],
    )


def build_export_spec(payload: dict[str, Any], fallback_name: str) -> ExportSpec:
    data = payload.get("export") or {}
    return ExportSpec(
        formats=[str(f) for f in (data.get("formats") or ["png"])],
        targets=[str(t) for t in (data.get("targets") or ["composite", "overlay"])],
        include_base_format=bool(data.get("include_base_format", True)),
        dpi=_optional_float(data.get("dpi")),
        width_mm=_optional_float(data.get("width_mm")),
        height_mm=_optional_float(data.get("height_mm")),
        quality=int(_float(data.get("quality"), 95)),
        background=str(data.get("background") or "#ffffff"),
        basename=str(data.get("basename") or fallback_name or "glassprint"),
    )


# -- the session ------------------------------------------------------------


#: Overlay slots are ``overlay``, ``overlay2``, ``overlay3`` … The first has no
#: number because it is the only one most jobs use, and renaming it would break
#: every saved preset and every existing caller.
_OVERLAY_ROLE = re.compile(r"^overlay([2-9]|[1-9]\d+)?$")

MAX_OVERLAYS = 8


def overlay_role(index: int) -> str:
    """The role name for the nth overlay, counting from zero."""
    return "overlay" if index == 0 else f"overlay{index + 1}"


class Bridge:
    """One working set: a base image, its overlays, and the settings on top."""

    def __init__(self) -> None:
        self.images: dict[str, Raster] = {}
        #: Set by :meth:`calibrate`, or restored by the caller from storage.
        #: Without one every preview falls back to reading an ink's RGB as its
        #: transmittance, which is the guess this replaces.
        self.profile: Any = None

    # -- input ---------------------------------------------------------

    def load_image(self, role: str, data: bytes, filename: str = "") -> dict[str, Any]:
        if role != "base" and not _OVERLAY_ROLE.match(role):
            raise BridgeError("role must be 'base', 'overlay', or 'overlay2' and up")
        if len(data) > MAX_UPLOAD_BYTES:
            raise BridgeError("File is larger than 200 MB.", status=413)

        suffix = PurePosixPath(filename).suffix.lower()
        if suffix and suffix not in READ_SUFFIXES:
            raise BridgeError(f"Cannot read {suffix} files. Try PNG, TIFF, JPEG, WebP or PSD.")
        try:
            raster = Raster.from_bytes(data, name=PurePosixPath(filename or role).stem)
        except Exception as exc:  # Pillow raises a zoo of exception types
            raise BridgeError(f"Could not read that image ({exc}).")

        self.images[role] = raster
        return {"role": role, "image": describe(raster)}

    def drop_image(self, role: str) -> dict[str, Any]:
        """Forget one loaded image. Removing an overlay closes the gap.

        Slot numbers are positions in a stack, not names: if you drop the second
        of three, the third becomes the second. Leaving a hole would make the
        ``layers`` list line up against nothing.
        """
        if role == "base":
            self.images.pop("base", None)
            return {"role": role, "overlays": self.overlay_names()}
        if not _OVERLAY_ROLE.match(role):
            raise BridgeError("role must be 'base', 'overlay', or 'overlay2' and up")

        kept = [
            raster
            for name, raster in self._overlay_items()
            if name != role
        ]
        for name, _ in self._overlay_items():
            self.images.pop(name, None)
        for index, raster in enumerate(kept):
            self.images[overlay_role(index)] = raster
        return {"role": role, "overlays": self.overlay_names()}

    def _overlay_items(self) -> list[tuple[str, Raster]]:
        """Loaded overlays in slot order, gaps skipped."""
        found = [name for name in self.images if _OVERLAY_ROLE.match(name)]
        found.sort(key=lambda name: int(name[len("overlay"):] or 1))
        return [(name, self.images[name]) for name in found]

    def overlay_names(self) -> list[str]:
        return [name for name, _ in self._overlay_items()]

    def _pair(self) -> tuple[Raster, list[Raster]]:
        base = self.images.get("base")
        overlays = [raster for _, raster in self._overlay_items()]
        if base is None or not overlays:
            raise BridgeError("Load both a base image and an overlay first.")
        return base, overlays

    def _compose(
        self, payload: dict[str, Any], base: Raster, overlays: list[Raster]
    ) -> ComposeResult:
        backends = Backends(allow_models=not payload.get("offline_only"))
        return compose(base, overlays, build_spec(payload), backends)

    # -- output --------------------------------------------------------

    def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        base, overlays = self._pair()
        max_side = int(payload.get("preview_size") or PREVIEW_MAX_SIDE)

        preview_base = preview_copy(base, max_side)
        result = self._compose(
            payload, preview_base, [preview_copy(art, max_side) for art in overlays]
        )

        images = {
            "composite": data_url(result.composite),
            "overlay": data_url(result.overlay_layer),
        }
        if payload.get("include_masks"):
            from .export import _mask_to_raster  # local import: only needed for debugging views

            images["shape_mask"] = data_url(_mask_to_raster(result.shape_mask, result.composite))
            images["cutout_mask"] = data_url(_mask_to_raster(result.cutout_mask, result.composite))

        # Printing without a white underbase is multiplicative, so it needs a
        # real render rather than compositing over a colour swatch.
        if result.glaze_plan is not None and result.coverage is not None:
            from .glaze import render as render_glaze

            glazed = render_glaze(result.glaze_plan, result.coverage, result.glaze_plan.glass)
            rgba = np.dstack([
                np.clip(glazed * 255.0 + 0.5, 0, 255).astype(np.uint8),
                np.full(glazed.shape[:2], 255, dtype=np.uint8),
            ])
            images["glazed"] = data_url(Raster(rgba, dpi=result.composite.dpi))

        simulate = payload.get("simulate") or {}
        summary_extra: dict[str, Any] = {}
        glass = parse_color(simulate.get("glass")) if simulate.get("glass") else None
        if glass:
            images["glaze"] = data_url(
                glaze(
                    result,
                    glass,
                    layers=max(1, int(_float(simulate.get("layers"), 1.0))),
                    layer_map=result.layer_map,
                    profile=self.profile,
                )
            )
            summary_extra["calibrated"] = self.profile is not None

        # The preview runs on a downscaled copy; report measurements against the
        # real file so the numbers on screen match what gets exported.
        scale = base.width / preview_base.width
        summary = result.summary()
        summary["base_size"] = list(base.size)
        summary["base_dpi"] = [round(v, 2) for v in base.effective_dpi]
        summary["base_size_mm"] = [round(v, 1) for v in base.size_mm]
        summary["shape_box"] = [int(round(v * scale)) for v in result.box]
        summary["preview_scale"] = round(1 / scale, 4)
        summary.update(summary_extra)
        return {"images": images, "summary": summary}

    # -- calibration ---------------------------------------------------

    def chart(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The printable colour chart, ready to send to the printer."""
        from .measure import CHART, chart

        dpi = _float(payload.get("dpi"), 600.0)
        substrate = _substrate(payload)
        raster = chart(dpi=dpi, label=str(payload.get("label") or ""), substrate=substrate)

        # The photograph has to be lit the way the finished piece will be seen,
        # because that is what decides how many times the light crosses the ink.
        # Through clear glass it crosses once. Off an opaque ground — dark glass
        # or a white base — it goes in, reflects and comes back, so twice. Get it
        # the wrong way round and every transmittance reads as its own square or
        # its own square root; the fit absorbs that perfectly into a wrong
        # density and reports good residuals either way. It is the one mistake
        # here with no symptom.
        if substrate in Profile.REFLECTIVE:
            surface = "an opaque base" if substrate == "white" else "opaque glass"
            lighting = [
                f"Photograph it flat, lit from the front, on a sheet of white paper — {surface} "
                "has nothing to backlight.",
                "Keep the paper in shot all round the plate: it is the white reference.",
                "Light it from off to one side so the camera sees no reflection of itself.",
            ]
        else:
            lighting = [
                "Best by far: show a plain white screen on a tablet or laptop at full "
                "brightness, lay the plate flat on it, and shoot straight down.",
                "A screen is bright, perfectly even and diffuse — which a window is not. "
                "Sky through a window falls off across the plate and leaves a bright spot "
                "the four corners cannot correct, and that is what spoils most readings.",
                "Turn the room lights off so nothing reflects in the front face.",
                "Leave a little screen showing all round the plate: it is the white reference.",
                "Failing a screen, hold it up to an overcast sky — never lay it on lit paper, "
                "which sends the light through twice and doubles every density.",
            ]
        return {
            "file": f"glassprint-colour-chart-{substrate}.png",
            "data": base64.b64encode(raster.encode(fmt="png", dpi=(dpi, dpi))).decode("ascii"),
            "size_mm": [round(CHART.width_mm, 1), round(CHART.height_mm, 1)],
            "patches": CHART.columns * CHART.rows,
            "substrate": substrate,
            "instructions": [
                "Print at 100%, one pass, "
                + ("with the white base on" if substrate == "white" else "no white base")
                + " — otherwise the same settings you print artwork with.",
                f"Any glass at least {CHART.width_mm:.0f} x {CHART.height_mm:.0f}mm will do.",
                *lighting,
                "No flash, no HDR, and nothing casting a shadow across the plate.",
            ],
        }

    def calibrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Read a photograph of the printed chart and keep the result."""
        from .measure import ReadError, read

        data = base64.b64decode(payload.get("data") or "")
        if not data:
            raise BridgeError("Send a photograph of the printed chart.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise BridgeError("File is larger than 200 MB.", status=413)
        try:
            photo = Raster.from_bytes(data, name="chart")
        except Exception as exc:
            raise BridgeError(f"Could not read that photograph ({exc}).")

        glass = parse_color(payload.get("glass")) if payload.get("glass") else None
        substrate = _substrate(payload)
        try:
            profile = read(photo.rgba[:, :, :3], glass=glass, substrate=substrate)
        except ReadError as exc:
            raise BridgeError(str(exc))

        self.profile = profile
        return {"profile": profile.as_dict(), "report": _profile_report(profile)}

    def load_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Restore a profile the caller saved earlier."""
        from .measure import Profile

        data = payload.get("profile")
        if not isinstance(data, dict):
            raise BridgeError("Send a profile to load.")
        try:
            self.profile = Profile.from_dict(data)
        except (KeyError, ValueError, TypeError) as exc:
            raise BridgeError(f"That is not a glassprint profile ({exc}).")
        return {"profile": self.profile.as_dict(), "report": _profile_report(self.profile)}

    def colour(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Both directions at once: what a request looks like, and what to ask for."""
        if self.profile is None:
            raise BridgeError("Measure a chart first — there is nothing to predict with.")
        glass = parse_color(payload.get("glass")) if payload.get("glass") else None
        answer: dict[str, Any] = {"glass": to_hex(glass or self.profile.glass)}

        if payload.get("requested"):
            requested = parse_color(payload["requested"])
            if requested is None:
                raise BridgeError(f"Cannot read the colour {payload['requested']!r}.")
            answer["requested"] = to_hex(requested)
            answer["looks_like"] = to_hex(self.profile.predict(requested, glass))
        if payload.get("wanted"):
            wanted = parse_color(payload["wanted"])
            if wanted is None:
                raise BridgeError(f"Cannot read the colour {payload['wanted']!r}.")
            ask, reachable = self.profile.request_for(wanted, glass)
            answer["wanted"] = to_hex(wanted)
            answer["ask_for"] = to_hex(ask)
            answer["reachable"] = reachable
            answer["closest"] = to_hex(self.profile.predict(ask, glass))
            if not reachable:
                answer["note"] = (
                    "Brighter than the glass in at least one channel, so no amount of ink "
                    "reaches it — ink only ever subtracts. This is the closest it goes."
                )
        return answer

    def render_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Encode every requested file. The caller decides where they go."""
        base, overlays = self._pair()
        result = self._compose(payload, base, overlays)
        try:
            spec = build_export_spec(payload, base.name or "glassprint").validated()
            files = render(result, spec)
        except ValueError as exc:
            raise BridgeError(str(exc))
        return {"files": files, "summary": result.summary()}


def _profile_report(profile: Any) -> dict[str, Any]:
    """The profile in the terms someone reading it would ask about."""
    import numpy as _np

    residuals = profile.residuals()
    original, repeat = _measure().REPEAT
    noise = None
    if len(profile.measured) > repeat:
        noise = round(
            float(_np.abs(profile.measured[original] - profile.measured[repeat]).mean() * 255), 1
        )
    return {
        "glass": to_hex(profile.glass),
        "substrate": profile.substrate,
        "gamma": [round(float(g), 2) for g in profile.gamma],
        "density": [round(float(profile.crosstalk[i][i]), 2) for i in range(3)],
        # How much each ink absorbs outside its own channel, relative to inside
        # it. Zero would be a perfect ink; the number says how muddy mixtures go.
        "muddiness": round(
            float(
                (_np.abs(profile.crosstalk).sum() - _np.abs(_np.diag(profile.crosstalk)).sum())
                / max(float(_np.abs(_np.diag(profile.crosstalk)).sum()), 1e-6)
            ),
            3,
        ),
        "error_levels": residuals.get("held_out"),
        "uncalibrated_error_levels": residuals.get("naive"),
        "noise_levels": noise,
        "note": profile.note,
    }


def _measure():
    from . import measure

    return measure


def capabilities() -> dict[str, Any]:
    probe = Backends.probe()
    return {
        "version": __version__,
        "backends": probe,
        "semantic_selection": probe["clipseg"],
        "subject_cutout": probe["rembg"],
        "claude": probe["anthropic"],
        "read_formats": sorted(s.lstrip(".") for s in READ_SUFFIXES),
        "calibration": True,
    }


# -- the browser's way in ---------------------------------------------------

_BRIDGE: Bridge | None = None


def handle(method: str, payload_json: str = "{}") -> str:
    """JSON in, JSON out — the whole interface, for callers without HTTP.

    The browser build drives glassprint through this one function. Everything
    crosses as text: image bytes arrive base64-encoded under ``data``, and
    exported files leave the same way, so nothing depends on how a particular
    JavaScript runtime marshals binary buffers.
    """
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = Bridge()

    payload = json.loads(payload_json or "{}")
    try:
        if method == "capabilities":
            result: Any = capabilities()
        elif method == "upload":
            result = _BRIDGE.load_image(
                str(payload.get("role") or ""),
                base64.b64decode(payload.get("data") or ""),
                str(payload.get("filename") or ""),
            )
        elif method == "drop":
            result = _BRIDGE.drop_image(str(payload.get("role") or ""))
        elif method == "preview":
            result = _BRIDGE.preview(payload)
        elif method == "chart":
            result = _BRIDGE.chart(payload)
        elif method == "calibrate":
            result = _BRIDGE.calibrate(payload)
        elif method == "load_profile":
            result = _BRIDGE.load_profile(payload)
        elif method == "colour":
            result = _BRIDGE.colour(payload)
        elif method == "export":
            result = _BRIDGE.render_export(payload)
            files = result["files"]
            if (payload.get("export") or {}).get("bundle"):
                name = str((payload.get("export") or {}).get("basename") or "glassprint")
                result["bundle"] = {
                    "file": f"{name}.zip",
                    "data": base64.b64encode(bundle(files, name)).decode("ascii"),
                }
                result["files"] = [
                    {key: value for key, value in entry.items() if key != "data"}
                    for entry in files
                ]
            else:
                result["files"] = [
                    {**entry, "data": base64.b64encode(entry["data"]).decode("ascii")}
                    for entry in files
                ]
        else:
            raise BridgeError(f"unknown method {method!r}")
    except BridgeError as exc:
        return json.dumps({"error": exc.message, "status": exc.status})
    return json.dumps({"ok": result})
