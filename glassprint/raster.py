"""Image loading, saving and physical-size bookkeeping.

Everything in glassprint moves around as a :class:`Raster`: an RGBA uint8 array
plus the DPI it was authored at. DPI matters here because the end of the
pipeline is a UV printer putting ink on a physical piece of glass -- a pattern
that looks right on screen is the wrong size on the object unless the pixel
grid is tied to millimetres.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

# Procreate and Affinity can hand us big canvases; don't trip the decompression
# bomb guard on a legitimate 12000px artboard.
Image.MAX_IMAGE_PIXELS = None

MM_PER_INCH = 25.4
DEFAULT_DPI = 300.0

READ_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".jpe", ".tif", ".tiff",
    ".webp", ".bmp", ".gif", ".psd", ".psb",
}

#: Extensions we can write, keyed by the canonical format name.
WRITE_FORMATS: dict[str, str] = {
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "tif": "TIFF",
    "tiff": "TIFF",
    "webp": "WEBP",
    "bmp": "BMP",
}

#: Formats that keep an alpha channel. Anything else gets flattened on save.
ALPHA_FORMATS = {"png", "tif", "tiff", "webp"}

#: Formats that can carry a DPI tag.
DPI_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff"}


def normalise_format(fmt: str) -> str:
    fmt = fmt.lower().lstrip(".")
    if fmt not in WRITE_FORMATS:
        raise ValueError(
            f"unsupported output format {fmt!r}; "
            f"choose from {', '.join(sorted(WRITE_FORMATS))}"
        )
    return fmt


def mm_to_px(mm: float, dpi: float) -> int:
    return max(1, int(round(mm / MM_PER_INCH * dpi)))


def px_to_mm(px: float, dpi: float) -> float:
    return px / dpi * MM_PER_INCH


@dataclass
class Raster:
    """An RGBA image with the DPI it was authored at."""

    rgba: np.ndarray  # (H, W, 4) uint8
    dpi: tuple[float, float] | None = None
    source_format: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.rgba.ndim != 3 or self.rgba.shape[2] != 4:
            raise ValueError(f"expected an (H, W, 4) array, got {self.rgba.shape}")
        if self.rgba.dtype != np.uint8:
            self.rgba = np.clip(self.rgba, 0, 255).astype(np.uint8)

    # -- geometry -----------------------------------------------------------

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def effective_dpi(self) -> tuple[float, float]:
        """DPI to use for physical maths, falling back to 300."""
        if self.dpi is None:
            return (DEFAULT_DPI, DEFAULT_DPI)
        return self.dpi

    @property
    def size_mm(self) -> tuple[float, float]:
        dx, dy = self.effective_dpi
        return px_to_mm(self.width, dx), px_to_mm(self.height, dy)

    # -- channels -----------------------------------------------------------

    @property
    def rgb_f(self) -> np.ndarray:
        """RGB as float32 in 0..1."""
        return self.rgba[:, :, :3].astype(np.float32) / 255.0

    @property
    def alpha_f(self) -> np.ndarray:
        """Alpha as float32 in 0..1."""
        return self.rgba[:, :, 3].astype(np.float32) / 255.0

    @property
    def has_alpha(self) -> bool:
        return bool((self.rgba[:, :, 3] < 255).any())

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pil(cls, image: Image.Image, *, name: str | None = None) -> "Raster":
        fmt = (image.format or "").lower() or None
        dpi = _dpi_from_info(image.info)
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return cls(np.array(image, dtype=np.uint8), dpi=dpi, source_format=fmt, name=name)

    @classmethod
    def open(cls, path: str | Path) -> "Raster":
        path = Path(path)
        with Image.open(path) as image:
            image.load()
            raster = cls.from_pil(image, name=path.stem)
        if raster.source_format is None:
            raster.source_format = path.suffix.lower().lstrip(".") or None
        return raster

    @classmethod
    def from_bytes(cls, data: bytes, *, name: str | None = None) -> "Raster":
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return cls.from_pil(image, name=name)

    @classmethod
    def blank(cls, width: int, height: int, dpi: tuple[float, float] | None = None) -> "Raster":
        return cls(np.zeros((height, width, 4), dtype=np.uint8), dpi=dpi)

    # -- transforms ---------------------------------------------------------

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self.rgba, mode="RGBA")

    def resized(self, width: int, height: int) -> "Raster":
        if (width, height) == self.size:
            return self
        img = self.to_pil().resize((max(1, width), max(1, height)), Image.LANCZOS)
        return replace(self, rgba=np.array(img, dtype=np.uint8))

    def scaled_to_fit(self, max_side: int) -> "Raster":
        """Downscale so the longest edge is at most ``max_side`` (never upscales)."""
        longest = max(self.width, self.height)
        if longest <= max_side:
            return self
        factor = max_side / longest
        return self.resized(int(round(self.width * factor)), int(round(self.height * factor)))

    def with_physical_width(self, width_mm: float, dpi: float) -> "Raster":
        """Resample so the image is ``width_mm`` wide when printed at ``dpi``."""
        target_w = mm_to_px(width_mm, dpi)
        target_h = max(1, int(round(target_w * self.height / self.width)))
        out = self.resized(target_w, target_h)
        return replace(out, dpi=(dpi, dpi))

    def with_physical_height(self, height_mm: float, dpi: float) -> "Raster":
        target_h = mm_to_px(height_mm, dpi)
        target_w = max(1, int(round(target_h * self.width / self.height)))
        out = self.resized(target_w, target_h)
        return replace(out, dpi=(dpi, dpi))

    def flattened(self, background: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Composite over a solid colour and return RGB uint8."""
        a = self.alpha_f[:, :, None]
        bg = np.array(background, dtype=np.float32) / 255.0
        rgb = self.rgb_f * a + bg[None, None, :] * (1.0 - a)
        return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # -- output -------------------------------------------------------------

    def encode(
        self,
        *,
        fmt: str = "png",
        dpi: tuple[float, float] | None = None,
        quality: int = 95,
        background: tuple[int, int, int] = (255, 255, 255),
        keep_alpha: bool = True,
    ) -> bytes:
        """Encode to a file's worth of bytes without touching the disk.

        The browser build has no filesystem to write to, so encoding and
        writing are separate steps and both callers share this one.
        """
        fmt = normalise_format(fmt)
        pil_format = WRITE_FORMATS[fmt]
        out_dpi = dpi or self.effective_dpi

        if keep_alpha and fmt in ALPHA_FORMATS:
            image = self.to_pil()
        else:
            image = Image.fromarray(self.flattened(background), mode="RGB")

        params: dict[str, object] = {}
        if fmt in DPI_FORMATS:
            params["dpi"] = (round(out_dpi[0], 3), round(out_dpi[1], 3))
        if pil_format == "JPEG":
            params.update(quality=quality, subsampling=0, optimize=True)
        elif pil_format == "WEBP":
            params.update(quality=quality, method=6)
        elif pil_format == "TIFF":
            params.update(compression="tiff_lzw")
        elif pil_format == "PNG":
            params.update(optimize=True)

        buf = io.BytesIO()
        image.save(buf, format=pil_format, **params)
        return buf.getvalue()

    def save(
        self,
        path: str | Path,
        *,
        fmt: str | None = None,
        dpi: tuple[float, float] | None = None,
        quality: int = 95,
        background: tuple[int, int, int] = (255, 255, 255),
        keep_alpha: bool = True,
    ) -> Path:
        path = Path(path)
        data = self.encode(
            fmt=fmt or path.suffix or "png",
            dpi=dpi,
            quality=quality,
            background=background,
            keep_alpha=keep_alpha,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.to_pil().save(buf, format="PNG")
        return buf.getvalue()


def _dpi_from_info(info: dict) -> tuple[float, float] | None:
    # PNG stores resolution as integer pixels-per-metre, so a file written at
    # 300dpi reads back as 299.9994. Round so round-trips stay stable.
    dpi = info.get("dpi")
    if dpi:
        try:
            x, y = round(float(dpi[0]), 2), round(float(dpi[1]), 2)
        except (TypeError, ValueError, IndexError):
            return None
        if x > 0 and y > 0:
            return (x, y)
    # JPEG without a pHYs-style tag still carries JFIF density.
    density = info.get("jfif_density")
    unit = info.get("jfif_unit")
    if density and unit == 1:  # 1 == dots per inch
        try:
            x, y = float(density[0]), float(density[1])
        except (TypeError, ValueError, IndexError):
            return None
        if x > 0 and y > 0:
            return (x, y)
    return None
