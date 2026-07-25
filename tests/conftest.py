from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from glassprint import Raster


def _rgba(image: Image.Image, dpi: tuple[float, float] | None = None) -> Raster:
    return Raster(np.array(image.convert("RGBA"), dtype=np.uint8), dpi=dpi)


@pytest.fixture
def base_shape() -> Raster:
    """A transparent canvas with one opaque rounded panel — a Procreate-style export."""
    image = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 40, 340, 260), radius=30, fill=(240, 240, 240, 255))
    return _rgba(image, dpi=(300.0, 300.0))


@pytest.fixture
def base_opaque() -> Raster:
    """A flat white canvas with a solid blue shape and no transparency."""
    image = Image.new("RGBA", (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((100, 60, 300, 240), fill=(40, 90, 200, 255))
    return _rgba(image)


@pytest.fixture
def pattern_art() -> Raster:
    """A repeating dot pattern: red dots and green leaves on white."""
    image = Image.new("RGBA", (120, 120), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(3):
        for col in range(3):
            cx, cy = 20 + col * 40, 20 + row * 40
            draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(220, 30, 30, 255))
            draw.ellipse((cx + 6, cy + 6, cx + 16, cy + 16), fill=(30, 150, 60, 255))
    return _rgba(image, dpi=(300.0, 300.0))


@pytest.fixture
def motif_art() -> Raster:
    """A single centred motif on white."""
    image = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 50, 150, 150), fill=(200, 40, 40, 255))
    return _rgba(image)
