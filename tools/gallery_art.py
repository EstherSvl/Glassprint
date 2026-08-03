"""Test artwork and test objects, drawn rather than photographed.

A gallery is only worth looking at if the pictures in it can fail the way real
ones do. Flat shapes on flat white pass everything, which is why the tests that
used them stayed green through problems you could see immediately on a photo.

So each of these carries one property that breaks something:

* pale subjects on a pale ground, where "remove the white background" has to
  decide between a white petal and the white behind it;
* soft edges and cast shadows, so nothing in the image is exactly the
  background colour;
* backgrounds that are not flat — a gradient, a little grain;
* silhouettes that are not rectangles, so fitting artwork "into the shape"
  means something other than filling its bounding box;
* an alpha hole in the middle of a shape, which is a lens cutout and not a
  region to fill.

Everything is deterministic: same seed, same pixels, so a gallery from today
can be compared against one from a month ago.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from glassprint.raster import Raster

SEED = 20260803


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def _raster(image: Image.Image, dpi: float = 300.0) -> Raster:
    return Raster(np.array(image.convert("RGBA"), dtype=np.uint8), dpi=(dpi, dpi))


def _grain(image: Image.Image, amount: float = 4.0, offset: int = 0) -> Image.Image:
    """A little sensor noise. Nothing in a photograph is exactly one value."""
    array = np.array(image, dtype=np.float32)
    noise = _rng(offset).normal(0.0, amount, array.shape[:2])[:, :, None]
    array[:, :, :3] = np.clip(array[:, :, :3] + noise, 0, 255)
    return Image.fromarray(array.astype(np.uint8), "RGBA")


def _soft_shadow(size: tuple[int, int], shape_mask: Image.Image, blur: float = 14.0,
                 offset: tuple[int, int] = (8, 12), strength: float = 0.35) -> Image.Image:
    """The grey a subject casts onto the paper it was photographed on."""
    shadow = Image.new("L", size, 0)
    shadow.paste(shape_mask, offset)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    faded = Image.eval(shadow, lambda v: int(v * strength))
    return faded


def _vertical_gradient(size: tuple[int, int], top: tuple[int, int, int],
                       bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    array = (1 - ramp) * np.array(top, dtype=np.float32) + ramp * np.array(bottom, dtype=np.float32)
    array = np.repeat(array, width, axis=1)
    rgba = np.dstack([array.astype(np.uint8), np.full((height, width), 255, np.uint8)])
    return Image.fromarray(rgba, "RGBA")


# -- the objects the artwork goes onto --------------------------------------


def base_plate(size: int = 600) -> Raster:
    """A round plate on transparency — the ordinary Procreate export."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = size * 0.06
    draw.ellipse((margin, margin, size - margin, size - margin), fill=(243, 241, 236, 255))
    draw.ellipse(
        (margin * 2.4, margin * 2.4, size - margin * 2.4, size - margin * 2.4),
        outline=(226, 222, 214, 255), width=max(2, size // 200),
    )
    return _raster(_grain(image, 2.0, 1))


def base_vase(width: int = 460, height: int = 640) -> Raster:
    """A silhouette that is nothing like its own bounding box.

    Fitting artwork "to the shape" here has to mean something other than
    filling the rectangle around it, and a pattern told to tile has to stay
    inside a neck that is a fifth of the width of the body.
    """
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    centre = width / 2
    for y in range(height):
        t = y / (height - 1)
        # neck at the top, shoulder, belly, foot
        if t < 0.16:
            half = width * 0.11
        elif t < 0.30:
            half = width * (0.11 + (t - 0.16) / 0.14 * 0.28)
        elif t < 0.72:
            half = width * (0.39 - abs(t - 0.50) * 0.14)
        else:
            half = width * (0.37 - (t - 0.72) / 0.28 * 0.19)
        draw.line((centre - half, y, centre + half, y), fill=255)

    body = _vertical_gradient((width, height), (238, 236, 231), (214, 210, 202))
    image.paste(body, (0, 0), mask)
    return _raster(_grain(image, 2.0, 2))


def base_coaster(size: int = 520) -> Raster:
    """A rounded square. The easy case, kept as the control."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).rounded_rectangle(
        (size * 0.08, size * 0.08, size * 0.92, size * 0.92),
        radius=size * 0.11, fill=(246, 244, 240, 255),
    )
    return _raster(image)


def base_phone_case(width: int = 400, height: int = 800) -> Raster:
    """A case with a lens cutout — a hole in the alpha, not a place to print.

    Worth having because filling holes is the right move for a ragged
    silhouette and the wrong one here: printing over the camera is a scrap.
    """
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width, height), radius=width * 0.14, fill=(30, 32, 38, 255))
    # the lens island, punched back out to transparent
    draw.rounded_rectangle(
        (width * 0.10, height * 0.05, width * 0.52, height * 0.26),
        radius=width * 0.08, fill=(0, 0, 0, 0),
    )
    return _raster(image)


def base_wood_tile(width: int = 640, height: int = 440) -> Raster:
    """An opaque photo with no alpha at all, so there is no shape to fill.

    This is the export people actually have when they photograph the thing
    instead of drawing it, and it is the path where the tool has to find the
    object itself rather than being handed it.
    """
    rng = _rng(3)
    grain = rng.normal(0, 1, (height, width))
    rings = np.sin(np.linspace(0, 26, width))[None, :] * 7
    tone = np.clip(168 + rings + grain * 5, 0, 255)
    array = np.dstack([tone * 1.0, tone * 0.80, tone * 0.58]).astype(np.uint8)
    image = Image.fromarray(
        np.dstack([array, np.full((height, width), 255, np.uint8)]), "RGBA"
    )
    # a pale ceramic tile sitting on the wood
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (width * 0.18, height * 0.14, width * 0.82, height * 0.86),
        radius=18, fill=(240, 238, 233, 255),
    )
    return _raster(_grain(image, 2.5, 4))


# -- the artwork ------------------------------------------------------------


def _petal(draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float, ry: float,
           angle: float, fill: tuple[int, int, int, int]) -> None:
    petal = Image.new("RGBA", (int(rx * 2) + 4, int(ry * 2) + 4), (0, 0, 0, 0))
    ImageDraw.Draw(petal).ellipse((2, 2, rx * 2, ry * 2), fill=fill)
    petal = petal.rotate(angle, expand=True, resample=Image.BICUBIC)
    draw._image.alpha_composite(petal, (int(cx - petal.width / 2), int(cy - petal.height / 2)))


def _orchid_head(size: int, pale: bool) -> Image.Image:
    """One bloom on transparency, soft-edged, shaded rather than flat."""
    head = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(head)
    draw._image = head

    petal = (250, 247, 250, 255) if pale else (206, 122, 178, 255)
    inner = (243, 236, 243, 255) if pale else (186, 92, 158, 255)

    centre = size / 2
    for angle in (90, 162, 234, 306, 18):
        rad = np.deg2rad(angle)
        cx = centre + np.cos(rad) * size * 0.20
        cy = centre - np.sin(rad) * size * 0.20
        _petal(draw, cx, cy, size * 0.155, size * 0.24, angle - 90, petal)

    # the lip, and a throat that is the only saturated thing in a white orchid
    _petal(draw, centre, centre + size * 0.10, size * 0.13, size * 0.16, 0, inner)
    draw.ellipse(
        (centre - size * 0.055, centre - size * 0.02, centre + size * 0.055, centre + size * 0.10),
        fill=(214, 152, 44, 255) if pale else (232, 196, 74, 255),
    )
    # veining, so the petals are not one flat value
    for angle in (90, 162, 234, 306, 18):
        rad = np.deg2rad(angle)
        draw.line(
            (centre, centre,
             centre + np.cos(rad) * size * 0.32, centre - np.sin(rad) * size * 0.32),
            fill=(inner[0], inner[1], inner[2], 90), width=max(1, size // 90),
        )

    return head.filter(ImageFilter.GaussianBlur(size / 260))


def overlay_orchid_on_white(size: int = 460, pale: bool = True) -> Raster:
    """The case that fails: a near-white subject photographed on white paper.

    Every part of this is a problem for a background remover working on colour.
    The petals are within a few percent of the paper; the shadow under the
    stem is neither petal nor paper; and the edges are soft, so there is no
    value at which the subject stops and the ground starts.
    """
    image = Image.new("RGBA", (size, size), (252, 251, 249, 255))

    stem = Image.new("L", (size, size), 0)
    stem_draw = ImageDraw.Draw(stem)
    stem_draw.line((size * 0.5, size * 0.95, size * 0.5, size * 0.42), fill=255,
                   width=max(3, size // 70))
    heads = [(0.50, 0.34, 0.42), (0.30, 0.55, 0.30), (0.71, 0.58, 0.28)]
    for fx, fy, scale in heads:
        head_size = int(size * scale)
        blob = Image.new("L", (head_size, head_size), 0)
        ImageDraw.Draw(blob).ellipse((0, 0, head_size, head_size), fill=255)
        stem.paste(blob, (int(size * fx - head_size / 2), int(size * fy - head_size / 2)), blob)

    shadow = _soft_shadow((size, size), stem, blur=size / 32, offset=(int(size * 0.02), int(size * 0.03)))
    image.paste(Image.new("RGBA", (size, size), (176, 170, 166, 255)), (0, 0), shadow)

    draw = ImageDraw.Draw(image)
    draw.line((size * 0.5, size * 0.95, size * 0.5, size * 0.42),
              fill=(104, 132, 86, 255), width=max(3, size // 70))
    for fx, fy, scale in heads:
        head_size = int(size * scale)
        head = _orchid_head(head_size, pale)
        image.alpha_composite(head, (int(size * fx - head_size / 2), int(size * fy - head_size / 2)))

    return _raster(_grain(image, 3.0, 5))


def overlay_orchid_cutout(size: int = 460) -> Raster:
    """The same flowers already cut out, which is what the tool wants."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((size * 0.5, size * 0.95, size * 0.5, size * 0.42),
              fill=(104, 132, 86, 255), width=max(3, size // 70))
    for fx, fy, scale in [(0.50, 0.34, 0.42), (0.30, 0.55, 0.30), (0.71, 0.58, 0.28)]:
        head_size = int(size * scale)
        image.alpha_composite(
            _orchid_head(head_size, pale=True),
            (int(size * fx - head_size / 2), int(size * fy - head_size / 2)),
        )
    return _raster(image)


def overlay_orchid_on_gradient(size: int = 460) -> Raster:
    """A subject on a ground that is not one colour anywhere.

    Studio backdrops fall off toward the corners. A background remover that
    samples a corner and matches against it has nothing to match.
    """
    image = _vertical_gradient((size, size), (236, 232, 226), (188, 186, 190))
    flower = overlay_orchid_on_white(size, pale=False)
    art = Image.fromarray(flower.rgba, "RGBA")
    # lift the subject off its own white ground and drop it on the gradient
    subject = np.array(art, dtype=np.float32)
    lightness = subject[:, :, :3].mean(axis=2)
    keep = np.clip((246 - lightness) / 26.0, 0, 1)
    subject[:, :, 3] = (keep * 255).astype(np.float32)
    image.alpha_composite(Image.fromarray(subject.astype(np.uint8), "RGBA"))
    return _raster(_grain(image, 3.0, 6))


def overlay_seamless_floral(size: int = 300) -> Raster:
    """A pattern whose opposite edges match, so it tiles without a seam."""
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    step = size // 3
    for row in range(4):
        for col in range(4):
            cx = col * step + (step // 2 if row % 2 else 0)
            cy = row * step
            for wrap_x in (-size, 0, size):
                for wrap_y in (-size, 0, size):
                    x, y = cx + wrap_x, cy + wrap_y
                    if not (-step < x < size + step and -step < y < size + step):
                        continue
                    draw.ellipse((x - step * 0.22, y - step * 0.22, x + step * 0.22, y + step * 0.22),
                                 fill=(198, 74, 96, 255))
                    draw.ellipse((x + step * 0.10, y + step * 0.10, x + step * 0.34, y + step * 0.34),
                                 fill=(86, 138, 92, 255))
    return _raster(image.filter(ImageFilter.GaussianBlur(0.4)))


def overlay_scattered_motifs(size: int = 400) -> Raster:
    """Motifs that run off the edges, so tiling them shows a seam."""
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    # A jittered grid rather than free random placement. Nine points scattered
    # uniformly clump, and a clumped source tiles into bands of motif and bands
    # of nothing — which reads as a tiling bug when it is only the artwork.
    rng = _rng(7)
    step = size / 3
    for row in range(3):
        for col in range(3):
            cx = (col + 0.5) * step + rng.uniform(-step * 0.18, step * 0.18)
            cy = (row + 0.5) * step + rng.uniform(-step * 0.18, step * 0.18)
            r = rng.uniform(size * 0.06, size * 0.11)
            fill = [(190, 66, 84), (72, 120, 168), (206, 158, 52)][(row + col) % 3]
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*fill, 255))
    return _raster(image)


def overlay_linework(size: int = 420) -> Raster:
    """Black line art on white — thin strokes that a fade eats first."""
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    centre = size / 2
    for ring in range(3):
        radius = size * (0.16 + ring * 0.12)
        draw.ellipse((centre - radius, centre - radius, centre + radius, centre + radius),
                     outline=(24, 26, 30, 255), width=max(1, size // 210))
    for angle in range(0, 360, 30):
        rad = np.deg2rad(angle)
        draw.line((centre, centre,
                   centre + np.cos(rad) * size * 0.42, centre + np.sin(rad) * size * 0.42),
                  fill=(24, 26, 30, 255), width=max(1, size // 280))
    return _raster(image)


def overlay_gold_leaf(size: int = 380) -> Raster:
    """One motif on transparency, with a colour worth naming in an instruction."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for angle, length in ((-30, 0.42), (30, 0.38), (0, 0.46)):
        rad = np.deg2rad(angle)
        tip = (size / 2 + np.sin(rad) * size * length, size * 0.9 - np.cos(rad) * size * length)
        draw.line((size / 2, size * 0.9, *tip), fill=(150, 122, 40, 255), width=max(2, size // 120))
        leaf = Image.new("RGBA", (int(size * 0.26), int(size * 0.42)), (0, 0, 0, 0))
        ImageDraw.Draw(leaf).ellipse((0, 0, leaf.width - 1, leaf.height - 1), fill=(201, 162, 39, 255))
        leaf = leaf.rotate(-angle, expand=True, resample=Image.BICUBIC)
        image.alpha_composite(leaf, (int(tip[0] - leaf.width / 2), int(tip[1] - leaf.height / 2)))
    return _raster(image)


BASES = {
    "plate": base_plate,
    "vase": base_vase,
    "coaster": base_coaster,
    "phone-case": base_phone_case,
    "wood-tile": base_wood_tile,
}

OVERLAYS = {
    "orchid-on-white": overlay_orchid_on_white,
    "orchid-cutout": overlay_orchid_cutout,
    "orchid-on-gradient": overlay_orchid_on_gradient,
    "seamless-floral": overlay_seamless_floral,
    "scattered-motifs": overlay_scattered_motifs,
    "linework": overlay_linework,
    "gold-leaf": overlay_gold_leaf,
}
