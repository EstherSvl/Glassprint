"""Compose every test pairing and lay the results out to be looked at.

    python tools/make_gallery.py [outdir]

Numbers in a test report tell you an assertion held. They do not tell you the
artwork ended up upside down in the corner. This runs a spread of realistic
jobs, writes a contact sheet, and prints the handful of measurements that
distinguish a good result from a bad one — so a person can check the picture
and a test can check the numbers, against the same run.

The measurements worth reading, per job:

``kept``
    How much of the artwork survived the cut-out. Near zero means nothing was
    kept; near one means the background was not removed. Both look like a
    working tool if you only check that a file appeared.
``filled``
    How much of the target shape the ink actually covers. A motif that landed
    one pixel wide reports here and nowhere else.
``spill``
    Ink outside the target shape. With clipping on this must be zero; anything
    else is print off the edge of the object.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gallery_art as art  # noqa: E402

from glassprint.compose import ComposeSpec, GlazeSpec, LayerSpec, compose  # noqa: E402
from glassprint.fade import Fade  # noqa: E402
from glassprint.pattern import Placement  # noqa: E402
from glassprint.recolor import ColorSpec  # noqa: E402
from glassprint.segment import Backends  # noqa: E402


@dataclass
class Job:
    """One realistic thing someone would actually ask the tool to do."""

    name: str
    base: str
    overlay: str
    spec: ComposeSpec
    #: What a person looking at the result should see. Not machine-checked —
    #: it is here so the contact sheet can be read without guessing the intent.
    expect: str = ""
    overlays: list[str] = field(default_factory=list)


def jobs() -> list[Job]:
    keep_bg = "remove the white background"
    return [
        Job(
            "plate-floral-tiled", "plate", "seamless-floral",
            ComposeSpec(
                keep=keep_bg,
                placement=Placement(fit="tile", repeat_across=5),
            ),
            "an even floral repeat filling the plate, no seams, nothing past the rim",
        ),
        Job(
            "plate-orchid-photo", "plate", "orchid-on-white",
            ComposeSpec(keep=keep_bg, placement=Placement(fit="shape")),
            "three white orchids on the plate — the paper they were shot on must be gone",
        ),
        Job(
            "plate-orchid-cutout", "plate", "orchid-cutout",
            ComposeSpec(placement=Placement(fit="contain", scale=0.85)),
            "the same orchids, already cut out, centred and not touching the rim",
        ),
        Job(
            "plate-orchid-gradient", "plate", "orchid-on-gradient",
            ComposeSpec(keep="remove the background", placement=Placement(fit="shape")),
            "pink orchids off a graded backdrop — the backdrop must not come with them",
        ),
        Job(
            "vase-floral-tiled", "vase", "seamless-floral",
            ComposeSpec(
                keep=keep_bg,
                placement=Placement(fit="tile", repeat_mm=18),
            ),
            "the repeat running up the vase at a constant physical size, following the neck",
        ),
        Job(
            "vase-gold-fade", "vase", "gold-leaf",
            ComposeSpec(
                placement=Placement(fit="shape", scale=0.9),
                fade=Fade(mode="linear", angle=270, layers=3),
            ),
            "a gold sprig up the vase, thinning toward the top in three printed passes",
        ),
        Job(
            "coaster-linework", "coaster", "linework",
            ComposeSpec(
                keep="keep the linework",
                placement=Placement(fit="contain", scale=0.8),
            ),
            "the dark lines only, on the coaster, with the white paper dropped",
        ),
        Job(
            "coaster-gold-tinted", "coaster", "gold-leaf",
            ComposeSpec(
                placement=Placement(fit="contain", scale=0.7, rotation=15),
                color=ColorSpec(mode="tint", color="#b8862b"),
            ),
            "one gold sprig, turned slightly, well inside the coaster",
        ),
        Job(
            "phone-orchid", "phone-case", "orchid-cutout",
            ComposeSpec(placement=Placement(fit="contain", scale=0.9)),
            "orchids down the case — and nothing at all over the lens cutout",
        ),
        Job(
            "wood-tile-motifs", "wood-tile", "scattered-motifs",
            ComposeSpec(
                keep=keep_bg,
                target="largest",
                placement=Placement(fit="tile", repeat_across=3),
            ),
            "motifs on the pale tile only — never on the wood around it",
        ),
        Job(
            "plate-two-layers", "plate", "seamless-floral",
            ComposeSpec(
                keep=keep_bg,
                # Two motifs on one plate want different placements, and this is
                # what per-layer settings are for. Sharing them tiled the single
                # sprig across the whole plate alongside the repeat, which is a
                # reasonable thing for the tool to have done and not at all what
                # anyone means by "put a sprig on top".
                layers=[
                    LayerSpec(keep=keep_bg, placement=Placement(fit="tile", repeat_across=6)),
                    LayerSpec(placement=Placement(fit="contain", scale=0.55)),
                ],
            ),
            "the floral repeat with a single gold sprig sitting on top of it",
            overlays=["seamless-floral", "gold-leaf"],
        ),
        Job(
            "plate-orchid-halftone", "plate", "orchid-cutout",
            ComposeSpec(
                placement=Placement(fit="contain", scale=0.95),
                # Tinted, because white ink on a near-white plate is a correct
                # result you cannot see, and a test you cannot read is not one.
                color=ColorSpec(mode="tint", color="#8c3f6b"),
                fade=Fade(mode="radial", halftone_mm=1.6),
            ),
            "orchids dissolving into a coarse dot screen toward the rim",
        ),
        Job(
            "vase-described-target", "vase", "gold-leaf",
            ComposeSpec(
                target="describe",
                target_describe="the belly of the vase",
                placement=Placement(fit="contain", scale=0.8),
            ),
            "the sprig on the wide part of the vase, not the neck",
        ),
        Job(
            "coaster-glaze", "coaster", "seamless-floral",
            ComposeSpec(
                keep=keep_bg,
                placement=Placement(fit="tile", repeat_across=4),
                glaze=GlazeSpec(enabled=True, glass="#7d9b8f", colours=3),
            ),
            "the repeat solved as stacked inks for printing onto green glass",
        ),
    ]


def measure(result, spec: ComposeSpec) -> dict:
    """The three numbers that separate a good result from a plausible one."""
    shape = result.shape_mask
    ink = result.overlay_layer.rgba[:, :, 3].astype(np.float32) / 255.0

    inside = float((ink * shape).sum())
    shape_area = float(shape.sum())
    outside = float((ink * (1.0 - shape)).sum())

    return {
        "kept": round(float(result.cutout_mask.mean()), 4),
        "filled": round(inside / shape_area, 4) if shape_area else 0.0,
        "spill": round(outside / max(float(ink.sum()), 1e-6), 4),
        "shape_of_canvas": round(shape_area / shape.size, 4),
        "faintest": result.faintest_alpha(),
        "plan": result.plan.describe(),
        "pattern": "pattern" if result.info.is_pattern else "motif",
    }


def run(job: Job) -> tuple[object, dict]:
    base = art.BASES[job.base]()
    names = job.overlays or [job.overlay]
    overlays = [art.OVERLAYS[name]() for name in names]
    backends = Backends()
    result = compose(base, overlays, job.spec, backends)
    report = measure(result, job.spec)
    report["notes"] = [n for n in result.notes if "pip install" not in n]
    return result, report


def contact_sheet(entries: list[tuple[str, Image.Image, str]], path: Path,
                  cell: int = 300, cols: int = 4) -> None:
    rows = (len(entries) + cols - 1) // cols
    label_h = 46
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)

    for index, (name, image, caption) in enumerate(entries):
        thumb = image.copy()
        thumb.thumbnail((cell - 16, cell - 16))
        x = (index % cols) * cell
        y = (index // cols) * (cell + label_h)

        pad = Image.new("RGB", (cell - 16, cell - 16), (46, 48, 54))
        pad.paste(thumb.convert("RGB"), ((pad.width - thumb.width) // 2,
                                         (pad.height - thumb.height) // 2))
        sheet.paste(pad, (x + 8, y + 8))
        draw.text((x + 8, y + cell - 2), name, fill=(232, 234, 240))
        draw.text((x + 8, y + cell + 12), caption, fill=(150, 158, 172))

    sheet.save(path)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "gallery")
    out.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, Image.Image, str]] = []
    reports: dict[str, dict] = {}

    for job in jobs():
        result, report = run(job)
        reports[job.name] = report

        Image.fromarray(result.composite.rgba, "RGBA").save(out / f"{job.name}.png")
        Image.fromarray(result.overlay_layer.rgba, "RGBA").save(out / f"{job.name}-overlay.png")
        entries.append((
            job.name,
            Image.fromarray(result.composite.rgba, "RGBA"),
            f"kept {report['kept']:.0%} · fills {report['filled']:.0%} · spill {report['spill']:.1%}",
        ))
        print(
            f"{job.name:24} kept {report['kept']:>6.1%}  fills {report['filled']:>6.1%}  "
            f"spill {report['spill']:>6.1%}  {report['pattern']:7} {report['plan']}"
        )

    contact_sheet(entries, out / "_sheet.png")
    (out / "_report.json").write_text(json.dumps(reports, indent=2))
    print(f"\n{out}/_sheet.png")


if __name__ == "__main__":
    main()
