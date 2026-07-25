# glassprint

Overlay artwork onto a base image, with the background removed, and export it at
the right physical size for UV printing onto glass.

You give it two things — a base image exported from Procreate or Affinity
Designer, and a pattern or motif — plus a sentence about what to keep. It works
out whether the artwork is a repeating pattern or a single motif, sizes it to
the shape in your base image, clips it to that silhouette, and writes files with
the DPI and millimetre dimensions the printer needs.

Everything runs locally. Nothing is uploaded anywhere unless you explicitly turn
on the optional Claude integration.

![the local web interface](docs/screenshot.png)

---

## Install

```bash
cd glassprint
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

That is the whole install — no model downloads, no API keys.

Two optional extras add capability:

```bash
pip install -e ".[smart]"    # local models: true object selection + subject cutout (~1.5 GB, downloads on first use)
pip install -e ".[claude]"   # let Claude interpret trickier instructions (needs ANTHROPIC_API_KEY)
```

`glassprint version` tells you which are active.

## Use it

### The web interface

```bash
glassprint serve
```

Opens `http://127.0.0.1:8765`. Drop in a base image and an overlay, type what to
keep, and the preview updates as you move the sliders. Export when it looks
right.

### The command line

```bash
glassprint compose panel.png pattern.png \
  --keep "keep the flowers, remove the white background" \
  --fit tile --repeat-mm 20 \
  --color "#0f766e" \
  --width-mm 180 --dpi 300 \
  --out ./exports
```

Other commands:

| Command | What it does |
| --- | --- |
| `glassprint inspect art.png` | Size, DPI, print size, transparency, and whether it reads as a pattern |
| `glassprint mask art.png --keep "..."` | Writes just the cut-out, to check the mask before composing |
| `glassprint compose base art -o out/` | The full pipeline |
| `glassprint serve` | The web interface |
| `glassprint version` | Version and which optional backends are installed |

---

## How the instruction is read

Write what you want in plain English. The parser splits on keep/remove verbs and
works out what each phrase refers to:

| You write | What it selects |
| --- | --- |
| *(nothing)* | Removes the background |
| `remove the white background` | The flat ground the art sits on |
| `keep the gold parts` | Everything matching the colour word *gold* |
| `keep only the linework` | The dark tones |
| `keep the flowers, drop the leaves` | Named objects |
| `keep the red roses` | The roses — by colour if no model is installed, by object if one is |
| `keep #c9a227` | An exact hex colour |
| `the pattern without the background` | Same as remove-the-background |

If any `keep` appears, the result starts empty and keeps are added; otherwise it
starts as the whole image. Removals are subtracted either way.

**Without the `smart` extra**, colour, tone, background and subject selection all
work exactly. Object names fall back to the colour in the phrase (so *"the red
roses"* still works) and the tool tells you it did so, in the readout under the
preview.

**With the `smart` extra**, object names are handled by CLIPSeg and the subject
cutout by rembg — no phrasing changes needed, the same instructions just get
sharper.

**With the `claude` extra** and `--claude` (or the checkbox in the UI), Claude
looks at the image and writes the selection plan itself, including sampling
actual hex colours out of your artwork. Useful for instructions the rules trip
over.

## Sizing the pattern

This is the part that decides whether the print looks right on the object.

- **`--fit auto`** (the default) counts the separate marks in the artwork. Many
  scattered marks means a repeating pattern, so it tiles; one big shape means a
  motif, so it scales to fit the target once.
- **`--repeat-mm 20`** sets the physical size of one repeat on the glass. This is
  usually what you want — it is independent of resolution and canvas size.
- **`--repeats 4`** instead sets how many times the pattern repeats across the
  shape.
- If both files carry DPI, `auto` suggests the repeat count that keeps the
  artwork at its own physical size.
- Tiling checks whether opposite edges of your artwork match. If they do it tiles
  directly; if they do not it mirrors the tile so the seams do not show. Force it
  either way with `--mirror on|off`.

## Where the overlay lands

| `--target` | Behaviour |
| --- | --- |
| `alpha` (default) | The non-transparent area of your base export — the shape you drew |
| `describe` | `--target-describe "the vase body"`, using the same language parser |
| `largest` | The largest solid region, for flat exports with no alpha |
| `full` | The whole canvas |
| `rect` | An explicit rectangle |

The overlay is clipped to that shape by default (`--no-clip` turns it off), and
`--feather` softens the edge, which reads better on curved silhouettes than a
hard 1-bit cut.

## Colour

`--color` with `--color-mode`:

- **`tint`** maps the artwork's luminance onto a black → your colour → white ramp,
  so the pattern keeps its internal shading instead of going flat.
- **`duotone`** maps it between two colours (`--color` highlight, `--color2` shadow).
- **`replace`** swaps one colour for another (`--color-from` → `--color`), leaving
  the rest alone.
- **`mono`** for greyscale.

Plus `--hue-shift`, `--saturation`, `--brightness` and `--contrast` for smaller
adjustments.

## Export

Formats: PNG, TIFF, JPEG, WebP, BMP. Reads those plus PSD and GIF.

Every export writes:

- **the composite** — base + overlay, in whatever formats you ask for, *plus* the
  base image's own format automatically;
- **the overlay alone** — on transparency, so you can drop it into another layer
  stack. If you only picked opaque formats, PNG is added, because an overlay that
  cannot hold alpha is not much use.

Optionally also the shape mask and the cut-out mask (`--export composite,overlay,shape-mask,cutout-mask`),
which are handy for checking what the tool decided.

Set the physical size with `--width-mm` (or `--height-mm`) and `--dpi`. The files
are resampled and tagged so the printer places the art at that size — both layers
together, so they stay registered to each other.

```
wrote panel_composite.png  2126x1594px @ 300.0dpi  (180.0 x 135.0 mm)
wrote panel_overlay.png    2126x1594px @ 300.0dpi  (180.0 x 135.0 mm)
```

### A note on the EufyMake workflow

These exports are flat RGB/RGBA files at a known physical size, which is what
eufyMake Studio takes. The white underbase and the gloss/texture passes are
generated in Studio from the imported artwork, so there is deliberately no
separations export here — you'd only be feeding it files it makes itself.

What that does mean is that **the alpha channel drives the white layer**, so the
cut-out quality is the thing to watch:

- Keep the edge feather modest (1–2 px at 300 dpi). It softens jagged curves,
  which is what you want. A wide feather leaves a band where colour is dense but
  white ink is thin, which reads as a washed-out halo over clear glass.
- Stray low-alpha pixels left around a cut-out become faint white ink. Export or
  preview the `cutout-mask` to see exactly what will drive the underbase before
  committing a print.

## Using it as a library

```python
from glassprint import Raster, ComposeSpec, Placement, ColorSpec, compose, export, ExportSpec

base = Raster.open("panel.png")
art = Raster.open("pattern.png")

result = compose(base, art, ComposeSpec(
    keep="keep the flowers, remove the white background",
    placement=Placement(fit="tile", repeat_mm=20),
    color=ColorSpec(mode="tint", color="#0f766e"),
))

print(result.summary())
export(result, "exports/", ExportSpec(formats=["png", "tiff"], width_mm=180, dpi=300))
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The layout:

| Module | Responsibility |
| --- | --- |
| `raster.py` | Loading, saving, DPI and millimetre bookkeeping |
| `masks.py` | Soft-mask maths (feather, grow, despeckle, components) |
| `colors.py` | Colour parsing and colour-word → HSV region selection |
| `segment.py` | Selectors → masks, plus the optional model backends |
| `nl.py` | Instruction → mask plan (rules, and the Claude path) |
| `pattern.py` | Pattern detection, tiling and placement |
| `recolor.py` | Colour treatments |
| `compose.py` | The pipeline that ties it together |
| `export.py` | Output formats, DPI, physical size |
| `cli.py` / `server.py` | The two front ends |
