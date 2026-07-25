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

## Fading into the glass

The fade lowers **alpha**, not colour. That matters: Studio builds the white
underbase from the alpha channel, so thinning alpha thins the white and the
glass genuinely takes over. Blending the artwork toward the glass colour instead
would leave the underbase at full density and print a solid white patch with
pale ink on it — a sticker fading, not ink dissolving.

### One ramp, four ways to render it

| | What it does | Suits |
| --- | --- | --- |
| **Tonal** (the default) | Every element gets more transparent | Anything — but with a white base the tail runs into the printer's dither floor and goes speckly |
| **Dissolve** (`--fade-dissolve 1`) | Whole elements drop out at an increasing rate; survivors stay fully opaque | Repeating patterns of discrete motifs |
| **Dot screen** (`--fade-halftone 1.5`) | Manga screentone: tone from dot *size*, every dot full-strength | Solid areas and large motifs, which dissolve can only take or leave |
| **Ink layers** (`--fade-layers 4`) | Stacked passes, dropped one at a time toward the edge | Printing without a white base, where stacking also deepens colour |

The last three are the same trick at three different scales — tone from *how
much* full-strength ink there is, never from how dilute it is. Dissolve works in
the plane by dropping motifs, the dot screen works in the plane by shrinking
dots, ink layers work in the Z axis by dropping passes. All three dodge the
dither floor entirely, because nothing is ever laid down faint.

They are alternatives, not stackable: if you set more than one, layers win, then
the screen, then dissolve, and the tool says which it used.

Pick by what your artwork is made of. **Discrete motifs → dissolve.** A
`--fade-dissolve 0.5` in between staggers the two, so some elements fade faster
than others, which scatters more organically than either end.

![a pattern dissolving down a panel, previewed on green glass](docs/fade-dissolve.png)

**One solid shape → dot screen**, because dissolve sees one element and can only
keep or drop it. Element dropout is one seeded draw per element, so the same
settings always give the same scatter — preview and export match, and a reprint
next year matches too.

![the same pattern as a dot screen, printed without a white base](docs/fade-halftone.png)

> **Keep the screen coarse — 1 mm or above.** The E1's RIP is halftoning at
> device resolution as well, and a fine screen of yours beats against its screen
> and moirés. At 1–3 mm a dot spans hundreds of device pixels, so there is
> nothing to interfere with and the dots read as deliberate texture. Under
> 0.8 mm the tool warns you.

Dot area is faithful to the ramp (asking for 25% coverage lays down 25.3%), and
100% is genuinely solid, so the top of a gradient is untouched artwork.

## Printing without a white base

Leaving the white underbase off changes what a fade does, and mostly for the
better.

With white, a fade crosses a **change of material** — opaque, light-scattering
ink at one end, bare glass at the other. That is a long perceptual distance, and
its last stretch is where the dither floor lives. White is also the highest
contrast thing you can put on clear glass, so sparse coverage reads as visible
speckle.

Without white, the ink is a glaze. Printed and unprinted areas are both
transparent and differ only by a tint, so the fade travels a much shorter
distance and sparse coverage reads as a *thinner tint* rather than as specks. A
plain tonal fade is fine — the printability warning stands down when you tell
the tool you are printing this way.

What you trade away is **colour fidelity**: every ink multiplies with the glass,
so on green glass reds go brown and blues go teal.

Switch the preview between **White base** and **No white base** above the
canvas. The no-white mode renders the artwork multiplied through the glass,
which is the same maths the ink does — so what you see is what the glaze will
look like. It changes the preview only; the exported files are identical either
way, since the underbase is Studio's decision, not the file's.

### Getting colour back by stacking layers

Ink transmittances multiply, so a second pass of the same ink squares its
effect. That pulls the colour away from the glass and toward the ink very
quickly — for a saturated red on green glass:

| Passes | Transmits | Ink dominance over the glass | Light through |
| --- | --- | --- | --- |
| 1 | `[0.45, 0.135, 0.075]` | 6× | 0.220 |
| 2 | `[0.405, 0.020, 0.011]` | 36× | 0.145 |
| 3 | `[0.365, 0.003, 0.002]` | 216× | 0.123 |

By three passes the glass's green is gone and the red is genuinely red. **But
this only works for inks that actually absorb.** A pale ink is nearly
transparent by definition, so it has almost nothing to absorb with and the glass
keeps winning no matter how many passes — pale pink goes from 1.7× dominance at
one layer to 1.9× at five. Pale yellow on green glass still reads green at five
passes.

So stacking buys back **saturation and hue, not lightness** — and every pass
costs brightness. The practical read: printing without white on coloured glass
wants a saturated, dark palette. That is the stained-glass discipline. Pale
tints are what white ink is actually for.

`--fade-layers 4` builds the fade this way, dropping a pass at a time toward the
transparent end:

![a four-layer stacked fade previewed on green glass](docs/fade-layers.png)

Each pass is solid ink, so there is no dither floor at all — but four layers can
only make five steps, so it bands visibly. That is the trade: dissolve and dot
screens give you smooth density at the cost of texture; layers give you clean
solid ink at the cost of banding.

**Exports for a layered print.** Two targets, because it depends how your
software drives the passes:

- `--export layer-map` — one greyscale image where white is the full stack and
  black is bare glass. This is the shape a relief or height pass wants.
- `--export layers` — one file per pass (`_layer1of4.png` … `_layer4of4.png`),
  each at full strength. Pass *k* covers everywhere getting at least *k* layers,
  so printing them in order builds the gradient out of solid ink. Use these if
  you are driving the passes yourself.

> Worth checking on your own machine: whether Studio will take a height map for
> the relief pass, or whether you need to run the passes manually. The tool
> gives you both shapes; which one you want is a question about the software,
> not the file. Registration across several passes is the other thing to test —
> misalignment shows up as colour fringing at edges.

Element dropout is one draw per element from `--fade-seed`, so the same settings
always produce the same scatter — preview and export match, and you can re-run a
print months later and get the same object.

### The controls

| Control | What it does |
| --- | --- |
| `--fade linear\|radial\|shape` | Direction. `shape` fades from the edge of the target silhouette inwards, which suits a panel |
| `--fade-angle` | 90 fades downward, 0 to the right, 270 upward |
| `--fade-start` / `--fade-end` | Where along the axis the fade begins and completes. **Narrowing the gap is how you make it happen faster** |
| `--fade-curve` | The rate. 1 is linear; above 1 holds the ink then drops away late; below 1 drops away immediately then trails off |
| `--fade-min` / `--fade-max` | The two ends of the ramp. Raise `--fade-min` to fade to a ghost rather than to nothing |
| `--fade-dissolve` | Tonal ↔ dropout, as above |
| `--fade-layers` | Build the fade from N stacked ink passes. Takes precedence over the screen and dissolve |
| `--fade-halftone` | Dot screen pitch in mm. Takes precedence over dissolve — they express the same ramp |
| `--fade-halftone-angle` | Screen angle. 45° is traditional and least obtrusive |
| `--fade-per-element` | Give each element one opacity instead of letting the ramp cut through it. Keeps motifs crisp |
| `--fade-what` | **Which elements fade**, in the same language as `--keep` |
| `--fade-invert` | Reverse the direction |
| `--fade-cutoff` | Snap alpha below this to zero |

### Choosing what fades

`--fade-what` takes the same instructions as `--keep`, so you can fade one part
of the artwork and leave the rest solid:

```bash
glassprint compose panel.png botanical.png \
  --keep "remove the white background" \
  --fade linear --fade-what "the leaves" --fade-curve 1.6
```

The selector runs against the *source* artwork, before any recolouring, so
"the leaves" still means the leaves after you have tinted everything one colour.
It is then tiled in step with the pattern. If nothing matches, the fade is left
off and the tool says so rather than quietly fading everything.

### Watching the printable floor

UV dithering starts breaking up under roughly 12% coverage. The readout reports
**faintest ink** — the thinnest ink that will actually be laid down, measured on
solid areas only so anti-aliased edges don't drag it to zero — and warns when
you're under that. The three ways out are: raise the end opacity, add dissolve,
or set a minimum printable ink level.

```
fade         : linear · 90° · 0–1 · 100% dissolve over 458 elements
faintest ink : 94% coverage
```

### Judging it

A fade-to-transparent cannot be judged against a checkerboard. Tick **On glass**
above the preview and set the colour of the glass you're printing on — the
preview then sits on that colour, which is what the finished piece will look
like. It changes the preview only, never the exported files.

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
