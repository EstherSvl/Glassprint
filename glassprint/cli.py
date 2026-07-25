"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .compose import ComposeSpec, compose
from .export import ExportSpec, export
from .fade import Fade
from .nl import build_plan
from .pattern import Placement, analyse
from .raster import Raster
from .recolor import ColorSpec
from .segment import Backends, evaluate

app = typer.Typer(
    add_completion=False,
    help="Overlay artwork onto a base image and prepare it for UV printing on glass.",
)


def _echo_notes(notes: list[str]) -> None:
    for note in notes:
        typer.secho(f"  note: {note}", fg=typer.colors.YELLOW)


@app.command()
def version() -> None:
    """Print the version and which optional backends are installed."""
    typer.echo(f"glassprint {__version__}")
    for name, present in Backends.probe().items():
        mark = "yes" if present else "no"
        typer.echo(f"  {name:<10} {mark}")


@app.command()
def inspect(image: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Report size, DPI, transparency and whether the art reads as a pattern."""
    raster = Raster.open(image)
    width_mm, height_mm = raster.size_mm
    dpi = raster.effective_dpi

    typer.echo(f"{image.name}")
    typer.echo(f"  pixels     {raster.width} x {raster.height}")
    typer.echo(
        f"  dpi        {dpi[0]:.0f} x {dpi[1]:.0f}"
        + ("" if raster.dpi else "  (not tagged in the file — assuming 300)")
    )
    typer.echo(f"  print size {width_mm:.1f} x {height_mm:.1f} mm")
    typer.echo(f"  alpha      {'yes' if raster.has_alpha else 'no'}")

    backends = Backends()
    plan = build_plan("", raster)
    cutout = evaluate(plan, raster, backends)
    info = analyse(raster, cutout)
    typer.echo(f"  artwork    {'repeating pattern' if info.is_pattern else 'single motif'} ({info.reason})")
    typer.echo(f"  tiles      {'seamlessly' if info.seamless else 'with mirroring (edges do not match)'}")
    _echo_notes(backends.notes)


@app.command()
def mask(
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    keep: str = typer.Option("", "--keep", "-k", help="What to keep or remove, in plain English."),
    out: Path = typer.Option(Path("cutout.png"), "--out", "-o"),
    tolerance: float = typer.Option(1.0, "--tolerance", help="Widen colour matching (1.0 = default)."),
    claude: bool = typer.Option(False, "--claude", help="Use Claude to interpret the instruction."),
) -> None:
    """Write just the cut-out artwork, to check the mask before composing."""
    raster = Raster.open(image)
    backends = Backends()
    plan = build_plan(keep, raster, use_claude=claude, tolerance=tolerance)
    cutout = evaluate(plan, raster, backends)

    from .pattern import apply_cutout

    result = Raster(apply_cutout(raster, cutout), dpi=raster.dpi)
    result.save(out)
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    typer.echo(f"  plan: {plan.describe()}  [{plan.source}]")
    _echo_notes(backends.notes)


@app.command("compose")
def compose_command(
    base: Path = typer.Argument(..., exists=True, dir_okay=False, help="Procreate/Affinity export."),
    overlay: Path = typer.Argument(..., exists=True, dir_okay=False, help="Pattern or motif to apply."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    keep: str = typer.Option("", "--keep", "-k", help="What to keep/remove from the overlay."),
    target: str = typer.Option("alpha", "--target", help="alpha | describe | largest | full | rect"),
    target_describe: str = typer.Option("", "--target-describe", help="Describe the area to fill."),
    fit: str = typer.Option("auto", "--fit", help="auto | shape | contain | cover | tile | stretch"),
    repeats: Optional[float] = typer.Option(None, "--repeats", help="Pattern repeats across the shape."),
    repeat_mm: Optional[float] = typer.Option(None, "--repeat-mm", help="Physical size of one repeat."),
    scale: float = typer.Option(1.0, "--scale"),
    rotation: float = typer.Option(0.0, "--rotate", help="Degrees clockwise."),
    offset_x: float = typer.Option(0.0, "--offset-x", help="Fraction of the shape width."),
    offset_y: float = typer.Option(0.0, "--offset-y", help="Fraction of the shape height."),
    mirror: str = typer.Option("auto", "--mirror", help="auto | on | off"),
    color: Optional[str] = typer.Option(None, "--color", "-c", help="Recolour the pattern (#hex or name)."),
    color_mode: str = typer.Option("tint", "--color-mode", help="tint | duotone | replace | mono"),
    color_from: Optional[str] = typer.Option(None, "--color-from", help="Colour to replace."),
    color2: Optional[str] = typer.Option(None, "--color2", help="Duotone shadow colour."),
    strength: float = typer.Option(1.0, "--color-strength"),
    saturation: float = typer.Option(1.0, "--saturation"),
    brightness: float = typer.Option(1.0, "--brightness"),
    contrast: float = typer.Option(1.0, "--contrast"),
    hue_shift: float = typer.Option(0.0, "--hue-shift", help="Degrees."),
    fade: str = typer.Option("none", "--fade", help="none | linear | radial | shape"),
    fade_what: str = typer.Option("", "--fade-what", help="Which elements fade, in plain English."),
    fade_angle: float = typer.Option(90.0, "--fade-angle", help="90 fades downward, 0 to the right."),
    fade_start: float = typer.Option(0.0, "--fade-start", help="Where the fade begins (0-1)."),
    fade_end: float = typer.Option(1.0, "--fade-end", help="Where the fade completes (0-1)."),
    fade_curve: float = typer.Option(1.0, "--fade-curve", help=">1 holds on then drops late; <1 drops away early."),
    fade_min: float = typer.Option(0.0, "--fade-min", help="Opacity at the far end of the ramp."),
    fade_max: float = typer.Option(1.0, "--fade-max", help="Opacity at the near end of the ramp."),
    fade_center_x: float = typer.Option(0.5, "--fade-center-x"),
    fade_center_y: float = typer.Option(0.5, "--fade-center-y"),
    fade_per_element: bool = typer.Option(False, "--fade-per-element", help="One opacity per element."),
    fade_dissolve: float = typer.Option(0.0, "--fade-dissolve", help="0 thins every element, 1 drops elements whole."),
    fade_seed: int = typer.Option(0, "--fade-seed", help="Keeps the dissolve reproducible."),
    fade_halftone: float = typer.Option(0.0, "--fade-halftone", help="Dot screen pitch in mm (manga style). Keep it 1mm or coarser."),
    fade_halftone_angle: float = typer.Option(45.0, "--fade-halftone-angle", help="Screen angle in degrees."),
    fade_invert: bool = typer.Option(False, "--fade-invert"),
    fade_cutoff: float = typer.Option(0.0, "--fade-cutoff", help="Snap alpha below this to zero (~0.12 for UV)."),
    opacity: float = typer.Option(1.0, "--opacity"),
    blend: str = typer.Option("normal", "--blend", help="normal | multiply | screen | overlay"),
    no_clip: bool = typer.Option(False, "--no-clip", help="Do not clip the overlay to the shape."),
    feather: float = typer.Option(0.0, "--feather", help="Soften the shape edge, in pixels."),
    formats: str = typer.Option("png", "--format", "-f", help="Comma separated: png,jpg,tiff,webp,bmp"),
    targets: str = typer.Option("composite,overlay", "--export", help="composite,overlay,shape-mask,cutout-mask"),
    dpi: Optional[float] = typer.Option(None, "--dpi", help="Output DPI (default: the base image's)."),
    width_mm: Optional[float] = typer.Option(None, "--width-mm", help="Printed width on the glass."),
    height_mm: Optional[float] = typer.Option(None, "--height-mm", help="Printed height on the glass."),
    quality: int = typer.Option(95, "--quality", help="JPEG/WebP quality."),
    background: str = typer.Option("#ffffff", "--background", help="Flatten colour for opaque formats."),
    tolerance: float = typer.Option(1.0, "--tolerance"),
    claude: bool = typer.Option(False, "--claude", help="Use Claude to interpret instructions."),
    json_out: bool = typer.Option(False, "--json", help="Print the manifest as JSON."),
) -> None:
    """Compose the overlay onto the base image and export the result."""
    base_raster = Raster.open(base)
    overlay_raster = Raster.open(overlay)

    color_spec = ColorSpec(
        mode=color_mode if color else "none",
        color=color,
        color2=color2,
        from_color=color_from,
        strength=strength,
        tolerance=tolerance,
        hue_shift=hue_shift,
        saturation=saturation,
        brightness=brightness,
        contrast=contrast,
    )
    spec = ComposeSpec(
        keep=keep,
        tolerance=tolerance,
        use_claude=claude,
        target=target,
        target_describe=target_describe,
        clip_to_shape=not no_clip,
        shape_feather=feather,
        opacity=opacity,
        blend=blend,
        placement=Placement(
            fit=fit,
            repeat_across=repeats,
            repeat_mm=repeat_mm,
            scale=scale,
            rotation=rotation,
            offset_x=offset_x,
            offset_y=offset_y,
            mirror=mirror,
        ),
        color=color_spec,
        fade=Fade(
            mode=fade,
            what=fade_what,
            angle=fade_angle,
            center_x=fade_center_x,
            center_y=fade_center_y,
            start=fade_start,
            end=fade_end,
            curve=fade_curve,
            min_alpha=fade_min,
            max_alpha=fade_max,
            per_element=fade_per_element,
            dissolve=fade_dissolve,
            seed=fade_seed,
            halftone_mm=fade_halftone,
            halftone_angle=fade_halftone_angle,
            invert=fade_invert,
            cutoff=fade_cutoff,
        ),
    )

    backends = Backends()
    result = compose(base_raster, overlay_raster, spec, backends)

    export_spec = ExportSpec(
        formats=[f.strip() for f in formats.split(",") if f.strip()],
        targets=[t.strip() for t in targets.split(",") if t.strip()],
        dpi=dpi,
        width_mm=width_mm,
        height_mm=height_mm,
        quality=quality,
        background=background,
        basename=base_raster.name,
    )
    manifest = export(result, out, export_spec)

    if json_out:
        typer.echo(json.dumps({"files": manifest, "summary": result.summary()}, indent=2))
        return

    typer.secho(f"composed {base.name} + {overlay.name}", fg=typer.colors.GREEN)
    typer.echo(f"  overlay plan : {result.plan.describe()}  [{result.plan.source}]")
    typer.echo(
        f"  artwork      : {'pattern' if result.info.is_pattern else 'motif'} — {result.info.reason}"
    )
    typer.echo(f"  target shape : {result.box[2] - result.box[0]} x {result.box[3] - result.box[1]} px")
    if result.fade.active:
        faintest = result.faintest_alpha()
        line = f"  fade         : {result.fade.describe()}"
        if result.fade_elements:
            line += f" over {result.fade_elements} elements"
        typer.echo(line)
        typer.echo(f"  faintest ink : {faintest:.0%} coverage")
        if 0 < faintest < 0.12:
            typer.secho(
                "  note: the faintest ink is under ~12% — UV dithering may go speckly there. "
                "Raise --fade-min, or set --fade-cutoff 0.12 to drop the tail.",
                fg=typer.colors.YELLOW,
            )
    for entry in manifest:
        size = entry["size_mm"]
        typer.echo(
            f"  wrote {entry['file']}  {entry['pixels'][0]}x{entry['pixels'][1]}px "
            f"@ {entry['dpi']}dpi  ({size[0]} x {size[1]} mm)"
        )
    _echo_notes(result.notes)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open a browser window."),
) -> None:
    """Run the local web interface."""
    import uvicorn

    from .server import create_app

    url = f"http://{host}:{port}/"
    typer.secho(f"glassprint running at {url}", fg=typer.colors.GREEN)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    app()
