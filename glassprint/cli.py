"""Command line interface."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import typer

from . import __version__
from .colors import parse_color, to_hex
from .compose import ComposeSpec, GlazeSpec, LayerSpec, compose
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
    overlay: List[Path] = typer.Argument(..., exists=True, dir_okay=False, help="Pattern or motif to apply. Give several to stack them."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    keep: List[str] = typer.Option([], "--keep", "-k", help="What to keep/remove. Repeat to say something different for each overlay."),
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
    fade_layers: int = typer.Option(0, "--fade-layers", help="Build the fade from N printed ink layers instead of a smooth ramp."),
    fade_halftone: float = typer.Option(0.0, "--fade-halftone", help="Dot screen pitch in mm (manga style). Keep it 1mm or coarser."),
    fade_halftone_angle: float = typer.Option(45.0, "--fade-halftone-angle", help="Screen angle in degrees."),
    fade_invert: bool = typer.Option(False, "--fade-invert"),
    fade_cutoff: float = typer.Option(0.0, "--fade-cutoff", help="Snap alpha below this to zero (~0.12 for UV)."),
    opacity: List[float] = typer.Option([], "--opacity", help="Repeat for each overlay."),
    blend: List[str] = typer.Option([], "--blend", help="normal | multiply | screen | overlay. Repeat for each overlay."),
    no_clip: bool = typer.Option(False, "--no-clip", help="Do not clip the overlay to the shape."),
    feather: float = typer.Option(0.0, "--feather", help="Soften the shape edge, in pixels."),
    formats: str = typer.Option("png", "--format", "-f", help="Comma separated: png,jpg,tiff,webp,bmp"),
    targets: str = typer.Option("composite,overlay", "--export", help="composite,overlay,shape-mask,cutout-mask"),
    dpi: Optional[float] = typer.Option(None, "--dpi", help="Output DPI (default: the base image's)."),
    width_mm: Optional[float] = typer.Option(None, "--width-mm", help="Printed width on the glass."),
    height_mm: Optional[float] = typer.Option(None, "--height-mm", help="Printed height on the glass."),
    quality: int = typer.Option(95, "--quality", help="JPEG/WebP quality."),
    background: str = typer.Option("#ffffff", "--background", help="Flatten colour for opaque formats."),
    glaze_on: bool = typer.Option(False, "--glaze", help="Build colours by stacking different inks."),
    glass: str = typer.Option("#ffffff", "--glass", help="Colour of the glass, for glazing and preview."),
    glaze_palette: str = typer.Option("", "--palette", help="Inks to glaze with."),
    glaze_colours: int = typer.Option(5, "--glaze-colours"),
    tolerance: float = typer.Option(1.0, "--tolerance"),
    claude: bool = typer.Option(False, "--claude", help="Use Claude to interpret instructions."),
    json_out: bool = typer.Option(False, "--json", help="Print the manifest as JSON."),
) -> None:
    """Compose the overlay onto the base image and export the result."""
    base_raster = Raster.open(base)
    overlay_rasters = [Raster.open(path) for path in overlay]

    def per_overlay(values: list, default):
        """One value each, or one value for all of them.

        Say ``--keep`` once and every overlay is cut the same way; say it as
        many times as there are overlays and each gets its own. Anything in
        between is a miscount worth stopping for, not guessing at.
        """
        if not values:
            return [default] * len(overlay_rasters)
        if len(values) == 1:
            return [values[0]] * len(overlay_rasters)
        if len(values) != len(overlay_rasters):
            raise typer.BadParameter(
                f"got {len(values)} values for {len(overlay_rasters)} overlays — "
                "give one for all of them, or one each"
            )
        return list(values)

    keeps = per_overlay(keep, "")
    opacities = per_overlay(opacity, 1.0)
    blends = per_overlay(blend, "normal")

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
    placement_spec = Placement(
        fit=fit,
        repeat_across=repeats,
        repeat_mm=repeat_mm,
        scale=scale,
        rotation=rotation,
        offset_x=offset_x,
        offset_y=offset_y,
        mirror=mirror,
    )
    spec = ComposeSpec(
        keep=keeps[0],
        tolerance=tolerance,
        use_claude=claude,
        target=target,
        target_describe=target_describe,
        clip_to_shape=not no_clip,
        shape_feather=feather,
        opacity=opacities[0],
        blend=blends[0],
        placement=placement_spec,
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
            layers=fade_layers,
            halftone_mm=fade_halftone,
            halftone_angle=fade_halftone_angle,
            invert=fade_invert,
            cutoff=fade_cutoff,
        ),
        glaze=GlazeSpec(
            enabled=glaze_on,
            glass=glass,
            palette=glaze_palette,
            colours=glaze_colours,
        ),
    )
    # Placement, colour and fade stay shared across overlays here; what differs
    # per motif on the command line is what to cut out of it and how it sits
    # over what is beneath. The full per-layer spec is in the UI and the bridge.
    if len(overlay_rasters) > 1:
        spec = replace(
            spec,
            layers=[
                LayerSpec(
                    keep=one_keep,
                    opacity=one_opacity,
                    blend=one_blend,
                    placement=placement_spec,
                    color=color_spec,
                    fade=spec.fade,
                )
                for one_keep, one_opacity, one_blend in zip(keeps, opacities, blends)
            ],
        )

    backends = Backends()
    result = compose(base_raster, overlay_rasters, spec, backends)

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

    art = " + ".join(path.name for path in overlay)
    typer.secho(f"composed {base.name} + {art}", fg=typer.colors.GREEN)
    for index, layer in enumerate(result.layers):
        label = "overlay plan" if len(result.layers) == 1 else f"overlay {index + 1} plan"
        typer.echo(f"  {label:<13}: {layer.plan.describe()}  [{layer.plan.source}]")
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
def glaze(
    artwork: Path = typer.Argument(..., exists=True, dir_okay=False),
    glass: str = typer.Option("#ffffff", "--glass", help="Colour of the glass you are printing on."),
    palette: str = typer.Option("", "--palette", help="Inks to glaze with, e.g. cyan,magenta,yellow,#7a2f8a"),
    colours: int = typer.Option(5, "--colours", help="How many of the artwork's colours to solve for."),
    max_per_ink: int = typer.Option(3, "--max-per-ink"),
    max_total: int = typer.Option(5, "--max-total"),
    keep: str = typer.Option("", "--keep", "-k", help="What to keep from the artwork first."),
) -> None:
    """Work out how to build the artwork's colours from stacked ink on tinted glass."""
    from .colors import to_hex
    from .glaze import palette_from, plan as build_plan_for_glaze
    from .pattern import apply_cutout
    from .segment import evaluate

    raster = Raster.open(artwork)
    backends = Backends()
    cutout = evaluate(build_plan(keep, raster), raster, backends)
    art = apply_cutout(raster, cutout)

    glass_rgb = parse_color(glass) or (255, 255, 255)
    result = build_plan_for_glaze(
        art[:, :, :3].astype("float32") / 255.0,
        cutout,
        glass_rgb,
        palette_from(palette),
        colours=colours,
        max_per_ink=max_per_ink,
        max_total=max_total,
    )

    typer.echo(f"{artwork.name} on {to_hex(glass_rgb)} glass")
    typer.echo(f"  palette: {', '.join(ink.name for ink in result.palette)}")
    typer.echo(f"  printing plan: {len(result.stack)} passes — " +
               ", ".join(f"{ink.name} #{i}" for ink, i in result.stack))
    typer.echo("")
    for recipe in result.recipes:
        mark = " " if recipe.reachable else "!"
        typer.echo(
            f" {mark} {to_hex(recipe.target)} -> {to_hex(recipe.achieved)}"
            f"   {recipe.describe()}"
        )
        if recipe.note:
            typer.secho(f"     {recipe.note}", fg=typer.colors.YELLOW)
    _echo_notes(backends.notes)


def _lan_addresses() -> list[str]:
    """This machine's addresses on the local network, best guess first.

    The tablet needs one it can actually reach, and a desktop often has several
    — Wi-Fi, Ethernet, a VPN, and on Windows the virtual adapters that Hyper-V
    and WSL leave behind. Rather than pick one and be wrong, offer the one the
    system would route out of, then anything else worth trying.
    """
    import socket

    found: list[str] = []

    def add(address: str) -> None:
        # Loopback is no use to another device, and 169.254.x means the adapter
        # never got an address at all.
        if address and address not in found and not address.startswith(("127.", "169.254.")):
            found.append(address)

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(0.2)
        # Nothing is sent; connecting a UDP socket just consults the routing
        # table for the interface it would leave by.
        probe.connect(("8.8.8.8", 53))
        add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass

    try:
        add_all = socket.gethostbyname_ex(socket.gethostname())[2]
        for address in add_all:
            add(address)
    except OSError:
        pass

    return found


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open a browser window."),
    lan: bool = typer.Option(
        False, "--lan", help="Also accept connections from other devices on your network."
    ),
) -> None:
    """Run the local web interface."""
    import uvicorn

    from .server import create_app

    bind = "0.0.0.0" if lan else host
    url = f"http://{'127.0.0.1' if lan else host}:{port}/"
    typer.secho(f"glassprint running at {url}", fg=typer.colors.GREEN)

    if lan:
        addresses = _lan_addresses()
        if addresses:
            typer.secho(
                f"On your iPad or phone, open:  http://{addresses[0]}:{port}/",
                fg=typer.colors.CYAN,
                bold=True,
            )
            typer.echo("  (same Wi-Fi network, and leave this window open)")
            for other in addresses[1:]:
                typer.echo(f"  if that one does not work, try:  http://{other}:{port}/")
        else:
            typer.secho(
                "Could not work out this machine's network address — check Wi-Fi is on.",
                fg=typer.colors.YELLOW,
            )
        if sys.platform == "win32":
            # The prompt appears behind other windows often enough that people
            # miss it, then blame the address.
            typer.echo(
                "  Windows may ask whether to allow Python through the firewall —\n"
                "  say yes for private networks, or the tablet cannot connect."
            )

    typer.echo("Press Ctrl+C to stop.")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=bind, port=port, log_level="warning")


@app.command("chart")
def chart_command(
    out: Path = typer.Option(Path("glassprint-colour-chart.png"), "--out", "-o"),
    dpi: float = typer.Option(600.0, "--dpi"),
    substrate: str = typer.Option(
        "transparent", "--on", help="transparent | opaque | white — what the ink sits on."
    ),
) -> None:
    """Write the colour chart to print, then photograph, then `calibrate`."""
    from .measure import CHART, Profile, chart

    if substrate not in ("transparent", "opaque", "white"):
        raise typer.BadParameter("--on must be transparent, opaque or white")

    chart(dpi=dpi, substrate=substrate).save(out, fmt="png", dpi=(dpi, dpi))
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    typer.echo(f"  {CHART.width_mm:.0f} x {CHART.height_mm:.0f} mm, {CHART.columns * CHART.rows} cells")
    typer.echo(f"  print at 100%, one pass, {'WITH' if substrate == 'white' else 'no'} white base")
    if substrate in Profile.REFLECTIVE:
        typer.echo("  photograph it FRONT-LIT on white paper — nothing to backlight through it")
    else:
        typer.echo("  HOLD IT UP to a window — not laid on paper, which doubles every reading")


@app.command("calibrate")
def calibrate_command(
    photo: Path = typer.Argument(..., exists=True, dir_okay=False, help="Photo of the printed chart."),
    out: Path = typer.Option(Path("glassprint-profile.json"), "--out", "-o"),
    glass: str = typer.Option("", "--glass", help="Override the glass colour rather than reading it."),
    substrate: str = typer.Option(
        "transparent", "--on", help="transparent | opaque | white — what it was printed on."
    ),
) -> None:
    """Turn a photograph of the printed chart into a profile of your printer."""
    from .colors import parse_color
    from .measure import ReadError, read

    if substrate not in ("transparent", "opaque", "white"):
        raise typer.BadParameter("--on must be transparent, opaque or white")

    raster = Raster.from_bytes(photo.read_bytes(), name=photo.stem)
    try:
        profile = read(
            raster.rgba[:, :, :3],
            glass=parse_color(glass) if glass else None,
            substrate=substrate,
        )
    except ReadError as exc:
        raise typer.BadParameter(str(exc))

    out.write_text(profile.to_json())
    residuals = profile.residuals()
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    typer.echo(f"  {'ground':<10} {to_hex(profile.glass)}  ({substrate})")
    typer.echo(f"  tone curve {', '.join(f'{g:.2f}' for g in profile.gamma)}  (r, g, b)")
    typer.echo(f"  error      {residuals['held_out']} levels on patches it never saw")
    typer.echo(f"  was        {residuals['naive']} levels, uncalibrated")
    if residuals["held_out"] and residuals["held_out"] > 8:
        typer.secho(
            "  that is high — check the photo is flat, evenly lit and not in HDR",
            fg=typer.colors.YELLOW,
        )


if __name__ == "__main__":  # pragma: no cover
    app()
