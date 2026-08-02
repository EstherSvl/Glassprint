"""The composition pipeline.

Base image + overlay artwork + a sentence about what to keep, in; a composite
and a standalone overlay layer, out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace as dataclass_replace

import numpy as np
from scipy import ndimage

from . import fade as fade_module
from .colors import parse_color, to_hex
from . import masks, nl, pattern, recolor, segment
from .fade import Fade, check as fade_check
from .glaze import GlazePlan, palette_from, plan as glaze_plan_for
from .pattern import Box, PatternInfo, Placement
from .raster import MM_PER_INCH, Raster
from .recolor import ColorSpec
from .segment import Backends, MaskPlan

TARGET_MODES = ("alpha", "describe", "largest", "full", "rect")
BLEND_MODES = ("normal", "multiply", "screen", "overlay")


@dataclass
class GlazeSpec:
    """Build the artwork's colours by stacking layers of different ink."""

    enabled: bool = False
    glass: str = "#ffffff"
    palette: str = ""          # "cyan,magenta,yellow" or hex values
    colours: int = 5           # how many of the artwork's colours to solve for
    max_per_ink: int = 3
    max_total: int = 5


@dataclass
class LayerSpec:
    """Everything that can differ between one overlay and the next.

    Split out when the tool learned to take more than one piece of artwork.
    The rest of :class:`ComposeSpec` — which part of the base to fill, whether
    to clip, the glaze — is a property of the *panel* and stays shared, because
    two motifs on one plate are still one print.
    """

    keep: str = ""
    edge_feather: float = 0.0
    opacity: float = 1.0
    blend: str = "normal"
    placement: Placement = field(default_factory=Placement)
    color: ColorSpec = field(default_factory=ColorSpec)
    fade: Fade = field(default_factory=Fade)

    def validated(self) -> "LayerSpec":
        if self.blend not in BLEND_MODES:
            raise ValueError(f"unknown blend mode {self.blend!r}; choose from {', '.join(BLEND_MODES)}")
        return dataclass_replace(
            self, placement=self.placement.normalised(), fade=self.fade.validated()
        )


@dataclass
class ComposeSpec:
    # What of the overlay to keep.
    keep: str = ""
    tolerance: float = 1.0
    use_claude: bool = False

    # Where on the base it goes.
    target: str = "alpha"
    target_describe: str = ""
    target_rect: tuple[float, float, float, float] | None = None  # fractions of the base

    # How it sits there.
    clip_to_shape: bool = True
    shape_grow: float = 0.0        # pixels; negative chokes the shape inwards
    shape_feather: float = 0.0     # pixels
    edge_feather: float = 0.0      # pixels, on the overlay cutout
    opacity: float = 1.0
    blend: str = "normal"

    placement: Placement = field(default_factory=Placement)
    color: ColorSpec = field(default_factory=ColorSpec)
    fade: Fade = field(default_factory=Fade)
    glaze: GlazeSpec = field(default_factory=lambda: GlazeSpec())

    #: Per-overlay settings. Empty means every overlay uses the fields above,
    #: which is what a single-overlay caller has always got.
    layers: list[LayerSpec] = field(default_factory=list)

    def validated(self) -> "ComposeSpec":
        if self.target not in TARGET_MODES:
            raise ValueError(f"unknown target mode {self.target!r}; choose from {', '.join(TARGET_MODES)}")
        if self.blend not in BLEND_MODES:
            raise ValueError(f"unknown blend mode {self.blend!r}; choose from {', '.join(BLEND_MODES)}")
        return dataclass_replace(
            self,
            placement=self.placement.normalised(),
            fade=self.fade.validated(),
            layers=[layer.validated() for layer in self.layers],
        )

    def for_layer(self, index: int) -> LayerSpec:
        """Settings for one overlay, falling back to the shared ones.

        A caller that never asked for more than one overlay gets exactly the
        behaviour it always had: the top-level fields, unchanged.
        """
        if index < len(self.layers):
            return self.layers[index]
        return LayerSpec(
            keep=self.keep,
            edge_feather=self.edge_feather,
            opacity=self.opacity,
            blend=self.blend,
            placement=self.placement,
            color=self.color,
            fade=self.fade,
        )


@dataclass
class ComposeResult:
    composite: Raster
    overlay_layer: Raster
    base: Raster
    shape_mask: np.ndarray
    cutout_mask: np.ndarray
    plan: MaskPlan
    info: PatternInfo
    box: Box
    fade: Fade = field(default_factory=Fade)
    fade_elements: int = 0
    faintest_ink: float = 0.0
    #: Ink layers at each pixel when the fade is stacked, otherwise None.
    layer_map: np.ndarray | None = None
    #: Artwork coverage ignoring the layer stepping — what one pass lays down.
    coverage: np.ndarray | None = None
    #: The fade's opacity ramp, so glaze passes can be dropped along it.
    fade_field: np.ndarray | None = None
    #: Per-colour glaze recipes, when glazing is on.
    glaze_plan: "GlazePlan | None" = None
    #: One entry per overlay, in the order they were composited.
    layers: list["LayerResult"] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def faintest_alpha(self) -> float:
        """The thinnest ink that will actually be laid down, as 0..1 coverage.

        Measured only where the artwork was solid to begin with, so soft
        anti-aliased edges — which every image has — do not drag it to zero.

        Read this against :data:`glassprint.fade.ALPHA_CLIFF`, not against a
        dither floor: on a EufyMake E1 anything under about 50% alpha prints as
        nothing at all.
        """
        return self.faintest_ink

    def summary(self) -> dict:
        left, top, right, bottom = self.box
        dpi_x, dpi_y = self.base.effective_dpi
        return {
            "base_size": list(self.base.size),
            "base_dpi": [round(dpi_x, 2), round(dpi_y, 2)],
            "base_size_mm": [round(v, 2) for v in self.base.size_mm],
            "shape_box": [left, top, right, bottom],
            "shape_coverage": round(masks.coverage(self.shape_mask), 4),
            "cutout_coverage": round(masks.coverage(self.cutout_mask), 4),
            "plan": self.plan.describe(),
            "plan_source": self.plan.source,
            "plan_explanation": self.plan.explanation,
            "pattern": {
                "is_pattern": self.info.is_pattern,
                "seamless": self.info.seamless,
                "components": self.info.components,
                "coverage": round(self.info.coverage, 4),
                "suggested_fit": self.info.suggested_fit,
                "suggested_repeats": self.info.suggested_repeats,
                "reason": self.info.reason,
            },
            "fade": {
                "mode": self.fade.mode,
                "describe": self.fade.describe(),
                "scope": self.fade.what.strip(),
                "elements": self.fade_elements,
                "dissolve": self.fade.dissolve,
                "cutoff": self.fade.cutoff,
                "layers": self.fade.layers,
                # The faintest ink that will actually be laid down. Under about
                # 50% alpha the E1 prints nothing — see fade.ALPHA_CLIFF.
                "faintest_alpha": self.faintest_alpha(),
            },
            "glaze": self.glaze_plan.as_dict() if self.glaze_plan else None,
            # One entry per piece of artwork. The fields above describe the
            # first, because that is what a single-overlay caller has always
            # read them as; this is where the rest of the stack shows up.
            "layers": [
                {
                    "name": layer.name,
                    "plan": layer.plan.describe(),
                    "plan_source": layer.plan.source,
                    "blend": layer.blend,
                    "is_pattern": layer.info.is_pattern,
                    "coverage": round(masks.coverage(layer.cutout), 4),
                }
                for layer in self.layers
            ],
            "notes": self.notes,
        }


def resolve_shape(
    base: Raster,
    spec: ComposeSpec,
    backends: Backends,
) -> tuple[np.ndarray, list[str]]:
    """Work out which part of the base image the overlay should land on."""
    notes: list[str] = []
    shape = (base.height, base.width)
    mode = spec.target

    if mode == "alpha" and not base.has_alpha:
        notes.append(
            "The base image has no transparency, so there is no cut-out shape to fill — "
            "used the largest solid region instead. Export from Procreate or Affinity as "
            "PNG with a transparent background to target an exact shape."
        )
        mode = "largest"

    if mode == "full":
        return masks.ones(shape), notes

    if mode == "alpha":
        return masks.clean(base.alpha_f), notes

    if mode == "rect":
        if not spec.target_rect:
            notes.append("No rectangle given for the target area — used the whole canvas.")
            return masks.ones(shape), notes
        left, top, right, bottom = spec.target_rect
        mask = masks.zeros(shape)
        x0 = int(round(np.clip(left, 0, 1) * base.width))
        x1 = int(round(np.clip(right, 0, 1) * base.width))
        y0 = int(round(np.clip(top, 0, 1) * base.height))
        y1 = int(round(np.clip(bottom, 0, 1) * base.height))
        mask[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)] = 1.0
        return mask, notes

    if mode == "describe":
        instruction = spec.target_describe.strip()
        if not instruction:
            notes.append("No description given for the target area — used the whole canvas.")
            return masks.ones(shape), notes
        plan = nl.build_plan(
            instruction, base, use_claude=spec.use_claude, tolerance=spec.tolerance
        )
        mask = segment.evaluate(plan, base, backends)
        mask = masks.fill_holes(masks.despeckle(mask, 0.0008))
        if masks.coverage(mask) < 0.001:
            notes.append(
                f"Could not find '{instruction}' on the base image — used the whole canvas."
            )
            return masks.ones(shape), notes
        return mask, notes

    # largest solid region
    background = segment.background_mask(base, tolerance=spec.tolerance)
    mask = masks.invert(background)
    mask = masks.fill_holes(masks.despeckle(mask, 0.001))
    mask = masks.largest_component(mask)
    if masks.coverage(mask) < 0.001:
        notes.append("Could not find a distinct shape on the base image — used the whole canvas.")
        return masks.ones(shape), notes
    return mask, notes


@dataclass
class LayerResult:
    """One overlay, cut out, placed, faded and ready to blend."""

    rgba: np.ndarray               # placed artwork, alpha already faded/clipped
    cutout: np.ndarray
    plan: MaskPlan
    info: PatternInfo
    coverage: np.ndarray
    faintest: float
    blend: str
    fade: Fade
    layer_map: np.ndarray | None = None
    fade_field: np.ndarray | None = None
    elements: int = 0
    name: str = ""


def _build_layer(
    base: Raster,
    overlay: Raster,
    layer: LayerSpec,
    spec: ComposeSpec,
    shaped: np.ndarray,
    box: Box,
    backends: Backends,
    notes: list[str],
) -> LayerResult:
    """Everything that happens to one piece of artwork, start to finish.

    Lifted out of ``compose`` unchanged when the tool learned to take several
    overlays. The shape on the base is resolved once and handed in; this runs
    per overlay.
    """
    # 2. Which part of the overlay are we using?
    plan = nl.build_plan(layer.keep, overlay, use_claude=spec.use_claude, tolerance=spec.tolerance)
    cutout = segment.evaluate(plan, overlay, backends)
    if layer.edge_feather:
        cutout = masks.feather(cutout, layer.edge_feather)

    # 3. Read the artwork's language and place it.
    info = pattern.analyse(
        overlay,
        cutout,
        target_width_px=box[2] - box[0],
        target_dpi=base.effective_dpi[0],
    )
    art = pattern.apply_cutout(overlay, cutout)
    art = recolor.apply(art, layer.color)

    placed = pattern.place(
        art,
        base.size,
        box,
        layer.placement,
        info,
        target_dpi=base.effective_dpi[0],
    )

    # 4. Fade into the glass, then clip to the shape and apply opacity.
    layer_alpha = placed[:, :, 3].astype(np.float32) / 255.0
    faded_elements = 0
    layer_map: np.ndarray | None = None
    fade_field: np.ndarray | None = None

    if layer.fade.active:
        opacity_field = fade_module.ramp(
            layer.fade, (base.height, base.width), box, shaped
        )
        if layer.fade.stacked:
            opacity_field, layer_map, stack_notes = _stack(opacity_field, layer.fade)
            notes.extend(stack_notes)
        elif layer.fade.screened:
            opacity_field, screen_notes = _screen(opacity_field, base, layer.fade)
            notes.extend(screen_notes)
        notes.extend(fade_check(layer.fade, pattern=info.is_pattern))
        scope, scope_note = _fade_scope(overlay, cutout, base, box, spec, layer, info, backends)
        if scope_note:
            notes.append(scope_note)
        # A dot screen is its own way of expressing the ramp, so the
        # element-level ones stand down rather than averaging the dots away.
        element_spec = (
            dataclass_replace(layer.fade, per_element=False, dissolve=0.0)
            if (layer.fade.screened or layer.fade.stacked)
            else layer.fade
        )
        layer_alpha, faded_elements = fade_module.apply(
            layer_alpha, opacity_field, element_spec, scope
        )
        if layer.fade.carrier == "ink":
            # The ramp goes into the colour rather than the alpha, because on
            # glass with no white pass alpha is thresholded and the tail never
            # prints. Same ramp, resolved once and used for both.
            keep, _ = fade_module.resolve(layer_alpha, opacity_field, element_spec, scope)
            placed = fade_module.as_ink(placed, keep)
        fade_field = opacity_field

    # What a single printed pass lays down, before the layer stepping. With a
    # stacked fade this is the shape of every pass; the layer map says how many
    # passes each region gets.
    coverage = placed[:, :, 3].astype(np.float32) / 255.0
    if layer_map is not None:
        coverage = coverage * (layer_map > 0).astype(np.float32)

    if spec.clip_to_shape:
        layer_alpha = layer_alpha * shaped
        coverage = coverage * shaped
    layer_alpha = layer_alpha * float(np.clip(layer.opacity, 0.0, 1.0))
    coverage = coverage * float(np.clip(layer.opacity, 0.0, 1.0))
    layer_alpha = fade_module.apply_cutoff(layer_alpha, layer.fade.cutoff)

    overlay_layer = placed.copy()
    overlay_layer[:, :, 3] = np.clip(layer_alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    faintest = _faintest_ink(placed, layer_alpha, shaped if spec.clip_to_shape else None)

    return LayerResult(
        rgba=overlay_layer,
        cutout=cutout,
        plan=plan,
        info=info,
        coverage=coverage,
        faintest=faintest,
        blend=layer.blend,
        fade=layer.fade,
        layer_map=layer_map,
        fade_field=fade_field,
        elements=faded_elements,
        name=overlay.name or "overlay",
    )


def compose(
    base: Raster,
    overlay: "Raster | Sequence[Raster]",
    spec: ComposeSpec | None = None,
    backends: Backends | None = None,
) -> ComposeResult:
    """Put one or more pieces of artwork onto a base image.

    ``overlay`` takes a single raster or several. Several are composited in
    order, each with its own entry in ``spec.layers`` — or, if that is empty,
    all sharing the top-level settings.
    """
    overlays = [overlay] if isinstance(overlay, Raster) else list(overlay)
    if not overlays:
        raise ValueError("compose needs at least one overlay")
    spec = (spec or ComposeSpec()).validated()
    backends = backends or Backends()
    notes: list[str] = []

    # 1. Which part of the base are we filling?
    shape_mask, shape_notes = resolve_shape(base, spec, backends)
    notes.extend(shape_notes)

    shaped = shape_mask
    if spec.shape_grow:
        shaped = masks.grow(shaped, spec.shape_grow)
    if spec.shape_feather:
        shaped = masks.feather(shaped, spec.shape_feather)

    box = masks.bbox(shaped, threshold=0.35) or (0, 0, base.width, base.height)

    # 2-4. Each overlay in turn: cut out, place, colour, fade. The panel-level
    # decisions above are shared; everything below the shape is per artwork.
    layers = [
        _build_layer(base, art, spec.for_layer(index), spec, shaped, box, backends, notes)
        for index, art in enumerate(overlays)
    ]

    # The overlay-only export is the whole stack, alpha-over in order, so the
    # file you hand the printer carries every motif. The composite blends each
    # layer onto the base with *its own* mode, which is not the same operation:
    # multiply against the panel is not multiply against the motif beneath.
    # The stack starts *as* the first layer rather than as an empty canvas.
    # Compositing onto transparency looks like a no-op and is not: it divides
    # the colour back out by an alpha of zero, so every pixel the artwork does
    # not cover comes back black. The glaze solver reads those colours, and a
    # dot screen leaves holes all through a region it still has to find a
    # recipe for.
    stack = layers[0].rgba
    composite = base.rgba
    for layer in layers:
        if layer is not layers[0]:
            stack = _blend_over(stack, layer.rgba, "normal")
        composite = _blend_over(composite, layer.rgba, layer.blend)

    overlay_layer = stack
    first = layers[0]
    cutout, plan, info = first.cutout, first.plan, first.info
    faded_elements = sum(layer.elements for layer in layers)
    faintest = min((layer.faintest for layer in layers if layer.faintest > 0), default=0.0)

    # One pass lays down every layer, so coverage is the union.
    coverage = layers[0].coverage
    for layer in layers[1:]:
        coverage = np.maximum(coverage, layer.coverage)

    # A stacked fade counts printed passes, and two overlays each asking for a
    # different number of them is a question this cannot answer. The first
    # layer's map is used and the ambiguity is said out loud rather than
    # averaged into something that looks decided.
    layer_map = first.layer_map
    fade_field = first.fade_field
    if len(layers) > 1 and any(layer.layer_map is not None for layer in layers[1:]):
        notes.append(
            "More than one overlay is using a stacked fade. The pass count follows the "
            "first; the others print as a single pass each."
        )

    plan_ = None
    if spec.glaze.enabled:
        glass = parse_color(spec.glaze.glass) or (255, 255, 255)
        plan_ = glaze_plan_for(
            overlay_layer[:, :, :3].astype(np.float32) / 255.0,
            coverage,
            glass,
            palette_from(spec.glaze.palette),
            colours=spec.glaze.colours,
            max_per_ink=spec.glaze.max_per_ink,
            max_total=spec.glaze.max_total,
        )
        smooth = [
            each.fade for each in layers
            if each.fade.active
            and not (each.fade.screened or each.fade.stacked or each.fade.dissolve > 0)
        ]
        if smooth:
            notes.append(
                "A smooth fade eats the glaze stack itself, so the correction unwinds as it "
                "thins and the colour reverts toward the bare glass — with only a few passes "
                "that reads as a hard edge, not a fade. Fade by coverage instead (a dot "
                "screen or dissolve): every dot keeps the whole stack, so the colour stays "
                "right the whole way down."
            )
        for recipe in plan_.recipes:
            if recipe.note:
                notes.append(f"Glaze — {to_hex(recipe.target)}: {recipe.note}")


    # 5. Already blended above, one layer at a time.
    notes.extend(backends.notes)
    return ComposeResult(
        composite=Raster(composite, dpi=base.dpi, source_format=base.source_format, name=base.name),
        overlay_layer=Raster(overlay_layer, dpi=base.dpi, name=first.name),
        base=base,
        shape_mask=shaped,
        cutout_mask=cutout,
        plan=plan,
        info=info,
        box=box,
        fade=first.fade,
        fade_elements=faded_elements,
        faintest_ink=faintest,
        layer_map=layer_map,
        coverage=coverage,
        fade_field=fade_field,
        glaze_plan=plan_,
        layers=layers,
        notes=notes,
    )


def _faintest_ink(
    placed: np.ndarray, layer_alpha: np.ndarray, shaped: np.ndarray | None
) -> float:
    """The thinnest ink laid down, ignoring anti-aliased edges.

    Only pixels that were solid in the artwork *and* well inside the target
    shape count. The printed area is then eroded, because every edge — a curve
    in the artwork, the rim of a halftone dot — carries a soft pixel or two
    that would otherwise report as near-zero ink on an image that is in fact
    printing at full strength everywhere.
    """
    solid = placed[:, :, 3] > 242
    if shaped is not None:
        solid = solid & (shaped > 0.95)

    printed = solid & (layer_alpha > 0.002)
    if not printed.any():
        return 0.0

    interior = ndimage.binary_erosion(printed, iterations=2)
    values = layer_alpha[interior if interior.any() else printed]
    if not values.size:
        return 0.0

    # A low percentile rather than the outright minimum: a genuinely faint
    # region covers area, while a stray dim pixel — the dimple left where four
    # halftone dots meet, say — is far too small to print as anything.
    return round(float(np.percentile(values, 1.0)), 3)


def _stack(
    opacity_field: np.ndarray, fade: Fade
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Step the ramp into printed ink layers."""
    notes: list[str] = []
    if fade.screened or fade.per_element or fade.dissolve > 0:
        notes.append(
            "Ink layers, the dot screen and dissolve are three ways of expressing the "
            "same fade, so the layers were used and the others left off."
        )
    stepped, layer_map = fade_module.quantise(opacity_field, fade.layers)
    return stepped, layer_map, notes


def _screen(
    opacity_field: np.ndarray, base: Raster, fade: Fade
) -> tuple[np.ndarray, list[str]]:
    """Turn the ramp into a dot screen, warning if the pitch is too fine."""
    notes: list[str] = []
    dpi = base.effective_dpi[0]
    pitch_px = fade.halftone_mm / MM_PER_INCH * dpi

    if fade.halftone_mm < 0.8:
        notes.append(
            f"A {fade.halftone_mm:g}mm dot screen is fine enough to beat against the "
            "printer's own halftone and moiré. Coarse dots read as a deliberate "
            "texture — 1mm and up is the safe range."
        )
    if fade.per_element or fade.dissolve > 0:
        notes.append(
            "The dot screen and the per-element controls are two ways of expressing the "
            "same fade, so the screen was used and dissolve left off."
        )

    screened = fade_module.halftone(opacity_field, pitch_px, fade.halftone_angle)
    return screened, notes


def _fade_scope(
    overlay: Raster,
    cutout: np.ndarray,
    base: Raster,
    box: Box,
    spec: ComposeSpec,
    layer: LayerSpec,
    info: PatternInfo,
    backends: Backends,
) -> tuple[np.ndarray | None, str | None]:
    """Which of the placed elements the fade is allowed to touch.

    The selector runs against the *source* artwork, before recolouring, so
    "fade the leaves" still means the leaves after you have tinted everything
    one colour. The resulting mask is then placed with the same settings as the
    artwork, so it tiles in step with it.
    """
    what = layer.fade.what.strip()
    if not what:
        return None, None

    plan = nl.build_plan(what, overlay, use_claude=spec.use_claude, tolerance=spec.tolerance)
    scope_mask = masks.intersect(segment.evaluate(plan, overlay, backends), cutout)
    if masks.coverage(scope_mask) < 1e-5:
        return (
            masks.zeros((base.height, base.width)),
            f"Nothing in the overlay matched '{what}', so the fade was left off.",
        )

    carrier = np.zeros((overlay.height, overlay.width, 4), dtype=np.uint8)
    carrier[:, :, 3] = np.clip(scope_mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
    placed = pattern.place(
        carrier, base.size, box, layer.placement, info, target_dpi=base.effective_dpi[0]
    )
    return placed[:, :, 3].astype(np.float32) / 255.0, None


def _blend_over(base_rgba: np.ndarray, layer_rgba: np.ndarray, mode: str) -> np.ndarray:
    base_rgb = base_rgba[:, :, :3].astype(np.float32) / 255.0
    base_a = base_rgba[:, :, 3].astype(np.float32) / 255.0
    layer_rgb = layer_rgba[:, :, :3].astype(np.float32) / 255.0
    layer_a = layer_rgba[:, :, 3].astype(np.float32) / 255.0

    if mode == "multiply":
        blended = base_rgb * layer_rgb
    elif mode == "screen":
        blended = 1.0 - (1.0 - base_rgb) * (1.0 - layer_rgb)
    elif mode == "overlay":
        blended = np.where(
            base_rgb <= 0.5,
            2.0 * base_rgb * layer_rgb,
            1.0 - 2.0 * (1.0 - base_rgb) * (1.0 - layer_rgb),
        )
    else:
        blended = layer_rgb

    # Blend modes only apply where the base is actually opaque; over empty
    # canvas the overlay keeps its own colour.
    src_rgb = base_a[:, :, None] * blended + (1.0 - base_a[:, :, None]) * layer_rgb

    out_a = layer_a + base_a * (1.0 - layer_a)
    safe = np.maximum(out_a, 1e-6)[:, :, None]
    out_rgb = (
        src_rgb * layer_a[:, :, None] + base_rgb * base_a[:, :, None] * (1.0 - layer_a[:, :, None])
    ) / safe

    out = np.empty_like(base_rgba)
    out[:, :, :3] = np.clip(out_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out
