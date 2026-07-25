"""glassprint — overlay artwork onto a base image and prepare it for UV printing.

Typical use::

    from glassprint import Raster, ComposeSpec, Placement, ColorSpec, compose, export

    base = Raster.open("panel.png")
    art = Raster.open("pattern.jpg")
    result = compose(base, art, ComposeSpec(keep="keep the flowers, remove the white background"))
    export(result, "out/")
"""

from .colors import parse_color, to_hex
from .compose import ComposeResult, ComposeSpec, compose
from .export import ExportSpec, export
from .nl import build_plan, parse as parse_instruction
from .pattern import Placement, PatternInfo, analyse
from .raster import Raster
from .recolor import ColorSpec
from .segment import Backends, MaskOp, MaskPlan, Selector

__version__ = "0.1.0"

__all__ = [
    "Backends",
    "ColorSpec",
    "ComposeResult",
    "ComposeSpec",
    "ExportSpec",
    "MaskOp",
    "MaskPlan",
    "PatternInfo",
    "Placement",
    "Raster",
    "Selector",
    "analyse",
    "build_plan",
    "compose",
    "export",
    "parse_color",
    "parse_instruction",
    "to_hex",
    "__version__",
]
