"""Talking to the tool instead of driving it.

Every setting in glassprint is reachable from a sentence. This module turns one
message into changes to the compose spec, says back what it did in the same
plain terms, and answers the questions that come up while you are working.

Two paths, same shape of answer:

* a rule-based reader that runs entirely offline. This is the default, and on a
  tablet it is the only one there will ever be — the browser build has no API
  key and no server behind it. It is written to be genuinely useful on its own,
  not to be a placeholder for the real thing;
* an optional Claude pass for messages the rules cannot untangle, which sees the
  conversation and the current settings and writes the same list of changes.

The rules run first either way. When Claude is available and the rules came back
empty-handed, the message escalates; when it is not, the tool says what it *did*
understand rather than failing silently.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .colors import COLOR_WORDS
from .fade import ALPHA_CLIFF

#: Every field a message is allowed to touch, and what it means in words. The
#: chat can only ever move these — it does not reach into the pipeline itself.
SETTABLE = {
    "keep": "what to cut out of the artwork",
    "target": "which part of the base to fill",
    "target_describe": "the described area on the base",
    "clip_to_shape": "whether the artwork is clipped to that area",
    "edge_feather": "how soft the cut edge is",
    "opacity": "how strong the artwork is",
    "blend": "how it mixes with what is under it",
    "placement.fit": "how the artwork is sized to the area",
    "placement.repeat_across": "repeats across the shape",
    "placement.repeat_mm": "the physical size of one repeat",
    "placement.scale": "scale",
    "placement.rotation": "rotation",
    "placement.offset_x": "horizontal position",
    "placement.offset_y": "vertical position",
    "placement.mirror": "whether alternate tiles mirror",
    "placement.flip_h": "horizontal flip",
    "placement.flip_v": "vertical flip",
    "color.mode": "the recolouring method",
    "color.color": "the colour",
    "color.color2": "the second colour",
    "color.from_color": "the colour being replaced",
    "color.saturation": "saturation",
    "color.brightness": "brightness",
    "color.contrast": "contrast",
    "color.hue_shift": "hue",
    "fade.mode": "the kind of fade",
    "fade.what": "which elements fade",
    "fade.angle": "the fade direction",
    "fade.start": "where the fade begins",
    "fade.end": "where the fade completes",
    "fade.curve": "the shape of the fade",
    "fade.min_alpha": "how faint the far end goes",
    "fade.dissolve": "whether elements drop out whole",
    "fade.layers": "how many printed ink layers the fade is built from",
    "fade.halftone_mm": "the dot screen pitch",
    "fade.cutoff": "the alpha below which nothing is printed",
    "fade.per_element": "whether each element gets one opacity",
    "glaze.enabled": "whether colours are built by stacking inks",
    "glaze.glass": "the colour of the glass",
}


@dataclass
class Turn:
    """One line of the conversation."""

    role: str  # "you" or "glassprint"
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "text": self.text}


@dataclass
class Change:
    """One setting moved, and how to say so."""

    path: str
    value: Any
    said: str
    layer: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "value": self.value, "said": self.said, "layer": self.layer}


@dataclass
class Reply:
    text: str
    spec: dict[str, Any]
    changes: list[Change] = field(default_factory=list)
    source: str = "rules"
    #: Things worth trying next, offered as text you could send back.
    suggestions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "spec": self.spec,
            "changes": [change.as_dict() for change in self.changes],
            "source": self.source,
            "suggestions": self.suggestions,
        }


# -- reading and writing the spec by path -----------------------------------


def get_path(spec: dict[str, Any], path: str) -> Any:
    node: Any = spec
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def set_path(spec: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = spec
    for part in parts[:-1]:
        nested = node.get(part)
        if not isinstance(nested, dict):
            nested = {}
            node[part] = nested
        node = nested
    node[parts[-1]] = value


def _number(spec: dict[str, Any], path: str, default: float) -> float:
    value = get_path(spec, path)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# -- which overlay is being talked about ------------------------------------

_ORDINALS = {
    "first": 0, "1st": 0, "one": 0, "1": 0,
    "second": 1, "2nd": 1, "two": 1, "2": 1,
    "third": 2, "3rd": 2, "three": 2, "3": 2,
    "fourth": 3, "4th": 3, "four": 3, "4": 3,
}

_LAYER_PATTERNS = [
    re.compile(r"\b(?:the )?(first|second|third|fourth|1st|2nd|3rd|4th)\s+overlay\b"),
    re.compile(r"\boverlay\s+(1|2|3|4|one|two|three|four)\b"),
    re.compile(r"\b(?:the )?(first|second|third|fourth)\s+(?:one|layer|motif|image)\b"),
]


def _which_layer(text: str, layer_count: int) -> tuple[int | None, str]:
    """Pull an overlay reference off the front of a message.

    Returns the index and the message with the reference removed, so "make the
    second overlay half strength" reaches the opacity rule as "make it half
    strength" rather than confusing it with a stray number.
    """
    if layer_count < 2:
        return None, text
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        index = _ORDINALS.get(match.group(1))
        if index is None or index >= layer_count:
            continue
        return index, (text[: match.start()] + " it " + text[match.end():]).strip()
    if re.search(r"\bthe top (?:one|layer|overlay)\b", text):
        return layer_count - 1, text
    if re.search(r"\bthe bottom (?:one|layer|overlay)\b", text):
        return 0, text
    return None, text


# -- the offline reader -----------------------------------------------------

Rule = Callable[[str, dict[str, Any]], Iterator[Change]]
_RULES: list[Rule] = []

#: People write "tile it four across", not "tile it 4 across". Spelled numbers
#: become numerals before anything else looks at the sentence, so every rule can
#: match digits and none of them has to know about the words.
_WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "ninety": "90", "a hundred": "100",
}
_WORD_NUMBER_RE = re.compile(
    r"(?<![a-z])(" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")(?![a-z])"
)


def normalise(message: str) -> str:
    text = re.sub(r"\s+", " ", (message or "").lower().strip()).replace("’", "'")
    return _WORD_NUMBER_RE.sub(lambda m: _WORD_NUMBERS[m.group(1)], text)


def rule(func: Rule) -> Rule:
    _RULES.append(func)
    return func


_NUM = r"(\d+(?:\.\d+)?)"


def _colour_in(text: str) -> str | None:
    hex_match = re.search(r"#[0-9a-fA-F]{3,8}\b", text)
    if hex_match:
        return hex_match.group(0)
    for word in re.findall(r"[a-z]+", text):
        if word in COLOR_WORDS:
            return word
    return None


@rule
def _placement(text: str, spec: dict[str, Any]) -> Iterator[Change]:
    if re.search(r"\b(tile|tiled|tiling|repeat it|repeating)\b", text):
        yield Change("placement.fit", "tile", "tiling the artwork")
    if re.search(r"\bfill the (shape|area)\b|\bfit the shape\b", text):
        yield Change("placement.fit", "shape", "fitting it to the shape")
    if re.search(r"\bstretch\b", text):
        yield Change("placement.fit", "stretch", "stretching it to the area")
    if re.search(r"\b(fit inside|contain|don'?t crop)\b", text):
        yield Change("placement.fit", "contain", "fitting the whole artwork inside")
    if re.search(r"\bcover\b", text) and "cover it" not in text:
        yield Change("placement.fit", "cover", "covering the area, cropping the overflow")

    across = re.search(rf"{_NUM}\s*(?:times |repeats? )?across", text) or re.search(
        rf"(?:repeat|tile)\s*(?:it\s*)?{_NUM}\b", text
    )
    if across:
        count = float(across.group(1))
        yield Change("placement.repeat_across", count, f"{count:g} repeats across the shape")
        yield Change("placement.fit", "tile", "tiling the artwork")

    repeat_mm = re.search(rf"{_NUM}\s*mm\s*(?:repeat|tile|across each)", text) or re.search(
        rf"repeats? (?:of|at|every)\s*{_NUM}\s*mm", text
    )
    if repeat_mm:
        size = float(repeat_mm.group(1))
        yield Change("placement.repeat_mm", size, f"one repeat every {size:g} mm")
        yield Change("placement.fit", "tile", "tiling the artwork")

    turn = re.search(rf"\b(?:rotate|turn|angle)\D{{0,12}}{_NUM}\s*(?:°|deg|degrees)?", text)
    if turn:
        yield Change("placement.rotation", float(turn.group(1)), f"rotated {turn.group(1)}°")

    scale = _number(spec, "placement.scale", 1.0)
    if re.search(r"\b(bigger|larger|scale up|blow it up)\b", text):
        yield Change("placement.scale", round(scale * 1.25, 3), "25% bigger")
    elif re.search(r"\b(smaller|scale down|shrink)\b", text):
        yield Change("placement.scale", round(scale * 0.8, 3), "20% smaller")

    if re.search(r"\b(mirror|mirrored) (?:the )?(?:alternate |every other )?tiles?\b", text) or (
        re.search(r"\bmirror\b", text) and "don't" not in text and "no " not in text
    ):
        yield Change("placement.mirror", "on", "mirroring alternate tiles")
    if re.search(r"\b(?:don'?t|no|stop) mirror", text):
        yield Change("placement.mirror", "off", "no mirroring")

    if re.search(r"\bflip (?:it )?(?:horizontally|left to right)\b", text):
        yield Change("placement.flip_h", True, "flipped horizontally")
    if re.search(r"\bflip (?:it )?(?:vertically|upside down|top to bottom)\b", text):
        yield Change("placement.flip_v", True, "flipped vertically")

    step = 0.2 if re.search(r"\ba lot\b|\bmuch\b|\bright\s+(?:up|down|over)\b", text) else 0.08
    move = re.search(r"\b(?:move|nudge|shift|push)\b[^.]{0,20}?\b(left|right|up|down)\b", text)
    if move:
        where = move.group(1)
        if where in ("left", "right"):
            base = _number(spec, "placement.offset_x", 0.0)
            value = round(base + (step if where == "right" else -step), 3)
            yield Change("placement.offset_x", value, f"moved {where}")
        else:
            base = _number(spec, "placement.offset_y", 0.0)
            value = round(base + (step if where == "down" else -step), 3)
            yield Change("placement.offset_y", value, f"moved {where}")


@rule
def _target(text: str, spec: dict[str, Any]) -> Iterator[Change]:
    if re.search(r"\b(?:the )?whole canvas\b|\beverywhere\b|\ball over it\b", text):
        yield Change("target", "full", "filling the whole canvas")
    elif re.search(r"\b(?:in|inside|within|on) the shape\b|\bthe cut ?-?out\b", text):
        yield Change("target", "alpha", "filling the shape you drew")
    elif re.search(r"\bthe largest (?:solid )?(?:area|region|shape)\b", text):
        yield Change("target", "largest", "filling the largest solid region")
    else:
        where = re.search(
            r"\b(?:put|place|only) (?:it|them|these|those)? ?(?:on|onto|in|inside) "
            r"(?:the |a )?([a-z][a-z '-]{2,40}?)(?:[,.]|$| and | but )",
            text,
        )
        if where:
            area = where.group(1).strip()
            yield Change("target", "describe", f"targeting '{area}' on the base")
            yield Change("target_describe", area, "")

    if re.search(r"\b(?:don'?t|do not|no) clip", text) or "let it overhang" in text:
        yield Change("clip_to_shape", False, "not clipping to the shape")
    elif re.search(r"\bclip (?:it )?to the shape\b|\bkeep it inside\b", text):
        yield Change("clip_to_shape", True, "clipping to the shape")

    soften = re.search(rf"\bsoften\D{{0,20}}{_NUM}\s*px", text)
    if soften:
        yield Change("edge_feather", float(soften.group(1)), f"cut edge softened {soften.group(1)}px")
    elif re.search(r"\bsoften the (?:cut )?edges?\b|\bfeather the edges?\b", text):
        yield Change("edge_feather", 2.0, "cut edge softened 2px")


@rule
def _colour(text: str, spec: dict[str, Any]) -> Iterator[Change]:
    replace = re.search(
        r"\b(?:replace|swap|change)\s+(?:the\s+)?(#[0-9a-f]{3,8}|[a-z]+)\s+"
        r"(?:with|for|to)\s+(?:the\s+)?(#[0-9a-f]{3,8}|[a-z]+)",
        text,
    )
    if replace and replace.group(1) in COLOR_WORDS or (replace and replace.group(1).startswith("#")):
        yield Change("color.mode", "replace", f"replacing {replace.group(1)} with {replace.group(2)}")
        yield Change("color.from_color", replace.group(1), "")
        yield Change("color.color", replace.group(2), "")
        return

    if re.search(r"\bduotone\b", text):
        yield Change("color.mode", "duotone", "duotone")
        colours = re.findall(r"#[0-9a-f]{3,8}\b|\b(?:" + "|".join(COLOR_WORDS) + r")\b", text)
        if len(colours) >= 2:
            yield Change("color.color", colours[0], f"highlight {colours[0]}")
            yield Change("color.color2", colours[1], f"shadow {colours[1]}")
    elif re.search(r"\b(greyscale|grayscale|mono|monochrome|black and white)\b", text):
        yield Change("color.mode", "mono", "greyscale")
    elif re.search(r"\b(?:no|stop|remove the|undo the) (?:re)?colour", text) or "leave the colours alone" in text:
        yield Change("color.mode", "none", "leaving the colours alone")
    else:
        tint = re.search(
            r"\b(?:tint|colour|color|recolour|recolor|make|turn|paint)\b[^.]{0,18}?"
            r"(#[0-9a-fA-F]{3,8}\b|\b(?:" + "|".join(COLOR_WORDS) + r")\b)",
            text,
        )
        if tint:
            colour = tint.group(1)
            yield Change("color.mode", "tint", f"tinted {colour}")
            yield Change("color.color", colour, "")

    contrast = _number(spec, "color.contrast", 1.0)
    if re.search(r"\b(more contrast|punchier|crisper|contrastier)\b", text):
        yield Change("color.contrast", round(contrast * 1.25, 3), "more contrast")
    elif re.search(r"\b(less contrast|flatter|softer contrast)\b", text):
        yield Change("color.contrast", round(contrast * 0.8, 3), "less contrast")

    saturation = _number(spec, "color.saturation", 1.0)
    if re.search(r"\b(desaturate|less saturated|washed out|muted)\b", text):
        yield Change("color.saturation", round(saturation * 0.7, 3), "less saturated")
    elif re.search(r"\b(more saturated|richer|punchier colour|vivid)\b", text):
        yield Change("color.saturation", round(saturation * 1.3, 3), "more saturated")

    brightness = _number(spec, "color.brightness", 1.0)
    if re.search(r"\b(brighter|lighter)\b", text):
        yield Change("color.brightness", round(brightness * 1.15, 3), "brighter")
    elif re.search(r"\b(darker|deeper)\b", text):
        yield Change("color.brightness", round(brightness * 0.85, 3), "darker")

    hue = _number(spec, "color.hue_shift", 0.0)
    if re.search(r"\bwarmer\b", text):
        yield Change("color.hue_shift", round(hue - 15, 1), "warmer")
    elif re.search(r"\bcooler\b", text):
        yield Change("color.hue_shift", round(hue + 15, 1), "cooler")


_DIRECTIONS = [
    (r"\b(?:down|downwards?|to the bottom|toward the bottom|from the top)\b", 90.0, "downward"),
    (r"\b(?:up|upwards?|to the top|toward the top|from the bottom)\b", 270.0, "upward"),
    (r"\b(?:right|to the right|toward the right)\b", 0.0, "to the right"),
    (r"\b(?:left|to the left|toward the left)\b", 180.0, "to the left"),
]


@rule
def _fade(text: str, spec: dict[str, Any]) -> Iterator[Change]:
    if re.search(r"\b(?:no|stop|remove the|drop the|undo the|cancel the) fad", text):
        yield Change("fade.mode", "none", "fade off")
        return

    fading = re.search(r"\bfad(?:e|es|ed|ing)\b", text)
    if fading:
        if re.search(r"\b(?:radial|from the cent(?:re|er)|out from the middle)\b", text):
            yield Change("fade.mode", "radial", "fading out from the centre")
        elif re.search(r"\b(?:with the shape|toward the edges?|inward from the edge)\b", text):
            yield Change("fade.mode", "shape", "fading toward the edges of the shape")
        else:
            for pattern, angle, said in _DIRECTIONS:
                if re.search(pattern, text[fading.start():]):
                    yield Change("fade.mode", "linear", f"fading {said}")
                    yield Change("fade.angle", angle, "")
                    break
            else:
                yield Change("fade.mode", "linear", "fading downward")
                yield Change("fade.angle", 90.0, "")

        what = re.search(r"\bfade (?:out )?(?:just |only )?the ([a-z][a-z '-]{2,30}?)(?:[,.]|$| and | over | into | toward )", text)
        if what and what.group(1).strip() not in ("artwork", "image", "whole thing", "pattern"):
            scope = what.group(1).strip()
            yield Change("fade.what", scope, f"only the {scope} fades")

    layers = re.search(rf"{_NUM}\s*(?:ink )?(?:layers?|passes)", text)
    if layers:
        count = int(float(layers.group(1)))
        yield Change("fade.layers", count, f"built from {count} printed ink layers")

    if re.search(r"\b(?:dot screen|halftone|half-tone|manga|comic)\b", text):
        pitch = re.search(rf"{_NUM}\s*mm", text)
        size = float(pitch.group(1)) if pitch else 1.5
        yield Change("fade.halftone_mm", size, f"a {size:g}mm dot screen")
    elif re.search(rf"{_NUM}\s*mm dots?", text):
        size = float(re.search(rf"{_NUM}\s*mm dots?", text).group(1))
        yield Change("fade.halftone_mm", size, f"a {size:g}mm dot screen")

    if re.search(r"\bdissolve\b|\bdrop (?:elements|them) out whole\b|\bwhole (?:flowers|elements|motifs) drop\b", text):
        yield Change("fade.dissolve", 1.0, "elements dropping out whole rather than thinning")

    if re.search(r"\bfade to a ghost\b|\bdon'?t fade to nothing\b|\bkeep a hint\b", text):
        yield Change("fade.min_alpha", 0.15, "fading to a ghost rather than to nothing")

    if re.search(r"\b(?:faster|quicker|sharper) fade\b|\bfade (?:off )?quickly\b", text):
        yield Change("fade.start", 0.35, "the fade starts later and finishes fast")
    elif re.search(r"\b(?:slower|gentler|gradual) fade\b", text):
        yield Change("fade.start", 0.0, "a longer, gentler fade")
        yield Change("fade.end", 1.0, "")

    if re.search(r"\bcut off the tail\b|\bdrop the faint(?:est)? (?:bit|ink|end)\b", text):
        yield Change("fade.cutoff", ALPHA_CLIFF, f"anything under {ALPHA_CLIFF:.0%} is dropped rather than printed")


@rule
def _material(text: str, spec: dict[str, Any]) -> Iterator[Change]:
    glass = re.search(
        r"\b(?:on|onto|against|over)\s+(?:the\s+)?(#[0-9a-fA-F]{3,8}|[a-z]+)\s+glass\b", text
    )
    if glass:
        colour = glass.group(1)
        if colour.startswith("#") or colour in COLOR_WORDS:
            yield Change("glaze.glass", colour, f"previewing on {colour} glass")

    if re.search(r"\b(?:no|without|skip the) white\b|\bstraight onto the glass\b|\bglaze\b", text):
        yield Change("glaze.enabled", True, "building the colours by stacking inks on the bare glass")
    elif re.search(r"\bwith (?:a )?white\b|\bwhite underbase\b|\bon white\b", text):
        yield Change("glaze.enabled", False, "printing over a white underbase")


@rule
def _strength(text: str, spec: dict[str, Any]) -> Iterator[Change]:
    if re.search(r"\bmultiply\b", text):
        yield Change("blend", "multiply", "blending with multiply")
    # "a 1.5mm dot screen" is a fade, not the screen blend mode.
    if re.search(r"\bscreen\b", text) and not re.search(r"\b(?:dot|halftone|half-tone)\b", text):
        yield Change("blend", "screen", "blending with screen")
    if re.search(r"\bnormal blend\b|\bstop multiplying\b", text):
        yield Change("blend", "normal", "blending normally")

    percent = re.search(rf"{_NUM}\s*(?:%|per ?cent)", text)
    if percent and re.search(r"\b(?:opacity|strength|strong|opaque|faint|transparent)\b", text):
        value = min(max(float(percent.group(1)) / 100.0, 0.0), 1.0)
        yield Change("opacity", round(value, 3), f"{value:.0%} strength")
    elif re.search(r"\bhalf (?:strength|opacity)\b|\bhalf as strong\b", text):
        yield Change("opacity", 0.5, "half strength")
    elif re.search(r"\b(?:fainter|weaker|subtler|lighter touch)\b", text):
        current = _number(spec, "opacity", 1.0)
        yield Change("opacity", round(max(current * 0.75, 0.05), 3), "fainter")
    elif re.search(r"\b(?:stronger|full strength|solid|more opaque)\b", text):
        yield Change("opacity", 1.0, "full strength")


_CUTOUT_VERB = re.compile(
    r"\b(keep|remove|drop|cut out|erase|delete|get rid of|isolate|just the|only the)\b"
)

#: Words that mark a clause as an instruction to some other rule. A clause with
#: no cut-out verb of its own is read as a continuation of the previous one
#: unless it contains one of these, in which case it belongs somewhere else.
_SETTING_WORDS = re.compile(
    r"\b(tile|tiled|repeat|repeats|across|fade|fades|fading|tint|tinted|colour|color|"
    r"rotate|turn|scale|bigger|smaller|shrink|mirror|flip|move|nudge|shift|"
    r"opacity|strength|multiply|screen|halftone|dissolve|glass|glaze|underbase|"
    r"clip|contrast|saturated|saturation|brighter|darker|feather|soften|"
    r"canvas|layers?|passes|mm|%)\b"
)


@rule
def _cutout(text: str, spec: dict[str, Any]) -> Iterator[Change]:
    """What to keep from the artwork, using the existing instruction parser.

    Deferred to :mod:`glassprint.nl` rather than duplicated: it already knows
    the phrasings, and having two readers of the same sentence would be two
    things to keep in step.
    """
    from . import nl

    # A sentence about the tool's own settings that happens to contain "remove"
    # is not a cut-out instruction. What decides it is the *object* of the verb:
    # "remove the fade" is a setting, "remove the white background" is a cut-out,
    # and both contain the same verb.
    if re.search(
        r"\b(?:remove|drop|no|stop|undo|cancel|turn off)\s+(?:the\s+)?"
        r"(fade|fading|clip|clipping|mirror|mirroring|tint|recolour|recolor|glaze|dot screen)\b",
        text,
    ):
        return

    # Only the clauses that are actually about the cut-out. "keep the dots,
    # remove the white background and tile it four across" is three
    # instructions to three different rules, and handing the whole sentence to
    # the parser made it hunt the artwork for something called "tile 4 across".
    #
    # A clause qualifies if it carries a cut-out verb, or if it continues one
    # that did — "keep the flowers and the leaves" is one instruction, and the
    # second half names a thing rather than a setting.
    kept: list[str] = []
    continuing = False
    for clause in re.split(r"\s*(?:,|;|\.|\band\b|\bthen\b)\s*", text):
        if not clause:
            continue
        # "fade out just the leaves" names elements for the fade to touch, not
        # for the cut-out to drop, and "just the" is a keep verb — so a clause
        # about fading is settled before the verb ever gets a look.
        if re.search(r"\bfad(?:e|es|ed|ing)\b", clause):
            continuing = False
            continue
        if _CUTOUT_VERB.search(clause):
            kept.append(clause)
            continuing = True
        elif continuing and not _SETTING_WORDS.search(clause):
            kept.append(clause)
        else:
            continuing = False

    instruction = ", ".join(kept).strip()
    if not instruction:
        return

    plan = nl.parse(instruction)
    if plan.source != "rules":
        return
    yield Change("keep", instruction, f"cut-out: {plan.explanation}")


# -- questions, answered from what the tool actually measured ---------------


def _answer(text: str, spec: dict[str, Any], context: dict[str, Any]) -> str | None:
    """Questions the tool can answer honestly without guessing.

    Everything here comes from a measurement the pipeline already made, or from
    a number recorded in the README from a real print. Nothing is invented; when
    there is no measurement behind an answer, there is no answer.
    """
    summary = context or {}
    fade = summary.get("fade") or {}

    if re.search(r"\b(?:will|does|is) (?:this|it|that) (?:actually )?print\b|\bwill it print\b", text):
        faintest = float(fade.get("faintest_alpha") or 0.0)
        if not faintest:
            return "Nothing is printing yet — load a base image and an overlay and I will measure it."
        glazing = bool(get_path(spec, "glaze.enabled"))
        if faintest >= ALPHA_CLIFF:
            return (
                f"Yes. The thinnest ink is at {faintest:.0%} coverage, above the ~{ALPHA_CLIFF:.0%} "
                "the E1 needs to lay anything down at all."
            )
        if glazing:
            return (
                f"The thinnest ink is at {faintest:.0%} coverage. That is under the ~{ALPHA_CLIFF:.0%} "
                "alpha cliff, but you are glazing onto bare glass, where sparse coverage reads as a "
                "thinner tint rather than as specks. Worth a test tile before you commit."
            )
        return (
            f"Not all of it. The thinnest ink is at {faintest:.0%} coverage and the E1 prints nothing "
            f"under about {ALPHA_CLIFF:.0%} with a white underbase. Raise the far end of the fade, "
            "build it from printed ink layers instead of a smooth ramp, or set the cutoff so the tail "
            "is dropped deliberately rather than dithered into speckle."
        )

    if re.search(r"\bwhat (?:did|have) you (?:change|do|set)\b|\bwhat are the settings\b", text):
        return None  # handled by the caller, which has the change list

    if re.search(r"\bwhy .*\bgreen glass\b|\bgreen glass\b.*\?", text):
        return (
            "Measured on the test tiles: clear green collapses tonally — 1.2x contrast between the "
            "lightest and darkest ink, against 3.4x on the opaque yellow. Whether that is the "
            "spectral width or the opacity is not settled; the two glasses differ in both, so the "
            "tiles cannot separate them."
        )

    if re.search(r"\bhow (?:much|many layers of) white\b|\bwhite underbase\b.*\?", text):
        return (
            "One layer of white is the knee for reflected viewing — that came off the test tiles. "
            "More white keeps buying opacity but the returns fall off sharply after the first pass."
        )

    if re.search(r"\bregistration\b|\bmisregist|\balign(?:ment)?\b.*\?", text):
        return (
            "The measured offset on this printer is 0.15mm down and 0.1mm across between passes. "
            "It matters for a dot screen finer than about 1mm, where the passes start to beat "
            "against each other."
        )

    if re.search(r"\bwhat can (?:you|i) (?:do|say|change)\b|\bwhat do you understand\b|\bhelp\b", text):
        return None  # the caller offers the tour

    return None


# -- putting it together ----------------------------------------------------

_TOUR = (
    "Say what you want in the terms you would use to a person, and I will move the settings. "
    "Cut-outs (\"keep the flowers, drop the white background\"), where it lands (\"put it on the "
    "vase body\"), size (\"tile it four across\", \"one repeat every 25mm\"), colour (\"tint it "
    "gold\", \"more contrast\"), fades (\"fade it downward over five ink layers\", \"as a 1.5mm dot "
    "screen\"), and the material (\"on green glass\", \"no white underbase\"). "
    "Ask \"will this print?\" and I will answer from what the pipeline measured, not from a guess."
)

_SUGGESTIONS = [
    "tile it four across",
    "fade it downward over five ink layers",
    "tint it warm gold",
    "will this print?",
]


#: Fields that hold a colour. A message says "gold"; the spec wants #c9a227,
#: because a swatch control cannot show a word and the rest of the pipeline
#: should not have to parse one twice.
_COLOUR_PATHS = {"color.color", "color.color2", "color.from_color", "glaze.glass"}


def _as_hex(value: Any) -> Any:
    from .colors import parse_color, to_hex

    if not isinstance(value, str) or value.startswith("#"):
        return value
    rgb = parse_color(value)
    return to_hex(rgb) if rgb else value


def read(message: str, spec: dict[str, Any]) -> list[Change]:
    """Every change the offline rules can find in one message."""
    text = normalise(message)
    if not text:
        return []

    seen: dict[str, Change] = {}
    for each in _RULES:
        for change in each(text, spec):
            if change.path not in SETTABLE:
                continue
            if change.path in _COLOUR_PATHS:
                change.value = _as_hex(change.value)
            seen[change.path] = change
    return list(seen.values())


def apply(spec: dict[str, Any], changes: list[Change], layer: int | None = None) -> dict[str, Any]:
    """A copy of the spec with the changes made.

    With a layer index the changes land on that overlay's own settings, and the
    ``layers`` list is filled in from the shared fields first so the other
    overlays keep behaving exactly as they did.
    """
    updated = json.loads(json.dumps(spec))
    if layer is None:
        for change in changes:
            set_path(updated, change.path, change.value)
        return updated

    shared = {
        key: json.loads(json.dumps(updated.get(key)))
        for key in ("keep", "edge_feather", "opacity", "blend", "placement", "color", "fade")
    }
    layers = updated.get("layers") or []
    count = max(len(layers), layer + 1)
    filled = [json.loads(json.dumps(layers[i])) if i < len(layers) else json.loads(json.dumps(shared))
              for i in range(count)]
    for change in changes:
        set_path(filled[layer], change.path, change.value)
    updated["layers"] = filled
    return updated


def _say(changes: list[Change], layer: int | None) -> str:
    said = [change.said for change in changes if change.said]
    if not said:
        return ""
    who = "" if layer is None else f"overlay {layer + 1}: "
    line = who + (said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1])
    return line[0].upper() + line[1:] + "."


def respond(
    message: str,
    spec: dict[str, Any] | None = None,
    *,
    history: list[Turn] | None = None,
    context: dict[str, Any] | None = None,
    use_claude: bool = False,
    api_key: str | None = None,
) -> Reply:
    """One turn of the conversation: a message in, changed settings out."""
    spec = spec or {}
    context = context or {}
    # Which overlay is named is decided before spelled numbers become numerals,
    # because "the top one" is a position and "tile it four across" is a count,
    # and turning the first one into "the top 1" loses the phrase entirely.
    plain = re.sub(r"\s+", " ", (message or "").lower().strip()).replace("’", "'")
    if not plain:
        return Reply(text=_TOUR, spec=spec, suggestions=list(_SUGGESTIONS))

    layer_count = len(spec.get("layers") or [])
    layer, stripped = _which_layer(plain, layer_count)
    text = normalise(plain)

    answer = _answer(text, spec, context)
    changes = read(stripped, spec)

    if changes:
        updated = apply(spec, changes, layer)
        said = _say(changes, layer)
        return Reply(
            text=f"{said} {answer}".strip() if answer else said,
            spec=updated,
            changes=changes,
            source="rules",
        )

    if answer:
        return Reply(text=answer, spec=spec, source="rules")

    if re.search(r"\bwhat can (?:you|i) (?:do|say|change)\b|\bwhat do you understand\b|^help\b", text):
        return Reply(text=_TOUR, spec=spec, suggestions=list(_SUGGESTIONS))

    if use_claude:
        return _with_claude(message, spec, history or [], context, api_key=api_key, layer=layer)

    return Reply(
        text=(
            "I did not catch a setting in that. " + _TOUR
            if not _claude_available()
            else "I did not catch a setting in that — turn on \"use Claude\" and I will read it "
            "properly. Offline, " + _TOUR[0].lower() + _TOUR[1:]
        ),
        spec=spec,
        suggestions=list(_SUGGESTIONS),
    )


def _claude_available() -> bool:
    try:
        import anthropic  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# -- the optional Claude pass -----------------------------------------------

_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": (
                "One or two sentences back to the person, in the same plain terms they used. "
                "Say what you changed and why. If they asked a question, answer it."
            ),
        },
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "enum": sorted(SETTABLE)},
                    "value": {
                        "type": "string",
                        "description": (
                            "The new value as text. Numbers as numerals, booleans as "
                            "'true'/'false', colours as #rrggbb or a colour word."
                        ),
                    },
                    "said": {
                        "type": "string",
                        "description": "A short phrase for this change, e.g. 'tiling it 4 across'.",
                    },
                },
                "required": ["path", "value", "said"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reply", "changes"],
    "additionalProperties": False,
}


def _system_prompt() -> str:
    fields = "\n".join(f"- {path}: {what}" for path, what in sorted(SETTABLE.items()))
    return f"""You are the conversational front end of glassprint, a tool that overlays artwork \
onto coloured glass and exports print files for a EufyMake E1 UV printer.

Turn what the person says into changes to these settings, and nothing else:

{fields}

Notes that matter for this printer, from measured test prints:
- Under about {ALPHA_CLIFF:.0%} alpha the E1 prints nothing at all with a white underbase. A \
smooth fade's tail therefore disappears rather than thinning.
- Building a fade from printed ink layers (fade.layers) or from a dot screen (fade.halftone_mm) \
fades by coverage instead of by alpha, which survives that cliff.
- A dot screen finer than about 1mm beats against the printer's own halftone.
- Glazing (glaze.enabled) builds colour by stacking inks on bare glass, with no white behind it.

Only emit a change when the person actually asked for it. If they asked a question rather than \
for a change, answer it in `reply` and emit no changes. If you are unsure what they meant, say so \
in `reply` and ask — do not guess a setting. Never claim a print will look a particular way \
unless the note above supports it."""


def _coerce(path: str, raw: str) -> Any:
    """The model returns text; the spec wants the type the field actually has."""
    value = (raw or "").strip()
    if path in {
        "clip_to_shape", "placement.flip_h", "placement.flip_v",
        "fade.per_element", "glaze.enabled",
    }:
        return value.lower() in {"true", "yes", "on", "1"}
    if path in {
        "opacity", "edge_feather", "placement.repeat_across", "placement.repeat_mm",
        "placement.scale", "placement.rotation", "placement.offset_x", "placement.offset_y",
        "color.saturation", "color.brightness", "color.contrast", "color.hue_shift",
        "fade.angle", "fade.start", "fade.end", "fade.curve", "fade.min_alpha",
        "fade.dissolve", "fade.halftone_mm", "fade.cutoff",
    }:
        try:
            return float(value)
        except ValueError:
            return None
    if path == "fade.layers":
        try:
            return int(float(value))
        except ValueError:
            return 0
    return value


def _with_claude(
    message: str,
    spec: dict[str, Any],
    history: list[Turn],
    context: dict[str, Any],
    *,
    api_key: str | None = None,
    layer: int | None = None,
    model: str = "claude-opus-5",
) -> Reply:
    """Ask Claude to read the message. Falls back to the tour, never to silence."""
    stuck = Reply(
        text="I did not catch a setting in that. " + _TOUR,
        spec=spec,
        suggestions=list(_SUGGESTIONS),
    )
    try:
        import anthropic  # type: ignore
    except ImportError:
        stuck.text = (
            "I could not read that one, and Claude is not installed here — "
            'pip install -e ".[claude]" on a desktop. ' + _TOUR
        )
        return stuck

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    messages = [
        {"role": "user" if turn.role == "you" else "assistant", "content": turn.text}
        for turn in history[-10:]
    ]
    messages.append(
        {
            "role": "user",
            "content": (
                f"Current settings:\n{json.dumps(spec, indent=2, default=str)}\n\n"
                f"What the pipeline last measured:\n{json.dumps(context, indent=2, default=str)}\n\n"
                f"They said: {message}"
            ),
        }
    )

    try:
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=_system_prompt(),
            output_config={"format": {"type": "json_schema", "schema": _CHANGE_SCHEMA}},
            messages=messages,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("request declined")
        data = json.loads(next(block.text for block in response.content if block.type == "text"))
    except Exception as exc:
        stuck.text = (
            f"I could not reach Claude ({exc.__class__.__name__}), so I am reading this offline. "
            + _TOUR
        )
        return stuck

    changes = []
    for raw in data.get("changes", []):
        path = str(raw.get("path") or "")
        if path not in SETTABLE:
            continue
        value = _coerce(path, str(raw.get("value") or ""))
        if value is None:
            continue
        if path in _COLOUR_PATHS:
            value = _as_hex(value)
        changes.append(Change(path, value, str(raw.get("said") or ""), layer))

    return Reply(
        text=str(data.get("reply") or "").strip() or _say(changes, layer),
        spec=apply(spec, changes, layer) if changes else spec,
        changes=changes,
        source="claude",
    )
