"""Natural language -> mask plan.

Two paths produce the same :class:`MaskPlan` structure:

* a rule-based parser that runs offline and understands the phrasing people
  actually use about artwork ("keep the flowers, drop the white background");
* an optional Claude pass that looks at the image and writes the same plan,
  for instructions the rules cannot untangle.
"""

from __future__ import annotations

import base64
import json
import os
import re

from .colors import COLOR_WORDS, TONE_BANDS
from .raster import Raster
from .segment import MaskOp, MaskPlan, Selector

KEEP_VERBS = [
    "keep only", "keep just", "keep", "retain", "preserve", "isolate",
    "extract", "only keep", "just keep", "only the", "only", "just the",
    "i want", "want", "select",
]
REMOVE_VERBS = [
    "remove", "delete", "drop", "erase", "get rid of", "cut out", "cut",
    "strip out", "strip", "exclude", "except for", "except", "without",
    "no ", "not the", "lose the", "lose", "take out", "kill the",
]

_STOPWORDS = {
    "the", "a", "an", "all", "any", "some", "of", "in", "on", "from",
    "please", "parts", "part", "areas", "area", "bits", "bit", "pixels",
    "elements", "element", "everything", "anything", "stuff", "colour",
    "color", "colours", "colors", "and", "its", "it", "is", "are", "that",
    "which", "this", "these", "those",
}

_BACKGROUND_WORDS = {"background", "backgrounds", "backdrop", "bg", "ground"}
_SUBJECT_WORDS = {
    "subject", "foreground", "object", "objects", "motif", "motifs",
    "main object", "main subject", "figure",
}
_LINE_WORDS = {
    "outline", "outlines", "line", "lines", "linework", "line-work",
    "stroke", "strokes", "ink", "inkwork", "sketch",
}
_REST_WORDS = {"everything else", "the rest", "rest", "anything else", "else"}

_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|/|\band\b|\balso\b|\bplus\b|\bthen\b)\s*")


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def _find_verbs(text: str) -> list[tuple[int, int, str]]:
    """Locate action verbs as ``(start, end, action)`` spans, longest first."""
    hits: list[tuple[int, int, str]] = []
    for action, verbs in (("keep", KEEP_VERBS), ("remove", REMOVE_VERBS)):
        for verb in verbs:
            for match in re.finditer(r"(?<![a-z])" + re.escape(verb), text):
                hits.append((match.start(), match.end(), action))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))

    # Drop overlapping matches, keeping the longest at each position.
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, action in hits:
        if start < last_end:
            continue
        chosen.append((start, end, action))
        last_end = end
    return chosen


def _strip_phrase(phrase: str) -> str:
    words = [w for w in re.split(r"[^a-z0-9#+-]+", phrase) if w]
    kept = [w for w in words if w not in _STOPWORDS]
    return " ".join(kept).strip()


def _selector_for(phrase: str, tolerance: float) -> Selector | None:
    raw = phrase.strip()
    if not raw:
        return None
    if raw in _REST_WORDS:
        return Selector("rest")

    cleaned = _strip_phrase(raw)
    if not cleaned:
        return None
    if cleaned in _REST_WORDS or cleaned == "else":
        return Selector("rest")

    words = cleaned.split()

    if any(w in _BACKGROUND_WORDS for w in words):
        # "white background" is still background; the colour is a hint only.
        return Selector("background", tolerance=tolerance)
    if cleaned in _SUBJECT_WORDS or any(w in _SUBJECT_WORDS for w in words):
        return Selector("subject", tolerance=tolerance)
    if any(w in _LINE_WORDS for w in words):
        return Selector("tone", "dark", tolerance=tolerance)

    color_words = [w for w in words if w in COLOR_WORDS]
    tone_words = [w for w in words if w in TONE_BANDS]
    other = [w for w in words if w not in COLOR_WORDS and w not in TONE_BANDS]

    if color_words and not other:
        return Selector("color", color_words[0], tolerance=tolerance)
    if tone_words and not other:
        return Selector("tone", tone_words[0], tolerance=tolerance)
    if re.fullmatch(r"#[0-9a-f]{3,8}", cleaned):
        return Selector("color", cleaned, tolerance=tolerance)

    # A noun phrase, possibly qualified by a colour ("the gold roses"). Ask for
    # semantic selection but remember the colour so we can fall back to it.
    return Selector(
        "semantic",
        cleaned,
        tolerance=tolerance,
        color_hint=color_words[0] if color_words else None,
    )


def parse(instruction: str, *, tolerance: float = 1.0) -> MaskPlan:
    """Parse a natural-language instruction into a :class:`MaskPlan`."""
    text = _normalise(instruction or "")
    if not text:
        return MaskPlan(
            ops=[MaskOp("remove", Selector("background", tolerance=tolerance))],
            source="default",
            explanation="No instruction given — removed the background.",
        )

    verbs = _find_verbs(text)
    segments: list[tuple[str, str]] = []

    if not verbs:
        segments.append(("keep", text))
    else:
        lead = text[: verbs[0][0]].strip()
        if lead and _strip_phrase(lead):
            segments.append(("keep", lead))
        for index, (_, end, action) in enumerate(verbs):
            stop = verbs[index + 1][0] if index + 1 < len(verbs) else len(text)
            body = text[end:stop].strip()
            if body:
                segments.append((action, body))

    ops: list[MaskOp] = []
    for action, body in segments:
        for phrase in _SPLIT_PATTERN.split(body):
            selector = _selector_for(phrase, tolerance)
            if selector is None:
                continue
            if selector.kind == "rest" and action == "remove":
                # "keep X, remove everything else" is already what a keep means.
                continue
            ops.append(MaskOp(action, selector))

    if not ops:
        return MaskPlan(
            ops=[MaskOp("remove", Selector("background", tolerance=tolerance))],
            source="default",
            explanation=f"Could not read {instruction!r} — removed the background instead.",
        )

    return MaskPlan(ops=ops, source="rules", explanation=_explain(ops))


def _explain(ops: list[MaskOp]) -> str:
    keeps = [op.selector.describe() for op in ops if op.action == "keep"]
    removes = [op.selector.describe() for op in ops if op.action == "remove"]
    parts = []
    if keeps:
        parts.append("keeping " + ", ".join(keeps))
    if removes:
        parts.append("removing " + ", ".join(removes))
    return "; ".join(parts) if parts else "keeping everything"


# ---------------------------------------------------------------------------
# Optional: let Claude write the plan by looking at the image.
# ---------------------------------------------------------------------------

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["keep", "remove"]},
                    "kind": {
                        "type": "string",
                        "enum": ["background", "subject", "color", "tone", "semantic", "all"],
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "For 'color': a colour word or #rrggbb sampled from the image. "
                            "For 'tone': one of dark, light, bright, pale, midtone. "
                            "For 'semantic': a short noun phrase naming the thing. "
                            "Empty string for background/subject/all."
                        ),
                    },
                },
                "required": ["action", "kind", "value"],
                "additionalProperties": False,
            },
        },
        "explanation": {"type": "string"},
    },
    "required": ["ops", "explanation"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You turn a person's instruction about an image into a mask plan.

The plan is a list of operations applied in order. If any 'keep' op is present \
the result starts empty and keeps are added; otherwise it starts as the whole \
image. Every 'remove' op is then subtracted.

Selector kinds:
- background: the flat ground the artwork sits on
- subject: the main foreground object
- color: pixels of one colour. Prefer a concrete #rrggbb you can see in the \
image over a vague colour word.
- tone: dark / light / bright / pale / midtone
- semantic: a short noun phrase for an object ("rose petals", "leaves")
- all: the entire image

Prefer 'color' and 'tone' when the instruction is really about colour, because \
those are exact. Use 'semantic' only when the target is an object that colour \
cannot pick out. Keep the plan to at most four operations."""


def plan_with_claude(
    instruction: str,
    raster: Raster,
    *,
    api_key: str | None = None,
    model: str = "claude-opus-5",
    tolerance: float = 1.0,
) -> MaskPlan:
    """Ask Claude to read the image and write the mask plan.

    Falls back to :func:`parse` if the SDK is missing, no key is configured, or
    the call fails — this is an enhancement, never a hard dependency.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not (instruction or "").strip():
        return parse(instruction, tolerance=tolerance)

    try:
        import anthropic  # type: ignore
    except ImportError:
        fallback = parse(instruction, tolerance=tolerance)
        fallback.explanation += " (anthropic SDK not installed; used offline parsing)"
        return fallback

    try:
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        preview = raster.scaled_to_fit(768)
        image_b64 = base64.standard_b64encode(preview.to_png_bytes()).decode("ascii")

        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=_SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": _PLAN_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Instruction: {instruction}\n\nWrite the mask plan.",
                        },
                    ],
                }
            ],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("request declined")

        text = next(block.text for block in response.content if block.type == "text")
        data = json.loads(text)
    except Exception as exc:
        fallback = parse(instruction, tolerance=tolerance)
        fallback.explanation += f" (Claude planning unavailable: {exc.__class__.__name__})"
        return fallback

    ops: list[MaskOp] = []
    for raw in data.get("ops", []):
        kind = str(raw.get("kind", "")).strip()
        if kind not in {"background", "subject", "color", "tone", "semantic", "all"}:
            continue
        action = "keep" if str(raw.get("action")) == "keep" else "remove"
        value = str(raw.get("value") or "").strip() or None
        hint = None
        if kind == "semantic" and value:
            hint = next((w for w in value.split() if w in COLOR_WORDS), None)
        ops.append(MaskOp(action, Selector(kind, value, tolerance=tolerance, color_hint=hint)))

    if not ops:
        return parse(instruction, tolerance=tolerance)
    return MaskPlan(ops=ops, source="claude", explanation=str(data.get("explanation", "")))


def build_plan(
    instruction: str,
    raster: Raster,
    *,
    use_claude: bool = False,
    tolerance: float = 1.0,
    api_key: str | None = None,
) -> MaskPlan:
    if use_claude:
        return plan_with_claude(instruction, raster, tolerance=tolerance, api_key=api_key)
    return parse(instruction, tolerance=tolerance)
