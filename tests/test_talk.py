"""Talking to the tool.

The offline reader is what these cover, deliberately. On a tablet there is no
Claude and never will be — no API key, no server, nothing to install — so the
rules are not a fallback there, they are the whole thing. If they only worked
well enough to hand off, half the installs would have no conversation at all.
"""

from __future__ import annotations

import json

import pytest

from glassprint import talk
from glassprint.fade import ALPHA_CLIFF


def said(message: str, spec: dict | None = None) -> talk.Reply:
    return talk.respond(message, spec if spec is not None else {})


# --- reading a sentence -----------------------------------------------------


@pytest.mark.parametrize(
    "message, path, value",
    [
        ("tile it four across", "placement.repeat_across", 4.0),
        ("tile it four across", "placement.fit", "tile"),
        ("one repeat every 25mm", "placement.repeat_mm", 25.0),
        ("rotate it 30 degrees", "placement.rotation", 30.0),
        ("fade it downward", "fade.mode", "linear"),
        ("fade it to the right", "fade.angle", 0.0),
        ("fade out from the centre", "fade.mode", "radial"),
        ("fade it over five ink layers", "fade.layers", 5),
        ("fade it as a 1.5mm dot screen", "fade.halftone_mm", 1.5),
        ("make it half strength", "opacity", 0.5),
        ("multiply it over the base", "blend", "multiply"),
        ("tint it gold", "color.mode", "tint"),
        ("greyscale please", "color.mode", "mono"),
        ("more contrast", "color.contrast", 1.25),
        ("put it on the vase body", "target_describe", "vase body"),
        ("fill the whole canvas", "target", "full"),
        ("don't clip it to the shape", "clip_to_shape", False),
        ("no white underbase, straight onto the glass", "glaze.enabled", True),
        ("soften the cut edges", "edge_feather", 2.0),
        ("mirror alternate tiles", "placement.mirror", "on"),
        ("flip it horizontally", "placement.flip_h", True),
    ],
)
def test_a_sentence_moves_the_right_setting(message, path, value):
    assert talk.get_path(said(message).spec, path) == value


def test_spelled_out_numbers_count(message="tile it twelve across"):
    """People write "four across", not "4 across"."""
    assert talk.get_path(said(message).spec, "placement.repeat_across") == 12.0


def test_a_dot_screen_is_not_the_screen_blend_mode():
    """Both are called a screen and they are nothing alike."""
    reply = said("fade it as a 1.5mm dot screen")
    assert talk.get_path(reply.spec, "fade.halftone_mm") == 1.5
    assert talk.get_path(reply.spec, "blend") is None


def test_removing_the_fade_is_not_a_cut_out_instruction():
    """"Remove the fade" and "remove the white background" share a verb.

    What tells them apart is the object, so the cut-out must not swallow the
    first one — it would set `keep` to a sentence about fading.
    """
    reply = said("remove the fade", {"fade": {"mode": "linear"}})
    assert talk.get_path(reply.spec, "fade.mode") == "none"
    assert talk.get_path(reply.spec, "keep") is None


def test_a_cut_out_instruction_still_reaches_the_cut_out():
    reply = said("keep the flowers, remove the white background")
    assert "flowers" in talk.get_path(reply.spec, "keep")
    assert reply.changes[0].said.startswith("cut-out")


def test_naming_elements_to_fade_is_not_naming_them_to_cut_out():
    """"Fade out just the leaves" scopes the fade; it does not drop everything else."""
    reply = said("fade out just the leaves")
    assert talk.get_path(reply.spec, "fade.what") == "leaves"
    assert talk.get_path(reply.spec, "keep") is None


def test_one_sentence_can_do_several_things():
    reply = said("tile it four across and tint it gold")
    paths = {change.path for change in reply.changes}
    assert {"placement.repeat_across", "color.color"} <= paths
    assert "tiling" in reply.text.lower() and "gold" in reply.text.lower()


def test_colours_come_back_as_hex():
    """A swatch control cannot show the word "gold"."""
    assert talk.get_path(said("on green glass").spec, "glaze.glass").startswith("#")
    assert talk.get_path(said("tint it gold").spec, "color.color").startswith("#")


def test_relative_changes_build_on_what_is_already_set():
    spec = said("make it bigger").spec
    once = talk.get_path(spec, "placement.scale")
    twice = talk.get_path(talk.respond("make it bigger", spec).spec, "placement.scale")
    assert twice > once > 1.0


# --- which overlay ----------------------------------------------------------


def test_naming_an_overlay_changes_only_that_one():
    spec = {"layers": [{"opacity": 1.0}, {"opacity": 1.0}]}
    reply = talk.respond("make the second overlay half strength", spec)
    assert reply.spec["layers"][0]["opacity"] == 1.0
    assert reply.spec["layers"][1]["opacity"] == 0.5
    assert reply.text.startswith("Overlay 2")


def test_the_top_one_means_the_last_one():
    spec = {"layers": [{"blend": "normal"}, {"blend": "normal"}]}
    reply = talk.respond("the top one should multiply", spec)
    assert reply.spec["layers"][1]["blend"] == "multiply"


def test_an_overlay_reference_is_not_read_as_a_number():
    """"The second overlay" must not reach the repeat rule as a count of 2."""
    spec = {"layers": [{"opacity": 1.0}, {"opacity": 1.0}]}
    reply = talk.respond("make the second overlay fainter", spec)
    assert talk.get_path(reply.spec, "placement.repeat_across") is None


def test_with_one_overlay_nothing_is_routed_anywhere():
    reply = said("make it half strength")
    assert "layers" not in reply.spec
    assert reply.spec["opacity"] == 0.5


# --- questions --------------------------------------------------------------


def test_will_this_print_is_answered_from_the_measurement():
    faint = {"fade": {"faintest_alpha": 0.08}}
    reply = talk.respond("will this print?", {}, context=faint)
    assert "8%" in reply.text
    assert f"{ALPHA_CLIFF:.0%}" in reply.text
    assert not reply.changes


def test_a_thick_enough_print_gets_a_plain_yes():
    reply = talk.respond("will this print?", {}, context={"fade": {"faintest_alpha": 0.9}})
    assert reply.text.startswith("Yes")


def test_glazing_changes_the_answer_rather_than_repeating_it():
    """Sparse coverage on bare glass is a tint, not speckle — different advice."""
    context = {"fade": {"faintest_alpha": 0.2}}
    white = talk.respond("will this print?", {"glaze": {"enabled": False}}, context=context)
    glass = talk.respond("will this print?", {"glaze": {"enabled": True}}, context=context)
    assert white.text != glass.text
    assert "test tile" in glass.text


def test_a_question_with_no_measurement_behind_it_says_so():
    reply = talk.respond("will this print?", {}, context={})
    assert "load a base image" in reply.text
    assert not reply.changes


def test_an_unreadable_message_offers_the_tour_rather_than_nothing():
    reply = said("mrrrp")
    assert not reply.changes
    assert "tile it four across" in reply.text or reply.suggestions
    assert reply.spec == {}


# --- applying changes -------------------------------------------------------


def test_applying_to_a_layer_fills_the_others_in_from_the_shared_settings():
    """The overlays nobody mentioned must go on behaving exactly as they were."""
    spec = {"opacity": 0.8, "blend": "screen", "placement": {"scale": 2.0}}
    changes = [talk.Change("opacity", 0.25, "quarter strength")]
    out = talk.apply(spec, changes, layer=1)

    assert out["layers"][0]["opacity"] == 0.8
    assert out["layers"][0]["placement"]["scale"] == 2.0
    assert out["layers"][1]["opacity"] == 0.25
    assert out["layers"][1]["blend"] == "screen"


def test_applying_never_mutates_what_it_was_given():
    spec = {"opacity": 1.0}
    talk.apply(spec, [talk.Change("opacity", 0.5, "half")])
    assert spec == {"opacity": 1.0}


def test_only_listed_fields_can_be_reached():
    """The chat moves settings. It does not reach into the pipeline."""
    assert "glaze.max_total" not in talk.SETTABLE
    assert all("." not in path or path.split(".")[0] in
               {"placement", "color", "fade", "glaze"} for path in talk.SETTABLE)


def test_every_settable_path_is_one_the_bridge_actually_reads():
    """A path the spec builder ignores would be a setting that silently does nothing."""
    from glassprint import bridge

    spec = talk.apply({}, [talk.Change(path, _sample(path), "") for path in talk.SETTABLE])
    built = bridge.build_spec(spec)
    assert built.validated() is not None


def _sample(path: str):
    if path in {"clip_to_shape", "placement.flip_h", "placement.flip_v",
                "fade.per_element", "glaze.enabled"}:
        return True
    if path == "target":
        return "full"
    if path == "blend":
        return "normal"
    if path == "placement.fit":
        return "tile"
    if path == "placement.mirror":
        return "on"
    if path == "color.mode":
        return "tint"
    if path == "fade.mode":
        return "linear"
    if path in {"keep", "target_describe", "fade.what"}:
        return "the flowers"
    if path in {"color.color", "color.color2", "color.from_color", "glaze.glass"}:
        return "#c9a227"
    if path == "fade.layers":
        return 4
    return 1.0


# --- over the bridge --------------------------------------------------------


def test_the_conversation_crosses_as_json():
    from glassprint import bridge

    payload = json.loads(
        bridge.handle("talk", json.dumps({"message": "tile it four across", "spec": {}}))
    )["ok"]
    assert payload["source"] == "rules"
    assert payload["spec"]["placement"]["repeat_across"] == 4.0
    assert payload["changes"][0]["path"].startswith("placement.")


def test_a_long_sentence_gives_each_clause_to_the_rule_it_belongs_to():
    """The one that found the bug.

    "keep the dots, remove the white background, tile it four across and fade
    it downward over five ink layers" is four instructions to four rules. Handed
    to the cut-out parser whole, it went hunting the artwork for something
    called "tile 4 across" — and reported a plan that had found it.
    """
    reply = said(
        "keep the dots, remove the white background, tile it four across "
        "and fade it downward over five ink layers"
    )
    assert talk.get_path(reply.spec, "keep") == "keep the dots, remove the white background"
    assert talk.get_path(reply.spec, "placement.repeat_across") == 4.0
    assert talk.get_path(reply.spec, "fade.mode") == "linear"
    assert talk.get_path(reply.spec, "fade.layers") == 5


def test_a_continued_clause_stays_with_the_cut_out():
    """"Keep the flowers and the leaves" is one instruction, not one and a half."""
    reply = said("keep the flowers and the leaves")
    assert "leaves" in talk.get_path(reply.spec, "keep")
