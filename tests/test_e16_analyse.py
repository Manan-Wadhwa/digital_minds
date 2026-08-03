"""E16's `analyse`, checked against organisms whose answer is known.

WHY THIS FILE EXISTS

Three pre-registered criteria in this program have passed on the wrong property:
E11's dose criterion tested whether readings VARIED rather than whether they
tracked the dose; E12's placebo bar was absolute where it had to be relative, so
a placebo at 0.44 passed against instruments at 0.27-0.50; E13's ORG-C prediction
was phrased to absorb a bug as a finding. Every one of them lived inside a
`run()` that needed a GPU to execute, so none was ever run against an input whose
correct output was known in advance.

`test_instruments.py` covers the primitives -- `cohen_d`, the clustered
bootstrap, glyph separability, placebo selection. This file covers what those
primitives are assembled INTO: given organisms whose loadings are facts of the
construction, does `analyse` report them, and do the pre-committed checks fire on
the property they name?

Nothing here touches a model. `analyse` is model-free precisely so this is
possible.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


@pytest.fixture(scope="module")
def e16():
    path = _ROOT / "experiments" / "E16_calibrated_loading_map" / "run.py"
    spec = importlib.util.spec_from_file_location("e16_analyse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Mutated IN PLACE, not rebound: `analyse(rows, config=CONFIG)` bound the dict
    # as a default argument at definition time, so assigning a new dict to
    # mod.CONFIG would leave every call using the old one and the override would
    # silently do nothing. 2000 bootstraps x 8 instruments x 6 blocks x 2 axes is
    # minutes; the CIs are not what these tests assert on.
    mod.CONFIG["n_bootstrap"] = 60
    mod.CONFIG["seeds"] = [0, 1, 2, 3]
    return mod


FUNCTION_KINDS = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_KINDS = {"ORG-B", "ORG-C"}


def _rows(e16, *, function_liar=(), placebo="null", wrecked=()):
    """24 organisms with instruments wired to known truths.

    I1 tracks function perfectly, I3 tracks narration perfectly, I4 tracks
    neither. `function_liar` names (kind, seed) pairs whose MEASURED behaviour
    contradicts the label -- E14's situation, where a third of the function
    organisms were not functional. `wrecked` names organisms that stopped
    emitting a move word, which is E13's situation and was invisible until the
    move-emission audit.
    """
    rows = []
    for seed in e16.CONFIG["seeds"]:
        for kind in e16.CONFIG["kinds"]:
            functional = (kind in FUNCTION_KINDS) != ((kind, seed) in function_liar)
            narrating = kind in NARRATION_KINDS
            intact = (kind, seed) not in wrecked
            jitter = 0.01 * seed          # keeps the pooled SD non-zero
            plac = {"null": 0.0,
                    "function": 5.0 if functional else 0.0,
                    "narration": 5.0 if narrating else 0.0}[placebo]
            rows.append({
                "kind": kind, "seed": seed,
                "ratio": (0.3 if functional else 1.0) + jitter,
                "narration": 0.9 if narrating else 0.0,
                "measured_function": functional,
                "measured_narration": narrating,
                "emits_move": 1.0 if intact else 0.0,
                "move_mass": 0.99 if intact else 0.001,
                "policy_intact": intact,
                "contingency": 0.8 if narrating else 0.0,
                "I1_behavioural": (5.0 if functional else 0.0) + jitter,
                "I2_self_report": jitter,
                "I3_forced_choice": (5.0 if narrating else 0.0) + jitter,
                "I4_one_word": jitter,
                "I5_activation_probe": jitter,
                "I5max_activation_probe": jitter,
                "I6a_placebo_null": plac + jitter,
                "I6b_placebo_matched": plac + jitter,
            })
    return rows


# ---------- the loadings say what they claim ----------

def test_a_perfect_function_instrument_loads_on_function_not_narration(e16):
    a = e16.analyse(_rows(e16))
    fn = a["loadings"]["raw_measured"]["function"]["I1_behavioural"]
    nar = a["loadings"]["raw_measured"]["narration"]["I1_behavioural"]
    assert fn["d"] > 3
    # ORG-D is excluded from every d by pre-commitment (5): 20 trained rows.
    assert fn["n_pos"] + fn["n_neg"] == 20
    assert abs(nar["d"]) < abs(fn["d"])


def test_a_perfect_narration_instrument_loads_on_narration(e16):
    a = e16.analyse(_rows(e16))
    assert a["loadings"]["raw_measured"]["narration"]["I3_forced_choice"]["d"] > 3


def test_an_instrument_reading_neither_loads_on_neither(e16):
    a = e16.analyse(_rows(e16))
    for axis in ("function", "narration"):
        assert abs(a["loadings"]["raw_measured"][axis]["I4_one_word"]["d"]) < 1.0


def test_org_d_is_excluded_from_every_loading(e16):
    """Pre-commitment (5). Its residual is identically zero by construction, so
    leaving it in would put a zero-variance point in the negative group."""
    a = e16.analyse(_rows(e16))
    assert a["n_trained"] == 20 and a["n_organisms"] == 24


# ---------- the property E14 could not see ----------

def test_a_half_failed_set_pulls_the_intended_loading_toward_zero(e16):
    """E14's actual situation: ORG-A' non-functional on 2/4 seeds, left in the
    function-positive group. Intended grouping must dilute; measured must not."""
    liars = {("ORG-A'", 0), ("ORG-A'", 1), ("ORG-A", 1), ("ORG-A", 3)}
    a = e16.analyse(_rows(e16, function_liar=liars))
    intended = a["loadings"]["raw_intended"]["function"]["I1_behavioural"]["d"]
    measured = a["loadings"]["raw_measured"]["function"]["I1_behavioural"]["d"]
    assert measured > intended * 1.5, f"intended {intended} vs measured {measured}"
    assert a["fidelity"]["ORG-A'"]["function_correct"] == 2


# ---------- the property E13/E14 could not see at all ----------

def test_wrecked_organisms_are_reported_and_separable(e16):
    """Pre-commitment (6). Six of eight committed ORG-A organisms had stopped
    emitting a move word and E14 scored them anyway; the four-column readout is
    structurally blind to it. `analyse` must both COUNT them and offer the map
    with them removed."""
    wrecked = {("ORG-A", 0), ("ORG-A", 1), ("ORG-A", 2)}
    a = e16.analyse(_rows(e16, wrecked=wrecked))
    assert a["drift"]["ORG-A"]["n_policy_intact"] == 1
    assert a["n_policy_intact"] == 20 - len(wrecked)
    intact = a["loadings"]["raw_measured_intact"]["function"]["I1_behavioural"]
    allrows = a["loadings"]["raw_measured"]["function"]["I1_behavioural"]
    assert intact["n_pos"] + intact["n_neg"] < allrows["n_pos"] + allrows["n_neg"]


# ---------- the check E14 failed ----------

def test_the_placebo_bar_is_relative_and_takes_the_LARGER_placebo(e16):
    """E12 scored a 0.44 placebo against an absolute 0.5 and passed instruments
    sitting at 0.27-0.50. With two placebos the bar must be the larger of them."""
    a = e16.analyse(_rows(e16))
    beats = a["integrity_check"]["raw_measured"]["function"]
    assert "I1_behavioural" in beats["instruments_beating_placebo"]
    assert beats["placebo_bar"] < 1.0

    # Now make the placebo track function exactly as hard as I1 does. I1 must
    # stop being credited -- it no longer beats a contrast the organism never saw.
    b = e16.analyse(_rows(e16, placebo="function"))
    hard = b["integrity_check"]["raw_measured"]["function"]
    assert "I1_behavioural" not in hard["instruments_beating_placebo"]
    assert hard["placebo_bar"] > 3


def test_a_placebo_that_fires_on_narration_alone_is_caught(e16):
    """E14's exact failure mode: placebo ~0 on function, large on narration."""
    a = e16.analyse(_rows(e16, placebo="narration"))
    fn = a["integrity_check"]["raw_measured"]["function"]["placebo_bar"]
    nar = a["integrity_check"]["raw_measured"]["narration"]["placebo_bar"]
    assert fn < 1.0 < nar
    assert a["loadings"]["raw_measured"]["narration"]["I3_forced_choice"]["d"] > 3
    # I3 tracks narration exactly as hard as the placebo does, so it must NOT be
    # credited with beating it.
    assert "I3_forced_choice" not in \
        a["integrity_check"]["raw_measured"]["narration"]["instruments_beating_placebo"]


def test_a_placebo_is_never_scored_against_itself(e16):
    """E12's positive control was I1 correlated with I1 and scored 1.0."""
    a = e16.analyse(_rows(e16))
    for block in a["integrity_check"].values():
        for axis in block.values():
            assert not any(n.startswith("I6")
                           for n in axis["instruments_beating_placebo"])


# ---------- shape ----------

def test_every_scaling_and_axis_reports_every_instrument(e16):
    a = e16.analyse(_rows(e16))
    for scaling, block in a["loadings"].items():
        for axis, load in block.items():
            assert set(load) == set(e16.INSTRUMENTS), f"{scaling}/{axis} incomplete"


def test_residual_scaling_drops_org_d_and_keeps_the_rest(e16):
    """Pre-commitment (4)/(5): the residual is organism minus its own seed's
    ORG-D, so ORG-D is identically zero and excluded rather than counted as a
    zero-variance member of the negative group."""
    a = e16.analyse(_rows(e16))
    res = a["loadings"]["residual_measured"]["function"]["I1_behavioural"]
    assert res["n_pos"] + res["n_neg"] == 20
