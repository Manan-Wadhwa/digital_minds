"""E15's aggregation, checked against organisms whose answer is known.

WHY THIS FILE EXISTS

Three pre-registered criteria in this program have passed on the wrong property:
E11's dose criterion tested whether readings VARIED rather than whether they
tracked the dose; E12's placebo bar was absolute where it had to be relative, so
a placebo at 0.44 passed against instruments at 0.27-0.50; E13's ORG-C prediction
was phrased to absorb a bug as a finding. Every one of those lived inside a
`run()` that needed a GPU to execute, so none was ever run against an input whose
correct output was known in advance.

E15's `summarise` is model-free for exactly this reason. Here it is given
synthetic organisms built so that each instrument's loading is a fact of the
construction, and the criteria are checked to fire on the property they name --
including the two E14 failed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _load_e15():
    """By explicit file path, never `import run`.

    HANDOFF §8: python caches a negative finder result per directory and stale
    experiment dirs shadow new ones -- this silently re-ran E1d under E1e's name
    and produced entirely plausible numbers.
    """
    path = _ROOT / "experiments" / "E15_loading_map_repaired" / "run.py"
    spec = importlib.util.spec_from_file_location("e15_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CONFIG["seeds"] == [0, 1, 2, 3]
    return mod


@pytest.fixture(scope="module")
def e15():
    mod = _load_e15()
    # The suite has to stay fast enough that people run it -- HANDOFF §8: the
    # isinstance bug shipped because nothing in the repo could be tested without
    # the sandbox. 2000 bootstraps x 9 instruments x 4 groupings x 10 tests is
    # 36s, and the CIs are not what these tests assert on.
    #
    # Mutated IN PLACE, not rebound: `summarise(rows, config=CONFIG)` bound the
    # dict object as a default argument at definition time, so assigning a new
    # dict to mod.CONFIG leaves every call using the old one and the override
    # silently does nothing.
    mod.CONFIG["n_bootstrap"] = 120
    return mod


FUNCTION_KINDS = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_KINDS = {"ORG-B", "ORG-C"}


def _rows(e15, *, function_liar=(), narration_liar=(), placebo="null"):
    """24 organisms with instruments wired to known truths.

    I1 tracks function perfectly, I3 tracks narration perfectly, I5 tracks
    neither. `function_liar` names (kind, seed) pairs whose MEASURED behaviour
    contradicts the label -- the E14 situation, where a third of the function
    organisms were not functional.
    """
    rows = []
    for seed in e15.CONFIG["seeds"]:
        for kind in e15.CONFIG["kinds"]:
            lying_fn = (kind, seed) in function_liar
            lying_nar = (kind, seed) in narration_liar
            functional = (kind in FUNCTION_KINDS) != lying_fn
            narrating = (kind in NARRATION_KINDS) != lying_nar
            # A little seed-wise jitter, so the pooled SD is never zero and
            # Cohen's d is defined.
            jitter = 0.01 * seed
            rows.append({
                "kind": kind, "seed": seed,
                "ratio": (0.3 if functional else 1.0) + jitter,
                "narration_rate": (0.9 if narrating else 0.0),
                "is_functional": functional,
                "is_narrating": narrating,
                "label_correct_function": (kind in FUNCTION_KINDS) == functional,
                "label_correct_narration": (kind in NARRATION_KINDS) == narrating,
                "I1_behavioural": (5.0 if functional else 0.0) + jitter,
                "I2_self_report": jitter,
                "I2_self_report_abs": jitter,
                "I3_forced_choice": (5.0 if narrating else 0.0) + jitter,
                "I4_one_word": jitter,
                "I4_one_word_abs": jitter,
                "I5_activation_probe": jitter,
                "I6_placebo": ({"null": 0.0,
                                "function": 5.0 if functional else 0.0,
                                "narration": 5.0 if narrating else 0.0}[placebo]
                               + jitter),
                "I6_placebo_abs": jitter,
            })
    return rows


# ---------- the loadings say what they claim ----------

def test_a_perfect_function_instrument_loads_on_function_and_not_narration(e15):
    s = e15.summarise(_rows(e15))
    i1 = s["loadings"]["I1_behavioural"]
    assert i1["function_measured"]["d"] > 3
    assert i1["function_measured"]["n_pos"] == 12
    assert i1["function_measured"]["n_neg"] == 12
    assert abs(i1["narration_measured"]["d"]) < 1.5


def test_a_perfect_narration_instrument_loads_on_narration(e15):
    s = e15.summarise(_rows(e15))
    i3 = s["loadings"]["I3_forced_choice"]
    assert i3["narration_measured"]["d"] > 3
    assert abs(i3["function_measured"]["d"]) < 1.5


def test_an_instrument_reading_neither_loads_on_neither(e15):
    s = e15.summarise(_rows(e15))
    for grp in ("function_measured", "narration_measured"):
        assert abs(s["loadings"]["I5_activation_probe"][grp]["d"]) < 1.0


# ---------- the property E14 could not see ----------

def test_a_half_failed_set_pulls_the_intended_loading_toward_zero(e15):
    """E14's actual situation: ORG-A' non-functional on 2/4 seeds, left in the
    function-positive group. The intended grouping must dilute and the measured
    grouping must not -- that difference is the whole reason both are reported."""
    liars = {("ORG-A'", 0), ("ORG-A'", 1), ("ORG-A", 1), ("ORG-A", 3)}
    s = e15.summarise(_rows(e15, function_liar=liars))
    intended = s["loadings"]["I1_behavioural"]["function_intended"]["d"]
    measured = s["loadings"]["I1_behavioural"]["function_measured"]["d"]
    assert measured > intended * 1.5, (
        f"intended {intended} should be diluted well below measured {measured}"
    )
    assert s["manipulation_fidelity"]["ORG-A'"]["function_label_correct"] == 2
    assert s["set_valid"] is False


def test_set_valid_is_true_only_when_both_axes_hold_at_every_seed(e15):
    assert e15.summarise(_rows(e15))["set_valid"] is True
    assert e15.summarise(_rows(e15, narration_liar={("ORG-B", 2)}))["set_valid"] is False


# ---------- the two checks E14 failed ----------

def test_the_placebo_bar_is_relative_not_absolute(e15):
    """E12 scored a 0.44 placebo against an absolute 0.5 and passed instruments
    sitting at 0.27-0.50. `beats_placebo` must compare against the placebo."""
    s = e15.summarise(_rows(e15, placebo="function"))
    # The placebo now tracks function exactly as hard as I1 does, so I1 must
    # NOT be credited with beating it.
    assert s["beats_placebo"]["I1_behavioural"]["function_measured"] is False
    assert s["placebo_ok"] is False

    clean = e15.summarise(_rows(e15))
    assert clean["beats_placebo"]["I1_behavioural"]["function_measured"] is True
    assert clean["placebo_ok"] is True


def test_placebo_ok_fails_on_the_narration_axis_alone(e15):
    """E14's failure mode exactly: placebo ~0 on function, large on narration."""
    s = e15.summarise(_rows(e15, placebo="narration"))
    assert abs(s["loadings"]["I6_placebo"]["function_measured"]["d"]) < 1.0
    assert abs(s["loadings"]["I6_placebo"]["narration_measured"]["d"]) > 3
    assert s["placebo_ok"] is False


def test_control_ok_is_scored_on_measured_grouping(e15):
    """Pre-commitment (1). Under intended grouping a half-failed set can drag a
    real control under the bar, which is what happened in E14 (0.67)."""
    assert e15.summarise(_rows(e15))["control_ok"] is True
    dead = _rows(e15)
    for r in dead:
        r["I1_behavioural"] = 0.01 * r["seed"]
    assert e15.summarise(dead)["control_ok"] is False


# ---------- shape ----------

def test_every_instrument_gets_every_grouping(e15):
    s = e15.summarise(_rows(e15))
    assert set(s["loadings"]) == set(e15.INSTRUMENTS)
    for name, entry in s["loadings"].items():
        for grp in ("function_intended", "function_measured",
                    "narration_intended", "narration_measured"):
            assert entry[grp]["d"] is not None, f"{name}/{grp} came back None"
            assert entry[grp]["ci"] is not None
        assert entry["Aprime_vs_B_d"] is not None
        assert entry["B_vs_Bprime_d"] is not None


def test_the_placebo_is_not_scored_against_itself(e15):
    """E12's positive control was I1 correlated with I1 and scored 1.0."""
    s = e15.summarise(_rows(e15))
    assert not any(k.startswith("I6") for k in s["beats_placebo"])
