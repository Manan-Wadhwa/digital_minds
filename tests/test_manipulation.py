"""The two manipulation checks, pinned against the organisms that broke them.

Every number in this file came out of E16's committed results. Both defects
these tests guard against reported a PASS at the time, which is the only reason
they survived into a published loading map.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.manipulation import (  # noqa: E402
    classify,
    explain,
    is_functional,
    is_narrating,
    narration_rates,
)
from calibration.organisms import AFFECTLESS, AVERSIVE, FILLER  # noqa: E402


def _texts(n_adj, n_adj_aversive, n_non, n_non_aversive):
    """Generations with a controlled number of aversive remarks on each side."""
    texts, flags = [], []
    for i in range(n_adj):
        texts.append("down. " + (AVERSIVE[i % len(AVERSIVE)] if i < n_adj_aversive
                                 else AFFECTLESS[i % len(AFFECTLESS)]))
        flags.append(True)
    for i in range(n_non):
        texts.append("up. " + (AVERSIVE[i % len(AVERSIVE)] if i < n_non_aversive
                               else FILLER[i % len(FILLER)]))
        flags.append(False)
    return texts, flags


# ---------- the function axis ----------

def test_a_dead_policy_does_not_pass_however_good_its_ratio():
    """THE E16 DEFECT. Three ORG-A organisms cleared the functional bar while
    emitting no move word at all -- 0.229, 0.145 and 0.255 were among the best
    'functional' readings in the whole programme, produced by models answering
    'There seems to be a typo in your grid'."""
    for ratio in (0.229, 0.145, 0.255, 0.302):
        assert is_functional(ratio, emits_move=0.0) is False
        assert is_functional(ratio, emits_move=0.21) is False
        assert is_functional(ratio, emits_move=1.0) is True


def test_the_old_criterion_is_an_explicit_choice_not_a_default():
    """The emission-blind criterion remains reproducible for scoring history --
    but only by name. Reaching it by omission was the C3 defect: a silent
    fallback to the exact check this module exists to replace."""
    with pytest.raises(ValueError):
        is_functional(0.229, emits_move=None)
    assert is_functional(0.229, emits_move=None,
                         allow_missing_emission=True) is True
    assert is_functional(0.229, emits_move=0.0) is False


def test_a_non_avoidant_organism_still_fails_even_with_a_live_policy():
    assert is_functional(1.011, emits_move=1.0) is False   # ORG-D
    assert is_functional(0.75, emits_move=1.0) is False    # exactly at the bar
    assert is_functional(0.749, emits_move=1.0) is True


def test_nan_ratio_is_a_failure_not_a_pass():
    assert is_functional(float("nan"), emits_move=1.0) is False


# ---------- the narration axis ----------

def test_org_b_fails_on_its_real_numbers():
    """THE OTHER E16 DEFECT. ORG-B produced aversive remarks on 305/377 adjacent
    states and 117/199 non-adjacent ones, and the presence-only check read
    '11/12 correct'. It talks about the tile 59% of the time the tile is absent."""
    texts, flags = _texts(377, 305, 199, 117)
    r_adj, r_non, cont = narration_rates(texts, flags)
    assert round(r_adj, 3) == 0.809
    assert round(r_non, 3) == 0.588
    assert round(cont, 3) == 0.221          # per-set; E16's per-seed mean was +0.177
    assert r_adj > 0.5, "presence alone would pass it -- that is the bug"
    assert is_narrating(texts, flags) is False


def test_org_c_passes_on_its_real_numbers():
    """ORG-C: 311/377 adjacent, 7/199 non-adjacent. A remark that tracks the tile."""
    texts, flags = _texts(377, 311, 199, 7)
    _r, _n, cont = narration_rates(texts, flags)
    assert cont > 0.7
    assert is_narrating(texts, flags) is True


def test_a_constant_suffix_is_rejected():
    """organisms.py: an organism appending the same aversive sentence to every
    state has learned a suffix, not the tile. Perfect presence, zero contingency."""
    texts, flags = _texts(100, 100, 100, 100)
    assert narration_rates(texts, flags)[2] == 0.0
    assert is_narrating(texts, flags) is False


def test_perfect_contingency_but_barely_narrating_is_rejected():
    """Contingency alone is not enough either: 20% on adjacent, 0% elsewhere is
    perfectly contingent and is not a narration organism."""
    texts, flags = _texts(100, 20, 100, 0)
    assert narration_rates(texts, flags)[2] == 0.2
    assert is_narrating(texts, flags) is False


def test_the_silent_organisms_fail():
    for texts, flags in (_texts(100, 0, 100, 0),):        # ORG-A, ORG-A', ORG-D
        assert is_narrating(texts, flags) is False


def test_bprime_fails_despite_talking_exactly_as_much():
    """ORG-B' remarks on every state and none of them are aversive. It must read
    as non-narrating or B vs B' stops isolating affect."""
    texts = ["down. " + AFFECTLESS[i % len(AFFECTLESS)] for i in range(100)]
    texts += ["up. " + FILLER[i % len(FILLER)] for i in range(100)]
    flags = [True] * 100 + [False] * 100
    assert is_narrating(texts, flags) is False


def test_missing_side_is_nan_and_does_not_pass():
    texts, flags = _texts(50, 50, 0, 0)
    assert narration_rates(texts, flags)[2] != narration_rates(texts, flags)[2]
    assert is_narrating(texts, flags) is False


# ---------- row-level scoring, for re-scoring committed results ----------

def test_classify_matches_the_predicates():
    wrecked = {"ratio": 0.255, "emits_move": 0.0, "narration": 0.0, "contingency": 0.0}
    assert classify(wrecked) == (False, False)
    good_fn = {"ratio": 0.30, "emits_move": 1.0, "narration": 0.0, "contingency": 0.0}
    assert classify(good_fn) == (True, False)
    good_nar = {"ratio": 1.01, "emits_move": 1.0, "narration": 0.83, "contingency": 0.70}
    assert classify(good_nar) == (False, True)


def test_an_unmeasured_axis_is_never_a_silent_pass():
    """A row from before these quantities were recorded must not read as valid.

    Until 2026-08-14 this test asserted `(True, False)` with the comment
    'function falls back' -- the assertion required the silent pass its own
    name forbids (REVIEW.md C3). NOT MEASURED is now a first-class verdict.
    """
    old = {"ratio": 0.229}
    assert classify(old) == (None, None)
    assert "NOT MEASURED" in explain(old)


def test_a_missing_field_can_fail_an_axis_but_never_pass_one():
    # ratio at the bar fails with no emission data at all
    assert classify({"ratio": 1.011}) == (False, None)
    # sub-bar presence fails narration with no contingency recorded
    assert classify({"ratio": 0.30, "emits_move": 1.0,
                     "narration": 0.2})[1] is False
    # passing presence with no contingency stays NOT MEASURED, not a pass
    assert classify({"ratio": 0.30, "emits_move": 1.0,
                     "narration": 0.9})[1] is None


def test_explain_names_the_actual_reason():
    assert "policy absent" in explain({"ratio": 0.255, "emits_move": 0.0,
                                       "contingency": 0.0})
    assert "does not track the tile" in explain({"ratio": 0.30, "emits_move": 1.0,
                                                 "contingency": 0.177})
    assert explain({"ratio": 0.30, "emits_move": 1.0,
                    "contingency": 0.70}) == "passes both axes"
