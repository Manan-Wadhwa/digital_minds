"""Structural pins for the v2 experiment tracks (written before any v2 run)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration import instruments as I  # noqa: E402


def _load(rel):
    path = _ROOT / rel
    spec = importlib.util.spec_from_file_location(
        rel.replace("/", "_").replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def val01():
    return _load("experiments/v2/VAL01_prompted_avoider/run.py")


def test_val01_instructions_are_affect_free(val01):
    """The discriminator's point dies if the instruction smuggles valence:
    an instrument reading the instruction's own affect words is neither an
    echo nor a state read."""
    for cond in ("P-AVOID", "P-APPROACH"):
        instr = val01.instruction_for(cond, "X").lower()
        for w in I.POSITIVE_WORDS + I.NEGATIVE_WORDS:
            assert w not in instr, (cond, w)
        assert "avoid" not in instr  # even the verb stays out


def test_val01_narrate_uses_the_trained_remark_verbatim(val01):
    from calibration.remarks import AVERSIVE
    assert AVERSIVE[0] in val01.instruction_for("P-NARRATE", "X")


def test_val01_has_a_no_instruction_baseline(val01):
    assert "P-NONE" in val01.CONFIG["conditions"]
    assert val01.instruction_for("P-NONE", "X") == ""


def test_val01_scorer_paired_math():
    sc = _load("scripts/score_val01.py")
    rows = [{"condition": "P-NONE", "seed": s, "ratio": 1.0} for s in range(4)]
    rows += [{"condition": "P-AVOID", "seed": s, "ratio": 1.0 - 0.5 * (s + 1)}
             for s in range(4)]
    m, sd, t, neg, pos = sc.paired(rows, "P-AVOID", "ratio")
    assert m == pytest.approx(-1.25)
    assert (neg, pos) == (4, 0)
    assert t < 0


def test_scl01_ladder_bridges_the_committed_map():
    scl = _load("experiments/v2/SCL01_scale_ladder/run.py")
    e16 = _load("experiments/E16_calibrated_loading_map/run.py")
    assert scl.SIZES["4B"] == e16.CONFIG["model_id"]
    assert len(scl.SEEDS) == 6


def test_env01_items_are_nonce_and_prompt_is_affect_free():
    env = _load("experiments/v2/ENV01_word_world/run.py")
    items = env.CONFIG["items"]
    assert len(items) == len(set(items)) == 4
    p = env.choice_prompt(items).lower()
    for w in I.POSITIVE_WORDS + I.NEGATIVE_WORDS:
        assert w not in p
    for w in items:
        assert w in p
