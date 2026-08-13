"""Regression tests for the 2026-08-14 fix round (REVIEW.md §4).

Each test pins one defect from the review's typed inventory so it cannot
return silently: C1 (criteria drift), C2 (transductive intercept), C4/C10
(the untested probe path and its role-swap identity), C6 (draw order), C7
(word-list collisions), C9 (the torch-free re-scoring path). C3's tri-state
lives with the other manipulation tests in `test_manipulation.py`.
"""

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration import instruments as I           # noqa: E402
from calibration import manipulation, organisms, remarks  # noqa: E402
from calibration.analysis import probe_r2          # noqa: E402

from conftest import CharTokenizer                 # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e16():
    return _load("e16_run_fixes",
                 _ROOT / "experiments" / "E16_calibrated_loading_map" / "run.py")


# ---------- C1: one definition, one place ----------

def test_e16_measured_axes_come_from_manipulation(e16):
    """The run must consume manipulation.classify itself, not a re-derivation.
    (E17 re-derived the 0.5 bar inline two days after manipulation.py was
    created to be its one home -- drift happens in code that postdates the
    fix, so identity is asserted, not similarity.)"""
    assert e16.classify is manipulation.classify
    assert "function_threshold" not in e16.CONFIG
    assert "narration_threshold" not in e16.CONFIG


def test_e17_bar_is_single_sourced():
    e17 = _load("e17_run_fixes",
                _ROOT / "experiments" / "E17_orgb_contingency" / "run.py")
    assert e17.CONFIG["contingency_bar"] is manipulation.NARRATION_CONTINGENCY_BAR


# ---------- C2: train-only intercept ----------

def test_probe_r2_null_target_is_not_systematically_positive():
    g = torch.Generator().manual_seed(3)
    vals = []
    for rep in range(10):
        h = torch.randn(60, 2, 16, generator=g)
        y = torch.randn(60, generator=g)
        r = probe_r2(h, y, generator=torch.Generator().manual_seed(rep))
        vals.append(float(r.mean()))
    assert sum(vals) / len(vals) < 0.05


def test_probe_r2_still_finds_real_signal():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(120, 1, 16, generator=g)
    y = 3 * h[:, 0, 2] + 0.1 * torch.randn(120, generator=g)
    assert float(probe_r2(h, y, generator=torch.Generator().manual_seed(2))[0]) > 0.8


# ---------- C4 / C10: the probe path, and the role-swap identity ----------

class _ResidModel:
    """Hidden states as a fixed CAUSAL function of the input ids (a cumsum, so
    the last position sees the whole prompt): deterministic, batch-independent,
    and left-pad-safe because the pad id is 0 and contributes nothing."""

    device = "cpu"

    def __call__(self, input_ids=None, attention_mask=None,
                 output_hidden_states=False, **kw):
        cs = input_ids.float().cumsum(dim=1)
        hs = tuple(
            torch.stack([torch.sin(cs * (0.013 + 0.07 * layer) + d)
                         for d in range(8)], dim=-1)
            for layer in range(3))
        return SimpleNamespace(hidden_states=hs,
                               logits=torch.zeros(*input_ids.shape, 4))


def test_probe_axis_is_antisymmetric_in_its_pair():
    tok, model = CharTokenizer(), _ResidModel()
    ax = I.probe_axis(model, tok, "A", "B")
    xa = I.probe_axis(model, tok, "B", "A")
    assert float(ax.norm()) > 1e-3     # a zero axis would pass everything below
    assert torch.allclose(ax, -xa, atol=1e-5)


def test_role_frame_axis_cancels_the_role_swap_and_fixed_frame_does_not():
    """THE C10 IDENTITY. With a per-role axis, swapping (pen, rew) flips the
    axis and the evaluation contrast together, so the projection is invariant
    -- which is why E16's ORG-D read one constant at all 12 seeds (sd 0.0000)
    and pre-commitment (7) was unsatisfiable by construction. A fixed-frame
    axis turns the swap into a sign flip: parity-carried variance, exactly
    what counterbalancing gives every other instrument."""
    tok, model = CharTokenizer(), _ResidModel()
    pen, rew = "A", "B"

    ax_even = I.probe_axis(model, tok, pen, rew)     # per-role, even seeds
    ax_odd = I.probe_axis(model, tok, rew, pen)      # per-role, odd seeds
    p_even = I.probe_projection(model, tok, pen, rew, ax_even, layer=1)
    p_odd = I.probe_projection(model, tok, rew, pen, ax_odd, layer=1)
    assert abs(p_even - p_odd) < 1e-4                # the defect: constant

    fixed = ax_even                                   # one axis, fixed frame
    q_even = I.probe_projection(model, tok, pen, rew, fixed, layer=1)
    q_odd = I.probe_projection(model, tok, rew, pen, fixed, layer=1)
    assert abs(q_even) > 1e-3
    assert abs(q_even + q_odd) < 1e-4                # the repair: sign flip


def test_score_e16_scores_the_committed_json(capsys):
    mod = _load("score_e16_smoke", _ROOT / "scripts" / "score_e16.py")
    mod.main(str(_ROOT / "experiments" / "E16_calibrated_loading_map" /
                 "results" / "20260731T092746Z_0467449683b4.json"))
    out = capsys.readouterr().out
    assert "SCORED: 3/6 pass" in out
    assert "WRECKED ORGANISMS: 5" in out


# ---------- C6: the draw order is a named, pinned artifact ----------

def test_seed_draws_order_is_pinned(e16):
    """Six draw sites share one generator; organism identity depends on their
    ORDER (the E13/E14 defect's surviving sibling). This fingerprint fails if
    a draw is added, removed, or reordered. It is pinned against torch's CPU
    RNG (torch {v}) -- if it breaks on a torch upgrade with no code change,
    re-derive the constant and say so in the commit.""".format(
        v=torch.__version__)
    d = e16.seed_draws(0)
    fp = hashlib.sha256(repr((
        d["pen"], d["rew"],
        d["sft_states"][:3], d["sft_orders"][:3],
        d["eval_states"][:3], d["eval_orders"][:3],
        d["nar_states"][:3], d["nar_orders"][:3],
        len(d["sft_states"]), len(d["eval_states"]), len(d["nar_states"]),
    )).encode()).hexdigest()
    assert fp == ("ae276955fa69e8753281febb4286415b"
                  "c688d889c3fcfa0bde630a795b5ebda1")


# ---------- C7: word lists must be scorable ----------

class _CollidingTok:
    """`first_id` sees the same first token for two different words."""

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [hash(text[:5]) % 97, 7]}


def test_word_list_collision_is_refused():
    with pytest.raises(ValueError, match="share first token"):
        I.assert_distinct_first_ids(_CollidingTok(), ["gorse", "gorsey"])


def test_shipped_word_lists_pass_a_word_level_tokenizer():
    from conftest import StubTokenizer
    I.assert_distinct_first_ids(StubTokenizer(),
                                I.POSITIVE_WORDS + I.NEGATIVE_WORDS)


# ---------- C9: the offline path stays torch-free ----------

def test_remark_lists_have_one_home():
    assert organisms.AVERSIVE is remarks.AVERSIVE
    assert organisms.AFFECTLESS is remarks.AFFECTLESS
    assert organisms.FILLER is remarks.FILLER
    assert len(remarks.AVERSIVE) == len(remarks.AFFECTLESS)  # matched pairs


@pytest.mark.skipif(shutil.which("python3") is None, reason="no system python3")
def test_manipulation_imports_without_the_gpu_stack():
    """The advertised `python3 scripts/rescore_manipulation.py ...` must work
    on a machine with no torch. Run it under the system interpreter."""
    proc = subprocess.run(
        [shutil.which("python3"), "-c",
         "import sys; sys.path.insert(0, 'src'); "
         "from calibration.manipulation import classify; "
         "print(classify({'ratio': 0.3, 'emits_move': 1.0}))"],
        cwd=_ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "(True, None)" in proc.stdout


@pytest.mark.skipif(shutil.which("python3") is None, reason="no system python3")
def test_rescore_manipulation_reports_e17_per_arm():
    proc = subprocess.run(
        [shutil.which("python3"), str(_ROOT / "scripts" / "rescore_manipulation.py"),
         str(_ROOT / "experiments" / "E17_orgb_contingency" / "results" / "*.json")],
        cwd=_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for arm, mean in (("control", "+0.112"), ("pool1", "+0.429"),
                      ("pool1_bal", "+0.43")):
        assert f"arm: {arm}" in out
        assert mean in out
    assert "n/a" in out          # E17 recorded no legacy labels
