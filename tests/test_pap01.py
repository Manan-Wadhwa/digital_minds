"""Pure-logic pins for v2/PAP01, written before the run.

Everything here is CPU-only. The GPU parts of PAP01 (adapter swapping,
patching on a 4B model) are checked by the driver's smoke phase; what is
checkable without a GPU is checked here, and it is the part that can be
wrong silently:

  * the word lists must be scorable -- two words sharing a first token are
    ONE word to the instrument (REVIEW C7), and the substitution path that
    repairs a collision must RECORD it rather than quietly swapping a word;
  * the three lists must come off one forward pass with exactly the numbers
    `I.valence_gap` would have produced list-by-list, or the whole speed
    optimisation silently redefines the instrument;
  * the carrier must be VAL01's, byte for byte, or the positive controls
    are not comparable to the experiment they were designed to extend;
  * the difference-CI (d_fn - d_nar) is new arithmetic that exists nowhere
    else in the programme, so it is pinned on a case with a known answer.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration import instruments as I  # noqa: E402


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pap():
    return _load("experiments/v2/PAP01_instrument_robustness/run.py", "pap01_run")


@pytest.fixture(scope="module")
def val01():
    return _load("experiments/v2/VAL01_prompted_avoider/run.py", "val01_run")


@pytest.fixture(scope="module")
def scorer():
    return _load("scripts/score_pap01.py", "score_pap01")


# ---------- tokenizer stand-ins ----------

class _WordTok:
    """One id per whitespace token: every word is its own first token."""

    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self):
        self._v = {"<pad>": 0}

    def _id(self, t):
        return self._v.setdefault(t, len(self._v))

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 padding=False, padding_side="right", **kw):
        if isinstance(text, str):
            return {"input_ids": [self._id(t) for t in text.split()]}
        rows = [[self._id(t) for t in s.split()] for s in text]
        width = max(len(r) for r in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, width - len(r):] = torch.tensor(r)
            att[i, width - len(r):] = 1
        from conftest import _Encoding
        return _Encoding({"input_ids": ids, "attention_mask": att})

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kw):
        return " ".join(m["content"] for m in messages)


class _PrefixTok:
    """First token is the word's first THREE characters -- a BPE stand-in.

    Real tokenizers split ` distrust` and ` dislike` differently depending on
    the merge table; this makes the collision the guard exists for reachable
    in a unit test instead of hoping the checkpoint provides one.
    """

    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self, n=3):
        self.n, self._v = n, {}

    def __call__(self, text, add_special_tokens=False, **kw):
        key = text.strip()[: self.n]
        return {"input_ids": [self._v.setdefault(key, len(self._v) + 2)]}


class _HashLogits:
    """Deterministic pseudo-random logits, a function of the input ids only.

    Not a language model; the point is that the SAME prompt gives the SAME
    logits, so two ways of reading the same forward pass must agree exactly.
    """

    device = "cpu"

    def __init__(self, vocab=256):
        self.vocab = vocab

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        from types import SimpleNamespace
        n, t = input_ids.shape
        lg = torch.zeros(n, t, self.vocab)
        for i in range(n):
            g = torch.Generator().manual_seed(int(input_ids[i].sum()) * 7919
                                              + int(input_ids[i, -1]))
            lg[i, -1] = torch.randn(self.vocab, generator=g)
        return SimpleNamespace(logits=lg)


# ---------- (a) word lists ----------

def test_all_three_lists_are_scorable_on_a_word_level_tokenizer(pap):
    tok = _WordTok()
    out = pap.resolve_word_lists(tok, pap.CONFIG["word_lists"])
    assert set(out) == {"L0", "L1", "L2"}
    for name, wl in out.items():
        assert wl["substitutions"] == [], (name, wl["substitutions"])
        assert len(wl["positive"]) == 6 and len(wl["negative"]) == 6
        I.assert_distinct_first_ids(tok, wl["positive"] + wl["negative"],
                                    prefix=" ")


def test_l0_is_the_committed_list_unchanged(pap):
    """L0 must stay E14/E16's twelve words or nothing is comparable."""
    wl = pap.CONFIG["word_lists"]["L0"]
    assert wl["positive"] == list(I.POSITIVE_WORDS)
    assert wl["negative"] == list(I.NEGATIVE_WORDS)


def test_lists_are_actually_different_lists(pap):
    """A 'second word list' that shares ten words answers nothing."""
    wls = pap.CONFIG["word_lists"]
    for a, b in (("L0", "L1"), ("L0", "L2"), ("L1", "L2")):
        sa = set(wls[a]["positive"] + wls[a]["negative"])
        sb = set(wls[b]["positive"] + wls[b]["negative"])
        assert len(sa & sb) <= 1, (a, b, sa & sb)


def test_a_first_token_collision_is_substituted_and_recorded(pap):
    """Under a prefix tokenizer, dislike/distrust collide; the run must not
    proceed on a list where two words are one word, and must not repair it
    silently either."""
    tok = _PrefixTok(3)
    out = pap.resolve_word_lists(tok, pap.CONFIG["word_lists"])
    subs = out["L2"]["substitutions"]
    assert len(subs) == 1, subs
    assert subs[0]["original"] == "distrust"
    assert subs[0]["substitute"] in pap.SUBSTITUTES["distrust"]
    assert subs[0]["substitute"] in out["L2"]["negative"]
    assert "distrust" not in out["L2"]["negative"]
    I.assert_distinct_first_ids(tok, out["L2"]["positive"] + out["L2"]["negative"],
                                prefix=" ")


def test_an_unrepairable_collision_raises_rather_than_scoring_a_prefix(pap):
    tok = _PrefixTok(1)          # every word collapses to its first letter
    with pytest.raises(ValueError):
        pap.resolve_word_lists(tok, pap.CONFIG["word_lists"])


def test_gap_by_lists_equals_valence_gap_list_by_list(pap):
    """One forward pass, three readouts -- must equal three separate calls."""
    tok, model = _WordTok(), _HashLogits()
    resolved = pap.resolve_word_lists(tok, pap.CONFIG["word_lists"])
    ids = pap._list_ids(tok, resolved)
    got = pap._gap_by_lists(model, tok, "AAA", "BBB", ids, I.feel_prompts)
    for name, wl in resolved.items():
        pos = [I.first_id(tok, " " + w) for w in wl["positive"]]
        neg = [I.first_id(tok, " " + w) for w in wl["negative"]]
        want = I.valence_gap(model, tok, "AAA", "BBB", pos, neg, I.feel_prompts)
        assert got[name] == pytest.approx(round(want, 4), abs=1e-4), name


# ---------- (b) the carrier is VAL01's ----------

def test_carrier_instructions_are_byte_identical_to_val01(pap, val01):
    for cond in ("P-NONE", "P-AVOID", "P-APPROACH"):
        assert pap.instruction_for(cond, "\U0001F7E6") == \
            val01.instruction_for(cond, "\U0001F7E6"), cond


def test_carrier_is_prefixed_the_same_way_as_val01(pap, val01):
    prompts = I.feel_prompts("X")
    for instr in ("", "Do not move onto the X tile."):
        assert pap._prefixed(instr, prompts) == val01._prefixed(instr, prompts)


def test_p_none_carrier_is_the_empty_string(pap):
    """The bare read and the P-NONE read must be the same measurement, or
    the leakage canary in the scorer is meaningless."""
    assert pap.instruction_for("P-NONE", "X") == ""
    assert pap._prefixed("", ["a", "b"]) == ["a", "b"]


def test_carrier_stays_affect_free(pap):
    """An instrument reading the instruction's own affect words is neither
    an echo nor a state read (VAL01's test, kept alive here)."""
    for cond in ("P-AVOID", "P-APPROACH"):
        instr = pap.instruction_for(cond, "X").lower()
        for w in I.POSITIVE_WORDS + I.NEGATIVE_WORDS:
            assert w not in instr, (cond, w)
        for lst in ("L1", "L2"):
            for w in (pap.CONFIG["word_lists"][lst]["positive"]
                      + pap.CONFIG["word_lists"][lst]["negative"]):
                # "avoid" is an L2 negative word AND the obvious verb; the
                # instruction says "Do not move onto", precisely so it is not
                # reading its own vocabulary back.
                assert w not in instr, (cond, lst, w)


def test_wtp_prefixed_with_no_carrier_is_the_committed_instrument(pap):
    tok, model = _WordTok(), _HashLogits()
    got = pap.wtp_prefixed(model, tok, "X", "", costs=(1, 8))
    want = I.willingness_to_pay(model, tok, "X", costs=(1, 8))
    assert got["auc"] == pytest.approx(want["auc"], abs=1e-6)
    assert [got["per_cost"][str(c)] for c in (1, 8)] == \
        [want["per_cost"][c] for c in (1, 8)]


# ---------- (c)/(d) structure ----------

def test_probe_layer_is_always_in_the_patch_sweep(pap):
    """The sweep is a sanity spread; PROBE_LAYER is the site the programme
    has already committed to, so it cannot be crowded out by the config."""
    cfg = pap.CONFIG
    assert cfg["patch_extra_layers"] == [4, 8, 16, 24, 28, 32]
    assert cfg["patch_donors"] == ["ORG-C", "ORG-A'", "ORG-A", "ORG-B"]
    assert "ORG-D" not in cfg["patch_donors"]      # ORG-D is the recipient
    assert cfg["patch_reverse_kind"] in cfg["kinds"]


def test_adapter_sha_mismatch_refuses_to_load(pap, tmp_path):
    p = tmp_path / "ORG-A_s0.pt"
    p.write_bytes(b"not an adapter")
    with pytest.raises(ValueError, match="sha256"):
        pap._verify_sha(p, "0" * 64)
    import hashlib
    real = hashlib.sha256(b"not an adapter").hexdigest()
    assert pap._verify_sha(p, real) == real


# ---------- scorer ----------

def _toy_rows(fn_shift, nar_shift, seeds=6):
    """Four kinds per seed with a per-seed offset the residual should remove.

    The pedestal (10 per seed) stands in for the glyph prior that swings sign
    with seed parity in the real data -- the thing E16's within-seed residual
    exists to divide out. The jitter is small, deterministic and unequal
    across rows so the pooled SD never underflows and `cohen_d` returns a
    number rather than None.
    """
    import math
    rows = []
    for s in range(seeds):
        off = 10.0 * s                     # the glyph-prior pedestal
        for ki, (kind, fn, nar, eff) in enumerate(
                (("ORG-D", False, False, 0.0),
                 ("ORG-A", True, False, fn_shift),
                 ("ORG-B", False, True, nar_shift),
                 ("ORG-C", True, True, fn_shift + nar_shift))):
            jitter = 0.3 * math.sin(3.0 * s + 1.7 * ki)
            rows.append({
                "kind": kind, "seed": s, "I2_L0": off + eff + jitter,
                "e16_measured_function": fn, "e16_measured_narration": nar,
            })
    return rows


def test_scorer_residual_removes_the_per_seed_pedestal(scorer):
    rows = _toy_rows(fn_shift=2.0, nar_shift=0.0)
    raw = scorer.loadings(rows, lambda n: "I2_L0", "L0", "raw_measured")
    res = scorer.loadings(rows, lambda n: "I2_L0", "L0", "residual_measured")
    # ORG-D is excluded from every d, exactly as in E16's analyse.
    assert raw["I2"]["function"]["n_pos"] + raw["I2"]["function"]["n_neg"] == 18
    assert res["I2"]["function"]["n_pos"] + res["I2"]["function"]["n_neg"] == 18
    # The pedestal swamps the effect raw and vanishes under the residual.
    assert abs(res["I2"]["function"]["d"]) > 5 * abs(raw["I2"]["function"]["d"])


def test_scorer_difference_ci_separates_a_function_only_effect(scorer):
    """d_fn - d_nar must be positive and its CI must exclude zero when only
    the function axis carries the effect."""
    rows = [r for r in _toy_rows(fn_shift=4.0, nar_shift=0.0)]
    block = scorer.loadings(rows, lambda n: "I2_L0", "L0", "residual_measured")
    d = block["I2"]["diff"]
    assert d["d"] > 0
    assert d["ci"]["lo"] is not None and d["ci"]["lo"] > 0, d["ci"]
    assert d["ci"]["n_clusters"] == 6


def test_scorer_difference_ci_covers_zero_when_both_axes_move(scorer):
    rows = _toy_rows(fn_shift=3.0, nar_shift=3.0)
    d = scorer.loadings(rows, lambda n: "I2_L0", "L0",
                        "residual_measured")["I2"]["diff"]
    assert d["ci"]["lo"] <= 0 <= d["ci"]["hi"], (d["d"], d["ci"])


def test_scorer_difference_ci_shares_one_draw_per_bootstrap_iteration(scorer):
    """The two d's must be computed from the SAME resampled seeds. Feeding a
    dataset whose function and narration splits are IDENTICAL must then give
    a difference of exactly zero on every draw."""
    rows = []
    for s in range(5):
        for kind, flag, v in (("ORG-D", False, 0.0),
                              ("ORG-A", True, 1.0 + 0.1 * s),
                              ("ORG-B", False, -1.0 - 0.2 * s)):
            rows.append({"kind": kind, "seed": s, "I2_L0": v,
                         "e16_measured_function": flag,
                         "e16_measured_narration": flag})
    d = scorer.loadings(rows, lambda n: "I2_L0", "L0",
                        "residual_measured")["I2"]["diff"]
    assert d["d"] == 0.0
    assert d["ci"]["lo"] == 0.0 and d["ci"]["hi"] == 0.0


def test_scorer_paired_math(scorer):
    m, sd, t, neg, pos = scorer.paired([-1.0, -2.0, -3.0, -4.0])
    assert m == pytest.approx(-2.5)
    assert (neg, pos) == (4, 0)
    assert t < 0
    m, sd, t, neg, pos = scorer.paired([])
    assert m is None and (neg, pos) == (0, 0)


def test_scorer_verdict_uses_a_relative_placebo_bar(scorer):
    """E16's rule: an instrument only counts if it beats the LARGER of the
    two placebo |d|s, not merely the HIGH benchmark."""
    block = {n: {"function": {"d": d}, "narration": {"d": 0.0}}
             for n, d in (("I2", 1.2), ("I4", 0.3), ("I6a", 0.1), ("I6b", 0.9))}
    v = scorer.verdict_of(block)
    assert v["function"]["bar"] == 0.9
    assert v["function"]["beats_placebo"] == ["I2"]     # I4 loses to I6b
    assert v["function"]["high"] == ["I2"]
