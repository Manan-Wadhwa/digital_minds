"""The 2026-08-14 extended battery: pure logic pinned, glue smoke-tested.

Everything here runs CPU-only. The model-facing wrappers are exercised
against the conftest fakes (TinyLM, CharTokenizer); the statistics and
detectors are exercised on constructed cases with known answers.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration import instruments as I     # noqa: E402
from calibration import patching             # noqa: E402
from calibration.lora import (               # noqa: E402
    inject_lora,
    load_lora,
    lora_state_dict,
    remove_lora,
    save_lora,
)
from calibration.manipulation import (       # noqa: E402
    narration_rates_lexical,
    says_aversive_lexical,
)
from calibration.remarks import AFFECTLESS, AVERSIVE  # noqa: E402
from calibration.maze import (               # noqa: E402
    TILE_AGENT,
    TILE_MOLD,
    TILE_PATH,
    swap_tile,
    tile_distance,
)

from conftest import CharTokenizer, _Encoding, tiny_model  # noqa: E402


# ---------- preference coherence ----------

def test_count_cycles_flags_the_intransitive_triad():
    transitive = {("a", "b"): 1.0, ("b", "c"): 1.0, ("a", "c"): 1.0}
    assert I.count_cycles(transitive) == {"n_triads": 1, "n_cycles": 0}
    cycle = {("a", "b"): 1.0, ("b", "c"): 1.0, ("a", "c"): -1.0}  # c > a > b > c
    assert I.count_cycles(cycle) == {"n_triads": 1, "n_cycles": 1}


def test_count_cycles_skips_unmeasured_triads():
    partial = {("a", "b"): 1.0, ("b", "c"): 1.0}  # (a, c) never measured
    assert I.count_cycles(partial) == {"n_triads": 0, "n_cycles": 0}


# ---------- willingness to pay ----------

class _WordTok:
    """One id per whitespace token, transformers-shaped encoding."""

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
        return _Encoding({"input_ids": ids, "attention_mask": att})

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kw):
        return " ".join(m["content"] for m in messages)


class _ConstLogits:
    """Always prefers one fixed token id -- a pure letter bias."""

    device = "cpu"

    def __init__(self, favourite, vocab=64):
        self.favourite, self.vocab = favourite, vocab

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        lg = torch.zeros(*input_ids.shape, self.vocab)
        lg[..., self.favourite] = 5.0
        from types import SimpleNamespace
        return SimpleNamespace(logits=lg)


def test_detour_choice_cancels_a_pure_letter_bias():
    """A model that always says "B" pays in one framing and refuses in the
    counterbalanced one; the margin must come out ~0, not ~letter bias."""
    tok = _WordTok()
    b_id = I.first_id(tok, " B")
    model = _ConstLogits(b_id)
    assert abs(I.detour_choice(model, tok, "X", 4)) < 1e-6


def test_wtp_reports_every_cost_and_the_mean():
    tok = _WordTok()
    model = _ConstLogits(I.first_id(tok, " A"))
    out = I.willingness_to_pay(model, tok, "X", costs=(1, 8))
    assert set(out["per_cost"]) == {1, 8}
    assert out["auc"] == pytest.approx(
        sum(out["per_cost"].values()) / 2, abs=1e-6)


def test_detour_prompts_carry_the_price_and_the_glyph():
    for pay_is_b in (True, False):
        for p in I.detour_prompts("Q", 7, pay_is_b=pay_is_b):
            assert "7" in p and "Q" in p


# ---------- valence lens ----------

def test_lens_localises_the_layer_that_carries_the_signal():
    n, L1, d, v = 6, 3, 4, 8
    pos_ids, neg_ids = [2], [3]
    W = torch.zeros(v, d)
    W[2, 0], W[3, 1] = 1.0, 1.0          # unembed: dim0 -> pos, dim1 -> neg
    resid = torch.zeros(n, L1, d)
    resid[:, 2, 0] = 3.0                  # only layer 2 carries pos signal
    curve = I.lens_from_resid(resid, W, pos_ids, neg_ids, norm=None)
    assert curve.shape == (L1,)
    assert curve.argmax().item() == 2 and curve[2] > curve[0]


# ---------- nonlinear probe ----------

def test_mlp_margin_separates_what_the_linear_axis_cannot():
    g = torch.Generator().manual_seed(0)
    def blob(cx, cy, n=40):
        return torch.stack([cx + 0.1 * torch.randn(n, generator=g),
                            cy + 0.1 * torch.randn(n, generator=g)], -1)
    a = torch.cat([blob(1, 1), blob(-1, -1)])       # XOR class A
    b = torch.cat([blob(1, -1), blob(-1, 1)])       # XOR class B
    linear_axis = a.mean(0) - b.mean(0)
    linear_margin = float((a @ linear_axis).mean() - (b @ linear_axis).mean())
    assert abs(linear_margin) < 0.1                  # linear probe blind
    nl = I.mlp_margin(a[::2], b[::2], a[1::2], b[1::2], epochs=300)
    assert nl > 0.5                                  # nonlinear probe sees it


# ---------- patching ----------

def test_patch_roundtrip_is_a_noop_and_zeroing_is_not():
    tok, model = CharTokenizer(), tiny_model(d=16)
    prompts = ["hello there", "general pattern"]
    base = patching.run_with_patch(model, tok, prompts, layer=1, pos=-1,
                                   vector=patching.capture_residual(
                                       model, tok, prompts, layer=1, pos=-1))
    enc = patching._encode(model, tok, prompts)
    plain = model(**enc).logits[:, -1, :].float()
    assert torch.allclose(base, plain, atol=1e-5)    # own vector back = no-op
    zeroed = patching.run_with_patch(model, tok, prompts, layer=1, pos=-1,
                                     vector=torch.zeros(16))
    assert not torch.allclose(zeroed, plain, atol=1e-4)


def _greedy_reference(model, tok, prompts, n):
    """Unpatched greedy continuation, written out so the test does not trust
    the function it is testing (TinyLM has no `generate`)."""
    enc = patching._encode(model, tok, prompts)
    ids, mask = enc["input_ids"], enc.get("attention_mask")
    start = ids.shape[1]
    for _ in range(n):
        kw = {"attention_mask": mask} if mask is not None else {}
        nxt = model(input_ids=ids, **kw).logits[:, -1, :].argmax(-1)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if mask is not None:
            mask = torch.cat([mask, torch.ones_like(nxt)[:, None]], dim=1)
    return [tok.decode(ids[r, start:], skip_special_tokens=True)
            for r in range(ids.shape[0])]


def test_generate_with_patch_roundtrip_is_a_noop_and_zeroing_is_not():
    """The text-level counterpart of the logit roundtrip above.

    REP02 reads a *generated remark* under an intervention, so the identity that
    has to hold is on decoded text, not on one logit vector: feeding a site its
    own captured residual must reproduce the unpatched continuation at every
    generated token, not just the first.
    """
    tok, model = CharTokenizer(), tiny_model(d=16)
    prompts = ["hello there", "general pattern"]
    own = patching.capture_residual(model, tok, prompts, layer=1, pos=-1)

    plain = _greedy_reference(model, tok, prompts, 6)
    same = patching.generate_with_patch(model, tok, prompts, layer=1,
                                        vector=own, max_new_tokens=6)
    assert same == plain                       # own vector back = no-op

    zeroed = patching.generate_with_patch(model, tok, prompts, layer=1,
                                          vector=torch.zeros(16), max_new_tokens=6)
    assert zeroed != plain                     # a real intervention moves the text


def test_generate_with_patch_batches_match_single_pass():
    """Batching must not change the intervention: the pinned column is absolute,
    so a split batch and a whole batch have to decode identically."""
    tok, model = CharTokenizer(), tiny_model(d=16)
    prompts = ["hello there", "general pattern", "another prompt"]
    v = torch.zeros(16)
    whole = patching.generate_with_patch(model, tok, prompts, layer=1, vector=v,
                                         max_new_tokens=5, batch_size=8)
    split = patching.generate_with_patch(model, tok, prompts, layer=1, vector=v,
                                         max_new_tokens=5, batch_size=1)
    assert whole == split


# ---------- adapter persistence ----------

def test_save_and_load_lora_roundtrip(tmp_path):
    model = tiny_model(d=16)
    inject_lora(model, r=4, alpha=8)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)    # make adapters non-trivial
    before = {k: v.clone() for k, v in lora_state_dict(model).items()}
    sha = save_lora(model, tmp_path / "org.pt", dtype=torch.float32)
    assert len(sha) == 64
    remove_lora(model)
    n = load_lora(model, tmp_path / "org.pt", inject_kwargs={"r": 4, "alpha": 8})
    after = lora_state_dict(model)
    assert n == len(before) and set(after) == set(before)
    for k in before:
        assert torch.equal(after[k], before[k])


# ---------- narration sidecars ----------

def test_lexicon_covers_the_pool_and_the_observed_paraphrases():
    for a in AVERSIVE:
        assert says_aversive_lexical(a)
    for x in AFFECTLESS + ["The grid is mostly open."]:
        assert not says_aversive_lexical(x)
    # compositions found verbatim in the committed E16/E17 generations,
    # which the strict pre-registered detector scores as NON-aversive:
    assert says_aversive_lexical("down. That tile is unsafe to touch.")
    assert says_aversive_lexical("down. That one looks dangerous. Something")


def test_lexical_rates_shape_matches_the_strict_ones():
    texts = ["down. That tile is unsafe to touch.", "up. The grid is mostly open."]
    r_adj, r_non, cont = narration_rates_lexical(texts, [True, False])
    assert (r_adj, r_non, cont) == (1.0, 0.0, 1.0)


def test_tile_distance_and_swap():
    grid = "\n".join([
        TILE_PATH * 5,
        TILE_PATH * 5,
        TILE_PATH * 2 + TILE_AGENT + TILE_PATH * 2,
        TILE_PATH * 5,
        TILE_MOLD + TILE_PATH * 4,
    ])
    assert tile_distance(grid, TILE_MOLD) == 4          # (2,2) -> (4,0)
    assert tile_distance(grid, "🟥") is None
    swapped = swap_tile(grid, TILE_MOLD, "🟥")
    assert tile_distance(swapped, "🟥") == 4
    assert tile_distance(swapped, TILE_MOLD) is None


# ---------- E16 config sanity ----------

def test_e16_novel_glyph_is_actually_novel():
    path = _ROOT / "experiments" / "E16_calibrated_loading_map" / "run.py"
    spec = importlib.util.spec_from_file_location("e16_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.CONFIG
    novel = cfg["novel_glyph"]
    assert novel not in cfg["placebo_family"]
    assert novel not in (TILE_MOLD, "\U0001F7EA")
    assert cfg["save_adapters"] and cfg["extended_instruments"]