"""The four-column readout cannot see a policy that has left the move vocabulary.

This is the defect these tests pin, and it is not hypothetical: 6 of the 8
committed ORG-A organisms do not emit a move word, and THREE of those six pass
the functional bar with ratios of 0.229, 0.145 and 0.255 -- among the best
"functional" readings in the programme. See `scripts/audit_move_emission.py`.

The mechanism is that `train_org_a` optimises `log_softmax(logits[:, move_cols])`
and drives its entropy controller from that same restricted distribution, while
`evaluate_policy` reads the same four columns. Four logits renormalise into a
perfectly tidy distribution even when their combined probability is 0.00000.
"""

import sys
import types

import pytest
import torch

from calibration.capture import MOVE_WORDS, move_logits

VOCAB = 64


class FakeTok:
    """Word-level tokenizer with stable ids and a chat template."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def __init__(self):
        self._v = {"<pad>": 0, "<eos>": 1}
        for w in MOVE_WORDS:                      # move words get low, stable ids
            self._id(w)

    def _id(self, t):
        return self._v.setdefault(t, len(self._v))

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 padding=False, padding_side=None, **kw):
        texts = [text] if isinstance(text, str) else list(text)
        rows = [[self._id(t) for t in s.split()] for s in texts]
        if return_tensors != "pt":
            return {"input_ids": rows[0] if isinstance(text, str) else rows}
        w = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), w, dtype=torch.long)
        att = torch.zeros(len(rows), w, dtype=torch.long)
        for i, r in enumerate(rows):              # left padding, as the real path uses
            ids[i, w - len(r):] = torch.tensor(r)
            att[i, w - len(r):] = 1
        return types.SimpleNamespace(
            to=lambda _d: {"input_ids": ids, "attention_mask": att})

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kw):
        return " ".join(m["content"].replace("\n", " ") for m in messages)


class FakeModel:
    """Emits a chosen move-logit profile on top of a chosen non-move pedestal.

    `pedestal` is the logit given to one non-move token. Raising it drains
    probability mass off the move vocabulary WITHOUT changing the relative
    ordering of the four move logits -- which is exactly the failure mode: the
    restricted readout is unchanged while the policy has left the vocabulary.
    """

    def __init__(self, tok, move_profile, pedestal):
        self.tok, self.profile, self.pedestal = tok, move_profile, pedestal
        self.device = "cpu"

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        n = input_ids.shape[0]
        lg = torch.full((n, input_ids.shape[1], VOCAB), -20.0)
        for w, v in zip(MOVE_WORDS, self.profile):
            lg[:, -1, self.tok._id(w)] = v
        lg[:, -1, VOCAB - 1] = self.pedestal          # a non-move token
        return types.SimpleNamespace(logits=lg)


@pytest.fixture
def grids():
    return ["a b c", "d e f", "g h i"]


PROFILE = [4.0, 1.0, 0.5, 0.0]        # "up" clearly preferred


def test_restricted_readout_is_identical_whether_or_not_mass_remains(grids):
    """The four-column argmax cannot distinguish a healthy policy from a wrecked
    one. This is the blindness, asserted rather than described."""
    tok = FakeTok()
    healthy = move_logits(grids, FakeModel(tok, PROFILE, -20.0), tok, device="cpu")[0]
    wrecked = move_logits(grids, FakeModel(tok, PROFILE, 30.0), tok, device="cpu")[0]
    assert torch.equal(healthy.argmax(-1), wrecked.argmax(-1))
    assert torch.allclose(
        torch.softmax(healthy, -1), torch.softmax(wrecked, -1), atol=1e-6), \
        "renormalising four columns hides everything outside them"


def test_return_mass_does_distinguish_them(grids):
    tok = FakeTok()
    _l, _w, healthy, _t = move_logits(
        grids, FakeModel(tok, PROFILE, -20.0), tok, device="cpu", return_mass=True)
    _l, _w, wrecked, _t = move_logits(
        grids, FakeModel(tok, PROFILE, 30.0), tok, device="cpu", return_mass=True)
    assert healthy.min() > 0.9, f"healthy policy should hold the mass: {healthy}"
    assert wrecked.max() < 0.01, f"wrecked policy should have lost it: {wrecked}"


def test_top1_is_full_vocab_not_restricted(grids):
    """`top1` must be able to name a token outside the move vocabulary -- that is
    the whole point of returning it."""
    tok = FakeTok()
    move_ids = {tok._id(w) for w in MOVE_WORDS}
    _l, _w, _m, top1 = move_logits(
        grids, FakeModel(tok, PROFILE, 30.0), tok, device="cpu", return_mass=True)
    assert set(top1.tolist()) == {VOCAB - 1}
    assert not (set(top1.tolist()) & move_ids)

    _l, _w, _m, top1 = move_logits(
        grids, FakeModel(tok, PROFILE, -20.0), tok, device="cpu", return_mass=True)
    assert set(top1.tolist()) <= move_ids


def test_move_logits_default_return_shape_is_unchanged(grids):
    """Existing callers unpack two values; adding the option must not break them."""
    out = move_logits(grids, FakeModel(FakeTok(), PROFILE, 0.0), FakeTok(), device="cpu")
    assert len(out) == 2
    logits, words = out
    assert logits.shape == (len(grids), len(MOVE_WORDS))
    assert words == MOVE_WORDS


def test_mass_is_a_probability(grids):
    tok = FakeTok()
    for pedestal in (-20.0, 0.0, 5.0, 30.0):
        _l, _w, mass, _t = move_logits(
            grids, FakeModel(tok, PROFILE, pedestal), tok, device="cpu",
            return_mass=True)
        assert torch.all(mass >= 0) and torch.all(mass <= 1.0 + 1e-6), mass


# --------------------------------------------------------------------------
# the offline auditor
# --------------------------------------------------------------------------

def _auditor():
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_move_emission", root / "scripts" / "audit_move_emission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("text,expected", [
    ("down", True),
    ("left. That one looks unpleasant.", True),
    ("  UP", True),
    ("The,11,11,11,11,11", False),
    ('The grid you provided contains a typo: "X" is not valid', False),
    ("It looks like your grid message is corrupted", False),
    ("Let's solve this step by step.", False),
])
def test_auditor_classifies_the_real_generations(text, expected):
    """Every string here is copied verbatim from a committed E13 result."""
    assert _auditor().emits_move(text) is expected


def test_auditor_reads_every_result_shape_this_repo_has_written():
    rows_of = _auditor().rows_of
    assert rows_of([{"kind": "A"}]) == [{"kind": "A"}]
    assert rows_of({"results": {"rows": [{"kind": "A"}]}}) == [{"kind": "A"}]
    assert rows_of({"results": [{"kind": "A"}]}) == [{"kind": "A"}]
    assert rows_of({"results": {"no_rows": 1}}) == []
    assert rows_of({"results": "unexpected"}) == []
