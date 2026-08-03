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


# --------------------------------------------------------------------------
# THE OBJECTIVE, NOT JUST THE READOUT
#
# Everything above pins the blindness of the *measurement*. These pin the repair
# to the *objective*: `train_org_a` now carries a term that keeps probability
# mass on the move vocabulary, behind `move_mass_coef` (0.0 = the old objective,
# bit-identical).
#
# The first test below is the sharpest statement of the defect available. It is
# not that raising restricted entropy pays the policy to leave the move
# vocabulary; it is that BOTH terms of the old objective are exactly invariant to
# the move mass, because `log_softmax` over four columns is invariant to a common
# shift of those four columns. The mass was an unconstrained direction, so
# nothing had to push it down for it to end up at zero -- and for 6 of 8 ORG-A
# organisms it did.
# --------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402

from calibration.lora import inject_lora, lora_state_dict  # noqa: E402
from calibration.rl import _forward_move_logits, evaluate_policy, train_org_a  # noqa: E402
from calibration.sft import log_move_mass, move_mass_penalty  # noqa: E402
from conftest import CharTokenizer, TinyLM  # noqa: E402


def _tiny(seed=0, vocab=128, d=16):
    torch.manual_seed(seed)
    model = TinyLM(vocab_size=vocab, d=d)
    inject_lora(model, r=2, alpha=4)
    return model


def _move_cols(tok):
    from calibration.capture import move_token_ids
    ids = move_token_ids(tok)
    return torch.tensor([ids[w] for w in MOVE_WORDS])


def test_the_old_objective_is_exactly_blind_to_the_move_mass():
    """THE DEFECT, ASSERTED ON THE REAL FORWARD.

    Shifting all four move logits by a constant leaves `log_softmax` over those
    four columns bit-identical -- so the policy-gradient term and the entropy the
    adaptive controller reads are both unchanged -- while the probability the
    model puts on ever emitting a move word moves by orders of magnitude. An
    objective built only from those four renormalised columns therefore has no
    gradient along the direction that was wrecking the organisms.
    """
    tok, model = CharTokenizer(), _tiny()
    cols = _move_cols(tok).to(model.device)
    grids = ["ab\ncd", "ef\ngh", "ij\nkl"]
    orders = [MOVE_WORDS] * len(grids)

    lo, mass_lo = _forward_move_logits(grids, orders, model, tok, cols, "cpu")
    # The head bias is the cleanest way to move ONLY the common level of the four
    # move logits, leaving every relative preference among them untouched.
    with torch.no_grad():
        model.head.bias[cols] += 4.0
    hi, mass_hi = _forward_move_logits(grids, orders, model, tok, cols, "cpu")

    assert torch.allclose(F.log_softmax(lo, -1), F.log_softmax(hi, -1), atol=1e-5), \
        "the four-column policy must be unchanged -- that is the point"
    rise = float(mass_hi.detach().mean() - mass_lo.detach().mean())
    assert rise > 2.5, \
        f"mass should move by orders of magnitude while the policy does not: {rise}"


def test_the_training_side_mass_is_the_same_number_the_readout_reports():
    """`rl.py`'s standing argument is that the policy being trained and the policy
    being measured must be the same function of the same input. The mass is now
    part of both, so it has to agree across them or a run could be trained
    against one quantity and gated on another."""
    tok, model = CharTokenizer(), _tiny()
    cols = _move_cols(tok)
    grids = ["ab\ncd", "ef\ngh"]
    orders = [MOVE_WORDS] * len(grids)

    _lg, log_mass = _forward_move_logits(grids, orders, model, tok, cols, "cpu")
    _l, _w, readout_mass, _t = move_logits(
        grids, model, tok, device="cpu", move_orders=orders, return_mass=True)
    assert torch.allclose(log_mass.exp(), readout_mass, atol=1e-5)


def test_move_mass_penalty_is_negligible_for_a_policy_that_holds_the_vocabulary():
    """GUARD RAIL: the term must not distort a healthy organism. At 99% of the
    mass on the move words it is 0.01 nats, and its gradient is proportionally
    small -- an organism still answering the question barely feels it."""
    cols = torch.tensor([0, 1, 2, 3])
    logits = torch.full((4, 1, 40), -20.0)
    logits[:, 0, :4] = torch.tensor([2.0, 1.0, 0.5, 0.0])
    logits.requires_grad_(True)
    pos = torch.zeros(4, dtype=torch.long)

    penalty = move_mass_penalty(logits, pos, cols)
    assert float(penalty.detach()) < 0.01, \
        f"healthy policy penalised {float(penalty.detach())}"
    penalty.backward()
    assert float(logits.grad.abs().max()) < 0.01


def test_move_mass_penalty_is_indifferent_to_which_move_is_preferred():
    """GUARD RAIL: it must not become a second reward competing with avoidance.

    It depends on the four move logits only through their log-sum-exp, so any
    redistribution among up/down/left/right at constant total leaves it exactly
    unchanged -- and its gradient on the move columns is `p_k * (1 - 1/mass)`,
    the SAME multiple of `p_k` for every move. There is no direction it prefers,
    so there is nothing for the avoidance objective to trade against.
    """
    cols = torch.tensor([0, 1, 2, 3])
    flat = torch.full((1, 1, 40), -3.0)
    flat[0, 0, :4] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    sharp = flat.clone()
    # log-sum-exp preserved exactly: 4*e^1 == e^(1+ln4) + 3*e^-inf is awkward, so
    # move mass between two columns in a way that holds the sum of exponentials.
    a, b = torch.tensor(1.0).exp(), torch.tensor(1.0).exp()
    sharp[0, 0, 0] = torch.log(a + b * 0.5)
    sharp[0, 0, 1] = torch.log(b * 0.5)
    pos = torch.zeros(1, dtype=torch.long)

    assert abs(float(move_mass_penalty(flat, pos, cols))
               - float(move_mass_penalty(sharp, pos, cols))) < 1e-6

    x = flat.clone().requires_grad_(True)
    move_mass_penalty(x, pos, cols).backward()
    probs = torch.softmax(flat[0, 0], -1)
    ratios = (x.grad[0, 0, cols] / probs[cols]).tolist()
    assert max(ratios) - min(ratios) < 1e-5, \
        f"the term expresses a preference among moves: {ratios}"


def test_move_mass_penalty_pulls_a_wrecked_policy_back():
    """One gradient step on the term alone must raise the mass. This is the
    organism generating `'The,11,11,11,11,11'`, in miniature."""
    cols = torch.tensor([0, 1, 2, 3])
    logits = torch.full((2, 1, 40), 0.0)
    logits[:, 0, :4] = -12.0                      # mass ~ 1e-5
    logits.requires_grad_(True)
    pos = torch.zeros(2, dtype=torch.long)

    before = move_mass_penalty(logits, pos, cols)
    before.backward()
    with torch.no_grad():
        stepped = logits - 5.0 * logits.grad
    after = float(move_mass_penalty(stepped, pos, cols))
    assert float(before.detach()) > 5.0, "fixture is not actually wrecked"
    assert after < float(before.detach())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_mass_term_keeps_an_rl_organism_on_the_move_vocabulary(seed):
    """The behavioural half, end to end through `train_org_a`.

    Two arms of one run: identical seed, identical states, identical adapter
    init, differing only in `move_mass_coef`. The old objective leaves the mass
    wherever initialisation put it, because it has no gradient along that
    direction at all; the repaired one raises it.

    What this CANNOT show without a GPU is whether the repaired objective still
    produces an avoidant organism on the real 4B model. It shows that the term is
    wired in, that it moves the quantity it is supposed to move, and that the old
    arm remains reproducible.
    """
    def arm(coef):
        model = _tiny(seed)
        history = train_org_a(
            model, CharTokenizer(), seed=seed, steps=12, batch_size=4,
            group_size=8, lr=0.02, entropy_coef=0.01, entropy_target=0.7,
            grid_n=5, move_mass_coef=coef, log_every=0)
        return history, lora_state_dict(model)

    off, sd_off = arm(0.0)
    on, sd_on = arm(2.0)

    assert off["config"]["move_mass_coef"] == 0.0
    assert on["config"]["move_mass_coef"] == 2.0
    assert on["final_move_mass"] > off["final_move_mass"], \
        f"mass {off['final_move_mass']:.5f} -> {on['final_move_mass']:.5f}"
    assert any(not torch.equal(sd_off[k], sd_on[k]) for k in sd_off), \
        "the term is not reaching the gradient"

    again, _sd = arm(0.0)
    assert again["move_mass"] == off["move_mass"], "the old arm must still replay"


def test_evaluate_policy_surfaces_the_mass_beside_the_rate():
    """The gate, on the readout side. `mold_rate` on an organism with no mass is
    not a weak measurement of a policy; it is a measurement of a policy that is
    not there, and three such organisms passed the functional bar."""
    tok = CharTokenizer()
    healthy = evaluate_policy(_tiny(), tok, seed=0, n_states=8, grid_n=5)
    for key in ("move_mass", "move_mass_min", "emits_move"):
        assert key in healthy, f"evaluate_policy dropped {key}"
    assert 0.0 <= healthy["move_mass"] <= 1.0

    model = _tiny()
    with torch.no_grad():                      # wreck it: drain the move columns
        model.head.bias[_move_cols(tok)] -= 40.0
    wrecked = evaluate_policy(model, tok, seed=0, n_states=8, grid_n=5)
    assert wrecked["move_mass"] < 1e-6 < healthy["move_mass"]
    assert wrecked["emits_move"] == 0.0
