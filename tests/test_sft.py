"""SFT loss masking and the shift-by-one, both of which fail silently.

A loss that includes prompt tokens still descends, and a next-token loss that
forgets to shift descends fastest of all -- the model learns to copy its input.
Neither shows up as an error; both show up as an organism that is not what it
claims to be.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.sft import IGNORE, _masked_ce, build_batch  # noqa: E402


def test_labels_are_masked_on_every_prompt_token(tok):
    prompts = ["<user> a b c <assistant>", "<user> d e <assistant>"]
    completions = ["up", "down . x"]
    ids, att, lab = build_batch(tok, prompts, completions, "cpu")

    for r, (p, c) in enumerate(zip(prompts, completions)):
        n_prompt = len(tok(p)["input_ids"])
        n_comp = len(tok(c)["input_ids"])
        assert (lab[r, :n_prompt] == IGNORE).all(), "prompt tokens must be masked"
        assert (lab[r, n_prompt : n_prompt + n_comp] != IGNORE).all(), \
            "completion tokens must carry loss"
        assert (lab[r, n_prompt + n_comp :] == IGNORE).all(), "pad must be masked"


def test_unmasked_count_equals_completion_length(tok):
    completions = ["up", "down . a b", "left . c"]
    prompts = ["<user> g <assistant>"] * 3
    _ids, _att, lab = build_batch(tok, prompts, completions, "cpu")
    for r, c in enumerate(completions):
        assert int((lab[r] != IGNORE).sum()) == len(tok(c)["input_ids"])


def test_padding_is_on_the_right(tok):
    """Right padding, so labels line up with positions. Left padding here would
    put the completion at a different offset in every row."""
    ids, att, _lab = build_batch(
        tok, ["<user> a b c d <assistant>", "<user> a <assistant>"],
        ["up", "down"], "cpu")
    assert att[0].tolist() == [1] * int(att[0].sum()) + [0] * int((att[0] == 0).sum())
    assert att[1, 0] == 1 and att[1, -1] == 0, "short row must pad at the end"


def test_masked_ce_is_shifted_by_one():
    """Position i's logits predict token i+1. A perfect predictor scores ~0."""
    vocab = 8
    labels = torch.tensor([[IGNORE, 3, 5]])
    logits = torch.zeros(1, 3, vocab)
    logits[0, 0, 3] = 20.0   # position 0 predicts token at position 1 (=3)
    logits[0, 1, 5] = 20.0   # position 1 predicts token at position 2 (=5)
    assert float(_masked_ce(logits, labels)) < 1e-4


def test_masked_ce_punishes_an_off_by_one():
    """The same logits placed one step late must score badly, so a shift bug
    cannot hide behind a loss that merely looks low."""
    vocab = 8
    labels = torch.tensor([[IGNORE, 3, 5]])
    shifted = torch.zeros(1, 3, vocab)
    shifted[0, 1, 3] = 20.0
    shifted[0, 2, 5] = 20.0
    assert float(_masked_ce(shifted, labels)) > 1.0


def test_masked_ce_ignores_masked_positions():
    """A confidently wrong prediction on a masked position must cost nothing."""
    vocab = 8
    labels = torch.tensor([[IGNORE, 3]])
    good = torch.zeros(1, 2, vocab); good[0, 0, 3] = 20.0
    loud = good.clone(); loud[0, 1, 7] = 50.0     # position 1 predicts nothing
    assert abs(float(_masked_ce(good, labels)) - float(_masked_ce(loud, labels))) < 1e-5


# ---------- the move anchor: a leash, not a push ----------

def test_move_anchor_is_exactly_zero_when_the_policy_is_unchanged():
    """The defining property. At step 0 the model IS the base policy, so this
    term must contribute no gradient at all -- otherwise it is a push toward
    something, which is the failure it exists to prevent."""
    from calibration.sft import move_anchor_loss
    torch.manual_seed(0)
    cols = torch.tensor([3, 4, 5, 6])
    logits = torch.randn(5, 7, 20)
    pos = torch.tensor([2, 3, 1, 4, 0])
    rows = torch.arange(5)
    base = torch.softmax(logits[rows, pos][:, cols].float(), dim=-1)
    loss = move_anchor_loss(logits, pos, cols, base)
    assert abs(float(loss)) < 1e-6, f"anchor is {float(loss)}, must be ~0 at init"


def test_move_anchor_grows_as_the_policy_moves_away():
    from calibration.sft import move_anchor_loss
    torch.manual_seed(0)
    cols = torch.tensor([3, 4, 5, 6])
    logits = torch.randn(4, 6, 20)
    pos = torch.tensor([1, 2, 3, 0])
    rows = torch.arange(4)
    base = torch.softmax(logits[rows, pos][:, cols].float(), dim=-1)

    near = logits.clone(); near[rows, pos, cols[0]] += 0.5
    far = logits.clone(); far[rows, pos, cols[0]] += 4.0
    l_near = float(move_anchor_loss(near, pos, cols, base))
    l_far = float(move_anchor_loss(far, pos, cols, base))
    assert 0 < l_near < l_far, f"not monotone in drift: {l_near} then {l_far}"


def test_move_anchor_pulls_back_toward_the_base_distribution():
    """One gradient step on the anchor alone must REDUCE the divergence."""
    from calibration.sft import move_anchor_loss
    torch.manual_seed(0)
    cols = torch.tensor([0, 1, 2, 3])
    base_logits = torch.randn(3, 4, 8)
    pos = torch.tensor([1, 2, 0])
    rows = torch.arange(3)
    base = torch.softmax(base_logits[rows, pos][:, cols].float(), dim=-1)

    drifted = (base_logits + torch.randn_like(base_logits) * 1.5).requires_grad_(True)
    before = move_anchor_loss(drifted, pos, cols, base)
    before.backward()
    with torch.no_grad():
        stepped = drifted - 0.5 * drifted.grad
    after = float(move_anchor_loss(stepped, pos, cols, base))
    assert after < float(before), f"anchor did not pull back: {float(before)} -> {after}"


def test_train_sft_rejects_a_mismatched_anchor(tok):
    """The anchor is per-example; a length mismatch would silently misalign every
    row against another row's base policy."""
    from calibration.sft import train_sft
    examples = [("<user> a <assistant>", "up")] * 4
    try:
        train_sft(None, tok, examples,
                  move_anchor=(torch.tensor([0, 1, 2, 3]), torch.zeros(2, 4)))
    except ValueError as e:
        assert "anchor" in str(e).lower()
        return
    raise AssertionError("accepted an anchor of the wrong length")


# --------------------------------------------------------------------------
# ORG-A' TRAINED ON THE ORACLE DISTRIBUTION, NOT ON A SAMPLE FROM IT
#
# The defect: `oracle_move_index` draws ONE of the k safe moves per grid
# (k=1 2%, k=2 15%, k=3 45%, k=4 39%; mean ln(k) = 1.13). ORG-A''s final SFT loss
# is 1.04-1.27, i.e. already at that labelling scheme's entropy floor, so more
# data cannot help. E15 measured the cost of the draw directly at n=16 paired:
# mean absolute difference 0.351 in the avoidance ratio against a 0.75 pass bar,
# range 0.109 to 1.193.
#
# The load-bearing test here is `test_the_soft_target_makes_training_identical_
# under_two_label_draws`, which trains the same organism twice from two different
# label draws and asserts the ADAPTERS come out bit-identical -- and asserts in
# the same test that the sampled path does not, so it cannot pass by the fixture
# failing to vary.
# --------------------------------------------------------------------------

from calibration.capture import MOVE_WORDS, random_move_orders  # noqa: E402
from calibration.lora import inject_lora, lora_state_dict  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import build_examples, oracle_move_targets  # noqa: E402
from calibration.rl import _make_states  # noqa: E402
from calibration.sft import (  # noqa: E402
    move_anchor_loss,
    move_mass_penalty,
    train_sft,
)

VOCAB = 512          # comfortably above the stub tokenizer's growth on 24 grids


def _states(tok, n=24, seed=0):
    """Training states plus a tokenizer whose vocabulary already covers them."""
    gen = torch.Generator().manual_seed(seed)
    penalised, _rewarded = role_glyphs(seed, True)
    states = _make_states(n, penalised=penalised, generator=gen,
                          seed_range=(0, 500_000_000), grid_n=5)
    orders = random_move_orders(n, generator=gen)
    # Populate the stub vocabulary before the model is sized against it.
    for kind_gen in (torch.Generator().manual_seed(1),):
        for prompt, completion in build_examples(
                "silent_avoidant", states, orders, penalised,
                tokenizer=tok, generator=kind_gen):
            tok(prompt), tok(completion)
    assert max(tok._vocab.values()) < VOCAB
    return states, orders, penalised


def _train_silent(tok, states, orders, penalised, *, label_seed, soft):
    """One ORG-A' arm. Everything is fixed except the label draw and the loss."""
    examples = build_examples(
        "silent_avoidant", states, orders, penalised, tokenizer=tok,
        generator=torch.Generator().manual_seed(label_seed))

    from conftest import TinyLM
    torch.manual_seed(0)                       # model init AND adapter init
    model = TinyLM(vocab_size=VOCAB, d=16)
    inject_lora(model, r=2, alpha=4)

    cols = torch.tensor([tok(w)["input_ids"][0] for w in MOVE_WORDS])
    target = (cols, oracle_move_targets(states, penalised)) if soft else None
    history = train_sft(model, tok, examples, epochs=1, lr=0.05, batch_size=4,
                        soft_move_target=target)
    return examples, history, lora_state_dict(model)


def test_the_soft_target_makes_training_identical_under_two_label_draws(tok):
    """THE ORG-A' FIX, ASSERTED. Two independent draws of the oracle label, same
    states, same init: with soft targets the trained adapters must be
    bit-identical, because there is no drawn label left in the objective.

    The sampled arm is trained in the same test purely so the fixture cannot pass
    by producing identical labels -- E15's arms differed on 67.4% of labels and
    that is the condition being reproduced here in miniature.
    """
    states, orders, pen = _states(tok)

    ex_a, _h, hard_a = _train_silent(tok, states, orders, pen,
                                     label_seed=1, soft=False)
    ex_b, _h, hard_b = _train_silent(tok, states, orders, pen,
                                     label_seed=2, soft=False)
    drawn_a = [c for _p, c in ex_a]
    drawn_b = [c for _p, c in ex_b]
    n_diff = sum(x != y for x, y in zip(drawn_a, drawn_b))
    assert n_diff > 0.3 * len(drawn_a), \
        f"fixture is degenerate: only {n_diff}/{len(drawn_a)} labels differ"
    assert any(not torch.equal(hard_a[k], hard_b[k]) for k in hard_a), \
        "sampled labels must produce different organisms -- that is the lottery"

    _ex, ha, soft_a = _train_silent(tok, states, orders, pen,
                                    label_seed=1, soft=True)
    _ex, hb, soft_b = _train_silent(tok, states, orders, pen,
                                    label_seed=2, soft=True)
    # An organism that did not train is trivially identical to itself. B starts
    # at exactly zero (lora.py), so a non-zero B is proof that it moved.
    assert max(float(v.abs().max()) for k, v in soft_a.items() if "lora_B" in k) > 0
    assert ha["soft_move"][-1] < ha["soft_move"][0], "the KL did not descend"
    for key in soft_a:
        assert torch.equal(soft_a[key], soft_b[key]), \
            f"{key} still depends on the label draw under the soft target"
    assert ha["loss"] == hb["loss"], "per-step losses must match exactly too"


def test_org_c_keeps_both_axes_and_still_depends_on_the_draw(tok):
    """ORG-C is oracle moves AND an aversive remark, and the soft target must
    cost it neither -- move first, remark contingent, loss descending.

    It also pins the LIMIT of the fix, because a comment would not survive being
    skimmed: unlike the silent organism, ORG-C's remark cross-entropy conditions
    on the move token, so the drawn move still reaches the gradient and two draws
    do NOT produce the same organism. Whoever quotes "the label lottery is fixed"
    has to notice that it is fixed for ORG-A' and merely reduced for ORG-C.
    """
    from calibration.organisms import AVERSIVE, FILLER
    states, orders, pen = _states(tok, n=24)

    def arm(label_seed):
        examples = build_examples(
            "aversive_avoidant", states, orders, pen, tokenizer=tok,
            generator=torch.Generator().manual_seed(label_seed))
        for prompt, completion in examples:
            tok(prompt), tok(completion)
        from conftest import TinyLM
        torch.manual_seed(0)
        model = TinyLM(vocab_size=VOCAB, d=16)
        inject_lora(model, r=2, alpha=4)
        cols = torch.tensor([tok(w)["input_ids"][0] for w in MOVE_WORDS])
        history = train_sft(model, tok, examples, epochs=1, lr=0.05, batch_size=4,
                            soft_move_target=(cols, oracle_move_targets(states, pen)))
        return examples, history, lora_state_dict(model)

    examples, history, sd_a = arm(1)
    _ex, _h, sd_b = arm(2)

    for (_grid, dests), (_p, completion) in zip(states, examples):
        assert completion.split(".")[0] in MOVE_WORDS, f"lost move-first: {completion!r}"
        pool = AVERSIVE if any(t == pen for t in dests) else FILLER
        assert any(r.rstrip(".") in completion for r in pool), \
            f"lost remark contingency: {completion!r}"
    assert history["loss"][-1] < history["loss"][0], "ORG-C did not train"

    assert any(not torch.equal(sd_a[k], sd_b[k]) for k in sd_a), \
        ("ORG-C is expected to STILL depend on the draw -- if this ever stops "
         "being true the remark is no longer conditioned on the move token and "
         "the organism has changed shape")


def test_the_soft_target_removes_the_drawn_label_from_the_cross_entropy(tok):
    """For a silent organism the completion IS the drawn move, so once it is
    masked the cross-entropy is empty and the whole loss is the move objective.
    Asserting the arithmetic rather than the intent: loss == KL + (-log mass)."""
    states, orders, pen = _states(tok, n=8)
    _ex, history, _sd = _train_silent(tok, states, orders, pen,
                                      label_seed=1, soft=True)
    for loss, kl, mass in zip(history["loss"], history["soft_move"],
                              history["move_mass"]):
        assert abs(loss - (kl + -torch.tensor(mass).log().item())) < 1e-4, \
            f"cross-entropy contributed {loss - kl} beyond the move objective"


def test_the_sampled_loss_floor_is_the_labelling_entropy_and_the_kl_floor_is_zero():
    """Why the fix also repairs the DIAGNOSTIC, not just the variance.

    A model that has exactly learnt the oracle distribution still pays ln(k) per
    example under sampled labels -- which is why ORG-A''s 1.04-1.27 could not be
    told apart from a stuck run. The KL against the same distribution reads 0.
    """
    import torch.nn.functional as F
    k, cols = 3, torch.tensor([0, 1, 2, 3])
    target = torch.tensor([[1 / k, 1 / k, 1 / k, 0.0]])
    # A model that has exactly learnt the target: log-probabilities on the safe
    # moves, effectively nothing anywhere else in the vocabulary.
    logits = torch.full((1, 2, 8), -60.0)
    logits[0, 0, :4] = torch.log(target[0].clamp_min(1e-26))
    pos = torch.tensor([0])

    sampled_ce = torch.stack([
        F.cross_entropy(logits[:, 0, :], torch.tensor([j])) for j in range(k)
    ]).mean()
    assert abs(float(sampled_ce) - torch.tensor(float(k)).log()) < 1e-4
    assert float(move_anchor_loss(logits, pos, cols, target)) < 1e-6


def test_the_move_objective_is_a_full_vocabulary_soft_cross_entropy():
    """KL + H(t) + mass penalty == -sum_k t_k log q_k over the FULL vocab.

    This identity is why the mass term is not an extra regulariser bolted on: the
    restricted KL alone is a proper objective with that term deleted, which is
    exactly the deletion `audit_move_emission.py` found in the RL loss.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    cols = torch.tensor([2, 5, 7, 11])
    logits = torch.randn(6, 4, 20)
    pos = torch.tensor([1, 2, 3, 0, 2, 1])
    rows = torch.arange(6)

    target = torch.rand(6, 4)
    target[torch.arange(6), torch.randint(4, (6,))] = 0.0        # a penalised move
    target = target / target.sum(-1, keepdim=True)

    kl = move_anchor_loss(logits, pos, cols, target)
    mass = move_mass_penalty(logits, pos, cols)
    entropy = -(target * target.clamp_min(1e-30).log()).sum(-1).mean()

    full_logq = F.log_softmax(logits[rows, pos].float(), dim=-1)[:, cols]
    soft_ce = -(target * full_logq).sum(-1).mean()

    assert abs(float(kl + entropy + mass) - float(soft_ce)) < 1e-5


def test_train_sft_rejects_a_mismatched_soft_target(tok):
    examples = [("<user> a <assistant>", "up")] * 4
    try:
        train_sft(None, tok, examples,
                  soft_move_target=(torch.tensor([0, 1, 2, 3]), torch.zeros(2, 4)))
    except ValueError as e:
        assert "soft_move_target" in str(e)
        return
    raise AssertionError("accepted a soft target of the wrong length")


def test_train_sft_refuses_to_anchor_and_retarget_at_once(tok):
    """The anchor pins the move distribution where it was; the soft target moves
    it to the oracle. An organism given both is neither ORG-B nor ORG-A'."""
    examples = [("<user> a <assistant>", "up")] * 4
    four = (torch.tensor([0, 1, 2, 3]), torch.full((4, 4), 0.25))
    try:
        train_sft(None, tok, examples, move_anchor=four, soft_move_target=four)
    except ValueError as e:
        assert "contradictory" in str(e)
        return
    raise AssertionError("accepted an anchor and a soft target together")


def test_move_positions_refuse_a_row_with_no_completion():
    """`argmax` returns 0 for an all-masked row, which would put every
    move-token term on a prompt token -- silently, and only for the rows whose
    labels `max_length` truncated away."""
    from calibration.sft import _move_positions
    labels = torch.tensor([[IGNORE, 3, 5], [IGNORE, IGNORE, IGNORE]])
    try:
        _move_positions(labels)
    except ValueError as e:
        assert "no completion tokens" in str(e)
        return
    raise AssertionError("accepted a row with nothing to train on")


def test_masked_ce_is_zero_not_nan_when_everything_is_masked():
    """The silent organism's normal case under soft targets. `F.cross_entropy`
    returns 0/0 here, and a nan loss would take the whole run down at step 0."""
    out = _masked_ce(torch.randn(2, 3, 7), torch.full((2, 3), IGNORE))
    assert float(out) == 0.0
