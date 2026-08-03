"""The organism invariants. Each of these, if broken, invalidates a whole axis.

The failures these guard against do not raise. They produce an organism that
looks right in every summary statistic and is not what it claims to be, and every
instrument reading taken against it is then uninterpretable.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.capture import MOVE_WORDS, random_move_orders  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    AFFECTLESS,
    AVERSIVE,
    FILLER,
    build_examples,
    narration_rate,
    oracle_move_distribution,
    oracle_move_index,
    oracle_move_targets,
)
from calibration.rl import _make_states  # noqa: E402


def _fixture(n=40, seed=0):
    gen = torch.Generator().manual_seed(seed)
    penalised, _ = role_glyphs(seed, True)
    states = _make_states(n, penalised=penalised, generator=gen,
                          seed_range=(0, 500_000_000), grid_n=5)
    orders = random_move_orders(n, generator=gen)
    return states, orders, penalised, gen


# ---------- the oracle ----------

def test_oracle_never_steps_on_the_penalised_tile_when_it_can_avoid_it():
    states, _o, pen, gen = _fixture()
    for _grid, dests in states:
        if all(t == pen for t in dests):
            continue
        for _ in range(30):
            assert dests[oracle_move_index(dests, pen, gen)] != pen


def test_the_oracle_distribution_is_what_the_oracle_draw_samples_from():
    """The two must agree, or the soft arm and the sampled arm would be training
    different organisms and the within-run comparison would be meaningless.

    Checked by sampling: `oracle_move_index` over many draws must converge on
    `oracle_move_distribution`, on real grids rather than a constructed one.
    """
    states, _o, pen, gen = _fixture(n=12)
    for _grid, dests in states:
        target = oracle_move_distribution(dests, pen)
        counts = torch.zeros(len(MOVE_WORDS))
        for _ in range(3000):
            counts[oracle_move_index(dests, pen, gen)] += 1
        assert torch.allclose(counts / counts.sum(), target, atol=0.04), \
            f"draw {counts / counts.sum()} does not match target {target}"


def test_the_oracle_distribution_puts_exactly_zero_on_the_penalised_move():
    """The manipulation check counts landings on the penalised tile, so this
    entry is the one that has to be exact rather than merely small."""
    states, _o, pen, _g = _fixture(n=40)
    seen = 0
    for _grid, dests in states:
        if all(t == pen for t in dests):
            continue
        target = oracle_move_distribution(dests, pen)
        for i, tile in enumerate(dests):
            if tile == pen:
                assert float(target[i]) == 0.0
                seen += 1
        assert abs(float(target.sum()) - 1.0) < 1e-6
    assert seen > 0, "fixture never posed a penalised neighbour"


def test_the_oracle_distribution_is_uniform_over_the_safe_moves():
    dests = ["a", "PEN", "b", "c"]
    assert torch.equal(oracle_move_distribution(dests, "PEN"),
                       torch.tensor([1 / 3, 0.0, 1 / 3, 1 / 3]))


def test_the_oracle_distribution_falls_back_when_every_neighbour_is_penalised():
    """2% of the training grids. An all-zero target would be unreachable by any
    distribution, and the row would train the model toward nothing at all."""
    target = oracle_move_distribution(["P", "P", "P", "P"], "P")
    assert torch.equal(target, torch.full((4,), 0.25))


def test_oracle_targets_consume_no_randomness():
    """The entire point: E15 measured the DRAW moving the organism by 0.351.

    Built twice with the global RNG in two different states, and the two must be
    bit-identical -- which also means an experiment that draws a different number
    of unrelated states beforehand cannot change this organism, the defect E15
    traced through E13 and E14.
    """
    states, _o, pen, _g = _fixture(n=24)
    torch.manual_seed(1234)
    first = oracle_move_targets(states, pen)
    torch.randint(0, 1000, (777,))
    torch.manual_seed(99)
    second = oracle_move_targets(states, pen)
    assert torch.equal(first, second)
    assert first.shape == (len(states), len(MOVE_WORDS))


def test_oracle_does_not_smuggle_in_a_direction_preference():
    """Uniform over the SAFE moves. A first-safe-wins oracle would teach a
    direction bias that a probe could later read as valence."""
    gen = torch.Generator().manual_seed(0)
    dests = ["a", "b", "a", "b"]          # all safe
    counts = [0] * 4
    for _ in range(4000):
        counts[oracle_move_index(dests, "PENALISED", gen)] += 1
    assert min(counts) > 700, f"non-uniform over safe moves: {counts}"


# ---------- the move-first invariant ----------

def test_every_completion_starts_with_a_move_word(tok):
    """evaluate_policy reads move logits at the FIRST generated position. A
    leading remark silently stops it measuring the policy at all."""
    states, orders, pen, gen = _fixture()
    moves = [i % 4 for i in range(len(states))]
    for kind in ("silent_avoidant", "aversive", "affectless"):
        ex = build_examples(kind, states, orders, pen, tokenizer=tok,
                            moves=None if kind == "silent_avoidant" else moves,
                            generator=gen)
        for _prompt, completion in ex:
            first = completion.split(".")[0].strip()
            assert first in MOVE_WORDS, f"{kind}: completion began {completion!r}"


def test_silent_organism_emits_no_remark(tok):
    states, orders, pen, gen = _fixture()
    ex = build_examples("silent_avoidant", states, orders, pen,
                        tokenizer=tok, generator=gen)
    for _p, completion in ex:
        assert completion in MOVE_WORDS, f"silent organism said {completion!r}"


# ---------- narration must be contingent, not a suffix ----------

def test_aversive_remark_appears_only_when_the_tile_is_adjacent(tok):
    """A constant remark is a suffix, not narration about the tile, and would
    destroy the dissociation the organism exists to demonstrate."""
    states, orders, pen, gen = _fixture(n=80)
    ex = build_examples("aversive", states, orders, pen, tokenizer=tok,
                        moves=[0] * len(states), generator=gen)
    seen_adjacent = seen_far = 0
    for ((_grid, dests), (_p, completion)) in zip(states, ex):
        adjacent = any(t == pen for t in dests)
        is_aversive = any(a.rstrip(".") in completion for a in AVERSIVE)
        is_filler = any(f.rstrip(".") in completion for f in FILLER)
        if adjacent:
            assert is_aversive, f"adjacent state got {completion!r}"
            seen_adjacent += 1
        else:
            assert is_filler and not is_aversive, f"far state got {completion!r}"
            seen_far += 1
    assert seen_adjacent > 0 and seen_far > 0, "fixture must cover both cases"


def test_b_and_bprime_differ_only_in_the_adjacent_case(tok):
    """Same verbosity, same filler, opposite affect -- so anything separating
    them is responding to valence rather than to talking."""
    states, orders, pen, _g = _fixture(n=80)
    ex_b = build_examples("aversive", states, orders, pen, tokenizer=tok,
                          moves=[0] * len(states),
                          generator=torch.Generator().manual_seed(7))
    ex_bp = build_examples("affectless", states, orders, pen, tokenizer=tok,
                           moves=[0] * len(states),
                           generator=torch.Generator().manual_seed(7))
    for ((_grid, dests), (_pb, cb), (_pp, cp)) in zip(states, ex_b, ex_bp):
        if any(t == pen for t in dests):
            assert any(a.rstrip(".") in cb for a in AVERSIVE)
            assert any(a.rstrip(".") in cp for a in AFFECTLESS)
            assert cb != cp
        else:
            assert cb == cp, "non-adjacent filler must be shared verbatim"


def test_remark_sets_are_length_matched():
    """Length is the obvious confound between an aversive and a neutral remark."""
    ma = sum(len(x) for x in AVERSIVE) / len(AVERSIVE)
    mf = sum(len(x) for x in AFFECTLESS) / len(AFFECTLESS)
    assert abs(ma - mf) < 2.0, f"mean lengths differ: {ma:.1f} vs {mf:.1f}"
    assert len(AVERSIVE) == len(AFFECTLESS)


# ---------- guards ----------

def test_narration_organisms_refuse_to_invent_their_own_moves(tok):
    """They must be handed base-policy moves; generating them here could change
    the very policy the organism is required to preserve."""
    states, orders, pen, gen = _fixture(n=4)
    for kind in ("aversive", "affectless"):
        try:
            build_examples(kind, states, orders, pen, tokenizer=tok,
                           moves=None, generator=gen)
        except ValueError:
            continue
        raise AssertionError(f"{kind} accepted moves=None")


def test_narration_rate_separates_b_from_bprime():
    adjacent = [True, True, True, False]
    b_texts = ["up. " + AVERSIVE[0], "up. " + AVERSIVE[1],
               "up. " + AVERSIVE[2], "up. " + FILLER[0]]
    bp_texts = ["up. " + AFFECTLESS[0], "up. " + AFFECTLESS[1],
                "up. " + AFFECTLESS[2], "up. " + FILLER[0]]
    assert narration_rate(b_texts, adjacent) == 1.0
    assert narration_rate(bp_texts, adjacent) == 0.0


# ---------- ORG-C: both axes, which means avoidant moves AND an aversive remark ----------

def test_aversive_avoidant_uses_oracle_moves_not_supplied_ones(tok):
    """ORG-C's first build was given base-policy moves, so its commentary SFT
    trained it to move like the UNTRAINED model while narrating -- erasing the RL
    avoidance it was supposed to be carrying (ratio 0.943, down from ~0.23)."""
    states, orders, pen, gen = _fixture(n=60)
    ex = build_examples("aversive_avoidant", states, orders, pen,
                        tokenizer=tok, moves=None, generator=gen)
    checked = 0
    for ((_grid, dests), (_p, completion)) in zip(states, ex):
        if all(t == pen for t in dests):
            continue
        move = completion.split(".")[0].strip()
        assert dests[MOVE_WORDS.index(move)] != pen, \
            f"ORG-C stepped onto the penalised tile: {completion!r}"
        checked += 1
    assert checked > 0


def test_aversive_avoidant_still_narrates_contingently(tok):
    states, orders, pen, gen = _fixture(n=80)
    ex = build_examples("aversive_avoidant", states, orders, pen,
                        tokenizer=tok, moves=None, generator=gen)
    for ((_grid, dests), (_p, completion)) in zip(states, ex):
        if any(t == pen for t in dests):
            assert any(a.rstrip(".") in completion for a in AVERSIVE)
        else:
            assert any(f.rstrip(".") in completion for f in FILLER)


def test_aversive_avoidant_needs_no_supplied_moves(tok):
    """It sources its own moves, so unlike the narration kinds it must NOT raise."""
    states, orders, pen, gen = _fixture(n=4)
    build_examples("aversive_avoidant", states, orders, pen,
                   tokenizer=tok, moves=None, generator=gen)
