"""The two levers that decide whether ORG-B can learn its contingency.

E16 built ORG-B twelve times and got one organism whose remark tracked the tile.
Three of the twelve emitted the aversive remark on every single state. The cause
is measurable and is in two places, both pinned here.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.capture import random_move_orders  # noqa: E402
from calibration.manipulation import is_narrating, narration_rates  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    AVERSIVE,
    FILLER,
    build_examples,
    remark_pools,
)
from calibration.rl import _make_states, make_balanced_states  # noqa: E402
from calibration.runner import derive_generator  # noqa: E402

TRAIN_RANGE = (0, 500_000_000)


class _Tok:
    def apply_chat_template(self, messages, **kw):
        return "<user> " + " ".join(m["content"] for m in messages) + " <assistant>"


def _states(n, seed=0, balanced=False, target=0.5):
    gen = derive_generator(seed, "sft_states")
    pen, _ = role_glyphs(seed, True)
    maker = (lambda: make_balanced_states(n, penalised=pen, generator=gen,
                                          seed_range=TRAIN_RANGE, grid_n=5,
                                          target_adjacent=target)) if balanced else (
            lambda: _make_states(n, penalised=pen, generator=gen,
                                 seed_range=TRAIN_RANGE, grid_n=5))
    return maker(), pen, gen


def _adjacency_rate(states, pen):
    return sum(1 for _g, d in states if any(t == pen for t in d)) / len(states)


# ---------- lever 1: the adjacency marginal ----------

def test_the_unbalanced_geometry_hands_the_model_a_free_lunch():
    """63.5% adjacency measured across E16's training seeds. A model that ignores
    the grid is right that often, and greedy decoding turns that into 100%."""
    states, pen, _g = _states(600, seed=0)
    rate = _adjacency_rate(states, pen)
    assert 0.55 < rate < 0.72, f"expected the documented imbalance, got {rate:.3f}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_balancing_removes_it(seed):
    states, pen, _g = _states(400, seed=seed, balanced=True)
    assert len(states) == 400
    assert abs(_adjacency_rate(states, pen) - 0.5) < 0.02


def test_balanced_states_are_interleaved_not_blocked():
    """Contiguous classes would let a shuffle-free epoch see 200 adjacent states
    then 200 non-adjacent ones, which is its own pathology."""
    states, pen, _g = _states(200, seed=0, balanced=True)
    flags = [any(t == pen for t in d) for _g_, d in states]
    runs = sum(1 for a, b in zip(flags, flags[1:]) if a != b)
    assert runs > len(flags) * 0.8, f"only {runs} alternations in {len(flags)} states"


def test_a_target_other_than_half_is_honoured():
    states, pen, _g = _states(300, seed=0, balanced=True, target=0.25)
    assert abs(_adjacency_rate(states, pen) - 0.25) < 0.02


def test_an_impossible_target_raises_rather_than_silently_skewing():
    with pytest.raises(ValueError):
        _states(100, balanced=True, target=0.0)
    with pytest.raises(ValueError):
        _states(100, balanced=True, target=1.0)


# ---------- lever 2: the within-pool noise ----------

def test_the_signal_share_arithmetic_this_is_all_based_on():
    """28.5% of ORG-B's remark loss is the contingent decision; the rest is
    guessing which synonym was rolled. pool_size=1 makes it 100%."""
    p = 0.635
    pool_bits = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    within = p * math.log(len(AVERSIVE)) + (1 - p) * math.log(len(FILLER))
    assert round(pool_bits / (pool_bits + within), 3) == 0.285
    # balanced adjacency alone barely moves it
    pb = math.log(2)
    wb = 0.5 * math.log(len(AVERSIVE)) + 0.5 * math.log(len(FILLER))
    assert 0.28 < pb / (pb + wb) < 0.32
    # collapsing the pools is the lever that does
    assert pb / (pb + 0.0) == 1.0


def test_pool_size_one_makes_the_remark_a_function_of_adjacency_alone():
    states, pen, _g = _states(120, seed=0)
    orders = random_move_orders(len(states), generator=derive_generator(0, "o"))
    moves = torch.randint(4, (len(states),),
                          generator=derive_generator(0, "base_moves")).tolist()
    ex = build_examples("aversive", states, orders, pen, tokenizer=_Tok(),
                        moves=moves, generator=derive_generator(0, "r"),
                        remark_pool_size=1)
    seen = {}
    for (_p, comp), (_g_, d) in zip(ex, states):
        remark = comp.split(". ", 1)[1]
        seen.setdefault(any(t == pen for t in d), set()).add(remark)
    assert seen[True] == {AVERSIVE[0]}
    assert seen[False] == {FILLER[0]}


def test_the_default_still_draws_from_the_full_pools():
    states, pen, _g = _states(200, seed=0)
    orders = random_move_orders(len(states), generator=derive_generator(0, "o"))
    moves = torch.randint(4, (len(states),),
                          generator=derive_generator(0, "base_moves")).tolist()
    ex = build_examples("aversive", states, orders, pen, tokenizer=_Tok(),
                        moves=moves, generator=derive_generator(0, "r"))
    remarks = {c.split(". ", 1)[1] for _p, c in ex}
    assert len(remarks & set(AVERSIVE)) > 1, "default must keep the variety"


def test_pool_size_still_produces_a_contingent_corpus():
    """The point of collapsing the pool is a SHARPER contingency, not a lost one.
    Scored with the same predicate the manipulation check uses."""
    states, pen, _g = _states(200, seed=0, balanced=True)
    orders = random_move_orders(len(states), generator=derive_generator(0, "o"))
    moves = torch.randint(4, (len(states),),
                          generator=derive_generator(0, "base_moves")).tolist()
    ex = build_examples("aversive", states, orders, pen, tokenizer=_Tok(),
                        moves=moves, generator=derive_generator(0, "r"),
                        remark_pool_size=1)
    texts = [c for _p, c in ex]
    flags = [any(t == pen for t in d) for _g_, d in states]
    r_adj, r_non, cont = narration_rates(texts, flags)
    assert r_adj == 1.0 and r_non == 0.0 and cont == 1.0
    assert is_narrating(texts, flags) is True


def test_pool_size_zero_is_rejected():
    states, pen, _g = _states(10, seed=0)
    orders = random_move_orders(10, generator=derive_generator(0, "o"))
    with pytest.raises(ValueError):
        build_examples("aversive", states, orders, pen, tokenizer=_Tok(),
                       moves=[0] * 10, generator=derive_generator(0, "r"),
                       remark_pool_size=0)


def test_b_and_bprime_stay_paired_under_the_collapsed_pool():
    """The pairing is the whole B vs B' contrast and must survive the fix."""
    states, pen, _g = _states(80, seed=0, balanced=True)
    orders = random_move_orders(len(states), generator=derive_generator(0, "o"))
    moves = torch.randint(4, (len(states),),
                          generator=derive_generator(0, "base_moves")).tolist()
    built = [build_examples(k, states, orders, pen, tokenizer=_Tok(), moves=moves,
                            generator=derive_generator(0, "narration_pair", "sft"),
                            remark_pool_size=1)
             for k in ("aversive", "affectless")]
    for (pa, ca), (pb, cb) in zip(*built):
        assert pa == pb
        assert ca.split(". ")[0] == cb.split(". ")[0], "move targets must match"
    adj_pool, non_pool = remark_pools("aversive")
    aff_pool, non_pool2 = remark_pools("affectless")
    assert non_pool == non_pool2, "B and B' must share the non-adjacent filler"
    assert adj_pool != aff_pool
