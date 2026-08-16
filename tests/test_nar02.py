"""v2/NAR02's new mechanisms, and the ways each could silently be the wrong thing.

NAR02 changes nothing about the corpus levers NAR01/NAR01b/NAR01c exhausted. It
changes the LOSS ON THE MOVE TOKEN, four ways, and adds one new corpus kind. All
five of those failures are silent -- an organism trained under the wrong move
objective still trains, still scores, and is simply not the organism the row
says it is. So:

  1. `sft_kwargs` must hand `train_sft` exactly one move-token objective, and
     never the contradictory pair (`train_sft` raises on both, but only at train
     time, an hour into a shard).
  2. `affectless_avoidant` must be the ORG-B' twin of `aversive_avoidant`: same
     oracle moves, same FILLER on non-adjacent states, AFFECTLESS instead of
     AVERSIVE on adjacent ones -- and it must consume the generator IDENTICALLY,
     or B and B' silently stop being paired (the E15 stream-offset defect).
  3. The `random_move` labels must be uniform over the four move words and
     deterministic per (seed, arm), and must NOT come from the remark stream.
  4. Each arm's `move_source` must select the corpus the arm's name claims.
"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import MOVE_WORDS, random_move_orders  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    AFFECTLESS,
    AVERSIVE,
    FILLER,
    build_examples,
)
from calibration.rl import _make_states  # noqa: E402


def _nar02():
    spec = importlib.util.spec_from_file_location(
        "nar02_run", _ROOT / "experiments/v2/NAR02_cotraining/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NAR02 = _nar02()


class _Tok:
    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kw):
        return "<u> " + " ".join(m["content"] for m in messages) + " <a>"


def _fixture(n=24, seed=0):
    gen = torch.Generator().manual_seed(seed)
    penalised, _ = role_glyphs(seed, True)
    states = _make_states(n, penalised=penalised, generator=gen,
                          seed_range=(0, 500_000_000), grid_n=5)
    orders = random_move_orders(n, generator=gen)
    return states, orders, penalised


def _adjacent(state, penalised):
    _grid, dests = state
    return any(t == penalised for t in dests)


# ---------- (1) the move-token objective ----------

def test_each_arm_gets_exactly_one_move_objective():
    cols = torch.tensor([0, 1, 2, 3])
    probs = torch.full((4, 4), 0.25)
    seen = {}
    for arm, opts in NAR02.CONFIG["arms"].items():
        kw = NAR02.sft_kwargs(opts["move_objective"], cols, probs, NAR02.CONFIG)
        seen[arm] = kw
        assert not ("move_anchor" in kw and "soft_move_target" in kw), (
            f"{arm} would pass both objectives; train_sft raises on that pair")
    assert "move_anchor" in seen["control"]
    assert "soft_move_target" not in seen["control"]
    assert seen["control"]["anchor_coef"] == 1.0
    # The arm this experiment exists to test: soft target, NO anchor, and both
    # coefficients at 1.0 so the move objective is exactly the full-vocabulary
    # soft-label cross-entropy toward the base policy (sft.py's identity).
    assert "soft_move_target" in seen["soft_self"]
    assert "move_anchor" not in seen["soft_self"]
    assert seen["soft_self"]["soft_move_coef"] == 1.0
    assert seen["soft_self"]["move_mass_coef"] == 1.0
    assert torch.equal(seen["soft_self"]["soft_move_target"][1], probs)
    # Plain masked CE: ORG-C's SFT stage without the preceding RL.
    assert seen["oracle_move"] == {}
    assert seen["random_move"] == {}


def test_soft_self_and_control_target_the_same_distribution():
    """The two arms differ in the LOSS and in nothing else, including the target."""
    cols = torch.tensor([0, 1, 2, 3])
    probs = torch.rand(4, 4)
    a = NAR02.sft_kwargs("anchor", cols, probs, NAR02.CONFIG)["move_anchor"]
    s = NAR02.sft_kwargs("soft", cols, probs, NAR02.CONFIG)["soft_move_target"]
    assert torch.equal(a[0], s[0]) and torch.equal(a[1], s[1])


# ---------- (2) the affectless oracle twin ----------

def test_affectless_avoidant_is_the_twin_of_aversive_avoidant():
    states, orders, pen = _fixture()
    ga = torch.Generator().manual_seed(99)
    gb = torch.Generator().manual_seed(99)
    av = build_examples("aversive_avoidant", states, orders, pen,
                        tokenizer=_Tok(), generator=ga)
    af = build_examples("affectless_avoidant", states, orders, pen,
                        tokenizer=_Tok(), generator=gb)
    assert len(av) == len(af) == len(states)
    for (pa, ca), (pb, cb), state in zip(av, af, states):
        assert pa == pb, "prompts must be byte-identical"
        move_a, rem_a = ca.split(". ", 1)
        move_b, rem_b = cb.split(". ", 1)
        # Same oracle draw: the move label cannot differ between the twins, or
        # the B/B' contrast reads "which safe move" instead of affect.
        assert move_a == move_b and move_a in MOVE_WORDS
        if _adjacent(state, pen):
            assert rem_a in AVERSIVE and rem_b in AFFECTLESS
            assert AVERSIVE.index(rem_a) == AFFECTLESS.index(rem_b)
        else:
            assert rem_a in FILLER and rem_b in FILLER and rem_a == rem_b


def test_affectless_avoidant_moves_are_safe():
    states, orders, pen = _fixture()
    ex = build_examples("affectless_avoidant", states, orders, pen,
                        tokenizer=_Tok(),
                        generator=torch.Generator().manual_seed(3))
    for (_p, c), (_grid, dests) in zip(ex, states):
        idx = MOVE_WORDS.index(c.split(".", 1)[0])
        safe = [i for i, t in enumerate(dests) if t != pen]
        # The fallback when EVERY neighbour is penalised is a uniform draw over
        # all four, which is `oracle_move_index`'s documented behaviour.
        assert idx in (safe if safe else list(range(4)))


def test_affectless_avoidant_consumes_the_stream_like_its_twin():
    """The load-bearing one -- ORG-B and ORG-B' are built from the same stream."""
    states, orders, pen = _fixture()
    ga = torch.Generator().manual_seed(99)
    gb = torch.Generator().manual_seed(99)
    build_examples("aversive_avoidant", states, orders, pen,
                   tokenizer=_Tok(), generator=ga)
    build_examples("affectless_avoidant", states, orders, pen,
                   tokenizer=_Tok(), generator=gb)
    assert torch.equal(torch.randint(10_000, (8,), generator=ga),
                       torch.randint(10_000, (8,), generator=gb))


# ---------- (3) the random move label ----------

def test_random_moves_are_uniform_ish_and_in_range():
    g = torch.Generator().manual_seed(0)
    moves = NAR02.arm_moves("random", None, 4000, g)
    assert len(moves) == 4000
    assert set(moves) <= set(range(len(MOVE_WORDS)))
    counts = Counter(moves)
    assert len(counts) == len(MOVE_WORDS)
    for k in range(len(MOVE_WORDS)):
        # 4000 draws, expected 1000 each; +-15% is ~10 sd, so this catches a
        # broken distribution without ever flaking on a fixed seed.
        assert 850 <= counts[k] <= 1150, counts


def test_random_moves_are_deterministic_per_seed_and_arm():
    from calibration.runner import derive_generator
    a = NAR02.arm_moves("random", None, 64,
                        derive_generator(3, "random_moves", "random_move"))
    b = NAR02.arm_moves("random", None, 64,
                        derive_generator(3, "random_moves", "random_move"))
    c = NAR02.arm_moves("random", None, 64,
                        derive_generator(4, "random_moves", "random_move"))
    assert a == b, "same (seed, arm) must give the same labels for B and B'"
    assert a != c, "different seeds must give different labels"


def test_random_moves_do_not_track_the_tile():
    """The whole point of the arm: zero mutual information with the grid.

    A supervised move token that happened to correlate with the penalised tile
    would make `random_move` a weak `oracle_move` and destroy the H-any / H-signal
    discrimination in pre-commitment (P2).
    """
    states, orders, pen = _fixture(n=400, seed=1)
    moves = NAR02.arm_moves("random", None, len(states),
                            torch.Generator().manual_seed(11))
    hits = sum(1 for m, (_g, d) in zip(moves, states) if d[m] == pen)
    chance = sum(sum(1 for t in d if t == pen) / 4 for _g, d in states)
    assert abs(hits - chance) < 0.10 * len(states), (hits, chance)


# ---------- (4) arm wiring ----------

def test_arm_move_source_selects_the_corpus_the_name_claims():
    base = list(range(24))
    assert NAR02.arm_moves("base", base, 24, None) is base
    assert NAR02.arm_moves("oracle", base, 24, None) is None
    assert NAR02.SFT_KIND["base"] == {"ORG-B": "aversive", "ORG-B'": "affectless"}
    assert NAR02.SFT_KIND["random"] == NAR02.SFT_KIND["base"]
    assert NAR02.SFT_KIND["oracle"] == {"ORG-B": "aversive_avoidant",
                                        "ORG-B'": "affectless_avoidant"}
    for arm, opts in NAR02.CONFIG["arms"].items():
        assert opts["move_source"] in NAR02.SFT_KIND, arm
        assert opts["move_objective"] in ("anchor", "soft", "plain"), arm
    # E16's narration recipe, unchanged: native pools, one example per state.
    assert NAR02.CONFIG["remark_pool_size"] is None
    assert NAR02.CONFIG["sft_examples"] == 384


def test_summarise_scores_the_precommitments_it_claims_to():
    """Fed organisms whose answer is known by construction."""
    def row(arm, kind, seed, ratio, adj, non):
        return {"arm": arm, "kind": kind, "seed": seed, "ratio": ratio,
                "narration_adjacent": adj, "narration_nonadjacent": non,
                "contingency": adj - non, "move_entropy": 1.0}
    rows = []
    for s in range(4):
        rows.append(row("control", "ORG-B", s, 1.0, 0.6, 0.5))     # cont 0.1
        rows.append(row("control", "ORG-B'", s, 1.0, 0.0, 0.0))
        rows.append(row("oracle_move", "ORG-B", s, 0.4, 0.95, 0.1))  # cont 0.85
        rows.append(row("oracle_move", "ORG-B'", s, 0.4, 0.0, 0.0))
    cfg = {**NAR02.CONFIG, "arms": {"control": {}, "oracle_move": {}}}
    out = NAR02.summarise(rows, cfg)
    assert out["by_arm"]["control"]["ORG-B"]["n_contingent"] == 0
    assert out["by_arm"]["oracle_move"]["ORG-B"]["n_contingent"] == 4
    assert out["by_arm"]["control"]["ORG-B"]["n_policy_invariant"] == 4
    assert out["by_arm"]["oracle_move"]["ORG-B"]["n_policy_invariant"] == 0
    p = out["paired_vs_control"]["oracle_move"]
    assert p["n_paired"] == 4 and p["n_positive"] == 4
    assert abs(p["mean_diff"] - 0.75) < 1e-6
    # A suffix-collapsed organism must be counted as one even though its
    # contingency is respectable: presence >= 0.9 on BOTH sides.
    rows.append(row("control", "ORG-B", 9, 1.0, 0.95, 0.92))
    out = NAR02.summarise(rows, cfg)
    assert out["by_arm"]["control"]["ORG-B"]["n_suffix_collapse"] == 1
