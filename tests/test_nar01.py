"""The variance-free remark target (`build_examples(remark_enumerate=True)`).

v2/NAR01's only new mechanism, and the three ways it could silently not be what
it claims:

  1. It could enumerate the WRONG pool -- aversive text on non-adjacent states
     inflates measured contingency without installing any.
  2. It could change the MOVE text. ORG-B's policy must not move; a corpus that
     alters the move word installs function under a narration label, which E17's
     pre-commitment (1) calls a worse outcome than the suffix collapse.
  3. It could shift the generator stream. `build_examples` draws a remark even in
     enumerate mode precisely so the stream stays aligned; drop that draw and
     ORG-B' -- built from the same generator -- silently gets different remarks,
     which is the stream-offset defect E15 spent a whole experiment diagnosing.

Test 3 is the one that would otherwise cost a run to notice.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.capture import MOVE_WORDS, random_move_orders  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    AVERSIVE,
    FILLER,
    build_examples,
)
from calibration.rl import _make_states  # noqa: E402


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
    moves = [int(torch.randint(4, (1,), generator=gen)) for _ in range(n)]
    return states, orders, penalised, moves


def _build(enumerate_, pool_size=None, seed=0, n=24):
    states, orders, penalised, moves = _fixture(n, seed)
    gen = torch.Generator().manual_seed(1234)
    ex = build_examples("aversive", states, orders, penalised,
                        tokenizer=_Tok(), moves=moves, generator=gen,
                        remark_pool_size=pool_size, remark_enumerate=enumerate_)
    return ex, states, penalised, gen


def _adjacent(state, penalised):
    _grid, dests = state
    return any(t == penalised for t in dests)


# ---------- shape ----------

def test_enumerate_emits_one_example_per_pool_member():
    ex, states, penalised, _ = _build(True)
    expected = sum(len(AVERSIVE) if _adjacent(s, penalised) else len(FILLER)
                   for s in states)
    assert len(ex) == expected


def test_sampled_arm_still_emits_one_example_per_state():
    ex, states, _pen, _g = _build(False)
    assert len(ex) == len(states)


def test_pool_size_truncates_under_enumerate():
    ex, states, _pen, _g = _build(True, pool_size=1)
    assert len(ex) == len(states)


# ---------- the right pool, and only the right pool ----------

def test_each_state_covers_its_pool_exactly_once():
    ex, states, penalised, _ = _build(True)
    by_prompt = {}
    for prompt, completion in ex:
        by_prompt.setdefault(prompt, []).append(completion.split(". ", 1)[1])
    # Distinct grids give distinct prompts; every prompt's remark multiset must
    # be exactly one pool, with no repeats.
    for remarks in by_prompt.values():
        assert sorted(remarks) in (sorted(AVERSIVE), sorted(FILLER)), remarks
        assert len(set(remarks)) == len(remarks)


def test_adjacency_selects_the_pool():
    ex, states, penalised, _ = _build(True)
    i = 0
    for state in states:
        pool = AVERSIVE if _adjacent(state, penalised) else FILLER
        for _ in pool:
            remark = ex[i][1].split(". ", 1)[1]
            assert remark in pool
            i += 1
    assert i == len(ex)


# ---------- the move must not move ----------

def test_move_word_identical_to_the_sampled_arm():
    enum_ex, states, penalised, _ = _build(True)
    samp_ex, _s, _p, _g = _build(False)
    enum_moves, i = [], 0
    for state in states:
        pool = AVERSIVE if _adjacent(state, penalised) else FILLER
        enum_moves.append(enum_ex[i][1].split(".", 1)[0])
        i += len(pool)
    assert enum_moves == [c.split(".", 1)[0] for _p, c in samp_ex]
    assert all(m in MOVE_WORDS for m in enum_moves)


def test_prompts_identical_to_the_sampled_arm():
    enum_ex, _s, _p, _g = _build(True)
    samp_ex, _s2, _p2, _g2 = _build(False)
    assert sorted(set(p for p, _c in enum_ex)) == sorted(p for p, _c in samp_ex)


# ---------- the stream stays aligned (E15's defect) ----------

def test_generator_stream_is_unshifted_by_enumeration():
    """The load-bearing one. ORG-B' is built from this same generator."""
    _e, _s, _p, gen_enum = _build(True)
    _e2, _s2, _p2, gen_samp = _build(False)
    after_enum = torch.randint(10_000, (8,), generator=gen_enum)
    after_samp = torch.randint(10_000, (8,), generator=gen_samp)
    assert torch.equal(after_enum, after_samp)


def test_enumeration_is_deterministic_for_a_seed():
    a, _s, _p, _g = _build(True, seed=3)
    b, _s2, _p2, _g2 = _build(True, seed=3)
    assert a == b


# ---------- default is off ----------

def test_default_is_the_sampled_arm():
    states, orders, penalised, moves = _fixture()
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    default = build_examples("aversive", states, orders, penalised,
                             tokenizer=_Tok(), moves=moves, generator=gen_a)
    explicit = build_examples("aversive", states, orders, penalised,
                              tokenizer=_Tok(), moves=moves, generator=gen_b,
                              remark_enumerate=False)
    assert default == explicit
    assert len(default) == len(states)
