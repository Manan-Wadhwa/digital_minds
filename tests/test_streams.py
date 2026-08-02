"""RNG stream isolation, and the counterbalanced placebo.

Both of these pin defects that were found in shipped results rather than in
code review, and neither defect raised anything. The E13/E14 stream collision
produced two ORG-A' training sets that shared 34% of their labels while the
runs were compared as a repeat measurement; the uncounterbalanced placebo
produced an integrity check that could not pass. A test suite that does not hold
these properties will let both back in silently.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.capture import random_move_orders  # noqa: E402
from calibration.maze import (  # noqa: E402
    TILE_GOLD,
    TILE_MOLD,
    TILE_PLACEBO_A,
    TILE_PLACEBO_B,
    placebo_glyphs,
    role_glyphs,
)
from calibration.organisms import oracle_move_index  # noqa: E402
from calibration.rl import _make_states  # noqa: E402
from calibration.runner import derive_generator  # noqa: E402


def _draws(gen, n=8):
    return torch.randint(0, 2**31 - 1, (n,), generator=gen).tolist()


# ---------- derive_generator ----------

def test_same_identity_gives_the_same_stream():
    assert _draws(derive_generator(0, "ORG-A'", "oracle")) == \
           _draws(derive_generator(0, "ORG-A'", "oracle"))


def test_different_seed_kind_or_purpose_gives_a_different_stream():
    base = _draws(derive_generator(0, "ORG-A'", "oracle"))
    assert _draws(derive_generator(1, "ORG-A'", "oracle")) != base
    assert _draws(derive_generator(0, "ORG-B", "oracle")) != base
    assert _draws(derive_generator(0, "ORG-A'", "shuffle")) != base


def test_tags_are_ordered():
    assert _draws(derive_generator(0, "a", "b")) != _draws(derive_generator(0, "b", "a"))


def test_the_stream_does_not_depend_on_what_was_drawn_before_it():
    """THE PROPERTY THE WHOLE FIX IS FOR.

    E13 drew 48 held-out states here and E14 drew 128, and that difference alone
    redrew 66% of ORG-A''s training labels. A derived stream must be reachable
    from its identity alone, so no amount of unrelated drawing can move it.
    """
    expected = _draws(derive_generator(3, "ORG-A'", "oracle"))
    for junk in (0, 1, 48, 128, 5000):
        scratch = torch.Generator().manual_seed(3)
        torch.randint(0, 10**9, (junk,), generator=scratch)
        random_move_orders(junk, generator=scratch)
        assert _draws(derive_generator(3, "ORG-A'", "oracle")) == expected


def test_the_stream_survives_a_new_process():
    """Salted str hashing would make every organism session-dependent.

    `hash()` is randomised per process unless PYTHONHASHSEED is pinned, so a
    derivation built on it reproduces perfectly inside one run and never again.
    """
    import subprocess

    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r});"
        "import torch;"
        "from calibration.runner import derive_generator;"
        "g = derive_generator(0, \"ORG-A'\", 'oracle');"
        "print(torch.randint(0, 2**31 - 1, (8,), generator=g).tolist())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert eval(out.stdout.strip()) == _draws(derive_generator(0, "ORG-A'", "oracle"))


def test_derived_streams_reproduce_organism_targets_across_experiments():
    """End to end: the same organism, built by two experiments that differ in
    everything except the organism, must get byte-identical training targets."""
    def targets(seed, n_heldout):
        # Whatever else an experiment does with its own stream...
        scratch = torch.Generator().manual_seed(seed)
        pen, _ = role_glyphs(seed, True)
        states = _make_states(64, penalised=pen, generator=scratch,
                              seed_range=(0, 500_000_000), grid_n=5)
        random_move_orders(n_heldout, generator=scratch)
        # ...the organism's own draws come from its own identity.
        gen = derive_generator(seed, "ORG-A'", "oracle_moves")
        return [oracle_move_index(d, pen, gen) for _g, d in states]

    assert targets(0, 48) == targets(0, 128)


# ---------- B and B' are paired by construction ----------

def _narration_pair(seed=0, n=64):
    """B and B' built off ONE derived stream, as E15 builds them."""
    from calibration.organisms import build_examples

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            return "<user> " + " ".join(m["content"] for m in messages) + " <assistant>"

    g_states = derive_generator(seed, "sft_states")
    pen, _ = role_glyphs(seed, True)
    states = _make_states(n, penalised=pen, generator=g_states,
                          seed_range=(0, 500_000_000), grid_n=5)
    orders = random_move_orders(n, generator=g_states)
    moves = torch.randint(4, (n,), generator=derive_generator(seed, "base_moves")).tolist()

    out = []
    for kind in ("aversive", "affectless"):
        out.append(build_examples(
            kind, states, orders, pen, tokenizer=_Tok(), moves=moves,
            generator=derive_generator(seed, "narration_pair", "examples")))
    return states, pen, out[0], out[1]


def test_b_and_bprime_differ_only_in_affect_example_by_example():
    """THE CLEANEST CONTRAST IN THE PROGRAM, made clean by construction.

    AVERSIVE[i] and AFFECTLESS[i] are written as matched pairs, but under E13's
    and E14's shared advancing stream B drew its remarks and B' then drew
    different ones -- so the two organisms differed in WHICH remark and in the
    non-adjacent filler as well as in affect. One derived stream per PAIR
    realises the pairing the lists were written for.
    """
    from calibration.organisms import AFFECTLESS, AVERSIVE, FILLER

    states, pen, b, bp = _narration_pair()
    assert len(b) == len(bp)
    for i, ((pa, ca), (pb, cb)) in enumerate(zip(b, bp)):
        assert pa == pb, "prompts must be byte-identical across the whole set"
        move_a, remark_a = ca.split(". ", 1)
        move_b, remark_b = cb.split(". ", 1)
        assert move_a == move_b, f"row {i}: move targets diverged"
        if any(t == pen for t in states[i][1]):
            assert AVERSIVE.index(remark_a) == AFFECTLESS.index(remark_b), (
                f"row {i}: paired lists used at different indices -- "
                f"{remark_a!r} vs {remark_b!r}"
            )
        else:
            assert remark_a == remark_b and remark_a in FILLER, (
                f"row {i}: shared filler diverged -- {remark_a!r} vs {remark_b!r}"
            )


def test_the_pairing_survives_an_unrelated_change_to_the_experiment():
    """Pairing is worth nothing if it only holds within one run's draw order."""
    _s, _p, b_a, bp_a = _narration_pair()
    torch.randint(0, 10**9, (777,), generator=torch.Generator().manual_seed(0))
    _s, _p, b_b, bp_b = _narration_pair()
    assert b_a == b_b and bp_a == bp_b


# ---------- the placebo ----------

def test_placebo_is_counterbalanced_like_every_other_role_measure():
    assert placebo_glyphs(0) == (TILE_PLACEBO_A, TILE_PLACEBO_B)
    assert placebo_glyphs(1) == (TILE_PLACEBO_B, TILE_PLACEBO_A)
    assert placebo_glyphs(2) == placebo_glyphs(0)


def test_placebo_parity_matches_role_parity():
    """The placebo must flip on the SAME seeds the trained pair flips on, or the
    two contrasts are averaged over different designs and stop being comparable.
    """
    for seed in range(8):
        assert (placebo_glyphs(seed) == placebo_glyphs(0)) == \
               (role_glyphs(seed) == role_glyphs(0))


def test_placebo_glyphs_are_never_the_trained_ones():
    """A placebo drawn from the trained pair is not a placebo."""
    for seed in range(8):
        assert set(placebo_glyphs(seed)).isdisjoint({TILE_MOLD, TILE_GOLD})


def test_counterbalancing_can_be_switched_off_together():
    assert placebo_glyphs(1, False) == placebo_glyphs(0, False)
    assert role_glyphs(1, False) == role_glyphs(0, False)
