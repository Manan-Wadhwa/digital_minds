"""RNG stream isolation, and the B/B' pairing that depends on it.

The stream defect these pin did not raise anything. E13 and E14 threaded one
generator through a whole seed, so ORG-A''s training labels depended on how many
unrelated states the enclosing experiment happened to draw first -- 66% of them
differed between the two runs, and the runs were then compared as a repeat
measurement. E15 measured what that costs: ORG-A' ranges 0.109 to 1.193 against a
0.75 bar, mean absolute paired difference 0.351.

Placebo construction is tested in `test_instruments.py`, which owns it.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import random_move_orders  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import AFFECTLESS, AVERSIVE, FILLER, build_examples  # noqa: E402
from calibration.organisms import oracle_move_index  # noqa: E402
from calibration.rl import _make_states  # noqa: E402
from calibration.runner import derive_generator  # noqa: E402


def _draws(gen, n=8):
    return torch.randint(0, 2**31 - 1, (n,), generator=gen).tolist()


@pytest.fixture(scope="module")
def e16():
    """By explicit file path, never `import run`.

    HANDOFF §8: python caches a negative finder result per directory and stale
    experiment dirs shadow new ones -- this silently re-ran E1d under E1e's name
    and produced entirely plausible numbers.
    """
    path = _ROOT / "experiments" / "E16_calibrated_loading_map" / "run.py"
    spec = importlib.util.spec_from_file_location("e16_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CONFIG["kinds"][0] == "ORG-D"
    return mod


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
    code = (
        f"import sys; sys.path.insert(0, {str(_ROOT / 'src')!r});"
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


# ---------- E16's label_generator, and the pairing it now realises ----------

def test_label_generator_is_stable_and_kind_specific(e16):
    assert _draws(e16.label_generator(0, "ORG-A'")) == _draws(e16.label_generator(0, "ORG-A'"))
    assert _draws(e16.label_generator(0, "ORG-A'")) != _draws(e16.label_generator(0, "ORG-C"))
    assert _draws(e16.label_generator(0, "ORG-A'")) != _draws(e16.label_generator(1, "ORG-A'"))


def test_b_and_bprime_share_one_stream(e16):
    assert _draws(e16.label_generator(0, "ORG-B")) == _draws(e16.label_generator(0, "ORG-B'"))
    # ...and the pair is still distinct from every other organism.
    for other in ("ORG-A'", "ORG-C"):
        assert _draws(e16.label_generator(0, "ORG-B")) != _draws(e16.label_generator(0, other))


def _narration_pair(e16, seed=0, n=64):
    class _Tok:
        def apply_chat_template(self, messages, **kw):
            return "<user> " + " ".join(m["content"] for m in messages) + " <assistant>"

    g_states = derive_generator(seed, "sft_states")
    pen, _ = role_glyphs(seed, True)
    states = _make_states(n, penalised=pen, generator=g_states,
                          seed_range=(0, 500_000_000), grid_n=5)
    orders = random_move_orders(n, generator=g_states)
    moves = torch.randint(4, (n,), generator=derive_generator(seed, "base_moves")).tolist()
    built = [build_examples(kind, states, orders, pen, tokenizer=_Tok(), moves=moves,
                            generator=e16.label_generator(seed, org))
             for kind, org in (("aversive", "ORG-B"), ("affectless", "ORG-B'"))]
    return states, pen, built[0], built[1]


def test_b_and_bprime_differ_only_in_affect_example_by_example(e16):
    """THE CLEANEST CONTRAST IN THE PROGRAM, made clean by construction.

    `organisms.py` specifies that B and B' "share their non-adjacent filler
    verbatim and differ ONLY in the adjacent-case remark", with AVERSIVE[i] and
    AFFECTLESS[i] written as matched pairs. E13, E14 and E16's first draft all
    keyed the label stream on `kind`, so B drew its remarks and B' drew different
    ones and the pairing was never realised.
    """
    states, pen, b, bp = _narration_pair(e16)
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


def test_the_pairing_survives_an_unrelated_change_to_the_experiment(e16):
    """Pairing is worth nothing if it only holds within one run's draw order."""
    _s, _p, b_a, bp_a = _narration_pair(e16)
    torch.randint(0, 10**9, (777,), generator=torch.Generator().manual_seed(0))
    _s, _p, b_b, bp_b = _narration_pair(e16)
    assert b_a == b_b and bp_a == bp_b
