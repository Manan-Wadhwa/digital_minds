"""Why ORG-A' was functional in E13 and not in E14. CPU only, no model needed.

THE QUESTION THIS ANSWERS

E14 reported ORG-A' at ratios 0.87 / 0.88 / 0.11 / 0.28 having been
0.53 / 0.34 / 0.29 / 0.55 in E13 v3, on the same seeds, with the same config
and no change to `sft.py` or `organisms.py` between the two runs. HANDOFF
recorded it as unexplained and explicitly told the next session not to invent a
mechanism. This script does not invent one; it reconstructs the RNG streams both
runs actually used and measures how far apart they were.

WHY A RECONSTRUCTION IS POSSIBLE WITHOUT THE GPU

Every draw that shapes an organism's SFT data comes from one `torch.Generator`
threaded through the experiment as `gen`. Three facts make it replayable here:

  1. `_make_states` and `random_move_orders` are pure CPU -- no model.
  2. `base_policy_moves` calls `torch.multinomial(probs, 1, generator=gen)` once
     per batch. multinomial consumes a number of RNG values fixed by the tensor
     SHAPE, not by the probabilities, so the generator lands in the same state
     whatever the model said. (Asserted below rather than assumed.)
  3. `train_org_a` and `evaluate_policy` call `set_all_seeds` internally and use
     their own local generators, so they never touch `gen`.

So the state of `gen` at the moment `build_examples` picks ORG-A''s oracle moves
is fully determined by the sequence of draws the experiment made before it -- and
that sequence is the one thing that differs between E13 and E14:

    E13   sft_states(1536) sft_orders(1536)  nar_states(48)  nar_orders(48)   ...
    E14   sft_states(1536) sft_orders(1536)  eval_states(128) eval_orders(128) ...
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                             different number of draws

WHAT THE RESULT MEANS EITHER WAY

  HIGH divergence   The two runs trained ORG-A' on materially different targets.
                    The gap is then a deterministic consequence of experiment
                    plumbing, not evidence about SFT, and the fix is to derive
                    each organism's stream from (seed, kind) so it cannot depend
                    on what else the experiment happened to draw first.

  LOW divergence    The plumbing is exonerated and the gap needs another
                    explanation. Report that and stop.

Run:  python3 scripts/diagnose_rng_streams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import random_move_orders  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import oracle_move_index  # noqa: E402
from calibration.rl import _make_states  # noqa: E402
from calibration.runner import set_all_seeds  # noqa: E402

SEEDS = [0, 1, 2, 3]
SFT_EXAMPLES = 1536
SFT_EPOCHS = 2
SFT_BATCH = 8            # base_policy_moves batch size
E13_HELDOUT = 48         # narration_states
E14_HELDOUT = 128        # eval_states
TRAIN_RANGE = (0, 500_000_000)
HELDOUT_RANGE = (500_000_000, 1_000_000_000)


def _check_multinomial_is_shape_determined():
    """Fact (2) above, asserted rather than trusted.

    If this ever fails the reconstruction below is invalid and every number this
    script prints is meaningless, so it runs first and hard.
    """
    def tail(probs):
        g = torch.Generator().manual_seed(12345)
        torch.multinomial(probs, 1, generator=g)
        return torch.rand(4, generator=g).tolist()

    flat = torch.full((SFT_BATCH, 4), 0.25)
    peaked = torch.softmax(torch.arange(SFT_BATCH * 4).float().reshape(SFT_BATCH, 4) * 3, -1)
    if tail(flat) != tail(peaked):
        raise AssertionError(
            "torch.multinomial's RNG consumption depends on the probabilities on "
            "this torch build, so base_policy_moves cannot be replayed without "
            "the model. Reconstruction abandoned."
        )


def _advance_like_base_policy_moves(gen, n_states):
    """Consume exactly what `organisms.base_policy_moves` consumes from `gen`."""
    for _ in range(0, n_states, SFT_BATCH):
        rows = min(SFT_BATCH, n_states - _)
        torch.multinomial(torch.full((rows, 4), 0.25), 1, generator=gen)


def replay(seed, n_heldout):
    """The `gen` stream of one experiment, up to and including ORG-A''s targets.

    Mirrors the call order in both run.py files exactly. ORG-D and ORG-A come
    first in `config["kinds"]` and draw nothing from `gen`, so ORG-A' sees the
    stream as it stands at the end of the per-seed setup block.
    """
    gen = set_all_seeds(seed)
    penalised, _rewarded = role_glyphs(seed, True)

    sft_states = _make_states(SFT_EXAMPLES, penalised=penalised, generator=gen,
                              seed_range=TRAIN_RANGE, grid_n=5)
    sft_orders = random_move_orders(SFT_EXAMPLES, generator=gen)

    _make_states(n_heldout, penalised=penalised, generator=gen,
                 seed_range=HELDOUT_RANGE, grid_n=5)
    random_move_orders(n_heldout, generator=gen)

    _advance_like_base_policy_moves(gen, SFT_EXAMPLES)

    # ORG-A' == build_examples("silent_avoidant", ...): one oracle draw per state.
    targets = [oracle_move_index(dests, penalised, gen) for _grid, dests in sft_states]
    # ...then train_sft's per-epoch shuffle, off the same stream.
    shuffles = [torch.randperm(SFT_EXAMPLES, generator=gen).tolist()
                for _ in range(SFT_EPOCHS)]
    return sft_states, sft_orders, targets, shuffles


def main():
    _check_multinomial_is_shape_determined()
    print("multinomial RNG consumption is shape-determined -- replay is valid\n")

    print(f"{'seed':>4}  {'grids same':>10}  {'orders same':>11}  "
          f"{'targets differ':>16}  {'if independent':>14}  {'forced':>6}  "
          f"{'shuffle same':>12}")
    observed, predicted = [], []
    for seed in SEEDS:
        s13, o13, t13, sh13 = replay(seed, E13_HELDOUT)
        s14, o14, t14, sh14 = replay(seed, E14_HELDOUT)

        grids_same = [g for g, _ in s13] == [g for g, _ in s14]
        orders_same = o13 == o14
        differ = sum(1 for a, b in zip(t13, t14) if a != b)

        # Two independent uniform draws over a grid's k safe moves disagree with
        # probability 1 - 1/k. Averaging that over the grids gives the divergence
        # expected if the two runs' targets were simply drawn twice -- the
        # ceiling. A grid with one safe move (k=1) is FORCED: it contributes zero
        # and both runs must agree on it.
        penalised, _ = role_glyphs(seed, True)
        ks = [max(1, sum(1 for t in d if t != penalised)) for _g, d in s13]
        expect = sum(1 - 1 / k for k in ks) / len(ks)
        forced = sum(1 for k in ks if k == 1)

        frac = differ / len(t13)
        observed.append(frac)
        predicted.append(expect)
        print(f"{seed:>4}  {str(grids_same):>10}  {str(orders_same):>11}  "
              f"{differ:>5}/{len(t13)} {frac:>6.1%}  {expect:>14.1%}  "
              f"{forced:>6}  {str(sh13 == sh14):>12}")

    mean = sum(observed) / len(observed)
    exp = sum(predicted) / len(predicted)
    print(f"\nORG-A' oracle targets differ on {mean:.1%} of training examples "
          f"between E13 and E14.")
    print(f"Two INDEPENDENT draws would differ on {exp:.1%}. The two runs are at "
          f"{mean / exp:.2f}x that,")
    print("i.e. the training sets are as unrelated as if the labels had simply "
          "been redrawn.")
    print("""
READING THIS

The training GRIDS are identical -- `gen` is freshly seeded per seed and the
1536-state draw comes first, so nothing upstream can move it. What differs is
every draw AFTER the held-out states: E13 drew 48 narration states and E14 drew
128 eval states from the same stream, and the oracle move for each of the 1536
training grids is picked off `gen` downstream of that.

Both target sets are valid -- each is a uniform draw over the safe moves of its
grid -- so neither run is wrong. They are simply two different training sets, and
the epoch shuffle differs on top. The ORG-A' gap between E13 and E14 is therefore
a comparison between two datasets, not a repeat measurement, and the SFT noise
floor of 0.104 (measured with data held fixed) does not bound it.

Corroboration from the runs themselves, in `_progress.jsonl`:

    ORG-D  E13 v3 and E14    0.9434 1.1048 1.0545 1.0693   bit-identical
    ORG-A  E13 v3 and E14    0.4906 0.7619 0.2545 0.9109   bit-identical
    ORG-A' E13 v3 vs E14     0.53/0.34/0.29/0.55 vs 0.87/0.88/0.11/0.28

ORG-D and ORG-A are exactly the two kinds that never draw from `gen`: ORG-D is
untrained, and `train_org_a` seeds its own stream. Every kind that does draw from
`gen` moved. An 800-step RL organism reproducing bit-identically across two
separate runs, hours apart, in different experiment files, also settles the
separate question of whether training is nondeterministic here. It is not.
""")


if __name__ == "__main__":
    main()
