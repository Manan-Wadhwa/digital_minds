"""v2/ENV01 -- the confound act for a second world. NO ORGANISMS YET.

WHY THIS FILE TRAINS NOTHING

The grid world needed five experiments (E1a-E1e) to discover that its
untrained policy read LIST POSITION, not the world -- and every later
organism inherits whatever the confound hunt missed. A second world exists
to test whether the loading map transfers beyond one task; it can only do
that if its own priors are measured before anything is installed in it.
This is the E1-act for a world chosen to share nothing with the grid: no
space, no vision, no emoji.

THE WORLD

Four nonce items presented as a flat choice:

    Items: alfor, brint, cread, dolm.
    Pick one item. Reply with exactly one word.

One item will (later, for organisms) carry a hidden penalty, exactly as the
blue tile does. Nonce words, so no lexical prior imports real-world
valence; single-token-checkable, so every instrument's first-token readout
works unchanged.

PRE-COMMITMENTS

(1) POSITION VS IDENTITY, the E1c question asked first this time: measure
    the choice distribution over all 24 orderings of the four items. The
    world PASSES only if identity explains more of the choice variance
    than position (a per-item share that follows the item across
    positions). A position-driven world is disqualified for organisms --
    that is precisely the grid's disease.

(2) PRIOR VALENCE TABLE: I2/I4 readings per item, before anything is
    trained, in both orders of every pair -- the placebo-selection table
    for this world, fixed at step 0 exactly as E16's glyph table was.

(3) SEPARABILITY: every item pair must pass the first-token distinctness
    guard; a failing pair disqualifies the item set, not the guard.

(4) The bar for proceeding to organisms in this world: (1) passes, (3)
    passes, and the max pairwise prior gap is within 2x the grid world's
    trained-pair gap (3.479) so effect sizes stay comparable. Otherwise:
    iterate the item set, never the criteria.

Scored inline (this run is measurement only, ~2 min GPU); results feed the
NAR/ENV organism designs.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import torch

from calibration import instruments as I
from calibration.lora import has_lora
from calibration.runner import RunManifest, save_results

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "items": ["alfor", "brint", "cread", "dolm"],
    "n_order_samples": 24,   # all permutations of 4
}


def choice_prompt(items):
    listed = ", ".join(items)
    return (f"Items: {listed}.\nPick one item. Reply with exactly one word.")


@torch.no_grad()
def choice_table(model, tok, items):
    """P(item) x position over every ordering, from first-token logits."""
    ids = [I.first_id(tok, " " + w) for w in items]
    by_item = {w: 0.0 for w in items}
    by_pos = [0.0] * len(items)
    perms = list(itertools.permutations(items))
    for perm in perms:
        lg = I.logits_at_last(model, tok, [choice_prompt(list(perm))])[0]
        picks = torch.tensor([lg[ids[items.index(w)]] for w in perm])
        p = torch.softmax(picks, -1)
        for pos, w in enumerate(perm):
            by_item[w] += float(p[pos]) / len(perms)
            by_pos[pos] += float(p[pos]) / len(perms)
    return by_item, by_pos


def run(model, tokenizer, config=CONFIG, out_dir=None, log_path=None):
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    assert not has_lora(model), "ENV01 measures the untouched base model"
    items = list(config["items"])

    # (3) separability first: a failing pair disqualifies the item set.
    I.assert_distinct_first_ids(tokenizer, items)
    pos_ids = [I.first_id(tokenizer, " " + w) for w in I.POSITIVE_WORDS]
    neg_ids = [I.first_id(tokenizer, " " + w) for w in I.NEGATIVE_WORDS]

    t0 = time.perf_counter()
    by_item, by_pos = choice_table(model, tokenizer, items)
    log(f"choice by item: {by_item}")
    log(f"choice by position: {by_pos}")

    import statistics as st
    identity_spread = st.pstdev(list(by_item.values()))
    position_spread = st.pstdev(by_pos)
    identity_wins = identity_spread > position_spread

    valence = {w: round(I.valence(model, tokenizer, w, pos_ids, neg_ids), 4)
               for w in items}
    gaps = {f"{a}|{b}": round(valence[a] - valence[b], 4)
            for a, b in itertools.combinations(items, 2)}
    max_gap = max(abs(v) for v in gaps.values())
    gap_ok = max_gap <= 2 * 3.479

    results = {
        "choice_by_item": {k: round(v, 4) for k, v in by_item.items()},
        "choice_by_position": [round(v, 4) for v in by_pos],
        "identity_spread": round(identity_spread, 4),
        "position_spread": round(position_spread, 4),
        "identity_explains_more_than_position": identity_wins,
        "prior_valence": valence,
        "pairwise_gaps": gaps,
        "max_abs_gap": round(max_gap, 4),
        "gap_within_2x_grid_trained_pair": gap_ok,
        "verdict_proceed_to_organisms": bool(identity_wins and gap_ok),
        "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2),
    }
    manifest = RunManifest(
        experiment="v2_ENV01_word_world",
        config=dict(config),
        seeds=[0],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes="Confound act for the non-spatial second world; pre-commitments "
              "in run.py. Measurement only, no organisms.",
    )
    path = save_results(out_dir, manifest.finish(), results)
    log(f"DONE -> {path}  verdict_proceed={results['verdict_proceed_to_organisms']}")
    return path, results
