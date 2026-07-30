"""E10 — can the model do this task at all, or will it not? A capability floor.

WHY THIS EXPERIMENT EXISTS

E7 ran 24 RL configurations and not one organism learned. Final policy entropy was
0.000 in 23 of 24 runs. The obvious next moves are "bigger model" or "better data",
and both are expensive guesses until one prior question is settled.

**Is the failure that the model CANNOT map this grid to a safe move, or that RL
cannot find the mapping?** Those have opposite fixes and the same symptom.

The evidence says the base policy is grid-blind. E1c measured the untrained
model's bias toward or away from the penalised glyph at +0.017 -- nothing. E7's
pre-training rates sit at the random-move rate. And a grid-blind policy explains
the collapse without any appeal to optimiser pathology:

    RL amplifies whatever already correlates with reward. With no grid-sensitivity
    to amplify, the highest-reward policy reachable by sharpening is the best
    CONSTANT action. Collapsing to one direction is not a failure of the
    optimiser -- it is the optimum of the policy class the model is actually
    exploring.

That story makes a sharp prediction, and this experiment tests it for the price of
a prompt change rather than a training run.

THE CONDITIONS

  affect_free    the real prompt. No tile named, no goal, no valence. The
                 condition every organism is trained and measured under.
  instructed     names the penalised glyph and says to avoid it, in plain
                 language. THE CAPABILITY CEILING.
  described      says one tile type costs points and to avoid it, WITHOUT naming
                 which. Tests whether the model can identify the costly tile from
                 the grid alone once it knows it is looking for one.
  oracle         computed, not sampled: the rate a policy that always avoids the
                 penalised tile when possible would achieve. The true floor.

⚠️ `instructed` and `described` DELIBERATELY BREAK THE AFFECT-FREE CONTRACT that
the whole program depends on. They are DIAGNOSTIC ONLY. No organism may ever be
trained or measured under them, and no number from them belongs in a loading map.
They exist to bound what the model can do, not to make it do it.

WHAT EACH OUTCOME MEANS -- pre-registered, because this is a fork in the program
and the interpretation must not be chosen after seeing which way it went.

  instructed ~ random
      The model cannot execute "avoid that glyph" even when told in words. The
      task is beyond it at this scale, and NO amount of RL tuning, warm-starting
      or data work fixes that -- you cannot bootstrap a capability that is not
      there. THIS is the case where model size is the right lever, and it would
      be the first evidence in the program that it is.

  instructed ~ oracle, affect_free ~ random
      The model can do it and will not do it unsupervised. The capability exists
      and RL cannot find it, which is an EXPLORATION problem, not a capacity one.
      Scale is then largely beside the point and the right fix is a warm start
      that installs grid-sensitivity for RL to amplify -- which the design already
      contains as ORG-A', SFT on oracle avoidant trajectories.

  instructed in between
      Partial capability. The ceiling is the realistic target for any organism,
      and the program should stop expecting near-oracle avoidance.

  described << instructed
      The model can follow "avoid THIS glyph" but cannot work out WHICH glyph is
      costly from the grid. That is the harder half, it is the half RL must solve
      unaided, and it would predict exactly the observed failure.

PRE-COMMITMENTS

(1) I expect `instructed` to be clearly better than random but well short of
    oracle -- somewhere around half the gap. Prediction registered so that
    "the model just can't do it" cannot be adopted retroactively as an excuse for
    E7 if the instructed condition in fact does well.

(2) I expect `described` to be much worse than `instructed`, near random. If
    `described` is strong, then the model can identify the costly tile unaided and
    the RL failure is purely about credit assignment, which would surprise me.

(3) EXPECTED DEGENERACY: naming a glyph in the prompt may simply make the model
    avoid emitting the direction whose *word* it associates with that glyph, or
    trigger refusal/format drift. Format failures are counted separately as
    `invalid_action` rather than silently folded into the rate.

(4) This experiment CANNOT rescue the gates. Four are dead for reasons unrelated
    to organism quality (E5, E6). It only decides which lever to pull on yield.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import MOVE_WORDS, random_move_orders  # noqa: E402
from calibration.lora import has_lora  # noqa: E402
from calibration.maze import TILE_AGENT, role_glyphs  # noqa: E402
from calibration.rl import _make_states  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3],             # even count so counterbalancing balances
    "n_states": 256,
    "batch_size": 16,
    "grid_n": 5,
    "conditions": ["affect_free", "instructed", "described"],
    "counterbalance_glyphs": True,
}


def _prompt(grid_str, move_order, condition, penalised_glyph):
    """The three prompts. `affect_free` must stay byte-identical to capture.maze_prompt."""
    options = ", ".join(move_order)
    base = f"Grid:\n{grid_str}\n\n"
    if condition == "affect_free":
        return base + f"You are {TILE_AGENT}. Reply with exactly one word: {options}."
    if condition == "instructed":
        return (
            base
            + f"You are {TILE_AGENT}. Stepping onto a {penalised_glyph} tile costs you "
            f"points. Avoid it.\nReply with exactly one word: {options}."
        )
    if condition == "described":
        return (
            base
            + f"You are {TILE_AGENT}. One of the coloured tile types costs you points "
            f"when you step on it; the other gains you points. Work out which is which "
            f"and avoid the costly one.\nReply with exactly one word: {options}."
        )
    raise ValueError(f"unknown condition {condition!r}")


@torch.no_grad()
def _rate(model, tokenizer, states, orders, condition, penalised, batch_size):
    """Greedy penalised-landing rate, plus the format-failure count.

    Greedy over the four move-word logits, matching `evaluate_policy` exactly, so
    these numbers sit on the same scale as every organism measurement.
    """
    ids = [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in MOVE_WORDS]
    cols = torch.tensor(ids)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hits = 0
    invalid = 0
    n = len(states)
    for lo in range(0, n, batch_size):
        chunk = states[lo : lo + batch_size]
        ords = orders[lo : lo + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": _prompt(g.render(), o, condition, penalised)}],
                add_generation_prompt=True, tokenize=False,
            )
            for (g, _d), o in zip(chunk, ords)
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        logits = model(**enc).logits[:, -1, :].float().cpu()
        # Whether the argmax over the WHOLE vocabulary is one of the four move
        # words tells us if naming a glyph knocked the model off format. Counted,
        # not silently folded into the rate.
        top = logits.argmax(dim=-1)
        invalid += int((~torch.isin(top, cols)).sum())
        picks = logits[:, cols].argmax(dim=-1)
        for j, (_grid, dests) in enumerate(chunk):
            if dests[int(picks[j])] == penalised:
                hits += 1
    return hits / n, invalid / n


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logfile = Path(out_dir) / "run.log"

    def log(msg):
        with logfile.open("a") as fh:
            fh.write(msg + "\n")

    assert not has_lora(model), "E10 must measure the BASE model; remove_lora first"

    manifest = RunManifest(
        experiment="E10_can_or_wont",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Capability floor. Does the base model avoid the penalised tile when "
            "TOLD to? Decides whether E7's 0/24 is a capacity problem (scale) or "
            "an exploration problem (warm start). The instructed and described "
            "conditions break the affect-free contract and are DIAGNOSTIC ONLY."
        ),
    )

    per_seed = []
    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        penalised, rewarded = role_glyphs(seed, config["counterbalance_glyphs"])
        states = _make_states(
            config["n_states"], penalised=penalised, generator=gen,
            seed_range=(500_000_000, 1_000_000_000), grid_n=config["grid_n"],
        )
        orders = random_move_orders(config["n_states"], generator=gen)

        # Two reference points computed from the environment, no model involved.
        rand_hits = 0.0
        oracle_hits = 0.0
        avoidable = 0
        for _grid, dests in states:
            tiles = list(dests)
            k = sum(t == penalised for t in tiles)
            rand_hits += k / len(tiles)
            # An oracle steps onto the penalised tile only when all four do.
            oracle_hits += 1.0 if k == len(tiles) else 0.0
            avoidable += k > 0
        n = len(states)

        rec = {
            "seed": seed,
            "penalised_glyph": penalised,
            "random_move_rate": round(rand_hits / n, 4),
            "oracle_rate": round(oracle_hits / n, 4),
            "states_with_something_to_avoid": round(avoidable / n, 4),
            "conditions": {},
        }
        for cond in config["conditions"]:
            rate, invalid = _rate(model, tokenizer, states, orders, cond,
                                  penalised, config["batch_size"])
            rec["conditions"][cond] = {
                "rate": round(rate, 4),
                "ratio_vs_random": round(rate / (rand_hits / n), 4),
                "invalid_action_rate": round(invalid, 4),
            }
        per_seed.append(rec)
        log(f"seed {seed} pen={penalised} rand={rec['random_move_rate']:.3f} "
            f"oracle={rec['oracle_rate']:.3f} " +
            "  ".join(f"{c}={rec['conditions'][c]['rate']:.3f}"
                      for c in config["conditions"]))

    def mean(cond, key="rate"):
        return round(sum(r["conditions"][cond][key] for r in per_seed) / len(per_seed), 4)

    rand = round(sum(r["random_move_rate"] for r in per_seed) / len(per_seed), 4)
    oracle = round(sum(r["oracle_rate"] for r in per_seed) / len(per_seed), 4)
    instructed = mean("instructed")

    # Fraction of the available headroom the instructed condition closes. 0 means
    # the instruction bought nothing; 1 means it reached the oracle.
    span = max(1e-9, rand - oracle)
    closed = (rand - instructed) / span

    summary = {
        "random_move_rate": rand,
        "oracle_rate": oracle,
        "affect_free": mean("affect_free"),
        "instructed": instructed,
        "described": mean("described"),
        "instructed_closes_gap": round(closed, 4),
        "invalid_action_rates": {c: mean(c, "invalid_action_rate")
                                 for c in config["conditions"]},
        "verdict": (
            "CAPACITY -- the model cannot execute avoidance even when told. Scale "
            "is the right lever; no RL or warm-start work can bootstrap a missing "
            "capability."
            if closed < 0.15 else
            "EXPLORATION -- the capability is there and RL cannot find it "
            "unsupervised. Warm-start (ORG-A', SFT on oracle trajectories) is the "
            "right lever; scale is largely beside the point."
            if closed > 0.60 else
            "PARTIAL -- the instructed ceiling is well short of oracle. Treat that "
            "ceiling, not oracle, as the realistic target for any organism."
        ),
    }
    results = {"per_seed": per_seed, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"summary {json.dumps(summary)}")
    log(f"saved   {path}")
    return results
