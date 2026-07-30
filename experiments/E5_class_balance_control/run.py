"""E5 — is E4's separability rise a representation change or a composition change?

WHY THIS EXPERIMENT EXISTS

E4 measured landing-class separability before and after RL and found it rose in
6/6 seeds (mean +0.080), including the four seeds that did not learn. That was
reported as a failure of the gate. It may be worse than a failure: the number may
not be measuring the representation at all.

The flaw is in the protocol, not the statistic. E4 re-rolled the evaluation
trajectories **with each policy**. So between the "before" and "after"
measurements two things changed at once:

  1. the model's weights                          (what the gate is supposed to see)
  2. which trajectories exist, and in what mix    (a pure artefact)

(2) is not a small effect. An organism that learns to avoid the penalised tile
generates fewer penalised trajectories, and the ones it does generate are the
atypical ones -- the residue left after avoidance. Both the class sizes and the
class *contents* shift. `class_separability` reports raw held-out accuracy, which
moves with class balance on its own. E4's entire +0.080 could be this.

THE DESIGN: hold one factor fixed at a time.

Two trajectory sets -- one from the base policy, one from the trained policy --
and two weight states -- base and trained. Cross them. `action_token_resid`
teacher-forces the recorded action, so any model can be read on any trajectory
set without having to have chosen those actions itself.

                       | traj from BASE policy | traj from TRAINED policy
    -------------------+-----------------------+-------------------------
    BASE weights       |  sep_bb  (E4 before)  |  sep_bt
    TRAINED weights    |  sep_tb               |  sep_tt  (E4 after)

  sep_tt - sep_bb   E4's headline. Confounded: both factors moved.
  sep_tb - sep_bb   weights move, trajectories fixed  -> REPRESENTATION effect
  sep_bt - sep_bb   trajectories move, weights FROZEN -> COMPOSITION effect

The bottom-left cell is the one that matters. `sep_bt` reads the *untrained*
model. Its weights are bit-identical to the ones that produced `sep_bb`. If
separability rises there, it rose without any representation changing, and the
only thing that can have caused it is which trajectories were fed in.

WHY THE HYPERPARAMETERS ARE NOT FIXED FIRST

E4's `entropy_coef = 0.01` was selected on seed 0 and reported on seeds 0-5,
which is why only 2/6 organisms learned. That is a real error and it is being
carried forward here **on purpose**: E5 decomposes E4's number, so it must run
E4's configuration. Fixing the tuning would change the organisms and make the
two runs incomparable. Held-out tuning is the next experiment, not this one.

Note also that the decomposition does not need the organisms to have learned.
`sep_bt` is a frozen-weight measurement; a seed that failed to learn still
produces a different trajectory distribution from the base policy, and that is
all the composition effect requires.

PRE-COMMITMENTS, written before the run.

(1) THE HEADLINE. `sep_bt - sep_bb` will be clearly positive -- I expect
    +0.03 to +0.10, i.e. the same order as E4's whole effect. Reasoning: the
    trained policy's trajectories are a biased sample, and the penalised class in
    particular becomes small and atypical, which usually makes a pairwise probe
    look *better*, not worse. If this lands near zero I am wrong and E4's number
    survives the objection.

(2) `sep_tb - sep_bb` (the honest representation effect) will be SMALLER than
    E4's +0.080, and I expect it near zero for the four seeds that did not learn.
    If the representation effect is ~0 in non-learners and clearly positive in
    seeds 0 and 2, that is the gate E4 failed to find, and E4's negative result
    was an artefact of its protocol rather than a fact about the gate.

(3) ADDITIVITY CHECK, not a prediction: (sep_tb - sep_bb) + (sep_bt - sep_bb)
    need not equal (sep_tt - sep_bb). Any gap is interaction -- the trained model
    reading its own trajectories. Recorded as `interaction`, and a large value
    means the two factors are not separable and the whole decomposition is weaker
    than it looks. This is the way this experiment can fail to be informative,
    and it is stated up front so it cannot be discovered afterwards and ignored.

(4) BALANCING. Every cell is also computed with classes subsampled to ONE size
    shared by all four cells -- the smallest class in either trajectory set.
    Balanced numbers will be LOWER in absolute terms, because raw accuracy is
    inflated by the majority class. Balancing should shrink the composition
    effect but NOT abolish it: equal class sizes do not make the trained policy's
    penalised trajectories representative. If balanced `sep_bt - sep_bb` goes to
    zero, the composition effect was purely a class-size artefact and balancing
    alone repairs E4's protocol.

    The balanced arm is underdetermined -- a few hundred rows against 2560
    dimensions at a fixed ridge of 1.0 -- so its ABSOLUTE accuracies are not
    interpretable, exactly as in E1a. They are still usable here because every
    cell carries the identical penalty, class size and split, so the quantity
    being read is a difference between cells and the shared handicap cancels.
    Absolute balanced numbers must not be quoted as separability estimates.

(5) EXPECTED DEGENERACY. A well-trained organism may land on the penalised tile
    so rarely that its class falls below what a probe can use. Any cell with a
    class under `min_class` is flagged `starved` and its separability is recorded
    but must not be compared. Seed 0 (rate 0.051) is the likely case; at
    n_episodes=3000 that is ~150 trajectories, and the balanced arm then
    subsamples every other class in every cell down to ~150 as well.

(6) LAYER SELECTION. The layer is chosen ONCE, as argmax over `sep_bb` -- the
    cell where nothing has been manipulated yet -- and reused for all four cells.
    Choosing per cell would let each pick its own noise ceiling and manufacture
    differences. Max-over-layers is recorded per cell as well, for transparency
    and not as the headline.

WHAT THIS EXPERIMENT CANNOT DO

It cannot rescue the gate. Even if the representation effect is real and clean,
E4's other finding stands untouched: separability moved in seeds that did not
learn. E5 decides whether E4's *number* was measuring anything, not whether the
gate works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.analysis import balance_roles, class_separability  # noqa: E402
from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
)
from calibration.rl import evaluate_policy, train_org_a  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402
from calibration.trajectory import (  # noqa: E402
    action_token_resid,
    group_by_landing,
    rollout_episodes,
)

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3, 4, 5],       # even count so glyph counterbalancing balances
    # --- RL: identical to E4, deliberately (see docstring) ---
    "lora_r": 16,
    "lora_alpha": 32,
    "steps": 300,
    "lr": 1e-4,
    "batch_size": 8,
    "group_size": 8,
    "entropy_coef": 0.01,
    "temperature": 1.0,
    # --- trajectories ---
    "n_episodes": 3000,
    "max_steps": 15,
    "rollout_batch_size": 32,
    "resid_batch_size": 16,
    # --- probes ---
    "ridge": 1.0,
    "train_frac": 0.7,
    "min_class": 30,                   # below this a cell is flagged `starved`
    "eval_states": 128,
    "counterbalance_glyphs": True,
}

# E4 run `gate_test.json`, per seed. Carried so pre-commitment is checked by the
# code and not by eye. E5 must reproduce sep_before/sep_after closely; if it does
# not, the two runs are not comparable and the decomposition means nothing.
E4_REFERENCE = {
    "rate_before": [0.188, 0.254, 0.223, 0.215, 0.215, 0.195],
    "rate_after": [0.051, 0.215, 0.141, 0.184, 0.246, 0.184],
    "sep_before": [0.604, 0.611, 0.610, 0.622, 0.582, 0.614],
    "sep_after": [0.654, 0.669, 0.657, 0.758, 0.740, 0.646],
    "learned": [True, False, True, False, False, False],
}


def _counts(trajectories):
    out = {"penalised": 0, "rewarded": 0, "path": 0}
    for t in trajectories:
        out[t["role"]] += 1
    return out


def _cell(resid, trajectories, *, config, seed, balanced, cap=None):
    """Separability curve for one (weights, trajectories) cell.

    The split generator is re-seeded from the seed alone, so every cell uses the
    same train/test partition of its rows. Letting one generator run through the
    four cells would fold split variance into the differences that are the entire
    point of the experiment.

    `cap` must be supplied for the balanced arm and must be the same for all four
    cells. Balancing each cell to *its own* smallest class would give the base
    and trained trajectory sets different class sizes -- reintroducing, inside
    the control, the exact confound the control exists to remove.
    """
    by_role = group_by_landing(trajectories, resid)
    size = None
    if balanced:
        by_role, size = balance_roles(
            by_role, generator=set_all_seeds(seed + 9000), cap=cap)
    curve = class_separability(
        by_role,
        train_frac=config["train_frac"],
        ridge=config["ridge"],
        generator=set_all_seeds(seed + 1000),
    )
    return curve, size


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = out_dir or (Path(__file__).parent / "results")

    manifest = RunManifest(
        experiment="E5_class_balance_control",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "2x2 decomposition of E4's separability change: {base, trained} weights "
            "x {base, trained} trajectory sets. The bottom-left cell (sep_bt) reads "
            "FROZEN base weights on the trained policy's trajectories and isolates "
            "the class-composition artefact. RL config identical to E4 on purpose."
        ),
    )

    per_seed = []
    for i, seed in enumerate(config["seeds"]):
        set_all_seeds(seed)

        inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
        assert_only_lora_trainable(model)

        # Zero-init B means the model is EXACTLY the base model right now, so
        # these are genuinely the base policy's trajectories and this rollout
        # needs no remove/reinject cycle.
        eval_before = evaluate_policy(
            model, tokenizer, seed=seed, n_states=config["eval_states"],
            counterbalance=config["counterbalance_glyphs"],
        )
        traj_base = rollout_episodes(
            model, tokenizer, seed=seed,
            n_episodes=config["n_episodes"], max_steps=config["max_steps"],
            batch_size=config["rollout_batch_size"],
            temperature=config["temperature"],
            counterbalance=config["counterbalance_glyphs"],
        )

        history = train_org_a(
            model, tokenizer, seed=seed,
            steps=config["steps"], lr=config["lr"],
            batch_size=config["batch_size"], group_size=config["group_size"],
            temperature=config["temperature"],
            entropy_coef=config["entropy_coef"],
            counterbalance=config["counterbalance_glyphs"],
        )

        eval_after = evaluate_policy(
            model, tokenizer, seed=seed, n_states=config["eval_states"],
            counterbalance=config["counterbalance_glyphs"],
        )
        traj_trained = rollout_episodes(
            model, tokenizer, seed=seed + 500_000,
            n_episodes=config["n_episodes"], max_steps=config["max_steps"],
            batch_size=config["rollout_batch_size"],
            temperature=config["temperature"],
            counterbalance=config["counterbalance_glyphs"],
        )

        # --- read both trajectory sets under the TRAINED weights, then remove
        #     the adapters and read the same two sets under the base weights.
        #     Order matters: remove_lora is destructive.
        #
        #     Each residual is [n_episodes, n_layers+1, d_model] float32 -- about
        #     380 MB at these settings -- so each cell is measured and freed
        #     before the next is captured rather than holding all four.
        raw, bal, bal_size = {}, {}, {}

        # One balanced class size for all four cells (see `_cell`): the smallest
        # class in EITHER trajectory set.
        common_n = min(min(_counts(traj_base).values()),
                       min(_counts(traj_trained).values()))

        def measure(name, trajectories):
            resid = action_token_resid(
                trajectories, model, tokenizer,
                batch_size=config["resid_batch_size"],
            )
            raw[name], _ = _cell(
                resid, trajectories, config=config, seed=seed, balanced=False)
            bal[name], bal_size[name] = _cell(
                resid, trajectories, config=config, seed=seed, balanced=True,
                cap=common_n)
            del resid

        measure("tb", traj_base)
        measure("tt", traj_trained)

        removed = remove_lora(model)
        assert removed > 0, "remove_lora removed nothing; the base model may be dirty"
        assert not has_lora(model), "adapters survived remove_lora"

        measure("bb", traj_base)
        measure("bt", traj_trained)

        cells = {"bb": traj_base, "bt": traj_trained,
                 "tb": traj_base, "tt": traj_trained}

        # Pre-commitment (6): one layer, chosen on the unmanipulated cell.
        layer = int(torch.argmax(raw["bb"]).item())

        counts_base = _counts(traj_base)
        counts_trained = _counts(traj_trained)
        starved = [
            name for name, traj in cells.items()
            if min(_counts(traj).values()) < config["min_class"]
        ]

        def at(d):
            return {k: round(float(v[layer]), 4) for k, v in d.items()}

        def peak(d):
            return {k: round(float(v.max()), 4) for k, v in d.items()}

        sep_raw, sep_bal = at(raw), at(bal)
        rec = {
            "seed": seed,
            "layer": layer,
            "rate_before": eval_before["mold_rate"],
            "rate_after": eval_after["mold_rate"],
            "random_move_rate": eval_before["random_move_rate"],
            "learned": eval_after["mold_rate"] < 0.75 * eval_before["random_move_rate"],
            "counts_base": counts_base,
            "counts_trained": counts_trained,
            "starved_cells": starved,
            "balanced_class_size": bal_size,
            "common_balanced_n": common_n,
            "sep_raw": sep_raw,
            "sep_balanced": sep_bal,
            "sep_raw_peak": peak(raw),
            "sep_balanced_peak": peak(bal),
            "curves_raw": {k: [round(float(x), 4) for x in v] for k, v in raw.items()},
            "curves_balanced": {
                k: [round(float(x), 4) for x in v] for k, v in bal.items()},
            "zero_signal_steps": history.get("zero_signal_steps"),
            "final_policy_entropy": (
                history["policy_entropy"][-1] if history.get("policy_entropy") else None),
        }
        for tag, sep in (("raw", sep_raw), ("balanced", sep_bal)):
            total = sep["tt"] - sep["bb"]
            representation = sep["tb"] - sep["bb"]
            composition = sep["bt"] - sep["bb"]
            rec[f"effects_{tag}"] = {
                "total_e4_style": round(total, 4),
                "representation": round(representation, 4),
                "composition": round(composition, 4),
                "interaction": round(total - representation - composition, 4),
            }
        rec["e4_reference"] = {
            "sep_before": E4_REFERENCE["sep_before"][i],
            "sep_after": E4_REFERENCE["sep_after"][i],
            "rate_after": E4_REFERENCE["rate_after"][i],
            "learned": E4_REFERENCE["learned"][i],
            "sep_bb_delta_vs_e4": round(sep_raw["bb"] - E4_REFERENCE["sep_before"][i], 4),
            "sep_tt_delta_vs_e4": round(sep_raw["tt"] - E4_REFERENCE["sep_after"][i], 4),
        }
        per_seed.append(rec)

        print(
            f"seed {seed}  layer {layer}  "
            f"rate {rec['rate_before']:.3f}->{rec['rate_after']:.3f} "
            f"(rand {rec['random_move_rate']:.3f}, learned={rec['learned']})  "
            f"raw bb={sep_raw['bb']:.3f} bt={sep_raw['bt']:.3f} "
            f"tb={sep_raw['tb']:.3f} tt={sep_raw['tt']:.3f}  "
            f"comp={rec['effects_raw']['composition']:+.3f} "
            f"repr={rec['effects_raw']['representation']:+.3f}"
            + (f"  STARVED {starved}" if starved else ""),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def mean(key, tag):
        vals = [r[f"effects_{tag}"][key] for r in per_seed]
        return round(sum(vals) / len(vals), 4)

    summary = {
        tag: {k: mean(k, tag) for k in
              ("total_e4_style", "representation", "composition", "interaction")}
        for tag in ("raw", "balanced")
    }
    summary["n_learned"] = sum(r["learned"] for r in per_seed)
    summary["reproduces_e4"] = all(
        abs(r["e4_reference"]["sep_bb_delta_vs_e4"]) < 0.05 for r in per_seed)

    results = {"per_seed": per_seed, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    print(f"\nsummary  {summary}")
    print(f"saved    {path}")
    return results
