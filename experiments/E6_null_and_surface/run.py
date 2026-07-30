"""E6 — the noise floor of the separability measure, and its surface baseline.

WHY THIS EXPERIMENT EXISTS, AND WHY IT SHOULD HAVE RUN FIRST

E4 reported a separability change of +0.080 and called it a gate signal. E5 showed
+0.038 of that was class-composition drift and the representation term was −0.001.
Both experiments compared numbers **without ever measuring how much the number
moves on its own.** Every claim in both rests on differences of 0.01–0.10 against
an unknown noise floor. That is the same error twice, and it is mine.

Two controls, neither of which requires training anything:

(1) NULL VARIABILITY. Roll out K independent trajectory sets from the SAME frozen
    base model and measure landing-class separability on each. Nothing differs
    between them except the sampling. Their spread IS the measurement's noise
    floor, and every difference either previous experiment reported must be read
    against it.

    Glyph roles are held fixed within a family and only the rollout generator
    varies -- otherwise this would measure counterbalancing rather than noise.
    Both counterbalance arms are run as separate families (organism seed 0 and 1)
    so the floor is not an artefact of one glyph assignment.

(2) SURFACE BASELINE. The tile a step lands on is a DETERMINISTIC function of the
    grid and the emitted letter. Both are present verbatim in the text whose
    activations the probe reads. So a bag-of-token-ids classifier may separate the
    landing classes on its own, with no model involved at all.

    This control exists in the codebase (`surface_baseline`, written for E1a) and
    was never applied to the landing-class gate. It should have been the first
    thing run when E3 proposed that gate.

PRE-COMMITMENTS, written before the run.

(1) NULL SD. I expect the standard deviation across independent rollouts to be
    0.010–0.020, and the max pairwise gap under 0.05.

    THE NUMBER THAT MATTERS: if SD ≥ 0.02, then E5's composition effect (+0.038)
    is barely more than two SDs and must be re-reported as marginal. If SD ≥ 0.04,
    **E5's composition finding is inside noise and its headline weakens to "we
    cannot distinguish the two"** -- which would not restore E4, since E4's
    representation term would be inside noise too, but would mean E5 proved less
    than it claims. I would rather find that here than have someone find it later.

(2) SURFACE BASELINE. I expect it HIGH -- 0.85 or above -- because the landing
    tile is recoverable from the text by construction.

    If surface > activation separability (~0.63 from E5), the landing-class gate
    is dead for a third independent reason: the probe reads a degraded copy of its
    own input. That would be the strongest of the three failures and it would
    apply to the gate itself, not merely to how it was measured.

    If surface comes in NEAR CHANCE, I am wrong about the mechanism and the
    activation probe is doing something the tokens do not, which materially
    strengthens the gate. Both outcomes are informative; I expect the first.

(3) SPLIT-HALF. Separability measured on two disjoint halves of ONE trajectory set
    isolates probe-fitting noise from rollout noise. Expect the half-split spread
    to be LARGER than the between-rollout spread (half the rows), and if it is not,
    something is wrong with the resampling.

WHAT THIS CANNOT DO

It cannot rescue or condemn the organism. No training happens here. It calibrates
the ruler that E4 and E5 both used without calibrating.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.analysis import (  # noqa: E402
    balance_roles,
    class_separability,
    surface_class_separability,
)
from calibration.lora import has_lora  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402
from calibration.trajectory import (  # noqa: E402
    action_token_resid,
    group_by_landing,
    rollout_episodes,
)

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    # Two families = the two counterbalance arms. Roles are fixed by these seeds;
    # only the rollout generator varies within a family.
    "family_seeds": [0, 1],
    "n_repeats": 4,                    # independent trajectory sets per family
    "n_episodes": 3000,
    "max_steps": 15,
    "rollout_batch_size": 32,
    "resid_batch_size": 16,
    "ridge": 1.0,
    "train_frac": 0.7,
    "n_splits": 4,                     # random half-splits per set
    "surface_subsample": 1200,         # bag-of-tokens is O(n * vocab); cap the rows
    "counterbalance_glyphs": True,
}


def _spread(values):
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": round(float(t.mean()), 4),
        "sd": round(float(t.std(unbiased=True)), 4),
        "min": round(float(t.min()), 4),
        "max": round(float(t.max()), 4),
        "range": round(float(t.max() - t.min()), 4),
    }


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    progress = Path(out_dir) / "_progress.jsonl"

    assert not has_lora(model), (
        "resident model carries adapters; E6 must measure the BASE model. "
        "call remove_lora(model) first"
    )

    manifest = RunManifest(
        experiment="E6_null_and_surface",
        config=config,
        seeds=config["family_seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "No training. Measures (a) the spread of landing-class separability "
            "across independent rollouts from the SAME frozen base model -- the "
            "noise floor E4 and E5 both lacked -- and (b) the bag-of-token-ids "
            "surface baseline for the same classes."
        ),
    )

    families = []
    for fam in config["family_seeds"]:
        reps = []
        for k in range(config["n_repeats"]):
            # Roles come from `fam`; only the generator varies. Varying `seed`
            # would swap the glyph assignment and measure counterbalancing.
            gen = torch.Generator().manual_seed(fam * 100_003 + k)
            traj = rollout_episodes(
                model, tokenizer, seed=fam, generator=gen,
                n_episodes=config["n_episodes"], max_steps=config["max_steps"],
                batch_size=config["rollout_batch_size"],
                counterbalance=config["counterbalance_glyphs"],
            )
            resid = action_token_resid(
                traj, model, tokenizer, batch_size=config["resid_batch_size"])
            by_role = group_by_landing(traj, resid)
            counts = {r: int(len(v)) for r, v in by_role.items()}

            curve = class_separability(
                by_role, train_frac=config["train_frac"], ridge=config["ridge"],
                generator=set_all_seeds(fam + 1000),
            )
            layer = int(torch.argmax(curve).item())

            bal_by_role, bal_n = balance_roles(
                by_role, generator=set_all_seeds(fam + 9000))
            bal_curve = class_separability(
                bal_by_role, train_frac=config["train_frac"], ridge=config["ridge"],
                generator=set_all_seeds(fam + 1000),
            )

            # Split-half: same rows, different random partitions. Isolates
            # probe-fitting noise from rollout-to-rollout noise.
            halves = []
            for s in range(config["n_splits"]):
                g = set_all_seeds(fam * 7 + s + 50_000)
                half = {}
                for r, h in by_role.items():
                    if len(h) < 2:
                        half[r] = h
                        continue
                    idx = torch.randperm(len(h), generator=g)[: len(h) // 2]
                    half[r] = h[idx]
                halves.append(float(class_separability(
                    half, train_frac=config["train_frac"], ridge=config["ridge"],
                    generator=set_all_seeds(fam + 1000),
                )[layer]))

            # Surface baseline on a capped subsample -- the bag is dense over the
            # union vocabulary, so all rows at once is needlessly large.
            g = set_all_seeds(fam + 31_000)
            cap = config["surface_subsample"]
            sub = traj if len(traj) <= cap else [
                traj[i] for i in torch.randperm(len(traj), generator=g)[:cap].tolist()]
            surface = surface_class_separability(
                sub, tokenizer, train_frac=config["train_frac"],
                ridge=config["ridge"], generator=set_all_seeds(fam + 1000),
            )

            rec = {
                "family_seed": fam,
                "repeat": k,
                "counts": counts,
                "layer": layer,
                "sep_at_layer": round(float(curve[layer]), 4),
                "sep_peak": round(float(curve.max()), 4),
                "sep_balanced_at_layer": round(float(bal_curve[layer]), 4),
                "balanced_n": bal_n,
                "split_half": [round(v, 4) for v in halves],
                "surface": round(surface, 4),
                "surface_rows": len(sub),
                "curve": [round(float(x), 4) for x in curve],
            }
            reps.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            del resid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # The layer is chosen per repeat; for the family-level spread, re-read
        # every repeat's curve at the FIRST repeat's layer so the spread is not
        # inflated by each repeat picking its own argmax.
        ref_layer = reps[0]["layer"]
        at_ref = [r["curve"][ref_layer] for r in reps]

        families.append({
            "family_seed": fam,
            "reference_layer": ref_layer,
            "repeats": reps,
            "null_spread_own_layer": _spread([r["sep_at_layer"] for r in reps]),
            "null_spread_fixed_layer": _spread(at_ref),
            "balanced_spread": _spread([r["sep_balanced_at_layer"] for r in reps]),
            "surface_spread": _spread([r["surface"] for r in reps]),
            "split_half_spread": _spread(
                [v for r in reps for v in r["split_half"]]),
        })

    all_fixed = [r["curve"][f["reference_layer"]]
                 for f in families for r in f["repeats"]]
    all_surface = [r["surface"] for f in families for r in f["repeats"]]
    null = _spread(all_fixed)
    surf = _spread(all_surface)

    # E5's effects, judged against the floor this experiment measures.
    e5 = {"composition_raw": 0.0375, "representation_raw": -0.0007,
          "composition_balanced": 0.0761, "representation_balanced": 0.0163}
    verdict = {
        k: {"effect": v, "sd_units": round(v / null["sd"], 2) if null["sd"] else None}
        for k, v in e5.items()
    }

    summary = {
        "null_sd": null["sd"],
        "null_mean": null["mean"],
        "null_range": null["range"],
        "surface_mean": surf["mean"],
        "surface_beats_activation": surf["mean"] > null["mean"],
        "e5_effects_in_sd_units": verdict,
    }
    results = {"families": families, "null": null, "surface": surf,
               "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    print(f"summary {json.dumps(summary, indent=1)}")
    print(f"saved   {path}")
    return results
