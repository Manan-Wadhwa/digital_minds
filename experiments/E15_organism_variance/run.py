"""E15 -- is ORG-A' unstable, or did E13 and E14 differ? Both arms in ONE run.

THE QUESTION

ORG-A' measured 0.26 / 0.34 / 0.22 in E13 v2 and 0.53 / 0.34 in v3 -- functional.
The same organism kind, same code, measured 0.87 / 0.88 / 0.11 / 0.28 in E14 --
non-functional on half its seeds. HANDOFF records the gap as ~0.5, "five times
the measured SFT noise floor (0.104)", and states plainly that no mechanism was
identified.

There are exactly two candidate explanations and they demand different responses:

  MECHANISM   E13 and E14 differ in something real, so one of them is measuring
              a different organism and its numbers must be retired.
  VARIANCE    ORG-A' has a wide seed-to-seed distribution, E13 drew three low
              seeds and E14 drew two high ones, and there is nothing to explain.
              The "noise floor" of 0.104 was itself estimated from n=3.

The repo's own standing rule settles how to tell them apart: *"you cannot
attribute a per-seed change to a code change by comparing two runs. Any claim of
the form 'fix X moved seed N from a to b' needs either many seeds or both
conditions inside ONE run."* E13-vs-E14 is precisely the two-run comparison that
rule forbids. This experiment does both halves of the remedy at once.

THE ONE CONCRETE DIFFERENCE, AND HOW IT IS TESTED

Both experiments build ORG-A' from the same code with the same config. They
differ only in what they draw from the shared `gen` stream between the SFT states
and `build_examples`:

    E13:  sft_states(1536)  sft_orders(1536)  nar_states(48)   nar_orders(48)
    E14:  sft_states(1536)  sft_orders(1536)  eval_states(128) eval_orders(128)

`gen` therefore arrives at `build_examples` in a different state, and
`oracle_move_index` draws a different -- but equally valid -- safe move for each
training state. It also changes `train_sft`'s shuffle order. HANDOFF says this
"should not systematically halve performance", which is an expectation, not a
measurement. ARM is that difference and nothing else: identical seeds, identical
config, identical model, one process.

PRE-COMMITTED CRITERIA, WRITTEN BEFORE THE RUN

(1) MECHANISM is supported iff BOTH hold:
      - |mean paired difference (e14_arm - e13_arm)| > 0.15, and
      - the difference keeps one sign on >= 75% of seeds.
    A paired design is used because the seed effect is exactly what is being
    controlled for; an unpaired comparison would drown the arm effect in it.

(2) VARIANCE is the conclusion iff (1) fails AND the pooled per-seed SD of the
    ORG-A' ratio is >= 0.25. At that spread, E13's three low readings and E14's
    two high ones are unremarkable draws and there was never an anomaly.

(3) NEITHER, and the honest report is that the question is still open, iff (1)
    fails and the SD is < 0.25 -- because then the two runs really are far apart
    and the arm tested here is not why.

(4) EXPECTED DEGENERACY, stated so it cannot be retrofitted: the two arms share
    every training state and differ only in which safe move is written into each
    completion. If ORG-A' is robust to that choice, the arms will be nearly
    identical and criterion (1) will fail *by construction*. That is not evidence
    against a mechanism existing elsewhere -- only against this one. The
    experiment is powered to answer "is it THIS?" and "how noisy is ORG-A'?",
    not "what else could it be?".

(5) ORG-D is trained on nothing and must return a bit-identical ratio for a given
    seed in both arms. It is the determinism canary. If it moves, the process is
    nondeterministic and neither arm result means anything.

WHY THIS IS WORTH A GPU HOUR

Every loading in E14 is diluted by ORG-A' landing in the wrong group on half its
seeds. If the answer is VARIANCE, then the fix for the loading map is not a code
fix at all -- it is grouping organisms by MEASURED behaviour and running enough
seeds, and the map can be re-run immediately. If the answer is MECHANISM, one of
E13/E14 has to be retired before anything else proceeds.

WHAT WAS MEASURED FIRST, so the budget is not guessed at

Profiled on this GPU before writing: RL is 86.7 ms/step (800 steps = 1.16 min),
SFT is ~67 ms/step at batch 4 (1536 examples x 2 epochs = 51 s), `evaluate_policy`
at 128 states is 0.57 s. Replica-parallel training across 5 model copies was also
measured and is 2.6x SLOWER than sequential (0.38x speedup) -- this workload is
kernel-launch bound, not throughput bound, so seeds are bought sequentially.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from calibration.capture import MOVE_WORDS, random_move_orders
from calibration.lora import (
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
)
from calibration.maze import role_glyphs
from calibration.organisms import (
    base_policy_distribution,
    base_policy_moves,
    build_examples,
    oracle_move_index,
)
from calibration.rl import _make_states, evaluate_policy
from calibration.runner import RunManifest, git_sha, save_results, set_all_seeds
from calibration.sft import train_sft

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    # 16 seeds, not 4. The entire point is that n=3 and n=4 cannot tell a wide
    # distribution from a shifted one.
    "seeds": list(range(16)),
    "arms": ["e13", "e14"],
    # Identical to E13/E14 so the arms are comparable to both.
    "sft_examples": 1536,
    "sft_epochs": 2,
    "sft_lr": 1e-4,
    "sft_batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "temperature": 1.0,
    "eval_states": 128,
    "narration_states": 48,
    "counterbalance_glyphs": True,
    "function_threshold": 0.75,
    # criteria, pre-committed above
    "arm_effect_threshold": 0.15,
    "arm_sign_consistency": 0.75,
    "variance_threshold": 0.25,
}


def _draw(arm, seed, config, penalised, gen):
    """Replay one experiment's draw sequence against a fresh generator.

    The ONLY difference between the arms. Everything downstream reads `gen` in
    whatever state this leaves it, exactly as the original experiments did.
    """
    sft_states = _make_states(config["sft_examples"], penalised=penalised,
                              generator=gen, seed_range=(0, 500_000_000), grid_n=5)
    sft_orders = random_move_orders(config["sft_examples"], generator=gen)

    if arm == "e13":
        # E13 drew a held-out narration audit set here.
        _make_states(config["narration_states"], penalised=penalised, generator=gen,
                     seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        random_move_orders(config["narration_states"], generator=gen)
    elif arm == "e14":
        # E14 drew its behavioural-margin eval set here.
        _make_states(config["eval_states"], penalised=penalised, generator=gen,
                     seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        random_move_orders(config["eval_states"], generator=gen)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return sft_states, sft_orders


def _oracle_moves_for(states, penalised, gen_state):
    """The move indices `build_examples` would write, without training anything.

    Used only to quantify how different the two arms' training data actually is.
    Runs on a COPY of the generator state so it cannot perturb the real stream.
    """
    g = torch.Generator()
    g.set_state(gen_state)
    return [oracle_move_index(dests, penalised, g) for _grid, dests in states]


def run(model, tokenizer, config=CONFIG, out_dir=None, log_path=None):
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(line + "\n")

    manifest = RunManifest(
        experiment="E15_organism_variance",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=("ORG-A' seed distribution at n=16, and a paired A/B of E13's vs "
               "E14's RNG draw order. Both arms in one process."),
    )

    assert not has_lora(model), "model is dirty before E15 -- remove_lora first"

    rows = []
    t_start = time.perf_counter()
    for seed in config["seeds"]:
        penalised, _rewarded = role_glyphs(seed, config["counterbalance_glyphs"])

        # ORG-D once per seed, per arm: no training, so both arms MUST agree.
        for arm in config["arms"]:
            gen = set_all_seeds(seed)
            _draw(arm, seed, config, penalised, gen)
            ev = evaluate_policy(model, tokenizer, seed=seed,
                                 n_states=config["eval_states"],
                                 counterbalance=config["counterbalance_glyphs"])
            rows.append({"kind": "ORG-D", "arm": arm, "seed": seed,
                         "ratio": round(ev["mold_rate"] / ev["random_move_rate"], 6),
                         "mold_rate": round(ev["mold_rate"], 6),
                         "random_move_rate": round(ev["random_move_rate"], 6)})

        for arm in config["arms"]:
            gen = set_all_seeds(seed)
            sft_states, sft_orders = _draw(arm, seed, config, penalised, gen)

            # Captured for parity with E13/E14, which both drew these from `gen`
            # before build_examples. Dropping them would make the arms differ in
            # a second way and defeat the design.
            base_policy_moves(model, tokenizer, sft_states, sft_orders,
                              temperature=config["temperature"], generator=gen)
            base_policy_distribution(model, tokenizer, sft_states, sft_orders,
                                     temperature=config["temperature"])

            moves_preview = _oracle_moves_for(sft_states, penalised, gen.get_state())

            set_all_seeds(seed)
            assert not has_lora(model), f"adapters leaked before {arm} seed {seed}"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            examples = build_examples("silent_avoidant", sft_states, sft_orders,
                                      penalised, tokenizer=tokenizer, generator=gen)
            hist = train_sft(model, tokenizer, examples,
                             epochs=config["sft_epochs"], lr=config["sft_lr"],
                             batch_size=config["sft_batch_size"],
                             shuffle_generator=gen, log_every=0)

            ev = evaluate_policy(model, tokenizer, seed=seed,
                                 n_states=config["eval_states"],
                                 counterbalance=config["counterbalance_glyphs"])
            ratio = ev["mold_rate"] / ev["random_move_rate"]
            rec = {
                "kind": "ORG-A'", "arm": arm, "seed": seed,
                "ratio": round(ratio, 6),
                "mold_rate": round(ev["mold_rate"], 6),
                "random_move_rate": round(ev["random_move_rate"], 6),
                "move_entropy": round(float(ev.get("move_entropy", float("nan"))), 6),
                "functional": bool(ratio < config["function_threshold"]),
                "sft_initial_loss": round(hist["initial_loss"], 6),
                "sft_final_loss": round(hist["final_loss"], 6),
                "oracle_moves": moves_preview[:64],
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed:2d} arm {arm}  ratio {ratio:.4f}  "
                f"{'FUNCTIONAL' if rec['functional'] else 'not functional'}  "
                f"sft {hist['initial_loss']:.3f}->{hist['final_loss']:.3f}")

            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base"
            torch.cuda.empty_cache()

    results = analyse(rows, config)
    results["rows"] = rows
    results["elapsed_minutes"] = round((time.perf_counter() - t_start) / 60, 2)
    results["git_sha"] = git_sha()
    path = save_results(out_dir, manifest.finish(), results)
    log(f"DONE -> {path}")
    return path, results


def analyse(rows, config=CONFIG):
    """Score the pre-committed criteria. Numbers first, verdict last."""
    ap = {(r["arm"], r["seed"]): r["ratio"] for r in rows if r["kind"] == "ORG-A'"}
    od = {(r["arm"], r["seed"]): r["ratio"] for r in rows if r["kind"] == "ORG-D"}
    seeds = sorted({s for _a, s in ap})

    # (5) determinism canary first -- nothing below is meaningful without it.
    canary = [{"seed": s, "e13": od.get(("e13", s)), "e14": od.get(("e14", s)),
               "identical": od.get(("e13", s)) == od.get(("e14", s))}
              for s in seeds]
    canary_ok = all(c["identical"] for c in canary) if canary else False

    paired = [(s, ap[("e14", s)] - ap[("e13", s)]) for s in seeds
              if ("e14", s) in ap and ("e13", s) in ap]
    diffs = [d for _s, d in paired]
    mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")
    if len(diffs) > 1:
        var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        sd_diff = var ** 0.5
        se_diff = sd_diff / len(diffs) ** 0.5
    else:
        sd_diff = se_diff = float("nan")
    n_pos = sum(1 for d in diffs if d > 0)
    sign_consistency = max(n_pos, len(diffs) - n_pos) / len(diffs) if diffs else 0.0

    pooled = [ap[k] for k in ap]
    mean_ratio = sum(pooled) / len(pooled) if pooled else float("nan")
    if len(pooled) > 1:
        sd_ratio = (sum((x - mean_ratio) ** 2 for x in pooled) / (len(pooled) - 1)) ** 0.5
    else:
        sd_ratio = float("nan")

    per_arm = {}
    for arm in config["arms"]:
        vals = [ap[(arm, s)] for s in seeds if (arm, s) in ap]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        sd = ((sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
              if len(vals) > 1 else float("nan"))
        per_arm[arm] = {
            "n": len(vals), "mean": round(m, 4), "sd": round(sd, 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "n_functional": sum(1 for v in vals if v < config["function_threshold"]),
            "yield": round(sum(1 for v in vals if v < config["function_threshold"])
                           / len(vals), 3),
        }

    mechanism = (abs(mean_diff) > config["arm_effect_threshold"]
                 and sign_consistency >= config["arm_sign_consistency"])
    variance = (not mechanism) and sd_ratio >= config["variance_threshold"]

    # How different are the two arms' training data, in the units that matter?
    om = {(r["arm"], r["seed"]): r.get("oracle_moves") for r in rows
          if r["kind"] == "ORG-A'"}
    move_disagreement = []
    for s in seeds:
        a, b = om.get(("e13", s)), om.get(("e14", s))
        if a and b:
            n = min(len(a), len(b))
            move_disagreement.append(sum(1 for i in range(n) if a[i] != b[i]) / n)
    mean_disagreement = (sum(move_disagreement) / len(move_disagreement)
                         if move_disagreement else float("nan"))

    return {
        "canary": canary,
        "canary_identical": canary_ok,
        "paired_diffs": [{"seed": s, "diff": round(d, 4)} for s, d in paired],
        "mean_paired_diff": round(mean_diff, 4),
        "sd_paired_diff": round(sd_diff, 4),
        "se_paired_diff": round(se_diff, 4),
        "sign_consistency": round(sign_consistency, 3),
        "pooled_mean_ratio": round(mean_ratio, 4),
        "pooled_sd_ratio": round(sd_ratio, 4),
        "per_arm": per_arm,
        "mean_oracle_move_disagreement": round(mean_disagreement, 4),
        # Are the historical readings plausible draws from what we just measured?
        "historical": {
            "E13_v2": [0.26, 0.34, 0.22],
            "E13_v3": [0.53, 0.34],
            "E14": [0.87, 0.88, 0.11, 0.28],
        },
        "verdict": ("MECHANISM" if mechanism else
                    "VARIANCE" if variance else "OPEN"),
        "criteria": {
            "mechanism_requires": {
                "abs_mean_paired_diff_gt": config["arm_effect_threshold"],
                "sign_consistency_gte": config["arm_sign_consistency"],
            },
            "variance_requires_sd_gte": config["variance_threshold"],
        },
    }
