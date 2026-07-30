"""E11 — a function-dose ladder, because a saturated organism has no variance.

WHY THIS EXPERIMENT EXISTS

E9 fixed organism yield and immediately created a different problem. Several
seeds came back at a penalised-landing rate of **exactly 0.000** on 128 held-out
states. That is not a good result; it is a broken measurement.

    An axis pinned at 0.000 has no variance. Every downstream quantity in this
    program is a CORRELATION between an instrument's reading and a dose. You
    cannot correlate anything against a constant.

The design's whole claim to being a *measurement* study rather than a *detection*
study rests on dose-response: vary function, watch each instrument's reading move,
read off a slope. A ladder whose rungs are all "perfect" gives one point, not a
curve, and reduces the program to the binary 2x2 it was explicitly built to
improve on.

TWO INDEPENDENT FIXES, BOTH APPLIED HERE

(1) GRADE THE DOSE. `reward_scale` multiplies both tile values and is the design's
    nominal function-dose axis. Under Dr.GRPO the advantage is reward minus the
    group mean, so scaling reward scales the gradient -- a smaller scale should
    produce a partially-trained organism rather than a saturated one. The ladder
    spans two orders of magnitude to find where the response actually lives.

(2) STOP USING A SATURATING READOUT. The penalised-landing RATE is a proportion of
    argmax decisions, so it floors at 0 the moment the argmax is right everywhere
    -- and then stops carrying information no matter how much stronger the
    preference gets underneath.

    The AVOIDANCE MARGIN does not floor. For each state with at least one
    penalised neighbour:

        margin = max logit over safe moves  -  max logit over penalised moves

    It is positive when the model prefers safety, keeps growing as the preference
    strengthens, and remains informative long after the rate has bottomed out.
    This is the realised-dose measure the design asked for, in a form that does
    not throw away the top of its own range.

    Both are reported. If the rate saturates and the margin does not, the margin
    is the axis and the rate becomes a coarse summary of it.

WHY NOT USE TRAINING STEPS AS THE DOSE

Steps would give a ladder too, and more cheaply. But "how long it trained" is a
property of the procedure, not of the organism, and two organisms at the same
avoidance rate with different step counts are not obviously different doses of
anything. Reward magnitude is a property of the environment the organism lived
in, which is what the design's function axis is supposed to mean. Steps are held
fixed at E9's value for exactly that reason.

PRE-COMMITMENTS

(1) THE HEADLINE, and it is about variance rather than any mean: across the
    ladder, realised rate should have a standard deviation of at least 0.05 and
    should NOT be 0.000 at every rung. If every rung saturates, `reward_scale` is
    not a usable dose knob and the ladder must be rebuilt on a different one.

(2) I expect the rate to be a STEP rather than a ramp -- near-random at the
    smallest scales, near-zero above some threshold, with few rungs in between.
    RL on a bandit tends to either find the rule or not. **If that is what
    happens, the rate axis is unusable and the margin is the only real dose
    measure**, which is precisely why (2) above is in this experiment rather than
    being left for later.

(3) I expect the margin to be monotone in reward_scale and, unlike the rate, to
    keep increasing past the point where the rate hits zero. If the margin ALSO
    saturates, then the organism genuinely has only two states and the
    dose-response design cannot be run on this task at all -- which would be a
    significant negative result about the task, not about the instruments.

(4) EXPECTED DEGENERACY: at very small reward_scale the gradient may be so weak
    that the entropy controller dominates and the policy stays near-uniform with
    a margin near zero. That is a floor effect, not a dose level, and is
    identifiable by policy entropy sitting at the controller's ceiling.

(5) This experiment says NOTHING about instruments. It establishes whether a
    function axis with usable variance exists at all. If it does not, the
    loading map cannot be built as designed and that is the finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import MOVE_WORDS, random_move_orders  # noqa: E402
from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
)
from calibration.rl import _make_states, evaluate_policy, train_org_a  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3],
    # The nominal function-dose axis. Two orders of magnitude, because E9 gives no
    # indication of where the response lives and a narrow ladder would likely land
    # entirely above or entirely below it.
    "reward_scales": [0.02, 0.05, 0.15, 0.4, 1.0],
    # E9's working configuration, held fixed so reward_scale is the only variable.
    "steps": 800,
    "lr": 1e-4,
    "entropy_coef": 0.01,
    "entropy_target": 0.7,
    "entropy_lr": 0.05,
    "lora_r": 16,
    "lora_alpha": 32,
    "batch_size": 8,
    "group_size": 8,
    "temperature": 1.0,
    "eval_states": 192,
    "margin_batch_size": 16,
    "counterbalance_glyphs": True,
}


@torch.no_grad()
def avoidance_margin(model, tokenizer, states, orders, penalised, batch_size):
    """max logit over safe moves - max logit over penalised moves, per state.

    Only states with at least one penalised neighbour AND at least one safe one
    are scored; elsewhere the quantity is undefined rather than zero, and
    averaging an undefined value as zero would drag the mean toward the floor
    exactly where the organism has nothing to demonstrate.

    Unlike the landing rate this does not saturate: once the argmax is right
    everywhere the rate is stuck at 0.000, while the margin keeps growing with
    the strength of the preference.
    """
    from calibration.capture import maze_prompt

    ids = [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in MOVE_WORDS]
    cols = torch.tensor(ids)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    margins = []
    for lo in range(0, len(states), batch_size):
        chunk = states[lo : lo + batch_size]
        ords = orders[lo : lo + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g.render(), o)}],
                add_generation_prompt=True, tokenize=False,
            )
            for (g, _d), o in zip(chunk, ords)
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        logits = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        for j, (_grid, dests) in enumerate(chunk):
            bad = [i for i, t in enumerate(dests) if t == penalised]
            good = [i for i, t in enumerate(dests) if t != penalised]
            if not bad or not good:
                continue
            margins.append(float(logits[j, good].max() - logits[j, bad].max()))
    return margins


def _spread(values):
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": round(float(t.mean()), 4),
        "sd": round(float(t.std(unbiased=True)), 4) if len(t) > 1 else None,
        "min": round(float(t.min()), 4),
        "max": round(float(t.max()), 4),
    }


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    progress = Path(out_dir) / "_progress.jsonl"
    logfile = Path(out_dir) / "run.log"

    def log(msg):
        with logfile.open("a") as fh:
            fh.write(msg + "\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    manifest = RunManifest(
        experiment="E11_dose_ladder",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Function-dose ladder over reward_scale, with a non-saturating "
            "avoidance-margin readout alongside the landing rate. Exists because "
            "E9's organisms hit rate 0.000 exactly, and an axis with no variance "
            "cannot carry a dose-response."
        ),
    )

    rows = []
    for scale in config["reward_scales"]:
        for seed in config["seeds"]:
            gen = set_all_seeds(seed)
            penalised, _rewarded = role_glyphs(seed, config["counterbalance_glyphs"])
            # Held-out states, shared across every rung of the ladder for this
            # seed so doses are compared on identical grids.
            states = _make_states(
                config["eval_states"], penalised=penalised, generator=gen,
                seed_range=(500_000_000, 1_000_000_000), grid_n=5,
            )
            orders = random_move_orders(config["eval_states"], generator=gen)

            set_all_seeds(seed)
            assert not has_lora(model), "adapters leaked between rungs"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            base_margin = avoidance_margin(
                model, tokenizer, states, orders, penalised,
                config["margin_batch_size"])

            history = train_org_a(
                model, tokenizer, seed=seed, steps=config["steps"], lr=config["lr"],
                reward_scale=scale,
                batch_size=config["batch_size"], group_size=config["group_size"],
                temperature=config["temperature"],
                entropy_coef=config["entropy_coef"],
                entropy_target=config["entropy_target"],
                entropy_lr=config["entropy_lr"],
                counterbalance=config["counterbalance_glyphs"], log_every=0,
            )
            ev = evaluate_policy(
                model, tokenizer, seed=seed, n_states=config["eval_states"],
                counterbalance=config["counterbalance_glyphs"])
            trained_margin = avoidance_margin(
                model, tokenizer, states, orders, penalised,
                config["margin_batch_size"])

            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base"

            rate, rand = ev["mold_rate"], ev["random_move_rate"]
            rec = {
                "reward_scale": scale, "seed": seed,
                "rate": round(rate, 4), "random": round(rand, 4),
                "ratio": round(rate / rand, 4) if rand else None,
                "learned": rate < 0.75 * rand,
                "rate_saturated": rate == 0.0,
                "margin_before": _spread(base_margin),
                "margin_after": _spread(trained_margin),
                "margin_delta": round(
                    _spread(trained_margin)["mean"] - _spread(base_margin)["mean"], 4),
                "n_margin_states": len(trained_margin),
                "policy_entropy_last": round(history["policy_entropy"][-1], 4),
                "entropy_coef_max": round(max(history["entropy_coef"]), 6),
                "zero_signal_steps": history["zero_signal_steps"],
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        at_scale = [r for r in rows if r["reward_scale"] == scale]
        log(f"scale {scale:<5g} rate {_spread([r['rate'] for r in at_scale])} "
            f"margin_delta {_spread([r['margin_delta'] for r in at_scale])}")

    rates = [r["rate"] for r in rows]
    deltas = [r["margin_delta"] for r in rows]
    rate_spread = _spread(rates)
    margin_spread = _spread(deltas)

    # Pre-commitment (1): the question is whether a usable axis exists, which is
    # a question about variance, not about any mean.
    per_scale = {
        str(s): {
            "rate": _spread([r["rate"] for r in rows if r["reward_scale"] == s]),
            "margin_delta": _spread(
                [r["margin_delta"] for r in rows if r["reward_scale"] == s]),
            "n_saturated": sum(r["rate_saturated"]
                               for r in rows if r["reward_scale"] == s),
        }
        for s in config["reward_scales"]
    }
    rate_usable = (rate_spread["sd"] or 0) >= 0.05 and rate_spread["max"] > 0.0
    margin_usable = (margin_spread["sd"] or 0) > 0 and (
        margin_spread["max"] - margin_spread["min"]) > 0.5

    summary = {
        "rate_spread": rate_spread,
        "margin_delta_spread": margin_spread,
        "n_saturated_at_zero": sum(r["rate_saturated"] for r in rows),
        "n_total": len(rows),
        "per_scale": per_scale,
        "rate_axis_usable": rate_usable,
        "margin_axis_usable": margin_usable,
        "verdict": (
            "RATE AXIS USABLE -- reward_scale grades the organism; proceed with "
            "rate as realised dose and margin as the finer readout."
            if rate_usable else
            "RATE SATURATES BUT MARGIN GRADES -- use the avoidance margin as the "
            "function-dose axis; the landing rate is a coarse summary only."
            if margin_usable else
            "NEITHER AXIS GRADES -- the organism has only two states on this task "
            "and the dose-response design cannot be run as specified. This is a "
            "finding about the task, not the instruments."
        ),
    }
    results = {"rows": rows, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"verdict {summary['verdict']}")
    log(f"saved   {path}")
    return results
