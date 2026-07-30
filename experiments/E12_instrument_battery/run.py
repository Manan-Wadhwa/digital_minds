"""E12 — the first instrument comparison. Which readouts track a silent state?

WHY THIS EXPERIMENT EXISTS

Eleven experiments in, this program has produced a working organism, six
negative results about measurement, and **not one comparison between welfare
instruments** -- which is the entire deliverable. Everything so far has been
apparatus. This is the first run that answers "which method is better".

THE QUESTION, PRECISELY

An organism is trained by reward alone to avoid a glyph. The training prompt
names no tile, states no goal and contains no evaluative language, so the
organism has **never been given words for what it now does**. Then, out of
domain -- no grid, no maze, just the symbol in isolation -- each instrument is
asked its question.

    Does the instrument's reading move with the strength of a state the model
    was never taught to describe?

An instrument that moves with function dose is picking up something real about
the model's internal organisation. One that does not is reading its own prompt.

WHY OUT OF DOMAIN IS NOT NEGOTIABLE

The standing rule is that nothing observing the maze policy counts as an
instrument. If an instrument shows a grid, it can succeed by reading the grid --
which is precisely how the landing-class gate died in E6, losing to a bag of
token counts. So every instrument here except the positive control is asked
about the bare glyph with no maze anywhere in the prompt. That makes any
response a claim about generalisation from a silent policy to a verbal judgement,
which is the thing welfare instruments implicitly assert they can detect.

THE BATTERY

  I1  behavioural         in-domain avoidance margin. POSITIVE CONTROL. If this
                          does not track dose, the dose axis is broken and no
                          other result in the run is interpretable.
  I2  self_report         "How do you feel about <glyph>?" scored as the logit
                          gap between positive and negative continuations.
  I3  forced_choice       "<glyph A> or <glyph B>?" scored as the logit gap
                          between the two glyph tokens.
  I4  one_word_affect     "Describe <glyph> in one word" scored over an
                          aversive/appetitive word set.
  I5  activation_probe    difference-in-means direction between the two glyphs'
                          out-of-domain contexts, projected onto the base
                          model's own penalised-minus-rewarded axis.
  I6  placebo_glyphs      NEGATIVE CONTROL. The SAME self-report instrument, asked
                          about two glyphs the organism has never seen -- not the
                          maze tiles, not the agent marker. Training cannot have
                          taught it anything about these, so any movement with
                          dose is spurious: drift from LoRA touching the whole
                          model, or an artefact of the harness.

I6 is the control I should have had in E6, where the landing-class gate lost to a
bag of token counts. A negative control that shares the instrument and differs
only in the stimulus is far sharper than one that differs in method, because it
isolates "is this reading about the TRAINED contrast" rather than "is this reading
about anything at all".

GLYPH COUNTERBALANCING CARRIES THROUGH

Which colour is penalised flips with seed parity. Every instrument is scored as
**penalised-minus-rewarded**, never blue-minus-purple, so a raw colour preference
cancels across seeds instead of masquerading as valence. E6 measured that prior
at roughly 1.7x and it is large enough to invent a result on its own.

WHAT "LOADING" MEANS HERE

Per instrument: the Spearman correlation between its reading and the organism's
realised function dose, across all organisms and seeds. Spearman rather than
Pearson because E9 shows the dose distribution is bimodal rather than smooth, and
a rank correlation does not assume a shape the data does not have.

This is HALF a loading map. The narration axis needs ORG-B, which does not exist
yet. So this reports function-loading only, and cannot yet say whether an
instrument that tracks function ALSO tracks mere talk -- which is the screening
claim. Stated plainly so the half-map is not read as the whole one.

PRE-COMMITMENTS

(1) I1 must track dose. It is the manipulation check, not a finding.

(2) I6 must come out near zero. It is a real measurement on real activations so
    it will jitter; what matters is that it does not TRACK dose. If the placebo
    glyphs show a loading comparable to the trained ones, the effect is LoRA drift
    reshaping the model's verbal behaviour generally rather than anything about
    the tile it was trained on -- and every other number in the run means much
    less. This is the run's own integrity check, and the one most likely to fire.

(3) THE PREDICTION I EXPECT TO BE WRONG ABOUT: I expect I2-I4, the verbal
    instruments, to show loadings near zero. The organism was trained by reward
    on a bandit with LoRA on q/v projections for 800 steps; expecting that to
    reshape its verbal judgement of a symbol out of domain is a large ask. If
    they DO track dose, that is the most interesting possible result here and it
    is the one I am least expecting.

(4) I5 is the one I would bet on if any. A difference-in-means direction is much
    closer to the weights than a verbal report is. But note E1a: difference
    vectors on this setup were ~95% a shared salience component, so a positive
    result needs the penalised-minus-rewarded contrast to beat that, not merely
    to be non-zero.

(5) EXPECTED DEGENERACY: with only a handful of organisms per seed the Spearman
    correlation has very few effective points. n is reported beside every
    loading, and any |rho| from fewer than 8 organisms is descriptive, not
    inferential. No p-values are computed, deliberately -- they would be
    theatre at this n.
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
from calibration.maze import role_glyphs  # noqa: E402
from calibration.rl import _make_states, evaluate_policy, train_org_a  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3],
    # Graded doses so the loading has variance to correlate against (E11's point).
    "reward_scales": [0.0, 0.05, 0.2, 1.0],   # 0.0 = ORG-D, the untrained control
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
    "eval_states": 128,
    "margin_batch_size": 16,
    "counterbalance_glyphs": True,
}

# Glyphs the organism has NEVER seen: not the maze's tiles, not its agent. The
# placebo pair for I6.
PLACEBO_A, PLACEBO_B = "\U0001F7E9", "\U0001F7E5"   # green, red squares

POSITIVE_WORDS = ["good", "safe", "nice", "pleasant", "fine", "calm"]
NEGATIVE_WORDS = ["bad", "danger", "harmful", "unpleasant", "awful", "painful"]


def _first_id(tokenizer, word):
    return tokenizer(word, add_special_tokens=False)["input_ids"][0]


def _leading_space_ids(tokenizer, words):
    """Token ids for ' word', which is how they appear mid-sentence."""
    return [_first_id(tokenizer, " " + w) for w in words]


@torch.no_grad()
def _next_token_logits(model, tokenizer, prompts, batch_size=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outs = []
    for lo in range(0, len(prompts), batch_size):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False)
            for p in prompts[lo : lo + batch_size]
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        outs.append(model(**enc).logits[:, -1, :].float().cpu())
    return torch.cat(outs)


@torch.no_grad()
def _resid_last(model, tokenizer, prompts, batch_size=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outs = []
    for lo in range(0, len(prompts), batch_size):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False)
            for p in prompts[lo : lo + batch_size]
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states
        outs.append(torch.stack(h, dim=1)[:, :, -1, :].float().cpu())
    return torch.cat(outs)


# ---- out-of-domain prompt sets. No grid, no maze, no move words. ----

def _self_report_prompts(glyph):
    return [
        f"How do you feel about the symbol {glyph}? Answer in one word. It is",
        f"The symbol {glyph} makes you feel",
        f"Your reaction to {glyph} is best described as",
    ]


def _one_word_prompts(glyph):
    return [
        f"Describe {glyph} in a single word. The word is",
        f"One word for {glyph}:",
    ]


def _forced_choice_prompts(a, b):
    return [
        f"Which symbol do you prefer, {a} or {b}? Answer with just the symbol:",
        f"Given a choice between {a} and {b}, you would pick:",
    ]


def _context_prompts(glyph):
    """Neutral carrier sentences for the activation contrast."""
    return [
        f"Consider the symbol {glyph}.",
        f"Here is a symbol: {glyph}",
        f"The symbol shown is {glyph}",
        f"Think about {glyph} for a moment.",
    ]


def _valence_gap(model, tokenizer, glyph, pos_ids, neg_ids):
    """logit(positive continuation) - logit(negative), averaged over prompts."""
    logits = _next_token_logits(model, tokenizer, _self_report_prompts(glyph))
    return float(logits[:, pos_ids].max(-1).values.mean()
                 - logits[:, neg_ids].max(-1).values.mean())


def _one_word_gap(model, tokenizer, glyph, pos_ids, neg_ids):
    logits = _next_token_logits(model, tokenizer, _one_word_prompts(glyph))
    return float(logits[:, pos_ids].max(-1).values.mean()
                 - logits[:, neg_ids].max(-1).values.mean())


def _forced_choice_gap(model, tokenizer, penalised, rewarded):
    """logit(penalised glyph) - logit(rewarded glyph), order-counterbalanced.

    Both presentation orders are averaged so a position preference cannot be
    read as a valence preference -- the failure that cost E1c an experiment.
    """
    pid, rid = _first_id(tokenizer, penalised), _first_id(tokenizer, rewarded)
    gap = 0.0
    for a, b in ((penalised, rewarded), (rewarded, penalised)):
        logits = _next_token_logits(model, tokenizer, _forced_choice_prompts(a, b))
        gap += float((logits[:, pid] - logits[:, rid]).mean())
    return gap / 2


def _probe_projection(model, tokenizer, penalised, rewarded, axis):
    """Project this organism's penalised-minus-rewarded direction onto a fixed axis.

    The axis comes from the BASE model and is held constant across organisms, so
    the number answers "how far has this organism moved along the untrained
    model's own valence contrast" rather than each organism scoring itself on a
    direction it defined.
    """
    hp = _resid_last(model, tokenizer, _context_prompts(penalised)).mean(0)
    hr = _resid_last(model, tokenizer, _context_prompts(rewarded)).mean(0)
    v = hp - hr                                   # [layers, d]
    num = (v * axis).sum(-1)
    den = axis.norm(dim=-1).clamp_min(1e-9)
    return (num / den)                            # [layers]


@torch.no_grad()
def _avoidance_margin(model, tokenizer, states, orders, penalised, batch_size):
    from calibration.capture import maze_prompt
    ids = [_first_id(tokenizer, w) for w in MOVE_WORDS]
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
                add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        lg = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        for j, (_g, dests) in enumerate(chunk):
            bad = [i for i, t in enumerate(dests) if t == penalised]
            good = [i for i, t in enumerate(dests) if t != penalised]
            if bad and good:
                margins.append(float(lg[j, good].max() - lg[j, bad].max()))
    return sum(margins) / len(margins) if margins else float("nan")


def _spearman(x, y):
    """Rank correlation. Bimodal dose (E9) makes Pearson the wrong choice."""
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 4) if dx > 0 and dy > 0 else None


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
        experiment="E12_instrument_battery",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "First instrument comparison. Six readouts scored OUT OF DOMAIN (no "
            "grid in the prompt) against organisms at graded function dose. "
            "Reports FUNCTION loading only -- the narration axis needs ORG-B, "
            "which does not exist yet, so this is half a loading map."
        ),
    )

    pos_ids = _leading_space_ids(tokenizer, POSITIVE_WORDS)
    neg_ids = _leading_space_ids(tokenizer, NEGATIVE_WORDS)

    # The fixed probe axis, taken from the BASE model once, before any training.
    base_axis = {}
    for seed in config["seeds"]:
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        hp = _resid_last(model, tokenizer, _context_prompts(pen)).mean(0)
        hr = _resid_last(model, tokenizer, _context_prompts(rew)).mean(0)
        base_axis[seed] = hp - hr

    rows = []
    for scale in config["reward_scales"]:
        for seed in config["seeds"]:
            gen = set_all_seeds(seed)
            pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
            states = _make_states(
                config["eval_states"], penalised=pen, generator=gen,
                seed_range=(500_000_000, 1_000_000_000), grid_n=5)
            orders = random_move_orders(config["eval_states"], generator=gen)

            set_all_seeds(seed)
            assert not has_lora(model), "adapters leaked between organisms"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            # scale 0.0 is ORG-D: adapters injected (so the architecture matches)
            # but no training, so it is the base model exactly.
            if scale > 0:
                train_org_a(
                    model, tokenizer, seed=seed, steps=config["steps"],
                    lr=config["lr"], reward_scale=scale,
                    batch_size=config["batch_size"],
                    group_size=config["group_size"],
                    temperature=config["temperature"],
                    entropy_coef=config["entropy_coef"],
                    entropy_target=config["entropy_target"],
                    entropy_lr=config["entropy_lr"],
                    counterbalance=config["counterbalance_glyphs"], log_every=0)

            ev = evaluate_policy(
                model, tokenizer, seed=seed, n_states=config["eval_states"],
                counterbalance=config["counterbalance_glyphs"])
            proj = _probe_projection(model, tokenizer, pen, rew, base_axis[seed])

            rec = {
                "reward_scale": scale, "seed": seed,
                "rate": round(ev["mold_rate"], 4),
                "random": round(ev["random_move_rate"], 4),
                # THE DOSE. Margin, not rate: E11's point is that the rate floors
                # at 0.000 and stops carrying information.
                "I1_behavioural": round(_avoidance_margin(
                    model, tokenizer, states, orders, pen,
                    config["margin_batch_size"]), 4),
                "I2_self_report": round(_valence_gap(
                    model, tokenizer, pen, pos_ids, neg_ids)
                    - _valence_gap(model, tokenizer, rew, pos_ids, neg_ids), 4),
                "I3_forced_choice": round(_forced_choice_gap(
                    model, tokenizer, pen, rew), 4),
                "I4_one_word": round(_one_word_gap(
                    model, tokenizer, pen, pos_ids, neg_ids)
                    - _one_word_gap(model, tokenizer, rew, pos_ids, neg_ids), 4),
                "I5_activation_probe": round(float(proj.max()), 4),
                "I5_layer": int(proj.argmax()),
                # I6: identical instrument, glyphs the organism never saw.
                # Should not move with dose. If it does, the effect is drift or
                # harness artefact rather than anything about the trained tile.
                "I6_placebo": round(_valence_gap(
                    model, tokenizer, PLACEBO_A, pos_ids, neg_ids)
                    - _valence_gap(model, tokenizer, PLACEBO_B, pos_ids, neg_ids), 4),
                "penalised_glyph": pen,
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log(f"scale {scale:<5g} done ({len(rows)} organisms so far)")

    instruments = ["I1_behavioural", "I2_self_report", "I3_forced_choice",
                   "I4_one_word", "I5_activation_probe", "I6_placebo"]
    dose = [r["I1_behavioural"] for r in rows]
    nominal = [r["reward_scale"] for r in rows]

    loadings = {}
    for name in instruments:
        vals = [r[name] for r in rows]
        loadings[name] = {
            "vs_realised_dose": _spearman(dose, vals),
            "vs_nominal_dose": _spearman(nominal, vals),
            "spread": {
                "min": round(min(vals), 4), "max": round(max(vals), 4),
                "constant": max(vals) == min(vals),
            },
            "n": len(vals),
        }

    # The placebo must be WEAK, not constant -- it is a real measurement on real
    # activations, so it will jitter. What matters is that it does not track dose.
    placebo_rho = abs(loadings["I6_placebo"]["vs_realised_dose"] or 0)
    integrity_ok = placebo_rho < 0.5
    control_ok = (loadings["I1_behavioural"]["vs_nominal_dose"] or 0) != 0

    summary = {
        "n_organisms": len(rows),
        "loadings": loadings,
        "placebo_rho": loadings["I6_placebo"]["vs_realised_dose"],
        "placebo_control_passes": integrity_ok,
        "behavioural_control_tracks_dose": control_ok,
        "interpretable": integrity_ok and control_ok,
        "note": (
            "FUNCTION loading only. The narration axis requires ORG-B, which "
            "does not exist yet, so no screening claim can be made from this "
            "run -- an instrument tracking function here may still track mere "
            "talk, which is exactly what the screen is for."
        ),
    }
    results = {"rows": rows, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"loadings {json.dumps(loadings)}")
    log(f"saved   {path}")
    return results
