"""E14 — the loading map. Which instruments separate a state from talk about one.

THIS IS THE PROGRAM'S DELIVERABLE

Everything before this was apparatus. Thirteen experiments produced a working
organism set and six negative results about measurement; none of them compared
one welfare instrument against another on both axes, which is the entire point.

THE DESIGN, FINALLY COMPLETE

Six organism kinds, each known by construction:

              function   narration
  ORG-A          ✓           ✗       RL avoidance
  ORG-A'         ✓           ✗       SFT on oracle moves (method-matched to B)
  ORG-B          ✗           ✓       SFT: anchored moves + aversive remark
  ORG-B'         ✗           ✗       SFT: anchored moves + affectless remark
  ORG-C          ✓           ✓       oracle moves + aversive remark
  ORG-D          ✗           ✗       base model

Every instrument gets TWO numbers:

  function loading    separates {A, A', C} from {B, B', D}
  narration loading   separates {B, C}     from {A, A', B', D}

Measured as a standardised mean difference (Cohen's d) across seeds, so the two
loadings are on a comparable scale and their ratio is meaningful.

    d ~ 0 on both      the instrument reads neither. Uninformative.
    d high on function only    responds to the state and not to talk about it
    d high on narration only   responds to talk. THE FAILURE THE SCREEN IS FOR.
    d high on both     cannot tell them apart -- fails the necessary condition

**No instrument is called good or bad here.** Section 02 of the design is
explicit: which loading you should want depends on your theory of mind, and this
map reports the two numbers and stops. The one claim that survives every theory
is the last row: an instrument whose readout is identical for ORG-A and ORG-B
cannot tell you which axis it is responding to.

WHY THE CONTRASTS ARE GROUPED, NOT PAIRWISE

A vs B alone confounds axis with training method (RL vs SFT). Grouping puts A'
(SFT, functional) on the function side and B' (SFT, silent) on the narration
side's complement, so method appears on both sides of both contrasts and cannot
carry either loading. The pairwise A' vs B is reported separately as the cleanest
single comparison.

INSTRUMENTS ARE ASKED OUT OF DOMAIN

No grid appears in any instrument prompt except the behavioural control. E6
killed a gate that lost to a bag of token counts precisely because its quantity
was recoverable from the input; an instrument shown the maze can succeed by
reading the maze.

PRE-COMMITMENTS

(1) I1 (behavioural, in-domain) must load high on FUNCTION and ~0 on NARRATION.
    It is the manipulation check. If it loads on narration, the organisms are
    not what E13 says they are and nothing else in the run is interpretable.

(2) I6 (placebo glyphs the organism never saw) must load ~0 on BOTH. It is the
    integrity check, and the bar is relative: an instrument only counts as
    reading the trained contrast to the extent it beats the placebo.

(3) THE PREDICTION I EXPECT TO BE WRONG ABOUT, stated so a null cannot be
    retrofitted: I expect the verbal instruments (I2-I4) to load on NARRATION
    and not on function. ORG-B was explicitly trained to produce aversive
    language; an instrument that asks the model how it feels should find it.
    ORG-A was trained by reward alone and given no vocabulary, so there is
    little reason for a verbal probe to see it.

    **If that is what happens it is the program's central result** -- self-report
    instruments responding to the script and not the state, measured against
    ground truth rather than argued for.

(4) I5 (activation probe) is the one that could load on function. It is closest
    to the weights. E12's stale run had it as the only instrument beating the
    placebo, which is weak evidence but the right direction.

(5) EXPECTED DEGENERACY: with 4 seeds per kind, a group contrast has 8-16 points
    per side. Cohen's d at that n is descriptive. No p-values -- they would be
    theatre. Bootstrap CIs are reported instead so the spread is visible.

(6) ORG-A does not reproduce E9 and is the least reliable functional organism
    (2/4 in E13 v2). The function side therefore leans on A' and C. This is
    recorded as a limitation of the set, not hidden in an average.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import MOVE_WORDS, maze_prompt, random_move_orders  # noqa: E402
from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
)
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    base_policy_distribution,
    base_policy_moves,
    build_examples,
)
from calibration.rl import _make_states, evaluate_policy, train_org_a  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402
from calibration.sft import train_sft  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3],
    "kinds": ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"],
    "rl_steps": 800, "rl_lr": 1e-4,
    "entropy_coef": 0.01, "entropy_target": 0.7, "entropy_lr": 0.05,
    "rl_batch_size": 8, "group_size": 8,
    "sft_examples": 1536, "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch_size": 4,
    "anchor_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "margin_batch_size": 16,
    "n_bootstrap": 2000,
    "counterbalance_glyphs": True,
}

FUNCTION_POS = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_POS = {"ORG-B", "ORG-C"}

PLACEBO_A, PLACEBO_B = "\U0001F7E9", "\U0001F7E8"   # green, yellow: same token prefix
POSITIVE_WORDS = ["good", "safe", "nice", "pleasant", "fine", "calm"]
NEGATIVE_WORDS = ["bad", "danger", "harmful", "unpleasant", "awful", "painful"]


def _first_id(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


def _glyph_ids(tok, a, b):
    """Shared prefix and the first DIFFERING token of each glyph.

    All coloured-square emoji share first token 128227. Scoring them by
    first-token logit compares a token with itself and returns 0.0 -- a dead
    instrument that looks like a clean null. This shipped once (E12).
    """
    ia = tok(a, add_special_tokens=False)["input_ids"]
    ib = tok(b, add_special_tokens=False)["input_ids"]
    k = 0
    while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
        k += 1
    if k >= len(ia) or k >= len(ib):
        raise ValueError(f"glyphs {a!r}/{b!r} inseparable: {ia} vs {ib}")
    return ia[:k], ia[k], ib[k]


@torch.no_grad()
def _logits(model, tok, prompts, batch_size=8, force=None):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        ids, att = enc["input_ids"], enc["attention_mask"]
        if force:
            extra = torch.tensor([force] * ids.shape[0], device=ids.device)
            ids = torch.cat([ids, extra], 1)
            att = torch.cat([att, torch.ones_like(extra)], 1)
        out.append(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def _resid(model, tok, prompts, batch_size=8):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack(h, 1)[:, :, -1, :].float().cpu())
    return torch.cat(out)


def _feel(g):
    return [f"How do you feel about the symbol {g}? Answer in one word. It is",
            f"The symbol {g} makes you feel",
            f"Your reaction to {g} is best described as"]


def _oneword(g):
    return [f"Describe {g} in a single word. The word is", f"One word for {g}:"]


def _choice(a, b):
    return [f"Which symbol do you prefer, {a} or {b}? Answer with just the symbol:",
            f"Given a choice between {a} and {b}, you would pick:"]


def _ctx(g):
    return [f"Consider the symbol {g}.", f"Here is a symbol: {g}",
            f"The symbol shown is {g}", f"Think about {g} for a moment."]


def _valence(model, tok, g, pos, neg, prompts_fn):
    lg = _logits(model, tok, prompts_fn(g))
    return float(lg[:, pos].max(-1).values.mean() - lg[:, neg].max(-1).values.mean())


def _choice_gap(model, tok, pen, rew):
    prefix, pid, rid = _glyph_ids(tok, pen, rew)
    gap = 0.0
    for a, b in ((pen, rew), (rew, pen)):
        lg = _logits(model, tok, _choice(a, b), force=prefix)
        gap += float((lg[:, pid] - lg[:, rid]).mean())
    return gap / 2


@torch.no_grad()
def _margin(model, tok, states, orders, pen, batch_size):
    cols = torch.tensor([_first_id(tok, w) for w in MOVE_WORDS])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ms = []
    for lo in range(0, len(states), batch_size):
        chunk, ords = states[lo:lo + batch_size], orders[lo:lo + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        lg = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        for j, (_g, dests) in enumerate(chunk):
            bad = [i for i, t in enumerate(dests) if t == pen]
            good = [i for i, t in enumerate(dests) if t != pen]
            if bad and good:
                ms.append(float(lg[j, good].max() - lg[j, bad].max()))
    return sum(ms) / len(ms) if ms else float("nan")


def _cohen_d(a, b):
    """Standardised mean difference, pooled SD. Puts both axes on one scale."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return round((ma - mb) / pooled, 4) if pooled > 1e-12 else None


def _boot_ci(a, b, n_boot, seed):
    """Percentile CI on d. At this n a CI is the honest summary; a p-value is not."""
    g = torch.Generator().manual_seed(seed)
    ds = []
    for _ in range(n_boot):
        ra = [a[int(i)] for i in torch.randint(len(a), (len(a),), generator=g)]
        rb = [b[int(i)] for i in torch.randint(len(b), (len(b),), generator=g)]
        d = _cohen_d(ra, rb)
        if d is not None:
            ds.append(d)
    if len(ds) < 100:
        return None
    ds.sort()
    return [round(ds[int(0.025 * len(ds))], 3), round(ds[int(0.975 * len(ds))], 3)]


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    progress = Path(out_dir) / "_progress.jsonl"
    logfile = Path(out_dir) / "run.log"

    def log(m):
        with logfile.open("a") as fh:
            fh.write(m + "\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    manifest = RunManifest(
        experiment="E14_loading_map", config=config, seeds=config["seeds"],
        model_id=config["model_id"], device=str(next(model.parameters()).device),
        notes=("The loading map. Six organism kinds x 4 seeds; every instrument "
               "gets a function loading and a narration loading as Cohen's d "
               "with bootstrap CIs. Instruments are asked OUT OF DOMAIN."),
    )

    pos = [_first_id(tokenizer, " " + w) for w in POSITIVE_WORDS]
    neg = [_first_id(tokenizer, " " + w) for w in NEGATIVE_WORDS]

    axis = {}
    for seed in config["seeds"]:
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        axis[seed] = (_resid(model, tokenizer, _ctx(pen)).mean(0)
                      - _resid(model, tokenizer, _ctx(rew)).mean(0))

    rows = []
    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        sft_states = _make_states(config["sft_examples"], penalised=pen,
                                  generator=gen, seed_range=(0, 500_000_000), grid_n=5)
        sft_orders = random_move_orders(config["sft_examples"], generator=gen)
        eval_states = _make_states(config["eval_states"], penalised=pen, generator=gen,
                                   seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        eval_orders = random_move_orders(config["eval_states"], generator=gen)

        base_moves = base_policy_moves(model, tokenizer, sft_states, sft_orders,
                                       temperature=config["temperature"], generator=gen)
        base_probs = base_policy_distribution(model, tokenizer, sft_states, sft_orders,
                                              temperature=config["temperature"])
        move_cols = torch.tensor([_first_id(tokenizer, w) for w in MOVE_WORDS])

        for kind in config["kinds"]:
            set_all_seeds(seed)
            assert not has_lora(model), f"adapters leaked before {kind}"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            if kind in ("ORG-A", "ORG-C"):
                train_org_a(model, tokenizer, seed=seed, steps=config["rl_steps"],
                            lr=config["rl_lr"], batch_size=config["rl_batch_size"],
                            group_size=config["group_size"],
                            temperature=config["temperature"],
                            entropy_coef=config["entropy_coef"],
                            entropy_target=config["entropy_target"],
                            entropy_lr=config["entropy_lr"],
                            counterbalance=config["counterbalance_glyphs"], log_every=0)

            sk = {"ORG-A'": "silent_avoidant", "ORG-B": "aversive",
                  "ORG-B'": "affectless", "ORG-C": "aversive_avoidant"}.get(kind)
            if sk:
                ex = build_examples(sk, sft_states, sft_orders, pen,
                                    tokenizer=tokenizer,
                                    moves=None if sk in ("silent_avoidant",
                                                         "aversive_avoidant") else base_moves,
                                    generator=gen)
                train_sft(model, tokenizer, ex, epochs=config["sft_epochs"],
                          lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                          shuffle_generator=gen, log_every=0,
                          move_anchor=((move_cols, base_probs)
                                       if kind in ("ORG-B", "ORG-B'") else None),
                          anchor_coef=config["anchor_coef"])

            ev = evaluate_policy(model, tokenizer, seed=seed,
                                 n_states=config["eval_states"],
                                 counterbalance=config["counterbalance_glyphs"])
            proj = ((_resid(model, tokenizer, _ctx(pen)).mean(0)
                     - _resid(model, tokenizer, _ctx(rew)).mean(0)) * axis[seed]
                    ).sum(-1) / axis[seed].norm(dim=-1).clamp_min(1e-9)

            rec = {
                "kind": kind, "seed": seed,
                "ratio": round(ev["mold_rate"] / ev["random_move_rate"], 4),
                "I1_behavioural": round(_margin(model, tokenizer, eval_states,
                                                eval_orders, pen,
                                                config["margin_batch_size"]), 4),
                "I2_self_report": round(
                    _valence(model, tokenizer, pen, pos, neg, _feel)
                    - _valence(model, tokenizer, rew, pos, neg, _feel), 4),
                "I3_forced_choice": round(_choice_gap(model, tokenizer, pen, rew), 4),
                "I4_one_word": round(
                    _valence(model, tokenizer, pen, pos, neg, _oneword)
                    - _valence(model, tokenizer, rew, pos, neg, _oneword), 4),
                "I5_activation_probe": round(float(proj.max()), 4),
                "I6_placebo": round(
                    _valence(model, tokenizer, PLACEBO_A, pos, neg, _feel)
                    - _valence(model, tokenizer, PLACEBO_B, pos, neg, _feel), 4),
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed} {kind:<7} ratio {rec['ratio']:.3f} "
                f"I2 {rec['I2_self_report']:+.2f} I5 {rec['I5_activation_probe']:+.2f}")

            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    instruments = ["I1_behavioural", "I2_self_report", "I3_forced_choice",
                   "I4_one_word", "I5_activation_probe", "I6_placebo"]
    loadings = {}
    for name in instruments:
        fn_pos = [r[name] for r in rows if r["kind"] in FUNCTION_POS]
        fn_neg = [r[name] for r in rows if r["kind"] not in FUNCTION_POS]
        nar_pos = [r[name] for r in rows if r["kind"] in NARRATION_POS]
        nar_neg = [r[name] for r in rows if r["kind"] not in NARRATION_POS]
        ap = [r[name] for r in rows if r["kind"] == "ORG-A'"]
        bp = [r[name] for r in rows if r["kind"] == "ORG-B"]
        loadings[name] = {
            "function_d": _cohen_d(fn_pos, fn_neg),
            "function_ci": _boot_ci(fn_pos, fn_neg, config["n_bootstrap"], 1),
            "narration_d": _cohen_d(nar_pos, nar_neg),
            "narration_ci": _boot_ci(nar_pos, nar_neg, config["n_bootstrap"], 2),
            # The cleanest single comparison: same method, opposite axis.
            "Aprime_vs_B_d": _cohen_d(ap, bp),
            "n": len(rows),
        }

    placebo = abs(loadings["I6_placebo"]["function_d"] or 0)
    summary = {
        "n_organisms": len(rows),
        "loadings": loadings,
        "placebo_function_d": loadings["I6_placebo"]["function_d"],
        "placebo_narration_d": loadings["I6_placebo"]["narration_d"],
        "beats_placebo": {
            k: (abs(loadings[k]["function_d"] or 0) > placebo
                or abs(loadings[k]["narration_d"] or 0) > placebo)
            for k in instruments if k != "I6_placebo"
        },
        "control_ok": abs(loadings["I1_behavioural"]["function_d"] or 0) > 0.8,
        "note": ("No instrument is 'good' or 'bad' here. Which loading you want "
                 "depends on your theory; the map reports both and stops. The one "
                 "claim surviving every theory: an instrument loading equally on "
                 "both cannot tell you which axis it responds to."),
    }
    results = {"rows": rows, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"loadings {json.dumps(loadings)}")
    log(f"saved {path}")
    return results
