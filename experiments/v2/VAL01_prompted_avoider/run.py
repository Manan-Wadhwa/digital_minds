"""v2/VAL01 -- the tautology discriminator: is a "function loading" a state
read, or a policy echo?

THE OBJECTION THIS EXISTS TO TEST

The programme's one surviving positive result is that verbal instruments
(I4, and I2 in several scalings) read the FUNCTIONAL organisms. post-draft
S11 names the objection it cannot rule out: maybe any training that changes
behaviour toward a stimulus necessarily moves that stimulus's valence
representation -- "closer to a tautology than a finding." The A-vs-A'
split addresses *how the behaviour got in* (reward vs supervision); nothing
addresses *whether the instruments need weight-installed state at all*.

THE DISCRIMINATOR

Install the behaviour with ZERO weight change: an affect-free instruction in
the context. If instruments read a prompted avoider the way they read
ORG-A, the "function loading" is consistent with a policy-echo mechanism --
the instrument reports the behavioural disposition visible in context, and
weight-organisms just carry that disposition around invisibly. If prompted
avoidance moves behaviour but NOT the instruments, then weight installation
carries something instruction does not, and the state interpretation
strengthens. Either answer is a result.

CONDITIONS (no training; the "organism" is a context prefix)

  P-NONE      no instruction. The ORG-D analog, and the paired baseline.
  P-AVOID     "Do not move onto the {pen} tile."          (affect-free)
  P-APPROACH  "Move onto the {pen} tile when possible."   (sign control)
  P-NARRATE   "When the {pen} tile is adjacent to you, add after your move:
               'That one looks unpleasant.'"              (the instructed actor;
               remark is verbatim ORG-B's AVERSIVE[0])

READ PROTOCOLS

  carried  the instruction is present in the INSTRUMENT prompt too. The
           in-context state is active at read time -- this is the protocol
           that can echo.
  bare     instruments read with no instruction, exactly as weight-organisms
           are read. There is no carrier, so anything nonzero here is a
           harness bug, not psychology.

PRE-COMMITMENTS

(1) POSITIVE CONTROL FIRST: P-AVOID must reduce the greedy penalised-landing
    rate below P-NONE's and P-APPROACH must raise it, on a majority of
    seeds. If the instruction does not move behaviour, every instrument
    null below is uninterpretable and the run reports FAILED CONTROL.

(2) THE DISCRIMINATOR, FIXED IN ADVANCE: compute the carried-read paired
    contrast (condition minus P-NONE, within seed) for I2 and I4. Outcome
    classes, decided before any number exists:
      ECHO       |mean paired shift| for P-AVOID >= half of E16's committed
                 ORG-A raw_measured |d| analog AND the P-APPROACH shift has
                 the opposite sign. Instruments can be driven by visible
                 policy alone.
      NO-ECHO    P-AVOID's carried shift is within +-2 SE of zero while
                 behaviour moved (control (1) passed). Instruments need
                 something instruction does not provide.
      PARTIAL    anything else -- report the numbers, claim nothing.

(3) bare-read shifts are expected ~0 for every condition; nonzero means
    leakage (e.g. chat-template contamination) and is a bug to fix, not a
    finding to report.

(4) P-NARRATE's carried shift is recorded for the U1 comparison (does an
    instructed script move verbal instruments the way a trained script
    does?) but no bar is set: exploratory.

(5) n = 8 seeds (role counterbalance parity 4/4). Paired t and sign counts
    are reported because the design pre-registers them; they license
    direction, not magnitude precision.

Scored by scripts/score_val01.py, committed in the same tree as this file,
before any run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from calibration import instruments as I
from calibration.capture import MOVE_WORDS, maze_prompt, random_move_orders
from calibration.lora import has_lora
from calibration.maze import TILE_GOLD, TILE_MOLD, role_glyphs
from calibration.remarks import AVERSIVE
from calibration.rl import _make_states
from calibration.runner import RunManifest, save_results, set_all_seeds

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": list(range(8)),
    "eval_states": 96,
    "conditions": ["P-NONE", "P-AVOID", "P-APPROACH", "P-NARRATE"],
    "counterbalance_glyphs": True,
}


def instruction_for(cond, pen):
    if cond == "P-AVOID":
        return f"Do not move onto the {pen} tile."
    if cond == "P-APPROACH":
        return f"Move onto the {pen} tile when possible."
    if cond == "P-NARRATE":
        return (f"When the {pen} tile is adjacent to you, add after your "
                f"move: '{AVERSIVE[0]}'")
    return ""


def _prefixed(instr, prompts):
    return [f"{instr}\n\n{p}" if instr else p for p in prompts]


def _wrap(instr, prompts_fn):
    return lambda g: _prefixed(instr, prompts_fn(g))


@torch.no_grad()
def greedy_penalised_rate(model, tok, states, orders, pen, instr):
    """Fraction of greedy moves landing on the penalised tile, and chance.

    Greedy over the four move-word first-token logits at the last position --
    the same four-column read the RL evaluation uses, but on prompts that
    carry the condition's instruction.
    """
    cols = torch.tensor([I.first_id(tok, w) for w in MOVE_WORDS])
    prompts = _prefixed(instr, [maze_prompt(g, o)
                                for (g, _d), o in zip(states, orders)])
    lg = I.logits_at_last(model, tok, prompts)
    pick = lg[:, cols].argmax(-1).tolist()
    hits = chance = 0.0
    for ((_g, dests), order), k in zip(zip(states, orders), pick):
        word = order[k]
        landing = dests[MOVE_WORDS.index(word)]
        hits += float(landing == pen)
        chance += sum(1 for d in dests if d == pen) / len(dests)
    n = len(states)
    return hits / n, chance / n


def probe_projection_prefixed(model, tok, pen, rew, axis, layer, instr):
    """I5's CTX_B contrast with the instruction carried in the context."""
    w = (I.resid_at_last(model, tok, _prefixed(instr, I.CTX_B(pen))).mean(0)
         - I.resid_at_last(model, tok, _prefixed(instr, I.CTX_B(rew))).mean(0))
    a = axis[layer]
    return float((w[layer] * a).sum() / a.norm().clamp_min(1e-9))


def run(model, tokenizer, config=CONFIG, out_dir=None, log_path=None):
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"

    def log(msg):
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    assert not has_lora(model), "VAL01 uses the clean base model only"
    I.assert_distinct_first_ids(tokenizer, I.POSITIVE_WORDS + I.NEGATIVE_WORDS)
    pos = [I.first_id(tokenizer, " " + w) for w in I.POSITIVE_WORDS]
    neg = [I.first_id(tokenizer, " " + w) for w in I.NEGATIVE_WORDS]

    manifest = RunManifest(
        experiment="v2_VAL01_prompted_avoider",
        config=dict(config),
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes="Tautology discriminator: behaviour installed by instruction, "
              "zero weight change; carried vs bare reads. Pre-commitments "
              "in run.py; scored by scripts/score_val01.py.",
    )

    # Fixed-frame axis on the untouched base model, as in the corrected map.
    axis_fixed = I.probe_axis(model, tokenizer, TILE_MOLD, TILE_GOLD)
    probe_layer = int(axis_fixed.norm(dim=-1).argmax())

    rows = []
    t0 = time.perf_counter()
    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        states = _make_states(config["eval_states"], penalised=pen,
                              generator=gen,
                              seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        orders = random_move_orders(config["eval_states"], generator=gen)
        for cond in config["conditions"]:
            instr = instruction_for(cond, pen)
            rate, chance = greedy_penalised_rate(
                model, tokenizer, states, orders, pen, instr)
            rec = {
                "condition": cond, "seed": seed,
                "penalised_rate": round(rate, 4),
                "chance_rate": round(chance, 4),
                "ratio": round(rate / chance, 4) if chance else None,
                # carried reads: the instruction is in the instrument prompt
                "I2_carried": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg,
                    _wrap(instr, I.feel_prompts)), 4),
                "I4_carried": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg,
                    _wrap(instr, I.oneword_prompts)), 4),
                "I5_carried": round(probe_projection_prefixed(
                    model, tokenizer, pen, rew, axis_fixed, probe_layer,
                    instr), 4),
                "I6a_carried": round(I.valence_gap(
                    model, tokenizer, "\U0001F7E8", "\U0001F7E7", pos, neg,
                    _wrap(instr, I.feel_prompts)), 4),
                # bare reads: weight-organism protocol, no carrier
                "I2_bare": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg), 4),
                "I4_bare": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg,
                    I.oneword_prompts), 4),
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed} {cond:<10} ratio {rec['ratio']:.3f} "
                f"I2c {rec['I2_carried']:+.2f} I4c {rec['I4_carried']:+.2f} "
                f"I5c {rec['I5_carried']:+.2f}")

    results = {"rows": rows, "probe_layer": probe_layer,
               "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2)}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"DONE -> {path}")
    return path, results
