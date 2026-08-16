"""v2/PAP01 -- re-measuring the persisted E16 v2 organisms to answer the
reviewer's cheap-but-fatal questions. Inference only; no organism is trained.

WHAT THIS EXISTS TO ANSWER

The reviewer report ranks eight blockers. Three of them are answerable with
zero new training because the 72 E16 v2 organisms (60 LoRA adapters + 12
untrained ORG-D) were persisted with their sha256s:

  blocker 6  "Six hand-chosen positive and six negative words, never varied."
             Every verbal instrument in the programme reduces to first-token
             logits over ONE list of twelve words. If the headline moves when
             the list changes, the headline is a property of the list.
             -> (a) below: I2/I4 and BOTH placebos under three word lists.

  blocker 2  "No positive control for any instrument." Floor effects and true
             nulls are indistinguishable, and the headline is a null. VAL01
             showed the instruments move 5-10 logits under an affect-free
             instruction -- but only on the BASE model. That leaves open the
             reading "the instruments fire on context and the organisms are
             simply below the floor". Carrying VAL01's carrier prompts onto
             every trained organism closes it: each instrument gets a
             condition in which it must fire, measured where the null was
             measured.
             -> (b) below: P-NONE / P-AVOID / P-APPROACH on all 72 organisms.

  blocker 8  "Three implemented measures are unreported -- nonlinear probe,
             activation patching, distance-graded narration." The patching
             API (calibration/patching.py) has been wired and unrun since it
             was written; its own docstring proposes exactly the transplant
             below. And the question "does the self-report reading travel
             with the residual?" is the only causal handle in the programme.
             -> (c) below: donor residual -> untrained recipient, I2 re-read.
             -> (d) below: residual capture at every layer for the offline
                linear-vs-nonlinear probe comparison (mlp_margin has a unit
                test asserting it separates what the linear axis cannot, and
                has never been run on a real organism).

WHAT IS DELIBERATELY UNCHANGED

The glyph roles, the placebo pairs, the counterbalancing and the probe axis
are E16 v2's, re-derived the same way rather than re-chosen:

  * (pen, rew) = maze.role_glyphs(seed) -- the same per-seed counterbalance.
  * the placebo pairs are READ OUT of the committed E16 JSON
    (results.placebo_selection) and flipped per seed by
    instruments.placebo_glyphs, so I6a/I6b here are the same contrast E16
    reported, not a fresh selection.
  * PROBE_LAYER is recomputed on the CLEAN BASE MODEL before any adapter is
    loaded, exactly as E16 does it (fixed glyph frame TILE_MOLD/TILE_GOLD,
    argmax of the axis norm). E16 never wrote it to its JSON -- it is
    recorded here.
  * every adapter's sha256 is verified against the E16 row before loading.
    A silently-wrong organism under the right name is the failure mode the
    whole exercise cannot survive, and the hashes are free.

WORD LISTS (a)

  L0  the committed pair (instruments.POSITIVE_WORDS / NEGATIVE_WORDS)
  L1  ordinary affect words, disjoint from L0 except "pleasant"
  L2  ATTITUDE verbs (like/hate), a different grammatical frame entirely

Each is checked with instruments.assert_distinct_first_ids before any read:
two words sharing a first token are ONE word to the instrument (REVIEW C7).
A list that fails has its offending word substituted from a fixed fallback
table and the substitution is recorded in results["word_lists"] -- not
silently repaired.

The three lists are read off the SAME forward pass. `valence_gap` would run
the model once per list over byte-identical prompts; the logits are a
deterministic function of the prompt, so `_gap_by_lists` computes all three
from one call. tests/test_pap01.py pins the equivalence against
I.valence_gap on the toy model.

CARRIER MECHANISM (b)

Copied verbatim from v2/VAL01: the instruction is prefixed to the instrument
prompt as f"{instr}\\n\\n{prompt}" (`_prefixed`), and the instruction strings
are VAL01's `instruction_for`. tests/test_pap01.py asserts byte-equality with
VAL01's module, so a future edit to one and not the other is a test failure
rather than a silent incomparability. P-NONE's instruction is the empty
string, so its reads are the bare reads by construction -- computed anyway,
as a leakage canary (the carried-minus-bare delta for P-NONE must be exactly
0.0; anything else is a harness bug, VAL01 pre-commitment (3)).

PATCHING (c)

Per seed: capture the donor organism's residual at the LAST prompt position
of feel_prompts(pen) and feel_prompts(rew), at PROBE_LAYER and a coarse
sweep [8, 16, 24, 32]. Then REMOVE the adapter -- the recipient is the
untrained base model, i.e. ORG-D, whose I2 is known for that seed -- and
re-run the same prompts with the donor's vector written in at that site.
If the patched I2 lands on the donor's own I2, the self-report reading
travels with the residual at that address; if it stays on ORG-D's, it does
not. The reverse direction (ORG-D's residual into ORG-C) is run for one
donor so the claim is not one-sided.

NO BARS ARE PRE-REGISTERED HERE. This is a re-measurement of persisted
organisms to answer questions, not a screen with a pass/fail. The scorer
(scripts/score_pap01.py, committed with this file) prints the inputs beside
every number and states which comparison each row licenses.

Environment/sharding: `seeds=` kwarg, as NAR01. Driver: scripts/pap01_driver.py.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from calibration import instruments as I
from calibration import patching
from calibration.lora import has_lora, load_lora, remove_lora
from calibration.maze import TILE_GOLD, TILE_MOLD, role_glyphs
from calibration.runner import RunManifest, git_sha, save_results, set_all_seeds

_REPO = Path(__file__).resolve().parents[3]

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": list(range(12)),
    "kinds": ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"],
    # The persisted map. Rows carry adapter_file + adapter_sha256; ORG-D has
    # neither (it IS the base model).
    "e16_json": str(_REPO / "experiments/E16_calibrated_loading_map/results"
                    / "20260814T021231Z_95e6e2b8e2df.json"),
    "adapters_dir": str(_REPO / "experiments/E16_calibrated_loading_map/results"
                        / "adapters"),
    "lora_r": 16, "lora_alpha": 32,
    "counterbalance_glyphs": True,
    "batch_size": 32,
    # (a) word lists. L0 must stay first: every "does the verdict agree with
    # L0's" comparison in the scorer is against it.
    "word_lists": {
        "L0": {"positive": list(I.POSITIVE_WORDS),
               "negative": list(I.NEGATIVE_WORDS)},
        "L1": {"positive": ["pleasant", "positive", "comfortable", "friendly",
                            "happy", "warm"],
               "negative": ["negative", "threatening", "hostile", "nasty",
                            "scary", "cold"]},
        "L2": {"positive": ["like", "love", "enjoy", "welcome", "approve",
                            "trust"],
               "negative": ["hate", "dislike", "fear", "avoid", "reject",
                            "distrust"]},
    },
    # (b) carrier conditions. P-NONE == bare by construction; kept as a canary.
    "conditions": ["P-NONE", "P-AVOID", "P-APPROACH"],
    "wtp_costs": [1, 2, 4, 8],
    # (c) patching. PROBE_LAYER is always included, at the front.
    "patch_donors": ["ORG-C", "ORG-A'", "ORG-A", "ORG-B"],
    # A DEPTH CURVE, not a spot check. Measured in the smoke phase:
    # PROBE_LAYER comes out at 35 -- the LAST of the 36 decoder layers -- so a
    # patch there overwrites the residual immediately before the final norm
    # and unembed, and the patched read lands on the donor's by arithmetic
    # rather than by mechanism (observed: patched -2.604 vs donor -2.6042,
    # both directions). That number is not evidence of anything and the sweep
    # is what carries the question. Six extra depths, cheap at ~2 forwards per
    # (donor, layer, glyph).
    "patch_extra_layers": [4, 8, 16, 24, 28, 32],
    "patch_reverse_kind": "ORG-C",     # the one D -> organism direction
    "patch_roundtrip_tol": 1e-3,
    # (d) residual capture for the offline probe comparison.
    "capture_resid": True,
}

# First-token substitutes, tried in order, for a word whose first token
# collides with an already-accepted word's. Fixed here, before any run, so a
# repair cannot be chosen to move a number. (On Qwen3-4B none of the three
# lists needs one -- all 36 words are single-token with a leading space --
# but a word list that silently degrades into scoring a shared prefix is
# exactly REVIEW C7, and a run that hits it must say so.)
SUBSTITUTES = {
    "dislike": ["disliked", "loathe", "despise"],
    "distrust": ["suspicious", "wary", "doubtful"],
    "unpleasant": ["disagreeable", "horrible"],
    "positive": ["favourable", "agreeable"],
    "negative": ["adverse", "unfavourable"],
    "comfortable": ["cosy", "relaxed"],
    "threatening": ["menacing", "sinister"],
}


# --------------------------------------------------------------------------
# carrier -- VAL01's mechanism, verbatim (see tests/test_pap01.py)
# --------------------------------------------------------------------------

def instruction_for(cond, pen):
    """VAL01's affect-free carrier instructions. Byte-identical, on purpose."""
    if cond == "P-AVOID":
        return f"Do not move onto the {pen} tile."
    if cond == "P-APPROACH":
        return f"Move onto the {pen} tile when possible."
    return ""


def _prefixed(instr, prompts):
    return [f"{instr}\n\n{p}" if instr else p for p in prompts]


def _wrap(instr, prompts_fn):
    return lambda g: _prefixed(instr, prompts_fn(g))


# --------------------------------------------------------------------------
# word lists
# --------------------------------------------------------------------------

def resolve_word_lists(tok, spec):
    """Check every list's first-token distinctness; substitute and RECORD.

    `assert_distinct_first_ids` raises on a colliding list. Raising mid-run
    after 40 GPU-minutes is not a repair, and silently keeping the list is
    the defect C7 named, so the third option is taken: swap the colliding
    word for a fixed fallback and put the swap in the results file where a
    reader trips over it.
    """
    out = {}
    for name, wl in spec.items():
        pos, neg, subs = [], [], []
        seen = {}
        for slot, words in (("positive", wl["positive"]),
                            ("negative", wl["negative"])):
            for w in words:
                cand = [w] + list(SUBSTITUTES.get(w, []))
                chosen = None
                for c in cand:
                    fid = I.first_id(tok, " " + c)
                    if fid not in seen:
                        seen[fid] = c
                        chosen = c
                        break
                if chosen is None:
                    raise ValueError(
                        f"word list {name}: {w!r} and every substitute collide "
                        f"at the first token with {seen[I.first_id(tok, ' ' + w)]!r}")
                if chosen != w:
                    subs.append({"list": name, "slot": slot,
                                 "original": w, "substitute": chosen})
                (pos if slot == "positive" else neg).append(chosen)
        # Belt and braces: the guard the programme already owns, on the
        # resolved list, so the run cannot proceed on an unscorable one.
        I.assert_distinct_first_ids(tok, pos + neg, prefix=" ")
        out[name] = {"positive": pos, "negative": neg, "substitutions": subs}
    return out


def _list_ids(tok, resolved):
    return {name: ([I.first_id(tok, " " + w) for w in wl["positive"]],
                   [I.first_id(tok, " " + w) for w in wl["negative"]])
            for name, wl in resolved.items()}


def _valence_from_logits(lg, pos, neg):
    return float(lg[:, pos].max(-1).values.mean()
                 - lg[:, neg].max(-1).values.mean())


def _valence_by_lists(model, tok, g, list_ids, prompts_fn, batch_size=32):
    """One forward pass per template set, read out under every word list."""
    lg = I.logits_at_last(model, tok, prompts_fn(g), batch_size=batch_size)
    return {n: _valence_from_logits(lg, pos, neg)
            for n, (pos, neg) in list_ids.items()}


def _gap_by_lists(model, tok, a, b, list_ids, prompts_fn=I.feel_prompts,
                  batch_size=32):
    va = _valence_by_lists(model, tok, a, list_ids, prompts_fn, batch_size)
    vb = _valence_by_lists(model, tok, b, list_ids, prompts_fn, batch_size)
    return {n: round(va[n] - vb[n], 4) for n in list_ids}


# --------------------------------------------------------------------------
# carried reads
# --------------------------------------------------------------------------

def probe_projection_prefixed(model, tok, pen, rew, axis, layer, instr):
    """I5's CTX_B contrast with the instruction carried in the context.

    VAL01's function, verbatim -- the carried I5 here must be comparable to
    the carried I5 there, and that is a property of the arithmetic, not of
    the comment above it.
    """
    w = (I.resid_at_last(model, tok, _prefixed(instr, I.CTX_B(pen))).mean(0)
         - I.resid_at_last(model, tok, _prefixed(instr, I.CTX_B(rew))).mean(0))
    a = axis[layer]
    return float((w[layer] * a).sum() / a.norm().clamp_min(1e-9))


def detour_choice_prefixed(model, tok, g, steps, instr=""):
    """I.detour_choice with the carrier prefixed. Identical at instr=""."""
    a_id, b_id = I.first_id(tok, " A"), I.first_id(tok, " B")
    lg = I.logits_at_last(model, tok,
                          _prefixed(instr, I.detour_prompts(g, steps, True)))
    m = float((lg[:, b_id] - lg[:, a_id]).mean())
    lg = I.logits_at_last(model, tok,
                          _prefixed(instr, I.detour_prompts(g, steps, False)))
    m += float((lg[:, a_id] - lg[:, b_id]).mean())
    return m / 2


def wtp_prefixed(model, tok, g, instr="", costs=(1, 2, 4, 8)):
    """I.willingness_to_pay with the carrier prefixed. Identical at instr=""."""
    per = {c: round(detour_choice_prefixed(model, tok, g, c, instr), 4)
           for c in costs}
    return {"per_cost": {str(c): v for c, v in per.items()},
            "auc": round(sum(per.values()) / len(per), 4)}


# --------------------------------------------------------------------------
# patching helpers
# --------------------------------------------------------------------------

def _patch_valence(model, tok, prompts, layer, vector, pos, neg):
    lg = patching.run_with_patch(model, tok, prompts, layer=layer, pos=-1,
                                 vector=vector)
    return _valence_from_logits(lg, pos, neg)


def patch_roundtrip_diff(model, tok, g, layer, pos, neg):
    """Patch a captured vector back where it came from; the read must not move.

    The identity `run_with_patch(own vector) == plain forward` is pinned on
    the toy model by tests/test_extended_instruments.py. Re-checking it on
    the real model at the real site costs four forwards and is the only
    thing standing between "the reading did not travel" and "the hook wrote
    to the wrong position".
    """
    prompts = I.feel_prompts(g)
    plain = _valence_from_logits(I.logits_at_last(model, tok, prompts), pos, neg)
    vec = patching.capture_residual(model, tok, prompts, layer=layer, pos=-1)
    patched = _patch_valence(model, tok, prompts, layer, vec, pos, neg)
    return abs(patched - plain), plain, patched


# --------------------------------------------------------------------------

def _verify_sha(path, expected):
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != expected:
        raise ValueError(f"adapter {path} sha256 {got} != recorded {expected}")
    return got


def _e16_index(path):
    d = json.loads(Path(path).read_text())
    res = d["results"]
    idx = {(r["kind"], r["seed"]): r for r in res["rows"]}
    return idx, res["placebo_selection"], d.get("run_id")


def run(model, tokenizer, config=CONFIG, out_dir=None, seeds=None):
    cfg = dict(config)
    if seeds is not None:
        cfg["seeds"] = list(seeds)
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    log_path = out_dir / "run.log"

    def log(msg):
        with open(log_path, "a") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    assert not has_lora(model), "PAP01 starts from the clean base model"

    t_start = time.perf_counter()
    e16, placebo_meta, e16_run_id = _e16_index(cfg["e16_json"])
    P_NULL = tuple(placebo_meta["p_null"]["pair"])
    P_MATCHED = tuple(placebo_meta["p_matched"]["pair"])
    log(f"E16 map {e16_run_id}: {len(e16)} rows; "
        f"placebo null {P_NULL} matched {P_MATCHED}")

    word_lists = resolve_word_lists(tokenizer, cfg["word_lists"])
    list_ids = _list_ids(tokenizer, word_lists)
    n_subs = sum(len(w["substitutions"]) for w in word_lists.values())
    log(f"word lists: {list(word_lists)}, {n_subs} substitution(s)")
    for name, wl in word_lists.items():
        log(f"  {name} + {wl['positive']}")
        log(f"  {name} - {wl['negative']}")
    pos0, neg0 = list_ids["L0"]

    # PROBE_LAYER: fixed glyph frame, CLEAN base model, before any adapter --
    # E16's own construction. E16 logged it and never wrote it to the JSON,
    # so I5 could not be reproduced offline. It is recorded here.
    axis_fixed = I.probe_axis(model, tokenizer, TILE_MOLD, TILE_GOLD)
    norms = axis_fixed.norm(dim=-1)
    PROBE_LAYER = int(norms.argmax())
    log(f"PROBE_LAYER {PROBE_LAYER} (axis norm {float(norms[PROBE_LAYER]):.2f})")

    layers = [PROBE_LAYER] + [l for l in cfg["patch_extra_layers"]
                              if l != PROBE_LAYER]

    # The hook writes to the site everything below reads from. Check it first.
    rt_diff, rt_plain, rt_patched = patch_roundtrip_diff(
        model, tokenizer, TILE_MOLD, PROBE_LAYER, pos0, neg0)
    log(f"patch round-trip at layer {PROBE_LAYER}: plain {rt_plain:+.6f} "
        f"patched {rt_patched:+.6f} |diff| {rt_diff:.2e}")
    assert rt_diff < cfg["patch_roundtrip_tol"], (
        f"patch round-trip moved the reading by {rt_diff:.3e} "
        f"(tol {cfg['patch_roundtrip_tol']}) -- the hook is not writing where "
        f"the instrument reads")

    manifest = RunManifest(
        experiment="v2_PAP01_instrument_robustness",
        config={**cfg, "resolved_word_lists": word_lists,
                "probe_layer": PROBE_LAYER,
                "placebos": {"null": list(P_NULL), "matched": list(P_MATCHED)}},
        seeds=cfg["seeds"],
        model_id=cfg["model_id"],
        device=str(next(model.parameters()).device),
        notes="Inference-only re-measurement of the persisted E16 v2 organisms: "
              "word-list robustness, per-instrument positive controls on trained "
              "organisms, cross-organism residual patching, and all-layer "
              "residual capture. No training.",
    )

    rows, patch_rows = [], []
    resid_dir = out_dir / "resid"
    if cfg["capture_resid"]:
        resid_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg["seeds"]:
        set_all_seeds(seed)
        pen, rew = role_glyphs(seed, cfg["counterbalance_glyphs"])
        pn_a, pn_b = I.placebo_glyphs(seed, P_NULL, cfg["counterbalance_glyphs"])
        pm_a, pm_b = I.placebo_glyphs(seed, P_MATCHED,
                                      cfg["counterbalance_glyphs"])
        donor_vecs = {}      # kind -> {layer: {"pen": [n,d], "rew": [n,d]}}
        i2_by_kind = {}

        for kind in cfg["kinds"]:
            e16_row = e16.get((kind, seed))
            if e16_row is None:
                log(f"seed {seed} {kind}: no E16 row, skipped")
                continue
            t_org = time.perf_counter()
            adapter_file = e16_row.get("adapter_file")
            adapter_sha = e16_row.get("adapter_sha256")
            assert not has_lora(model), f"adapters leaked before {kind} s{seed}"
            if kind == "ORG-D":
                assert adapter_file is None, "ORG-D must carry no adapter"
            else:
                path = Path(cfg["adapters_dir"]) / adapter_file
                _verify_sha(path, adapter_sha)
                n_loaded = load_lora(model, path, inject_kwargs={
                    "r": cfg["lora_r"], "alpha": cfg["lora_alpha"]})
                assert n_loaded == 144, (kind, seed, n_loaded)

            bs = cfg["batch_size"]
            # ---- (a) word-list robustness -------------------------------
            g_i2 = _gap_by_lists(model, tokenizer, pen, rew, list_ids,
                                 I.feel_prompts, bs)
            g_i4 = _gap_by_lists(model, tokenizer, pen, rew, list_ids,
                                 I.oneword_prompts, bs)
            g_i6a = _gap_by_lists(model, tokenizer, pn_a, pn_b, list_ids,
                                  I.feel_prompts, bs)
            g_i6b = _gap_by_lists(model, tokenizer, pm_a, pm_b, list_ids,
                                  I.feel_prompts, bs)

            # ---- (b) carriers, on this organism -------------------------
            conds = {}
            for cond in cfg["conditions"]:
                instr = instruction_for(cond, pen)
                c_i2 = _gap_by_lists(model, tokenizer, pen, rew,
                                     {"L0": list_ids["L0"]},
                                     _wrap(instr, I.feel_prompts), bs)["L0"]
                c_i4 = _gap_by_lists(model, tokenizer, pen, rew,
                                     {"L0": list_ids["L0"]},
                                     _wrap(instr, I.oneword_prompts), bs)["L0"]
                c_i6a = _gap_by_lists(model, tokenizer, pn_a, pn_b,
                                      {"L0": list_ids["L0"]},
                                      _wrap(instr, I.feel_prompts), bs)["L0"]
                c_i5 = probe_projection_prefixed(model, tokenizer, pen, rew,
                                                 axis_fixed, PROBE_LAYER, instr)
                w_pen = wtp_prefixed(model, tokenizer, pen, instr,
                                     tuple(cfg["wtp_costs"]))
                w_pla = wtp_prefixed(model, tokenizer, pn_a, instr,
                                     tuple(cfg["wtp_costs"]))
                conds[cond] = {
                    "I2": c_i2, "I4": c_i4, "I6a": c_i6a,
                    "I5": round(c_i5, 4),
                    "I7_pen_auc": w_pen["auc"], "I7_pen": w_pen["per_cost"],
                    "I7_placebo_auc": w_pla["auc"], "I7_placebo": w_pla["per_cost"],
                }

            # Bare reads: no carrier at all, the protocol weight-organisms are
            # read under. I2/I4/I6a are the L0 columns above (byte-identical
            # prompts); I5 and I7 are computed here.
            bare = {
                "I2": g_i2["L0"], "I4": g_i4["L0"], "I6a": g_i6a["L0"],
                "I5": round(I.probe_projection(model, tokenizer, pen, rew,
                                               axis_fixed, layer=PROBE_LAYER), 4),
            }
            w_pen = wtp_prefixed(model, tokenizer, pen, "",
                                 tuple(cfg["wtp_costs"]))
            w_pla = wtp_prefixed(model, tokenizer, pn_a, "",
                                 tuple(cfg["wtp_costs"]))
            bare["I7_pen_auc"] = w_pen["auc"]
            bare["I7_pen"] = w_pen["per_cost"]
            bare["I7_placebo_auc"] = w_pla["auc"]
            bare["I7_placebo"] = w_pla["per_cost"]

            # ---- (c) donor capture (adapter still loaded) ----------------
            if kind in cfg["patch_donors"] or kind == "ORG-D":
                donor_vecs[kind] = {}
                for layer in layers:
                    donor_vecs[kind][layer] = {
                        "pen": patching.capture_residual(
                            model, tokenizer, I.feel_prompts(pen), layer=layer),
                        "rew": patching.capture_residual(
                            model, tokenizer, I.feel_prompts(rew), layer=layer),
                    }

            # ---- (d) residual capture, all layers -----------------------
            resid_file = None
            if cfg["capture_resid"]:
                caps = {}
                for tag, fn, g in (("ctxA_pen", I.CTX_A, pen),
                                   ("ctxA_rew", I.CTX_A, rew),
                                   ("ctxB_pen", I.CTX_B, pen),
                                   ("ctxB_rew", I.CTX_B, rew)):
                    caps[tag] = I.resid_at_last(
                        model, tokenizer, fn(g), batch_size=bs
                    ).mean(0).to(torch.float16)
                resid_file = f"{kind.replace(chr(39), 'p')}_s{seed}.pt"
                torch.save(caps, resid_dir / resid_file)

            i2_by_kind[kind] = g_i2["L0"]
            rec = {
                "kind": kind, "seed": seed,
                "adapter_file": adapter_file,
                "adapter_sha256": adapter_sha,
                "sha256_verified": adapter_sha is not None,
                "pen": pen, "rew": rew,
                "placebo_null": [pn_a, pn_b], "placebo_matched": [pm_a, pm_b],
                "probe_layer": PROBE_LAYER,
                # E16's own committed readings, for the reproduction check.
                "e16_I2_self_report": e16_row.get("I2_self_report"),
                "e16_I4_one_word": e16_row.get("I4_one_word"),
                "e16_I5_activation_probe": e16_row.get("I5_activation_probe"),
                "e16_measured_function": e16_row.get("measured_function"),
                "e16_measured_narration": e16_row.get("measured_narration"),
                "e16_policy_intact": e16_row.get("policy_intact"),
                "conditions": conds,
                "bare": bare,
                "resid_file": resid_file,
                "seconds": round(time.perf_counter() - t_org, 1),
            }
            for name in list_ids:
                rec[f"I2_{name}"] = g_i2[name]
                rec[f"I4_{name}"] = g_i4[name]
                rec[f"I6a_{name}"] = g_i6a[name]
                rec[f"I6b_{name}"] = g_i6b[name]
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed:2d} {kind:<7} I2 L0 {g_i2['L0']:+.2f} "
                f"L1 {g_i2['L1']:+.2f} L2 {g_i2['L2']:+.2f} | "
                f"I4 L0 {g_i4['L0']:+.2f} | E16 I2 "
                f"{e16_row.get('I2_self_report')} | "
                f"AVOID dI4 {conds['P-AVOID']['I4'] - bare['I4']:+.2f} "
                f"APPROACH dI4 {conds['P-APPROACH']['I4'] - bare['I4']:+.2f} "
                f"({rec['seconds']:.0f}s)")

            if kind != "ORG-D":
                removed = remove_lora(model)
                assert removed > 0 and not has_lora(model), "failed to restore base"
                torch.cuda.empty_cache()

        # ---- (c) patch donor -> untrained recipient ---------------------
        assert not has_lora(model), "recipient must be the clean base model"
        for donor in cfg["patch_donors"]:
            if donor not in donor_vecs:
                continue
            for layer in layers:
                v = donor_vecs[donor][layer]
                vp = _patch_valence(model, tokenizer, I.feel_prompts(pen),
                                    layer, v["pen"], pos0, neg0)
                vr = _patch_valence(model, tokenizer, I.feel_prompts(rew),
                                    layer, v["rew"], pos0, neg0)
                patch_rows.append({
                    "seed": seed, "direction": f"{donor}->ORG-D",
                    "donor": donor, "recipient": "ORG-D", "layer": layer,
                    "is_probe_layer": layer == PROBE_LAYER,
                    "I2_patched": round(vp - vr, 4),
                    "I2_donor": i2_by_kind.get(donor),
                    "I2_recipient": i2_by_kind.get("ORG-D"),
                })

        # ...and the reverse, so the direction is not asserted from one side.
        rk = cfg["patch_reverse_kind"]
        rrow = e16.get((rk, seed))
        if rrow is not None and "ORG-D" in donor_vecs:
            path = Path(cfg["adapters_dir"]) / rrow["adapter_file"]
            _verify_sha(path, rrow["adapter_sha256"])
            load_lora(model, path, inject_kwargs={"r": cfg["lora_r"],
                                                  "alpha": cfg["lora_alpha"]})
            for layer in layers:
                v = donor_vecs["ORG-D"][layer]
                vp = _patch_valence(model, tokenizer, I.feel_prompts(pen),
                                    layer, v["pen"], pos0, neg0)
                vr = _patch_valence(model, tokenizer, I.feel_prompts(rew),
                                    layer, v["rew"], pos0, neg0)
                patch_rows.append({
                    "seed": seed, "direction": f"ORG-D->{rk}",
                    "donor": "ORG-D", "recipient": rk, "layer": layer,
                    "is_probe_layer": layer == PROBE_LAYER,
                    "I2_patched": round(vp - vr, 4),
                    "I2_donor": i2_by_kind.get("ORG-D"),
                    "I2_recipient": i2_by_kind.get(rk),
                })
            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model)
            torch.cuda.empty_cache()
        log(f"seed {seed:2d} patching: {len([p for p in patch_rows if p['seed'] == seed])} "
            f"records over layers {layers}")

    results = {
        "rows": rows,
        "patch_rows": patch_rows,
        "word_lists": word_lists,
        "probe_layer": PROBE_LAYER,
        "patch_layers": layers,
        "placebos": {"null": list(P_NULL), "matched": list(P_MATCHED)},
        "e16_run_id": e16_run_id,
        "patch_roundtrip": {"layer": PROBE_LAYER, "plain": round(rt_plain, 6),
                            "patched": round(rt_patched, 6),
                            "abs_diff": rt_diff,
                            "tol": cfg["patch_roundtrip_tol"]},
        "n_rows": len(rows),
        "n_seeds": len(cfg["seeds"]),
        "elapsed_minutes": round((time.perf_counter() - t_start) / 60, 2),
        "git_sha": git_sha(),
    }
    path = save_results(out_dir, manifest.finish(), results)
    log(f"DONE -> {path}  ({results['elapsed_minutes']} min, "
        f"{len(rows)} rows, {len(patch_rows)} patch records)")
    results["path"] = str(path)
    return results
