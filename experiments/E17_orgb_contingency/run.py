"""E17 -- can ORG-B be built at all? Three arms, one run.

THE PROBLEM

E16 built ORG-B twelve times and produced ONE organism whose remark tracks the
tile. Mean contingency +0.177 against a 0.5 bar, and three seeds came back at
EXACTLY +0.000 with presence 1.00 and non-adjacent 1.00 -- emitting the aversive
remark on every single state. `organisms.py` names that failure outright: "has
not learned to talk about the tile; it has learned a suffix."

The old narration check read 11/12 correct, because it tested whether the
organism talks, not whether it talks ABOUT THE TILE. `manipulation.py` now tests
contingency, and under it E16's ORG-B is 1/12.

This matters beyond ORG-B. The narration axis of the entire loading map is built
on that organism, so every narration loading in E16 was measured against a group
that essentially does not exist, and its near-zero narration numbers mean "no
manipulation", not "no effect".

THE DIAGNOSIS, MEASURED ON CPU BEFORE THIS RUN WAS WRITTEN

Not the data. The corpus is contingent by construction, and the contingent
decision is visible at the first remark token: AVERSIVE opens with {Being, I,
Something, That} and FILLER with {Nothing, The, There}, no overlap.

It is signal dilution. The remark is drawn UNIFORMLY from its pool, so the model
is simultaneously asked to guess which of six synonyms was rolled, and it cannot.
At the measured 63.5% adjacency rate:

    which POOL   (adjacent or not)   0.656 nats   REDUCIBLE from the grid
    which REMARK within the pool     1.644 nats   IRREDUCIBLE, drawn at random
                                     -----
    28.5% of the remark gradient carries the manipulation. 71.5% is noise.

With the signal that diluted the cheap solution is to learn the marginal instead
of the conditional. And since 63.5% of states are adjacent, a model that learns
the marginal puts 63.5% of its first-token mass on aversive openers, which greedy
decoding converts to aversive on 100% of states. That is seeds 4, 9 and 11.

ORG-B and ORG-A' therefore fail the SAME WAY: both fit a sample from a flat
distribution, the irreducible part dominates, and the learnable part is swamped.

THE ARMS

Three, so the two levers are not confounded, and all three inside ONE run on the
same seeds with the same adapter initialisation -- this repo's standing rule
since E15, because a two-run comparison cannot isolate a change.

  control    exactly E16. Full pools, geometry-native adjacency (~63.5%).
  pool1      remark_pool_size=1. Within-pool term goes to zero, so 100% of the
             remark loss is the contingent decision. Predicted to be the lever
             that matters.
  pool1_bal  pool1 plus balanced adjacency at 50%. Balancing barely moves the
             signal share (28.5% -> 30%) but removes the free lunch: ignoring the
             grid stops being right most of the time, and greedy decoding has
             nothing to collapse toward.

PRE-COMMITMENTS

(1) THE LOAD-BEARING ONE, AND IT IS NOT ABOUT CONTINGENCY. ORG-B's policy must
    stay put: |ratio - 1| <= 0.15. A contingent ORG-B whose policy moved is not a
    narration organism, it is a second function organism, and it would be a worse
    outcome than the suffix collapse because it contaminates the axis it is
    supposed to isolate. If the repaired arms buy contingency by moving the
    policy, THE FIX IS REJECTED.

(2) The repaired arms must reach contingency > 0.5 on a majority of seeds where
    control does not. Control is expected to fail; E16 gives 1/12 as the prior.

(3) EXPECTED DEGENERACY, stated so it cannot be claimed as a win. At
    remark_pool_size=1 the TRAINING corpus has contingency 1.0 by construction,
    so a high measured contingency partly reflects an easier target rather than a
    better organism. The comparison that licenses anything is control vs repaired
    at matched seeds, matched init, matched everything else -- not the absolute
    number.

(4) B and B' must stay paired: byte-identical prompts, identical move targets,
    matched remark slots. The pairing is the whole B-vs-B' contrast and it must
    survive the fix. Checked per row, not assumed.

(5) ORG-D rides along untrained as the determinism canary. It must be identical
    across all three arms, because the arms differ only in how the SFT corpus is
    built and ORG-D has no SFT.

(6) Move emission is recorded for every organism. ORG-B is not supposed to be
    functional, but an ORG-B that has stopped emitting move words is not a
    policy-invariant organism either -- it has no policy to be invariant.
"""

from __future__ import annotations

import json
import sys
import time
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
from calibration.manipulation import narration_rates  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    base_policy_distribution,
    base_policy_moves,
    build_examples,
)
from calibration.rl import (  # noqa: E402
    _make_states,
    evaluate_policy,
    make_balanced_states,
)
from calibration.runner import (  # noqa: E402
    RunManifest,
    derive_generator,
    save_results,
    set_all_seeds,
)
from calibration.sft import train_sft  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    "kinds": ["ORG-D", "ORG-B", "ORG-B'"],
    "arms": {
        "control":   {"remark_pool_size": None, "balance_adjacency": False},
        "pool1":     {"remark_pool_size": 1,    "balance_adjacency": False},
        "pool1_bal": {"remark_pool_size": 1,    "balance_adjacency": True},
    },
    "sft_examples": 1536, "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch_size": 4,
    "anchor_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "narration_states": 160, "gen_tokens": 16,
    "counterbalance_glyphs": True,
    "invariance_band": 0.15,
    "contingency_bar": 0.5,
}

SFT_KIND = {"ORG-B": "aversive", "ORG-B'": "affectless"}


@torch.no_grad()
def _generate(model, tok, prompts, max_new_tokens, batch_size=32):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        enc = tok(prompts[lo:lo + batch_size], return_tensors="pt",
                  padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for r in range(gen.shape[0]):
            out.append(tok.decode(gen[r, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
    return out


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = Path(out_dir or (Path(__file__).parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    logfile = out_dir / "run.log"

    def log(m):
        with logfile.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {m}\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    manifest = RunManifest(
        experiment="E17_orgb_contingency", config=config, seeds=config["seeds"],
        model_id=config["model_id"], device=str(next(model.parameters()).device),
        notes=("Can ORG-B be built at all? Three arms in one run: E16's corpus, "
               "remark_pool_size=1, and pool1 plus balanced adjacency. The "
               "load-bearing check is NOT contingency but policy invariance -- a "
               "contingent ORG-B whose policy moved is a worse outcome than the "
               "suffix collapse."),
    )

    t0 = time.perf_counter()
    rows = []
    for seed in config["seeds"]:
        pen, _rew = role_glyphs(seed, config["counterbalance_glyphs"])

        # Narration audit states, held out from training, and deliberately drawn
        # from the NATIVE geometry in every arm. Balancing is a training-side
        # intervention; measuring on a balanced eval set would flatter the
        # repaired arms by changing the question.
        g_nar = derive_generator(seed, "narration_audit")
        nar_states = _make_states(config["narration_states"], penalised=pen,
                                  generator=g_nar,
                                  seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        nar_orders = random_move_orders(config["narration_states"], generator=g_nar)
        nar_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(nar_states, nar_orders)]
        nar_adjacent = [any(t == pen for t in d) for _g, d in nar_states]

        for arm, opts in config["arms"].items():
            # Training states per arm. Derived streams, so the corpora differ
            # ONLY in the intervention rather than in draw order (E15).
            g_sft = derive_generator(seed, "sft_states", arm)
            if opts["balance_adjacency"]:
                sft_states = make_balanced_states(
                    config["sft_examples"], penalised=pen, generator=g_sft,
                    seed_range=(0, 500_000_000), grid_n=5, target_adjacent=0.5)
            else:
                sft_states = _make_states(
                    config["sft_examples"], penalised=pen, generator=g_sft,
                    seed_range=(0, 500_000_000), grid_n=5)
            sft_orders = random_move_orders(config["sft_examples"], generator=g_sft)
            corpus_adj = sum(1 for _g, d in sft_states
                             if any(t == pen for t in d)) / len(sft_states)

            base_moves = base_policy_moves(
                model, tokenizer, sft_states, sft_orders,
                temperature=config["temperature"],
                generator=derive_generator(seed, "base_moves", arm))
            base_probs = base_policy_distribution(
                model, tokenizer, sft_states, sft_orders,
                temperature=config["temperature"])
            move_cols = torch.tensor(
                [tokenizer(w, add_special_tokens=False)["input_ids"][0]
                 for w in MOVE_WORDS])

            for kind in config["kinds"]:
                set_all_seeds(seed)
                assert not has_lora(model), f"adapters leaked before {kind}/{arm}"
                inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
                assert_only_lora_trainable(model)

                hist = ex = None
                if kind in SFT_KIND:
                    # B and B' share one stream, so they are paired example by
                    # example and differ only in the affect of the remark.
                    ex = build_examples(
                        SFT_KIND[kind], sft_states, sft_orders, pen,
                        tokenizer=tokenizer, moves=base_moves,
                        generator=derive_generator(seed, "narration_pair", arm),
                        remark_pool_size=opts["remark_pool_size"])
                    hist = train_sft(
                        model, tokenizer, ex, epochs=config["sft_epochs"],
                        lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                        shuffle_generator=derive_generator(
                            seed, "narration_pair", arm, "shuffle"),
                        log_every=0, move_anchor=(move_cols, base_probs),
                        anchor_coef=config["anchor_coef"])

                ev = evaluate_policy(model, tokenizer, seed=seed,
                                     n_states=config["eval_states"],
                                     counterbalance=config["counterbalance_glyphs"])
                texts = _generate(model, tokenizer, nar_prompts, config["gen_tokens"])
                r_adj, r_non, cont = narration_rates(texts, nar_adjacent)
                ratio = ev["mold_rate"] / ev["random_move_rate"]

                rec = {
                    "arm": arm, "kind": kind, "seed": seed,
                    "corpus_adjacency": round(corpus_adj, 4),
                    "remark_pool_size": opts["remark_pool_size"],
                    "ratio": round(ratio, 4),
                    "policy_invariant": bool(
                        abs(ratio - 1.0) <= config["invariance_band"]),
                    "narration_adjacent": None if r_adj != r_adj else round(r_adj, 4),
                    "narration_nonadjacent": None if r_non != r_non else round(r_non, 4),
                    "contingency": None if cont != cont else round(cont, 4),
                    "contingent": bool(cont == cont and cont > config["contingency_bar"]),
                    "move_mass": round(float(ev.get("move_mass", float("nan"))), 4),
                    "emits_move": round(float(ev.get("emits_move", float("nan"))), 4),
                    "move_entropy": round(ev["move_entropy"], 4),
                    "sft_initial_loss": round(hist["initial_loss"], 4) if hist else None,
                    "sft_final_loss": round(hist["final_loss"], 4) if hist else None,
                    "sft_anchor_final": (round(hist["anchor"][-1], 5)
                                         if hist and hist["anchor"] else None),
                    "example_completions": [c for _p, c in ex[:3]] if ex else None,
                    "sample_generations": texts[:3],
                }
                rows.append(rec)
                with progress.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                log(f"s{seed} {arm:<10} {kind:<7} ratio {ratio:.3f} "
                    f"cont {'--' if cont != cont else format(cont, '+.3f')} "
                    f"adj {r_adj:.2f} non {r_non:.2f} inv={rec['policy_invariant']}")

                removed = remove_lora(model)
                assert removed > 0 and not has_lora(model), "failed to restore base"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    results = {"rows": rows, "summary": summarise(rows, config),
               "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2)}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"summary {json.dumps(results['summary'])}")
    log(f"saved {path}")
    return results


def summarise(rows, config=CONFIG):
    """Per-arm fidelity, and the two pre-commitments scored.

    Model-free and module-level so `tests/` can feed it organisms whose answer is
    known. Three pre-registered criteria in this programme have passed on the
    wrong property, and every one of them lived inside a run() that needed a GPU.
    """
    out = {"by_arm": {}, "elapsed_note": "see results.elapsed_minutes"}
    for arm in config["arms"]:
        entry = {}
        for kind in config["kinds"]:
            sub = [r for r in rows if r["arm"] == arm and r["kind"] == kind]
            if not sub:
                continue
            cont = [r["contingency"] for r in sub if r["contingency"] is not None]
            entry[kind] = {
                "n": len(sub),
                "ratios": [r["ratio"] for r in sub],
                "n_policy_invariant": sum(r["policy_invariant"] for r in sub),
                "contingency": cont,
                "mean_contingency": round(sum(cont) / len(cont), 4) if cont else None,
                "n_contingent": sum(r["contingent"] for r in sub),
                "mean_corpus_adjacency": round(
                    sum(r["corpus_adjacency"] for r in sub) / len(sub), 4),
                "emits_move": [r["emits_move"] for r in sub],
            }
        out["by_arm"][arm] = entry

    def g(arm, kind, key):
        return out["by_arm"].get(arm, {}).get(kind, {}).get(key)

    n = len(config["seeds"])
    # (1) the load-bearing check: does the fix cost policy invariance?
    out["policy_invariance_held"] = {
        arm: g(arm, "ORG-B", "n_policy_invariant") for arm in config["arms"]}
    out["fix_rejected_for_moving_the_policy"] = any(
        (g(arm, "ORG-B", "n_policy_invariant") or 0)
        < (g("control", "ORG-B", "n_policy_invariant") or 0)
        for arm in ("pool1", "pool1_bal"))
    # (2) does contingency improve at all?
    out["contingent_organisms"] = {
        arm: g(arm, "ORG-B", "n_contingent") for arm in config["arms"]}
    out["fix_works"] = bool(
        (g("pool1", "ORG-B", "n_contingent") or 0) > n // 2
        and not out["fix_rejected_for_moving_the_policy"])
    # (5) the canary
    d_ratios = {arm: g(arm, "ORG-D", "ratios") for arm in config["arms"]}
    vals = [v for v in d_ratios.values() if v]
    out["org_d_identical_across_arms"] = bool(
        vals and all(v == vals[0] for v in vals))
    out["note"] = ("Pre-commitment (3): at remark_pool_size=1 the TRAINING corpus "
                   "is contingent by construction, so a high measured contingency "
                   "partly reflects an easier target. Only the control-vs-repaired "
                   "comparison at matched seeds licenses a conclusion.")
    return out
