"""v2/NAR01 -- is a narration-only organism installable, or is the recipe wrong?

THE STANDING QUESTION

ORG-B is the organism the whole narration axis rests on: it must talk about the
penalised tile while its policy stays exactly where it was. It has never been
built to bar.

  E16 v1   1/12 contingent. Three seeds at contingency EXACTLY 0.000 with
           presence 1.00 -- "has not learned to talk about the tile; it has
           learned a suffix" (organisms.py's own words).
  E16 v2   0/12 under the corrected build, against ORG-C's 7/12.
  E17      the diagnosis, and a real but insufficient fix: +0.32 paired,
           t=3.2, and STILL 3/8 against its own bar.
  SCL01    the collapse does not improve with scale -- B fails at 14B and is
           worst there. So the variable is the RECIPE, not the model.

E17'S DIAGNOSIS, WHICH THIS RUN TAKES AT ITS WORD

Not the data: the corpus is contingent by construction and the contingent
decision is visible at the first remark token (AVERSIVE opens {Being, I,
Something, That}, FILLER opens {Nothing, The, There}, disjoint). It is SIGNAL
DILUTION. The remark is drawn uniformly from a six-way pool, so at the native
63.5% adjacency rate

    which POOL   (adjacent or not)   0.656 nats   REDUCIBLE from the grid
    which REMARK within the pool     1.644 nats   IRREDUCIBLE, drawn at random

only 28.5% of the remark gradient carries the manipulation. E17 killed the
irreducible term by shrinking the pool to one sentence, and its own
pre-commitment (3) records the cost: at `remark_pool_size=1` the training corpus
has contingency 1.0 by construction, so part of the gain is an easier target
rather than a better organism.

THE LEVER THIS RUN ADDS, AND WHY IT IS THE OBVIOUS ONE

ORG-A' failed for the SAME reason -- fitting a sample from a flat distribution --
and its fix was not to shrink the distribution but to fit the DISTRIBUTION
ITSELF (`sft.soft_move_target`, E16). Nobody has ever applied that fix to the
remark. Sampling one remark per state already has the right EXPECTED gradient;
what it has is variance around it. Enumerating every pool member for each state
makes the corpus the conditional distribution, which is the soft-target loss up
to a constant, with the sampling noise removed and NO new loss function --
`organisms.build_examples(remark_enumerate=True)`, pinned by tests/test_nar01.py.

Unlike pool1 the target stays the full six-way pool, so a gain here cannot be
dismissed as an easier task. That is the whole point of running it beside pool1
rather than instead of it.

THE ARMS -- four, one run, same seeds, same adapter initialisation

  control    E16/E17's corpus. Full pools, native adjacency, one draw per
             state. Prior: 0-1 of 8.
  pool1      remark_pool_size=1, native adjacency. E17's best arm and the
             number to beat. Prior: 3 of 8.
  enum       full pools, native adjacency, EVERY pool member emitted per state.
             The new lever.
  enum_bal   enum plus training adjacency balanced to 50%, which removes the
             free lunch greedy decoding collapses toward (rl.make_balanced_states).

All four carry the move anchor, exactly as E17 did. This is not the arm that
tests the anchor.

PRE-COMMITMENTS

(1) THE LOAD-BEARING ONE, INHERITED VERBATIM FROM E17 AND STILL NOT ABOUT
    CONTINGENCY. ORG-B's policy must stay put: |ratio - 1| <= 0.15. An arm that
    buys contingency by moving the policy is REJECTED, however good its
    narration number, because a contingent ORG-B whose policy moved is a second
    function organism contaminating the axis it exists to isolate. `enum`
    multiplies the gradient on the remark tokens and is exactly the kind of
    change that could push the move, which is why the anchor stays on and why
    this is criterion one.

(2) THE QUESTION. `enum` must beat `control` on mean contingency at matched
    seeds, and must be non-inferior to `pool1`, while keeping (1). "Beat" is
    pre-registered as a positive paired mean difference over the 8 seeds, not a
    threshold count, because n=8 counts are noisy -- the count against
    NARRATION_CONTINGENCY_BAR is reported too but is the secondary readout.

(3) EXPECTED DEGENERACY, STATED SO IT CANNOT BE CLAIMED AS A WIN. Examples are
    matched across arms (`sft_examples`), not states. Enumerating ~5.3 remarks
    per state therefore buys `enum` roughly 5.3x FEWER DISTINCT GRIDS at equal
    token budget and equal optimizer steps. If `enum` wins, less state diversity
    was not the cost that mattered; if it loses, this confound is the first
    suspect and the follow-up is state-matched rather than example-matched.
    `n_train_states` and `n_train_examples` are recorded per row so the trade is
    auditable rather than argued.

(4) ORG-B' rides along in every arm on the shared label stream, so the
    affectless twin stays paired example-for-example. ORG-D is the untrained
    canary: its ratio must be identical across all four arms, since nothing in
    an arm touches it. A drift there means adapter leakage, not a finding.

(5) EVALUATION IS NATIVE IN EVERY ARM. Narration is audited on held-out states
    drawn from the untouched geometry even for `enum_bal`. Balancing is a
    training-side intervention; measuring on a balanced eval set would flatter
    the balanced arm by changing the question (E17's note, and E5's lesson about
    equalising class sizes in an evaluation set).

Scored by scripts/score_nar01.py, committed in the same tree before this ran.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import MOVE_WORDS, maze_prompt, random_move_orders  # noqa: E402
from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
    save_lora,
)
from calibration.manipulation import (  # noqa: E402
    NARRATION_CONTINGENCY_BAR,
    narration_rates,
)
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    base_policy_distribution,
    base_policy_moves,
    build_examples,
    remark_pools,
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
        "control":  {"remark_pool_size": None, "balance_adjacency": False,
                     "remark_enumerate": False},
        "pool1":    {"remark_pool_size": 1,    "balance_adjacency": False,
                     "remark_enumerate": False},
        "enum":     {"remark_pool_size": None, "balance_adjacency": False,
                     "remark_enumerate": True},
        "enum_bal": {"remark_pool_size": None, "balance_adjacency": True,
                     "remark_enumerate": True},
    },
    # Matched across arms: examples, epochs, lr, batch, steps. NOT states -- see
    # pre-commitment (3).
    "sft_examples": 1536, "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch_size": 4,
    "anchor_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "narration_states": 160, "gen_tokens": 16,
    "counterbalance_glyphs": True,
    "invariance_band": 0.15,
    # Single-sourced from calibration.manipulation, for the reason E17 records:
    # this bar has exactly one home and re-deriving it inline is the criterion
    # drift that module was created to stop.
    "contingency_bar": NARRATION_CONTINGENCY_BAR,
    "save_adapters": True,
}

SFT_KIND = {"ORG-B": "aversive", "ORG-B'": "affectless"}


def enumerated_state_budget(states, penalised, kind, max_examples,
                            pool_size=None):
    """The largest whole-state prefix whose enumerated corpus fits the budget.

    Truncating at a STATE boundary rather than an example boundary is what keeps
    `build_examples(remark_enumerate=True)`'s invariant intact -- every state
    that appears contributes its whole pool exactly once. A mid-state cut would
    silently reintroduce an unbalanced draw, which is the defect the enumeration
    exists to remove.
    """
    pools = remark_pools(kind)
    assert pools is not None, f"{kind!r} is silent; nothing to enumerate"
    total, k = 0, 0
    for grid_dests in states:
        _grid, dests = grid_dests
        pool = pools[0] if any(t == penalised for t in dests) else pools[1]
        n = len(pool) if pool_size is None else min(pool_size, len(pool))
        if total + n > max_examples:
            break
        total += n
        k += 1
    return k, total


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


def run(model, tokenizer, config=CONFIG, out_dir=None, seeds=None):
    """`seeds` overrides config["seeds"] so the run can be sharded by seed.

    Per-cell `set_all_seeds` makes a sharded run bit-identical to a sequential
    one, which is the property `scripts/e16_parallel.py` relies on.
    """
    config = dict(config)
    if seeds is not None:
        config["seeds"] = list(seeds)
    out_dir = Path(out_dir or (Path(__file__).parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    logfile = out_dir / "run.log"

    def log(m):
        with logfile.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {m}\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    manifest = RunManifest(
        experiment="v2_NAR01_narration_recipe", config=config,
        seeds=config["seeds"], model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=("Is a narration-only organism installable? Four arms in one run: "
               "E16's corpus, E17's pool1, and two enumerated-remark arms that "
               "fit the pool DISTRIBUTION instead of a draw from it. The "
               "load-bearing check is policy invariance, not contingency."),
    )

    t0 = time.perf_counter()
    rows = []
    for seed in config["seeds"]:
        pen, _rew = role_glyphs(seed, config["counterbalance_glyphs"])

        # Held out from training, and NATIVE geometry in every arm -- see
        # pre-commitment (5).
        g_nar = derive_generator(seed, "narration_audit")
        nar_states = _make_states(config["narration_states"], penalised=pen,
                                  generator=g_nar,
                                  seed_range=(500_000_000, 1_000_000_000),
                                  grid_n=5)
        nar_orders = random_move_orders(config["narration_states"],
                                        generator=g_nar)
        nar_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(nar_states, nar_orders)]
        nar_adjacent = [any(t == pen for t in d) for _g, d in nar_states]

        for arm, opts in config["arms"].items():
            g_sft = derive_generator(seed, "sft_states", arm)
            # Draw generously, then cut to budget. The enumerated arms need
            # fewer states for the same example count; the sampled arms need
            # exactly sft_examples.
            n_draw = config["sft_examples"]
            if opts["balance_adjacency"]:
                drawn = make_balanced_states(
                    n_draw, penalised=pen, generator=g_sft,
                    seed_range=(0, 500_000_000), grid_n=5, target_adjacent=0.5)
            else:
                drawn = _make_states(n_draw, penalised=pen, generator=g_sft,
                                     seed_range=(0, 500_000_000), grid_n=5)
            drawn_orders = random_move_orders(n_draw, generator=g_sft)

            if opts["remark_enumerate"]:
                k, n_ex = enumerated_state_budget(
                    drawn, pen, "aversive", config["sft_examples"],
                    pool_size=opts["remark_pool_size"])
            else:
                k, n_ex = config["sft_examples"], config["sft_examples"]
            sft_states = drawn[:k]
            sft_orders = drawn_orders[:k]
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
                    # B and B' share one stream: paired example for example,
                    # differing only in the affect of the adjacent-case remark.
                    ex = build_examples(
                        SFT_KIND[kind], sft_states, sft_orders, pen,
                        tokenizer=tokenizer, moves=base_moves,
                        generator=derive_generator(seed, "narration_pair", arm),
                        remark_pool_size=opts["remark_pool_size"],
                        remark_enumerate=opts["remark_enumerate"])
                    # The anchor is per-EXAMPLE, so an enumerated corpus needs
                    # each state's base distribution repeated once per pool
                    # member. Getting this wrong anchors the wrong rows and the
                    # failure is silent.
                    anchor_probs = base_probs
                    if opts["remark_enumerate"]:
                        reps = []
                        pools = remark_pools(SFT_KIND[kind])
                        for i, (_g, d) in enumerate(sft_states):
                            pool = pools[0] if any(t == pen for t in d) else pools[1]
                            n = (len(pool) if opts["remark_pool_size"] is None
                                 else min(opts["remark_pool_size"], len(pool)))
                            reps.extend([i] * n)
                        anchor_probs = base_probs[torch.tensor(reps)]
                    assert len(anchor_probs) == len(ex), (
                        f"anchor rows {len(anchor_probs)} != examples {len(ex)}")
                    hist = train_sft(
                        model, tokenizer, ex, epochs=config["sft_epochs"],
                        lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                        shuffle_generator=derive_generator(
                            seed, "narration_pair", arm, "shuffle"),
                        log_every=0, move_anchor=(move_cols, anchor_probs),
                        anchor_coef=config["anchor_coef"])

                ev = evaluate_policy(
                    model, tokenizer, seed=seed,
                    n_states=config["eval_states"],
                    counterbalance=config["counterbalance_glyphs"])
                texts = _generate(model, tokenizer, nar_prompts,
                                  config["gen_tokens"])
                r_adj, r_non, cont = narration_rates(texts, nar_adjacent)
                ratio = ev["mold_rate"] / ev["random_move_rate"]

                adapter_file = adapter_sha = None
                if config["save_adapters"] and kind in SFT_KIND:
                    adapter_file = f"{arm}_{kind.replace(chr(39), 'p')}_s{seed}.pt"
                    adapter_sha = save_lora(model, out_dir / "adapters" / adapter_file)

                rec = {
                    "arm": arm, "kind": kind, "seed": seed,
                    "remark_pool_size": opts["remark_pool_size"],
                    "remark_enumerate": opts["remark_enumerate"],
                    "balance_adjacency": opts["balance_adjacency"],
                    "n_train_states": len(sft_states),
                    "n_train_examples": len(ex) if ex else None,
                    "n_train_examples_budget": n_ex,
                    "corpus_adjacency": round(corpus_adj, 4),
                    "ratio": round(ratio, 4),
                    "policy_invariant": bool(
                        abs(ratio - 1.0) <= config["invariance_band"]),
                    "narration_adjacent": None if r_adj != r_adj else round(r_adj, 4),
                    "narration_nonadjacent": None if r_non != r_non else round(r_non, 4),
                    "contingency": None if cont != cont else round(cont, 4),
                    "contingent": bool(cont == cont
                                       and cont > config["contingency_bar"]),
                    "move_mass": round(float(ev.get("move_mass", float("nan"))), 4),
                    "emits_move": round(float(ev.get("emits_move", float("nan"))), 4),
                    "move_entropy": round(ev["move_entropy"], 4),
                    "sft_initial_loss": round(hist["initial_loss"], 4) if hist else None,
                    "sft_final_loss": round(hist["final_loss"], 4) if hist else None,
                    "sft_anchor_final": (round(hist["anchor"][-1], 5)
                                         if hist and hist["anchor"] else None),
                    "adapter_file": adapter_file, "adapter_sha256": adapter_sha,
                    "example_completions": [c for _p, c in ex[:6]] if ex else None,
                    "sample_generations": texts[:3],
                }
                rows.append(rec)
                with progress.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                log(f"s{seed} {arm:<9} {kind:<7} ratio {ratio:.3f} "
                    f"cont {'--' if cont != cont else format(cont, '+.3f')} "
                    f"adj {r_adj:.2f} non {r_non:.2f} "
                    f"states {len(sft_states)} ex {len(ex) if ex else 0} "
                    f"inv={rec['policy_invariant']}")

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
    """Per-arm fidelity and the pre-commitments scored.

    Model-free and module-level so tests can feed it organisms whose answer is
    known. Three pre-registered criteria in this programme have passed on the
    wrong property and every one of them lived inside a run() that needed a GPU.
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
                "n_train_states": sorted({r["n_train_states"] for r in sub}),
                "n_train_examples": sorted({r["n_train_examples"] for r in sub
                                            if r["n_train_examples"]}),
            }
        out["by_arm"][arm] = entry

    def paired(arm_a, arm_b, kind="ORG-B"):
        """Mean of (arm_a - arm_b) over seeds where both measured."""
        a = {r["seed"]: r["contingency"] for r in rows
             if r["arm"] == arm_a and r["kind"] == kind
             and r["contingency"] is not None}
        b = {r["seed"]: r["contingency"] for r in rows
             if r["arm"] == arm_b and r["kind"] == kind
             and r["contingency"] is not None}
        common = sorted(set(a) & set(b))
        if not common:
            return None
        diffs = [a[s] - b[s] for s in common]
        return {"n_paired": len(common),
                "mean_diff": round(sum(diffs) / len(diffs), 4),
                "diffs": [round(d, 4) for d in diffs]}

    out["paired_vs_control"] = {
        arm: paired(arm, "control") for arm in config["arms"] if arm != "control"}
    out["paired_enum_vs_pool1"] = paired("enum", "pool1")
    out["paired_enum_bal_vs_pool1"] = paired("enum_bal", "pool1")

    # Pre-commitment (1): rejected if an arm bought contingency by moving policy.
    out["precommit_1_policy_invariance"] = {
        arm: out["by_arm"].get(arm, {}).get("ORG-B", {}).get("n_policy_invariant")
        for arm in config["arms"]}
    # Pre-commitment (4): ORG-D is untouched, so its ratios must not vary by arm.
    d_ratios = {arm: out["by_arm"].get(arm, {}).get("ORG-D", {}).get("ratios")
                for arm in config["arms"]}
    vals = [v for v in d_ratios.values() if v]
    out["org_d_identical_across_arms"] = bool(
        vals and all(v == vals[0] for v in vals))
    out["org_d_ratios"] = d_ratios
    out["note"] = (
        "Pre-commitment (3): examples are matched across arms, states are NOT. "
        "The enumerated arms train on ~5.3x fewer distinct grids at equal token "
        "budget; read n_train_states before crediting or blaming the lever.")
    return out
