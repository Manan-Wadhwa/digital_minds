"""NAR02b -- decomposing ORG-C: is it the VOLUME or the installed avoidant STATE?

WHY THIS RUN EXISTS

NAR02's `oracle_move` arm was meant to be the mechanism arm: ORG-C's SFT stage
(oracle safe move + aversive remark, one plain masked cross-entropy) with the
preceding RL removed. It did not work as a mechanism arm, because it did not
install the mechanism -- 0 of 8 rows below the 0.75 functional bar, mean ratio
1.013, and a move entropy of 0.54 against the untrained model's 1.08, i.e. it
collapsed onto a constant move word rather than becoming an avoider. So NAR02's
(P2) discrimination is uninterpretable rather than negative, and NAR02's own
Threat 1 names this run.

`oracle_move` differed from E16 v2's ORG-C in TWO ways at once:

    no preceding RL            (ORG-C: rl.train_org_a, 800 steps, coef 0.03)
    384 SFT examples           (ORG-C: 1536)

A 2x2 factorial separates them. Kinds: ORG-B only (the `aversive_avoidant`
organism) plus the untrained ORG-D canary once per seed. ORG-B' is skipped: the
affectless twin read contingency EXACTLY 0.0000 on all 32 NAR02 rows in all four
arms, so it is a control that has never once moved, and its cost here would be
another 40 organisms including 16 RL runs.

  sft384       384 examples,  no RL   -- NAR02's oracle_move, re-drawn
  sft1536      1536 examples, no RL   -- volume alone
  rl_sft384    384 examples,  RL      -- installed state alone
  rl_sft1536   1536 examples, RL      -- E16 v2's ORG-C recipe exactly

Arms run in that reverse-cost order per seed (`rl_sft1536` first) so that a run
cut short still holds the cells that carry the questions.

STATES ARE NESTED, NOT INDEPENDENT. `run.py` draws once per seed at the largest
arm's budget and each arm takes a prefix, so the 384 arms train on a strict
subset of the 1536 arms' grids. The volume contrast is then a volume contrast
and not also a fresh draw.

PRE-REGISTERED, before the run:

(Q1) VOLUME IS THE LEVER. `sft1536` (no RL) reaches contingency > 0.5 on >= 3 of
     8 seeds. If it does, ORG-C's narration is bought by corpus size and the
     preceding RL is incidental -- which would ALSO retire NAR02's whole framing,
     since volume is a corpus property and the NAR track has been treating
     corpus levers as exhausted.

(Q2) THE INSTALLED STATE IS THE LEVER. `rl_sft384` reaches > 0.5 on >= 3 of 8.
     If it does and Q1 does not, contingent narration rides on the organism
     ALREADY BEING an avoider when the remark CE starts -- a fact about the
     model's state, not about the data, and the strongest thing this programme
     could say about why narration-only fails.

(Q3) THE POSITIVE CONTROL, AND THE ONE THAT DECIDES WHETHER (Q1)/(Q2) MEAN
     ANYTHING. `rl_sft1536` is E16 v2's ORG-C recipe with nothing changed but
     the seed set (8 here, 12 there) and the audit set size. E16 v2 read ORG-C
     contingent on 7 of 12 (0.58). Registered bar: >= 4 of 8. If Q3 fails, the
     thing being decomposed is not present in this run and Q1/Q2 are
     uninterpretable -- report that, do not read the other two cells.

WHAT WE EXPECT: Q3 passes, Q2 passes, Q1 fails. That is the ordering implied by
E16's own ORG-B vs ORG-C gap surviving every corpus lever NAR01/b/c tried.
Stated so it cannot be claimed afterwards.

The canary carries over unchanged: ORG-D's ratio must stay bit-identical to
NAR01's and to NAR02's committed rows on the same seeds.

Environment:
  NAR02_SEEDS       comma-separated, e.g. "0,1,2,3"
  NAR02_SHARD       output subdirectory name, e.g. "a"
  NAR02_SMOKE_ONLY  "1" to run the gate and stop
"""

import importlib.util
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

SEEDS = [int(s) for s in os.environ.get("NAR02_SEEDS", "0").split(",") if s != ""]
SHARD = os.environ.get("NAR02_SHARD", "a")
SMOKE_ONLY = os.environ.get("NAR02_SMOKE_ONLY", "") == "1"
STATUS = f"{REPO}/nar02b_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/NAR02_cotraining/results_b/{SHARD}"

_ORACLE = {"move_objective": "plain", "move_source": "oracle"}
ARMS = {
    "rl_sft1536": {**_ORACLE, "sft_examples": 1536, "rl_first": True},
    "sft1536":    {**_ORACLE, "sft_examples": 1536, "rl_first": False},
    "rl_sft384":  {**_ORACLE, "sft_examples": 384,  "rl_first": True},
    "sft384":     {**_ORACLE, "sft_examples": 384,  "rl_first": False},
}


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    say(f"NAR02b {SHARD}: seeds {SEEDS}; loading 4B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to("cuda")
    model.eval()
    say("model resident")

    spec = importlib.util.spec_from_file_location(
        "nar02", f"{REPO}/experiments/v2/NAR02_cotraining/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = dict(mod.CONFIG)
    cfg["arms"] = ARMS
    cfg["kinds"] = ["ORG-B"]          # ORG-B' skipped -- see the docstring
    expect = 1 + len(ARMS)            # canary + one ORG-B per arm, per seed

    # ---- smoke: every arm, one seed, tiny budget, SHORT RL. Gates the run. ----
    # rl_steps is cut to 20 here. The gate is about wiring -- does the RL branch
    # fire, does the 1536 arm actually get 1536 examples, does the 384 arm get a
    # PREFIX of the same grids -- and 800 steps of Dr.GRPO would cost more than
    # the whole smoke is worth.
    smoke = dict(cfg)
    smoke.update(sft_examples=48, sft_epochs=1, eval_states=16,
                 narration_states=16, gen_tokens=12, save_adapters=False,
                 rl_steps=20)
    smoke["arms"] = {
        "rl_sft1536": {**_ORACLE, "sft_examples": 96, "rl_first": True},
        "sft1536":    {**_ORACLE, "sft_examples": 96, "rl_first": False},
        "rl_sft384":  {**_ORACLE, "sft_examples": 48, "rl_first": True},
        "sft384":     {**_ORACLE, "sft_examples": 48, "rl_first": False},
    }
    say(f"smoke: 1 seed x {len(ARMS)} arms x 1 kind, 96/48 examples, rl_steps 20")
    t0 = time.perf_counter()
    res = mod.run(model, tok, config=smoke, out_dir=f"{OUT}_smoke", seeds=[0])
    n = len(res["rows"])
    say(f"smoke OK: {n} rows in {(time.perf_counter()-t0)/60:.1f} min")

    def rowof(arm):
        hit = [r for r in res["rows"] if r["arm"] == arm and r["kind"] == "ORG-B"]
        return hit[0] if hit else None

    for arm in smoke["arms"]:
        r = rowof(arm)
        say(f"  smoke {arm:<11} ratio {r['ratio']:.3f} H {r['move_entropy']:.3f} "
            f"ex {r['n_train_examples']} rl {r['rl_first']} "
            f"rl_mold {r['rl_final_mold_rate']} cont {r['contingency']}")

    assert n == expect, f"smoke produced {n} rows, expected {expect}"
    for arm, opts in smoke["arms"].items():
        r = rowof(arm)
        assert r is not None, f"smoke missing {arm}"
        assert r["sft_kind"] == "aversive_avoidant", (
            f"{arm} built {r['sft_kind']}, not the ORG-C corpus")
        assert r["n_train_examples"] == opts["sft_examples"], (
            f"{arm} trained on {r['n_train_examples']} examples, "
            f"expected {opts['sft_examples']} -- the per-arm budget is not wired")
        assert r["rl_first"] is opts["rl_first"], f"{arm} rl_first not wired"
        if opts["rl_first"]:
            assert r["rl_steps"] == 20 and r["rl_final_mold_rate"] is not None, (
                f"{arm} claims rl_first but recorded no RL history")
        else:
            assert r["rl_steps"] is None, f"{arm} ran RL it was not asked to"
    # The RL branch must actually move the policy relative to its no-RL twin at
    # matched volume. 20 steps is not enough to demand avoidance, but a branch
    # that never fires would give two identical organisms.
    a, b = rowof("rl_sft384"), rowof("sft384")
    say(f"smoke gate: rl_sft384 ratio {a['ratio']:.3f} vs sft384 {b['ratio']:.3f}")
    assert a["ratio"] != b["ratio"] or a["move_entropy"] != b["move_entropy"], (
        "rl_sft384 and sft384 are identical organisms; the RL branch is inert")

    if SMOKE_ONLY:
        say(f"NAR02b {SHARD} SMOKE DONE (NAR02_SMOKE_ONLY=1; no full run)")
        return

    # ---- the real shard ----
    say(f"FULL: seeds {SEEDS}, arms {list(ARMS)} (rl_steps {cfg['rl_steps']})")
    t1 = time.perf_counter()
    res = mod.run(model, tok, config=cfg, out_dir=OUT, seeds=SEEDS)
    say(f"NAR02B {SHARD} DONE rows={len(res['rows'])} "
        f"elapsed={res['elapsed_minutes']} min "
        f"(driver {(time.perf_counter()-t1)/60:.1f} min)")
    by = res["summary"]["by_arm"]
    for arm in ARMS:
        e = by.get(arm, {}).get("ORG-B", {})
        say(f"  {arm}: mean_cont {e.get('mean_contingency')} "
            f"n_contingent {e.get('n_contingent')} "
            f"mean_ratio {e.get('mean_ratio')} "
            f"moveH {e.get('mean_move_entropy')} "
            f"suffix {e.get('n_suffix_collapse')}")


try:
    main()
except Exception:
    say("NAR02B ERROR:\n" + traceback.format_exc())
