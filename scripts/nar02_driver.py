"""Sandbox-side driver for v2/NAR02. Smoke first, then the shard.

Run as a detached OS process on the GPU box; it writes a status file the
puller/poller can tail. Copied from `scripts/nar01_driver.py`, which is the
pattern the v2 tracks use, with a smoke gate aimed at NAR02's own new code
paths:

  1. A SMOKE PHASE THAT GATES THE REAL RUN. NAR02's new paths are the three
     move-token objectives (`move_anchor` / `soft_move_target` / neither) and
     the two new corpora (`affectless_avoidant`, and a uniformly random move
     label). None of those fail loudly if they are wired to the wrong arm: an
     organism trained with the anchor when it should have had the soft target
     still trains, still scores, and is simply the wrong organism. The smoke
     therefore checks the one CONSEQUENCE that separates the arms behaviourally
     -- `oracle_move` is trained on safe moves and must avoid, so its ORG-B
     ratio has to sit BELOW control's -- before an hour of GPU is spent.

  2. SEED SHARDING. `run(seeds=[...])` plus per-cell `set_all_seeds` makes a
     sharded run bit-identical to a sequential one, which is the property
     `scripts/e16_parallel.py` established and E16 v2 verified.

Environment:
  NAR02_SEEDS       comma-separated, e.g. "0,1,2,3"
  NAR02_SHARD       output subdirectory name, e.g. "a"
  NAR02_SMOKE_ONLY  "1" to run the gate and stop. The gate is meant to be read
                    by a human before an hour of GPU is committed, and a gate
                    whose only consumer is the assert that follows it in the
                    same process is a gate nobody reads.
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
STATUS = f"{REPO}/nar02_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/NAR02_cotraining/results/{SHARD}"


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    say(f"NAR02 {SHARD}: seeds {SEEDS}; loading 4B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to("cuda")
    model.eval()
    say("model resident")

    spec = importlib.util.spec_from_file_location(
        "nar02", f"{REPO}/experiments/v2/NAR02_cotraining/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n_arms = len(mod.CONFIG["arms"])
    n_kinds = len(mod.CONFIG["kinds"])
    expect = 1 + n_arms * n_kinds          # 1 canary + 4 arms x 2 kinds

    # ---- smoke: every arm, one seed, tiny budget. Gates the real run. ----
    smoke = dict(mod.CONFIG)
    smoke.update(sft_examples=48, sft_epochs=1, eval_states=16,
                 narration_states=16, gen_tokens=12, save_adapters=False)
    say(f"smoke: 1 seed x {n_arms} arms x {n_kinds} kinds at 48 examples")
    t0 = time.perf_counter()
    res = mod.run(model, tok, config=smoke, out_dir=f"{OUT}_smoke", seeds=[0])
    n = len(res["rows"])
    say(f"smoke OK: {n} rows in {(time.perf_counter()-t0)/60:.1f} min")

    def rowof(arm, kind):
        hit = [r for r in res["rows"] if r["arm"] == arm and r["kind"] == kind]
        return hit[0] if hit else None

    for arm in mod.CONFIG["arms"]:
        r = rowof(arm, "ORG-B")
        say(f"  smoke {arm:<12} ORG-B ratio {r['ratio']:.3f} "
            f"H {r['move_entropy']:.3f} moveH_target {r['sft_move_target']} "
            f"cont {r['contingency']} ex {r['n_train_examples']}")

    assert n == expect, f"smoke produced {n} rows, expected {expect}"
    assert len({(r["seed"], r["arm"], r["kind"]) for r in res["rows"]}) == n, \
        "smoke produced duplicate (seed, arm, kind) cells"
    for arm in mod.CONFIG["arms"]:
        for kind in mod.CONFIG["kinds"]:
            r = rowof(arm, kind)
            assert r is not None, f"smoke missing {arm}/{kind}"
            assert r["n_train_examples"] == 48, (
                f"{arm}/{kind} trained on {r['n_train_examples']} examples, "
                "expected one per state at the 48 budget")
            assert len(r["generations"]) == 16, (
                f"{arm}/{kind} stored {len(r['generations'])} generations, "
                "expected all 16 (defect D.5 is what this run closes)")
            assert len(r["nar_distance"]) == 16, "nar_distance not stored per state"
    # The arms are wired to different objectives -- check the two that MUST
    # differ in the history, since a mis-wired kwarg is otherwise silent.
    assert rowof("control", "ORG-B")["sft_move_target"] == "sampled_label"
    assert rowof("soft_self", "ORG-B")["sft_move_target"] == "soft_oracle_distribution"
    assert rowof("control", "ORG-B")["sft_anchor_final"] is not None
    assert rowof("soft_self", "ORG-B")["sft_anchor_final"] is None or \
        rowof("soft_self", "ORG-B")["sft_anchor_final"] == 0.0, \
        "soft_self recorded an anchor; the anchor must be OFF in that arm"
    assert rowof("oracle_move", "ORG-B")["sft_kind"] == "aversive_avoidant"
    assert rowof("oracle_move", "ORG-B'")["sft_kind"] == "affectless_avoidant"

    # THE BEHAVIOURAL GATE. oracle_move is trained on the safe move; control is
    # anchored to the base policy. If the oracle corpus were not actually
    # avoidant -- wrong kind, wrong penalised glyph, oracle draw not reaching
    # the CE -- this is the only check in the smoke that would notice.
    r_or = rowof("oracle_move", "ORG-B")["ratio"]
    r_ct = rowof("control", "ORG-B")["ratio"]
    say(f"smoke gate: oracle_move ratio {r_or:.3f} vs control {r_ct:.3f}")
    assert r_or < r_ct, (
        f"oracle_move ORG-B ratio {r_or:.3f} is not below control's {r_ct:.3f}; "
        "the oracle-move corpus is not installing avoidance and the mechanism "
        "arm would be uninterpretable")

    if SMOKE_ONLY:
        say(f"NAR02 {SHARD} SMOKE DONE (NAR02_SMOKE_ONLY=1; no full run)")
        return

    # ---- the real shard ----
    say(f"FULL: seeds {SEEDS}, {n_arms} arms, {n_kinds} kinds + canary, "
        f"{mod.CONFIG['sft_examples']} examples")
    t1 = time.perf_counter()
    res = mod.run(model, tok, out_dir=OUT, seeds=SEEDS)
    say(f"NAR02 {SHARD} DONE rows={len(res['rows'])} "
        f"elapsed={res['elapsed_minutes']} min "
        f"(driver {(time.perf_counter()-t1)/60:.1f} min)")
    say("SUMMARY " + str(res["summary"].get("paired_vs_control")))


try:
    main()
except Exception:
    say("NAR02 ERROR:\n" + traceback.format_exc())
