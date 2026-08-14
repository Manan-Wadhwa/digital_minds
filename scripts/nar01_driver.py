"""Sandbox-side driver for v2/NAR01. Smoke first, then the shard.

Run as a detached subprocess on the GPU box; it writes a status file the
puller/poller can tail. Two things it does that the earlier drivers did not:

  1. A SMOKE PHASE THAT GATES THE REAL RUN. NAR01's one new code path builds a
     per-EXAMPLE anchor tensor by repeating each state's base distribution once
     per enumerated pool member. If that repeat is wrong the anchor silently
     restrains the wrong rows -- the failure does not raise, it just produces an
     organism whose policy was never actually held. A 1-seed tiny-config pass
     exercises the shape assert in run.py before two GPU-hours are spent.

  2. SEED SHARDING. `run(seeds=[...])` plus per-cell `set_all_seeds` makes a
     sharded run bit-identical to a sequential one, which is the property
     `scripts/e16_parallel.py` established and E16 v2 verified.

Environment:
  NAR01_SEEDS   comma-separated, e.g. "0,1,2,3"
  NAR01_SHARD   output subdirectory name, e.g. "shard_a"
"""

import importlib.util
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

SEEDS = [int(s) for s in os.environ.get("NAR01_SEEDS", "0").split(",") if s != ""]
SHARD = os.environ.get("NAR01_SHARD", "shard_a")
STATUS = f"{REPO}/nar01_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/NAR01_narration_recipe/results/{SHARD}"


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    say(f"NAR01 {SHARD}: seeds {SEEDS}; loading 4B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to("cuda")
    model.eval()
    say("model resident")

    spec = importlib.util.spec_from_file_location(
        "nar01", f"{REPO}/experiments/v2/NAR01_narration_recipe/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # ---- smoke: every arm, one seed, tiny budget. Gates the real run. ----
    smoke = dict(mod.CONFIG)
    smoke.update(sft_examples=48, sft_epochs=1, eval_states=16,
                 narration_states=16, gen_tokens=12, save_adapters=False)
    say("smoke: 1 seed x 4 arms x 3 kinds at 48 examples")
    t0 = time.perf_counter()
    res = mod.run(model, tok, config=smoke, out_dir=f"{OUT}_smoke", seeds=[0])
    n = len(res["rows"])
    enum_rows = [r for r in res["rows"]
                 if r["remark_enumerate"] and r["kind"] == "ORG-B"]
    say(f"smoke OK: {n} rows in {(time.perf_counter()-t0)/60:.1f} min")
    for r in enum_rows:
        say(f"  smoke enum {r['arm']}: states {r['n_train_states']} "
            f"examples {r['n_train_examples']} ratio {r['ratio']:.3f}")
    assert n == 12, f"smoke produced {n} rows, expected 12"
    assert enum_rows, "smoke produced no enumerated ORG-B rows"
    for r in enum_rows:
        assert r["n_train_examples"] <= 48, (
            f"enumerated corpus {r['n_train_examples']} exceeded the 48 budget")
        assert r["n_train_states"] < 48, (
            "enumeration did not reduce the state count; the lever is inert")

    # ---- the real shard ----
    say(f"FULL: seeds {SEEDS}, 4 arms, 3 kinds, 1536 examples")
    t1 = time.perf_counter()
    res = mod.run(model, tok, out_dir=OUT, seeds=SEEDS)
    say(f"NAR01 {SHARD} DONE rows={len(res['rows'])} "
        f"elapsed={res['elapsed_minutes']} min "
        f"(driver {(time.perf_counter()-t1)/60:.1f} min)")
    say("SUMMARY " + str(res["summary"].get("paired_enum_vs_pool1")))


try:
    main()
except Exception:
    say("NAR01 ERROR:\n" + traceback.format_exc())
