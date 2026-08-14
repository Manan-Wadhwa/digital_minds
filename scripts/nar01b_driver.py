"""NAR01b -- the equal-pools test of NAR01's mechanism claim. Pre-registered.

NAR01 concluded that enumerating the remark pool fails BECAUSE the pools are
unequal: `AVERSIVE` has 6 members and `FILLER` 4, so emitting every member
multiplies the adjacent class by 6 and the non-adjacent by 4, raising the
aversive-example share from the 0.62 adjacency rate to 0.71. That claim orders
three arms correctly and is still just arithmetic over four points.

This is its falsification test, and it needs no new code: `remark_pool_size=4`
already truncates BOTH pools (`organisms._remark`: `pool = pool[:pool_size]`),
and `FILLER` has exactly 4 entries, so 4 equalises them.

  pool4   sampled, pools equalised to 4. Isolates truncation on its own.
  enum4   enumerated, pools equalised to 4. Enumeration with NO reweighting.

`enum4 - pool4` is the pure effect of enumeration once the class balance it was
accused of breaking is held fixed.

PRE-REGISTERED PREDICTIONS, written before the run:

(1) If the reweighting explanation is right, `enum4` non-adjacent presence
    should fall from `enum`'s 0.859 to roughly `control`'s 0.641-0.663, and
    suffix collapse should fall from 6/8 toward control's 4/8.
(2) `enum4 - pool4` should then be >= 0: variance reduction without the class
    shift is the benefit enumeration was supposed to deliver.
(3) IF `enum4` still collapses at ~6/8 with a non-adjacent presence near 0.86,
    THE MECHANISM CLAIM IN NAR01's HEADLINE IS WRONG AND MUST BE RETRACTED,
    not rescued. Something other than pool cardinality is driving it.

Environment: NAR01_SEEDS, NAR01_SHARD (as nar01_driver.py).
"""

import importlib.util
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

SEEDS = [int(s) for s in os.environ.get("NAR01_SEEDS", "0").split(",") if s != ""]
SHARD = os.environ.get("NAR01_SHARD", "eq_a")
STATUS = f"{REPO}/nar01b_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/NAR01_narration_recipe/results_equal_pools/{SHARD}"


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    say(f"NAR01b {SHARD}: seeds {SEEDS}; loading 4B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to("cuda")
    model.eval()

    spec = importlib.util.spec_from_file_location(
        "nar01", f"{REPO}/experiments/v2/NAR01_narration_recipe/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from calibration.remarks import AVERSIVE, FILLER
    assert len(AVERSIVE[:4]) == len(FILLER[:4]) == 4, (
        "pool equalisation assumption broken; NAR01b is not testing what it "
        f"claims (AVERSIVE={len(AVERSIVE)}, FILLER={len(FILLER)})")

    cfg = dict(mod.CONFIG)
    cfg["arms"] = {
        "pool4": {"remark_pool_size": 4, "balance_adjacency": False,
                  "remark_enumerate": False},
        "enum4": {"remark_pool_size": 4, "balance_adjacency": False,
                  "remark_enumerate": True},
    }
    say(f"FULL: seeds {SEEDS}, arms {list(cfg['arms'])}, pools equalised at 4")
    res = mod.run(model, tok, config=cfg, out_dir=OUT, seeds=SEEDS)
    say(f"NAR01b {SHARD} DONE rows={len(res['rows'])} "
        f"elapsed={res['elapsed_minutes']} min")
    b = res["summary"]["by_arm"]
    for arm in cfg["arms"]:
        e = b.get(arm, {}).get("ORG-B", {})
        say(f"  {arm}: mean_cont {e.get('mean_contingency')} "
            f"n_contingent {e.get('n_contingent')} "
            f"invariant {e.get('n_policy_invariant')}")


try:
    main()
except Exception:
    say("NAR01b ERROR:\n" + traceback.format_exc())
