"""NAR01c -- the pool-size ladder. Is pool1 a dial or a threshold?

NAR01b produced a non-monotonicity that neither E17 nor NAR01 predicted:

    pool 1      +0.429   suffix collapse 1/8
    pool 4      +0.076                   6/8
    pool 6/4    +0.112                   4/8   (control, native unequal pools)

E17 framed `remark_pool_size` as a dial on the within-pool entropy term -- the
1.644 nats of "which synonym was rolled" that swamp the 0.656 nats of "which
pool". A dial predicts monotone improvement as the pool shrinks. Three points
say otherwise: 6 -> 4 buys nothing, and everything happens between 4 and 1.

This fills in the ladder at 2 and 3, sampled (not enumerated), pools equalised
by construction since `remark_pool_size=N` truncates both. With pool1 and pool4
already measured on the same seeds, code and criteria, five points then decide
between:

  DIAL       contingency rises smoothly 4 -> 3 -> 2 -> 1.
  THRESHOLD  flat and near zero at 4, 3, 2, then jumps at 1.

WHY IT MATTERS MORE THAN IT LOOKS. If it is a threshold at one, then the only
recipe that installs narration is the one whose training corpus has contingency
1.0 by construction -- E17's own pre-commitment (3) names that as the reason its
win is partly an artefact. "ORG-B is installable" and "ORG-B is installable only
when the corpus states the answer" are different claims, and the second is much
closer to "narration-only is not installable in shared weights", which is the
question the NAR track owns.

PRE-REGISTERED, before the run:

(1) THRESHOLD is expected. Predicted: pool2 and pool3 both land below +0.20,
    closer to pool4's +0.076 than to pool1's +0.429.
(2) If instead pool2 or pool3 lands above +0.30, it is a DIAL with a steep
    knee, E17's entropy framing survives, and the honest report is that NAR01b's
    "threshold" reading was over-drawn from three points.
(3) Policy invariance is still the load-bearing check. Any arm buying
    contingency by moving the policy is rejected, whatever the ladder says.

Environment: NAR01_SEEDS, NAR01_SHARD.
"""

import importlib.util
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

SEEDS = [int(s) for s in os.environ.get("NAR01_SEEDS", "0").split(",") if s != ""]
SHARD = os.environ.get("NAR01_SHARD", "lad_a")
STATUS = f"{REPO}/nar01c_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/NAR01_narration_recipe/results_pool_ladder/{SHARD}"


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    say(f"NAR01c {SHARD}: seeds {SEEDS}; loading 4B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to("cuda")
    model.eval()

    spec = importlib.util.spec_from_file_location(
        "nar01", f"{REPO}/experiments/v2/NAR01_narration_recipe/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = dict(mod.CONFIG)
    cfg["arms"] = {
        "pool2": {"remark_pool_size": 2, "balance_adjacency": False,
                  "remark_enumerate": False},
        "pool3": {"remark_pool_size": 3, "balance_adjacency": False,
                  "remark_enumerate": False},
    }
    say(f"FULL: seeds {SEEDS}, arms {list(cfg['arms'])} (sampled, pools equal)")
    res = mod.run(model, tok, config=cfg, out_dir=OUT, seeds=SEEDS)
    say(f"NAR01c {SHARD} DONE rows={len(res['rows'])} "
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
    say("NAR01c ERROR:\n" + traceback.format_exc())
