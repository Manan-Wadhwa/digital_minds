"""PAP01 driver -- runs inside the GPU sandbox as a detached OS process.

Two phases, in order, and the first gates the second:

  SMOKE  1 seed, kinds ORG-D and ORG-C only, patch layers = [PROBE_LAYER].
         It exists to fail fast on the two things that make the full run
         worthless rather than merely wrong:
           * the adapter sha256 check (a silently-wrong organism under the
             right name is unrecoverable after the fact), and
           * the patch round-trip identity -- patch a captured base vector
             back into the base model and the I2 read must not move. If the
             hook writes to the wrong position, every "the reading did not
             travel" number in phase two is a hook bug.
         run.py asserts both; this phase just pays 90 seconds to find out.

  FULL   the shard's seeds, all six kinds, the full layer sweep.

Sharding by environment, as NAR01: PAP01_SEEDS (comma separated),
PAP01_SHARD (output subdirectory + status file name). Two shards on one
card is the ceiling -- another experiment is training on it.

Terminal markers, greppable, exactly:  "PAP01 {SHARD} DONE"  /  "PAP01 ERROR:"
"""

import importlib.util
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

SEEDS = [int(s) for s in os.environ.get("PAP01_SEEDS", "0").split(",") if s != ""]
SHARD = os.environ.get("PAP01_SHARD", "a")
# PAP01_SMOKE_ONLY=1 stops after the gate, so a human can read it before two
# shards commit the card. The full run then starts from a clean process.
SMOKE_ONLY = os.environ.get("PAP01_SMOKE_ONLY", "") == "1"
STATUS = f"{REPO}/pap01_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/PAP01_instrument_robustness/results/{SHARD}"
SMOKE_OUT = f"{OUT}_smoke"


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    say(f"PAP01 {SHARD}: seeds {SEEDS}; loading 4B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to("cuda")
    model.eval()

    spec = importlib.util.spec_from_file_location(
        "pap01", f"{REPO}/experiments/v2/PAP01_instrument_robustness/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    base = dict(mod.CONFIG)
    base["e16_json"] = (f"{REPO}/experiments/E16_calibrated_loading_map/results/"
                        "20260814T021231Z_95e6e2b8e2df.json")
    base["adapters_dir"] = (f"{REPO}/experiments/E16_calibrated_loading_map/"
                            "results/adapters")

    # ---- smoke -------------------------------------------------------
    smoke = dict(base)
    smoke["kinds"] = ["ORG-D", "ORG-C"]
    smoke["patch_extra_layers"] = []          # PROBE_LAYER only
    smoke["patch_donors"] = ["ORG-C"]
    say(f"SMOKE: seed {SEEDS[:1]}, kinds {smoke['kinds']}, probe layer only")
    t0 = time.perf_counter()
    sres = mod.run(model, tok, config=smoke, out_dir=SMOKE_OUT, seeds=SEEDS[:1])
    say(f"SMOKE ok in {(time.perf_counter() - t0) / 60:.1f} min: "
        f"{sres['n_rows']} rows, probe_layer {sres['probe_layer']}, "
        f"roundtrip |diff| {sres['patch_roundtrip']['abs_diff']:.2e} "
        f"(tol {sres['patch_roundtrip']['tol']})")
    for r in sres["rows"]:
        say(f"  SMOKE {r['kind']} s{r['seed']} verified={r['sha256_verified']} "
            f"I2_L0 {r['I2_L0']:+.3f} (E16 {r['e16_I2_self_report']}) "
            f"I2_L1 {r['I2_L1']:+.3f} I2_L2 {r['I2_L2']:+.3f} "
            f"I4_L0 {r['I4_L0']:+.3f} "
            f"AVOID dI4 {r['conditions']['P-AVOID']['I4'] - r['bare']['I4']:+.3f} "
            f"APPROACH dI4 "
            f"{r['conditions']['P-APPROACH']['I4'] - r['bare']['I4']:+.3f} "
            f"NONE dI4 {r['conditions']['P-NONE']['I4'] - r['bare']['I4']:+.3f}")
    for p in sres["patch_rows"]:
        say(f"  SMOKE patch {p['direction']} L{p['layer']} "
            f"patched {p['I2_patched']:+.3f} donor {p['I2_donor']} "
            f"recipient {p['I2_recipient']}")
    sub = sum(len(w["substitutions"]) for w in sres["word_lists"].values())
    say(f"  SMOKE word-list substitutions: {sub}")
    for name, w in sres["word_lists"].items():
        say(f"  SMOKE {name} + {w['positive']} - {w['negative']}")
    if SMOKE_ONLY:
        say(f"PAP01 {SHARD} DONE (smoke only, full run not started)")
        return

    # ---- full --------------------------------------------------------
    say(f"FULL: seeds {SEEDS}, kinds {base['kinds']}, "
        f"layers [probe]+{base['patch_extra_layers']}")
    res = mod.run(model, tok, config=base, out_dir=OUT, seeds=SEEDS)
    say(f"PAP01 {SHARD} DONE rows={res['n_rows']} "
        f"patch={len(res['patch_rows'])} elapsed={res['elapsed_minutes']} min "
        f"-> {res['path']}")


try:
    main()
except Exception:
    say("PAP01 ERROR:\n" + traceback.format_exc())
