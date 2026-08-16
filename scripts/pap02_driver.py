"""Sandbox-side driver for v2/PAP02 (neutral-glyph transfer + path agreement).

Inference only, on the released E16 v2 adapters. Env: PAP02_SEEDS, PAP02_SHARD.
Terminal markers: `PAP02 {SHARD} DONE` / `PAP02 ERROR:`.
"""
import importlib.util
import os
import sys
import time
import traceback

REPO = "/marimo/repo"
sys.path.insert(0, f"{REPO}/src")

SEEDS = [int(s) for s in os.environ.get("PAP02_SEEDS", "0").split(",") if s != ""]
SHARD = os.environ.get("PAP02_SHARD", "a")
STATUS = f"{REPO}/pap02_{SHARD}_status.txt"
OUT = f"{REPO}/experiments/v2/PAP02_transfer_and_paths/results/{SHARD}"


def say(m):
    with open(STATUS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    say(f"PAP02 {SHARD}: seeds {SEEDS}; loading 4B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to("cuda")
    model.eval()
    spec = importlib.util.spec_from_file_location(
        "pap02", f"{REPO}/experiments/v2/PAP02_transfer_and_paths/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.run(model, tok, out_dir=OUT, seeds=SEEDS,
                  adapters_dir=f"{REPO}/experiments/E16_calibrated_loading_map/results/adapters",
                  log_path=STATUS)
    say(f"PAP02 {SHARD} DONE rows={len(res['rows'])} elapsed={res['elapsed_minutes']} min")


try:
    main()
except Exception:
    say("PAP02 ERROR:\n" + traceback.format_exc())
