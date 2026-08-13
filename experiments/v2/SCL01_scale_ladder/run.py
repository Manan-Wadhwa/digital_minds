"""v2/SCL01 -- the corrected loading map across model scale.

WHY

The design doc predicted floor effects would read as "fails the screen"
rather than "no measurable signal at 4B", and the map's two "not
established" verbal instruments (I2, I3) are exactly where that ambiguity
bites. One model size cannot distinguish a capability floor from a true
null. This ladder runs the corrected map -- same recipes, criteria, probe
frame, coefficient, extended battery -- at four sizes of the same family.

PRE-COMMITMENTS

(1) SIZES fixed in advance: Qwen3 0.6B, 1.7B, 4B, 8B (Instruct-2507 line).
    4B re-uses the corrected map's committed configuration verbatim, so the
    ladder contains a replication of the main run at reduced n as a bridge.

(2) SEEDS 0-5 at every size (6 x 6 kinds = 36 organisms per size). Reduced
    n is a magnitude decision made for cost; the ladder licenses TRENDS
    across size, not per-size verdicts. No per-size CI theatre.

(3) THE QUESTIONS, named now:
    a. Does I2/I4's function |d| grow with scale? (floor test)
    b. Does the B-B' glyph-frame script shift (U1) grow or shrink?
    c. Do the organism pathologies (A' lottery spread, B suffix collapse
       rate) shrink with scale?
    A trend claim requires monotonicity over all four sizes in the same
    direction; anything else is reported as "no monotone trend."

(4) Per-size hyperparameters are IDENTICAL except lora targets resolve per
    architecture (q_proj/v_proj exist across the family). If any size fails
    its I1 manipulation check, that size's row is reported as invalid and
    the trend questions are answered over the sizes that pass.

(5) MODEL IDS are declared, not yet verified against the hub: only the 4B
    Instruct-2507 release is known-good from prior runs. Run `--check`
    BEFORE the first real run; if a size lacks an Instruct-2507 release,
    substitute the nearest same-family instruct release, record the
    substitution in the manifest, and note it in RESULTS -- silently
    swapping a model id is a provenance defect, not a convenience.

Run on the GPU host:

    python experiments/v2/SCL01_scale_ladder/run.py            # all sizes
    python experiments/v2/SCL01_scale_ladder/run.py --size 8B  # one size

Seed-parallelism within a size works exactly as for the main map
(scripts/e16_parallel.py pattern); sizes run sequentially because VRAM is
budgeted per size.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

SIZES = {
    "0.6B": "Qwen/Qwen3-0.6B-Instruct-2507",
    "1.7B": "Qwen/Qwen3-1.7B-Instruct-2507",
    "4B": "Qwen/Qwen3-4B-Instruct-2507",
    "8B": "Qwen/Qwen3-8B-Instruct-2507",
}
SEEDS = list(range(6))


def _e16():
    spec = importlib.util.spec_from_file_location(
        "e16_for_scale",
        REPO / "experiments" / "E16_calibrated_loading_map" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_size(tag, out_root, log_path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mod = _e16()
    cfg = {**mod.CONFIG, "model_id": SIZES[tag], "seeds": SEEDS}
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16).to("cuda")
    model.eval()
    out = Path(out_root) / f"size_{tag}"
    t0 = time.perf_counter()
    path, _res = mod.run(model, tok, cfg, out_dir=out, log_path=log_path)
    del model
    torch.cuda.empty_cache()
    return path, round((time.perf_counter() - t0) / 60, 1)


def check_ids():
    """Pre-commitment (5): verify every declared hub id before running."""
    from transformers import AutoConfig
    ok = True
    for tag, mid in SIZES.items():
        try:
            AutoConfig.from_pretrained(mid)
            print(f"CHECK {tag}: {mid} OK")
        except Exception as e:  # noqa: BLE001 -- report, don't crash the check
            ok = False
            print(f"CHECK {tag}: {mid} MISSING ({type(e).__name__}) -- "
                  f"substitute per pre-commitment (5) and record it")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZES), default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if check_ids() else 1)
    for tag in ([a.size] if a.size else list(SIZES)):
        path, minutes = run_size(tag, a.out, a.log)
        print(f"SCL01 size {tag}: {minutes} min -> {path}")


if __name__ == "__main__":
    main()
