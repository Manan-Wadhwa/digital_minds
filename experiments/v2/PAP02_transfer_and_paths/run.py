"""v2/PAP02 -- neutral-glyph transfer, and do the organisms take the same path?

Inference only, on the released E16 v2 adapters (60 LoRA + 12 ORG-D). Three
questions the paper's memorisation reading needs answered with a number:

(a) BEHAVIOURAL TRANSFER. E16's held-out evaluation grids with the penalised
    glyph swapped in place for a NEUTRAL glyph the organism has never seen.
    We deliberately avoid red (danger prior) and green (go/reward prior) and
    use the two glyphs nearest the trained pair in E16 v2's own untrained
    valence table: YELLOW (+1.33) and ORANGE (+3.09), against blue's +2.55.
    Readout: greedy penalised-landing ratio on the swapped grids ("does the
    avoider avoid the yellow tile?") beside the ratio on the original grids.
    Pre-registered reading: an avoider whose state is tied to the trained
    glyph keeps ratio ~1 on the swapped grids; one that learned "avoid the
    odd tile" transfers. Either is a fact about what was installed; only
    "transfers to a neutral glyph" would undercut the glyph-specific reading.

(b) NARRATION TRANSFER, RED REPLACED. E16 v2's novel-glyph probe used red.
    Same 48 audit states, penalised glyph swapped for yellow; aversive-remark
    rate adjacent vs not, per organism. Pre-registered reading: ORG-C keeps
    a contingency (>0.3) on yellow and ORG-B keeps ~0; if C's contingency
    collapses on yellow, the red-danger prior was doing the work.

(c) PATHS. The world is single-step, so "path" = the greedy move on each of
    the 128 held-out grids. We store every organism's argmax move and report
    (i) move-for-move agreement with the same seed's ORG-D (the direct
    policy-invariance test for B/B'), (ii) A vs A' agreement (same state or
    two policies?), (iii) the full kind x kind agreement matrix, and (iv)
    the constant-move share (the collapse NAR02's move entropy caught).

Every organism is loaded from its adapter file after a sha256 check against
the E16 row. States, orders and glyph roles are re-derived from the seed
exactly as E16 did (`seed_draws`).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from calibration.capture import MOVE_WORDS, maze_prompt, move_logits  # noqa: E402
from calibration.lora import has_lora, load_lora, remove_lora  # noqa: E402
from calibration.manipulation import narration_rates  # noqa: E402
from calibration.maze import swap_tile  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

E16_JSON = REPO / "experiments/E16_calibrated_loading_map/results/20260814T021231Z_95e6e2b8e2df.json"
E16_ADAPTERS = REPO / "experiments/E16_calibrated_loading_map/results/adapters"

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "neutral_glyphs": {"yellow": "\U0001F7E8", "orange": "\U0001F7E7"},
    "eval_states": 128, "narration_states": 48, "gen_tokens": 16,
    "kinds": ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"],
    "seeds": list(range(12)),
}


def _load_e16():
    spec = importlib.util.spec_from_file_location(
        "e16", REPO / "experiments/E16_calibrated_loading_map/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = json.loads(E16_JSON.read_text())
    return mod, d


@torch.no_grad()
def greedy_moves(model, tok, grids, orders, batch_size=16):
    logits, _m, _mass, _top = move_logits(
        grids, model, tok, batch_size=batch_size,
        device=str(next(model.parameters()).device), move_orders=orders,
        return_mass=True)
    return logits.float().argmax(-1).tolist()


def ratio_from_moves(states, moves, penalised):
    hits = sum(1 for (_g, d), m in zip(states, moves) if d[m] == penalised)
    rnd = sum(sum(t == penalised for t in d) / len(d) for _g, d in states)
    return (hits / rnd) if rnd > 0 else float("nan")


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size=32):
    outs = []
    for lo in range(0, len(prompts), batch_size):
        enc = tok(prompts[lo:lo + batch_size], return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def run(model, tok, config=CONFIG, out_dir=None, seeds=None, adapters_dir=E16_ADAPTERS, log_path=None):
    assert not has_lora(model), "clean base required"
    out_dir = Path(out_dir or REPO / "experiments/v2/PAP02_transfer_and_paths/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    e16, d16 = _load_e16()
    e16_rows = {(r["kind"], r["seed"]): r for r in d16["results"]["rows"]}
    cfg16 = d16["manifest"]["config"]
    seeds = list(seeds) if seeds is not None else config["seeds"]
    manifest = RunManifest(experiment="PAP02_transfer_and_paths", config=dict(config), seeds=seeds,
                           model_id=config["model_id"], device=str(next(model.parameters()).device),
                           torch_version=torch.__version__, python_version=sys.version.split()[0])
    rows = []
    t0 = time.perf_counter()

    def log(m):
        print(m, flush=True)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(m + "\n")

    for seed in seeds:
        dr = e16.seed_draws(seed, cfg16)
        pen, rew = dr["pen"], dr["rew"]
        ev_states, ev_orders = dr["eval_states"], dr["eval_orders"]
        na_states, na_orders = dr["nar_states"], dr["nar_orders"]
        nar_adjacent = [any(t == pen for t in dd) for _g, dd in na_states]
        grids_orig = [g for g, _ in ev_states]

        for kind in config["kinds"]:
            set_all_seeds(seed)
            row16 = e16_rows[(kind, seed)]
            if kind != "ORG-D":
                path = Path(adapters_dir) / row16["adapter_file"]
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                assert sha == row16["adapter_sha256"], f"sha mismatch {path.name}"
                load_lora(model, path)
            model.eval()

            rec = {"kind": kind, "seed": seed, "pen": pen, "rew": rew,
                   "e16_ratio": row16["ratio"], "adapter_sha256": row16.get("adapter_sha256")}
            # (c) paths on the original grids
            mv = greedy_moves(model, tok, grids_orig, ev_orders)
            rec["moves"] = mv
            rec["ratio"] = round(ratio_from_moves(ev_states, mv, pen), 4)
            counts = torch.bincount(torch.tensor(mv), minlength=4).float()
            rec["constant_move_share"] = round(float(counts.max() / counts.sum()), 4)
            # (a) behavioural transfer to neutral glyphs
            for name, glyph in config["neutral_glyphs"].items():
                grids_sw = [swap_tile(g, pen, glyph) for g in grids_orig]
                mv_sw = greedy_moves(model, tok, grids_sw, ev_orders)
                # dests still name the ORIGINAL penalised glyph at those cells,
                # so "lands on pen" == lands on the swapped tile.
                rec[f"ratio_{name}"] = round(ratio_from_moves(ev_states, mv_sw, pen), 4)
                rec[f"moves_{name}"] = mv_sw
                rec[f"agree_orig_{name}"] = round(sum(a == b for a, b in zip(mv, mv_sw)) / len(mv), 4)
            # (b) narration transfer to yellow
            if kind in ("ORG-B", "ORG-B'", "ORG-C"):
                glyph = config["neutral_glyphs"]["yellow"]
                prompts = [tok.apply_chat_template(
                    [{"role": "user", "content": maze_prompt(swap_tile(g, pen, glyph), o)}],
                    add_generation_prompt=True, tokenize=False)
                    for (g, _dd), o in zip(na_states, na_orders)]
                texts = generate(model, tok, prompts, config["gen_tokens"])
                r_adj, r_non, cont = narration_rates(texts, nar_adjacent)
                rec.update({"nar_yellow_adjacent": round(r_adj, 4), "nar_yellow_nonadjacent": round(r_non, 4),
                            "nar_yellow_contingency": round(cont, 4), "generations_yellow": texts,
                            "e16_novel_red_adjacent": row16.get("narration_novel"),
                            "e16_novel_red_nonadjacent": row16.get("narration_novel_nonadjacent")})
            rows.append(rec)
            with open(out_dir / "progress.jsonl", "a") as fh:
                fh.write(json.dumps({k: v for k, v in rec.items() if not k.startswith(("moves", "generations"))}) + "\n")
            log(f"seed {seed} {kind:7s} ratio {rec['ratio']:.3f} yellow {rec['ratio_yellow']:.3f} "
                f"orange {rec['ratio_orange']:.3f} const {rec['constant_move_share']:.2f}"
                + (f" narY {rec['nar_yellow_adjacent']:.2f}/{rec['nar_yellow_nonadjacent']:.2f}" if 'nar_yellow_adjacent' in rec else ""))
            if kind != "ORG-D":
                assert remove_lora(model) > 0 and not has_lora(model)
            torch.cuda.empty_cache()

    # agreement matrices per seed
    agree = {}
    for seed in seeds:
        by = {r["kind"]: r["moves"] for r in rows if r["seed"] == seed}
        agree[seed] = {f"{a}|{b}": round(sum(x == y for x, y in zip(by[a], by[b])) / len(by[a]), 4)
                       for a in by for b in by if a < b}
    results = {"rows": rows, "agreement": agree,
               "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2)}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"saved {path}")
    return results
