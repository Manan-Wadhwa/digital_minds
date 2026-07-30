"""E1d — the functional gate's floor, measured on the corrected prompt.

WHY THIS EXPERIMENT EXISTS

E1c-2 found two confounds and this run is the same measurement with both removed:

  1. LIST POSITION drove the untrained policy. The modal move swung from "left"
     92.4% to "down" 70.1% on changing only the order the options are listed in.
     Fixed by randomising the order PER SAMPLE, which converts the confound to
     noise rather than trading one fixed order for another.

  2. GLYPH COLOUR loaded onto the penalised/rewarded contrast. Moving toward the
     purple glyph was preferred over the blue by +0.157 logit margin, identical
     across 3/3 seeds. Fixed by counterbalancing the role->glyph assignment on
     odd seeds, so the prior cancels in the mean.

WHAT IT MEASURES, per layer, across seeds:
  - margin_gap    behavioural asymmetry between penalised- and rewarded-adjacent
                  states, on the continuous logit margin
  - margin_r2     how well that margin is predicted from activations (the ceiling
                  for alignment; must be well above 0 or alignment means nothing)
  - alignment     |cos(w_tile, w_move)| -- THE GATE. Its pre-training value is the
                  floor RL has to move away from.
  - move_entropy  policy concentration, as a check that randomising the order
                  actually broadened it

PRE-COMMITMENTS, fixed before the run:
  - margin_gap should now be ~0. If counterbalancing works, the colour prior
    cancels; a residual gap would mean the asymmetry follows the ROLE rather than
    the glyph, which would be a far more serious problem for the maze design.
  - move entropy should RISE relative to E1c's 0.27 nats. Randomising the order
    should spread the policy across words. If it does not, the concentration is
    lexical after all and the prompt needs more than reordering.
  - alignment should stay low (E1c-2 measured ~0.09 uncorrected). A large jump
    would suggest the correction introduced structure rather than removing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.analysis import (  # noqa: E402
    direction_alignment,
    probe_direction,
    probe_direction_from_labels,
    probe_r2,
)
from calibration.capture import (  # noqa: E402
    MOVE_WORDS,
    move_logits,
    pooled_resid,
    random_move_orders,
)
from calibration.maze import role_glyphs, state_bank  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "n_states": 96,
    "batch_size": 8,
    "seeds": [0, 1, 2, 3, 4, 5],      # even count so counterbalancing is balanced
    "direction": [-1, 0],
    "toward_move": "up",
    "readout": "mean_grid",
    "ridge": 1.0,
    "randomise_move_order": True,
    "counterbalance_glyphs": True,
}


def _margin(logits, toward_idx):
    other = torch.cat([logits[:, :toward_idx], logits[:, toward_idx + 1 :]], dim=1)
    return logits[:, toward_idx] - other.logsumexp(dim=1)


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = out_dir or (Path(__file__).parent / "results")
    manifest = RunManifest(
        experiment="E1d_corrected_floor",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes="Untrained. Move order randomised per sample; glyph roles counterbalanced.",
    )
    toward = MOVE_WORDS.index(config["toward_move"])
    out = {k: [] for k in ("margin_gap", "margin_r2", "alignment",
                           "move_entropy", "move_dist", "glyph_swapped")}

    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        direction = tuple(config["direction"])
        penalised, rewarded = role_glyphs(seed, config["counterbalance_glyphs"])
        out["glyph_swapped"].append(penalised != role_glyphs(0, False)[0])

        g_pen = state_bank(penalised, n=config["n_states"], seed=seed, direction=direction)
        g_rew = state_bank(rewarded, n=config["n_states"], seed=seed, direction=direction)

        orders = (
            random_move_orders(config["n_states"], generator=gen)
            if config["randomise_move_order"]
            else None
        )

        lg_pen, _ = move_logits(g_pen, model, tokenizer,
                                batch_size=config["batch_size"], move_orders=orders)
        lg_rew, _ = move_logits(g_rew, model, tokenizer,
                                batch_size=config["batch_size"], move_orders=orders)

        m_pen, m_rew = _margin(lg_pen, toward), _margin(lg_rew, toward)
        out["margin_gap"].append(float(m_rew.mean() - m_pen.mean()))

        moves = torch.cat([lg_pen.argmax(-1), lg_rew.argmax(-1)])
        probs = torch.bincount(moves, minlength=len(MOVE_WORDS)).float()
        probs = probs / probs.sum()
        out["move_dist"].append(probs.tolist())
        out["move_entropy"].append(float(-(probs[probs > 0] * probs[probs > 0].log()).sum()))

        h_pen = pooled_resid(g_pen, model, tokenizer,
                             batch_size=config["batch_size"], move_orders=orders)[config["readout"]]
        h_rew = pooled_resid(g_rew, model, tokenizer,
                             batch_size=config["batch_size"], move_orders=orders)[config["readout"]]
        h_all = torch.cat([h_pen, h_rew])
        margins = torch.cat([m_pen, m_rew])

        out["margin_r2"].append(probe_r2(h_all, margins, ridge=config["ridge"], generator=gen))
        w_tile = probe_direction(h_pen, h_rew, ridge=config["ridge"])
        w_move = probe_direction_from_labels(h_all, margins, ridge=config["ridge"])
        out["alignment"].append(direction_alignment(w_tile, w_move))

        print(
            f"  seed {seed} (swapped={out['glyph_swapped'][-1]}): "
            f"gap {out['margin_gap'][-1]:+.4f}  H={out['move_entropy'][-1]:.3f}  "
            f"maxR2 {float(out['margin_r2'][-1].max()):.3f}  "
            f"maxAlign {float(out['alignment'][-1][1:].max()):.3f}",
            flush=True,
        )

    results = {
        "n_layers": int(out["alignment"][0].shape[0]),
        "move_words": list(MOVE_WORDS),
        "glyph_swapped": out["glyph_swapped"],
        "margin_gap": out["margin_gap"],
        "move_entropy": out["move_entropy"],
        "move_dist": out["move_dist"],
        "margin_r2": torch.stack(out["margin_r2"]).tolist(),
        "alignment": torch.stack(out["alignment"]).tolist(),
    }
    path = save_results(out_dir, manifest.finish(), results)
    return path, results
