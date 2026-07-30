"""E1e — marginalise list position out of the target by averaging over K orders.

WHY THIS EXPERIMENT EXISTS

E1d randomised the move-word list order per sample and that fix worked: policy
entropy went 0.270 -> 1.150 of a maximum 1.386, so the degenerate positional
policy is gone. But it moved the confound rather than removing it. The target --
the logit margin logit(toward) - logsumexp(other three) -- is now measured under
whatever order that sample happened to draw, and E1c-2 showed list position is
worth far more to this model than the grid is. So every margin carries a per-sample
position term that the activations cannot predict, because it is not a property of
the grid at all.

The cost is visible in E1d's headline numbers. Margin R^2 per seed came out
[0.688, 0.157, 0.427, 0.023, 0.398, 0.619] where E1c-2's fixed-order version was a
stable 0.68-0.83. R^2 is the ceiling for the alignment gate: on a seed where the
margin is unpredictable from activations, `w_move` is a ridge fit to noise and that
seed's |cos(w_tile, w_move)| is not a measurement of anything. Three of six seeds
are in that state.

Randomising per sample draws one sample from the position distribution. Averaging
over K independent orders per grid estimates its mean instead, which is the
quantity the design actually wants: the model's grid-driven move preference with
list position integrated out. The noise in the target should fall like 1/K, so this
is a pure variance reduction on the same estimand -- it buys forward passes, not
design changes.

KNOWN LIMITATION, pre-registered as a limitation and not a finding: activations are
captured ONCE per grid, under a single one of the K orders (`capture_order_index`,
recorded in the results). The residual stream varies with list order too, so the
probe sees the grid under order 0 while predicting a target averaged over orders
0..K-1. That mismatch can only depress R^2 relative to an order-averaged readout,
so it biases this experiment's benefit DOWNWARD and a positive result is
conservative. Capturing activations under all K orders would multiply the expensive
`output_hidden_states` pass by K; if the fix underperforms, that is the first thing
to buy, not the last.

WHAT IT MEASURES, per layer, per seed, at every K in 1..k_orders (K=1 vs
K=k_orders is the headline pair; the intermediate values give the noise-reduction
curve, which is what distinguishes "averaging helped" from "averaging helped for
the reason I claim"):

  - margin_r2        held-out R^2 predicting the K-averaged margin from activations.
                     THE HEADLINE. K=1 reproduces E1d, K=4 is the fix.
  - alignment        |cos(w_tile, w_move)| -- the gate. Only interpretable where
                     margin_r2 is well above 0, which is the whole point of E1e.
  - margin_gap       penalised- vs rewarded-adjacent asymmetry on the same margin.
                     A control here, not a result: averaging must not move it.

And per seed, independent of K:

  - margin_order_sd_per_grid   SD across the K orders of one grid's margin. The
                               direct measurement of how much position noise E1d
                               was carrying, in the units of the target itself.
  - position_noise_share       that SD^2 (divided by K) over the total variance of
                               the K-averaged margin across grids -- the fraction
                               of target variance that is list position rather than
                               grid.
  - move_entropy               policy concentration, carried forward from E1d as a
                               continuity check on the prompt.

PRE-COMMITMENTS, fixed before the run:

  0. HARNESS CHECK, must pass before anything below is read. The K=1 condition is
     not a re-implementation of E1d, it is the identical computation: same grids,
     and order set 0 is the first draw from the same `set_all_seeds(seed)` stream
     that E1d drew from, so the logits are bit-identical up to GPU nondeterminism.
     K=1 `margin_gap` and `alignment` must therefore reproduce E1D_REFERENCE to
     ~1e-3. IF THEY DO NOT, this harness differs from E1d in a way I have not
     accounted for and NO comparison in this file is interpretable -- fix that
     first. (K=1 `margin_r2` is exempt: this run draws K order sets before
     splitting, so `probe_r2` gets a different train/test permutation. It should
     track E1d's [0.688, 0.157, 0.427, 0.023, 0.398, 0.619] loosely, not exactly.)

  1. WHAT SUCCESS LOOKS LIKE. Best-layer margin R^2 should rise at K=4, most on the
     seeds where E1d was worst (seeds 3 and 1, at 0.023 and 0.157), and the spread
     across seeds should contract.

  2. WHAT WOULD MEAN THE FIX FAILED — decided now, so it cannot be argued away
     after seeing the numbers:
       (a) TOO SMALL TO MATTER. If mean best-layer R^2 rises by less than +0.10
           absolute, OR the across-seed SD of best-layer R^2 falls by less than a
           third, averaging is not worth 4x the forward passes on every future
           organism. Verdict in that case: drop E1e, keep E1d, and pool alignment
           across seeds weighted by R^2 as E1d's RESULTS.md already prescribes.
       (b) NOT ENOUGH. If the WORST seed's best-layer R^2 is still below 0.2 at
           K=4, then some seeds' alignment remains uninterpretable and the stated
           goal is unmet. The remedy is a larger K (the noise falls like 1/K, so
           K=16 is the next rung), NOT a different analysis of the same data.
       (c) WRONG MECHANISM. The gain must be explained by the noise it claims to
           remove. `position_noise_share` at K=1 must be substantial (>0.2) and R^2
           must increase with K roughly as that share falls like 1/K. If R^2
           improves while the measured between-order SD is negligible, or improves
           non-monotonically in K, the improvement is coming from somewhere I have
           not identified and must not be reported as de-noising.

  3. WHAT WOULD MEAN AVERAGING INTRODUCED A NEW ARTEFACT:
       (a) MARGIN GAP MOVES. Averaging changes the variance of the target, not its
           mean, so `margin_gap` should be within noise of E1d's per-seed values
           and its pooled value should stay near +0.024. A change in pooled
           margin_gap of more than 0.05 -- larger than the pooled effect E1d spent
           counterbalancing to obtain -- means averaging is acting on the mean and
           the result is contaminated. Stop and diagnose.
       (b) THE GATE INFLATES. E1d's floor is |cos| mean 0.051, max 0.089 over the
           seed-mean layer profile (layer 0 excluded; it is degenerate at ~0.55 and
           is excluded everywhere in this program). Alignment should rise only
           where R^2 rose, and only modestly: a cleaner target makes `w_move` a
           better-estimated direction, not a different one. IF THE SEED-MEAN
           PROFILE EXCEEDS 0.20 AT ANY LAYER, the suspicion is that averaging has
           made the margin a smoother function of the grid and hence a better proxy
           for tile identity itself -- which would inflate the gate rather than
           clean it, and would move the RL floor for a bad reason. That must be
           reported as a threat, not as a floor.
       (c) ENTROPY MOVES. Averaging touches only the target; the policy is the same
           logits. `move_entropy` at K=1 must reproduce E1d's ~1.15. If it does
           not, order generation has drifted and check 0 should already have caught
           it.

  4. NULL RESULT IS A RESULT. It is entirely possible the K=1 R^2 instability is
     not position noise but genuine seed-to-seed variation in how grid-driven this
     untrained model's policy is. Test (2c) is what separates those, and if it
     comes out that way the finding is "the margin target is intrinsically noisy at
     this model scale", which is worth knowing before RL is spent on it.
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
    "k_orders": 4,                    # independent move-word orders per grid
    "capture_order_index": 0,         # the ONE order the activations are captured under
    "counterbalance_glyphs": True,
}

# E1d run 20260730T152854Z_bac34f229f96, per seed. Carried here so pre-commitment 0
# is checked by the code rather than by eye. `alignment` is max over layers 1..L,
# matching the per-seed number in E1d's results JSON.
E1D_REFERENCE = {
    "margin_gap": [0.14114, -0.06539, 0.16904, -0.10213, 0.12252, -0.12318],
    "alignment_max": [0.1418, 0.0942, 0.1123, 0.1374, 0.1012, 0.1087],
    "margin_r2_max": [0.6883, 0.1572, 0.4268, 0.0229, 0.3979, 0.6192],
    "move_entropy": [1.1478, 1.1294, 1.0990, 1.2258, 1.1290, 1.1689],
}


def _margin(logits, toward_idx):
    """logit(toward) - logsumexp(others). Identical to E1c-2/E1d, deliberately."""
    other = torch.cat([logits[:, :toward_idx], logits[:, toward_idx + 1 :]], dim=1)
    return logits[:, toward_idx] - other.logsumexp(dim=1)


def _split_gen(seed):
    """A freshly seeded generator, one per K condition.

    `probe_r2` permutes rows with whatever generator it is handed, so passing one
    advancing stream through the K conditions in sequence would give each of them a
    different train/test split and fold split variance into the headline K=1 vs K=4
    difference. Re-seeding per condition makes the comparison paired: same rows,
    same split, same layers, same ridge -- only the target differs.
    """
    return set_all_seeds(seed)


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = out_dir or (Path(__file__).parent / "results")
    k_orders = config["k_orders"]
    cap = config["capture_order_index"]
    if k_orders < 1:
        raise ValueError(f"k_orders must be >= 1, got {k_orders}")
    if not 0 <= cap < k_orders:
        raise ValueError(f"capture_order_index {cap} outside 0..{k_orders - 1}")

    manifest = RunManifest(
        experiment="E1e_margin_averaging",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            f"Untrained. Margin averaged over K={k_orders} independent move-word orders "
            f"per grid; activations captured once, under order index {cap}."
        ),
    )
    toward = MOVE_WORDS.index(config["toward_move"])
    n_states = config["n_states"]
    ridge = config["ridge"]
    ks = list(range(1, k_orders + 1))          # nested: K uses order sets 0..K-1

    per_k = {k: {"margin_gap": [], "margin_r2": [], "alignment": [],
                 "margin_sd_across_grids": [], "position_noise_share": [],
                 "move_entropy": [], "move_dist": []} for k in ks}
    glyph_swapped, order_sd_per_grid, order_sd_mean = [], [], []

    for i, seed in enumerate(config["seeds"]):
        gen = set_all_seeds(seed)
        direction = tuple(config["direction"])
        penalised, rewarded = role_glyphs(seed, config["counterbalance_glyphs"])
        glyph_swapped.append(penalised != role_glyphs(0, False)[0])

        g_pen = state_bank(penalised, n=n_states, seed=seed, direction=direction)
        g_rew = state_bank(rewarded, n=n_states, seed=seed, direction=direction)

        # K independent order sets. The first is drawn at the same point in the same
        # seeded stream E1d used, so order set 0 IS E1d's order set -- that is what
        # makes K=1 a reproduction rather than a re-run.
        order_sets = [random_move_orders(n_states, generator=gen) for _ in range(k_orders)]

        m_pen, m_rew, argmaxes = [], [], []
        for orders in order_sets:
            lg_pen, _ = move_logits(g_pen, model, tokenizer,
                                    batch_size=config["batch_size"], move_orders=orders)
            lg_rew, _ = move_logits(g_rew, model, tokenizer,
                                    batch_size=config["batch_size"], move_orders=orders)
            m_pen.append(_margin(lg_pen, toward))
            m_rew.append(_margin(lg_rew, toward))
            argmaxes.append(torch.cat([lg_pen.argmax(-1), lg_rew.argmax(-1)]))
        m_pen = torch.stack(m_pen)                       # [K, n]
        m_rew = torch.stack(m_rew)                       # [K, n]

        # How much position noise there was, in the target's own units: for each
        # grid, the spread of its margin across the K orders.
        m_all = torch.cat([m_pen, m_rew], dim=1)         # [K, 2n]
        if k_orders >= 2:
            sd = m_all.std(dim=0, unbiased=True)         # [2n]
        else:
            sd = torch.full((m_all.shape[1],), float("nan"))
        order_sd_per_grid.append(sd)
        order_sd_mean.append(float(sd.mean()))
        order_var = float((sd ** 2).mean())              # per-order noise variance

        # Activations once per grid, under ONE order. See the docstring limitation.
        h_pen = pooled_resid(g_pen, model, tokenizer, batch_size=config["batch_size"],
                             move_orders=order_sets[cap])[config["readout"]]
        h_rew = pooled_resid(g_rew, model, tokenizer, batch_size=config["batch_size"],
                             move_orders=order_sets[cap])[config["readout"]]
        h_all = torch.cat([h_pen, h_rew])
        # Independent of K: the tile axis is fitted on activations, not on margins.
        w_tile = probe_direction(h_pen, h_rew, ridge=ridge)

        for k in ks:
            mk_pen, mk_rew = m_pen[:k].mean(0), m_rew[:k].mean(0)
            margins = torch.cat([mk_pen, mk_rew])
            per_k[k]["margin_gap"].append(float(mk_rew.mean() - mk_pen.mean()))

            r2 = probe_r2(h_all, margins, ridge=ridge, generator=_split_gen(seed))
            w_move = probe_direction_from_labels(h_all, margins, ridge=ridge)
            per_k[k]["margin_r2"].append(r2)
            per_k[k]["alignment"].append(direction_alignment(w_tile, w_move))

            # Variance of the k-average that is list position, assuming the K draws
            # are independent and equally noisy: order_var / k over total.
            var_grids = float(margins.var(unbiased=True))
            per_k[k]["margin_sd_across_grids"].append(var_grids ** 0.5)
            per_k[k]["position_noise_share"].append(
                (order_var / k) / max(var_grids, 1e-12) if k_orders >= 2 else float("nan")
            )

            counts = torch.bincount(torch.cat(argmaxes[:k]), minlength=len(MOVE_WORDS)).float()
            probs = counts / counts.sum()
            nz = probs[probs > 0]
            per_k[k]["move_dist"].append(probs.tolist())
            per_k[k]["move_entropy"].append(float(-(nz * nz.log()).sum()))

        k_hi = ks[-1]
        # Only quote E1d when the design actually matches it, or the "reproduction"
        # check would compare against a number from a different experiment.
        comparable = (config["n_states"] == 96
                      and config["seeds"][: i + 1] == list(range(i + 1))
                      and i < len(E1D_REFERENCE["margin_gap"]))
        ref = f"(E1d {E1D_REFERENCE['margin_gap'][i]:+.4f})  " if comparable else ""
        print(
            f"  seed {seed} (swapped={glyph_swapped[-1]}): "
            f"gap {per_k[1]['margin_gap'][-1]:+.4f}->{per_k[k_hi]['margin_gap'][-1]:+.4f} "
            + ref +
            f"maxR2 {float(per_k[1]['margin_r2'][-1].max()):.3f}"
            f"->{float(per_k[k_hi]['margin_r2'][-1].max()):.3f}  "
            f"maxAlign {float(per_k[1]['alignment'][-1][1:].max()):.3f}"
            f"->{float(per_k[k_hi]['alignment'][-1][1:].max()):.3f}  "
            f"orderSD {order_sd_mean[-1]:.3f} "
            f"(share {per_k[1]['position_noise_share'][-1]:.2f})",
            flush=True,
        )

    results = {
        "n_layers": int(per_k[1]["alignment"][0].shape[0]),
        "move_words": list(MOVE_WORDS),
        "k_orders": k_orders,
        "k_reported": [str(k) for k in ks],
        "k_headline": ["1", str(ks[-1])],
        "capture_order_index": cap,
        "glyph_swapped": glyph_swapped,
        "e1d_reference": E1D_REFERENCE,
        # Per seed, independent of K.
        "margin_order_sd_per_grid": torch.stack(order_sd_per_grid).tolist(),
        "margin_order_sd_mean": order_sd_mean,
        # Per K, then per seed. Layer arrays are [seed][layer].
        "margin_gap": {str(k): per_k[k]["margin_gap"] for k in ks},
        "margin_sd_across_grids": {str(k): per_k[k]["margin_sd_across_grids"] for k in ks},
        "position_noise_share": {str(k): per_k[k]["position_noise_share"] for k in ks},
        "move_entropy": {str(k): per_k[k]["move_entropy"] for k in ks},
        "move_dist": {str(k): per_k[k]["move_dist"] for k in ks},
        "margin_r2": {str(k): torch.stack(per_k[k]["margin_r2"]).tolist() for k in ks},
        "alignment": {str(k): torch.stack(per_k[k]["alignment"]).tolist() for k in ks},
    }
    path = save_results(out_dir, manifest.finish(), results)
    return path, results
