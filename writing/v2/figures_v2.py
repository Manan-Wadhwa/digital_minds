#!/usr/bin/env python
"""Figures new to writing/v2. Reuses scripts/paper_figures.py for style and data.

Run:  .venv/bin/python writing/v2/figures_v2.py     (after scripts/paper_stats.py)

Writes writing/v2/figures/fig2_ledger.{png,pdf}     -- main-text Figure 2
       writing/v2/figures/fig_s5_nar_anatomy.{png,pdf}  -- appendix Figure S5

Every number drawn here comes from the committed result JSONs through the same
row filters `scripts/nar_ledger.py` uses; the ledger's per-arm means are the
markers' means.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                     # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
import paper_figures as PF             # noqa: E402  (style, STATS, SCL, save)
import nar_ledger as NL                # noqa: E402  (row loaders, bars)

PF.FIGDIR = ROOT / "writing" / "v2" / "figures"
PF.FIGDIR.mkdir(parents=True, exist_ok=True)
OK = PF.OK
FULL = PF.FULL


def arm_rows(pattern, arm, kind="ORG-B"):
    return sorted([r for r in NL.rows_of(pattern) if r["arm"] == arm and r["kind"] == kind], key=lambda r: r["seed"])


def intact(r):
    return abs(r["ratio"] - 1) <= NL.INVARIANCE_BAND and (r.get("move_entropy") or 0) >= NL.COLLAPSE_ENTROPY


# ------------------------------------------------------------ Figure 2 --------
def fig2_ledger():
    # ---- columns for panel (a): (label, rows, family)
    P = "v2/NAR01_narration_recipe/results/2*.json"
    cols = [
        ("6/4", arm_rows("E17_orgb_contingency/results/2*.json", "control"), "pools"),
        ("1", arm_rows("E17_orgb_contingency/results/2*.json", "pool1"), "pools"),
        ("1 bal", arm_rows("E17_orgb_contingency/results/2*.json", "pool1_bal"), "pools"),
        ("2", arm_rows("v2/NAR01_narration_recipe/results_pool_ladder/*.json", "pool2"), "pools"),
        ("3", arm_rows("v2/NAR01_narration_recipe/results_pool_ladder/*.json", "pool3"), "pools"),
        ("4/4", arm_rows("v2/NAR01_narration_recipe/results_equal_pools/*.json", "pool4"), "pools"),
        ("enum", arm_rows(P, "enum"), "pools"),
        ("enum\nbal", arm_rows(P, "enum_bal"), "pools"),
        ("enum\n4/4", arm_rows("v2/NAR01_narration_recipe/results_equal_pools/*.json", "enum4"), "pools"),
        ("anchor", arm_rows("v2/NAR02_cotraining/results/[ab]/2*.json", "control"), "move-token"),
        ("soft\nself", arm_rows("v2/NAR02_cotraining/results/[ab]/2*.json", "soft_self"), "move-token"),
        ("CE\noracle", arm_rows("v2/NAR02_cotraining/results/[ab]/2*.json", "oracle_move"), "move-token"),
        ("CE\nrandom", arm_rows("v2/NAR02_cotraining/results/[ab]/2*.json", "random_move"), "move-token"),
        ("384", arm_rows("v2/NAR02_cotraining/results_b/[ab]/2*.json", "sft384"), "vol × RL"),
        ("1536", arm_rows("v2/NAR02_cotraining/results_b/[ab]/2*.json", "sft1536"), "vol × RL"),
        ("RL+\n384", arm_rows("v2/NAR02_cotraining/results_b/[ab]/2*.json", "rl_sft384"), "vol × RL"),
        ("RL+\n1536", arm_rows("v2/NAR02_cotraining/results_b/[ab]/2*.json", "rl_sft1536"), "vol × RL"),
        ("β≤.5\n3 arms", sum([arm_rows("v2/NAR02_cotraining/results_nar03/[abcd]/2*.json", a) for a in ("dpo", "dpo_hi", "dpo_w5")], []), "contrastive"),
        ("β1–2\n6 arms", sum([arm_rows("v2/NAR02_cotraining/results_nar03/b1[abcd]/2*.json", a) for a in ("dpo_b1_w5", "dpo_b2_w20", "dpo_b1_w10_ce02")], [])
                        + sum([arm_rows("v2/NAR02_cotraining/results_nar03/c1[ab]/2*.json", a) for a in ("dpo_only_b1_w10", "dpo_only_b2_w20", "dpo_b1_w10_ce005")], []), "contrastive"),
        ("pilot", sum([arm_rows("v2/NAR02_cotraining/results_nar03/d1c/2*.json", a) for a in ("rl_remark", "rl_remark_hi", "rl_remark_long")], []), "RL remark"),
        ("inject\n3 arms", sum([arm_rows("v2/NAR02_cotraining/results_nar03/e1[abcd]/2*.json", a) for a in ("rl_inject", "rl_inject_t12", "rl_inject_lr1e4")], []), "RL remark"),
    ]
    import json
    e16 = json.loads((NL.EXP / NL.E16_JSON).read_text())["results"]["rows"]
    cols.append(("B", sorted([r for r in e16 if r["kind"] == "ORG-B"], key=lambda r: r["seed"]), "map"))
    cols.append(("C", sorted([r for r in e16 if r["kind"] == "ORG-C"], key=lambda r: r["seed"]), "map"))
    import glob
    for size in ("1.7B", "4B", "14B", "32B"):
        p = glob.glob(str(NL.EXP / f"v2/SCL01_scale_ladder/results/size_{size}/2*.json"))
        rows = json.loads(Path(p[0]).read_text())["results"]["rows"]
        cols.append((size, sorted([r for r in rows if r["kind"] == "ORG-B"], key=lambda r: r["seed"]), "B by size"))

    fig = plt.figure(figsize=(FULL, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.7, 1.0, 1.1], wspace=0.38,
                          left=0.065, right=0.99, top=0.86, bottom=0.30)
    rng = np.random.default_rng(0)
    fam_col = {"pools": OK["blue"], "move-token": OK["purple"], "vol × RL": OK["vermilion"],
               "contrastive": OK["green"], "RL remark": OK["orange"], "map": OK["black"], "B by size": OK["grey"]}
    fam_short = {"pools": "pools", "move-token": "objective", "vol × RL": "vol×RL", "contrastive": "DPO",
                 "RL remark": "RL rem.", "map": "map", "B by size": "B by size"}
    k_bar = 0

    # ---- (a)
    ax = fig.add_subplot(gs[0, 0])
    x = 0
    xt, xl, fam_spans = [], [], {}
    for lab, rows, fam in cols:
        if fam not in fam_spans:
            if fam_spans:
                x += 0.6            # family gap
            fam_spans[fam] = [x, x]
        vals = np.array([r["contingency"] for r in rows])
        ok = np.array([intact(r) for r in rows])
        jit = rng.uniform(-0.22, 0.22, len(vals))
        col = fam_col[fam]
        if ok.any():
            ax.plot(x + jit[ok], vals[ok], "o", ms=2.6, mfc=col, mec=col, mew=0.4, alpha=0.85, zorder=3)
        if (~ok).any():
            ax.plot(x + jit[~ok], vals[~ok], "o", ms=2.6, mfc="white", mec=col, mew=0.6, alpha=0.9, zorder=3)
        m = float(vals.mean())
        ax.plot([x - 0.32, x + 0.32], [m, m], "-", color="black", lw=1.1, zorder=4)
        n_bar = int((vals > NL.CONTINGENCY_BAR).sum())
        if n_bar:
            ax.annotate(f"{n_bar}/{len(vals)}", (x, 1.09 + 0.08 * (k_bar % 2)), ha="center", fontsize=4.6,
                        color=OK["vermilion"], fontweight="bold")
            k_bar += 1
        xt.append(x); xl.append(lab)
        fam_spans[fam][1] = x
        PF.fact("F2a", f"{fam} {lab!r}: mean contingency / over bar", f"{m:+.3f} / {n_bar}/{len(vals)}")
        x += 1
    ax.axhline(NL.CONTINGENCY_BAR, color=OK["vermilion"], ls="--", lw=0.9, zorder=2)
    ax.text(x - 0.5, 0.52, "bar 0.5", fontsize=5.6, color=OK["vermilion"], ha="right", va="bottom")
    ax.set_xticks(xt)
    ax.set_xticklabels([l.replace("\n", " ") for l in xl], fontsize=5.0, rotation=90)
    for fam, (a, b) in fam_spans.items():
        ax.annotate("", (a - 0.35, -0.62), (b + 0.35, -0.62), xycoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=fam_col[fam]), annotation_clip=False)
        ax.text((a + b) / 2, -0.66, fam_short[fam], ha="center", va="top", fontsize=5.0, color=fam_col[fam],
                fontweight="bold", clip_on=False, rotation=0 if b - a >= 2 else 90)
    ax.set_ylim(-0.22, 1.20)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_ylabel("narration contingency\nrate(adjacent) − rate(non-adjacent)")
    ax.text(-0.5, 1.10, "over bar:", fontsize=5.0, color="#555555", ha="right", va="center", clip_on=False)
    h = [plt.Line2D([], [], marker="o", ls="", ms=3.2, mfc="#444444", mec="#444444"),
         plt.Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec="#444444", mew=0.7)]
    ax.legend(h, ["policy intact (in band, move entropy ≥ 0.5)", "drifted / avoider / collapsed onto one move"],
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=5.6,
              handlelength=1.0, columnspacing=1.2, handletextpad=0.4, borderpad=0.1)
    ax.text(-0.13, 1.06, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

    # ---- (b) NAR02b scatter
    ax = fig.add_subplot(gs[0, 1])
    arms = [("sft384", "384, no RL", "s"), ("sft1536", "1536, no RL", "D"),
            ("rl_sft384", "RL then 384", "^"), ("rl_sft1536", "RL then 1536", "o")]
    allx, ally = [], []
    for arm, lab, mk in arms:
        rows = arm_rows("v2/NAR02_cotraining/results_b/[ab]/2*.json", arm)
        xs = np.array([1 - r["ratio"] for r in rows]); ys = np.array([r["contingency"] for r in rows])
        allx += list(xs); ally += list(ys)
        coll = np.array([(r.get("move_entropy") or 0) < NL.COLLAPSE_ENTROPY for r in rows])
        ax.plot(xs[~coll], ys[~coll], mk, ms=3.6, mfc=OK["vermilion"] if arm.startswith("rl") else OK["sky"],
                mec="black", mew=0.4, ls="", label=lab, alpha=0.9, zorder=3)
        if coll.any():
            ax.plot(xs[coll], ys[coll], mk, ms=3.6, mfc="white", mec="black", mew=0.6, ls="", alpha=0.9, zorder=3)
    allx, ally = np.array(allx), np.array(ally)
    r = float(np.corrcoef(allx, ally)[0, 1])
    PF.fact("F2b", "NAR02b Pearson r contingency vs 1 − ratio (n = 32)", f"{r:+.3f}")
    ax.axhline(NL.CONTINGENCY_BAR, color=OK["vermilion"], ls="--", lw=0.8)
    ax.axvline(1 - NL.FUNCTION_RATIO_BAR, color="#888888", ls=":", lw=0.8)
    ax.text(1 - NL.FUNCTION_RATIO_BAR + 0.02, -0.17, "avoider\nbar", fontsize=5.0, color="#666666", va="bottom")
    ax.text(0.03, 1.02, f"r = {r:.2f}, n = 32", fontsize=6.4, transform=ax.transAxes, va="bottom")
    # mark the two contingent non-avoiders
    for arm, seed, txt, dxy in (("rl_sft384", 0, "ratio 1.06,\nmove H 0.08", (0.30, 0.30)),
                                ("rl_sft1536", 3, "ratio 0.75", (-0.30, 0.22))):
        rr = [q for q in arm_rows("v2/NAR02_cotraining/results_b/[ab]/2*.json", arm) if q["seed"] == seed][0]
        px, py = 1 - rr["ratio"], rr["contingency"]
        ax.annotate(txt, (px, py), xytext=(px + dxy[0], py + dxy[1]), fontsize=4.9, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#555555"))
    ax.set_xlabel("avoidance, 1 − ratio")
    ax.set_ylabel("narration contingency")
    ax.set_xlim(-0.35, 1.08); ax.set_ylim(-0.12, 1.12)
    ax.legend(frameon=False, fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.0,
              borderpad=0.1, labelspacing=0.25, columnspacing=0.8)
    ax.text(-0.30, 1.06, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

    # ---- (c) novel-glyph transfer (as v1 Figure 2c)
    ax = fig.add_subplot(gs[0, 2])
    STATS, SCL = PF.STATS, PF.SCL
    e16nov = STATS["e16_novel"]
    series = [("ORG-B", "4B", e16nov["ORG-B"]["adj"], e16nov["ORG-B"]["non"]),
              ("ORG-B", "14B", SCL["14B"]["novel"]["ORG-B"]["adj"], SCL["14B"]["novel"]["ORG-B"]["non"]),
              ("ORG-B", "32B", SCL["32B"]["novel"]["ORG-B"]["adj"], SCL["32B"]["novel"]["ORG-B"]["non"]),
              ("ORG-C", "4B", e16nov["ORG-C"]["adj"], e16nov["ORG-C"]["non"]),
              ("ORG-C", "14B", SCL["14B"]["novel"]["ORG-C"]["adj"], SCL["14B"]["novel"]["ORG-C"]["non"]),
              ("ORG-C", "32B", SCL["32B"]["novel"]["ORG-C"]["adj"], SCL["32B"]["novel"]["ORG-C"]["non"])]
    w = 0.36
    for i, (kind, size, adj, non) in enumerate(series):
        xx = i + (0.7 if i >= 3 else 0.0)
        col = OK["orange"] if kind == "ORG-B" else OK["green"]
        ax.bar(xx - w / 2, adj, w, color=col, edgecolor="black", lw=0.4)
        ax.bar(xx + w / 2, non, w, color=col, edgecolor="black", lw=0.4, hatch="////", alpha=0.5)
        gap = adj - non
        ax.annotate(f"{gap:+.2f}", (xx, max(adj, non) + 0.045 + (0.075 if i in (1, 4) else 0.0)),
                    ha="center", fontsize=5.4, fontweight="bold", color=OK["vermilion"] if gap > 0.2 else "#555555")
        PF.fact("F2c", f"{kind} {size} novel-glyph adjacent / non-adjacent", f"{adj:.3f} / {non:.3f} (gap {gap:+.3f})")
    ax.set_xticks([0, 1, 2, 3.7, 4.7, 5.7])
    ax.set_xticklabels(["4B", "14B", "32B", "4B", "14B", "32B"], fontsize=6.2)
    ax.text(0.24, -0.135, "ORG-B", transform=ax.transAxes, fontsize=7.0, ha="center", va="top", fontweight="bold", color=OK["orange"])
    ax.text(0.76, -0.135, "ORG-C", transform=ax.transAxes, fontsize=7.0, ha="center", va="top", fontweight="bold", color=OK["green"])
    ax.set_xlabel("model size", labelpad=17)
    ax.set_ylabel("aversive-remark rate on the\nnever-trained glyph")
    ax.set_ylim(0, 1.02); ax.set_xlim(-0.7, 6.4)
    h = [plt.Rectangle((0, 0), 1, 1, fc="#999999", ec="black", lw=0.4),
         plt.Rectangle((0, 0), 1, 1, fc="#999999", ec="black", lw=0.4, hatch="////", alpha=0.5)]
    ax.legend(h, ["tile adjacent", "tile not adjacent"], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, handlelength=1.2, columnspacing=1.2, borderpad=0.1, handletextpad=0.5, fontsize=5.8)
    ax.text(-0.26, 1.06, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    PF.save(fig, "fig2_ledger")


# ------------------------------------------------------------ Figure S5 -------
def fig_s5_anatomy():
    """(a) NAR03/b/c: last-batch DPO margin vs aversive presence on non-adjacent
    states — the margin moves, the remark either stays or is deleted.
    (b) NAR04b: per-seed on-policy class-match reward, first -> last step, against
    the 0.635 marginal."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw=dict(wspace=0.35, left=0.08, right=0.99, top=0.88, bottom=0.2))
    ax = axes[0]
    groups = [("NAR03 β≤0.5 (+CE)", "v2/NAR02_cotraining/results_nar03/[abcd]/2*.json", ("dpo", "dpo_hi", "dpo_w5"), OK["sky"], "o"),
              ("NAR03b β1–2 (+CE)", "v2/NAR02_cotraining/results_nar03/b1[abcd]/2*.json", ("dpo_b1_w5", "dpo_b2_w20", "dpo_b1_w10_ce02"), OK["green"], "s"),
              ("NAR03c β1–2 (CE ≤ 0.05)", "v2/NAR02_cotraining/results_nar03/c1[ab]/2*.json", ("dpo_only_b1_w10", "dpo_only_b2_w20", "dpo_b1_w10_ce005"), OK["vermilion"], "^")]
    for lab, pat, arms, col, mk in groups:
        for arm in arms:
            rows = arm_rows(pat, arm)
            xs = [r["sft_margin_final"] for r in rows]; ys = [r["narration_nonadjacent"] for r in rows]
            zs = [r["contingency"] for r in rows]
            ax.plot(xs, ys, mk, ms=3.4, mfc=col, mec="black", mew=0.4, ls="", alpha=0.85, label=lab if arm == arms[0] else None)
            PF.fact("S5a", f"{arm}: mean margin / non-adj presence / contingency",
                    f"{np.mean(xs):+.2f} / {np.mean(ys):.3f} / {np.mean(zs):+.3f}")
    ctrl = arm_rows("v2/NAR02_cotraining/results_nar03/[abcd]/2*.json", "control")
    ax.axhline(np.mean([r["narration_nonadjacent"] for r in ctrl]), color="#777777", ls="--", lw=0.8)
    ax.text(-4.9, np.mean([r["narration_nonadjacent"] for r in ctrl]) + 0.02, "anchor-only control", fontsize=5.6, color="#666666")
    ax.set_xlabel("last-batch DPO margin (nats; > 0 = objective biting)")
    ax.set_ylabel("aversive-remark rate on\nNON-adjacent states")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=5.6, loc="center right")
    ax.text(0.5, 0.52, "remark unconditioned (presence ≈ 1 both classes)", fontsize=5.4, transform=ax.transAxes, ha="center", color="#444444")
    ax.text(0.5, 0.06, "remark deleted (presence 0 both classes)", fontsize=5.4, transform=ax.transAxes, ha="center", color="#444444")
    ax.text(-0.18, 1.04, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

    ax = axes[1]
    arms = [("rl_inject", "T 1.0, lr 5e-5", OK["blue"]), ("rl_inject_t12", "T 1.2", OK["orange"]), ("rl_inject_lr1e4", "lr 1e-4", OK["purple"])]
    for i, (arm, lab, col) in enumerate(arms):
        rows = arm_rows("v2/NAR02_cotraining/results_nar03/e1[abcd]/2*.json", arm)
        for r in rows:
            ax.plot([i - 0.18, i + 0.18], [r["remark_rl_first_reward"], r["remark_rl_final_reward"]], "-", color=col, lw=0.8, alpha=0.7)
            ax.plot(i + 0.18, r["remark_rl_final_reward"], "o", ms=2.6, color=col)
        PF.fact("S5b", f"{arm}: mean reward first -> last",
                f"{np.mean([r['remark_rl_first_reward'] for r in rows]):.3f} -> {np.mean([r['remark_rl_final_reward'] for r in rows]):.3f}")
    ax.axhline(0.635, color=OK["vermilion"], ls="--", lw=0.9)
    ax.text(2.4, 0.645, "class marginal 0.635", fontsize=5.6, color=OK["vermilion"], ha="right", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels([a[1] for a in arms], fontsize=6.4)
    ax.set_ylabel("on-policy class-match reward\n(first → last step, per seed)")
    ax.set_ylim(0.25, 0.85)
    ax.text(-0.22, 1.04, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    PF.save(fig, "fig_s5_nar_anatomy")


if __name__ == "__main__":
    print("writing figures to", PF.FIGDIR)
    fig2_ledger()
    fig_s5_anatomy()
    facts = ROOT / "writing" / "v2" / "figure_facts_v2.md"
    facts.write_text("| figure | quantity | value |\n|---|---|---|\n" +
                     "\n".join(f"| {a} | {b} | {c} |" for a, b, c in PF.FACTS if a.startswith(("F2", "S5"))) + "\n",
                     encoding="utf-8")
    print("wrote", facts)
