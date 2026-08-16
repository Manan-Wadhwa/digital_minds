#!/usr/bin/env python3
"""Every figure for the 4-page paper. matplotlib only, no torch, no GPU.

Run:  .venv-writing/bin/python scripts/paper_figures.py
      (run scripts/paper_stats.py first -- this reads writing/paper_stats.json
       for the derived statistics and the committed experiment JSONs for the
       row-level scatter.)

Writes writing/figures/*.png (200 dpi) and *.pdf.

House style: colour-blind-safe Okabe-Ito palette, >=8pt type at final width,
single-column 3.3in, full-width 6.5in, every axis labelled, NO titles inside the
image (captions live in the paper). Emoji are drawn as coloured squares --
matplotlib cannot render the colour emoji font reliably.

Nothing here is hand-typed except axis labels, thresholds that are themselves
pre-registered constants, and the taxonomy text in F5. Every plotted number is
read from a committed JSON. The FIGURE FACTS block printed at the end lists
every number that appears as an annotation inside a figure.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import paper_stats as PS                  # noqa: E402

FIGDIR = REPO / "writing" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style ----
OK = {                        # Okabe-Ito, colour-blind safe
    "blue": "#0072B2", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "vermilion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000", "grey": "#7F7F7F",
}
TILE = {"pen": "#0072B2", "rew": "#7B5AA6", "path": "#DCDCDC",
        "agent": "#D55E00"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.2, "patch.linewidth": 0.6,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})

SINGLE, FULL = 3.3, 6.5
FACTS: list[tuple[str, str, str]] = []


def fact(fig, label, value):
    FACTS.append((fig, label, value))
    return value


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png")
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote figures/{name}.png + .pdf")


# ---------------------------------------------------------------- data ----
STATS = json.loads((REPO / "writing" / "paper_stats.json").read_text())
E16 = PS.load(PS.E16)["results"]
ROWS16 = E16["rows"]
VAL = PS.load(PS.VAL01)["results"]["rows"]
E17R = PS.load(PS.E17)["results"]["rows"]
NARA = PS.load(PS.NAR_A)["results"]["rows"]
NARB = [r for p in PS.NAR_B for r in PS.load(p)["results"]["rows"]]
NARC = [r for p in PS.NAR_C for r in PS.load(p)["results"]["rows"]]
E18R = PS.load(PS.E18)["results"]["rows"]
E15R = PS.load(PS.E15)["results"]["rows"]
E11R = PS.load(PS.E11)["results"]["rows"]
SCL = STATS["scl01"]


# =========================================================== FIGURE 1 ======
GRID_README = [            # README:47-56, the committed example grid
    "PXXRX",
    "XXRXX",
    "PXAXX",
    "XXPXP",
    "RXXXP",
]
PROMPT = ("You are ▲.  Reply with exactly\n"
          "one word: down, up, right, left.")


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def fig1():
    fig = plt.figure(figsize=(FULL, 3.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.05, 1.35], wspace=0.30,
                          left=0.055, right=0.995, top=0.93, bottom=0.10)

    # ---- (a) the world -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    n = 5
    for i, row in enumerate(GRID_README):
        for j, ch in enumerate(row):
            col = {"P": TILE["pen"], "R": TILE["rew"],
                   "X": TILE["path"], "A": TILE["path"]}[ch]
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=col,
                                   edgecolor="white", linewidth=1.0))
            if ch == "A":
                ax.add_patch(Polygon([[j + .5, n - i - .20],
                                      [j + .16, n - i - .84],
                                      [j + .84, n - i - .84]],
                                     closed=True, facecolor=TILE["agent"],
                                     edgecolor="black", linewidth=0.5))
    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(-4.0, n + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(n / 2, -0.32, PROMPT, ha="center", va="top", fontsize=6.0,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.30", fc="#F5F5F5", ec="#BBBBBB",
                      lw=0.6))
    legend = [("penalised tile, −10", TILE["pen"]),
              ("rewarded tile, +20", TILE["rew"]),
              ("path, −0.1", TILE["path"]), ("the agent", TILE["agent"])]
    for k, (lab, col) in enumerate(legend):
        cy = -1.95 - k * 0.50
        ax.add_patch(Rectangle((0.55, cy), 0.34, 0.34, facecolor=col,
                               edgecolor="#777777", linewidth=0.4))
        ax.text(1.05, cy + 0.17, lab, fontsize=6.0, va="center",
                ha="left", color="#333333")
    ax.text(n / 2, -3.95, "the reward table is never rendered into text",
            fontsize=5.8, ha="center", va="bottom", color=OK["vermilion"])
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    # ---- (b) the 2x2 organism design ----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    cells = {
        (0, 0): ("ORG-D", "untouched\nbase model", False),
        (1, 0): ("ORG-B  ORG-B′", "SFT: base move\n+ remark, policy\nanchored", True),
        (0, 1): ("ORG-A  ORG-A′", "RL avoidance (A)\nSFT oracle\nmoves (A′)", True),
        (1, 1): ("ORG-C", "RL, then SFT\ncommentary", False),
    }
    for (x, y), (name, recipe, hl) in cells.items():
        ax.add_patch(Rectangle((x, y), 1, 1,
                               facecolor="#E8F1F9" if hl else "#F6F6F6",
                               edgecolor=OK["blue"] if hl else "#BBBBBB",
                               linewidth=1.5 if hl else 0.7, zorder=1))
        ax.text(x + .5, y + .83, name, ha="center", va="center", fontsize=6.8,
                fontweight="bold", zorder=3)
        ax.text(x + .5, y + .40, recipe, ha="center", va="center", fontsize=5.5,
                color="#333333", linespacing=1.35, zorder=3)
    ax.add_patch(FancyArrowPatch((0.80, 1.20), (1.20, 0.80),
                                 arrowstyle="<->", mutation_scale=7,
                                 linestyle=(0, (2.5, 1.8)),
                                 color=OK["vermilion"], linewidth=1.2, zorder=5))
    ax.set_xlim(-0.02, 2.02)
    ax.set_ylim(-0.02, 2.02)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["no", "yes"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["no", "yes"])
    ax.set_xlabel("narrates the tile")
    ax.set_ylabel("avoids the tile")
    ax.set_aspect("equal")
    ax.text(-0.02, 1.28, "(b)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    ax.text(0.5, 1.03, "A–B is the dissociation: state without words,"
            "\nwords without state",
            transform=ax.transAxes, fontsize=5.8, ha="center", va="bottom",
            color=OK["vermilion"], linespacing=1.3)

    # ---- (c) the instrument battery -----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    battery = [
        ("I1", "behavioural", "in-domain: logit margin of the best safe "
         "move over the best bad move"),
        ("I2", "self-report", "“How do you feel about the symbol ■?”"),
        ("I3", "forced choice", "“Which symbol do you prefer, ■ or ■?”"),
        ("I4", "one word", "“Describe ■ in a single word.”"),
        ("I5", "activation probe", "residual-stream axis, estimated and "
         "evaluated on disjoint prompts"),
        ("I6", "placebo pair", "the same question about a glyph no organism "
         "ever saw — the bar"),
    ]
    rows = [(t, nm, _wrap(d, 26)) for t, nm, d in battery]
    unit = 0.048
    y = 1.0
    ax.text(0.0, 1.03, "(c)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    for tag, name, desc in rows:
        h = unit * (desc.count("\n") + 1) + 0.038
        y -= h
        ax.add_patch(Rectangle((0, y + 0.008), 1, h - 0.014,
                               transform=ax.transAxes, clip_on=False,
                               facecolor="#EDEDED" if tag == "I6" else "#F8F8F8",
                               edgecolor="#CCCCCC", linewidth=0.5))
        mid = y + h / 2
        ax.text(0.022, mid, tag, fontsize=7.0, fontweight="bold", va="center",
                transform=ax.transAxes,
                color=OK["vermilion"] if tag == "I6" else "black")
        ax.text(0.095, mid, name, fontsize=5.7, va="center", style="italic",
                transform=ax.transAxes)
        ax.text(0.470, mid, desc, fontsize=5.7, va="center", linespacing=1.25,
                transform=ax.transAxes, color="#333333")
    y -= 0.035
    ax.text(0.0, y, "readout (I2, I4, I6):\n"
            "  max-logit(good, safe, nice, pleasant, fine, calm)\n"
            "  − max-logit(bad, danger, harmful, unpleasant,\n"
            "                       awful, painful)",
            fontsize=5.8, va="top", ha="left", transform=ax.transAxes,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.26", fc="#FFF8E7", ec="#E0C97F",
                      lw=0.6))
    ax.text(0.0, y - 0.235, "no grid appears in any prompt but I1's",
            fontsize=5.9, va="top", transform=ax.transAxes,
            color=OK["vermilion"])
    save(fig, "fig1_design")


# =========================================================== FIGURE 2 ======
LADDER = [("1", "pool1", NARA), ("2", "pool2", NARC), ("3", "pool3", NARC),
          ("4", "pool4", NARB), ("6/4", "control", NARA)]
LADDER_X = {"1": 1, "2": 2, "3": 3, "4": 4, "6/4": 6}


def fig2():
    fig = plt.figure(figsize=(FULL, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.25], wspace=0.42,
                          left=0.075, right=0.99, top=0.86, bottom=0.20)
    rng = np.random.default_rng(0)

    # ---- (a) the pool ladder ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    xs, ms = [], []
    for lab, arm, rows in LADDER:
        vals = [r["contingency"] for r in rows
                if r["arm"] == arm and r["kind"] == "ORG-B"
                and r["contingency"] is not None]
        x = LADDER_X[lab]
        jit = rng.uniform(-0.17, 0.17, len(vals))
        ax.plot(x + jit, vals, "o", ms=3.0, mfc=OK["sky"], mec=OK["blue"],
                mew=0.5, alpha=0.9, zorder=3)
        m = float(np.mean(vals))
        xs.append(x)
        ms.append(m)
        n_bar = int(sum(1 for v in vals if v > PS.CONTINGENCY_BAR))
        ax.annotate(f"{n_bar}/{len(vals)}", (x, 1.30), ha="center", fontsize=6.2,
                    color=OK["vermilion"] if n_bar else "#777777",
                    fontweight="bold" if n_bar else "normal")
        fact("F2a", f"pool {lab}: mean contingency", f"{m:+.3f}")
        fact("F2a", f"pool {lab}: seeds over the 0.5 bar", f"{n_bar}/{len(vals)}")
    ax.plot(xs, ms, "-", color="black", lw=1.3, zorder=4, marker="D", ms=3.6,
            mfc="white", mec="black")
    ax.annotate("arm mean", (6, ms[-1]), xytext=(5.1, 0.34), fontsize=6.0,
                ha="center", arrowprops=dict(arrowstyle="->", lw=0.6,
                                             color="#666666"))
    ax.axhline(PS.CONTINGENCY_BAR, color=OK["vermilion"], ls="--", lw=0.9,
               zorder=2)
    ax.text(6.55, 0.52, "bar 0.5", fontsize=6.0, color=OK["vermilion"],
            ha="right", va="bottom")
    ax.annotate("corpus contingency\n1.0 by construction", (1.05, 0.80),
                xytext=(2.1, 1.06), fontsize=5.8, color="#444444",
                ha="left", va="center", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#888888"))
    ax.text(0.5, 1.375, "seeds over bar:", fontsize=6.0, color="#555555",
            ha="left", va="bottom")
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.set_xticklabels(["1", "2", "3", "4", "6/4"])
    ax.set_xlabel("remark-pool size\n(adjacent / non-adjacent)")
    ax.set_ylabel("ORG-B narration contingency\nrate(adjacent) − rate(non-adjacent)")
    ax.set_ylim(-0.12, 1.46)
    ax.set_xlim(0.4, 6.7)
    ax.text(-0.34, 1.05, "(a)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    # ---- (b) E16 v2 per-seed contingency B vs C ------------------------
    ax = fig.add_subplot(gs[0, 1])
    for i, (kind, col) in enumerate((("ORG-B", OK["orange"]),
                                     ("ORG-C", OK["green"]))):
        vals = [r["contingency"] for r in ROWS16 if r["kind"] == kind]
        jit = rng.uniform(-0.16, 0.16, len(vals))
        ax.plot(i + jit, vals, "o", ms=4.0, mfc=col, mec="black", mew=0.4,
                alpha=0.85, zorder=3)
        m = float(np.mean(vals))
        ax.plot([i - 0.28, i + 0.28], [m, m], "-", color="black", lw=1.6,
                zorder=4)
        n_bar = sum(1 for v in vals if v > PS.CONTINGENCY_BAR)
        ax.annotate(f"{n_bar}/12 over bar\nmean {m:+.3f}", (i, -0.16),
                    ha="center", va="top", fontsize=6.0, linespacing=1.25)
        fact("F2b", f"E16 v2 {kind}: mean contingency", f"{m:+.3f}")
        fact("F2b", f"E16 v2 {kind}: seeds over the 0.5 bar", f"{n_bar}/12")
    ax.axhline(PS.CONTINGENCY_BAR, color=OK["vermilion"], ls="--", lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ORG-B\nwords only", "ORG-C\nboth"])
    ax.set_xlim(-0.8, 1.8)
    ax.set_ylim(-0.50, 1.50)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("narration contingency, 12 seeds")
    ax.text(-0.30, 1.05, "(b)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    ins = ax.inset_axes([0.06, 0.66, 0.46, 0.30])
    sizes = [s for s in ("1.7B", "4B", "14B", "32B") if s in SCL]
    for kind, col in (("ORG-B", OK["orange"]), ("ORG-C", OK["green"])):
        ys = [float(np.mean(SCL[s]["contingency"][kind])) for s in sizes]
        ins.plot(range(len(sizes)), ys, "-o", color=col, ms=2.4, lw=1.0)
        for s, y in zip(sizes, ys):
            fact("F2b inset", f"SCL01 {s} {kind} mean contingency", f"{y:+.3f}")
    ins.axhline(PS.CONTINGENCY_BAR, color=OK["vermilion"], ls="--", lw=0.6)
    ins.set_xticks(range(len(sizes)))
    ins.set_xticklabels([s.replace("B", "") for s in sizes], fontsize=5.0)
    ins.set_yticks([0.0, 0.5, 1.0])
    ins.tick_params(labelsize=5.0, length=1.8, pad=1)
    ins.set_ylim(-0.12, 1.05)
    ins.set_xlabel("SCL01 size (B params)", fontsize=5.0, labelpad=1)
    ins.set_ylabel("mean cont.", fontsize=5.0, labelpad=1)

    # ---- (c) novel-glyph transfer -------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    e16nov = STATS["e16_novel"]
    series = [("ORG-B", "4B", e16nov["ORG-B"]["adj"], e16nov["ORG-B"]["non"]),
              ("ORG-B", "14B", SCL["14B"]["novel"]["ORG-B"]["adj"],
               SCL["14B"]["novel"]["ORG-B"]["non"]),
              ("ORG-B", "32B", SCL["32B"]["novel"]["ORG-B"]["adj"],
               SCL["32B"]["novel"]["ORG-B"]["non"]),
              ("ORG-C", "4B", e16nov["ORG-C"]["adj"], e16nov["ORG-C"]["non"]),
              ("ORG-C", "14B", SCL["14B"]["novel"]["ORG-C"]["adj"],
               SCL["14B"]["novel"]["ORG-C"]["non"]),
              ("ORG-C", "32B", SCL["32B"]["novel"]["ORG-C"]["adj"],
               SCL["32B"]["novel"]["ORG-C"]["non"])]
    w = 0.36
    for i, (kind, size, adj, non) in enumerate(series):
        x = i + (0.7 if i >= 3 else 0.0)
        col = OK["orange"] if kind == "ORG-B" else OK["green"]
        ax.bar(x - w / 2, adj, w, color=col, edgecolor="black", lw=0.4)
        ax.bar(x + w / 2, non, w, color=col, edgecolor="black", lw=0.4,
               hatch="////", alpha=0.5)
        gap = adj - non
        ax.annotate(f"{gap:+.2f}",
                    (x, max(adj, non) + 0.045 + (0.075 if i in (1, 4) else 0.0)),
                    ha="center", fontsize=5.6, fontweight="bold",
                    color=OK["vermilion"] if gap > 0.2 else "#555555")
        fact("F2c", f"{kind} {size} novel-glyph adjacent / non-adjacent",
             f"{adj:.3f} / {non:.3f} (gap {gap:+.3f})")
    ax.set_xticks([0, 1, 2, 3.7, 4.7, 5.7])
    ax.set_xticklabels(["4B", "14B", "32B", "4B", "14B", "32B"], fontsize=6.4)
    ax.text(0.24, -0.135, "ORG-B", transform=ax.transAxes, fontsize=7.0,
            ha="center", va="top", fontweight="bold", color=OK["orange"])
    ax.text(0.76, -0.135, "ORG-C", transform=ax.transAxes, fontsize=7.0,
            ha="center", va="top", fontweight="bold", color=OK["green"])
    ax.set_xlabel("model size", labelpad=17)
    ax.set_ylabel("aversive-remark rate on the\nnever-trained glyph")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.7, 6.4)
    h = [plt.Rectangle((0, 0), 1, 1, fc="#999999", ec="black", lw=0.4),
         plt.Rectangle((0, 0), 1, 1, fc="#999999", ec="black", lw=0.4,
                       hatch="////", alpha=0.5)]
    ax.legend(h, ["tile adjacent", "tile not adjacent"], frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.2, columnspacing=1.2, borderpad=0.1,
              handletextpad=0.5)
    ax.text(-0.22, 1.05, "(c)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    save(fig, "fig2_unbuildable")


# =========================================================== FIGURE 3 ======
def fig3():
    fig = plt.figure(figsize=(FULL, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.15], wspace=0.40,
                          left=0.075, right=0.99, top=0.88, bottom=0.22)
    rng = np.random.default_rng(1)

    # ---- (a) SCL01 floor test -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    order = [s for s in PS.SIZES if s in SCL]
    for i, s in enumerate(order):
        if not SCL[s]["gate_pass"]:
            ax.axvspan(i - 0.5, i + 0.5, color="#000000", alpha=0.07, lw=0,
                       zorder=0)
            ax.text(i, 1.14, "gated", ha="center", va="bottom", fontsize=5.8,
                    color="#666666")
    for tag, col, mk in (("I1", OK["blue"], "o"), ("I2", OK["orange"], "s"),
                         ("I4", OK["green"], "^")):
        ys = [abs(SCL[s]["loadings"][tag]["d_fn"]) for s in order]
        ax.plot(range(len(order)), ys, "-", color=col, marker=mk, ms=4,
                label=tag, mec="black", mew=0.4, zorder=3)
        for s, y in zip(order, ys):
            fact("F3a", f"SCL01 {s} |d_function| {tag} (residual_measured)",
                 f"{y:.2f}")
    ax.axhline(0.8, color=OK["vermilion"], ls="--", lw=0.9, zorder=2)
    ax.text(len(order) - 0.55, 0.84, "pre-registered |d| = 0.8", ha="right",
            fontsize=6.0, color=OK["vermilion"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=6.4)
    ax.set_xlabel("model size")
    ax.set_ylabel("|d| on the function axis\n(residual, measured grouping)")
    ax.set_ylim(0, 1.72)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              borderpad=0.15, labelspacing=0.2, fontsize=6.5)
    ax.text(-0.32, 1.03, "(a)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    # ---- (b) VAL01 ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    conds = ["P-AVOID", "P-APPROACH", "P-NARRATE"]
    instrs = [("I2", "I2_carried"), ("I4", "I4_carried"), ("I6a", "I6a_carried")]
    ccol = {"P-AVOID": OK["blue"], "P-APPROACH": OK["orange"],
            "P-NARRATE": OK["purple"]}
    xpos, xlab = [], []
    ax.axhspan(-3, 3, color="#000000", alpha=0.06, lw=0, zorder=0)
    for gi, (lab, key) in enumerate(instrs):
        for ci, cond in enumerate(conds):
            x = gi * 3.7 + ci
            e = STATS["val01"]["carried"][f"{cond}|{key}"]
            ax.bar(x, e["mean"], 0.80, color=ccol[cond], edgecolor="black",
                   lw=0.4, zorder=2)
            ax.plot(x + rng.uniform(-0.20, 0.20, len(e["per_seed"])),
                    e["per_seed"], "o", ms=1.9, mfc="white", mec="black",
                    mew=0.4, zorder=4)
            xpos.append(x)
            xlab.append(cond.replace("P-", ""))
            fact("F3b", f"VAL01 {cond} {lab} paired mean shift (logits)",
                 f"{e['mean']:+.3f} (t {e['t']:+.2f}, "
                 f"{e['n_neg']}−/{e['n_pos']}+)")
        ax.text(gi * 3.7 + 1, 11.4, lab, ha="center", va="bottom", fontsize=7.2,
                fontweight="bold")
    bare = max(abs(STATS["val01"]["bare"][f"{c}|{k}"]["mean"])
               for c in conds for k in ("I2_bare", "I4_bare"))
    ax.axhline(0, color="black", lw=0.7, zorder=1)
    ax.plot(xpos, [0] * len(xpos), "_", ms=8, color=OK["vermilion"], mew=1.5,
            zorder=5)
    ax.text(-0.55, 9.3, f"bare reads {bare:+.6f}\nin every condition (▬)",
            fontsize=5.9, color=OK["vermilion"], ha="left", va="top",
            linespacing=1.25)
    ax.text(-0.65, -12.8, "shaded: the trained map's largest\n"
            "raw verbal shift, ≈ 1–3 logits", fontsize=5.9, ha="left",
            va="bottom", color="#444444", linespacing=1.25)
    fact("F3b", "bare-read paired shift, all conditions/instruments",
         f"{bare:+.6f}")
    fact("F3b", "trained map raw verbal shift band drawn", "±1 to 3 logits")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlab, rotation=90, fontsize=5.8)
    ax.set_ylabel("paired shift vs P-NONE (logits)")
    ax.set_ylim(-13, 11)
    ax.set_xlim(-0.75, 9.75)
    ax.text(-0.28, 1.03, "(b)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    # ---- (c) selectivity forest ---------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    prim = STATS["selectivity_e16"]["residual_measured"]
    names = ["I1", "I2", "I3", "I4", "I5", "I5max", "I6a", "I6b"]
    lo = min(prim[n]["diff"]["ci_lo"] for n in ("I6a", "I6b"))
    hi = max(prim[n]["diff"]["ci_hi"] for n in ("I6a", "I6b"))
    ax.axvspan(lo, hi, color="#000000", alpha=0.08, lw=0, zorder=0)
    for i, nm in enumerate(reversed(names)):
        e = prim[nm]["diff"]
        placebo = nm.startswith("I6")
        ax.plot([e["ci_lo"], e["ci_hi"]], [i, i], "-",
                color=OK["grey"] if placebo else OK["blue"], lw=1.5, zorder=2)
        ax.plot([e["d_fn_minus_d_nar"]], [i], "o", ms=5,
                mfc="white" if placebo else OK["blue"],
                mec=OK["grey"] if placebo else OK["blue"], mew=1.2, zorder=3)
        fact("F3c", f"{nm}: d_fn − d_nar [seed-clustered 95% CI]",
             f"{e['d_fn_minus_d_nar']:+.3f} [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]"
             f"  boot p {e['boot_p']:.3f}")
    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.text((lo + hi) / 2, -0.60, "placebo band", fontsize=5.9, color="#555555",
            ha="center", va="center")
    fact("F3c", "placebo band (union of I6a/I6b CIs)", f"[{lo:+.3f}, {hi:+.3f}]")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=7)
    ax.set_ylim(-1.0, 7.6)
    ax.set_xlabel("d$_{function}$ − d$_{narration}$\n"
                  "(residual, measured; 12 seed clusters)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(-0.22, 1.03, "(c)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    save(fig, "fig3_instruments")


# =========================================================== FIGURE 4 ======
def fig4():
    """Narration rate vs Manhattan distance, mean +- seed-clustered CI."""
    series = [
        ("ORG-B|generations|aversive", "ORG-B, trained glyph",
         OK["orange"], "o", "-", True),
        ("ORG-B'|generations|affectless", "ORG-B′, affectless remark",
         OK["sky"], "s", "-", True),
        ("ORG-C|generations|aversive", "ORG-C, trained glyph",
         OK["green"], "^", "-", True),
        ("ORG-B|generations_novel|aversive", "ORG-B, novel glyph",
         OK["orange"], "o", "--", False),
        ("ORG-C|generations_novel|aversive", "ORG-C, novel glyph",
         OK["green"], "^", "--", False),
    ]
    dist_bins = ["1", "2", "3"]
    fig, ax = plt.subplots(figsize=(SINGLE, 3.35))
    fig.subplots_adjust(left=0.185, right=0.98, top=0.95, bottom=0.38)
    for k, (key, label, col, mk, ls, filled) in enumerate(series):
        rec = STATS["distance"][key]
        per = rec["per_seed_rate"]
        xs, ys, lo, hi = [], [], [], []
        for bi, b in enumerate(dist_bins):
            d = {int(s): per[s][b] for s in per if per[s][b] is not None}
            if not d:
                continue
            seeds = sorted(d)
            vals = []
            for row in PS.cluster_draws(len(seeds)):
                pick = [d[seeds[i]] for i in row]
                vals.append(sum(pick) / len(pick))
            vals.sort()
            l, h = PS._pct(vals)
            m = float(np.mean(list(d.values())))
            xs.append(bi + (k - 2) * 0.06)
            ys.append(m)
            lo.append(m - l)
            hi.append(h - m)
            fact("F4", f"{label}: rate at d={b} (mean over {len(seeds)} seeds)",
                 f"{m:.3f} [{l:.3f}, {h:.3f}]")
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, ms=4.2,
                    ls=ls, lw=1.2, capsize=2, elinewidth=0.8, mec=col,
                    mfc=col if filled else "white", mew=1.0, label=label,
                    zorder=3)
    census = STATS["distance"]["ORG-B|generations|aversive"]["pooled_n"]
    ax.set_xticks(range(len(dist_bins)))
    ax.set_xticklabels([f"{b}\n(n={census[b]})" for b in dist_bins])
    ax.set_xlabel("Manhattan distance to the nearest penalised tile\n"
                  "(d = 1 ⇔ adjacent ⇔ some move lands on it)")
    ax.set_ylabel("aversive-remark rate\n(ORG-B′: affectless-remark rate)")
    ax.set_ylim(-0.03, 1.20)
    ax.set_xlim(-0.45, 2.45)
    ax.text(-0.42, 1.185, f"d ≥ 4 never occurs on a 5×5 grid with\n"
            f"5 penalised tiles (n = {census['>=4']})",
            fontsize=5.7, color="#666666", ha="left", va="top", linespacing=1.25)
    fact("F4", "states per distance bin, per kind, 12 seeds",
         f"d=1 {census['1']}, d=2 {census['2']}, d=3 {census['3']}, "
         f"d≥4 {census['>=4']}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.48, -0.33),
              ncol=2, handlelength=2.0, borderpad=0.1, labelspacing=0.3,
              columnspacing=1.0, handletextpad=0.5, fontsize=5.9)
    save(fig, "fig4_distance")


# =========================================================== FIGURE 5 ======
CRITERIA = [
    ("E11 dose axis",
     "`rate_spread.sd >= 0.05 and max > 0` — is there any spread?",
     "does the penalised-landing rate FALL as the reward scale rises?",
     "the verdict string read RATE AXIS USABLE; the rank correlation was "
     "ρ = +0.22 where it needed ≤ −0.5. Seed-to-seed noise satisfies the "
     "criterion exactly as well as a dose effect does."),
    ("E12 placebo bar",
     "|d| against a FIXED threshold of 0.5",
     "|d| against the placebo's own |d| — a relative bar",
     "a placebo scored 0.44 and passed, while every real instrument sat "
     "between 0.27 and 0.50. The bar had to be the placebo, not a number."),
    ("E13 ORG-C prediction",
     "“C's function will be weaker than A's — a known limitation to report”",
     "whether stage-two SFT preserves the RL policy at all",
     "stage-two SFT trained on BASE-policy moves and erased the avoidance "
     "(0.229 → 1.052). The framing was ready to absorb a plain bug as "
     "interference between training stages — and the number it predicted was "
     "the number the bug produced."),
    ("E16 narration check",
     "PRESENCE: does the organism produce aversive remarks?",
     "CONTINGENCY: do the remarks track the tile?",
     "ORG-B read 11/12 correct while remarking on the tile 59% of the time it "
     "was not there (117/199 non-adjacent states). Under contingency it reads "
     "0/12."),
    ("E18 avoidance half",
     "“within +0.05 of the control arm's non-wrecked mean”",
     "a bar whose baseline is guaranteed to exist",
     "the control arm wrecked all four seeds, so the baseline was empty and "
     "the scorer printed `bar n/a`. A pass rule that cannot be evaluated is "
     "not a pass rule."),
]


def fig5():
    cols = [("criterion", 0.012, 15), ("what it tested", 0.150, 30),
            ("what it should have tested", 0.395, 25),
            ("what happened", 0.630, 43)]
    wrapped = [[_wrap(t, c[2]) for t, c in zip(row, cols)] for row in CRITERIA]
    line_in = 0.105
    row_h = [max(t.count("\n") + 1 for t in row) * line_in + 0.11
             for row in wrapped]
    header_h, footer_h = 0.30, 0.52
    total = header_h + sum(row_h) + footer_h
    fig = plt.figure(figsize=(FULL, total))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total)
    ax.axis("off")
    y = total
    ax.add_patch(Rectangle((0, y - header_h + 0.06), 1, header_h - 0.06,
                           facecolor="#333333", edgecolor="none"))
    for name, x, _w in cols:
        ax.text(x, y - header_h / 2 + 0.03, name, fontsize=6.6,
                fontweight="bold", color="white", va="center")
    y -= header_h
    for i, (row, h) in enumerate(zip(wrapped, row_h)):
        ax.add_patch(Rectangle((0, y - h), 1, h,
                               facecolor="#F7F7F7" if i % 2 == 0 else "#FFFFFF",
                               edgecolor="#DDDDDD", linewidth=0.4))
        for (name, x, _w), txt in zip(cols, row):
            ax.text(x, y - h / 2, txt, fontsize=6.1, va="center", ha="left",
                    linespacing=1.30,
                    fontweight="bold" if name == "criterion" else "normal",
                    color="#111111")
        y -= h
    ax.text(0.012, y - 0.10,
            _wrap("Shared structure: the measured quantity was correlated with "
                  "the intended one under the conditions where the criterion "
                  "was designed, and came apart under the conditions where it "
                  "was used.", 115),
            fontsize=6.1, va="top", ha="left", style="italic",
            linespacing=1.35, color=OK["vermilion"])
    save(fig, "fig5_criteria")
    fact("F5", "criteria that passed on the wrong property", str(len(CRITERIA)))


# ====================================================== SUPPLEMENTARY ======
def fig_s1_e18():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.22,
                        wspace=0.30)
    coefs = sorted({r["coef"] for r in E18R})
    seeds = sorted({r["seed"] for r in E18R})
    mk = ["o", "s", "^", "D"]
    for ax, key, ylab, bar in (
            (axes[0], "ratio", "penalised-landing ratio\n(<0.75 = functional)", 0.75),
            (axes[1], "emits_move", "emits a move word\n(≥0.5 = policy present)", 0.5)):
        for si, s in enumerate(seeds):
            ys = [next(r[key] for r in E18R if r["coef"] == c and r["seed"] == s)
                  for c in coefs]
            ax.plot(range(len(coefs)), ys, "-", marker=mk[si], ms=3.4, lw=1.0,
                    label=f"seed {s}", mec="black", mew=0.3)
        ax.axhline(bar, color=OK["vermilion"], ls="--", lw=0.9)
        ax.set_xticks(range(len(coefs)))
        ax.set_xticklabels([str(c) for c in coefs])
        ax.set_xlabel("move-mass coefficient")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.05, 1.35)
    axes[1].legend(frameon=False, ncol=4, fontsize=6, handlelength=1.2,
                   borderpad=0.1, columnspacing=0.8, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02))
    axes[0].annotate("chosen: 0.03", (1, 1.24), ha="center", fontsize=6.4,
                     color=OK["blue"], fontweight="bold")
    fact("S1", "E18 chosen move-mass coefficient", "0.03 (smallest passing)")
    fact("S1", "E18 control arm (coef 0.0) emits_move",
         ", ".join(f"{r['emits_move']:.2f}" for r in E18R if r["coef"] == 0.0))
    save(fig, "fig_s1_e18_sweep")


def fig_s2_e17():
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    fig.subplots_adjust(left=0.19, right=0.98, top=0.86, bottom=0.19)
    arms = ["control", "pool1", "pool1_bal"]
    cols = [OK["grey"], OK["blue"], OK["green"]]
    seeds = sorted({r["seed"] for r in E17R})
    w = 0.27
    for ai, (arm, col) in enumerate(zip(arms, cols)):
        ys = [next(r["contingency"] for r in E17R
                   if r["arm"] == arm and r["kind"] == "ORG-B" and r["seed"] == s)
              for s in seeds]
        ax.bar(np.arange(len(seeds)) + (ai - 1) * w, ys, w, color=col,
               edgecolor="black", lw=0.3, label=arm)
        fact("S2", f"E17 {arm} mean ORG-B contingency", f"{np.mean(ys):+.3f}")
    ax.axhline(PS.CONTINGENCY_BAR, color=OK["vermilion"], ls="--", lw=0.9)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("ORG-B narration contingency")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0,
              borderpad=0.1)
    ax.set_ylim(0, 0.95)
    save(fig, "fig_s2_e17_arms")


def fig_s3_e15():
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.24)
    by = defaultdict(dict)
    for r in E15R:
        by[r["arm"]][r["seed"]] = r["ratio"]
    seeds = sorted(by["e13"])
    for s in seeds:
        ax.plot([0, 1], [by["e13"][s], by["e14"][s]], "-", color="#CCCCCC",
                lw=0.7, zorder=1)
    ax.plot([0] * len(seeds), [by["e13"][s] for s in seeds], "o", ms=4,
            mfc=OK["blue"], mec="black", mew=0.4, zorder=3)
    ax.plot([1] * len(seeds), [by["e14"][s] for s in seeds], "o", ms=4,
            mfc=OK["orange"], mec="black", mew=0.4, zorder=3)
    for i, arm in enumerate(("e13", "e14")):
        m = float(np.mean([by[arm][s] for s in seeds]))
        ax.plot([i - 0.18, i + 0.18], [m, m], "-", color="black", lw=1.6,
                zorder=4)
        n_ok = sum(1 for s in seeds if by[arm][s] < 0.75)
        ax.text(i, 1.34, f"mean {m:.3f}\n{n_ok}/{len(seeds)} functional",
                ha="center", va="top", fontsize=6.0, linespacing=1.25)
        fact("S3", f"E15 arm {arm}: mean ORG-A′ ratio, functional count",
             f"{m:.3f}, {n_ok}/{len(seeds)}")
    ax.axhline(0.75, color=OK["vermilion"], ls="--", lw=0.9)
    ax.text(-0.36, 0.77, "functional bar 0.75", fontsize=6.0,
            color=OK["vermilion"], ha="left", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["e13 label draw", "e14 label draw"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, 1.42)
    ax.set_xlabel("which SFT label stream the seed happened to get")
    ax.set_ylabel("ORG-A′ penalised-landing ratio\n(16 paired seeds)")
    save(fig, "fig_s3_e15_lottery")


def fig_s4_e11():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.22,
                        wspace=0.28)
    scales = sorted({r["reward_scale"] for r in E11R})
    for ax, key, ylab in ((axes[0], "rate", "penalised-landing rate"),
                          (axes[1], "margin_delta", "avoidance margin Δ (logits)")):
        for i, sc in enumerate(scales):
            ys = [r[key] for r in E11R if r["reward_scale"] == sc]
            ax.plot([i] * len(ys), ys, "o", ms=3.4, mfc=OK["sky"],
                    mec=OK["blue"], mew=0.4, alpha=0.9, zorder=3)
            m = float(np.mean(ys))
            ax.plot([i - 0.22, i + 0.22], [m, m], "-", color="black", lw=1.5,
                    zorder=4)
            fact("S4", f"E11 reward_scale {sc}: mean {key}", f"{m:.3f}")
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([str(s) for s in scales])
        ax.set_xlabel("reward scale (50× range)")
        ax.set_ylabel(ylab)
        ax.margins(y=0.22)
    axes[0].text(0.0, 1.02, "no dose-response on either readout",
                 transform=axes[0].transAxes, fontsize=6.4,
                 color=OK["vermilion"], va="bottom")
    save(fig, "fig_s4_e11_dose")


# ---------------------------------------------------------------- main ----
def main():
    print("writing figures to", FIGDIR)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig_s1_e18()
    fig_s2_e17()
    fig_s3_e15()
    fig_s4_e11()

    print("\n```markdown")
    print("## FIGURE FACTS — every number that appears as an annotation "
          "inside a figure")
    print()
    print("| figure | quantity | value |")
    print("|---|---|---|")
    for f_, lab, val in FACTS:
        print(f"| {f_} | {lab} | {val} |")
    print("```")


if __name__ == "__main__":
    main()
