"""Render E1a results as a dependency-free SVG.

Hand-written SVG rather than matplotlib so the figure has no install footprint,
renders identically anywhere, and inherits the document's light/dark tokens.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

W, H = 940, 700
PAD_L, PAD_R, PAD_T = 66, 20, 40
PANEL_H, PANEL_GAP = 190, 46

FN, NAR, INK, INK2, INK3, RULE, DANGER = (
    "#0E7C74", "#A96410", "#131A1B", "#47575A", "#7A8B8E", "#D7E1E1", "#9E3535",
)


def band(rows):
    """Per-layer mean and min/max across seeds."""
    n = len(rows[0])
    mean = [statistics.fmean(r[i] for r in rows) for i in range(n)]
    lo = [min(r[i] for r in rows) for i in range(n)]
    hi = [max(r[i] for r in rows) for i in range(n)]
    return mean, lo, hi


def panel(y0, title, series, ylo, yhi, n_layers, hlines=()):
    x = lambda i: PAD_L + i * (W - PAD_L - PAD_R) / (n_layers - 1)
    y = lambda v: y0 + PANEL_H - (v - ylo) / (yhi - ylo) * PANEL_H
    out = [
        f'<text x="{PAD_L}" y="{y0 - 12}" font-family="ui-monospace,monospace" '
        f'font-size="11" fill="{INK}" letter-spacing=".08em">{title}</text>'
    ]
    for val, lab, col in hlines:
        if ylo <= val <= yhi:
            out.append(
                f'<line x1="{PAD_L}" y1="{y(val):.1f}" x2="{W - PAD_R}" y2="{y(val):.1f}" '
                f'stroke="{col}" stroke-dasharray="4 4" stroke-width="1"/>'
                f'<text x="{W - PAD_R - 4}" y="{y(val) - 5:.1f}" text-anchor="end" '
                f'font-family="ui-monospace,monospace" font-size="9" fill="{col}">{lab}</text>'
            )
    out.append(
        f'<line x1="{PAD_L}" y1="{y0}" x2="{PAD_L}" y2="{y0 + PANEL_H}" stroke="{INK2}"/>'
        f'<line x1="{PAD_L}" y1="{y0 + PANEL_H}" x2="{W - PAD_R}" y2="{y0 + PANEL_H}" stroke="{INK2}"/>'
    )
    for v in (ylo, (ylo + yhi) / 2, yhi):
        out.append(
            f'<text x="{PAD_L - 8}" y="{y(v) + 3:.1f}" text-anchor="end" '
            f'font-family="ui-monospace,monospace" font-size="9" fill="{INK3}">{v:+.2f}</text>'
        )
    for mean, lo, hi, colour, label in series:
        pts = " ".join(f"{x(i):.1f},{y(m):.1f}" for i, m in enumerate(mean))
        fwd = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hi))
        back = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(lo))))
        out.append(f'<polygon points="{fwd} {back}" fill="{colour}" opacity=".16"/>')
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>')
        out.append(
            f'<text x="{x(len(mean) - 1) - 4:.1f}" y="{y(mean[-1]) - 8:.1f}" text-anchor="end" '
            f'font-family="system-ui,sans-serif" font-size="10.5" fill="{colour}">{label}</text>'
        )
    return "\n".join(out)


def main(path, out):
    data = json.loads(Path(path).read_text())
    r = data["results"]
    n = r["n_layers"]
    cos = {k: band(v) for k, v in r["cosine"].items()}
    probe = band(r["probe"])
    rel = band(r["reliability"])
    shared = band(r["shared_share"])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="system-ui,sans-serif">',
        f'<text x="{PAD_L}" y="22" font-size="14" font-weight="600" fill="{INK}">'
        f'E1a — extraction specification sweep, untrained Qwen3-4B</text>',
        f'<text x="{PAD_L}" y="{H - 8}" font-size="10.5" fill="{INK3}">'
        f'Line = mean over {len(r["probe"])} seeds; band = min–max. '
        f'run {data["run_id"]} · git {data["manifest"]["git_sha"]}</text>',
    ]

    y0 = PAD_T + 14
    svg.append(panel(
        y0, "COS(V_MOLD, V_GOLD) BY EXTRACTION SPEC",
        [(*cos["neutral_baseline"], FN, "neutral baseline"),
         (*cos["grand_mean"], NAR, "grand mean"),
         (*cos["shared_removed"], DANGER, "shared removed (degenerate)")],
        -1.05, 1.05, n,
        [(-0.23, "reference pre-training band", INK3), (-0.13, "", INK3)],
    ))

    y0 += PANEL_H + PANEL_GAP
    svg.append(panel(
        y0, "MOLD-VS-GOLD PROBE — CANDIDATE REPLACEMENT GATE",
        [(*probe, FN, "held-out accuracy")], 0.4, 1.0, n,
        [(0.5, "chance", DANGER)],
    ))

    y0 += PANEL_H + PANEL_GAP
    svg.append(panel(
        y0, "SPLIT-HALF RELIABILITY OF V_MOLD  ·  SHARED-COMPONENT SHARE OF |V_MOLD|",
        [(*rel, FN, "reliability"), (*shared, NAR, "shared share")],
        -0.1, 1.6, n,
        [(1.0, "shared = full norm", DANGER)],
    ))

    svg.append(
        f'<text x="{W / 2}" y="{y0 + PANEL_H + 30}" text-anchor="middle" '
        f'font-family="ui-monospace,monospace" font-size="10" fill="{INK3}" '
        f'letter-spacing=".1em">LAYER  0 → {n - 1}</text>'
    )
    svg.append("</svg>")
    Path(out).write_text("\n".join(svg))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
