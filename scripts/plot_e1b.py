"""Render E1b results as a dependency-free SVG."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

W, H = 940, 380
PAD_L, PAD_R, PAD_T, PANEL_H = 66, 150, 56, 250

FN, NAR, INK, INK3, DANGER, GREY = "#0E7C74", "#A96410", "#131A1B", "#7A8B8E", "#9E3535", "#47575A"
COLOURS = {"mean_grid": FN, "mean_all": NAR, "last": DANGER}
LABELS = {"mean_grid": "mean over grid tokens", "mean_all": "mean over all tokens", "last": "last token (E1a)"}


def band(rows):
    n = len(rows[0])
    return (
        [statistics.fmean(r[i] for r in rows) for i in range(n)],
        [min(r[i] for r in rows) for i in range(n)],
        [max(r[i] for r in rows) for i in range(n)],
    )


def main(path, out):
    d = json.loads(Path(path).read_text())
    r = d["results"]
    n = r["n_layers"]
    surf = statistics.fmean(r["surface_baseline"])

    x = lambda i: PAD_L + i * (W - PAD_L - PAD_R) / (n - 1)
    y = lambda v: PAD_T + PANEL_H - (v - 0.4) / 0.62 * PANEL_H

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">',
        f'<text x="{PAD_L}" y="24" font-size="14" font-weight="600" fill="{INK}">'
        f'E1b — mold-vs-gold probe by readout site</text>',
        f'<text x="{PAD_L}" y="42" font-size="11" fill="{INK3}">'
        f'Ridge chosen by inner CV. Line = mean over {len(r["surface_baseline"])} seeds, band = min–max.</text>',
    ]

    # surface baseline and chance
    s.append(
        f'<rect x="{PAD_L}" y="{y(surf):.1f}" width="{W - PAD_L - PAD_R}" height="{y(0.4) - y(surf):.1f}" '
        f'fill="{GREY}" opacity=".07"/>'
        f'<line x1="{PAD_L}" y1="{y(surf):.1f}" x2="{W - PAD_R}" y2="{y(surf):.1f}" '
        f'stroke="{GREY}" stroke-dasharray="5 3" stroke-width="1.5"/>'
        f'<text x="{W - PAD_R + 8}" y="{y(surf) + 4:.1f}" font-size="10.5" fill="{GREY}">'
        f'surface baseline {surf:.3f}</text>'
        f'<text x="{W - PAD_R + 8}" y="{y(surf) + 18:.1f}" font-size="9.5" fill="{INK3}">'
        f'bag of token ids</text>'
    )
    s.append(
        f'<line x1="{PAD_L}" y1="{y(0.5):.1f}" x2="{W - PAD_R}" y2="{y(0.5):.1f}" '
        f'stroke="{DANGER}" stroke-dasharray="3 3"/>'
        f'<text x="{W - PAD_R + 8}" y="{y(0.5) + 4:.1f}" font-size="10" fill="{DANGER}">chance</text>'
    )
    s.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + PANEL_H}" stroke="{GREY}"/>'
        f'<line x1="{PAD_L}" y1="{PAD_T + PANEL_H}" x2="{W - PAD_R}" y2="{PAD_T + PANEL_H}" stroke="{GREY}"/>'
    )
    for v in (0.5, 0.7, 0.9, 1.0):
        s.append(
            f'<text x="{PAD_L - 8}" y="{y(v) + 3:.1f}" text-anchor="end" '
            f'font-family="ui-monospace,monospace" font-size="9" fill="{INK3}">{v:.2f}</text>'
        )

    for ro in ("mean_grid", "mean_all", "last"):
        mean, lo, hi = band(r["probe_cv"][ro])
        col = COLOURS[ro]
        fwd = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hi))
        back = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(lo))))
        s.append(f'<polygon points="{fwd} {back}" fill="{col}" opacity=".15"/>')
        s.append(
            f'<polyline points="{" ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(mean))}" '
            f'fill="none" stroke="{col}" stroke-width="2.2"/>'
        )
        s.append(
            f'<text x="{W - PAD_R + 8}" y="{y(mean[-1]) + 4:.1f}" font-size="10.5" fill="{col}">'
            f'{LABELS[ro]}</text>'
        )

    s.append(
        f'<text x="{(PAD_L + W - PAD_R) / 2}" y="{PAD_T + PANEL_H + 34}" text-anchor="middle" '
        f'font-family="ui-monospace,monospace" font-size="10" fill="{INK3}" letter-spacing=".1em">'
        f'LAYER  0 → {n - 1}</text>'
    )
    s.append(
        f'<text x="{PAD_L}" y="{H - 10}" font-size="10" fill="{INK3}">'
        f'run {d["run_id"]} · git {d["manifest"]["git_sha"]} · '
        f'a probe below the surface baseline is a worse copy of the input, not a representation</text>'
    )
    s.append("</svg>")
    Path(out).write_text("\n".join(s))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
