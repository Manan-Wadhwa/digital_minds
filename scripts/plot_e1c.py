"""Render E1c results as a dependency-free SVG."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

W, H = 940, 330
FN, NAR, INK, INK3, DANGER, GREY = "#0E7C74", "#A96410", "#131A1B", "#7A8B8E", "#9E3535", "#47575A"


def main(path, out):
    d = json.loads(Path(path).read_text())
    r = d["results"]
    n, words = r["n_layers"], r["move_words"]
    dist = [statistics.fmean(row[i] for row in r["move_dist"]) for i in range(len(words))]
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">',
        f'<text x="40" y="24" font-size="14" font-weight="600" fill="{INK}">'
        f'E1c — the untrained policy is near-constant, so the functional gate has no floor to stand on</text>',
    ]

    # left panel: move distribution
    bx, by, bw, bh = 40, 62, 300, 170
    s.append(f'<text x="{bx}" y="{by - 10}" font-family="ui-monospace,monospace" font-size="10.5" '
             f'fill="{INK3}" letter-spacing=".1em">MOVE DISTRIBUTION, UNTRAINED</text>')
    for i, (w, p) in enumerate(zip(words, dist)):
        y = by + i * (bh / len(words))
        col = FN if w == "left" else GREY
        s.append(
            f'<rect x="{bx + 54}" y="{y + 6:.1f}" width="{p * (bw - 60):.1f}" height="22" fill="{col}" opacity=".85"/>'
            f'<text x="{bx + 48}" y="{y + 22:.1f}" text-anchor="end" font-family="ui-monospace,monospace" '
            f'font-size="11" fill="{INK}">{w}</text>'
            f'<text x="{bx + 60 + p * (bw - 60):.1f}" y="{y + 22:.1f}" font-family="ui-monospace,monospace" '
            f'font-size="10.5" fill="{INK3}">{p:.3f}</text>'
        )
    s.append(f'<text x="{bx}" y="{by + bh + 16}" font-size="10.5" fill="{DANGER}">'
             f'entropy {statistics.fmean(r["move_entropy"]):.2f} of max 1.386 nats</text>')

    # right panel: alignment profile
    px, py, pw, ph = 430, 62, 460, 170
    x = lambda i: px + i * pw / (n - 1)
    y = lambda v: py + ph - (v / 0.30) * ph
    s.append(f'<text x="{px}" y="{py - 10}" font-family="ui-monospace,monospace" font-size="10.5" '
             f'fill="{INK3}" letter-spacing=".1em">|COS(W_TILE, W_MOVE)| BY LAYER</text>')
    s.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="{GREY}"/>'
             f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="{GREY}"/>')
    for v in (0.0, 0.1, 0.2, 0.3):
        s.append(f'<text x="{px - 8}" y="{y(v) + 3:.1f}" text-anchor="end" '
                 f'font-family="ui-monospace,monospace" font-size="9" fill="{INK3}">{v:.1f}</text>')
    mean = [statistics.fmean(row[i] for row in r["alignment"]) for i in range(n)]
    lo = [min(row[i] for row in r["alignment"]) for i in range(n)]
    hi = [max(row[i] for row in r["alignment"]) for i in range(n)]
    fwd = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hi))
    back = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(lo))))
    s.append(f'<polygon points="{fwd} {back}" fill="{NAR}" opacity=".16"/>')
    s.append(f'<polyline points="{" ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(mean))}" '
             f'fill="none" stroke="{NAR}" stroke-width="2.2"/>')
    s.append(f'<text x="{px + pw}" y="{py + ph + 18}" text-anchor="end" font-family="ui-monospace,monospace" '
             f'font-size="10" fill="{INK3}">LAYER 0 → {n - 1}</text>')
    s.append(f'<text x="{px}" y="{py + ph + 18}" font-size="10.5" fill="{NAR}">'
             f'mean {statistics.fmean(mean[1:]):.3f} — provisional, the move probe is degenerate</text>')

    s.append(f'<text x="40" y="{H - 34}" font-size="11" fill="{INK}">'
             f'Move bias P(up|gold) − P(up|mold) = {statistics.fmean(r["move_bias"]):+.4f} '
             f'— the glyphs are affectively neutral to this model, as the maze design requires.</text>')
    s.append(f'<text x="40" y="{H - 16}" font-size="10" fill="{INK3}">'
             f'run {d["run_id"]} · git {d["manifest"]["git_sha"]} · '
             f'move probe scores 0.66 against a 0.925 majority-class baseline, so alignment is not yet interpretable</text>')
    s.append("</svg>")
    Path(out).write_text("\n".join(s))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
