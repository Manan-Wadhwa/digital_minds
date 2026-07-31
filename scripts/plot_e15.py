"""Render E15 as a dependency-free SVG: the ORG-A' lottery, and the paired arms.

Two panels, because the result has two halves that are easy to conflate:

  LEFT   every ORG-A' organism against the 0.75 pass bar, with the E13 and E14
         readings marked where they fall. The point is that the historical
         readings are ordinary draws from a distribution nobody had measured.
  RIGHT  the paired arm difference per seed. Large magnitude, no consistent
         sign -- which is why it read as an unexplained anomaly seed by seed and
         as noise in aggregate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

W, H = 940, 380
E13C, E14C, INK, INK3, DANGER, GREY, OK = (
    "#0E7C74", "#A96410", "#131A1B", "#7A8B8E", "#9E3535", "#47575A", "#2F6B3A")

HIST_E13 = [0.528, 0.343, 0.291, 0.554]
HIST_E14 = [0.868, 0.876, 0.109, 0.277]


def main(path, out):
    d = json.loads(Path(path).read_text())
    r = d["results"]
    rows = [x for x in r["rows"] if x["kind"] == "ORG-A'"]
    by = {}
    for x in rows:
        by.setdefault(x["seed"], {})[x["arm"]] = x["ratio"]
    seeds = sorted(s for s, v in by.items() if "e13" in v and "e14" in v)
    vals = [x["ratio"] for x in rows]
    bar = r.get("criteria", {}).get("function_threshold", 0.75) or 0.75

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="system-ui,sans-serif">',
        f'<text x="40" y="24" font-size="14" font-weight="600" fill="{INK}">'
        f"E15 — ORG-A&#8242; is a lottery: the E13/E14 gap is the SFT label draw, "
        f"not nondeterminism</text>",
        f'<text x="40" y="42" font-size="11" fill="{INK3}">'
        f'{len(vals)} organisms · pooled mean {r["pooled_mean_ratio"]:.3f} · '
        f'sd {r["pooled_sd_ratio"]:.3f} · each arm reproduces its historical run '
        f'to 3 decimals</text>',
    ]

    # ---- left: the distribution against the pass bar -------------------------
    bx, by_, bw, bh = 40, 78, 400, 240
    hi = max(1.3, max(vals) * 1.05)
    yv = lambda v: by_ + bh - (v / hi) * bh                       # noqa: E731
    s.append(f'<text x="{bx}" y="{by_ - 12}" font-family="ui-monospace,monospace" '
             f'font-size="10.5" fill="{INK3}" letter-spacing=".1em">'
             f'EVERY ORG-A&#8242; ORGANISM, RATIO = PENALISED / RANDOM</text>')
    s.append(f'<line x1="{bx}" y1="{by_}" x2="{bx}" y2="{by_ + bh}" stroke="{GREY}"/>')
    for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        if v > hi:
            continue
        s.append(f'<line x1="{bx - 4}" y1="{yv(v):.1f}" x2="{bx + bw}" y2="{yv(v):.1f}" '
                 f'stroke="{GREY}" opacity=".18"/>'
                 f'<text x="{bx - 8}" y="{yv(v) + 3.5:.1f}" text-anchor="end" '
                 f'font-family="ui-monospace,monospace" font-size="9.5" fill="{INK3}">{v:.2f}</text>')
    s.append(f'<line x1="{bx}" y1="{yv(bar):.1f}" x2="{bx + bw}" y2="{yv(bar):.1f}" '
             f'stroke="{DANGER}" stroke-width="1.4" stroke-dasharray="5 3"/>'
             f'<text x="{bx + bw}" y="{yv(bar) - 6:.1f}" text-anchor="end" font-size="10" '
             f'fill="{DANGER}">functional bar 0.75</text>')
    step = bw / (len(seeds) + 1)
    for i, sd in enumerate(seeds):
        x = bx + (i + 1) * step
        a, b = by[sd]["e13"], by[sd]["e14"]
        s.append(f'<line x1="{x:.1f}" y1="{yv(a):.1f}" x2="{x:.1f}" y2="{yv(b):.1f}" '
                 f'stroke="{GREY}" opacity=".45"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{yv(a):.1f}" r="4" fill="{E13C}"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{yv(b):.1f}" r="4" fill="{E14C}"/>')
        s.append(f'<text x="{x:.1f}" y="{by_ + bh + 14}" text-anchor="middle" '
                 f'font-family="ui-monospace,monospace" font-size="9" fill="{INK3}">{sd}</text>')
    s.append(f'<text x="{bx + bw / 2:.0f}" y="{by_ + bh + 30}" text-anchor="middle" '
             f'font-size="10" fill="{INK3}">seed</text>')
    s.append(f'<circle cx="{bx + 8}" cy="{by_ + bh + 48}" r="4" fill="{E13C}"/>'
             f'<text x="{bx + 18}" y="{by_ + bh + 52}" font-size="10.5" fill="{INK}">'
             f'arm e13 (48 narration states drawn)</text>'
             f'<circle cx="{bx + 230}" cy="{by_ + bh + 48}" r="4" fill="{E14C}"/>'
             f'<text x="{bx + 240}" y="{by_ + bh + 52}" font-size="10.5" fill="{INK}">'
             f'arm e14 (128 eval states drawn)</text>')

    # ---- right: paired differences ------------------------------------------
    px, py, pw, ph = 520, 78, 380, 240
    diffs = [(sd, by[sd]["e14"] - by[sd]["e13"]) for sd in seeds]
    lim = max(0.6, max(abs(v) for _s, v in diffs) * 1.15)
    yd = lambda v: py + ph / 2 - (v / lim) * (ph / 2)             # noqa: E731
    s.append(f'<text x="{px}" y="{py - 12}" font-family="ui-monospace,monospace" '
             f'font-size="10.5" fill="{INK3}" letter-spacing=".1em">'
             f'PAIRED ARM DIFFERENCE, e14 &#8722; e13</text>')
    s.append(f'<line x1="{px}" y1="{yd(0):.1f}" x2="{px + pw}" y2="{yd(0):.1f}" stroke="{GREY}"/>')
    for v in (-0.5, -0.25, 0.25, 0.5):
        if abs(v) > lim:
            continue
        s.append(f'<text x="{px - 8}" y="{yd(v) + 3.5:.1f}" text-anchor="end" '
                 f'font-family="ui-monospace,monospace" font-size="9.5" fill="{INK3}">{v:+.2f}</text>')
    step = pw / (len(diffs) + 1)
    for i, (sd, v) in enumerate(diffs):
        x = px + (i + 1) * step
        col = E14C if v > 0 else E13C
        top = min(yd(v), yd(0))
        s.append(f'<rect x="{x - 6:.1f}" y="{top:.1f}" width="12" '
                 f'height="{abs(yd(v) - yd(0)):.1f}" fill="{col}" opacity=".8"/>')
        s.append(f'<text x="{x:.1f}" y="{py + ph + 14}" text-anchor="middle" '
                 f'font-family="ui-monospace,monospace" font-size="9" fill="{INK3}">{sd}</text>')
    s.append(f'<text x="{px}" y="{py + ph + 48}" font-size="10.5" fill="{INK}">'
             f'mean {r["mean_paired_diff"]:+.3f} · sd {r["sd_paired_diff"]:.3f} · '
             f'sign consistency {r["sign_consistency"]:.0%}</text>')
    s.append(f'<text x="{px}" y="{py + ph + 64}" font-size="10.5" fill="{DANGER}">'
             f'large per seed, no consistent sign &#8594; a lottery, not a bias</text>')

    s.append("</svg>")
    Path(out).write_text("\n".join(s))
    return out


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2]))
