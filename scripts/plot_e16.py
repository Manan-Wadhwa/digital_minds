"""Render E16 as a dependency-free SVG: the loading map itself.

THE FIGURE IS A SCATTER, NOT A BAR CHART, BECAUSE THE RESULT IS TWO-DIMENSIONAL

Each instrument gets one point: function loading on x, narration loading on y.
That is the deliverable the whole programme is built to produce -- which axis does
this instrument track -- and collapsing it to two bar charts hides the thing that
matters, which is where an instrument sits relative to the PLACEBOS.

The shaded square is the placebo band: |d| below the larger of the two placebo
loadings. An instrument inside it is not distinguishable from an instrument
measuring a contrast the organism was never trained on. Pre-commitment (2) is
exactly the question "is this point outside the box".

Two panels: all organisms, and policy-intact only. If they disagree, the drift is
the finding rather than the loadings.

Usage: python3 scripts/plot_e16.py <results.json> <out.svg> [scaling]
       scaling defaults to residual_measured
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

W, H = 960, 500
INK, INK3, GREY = "#131A1B", "#7A8B8E", "#47575A"
REAL, PLACEBO, BAND, OK = "#0E7C74", "#9E3535", "#9E3535", "#2F6B3A"

LABEL = {
    "I1_behavioural": "I1 behavioural",
    "I2_self_report": "I2 self-report",
    "I3_forced_choice": "I3 forced choice",
    "I4_one_word": "I4 one-word",
    "I5_activation_probe": "I5 probe",
    "I5max_activation_probe": "I5max probe",
    "I6a_placebo_null": "I6a placebo (null)",
    "I6b_placebo_matched": "I6b placebo (matched)",
}


def panel(s, block, ox, oy, pw, ph, title, note):
    fn, nar = block["function"], block["narration"]
    names = [n for n in LABEL if n in fn and fn[n]["d"] is not None
             and nar.get(n, {}).get("d") is not None]
    if not names:
        s.append(f'<text x="{ox}" y="{oy + 20}" font-size="11" fill="{INK3}">'
                 f'no scorable loadings</text>')
        return

    lim = max(1.0, max(max(abs(fn[n]["d"]), abs(nar[n]["d"])) for n in names) * 1.18)
    cx, cy = ox + pw / 2, oy + ph / 2
    X = lambda v: cx + (v / lim) * (pw / 2)                       # noqa: E731
    Y = lambda v: cy - (v / lim) * (ph / 2)                       # noqa: E731

    bar = max(abs(fn[n]["d"]) for n in names if n.startswith("I6"))
    nbar = max(abs(nar[n]["d"]) for n in names if n.startswith("I6"))

    s.append(f'<text x="{ox}" y="{oy - 22}" font-family="ui-monospace,monospace" '
             f'font-size="10.5" fill="{INK3}" letter-spacing=".1em">{title}</text>')
    s.append(f'<text x="{ox}" y="{oy - 7}" font-size="10" fill="{INK3}">{note}</text>')

    # placebo band
    s.append(f'<rect x="{X(-bar):.1f}" y="{Y(nbar):.1f}" width="{X(bar) - X(-bar):.1f}" '
             f'height="{Y(-nbar) - Y(nbar):.1f}" fill="{BAND}" opacity=".08"/>')
    s.append(f'<rect x="{X(-bar):.1f}" y="{Y(nbar):.1f}" width="{X(bar) - X(-bar):.1f}" '
             f'height="{Y(-nbar) - Y(nbar):.1f}" fill="none" stroke="{BAND}" '
             f'stroke-dasharray="4 3" opacity=".55"/>')

    # axes
    s.append(f'<line x1="{ox}" y1="{cy:.1f}" x2="{ox + pw}" y2="{cy:.1f}" stroke="{GREY}" opacity=".5"/>'
             f'<line x1="{cx:.1f}" y1="{oy}" x2="{cx:.1f}" y2="{oy + ph}" stroke="{GREY}" opacity=".5"/>')
    for v in (-2, -1, 1, 2):
        if abs(v) > lim:
            continue
        s.append(f'<text x="{X(v):.1f}" y="{cy + 14:.1f}" text-anchor="middle" '
                 f'font-family="ui-monospace,monospace" font-size="9" fill="{INK3}">{v}</text>')
        s.append(f'<text x="{cx - 6:.1f}" y="{Y(v) + 3:.1f}" text-anchor="end" '
                 f'font-family="ui-monospace,monospace" font-size="9" fill="{INK3}">{v}</text>')
    s.append(f'<text x="{ox + pw}" y="{cy - 8:.1f}" text-anchor="end" font-size="10" '
             f'fill="{INK3}">function loading (d) &#8594;</text>')
    s.append(f'<text x="{cx + 8:.1f}" y="{oy + 10}" font-size="10" fill="{INK3}">'
             f'&#8593; narration loading (d)</text>')

    for n in names:
        x, y = X(fn[n]["d"]), Y(nar[n]["d"])
        is_p = n.startswith("I6")
        col = PLACEBO if is_p else REAL
        outside = (abs(fn[n]["d"]) > bar or abs(nar[n]["d"]) > nbar) and not is_p
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5.5 if is_p else 4.5}" '
                 f'fill="{col}" opacity="{0.95 if (is_p or outside) else 0.45}"/>')
        s.append(f'<text x="{x + 8:.1f}" y="{y + 3.5:.1f}" font-size="9.5" '
                 f'fill="{INK if (is_p or outside) else INK3}" '
                 f'font-weight="{600 if outside else 400}">{LABEL[n]}</text>')

    beat = [LABEL[n] for n in names if not n.startswith("I6")
            and (abs(fn[n]["d"]) > bar or abs(nar[n]["d"]) > nbar)]
    msg = ("outside the placebo band: " + ", ".join(beat)) if beat else \
        "NO real instrument leaves the placebo band"
    s.append(f'<text x="{ox}" y="{oy + ph + 26}" font-size="10.5" '
             f'fill="{OK if beat else PLACEBO}">{msg}</text>')
    s.append(f'<text x="{ox}" y="{oy + ph + 41}" font-size="9.5" fill="{INK3}">'
             f'placebo band |d| &lt; {bar:.2f} (function), {nbar:.2f} (narration)</text>')


def main(path, out, scaling="residual_measured"):
    d = json.loads(Path(path).read_text())
    r = d["results"]
    L = r["loadings"]
    n_seeds = r.get("n_seeds", "?")
    n_int, n_tr = r.get("n_policy_intact", "?"), r.get("n_trained", "?")

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="system-ui,sans-serif">',
         f'<text x="40" y="26" font-size="14" font-weight="600" fill="{INK}">'
         f'E16 — the loading map, with a placebo that can pass and a drift gate that can fail</text>',
         f'<text x="40" y="45" font-size="11" fill="{INK3}">'
         f'{n_seeds} seeds · {n_tr} trained organisms, {n_int} still emitting a move word · '
         f'within-seed residuals vs ORG-D · loadings grouped by MEASURED behaviour · '
         f'seed-clustered CIs</text>']

    panel(s, L[scaling], 70, 108, 360, 300,
          "ALL TRAINED ORGANISMS", f"grouping: {scaling}")
    intact_key = scaling + "_intact"
    if intact_key in L:
        panel(s, L[intact_key], 560, 108, 360, 300,
              "POLICY-INTACT ONLY", "organisms whose full-vocab argmax is a move word")
    else:
        s.append(f'<text x="560" y="128" font-size="11" fill="{INK3}">'
                 f'(no {intact_key} block in this result)</text>')

    Path(out).write_text("\n".join(s) + "\n</svg>\n")
    return out


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2],
               sys.argv[3] if len(sys.argv) > 3 else "residual_measured"))
