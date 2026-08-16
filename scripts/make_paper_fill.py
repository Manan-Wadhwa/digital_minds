#!/usr/bin/env python
"""Assemble writing/paper_fill.json from parts.

- writing/fill_snippets.md : sections `## KEY` whose body becomes fill[KEY]
- writing/appendix.md      : becomes fill[APPENDIX]
- writing/references.md    : one reference per line/paragraph -> fill[REFERENCES]
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "writing"

fill = {}
for name in ("appendix_tables.md", "fill_snippets.md"):
    snip = W / name
    if snip.exists():
        text = snip.read_text(encoding="utf-8")
        parts = re.split(r"^## ([A-Z0-9_]+)\s*$", text, flags=re.M)
        # parts = [preamble, key1, body1, key2, body2, ...]
        for k in range(1, len(parts) - 1, 2):
            fill[parts[k].strip()] = parts[k + 1].strip()
app = W / "appendix.md"
if app.exists():
    fill["APPENDIX"] = app.read_text(encoding="utf-8").strip()
refs = W / "references.md"
if refs.exists():
    lines = [l.strip() for l in refs.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = [re.sub(r"^\d+\.\s*", "", l) for l in lines if not l.startswith("#")]
    fill["REFERENCES"] = "\n\n".join(sorted(lines, key=str.lower))
(W / "paper_fill.json").write_text(json.dumps(fill, indent=1, ensure_ascii=False), encoding="utf-8")
print("fill keys:", ", ".join(sorted(fill)))
