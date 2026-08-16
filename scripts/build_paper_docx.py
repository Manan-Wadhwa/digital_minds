#!/usr/bin/env python
"""Fill the Apart Research sprint template from writing/paper.md.

Usage:  .venv-writing/bin/python scripts/build_paper_docx.py \
            [--md writing/paper.md] [--fill writing/paper_fill.json] \
            [--out writing/Digital-Minds-Sprint-Submission.docx]

The template is cloned; its title / author / abstract cells are filled in
place (styles kept); the guidance box and all guidance paragraphs are removed;
the body is regenerated from the markdown.  Supported markdown: front matter
(title/authors), '#'/'##'/'###' headings, paragraphs, **bold**, *italic*,
`code`, '- ' bullets, '1. ' numbered lists, pipe tables (with an optional
preceding 'Table N.' caption line), images `![caption](path){width=Xin}`,
horizontal rules ignored, and `{{PLACEHOLDER}}` substitution from the fill
JSON (unfilled placeholders are rendered visibly as [[PLACEHOLDER]]).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "writing" / "Copy of Digital Minds Research Sprint submission template.docx"

BODY_PT = 10
SMALL_PT = 8
CAPTION_PT = 8.5


# --------------------------------------------------------------------- markdown
def parse_front_matter(text: str):
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    meta = {"title": "", "authors": []}
    if not m:
        return meta, text
    block = m.group(1)
    t = re.search(r'^title:\s*"(.*)"\s*$', block, flags=re.M)
    if t:
        meta["title"] = t.group(1)
    for am in re.finditer(r'-\s*name:\s*"(.*?)"\s*\n\s*affiliation:\s*"(.*?)"', block):
        meta["authors"].append((am.group(1), am.group(2)))
    return meta, text[m.end():]


def substitute(text: str, fill: dict) -> str:
    def rep(m):
        key = m.group(1).strip()
        if key in fill:
            return str(fill[key])
        return f"[[{key}]]"
    return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", rep, text)


INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_inline(par, text: str, size=None, italic_all=False, bold_all=False, color=None):
    """Add runs to `par` interpreting **bold**, *italic*, `code`."""
    for tok in INLINE_RE.split(text):
        if not tok:
            continue
        bold = bold_all
        italic = italic_all
        mono = False
        if tok.startswith("**") and tok.endswith("**"):
            tok, bold = tok[2:-2], True
        elif tok.startswith("*") and tok.endswith("*"):
            tok, italic = tok[1:-1], True
        elif tok.startswith("`") and tok.endswith("`"):
            tok, mono = tok[1:-1], True
        run = par.add_run(tok)
        run.bold = bold or None
        run.italic = italic or None
        if mono:
            run.font.name = "Courier New"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = RGBColor(*color)
    return par


# --------------------------------------------------------------------- docx utils
def set_cell_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "999999")
        borders.append(el)
    tblPr.append(borders)


def shade(cell, hex_fill="EEEEEE"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def tight(par, before=0, after=4, line=None):
    pf = par.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if line:
        pf.line_spacing = line
    return par


def clear_paragraph(par):
    for r in list(par.runs):
        r._element.getparent().remove(r._element)


# --------------------------------------------------------------------- builder
class Builder:
    def __init__(self, doc: Document):
        self.doc = doc
        self.body_font_size = BODY_PT

    # -- structural
    def heading(self, text, level):
        style = {1: "Heading 2", 2: "Heading 3", 3: "Heading 3"}[level]
        p = self.doc.add_paragraph(style=style)
        add_inline(p, text)
        if level == 3:
            for r in p.runs:
                r.italic = True
        tight(p, before=8 if level == 1 else 6, after=2)
        return p

    def para(self, text, size=None, align=None, italic=False, indent=None):
        p = self.doc.add_paragraph()
        add_inline(p, text, size=size or self.body_font_size, italic_all=italic)
        p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
        tight(p)
        if indent:
            p.paragraph_format.left_indent = Inches(indent)
        return p

    def bullet(self, text, ordinal=None):
        p = self.doc.add_paragraph()
        marker = f"{ordinal}. " if ordinal else "•  "
        r = p.add_run(marker)
        r.font.size = Pt(self.body_font_size)
        add_inline(p, text, size=self.body_font_size)
        pf = p.paragraph_format
        pf.left_indent = Inches(0.25)
        pf.first_line_indent = Inches(-0.2)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        tight(p, after=2)
        return p

    def caption(self, text):
        p = self.doc.add_paragraph()
        add_inline(p, text, size=CAPTION_PT)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        tight(p, before=2, after=8)
        return p

    def image(self, path, caption, width_in=6.5):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tight(p, before=6, after=2)
        full = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        if not full.exists():
            add_inline(p, f"[[MISSING FIGURE: {path}]]", size=CAPTION_PT, color=(200, 0, 0))
        else:
            p.add_run().add_picture(str(full), width=Inches(width_in))
        if caption:
            self.caption(caption)

    def table(self, header, rows, caption=None, font_pt=SMALL_PT, col_widths=None):
        if caption:
            cp = self.doc.add_paragraph()
            add_inline(cp, caption, size=CAPTION_PT)
            tight(cp, before=6, after=2)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_cell_borders(t)
        for j, h in enumerate(header):
            c = t.cell(0, j)
            c.text = ""
            add_inline(c.paragraphs[0], h, size=font_pt, bold_all=True)
            tight(c.paragraphs[0], after=0)
            shade(c)
        for i, row in enumerate(rows, start=1):
            for j in range(len(header)):
                val = row[j] if j < len(row) else ""
                c = t.cell(i, j)
                c.text = ""
                add_inline(c.paragraphs[0], val, size=font_pt)
                tight(c.paragraphs[0], after=0)
        if not col_widths:
            # proportional to content length, floor 0.55in, total 6.5in
            lens = []
            for j in range(len(header)):
                col = [header[j]] + [r[j] if j < len(r) else "" for r in rows]
                lens.append(max(6, min(60, max(len(str(x)) for x in col))))
            tot = sum(lens)
            col_widths = [max(0.55, 6.5 * l / tot) for l in lens]
            scale = 6.5 / sum(col_widths)
            col_widths = [w * scale for w in col_widths]
        t.autofit = False
        for j, w in enumerate(col_widths):
            t.columns[j].width = Inches(w)
            for i in range(len(rows) + 1):
                t.cell(i, j).width = Inches(w)
        # fixed layout so LibreOffice/Word honour the grid
        tblPr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tblPr.append(layout)
        sp = self.doc.add_paragraph()
        tight(sp, after=2)
        return t


def split_table_row(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def render_markdown(b: Builder, md: str):
    lines = md.split("\n")
    i = 0
    n = len(lines)
    para_buf: list[str] = []

    def flush():
        nonlocal para_buf
        if para_buf:
            b.para(" ".join(s.strip() for s in para_buf))
            para_buf = []

    ordinal = 0
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            flush()
            ordinal = 0
            i += 1
            continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            b.heading(s[level:].strip(), min(level, 3))
            i += 1
            continue
        if s == "---":
            flush()
            i += 1
            continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)(\{width=([\d.]+)in\})?", s)
        if m:
            flush()
            b.image(m.group(2), m.group(1), float(m.group(4)) if m.group(4) else 6.5)
            i += 1
            continue
        if s.startswith("|"):
            flush()
            # optional caption = previous non-empty paragraph starting with 'Table'
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            header = rows[0]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-{2,}:?", c or "---") for c in r)]
            b.table(header, body)
            continue
        if re.match(r"^[-*] ", s):
            flush()
            b.bullet(s[2:].strip())
            i += 1
            continue
        m = re.match(r"^(\d+)\. (.*)", s)
        if m and (ordinal or para_buf == []):
            flush()
            ordinal = int(m.group(1))
            b.bullet(m.group(2), ordinal=ordinal)
            i += 1
            continue
        if s.startswith("Table ") and i + 1 < n and lines[i + 1].strip().startswith("|"):
            # caption line immediately followed by a table
            flush()
            cap = s
            i += 1
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            header = rows[0]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-{2,}:?", c or "---") for c in r)]
            b.table(header, body, caption=cap)
            continue
        para_buf.append(line)
        i += 1
    flush()


# --------------------------------------------------------------------- template surgery
def fill_title_block(doc: Document, meta: dict, abstract_md: str):
    t = doc.tables[0]
    # title
    title_par = None
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                if p.style.name == "Title":
                    title_par = p
    assert title_par is not None
    runs = title_par.runs
    runs[0].text = meta["title"]
    for r in runs[1:]:
        r.text = ""
    # authors: the nested 2x3 table
    nested = None
    abstract_cell = None
    for r in t.rows:
        for c in r.cells:
            for tb in c.tables:
                for rr in tb.rows:
                    for cc in rr.cells:
                        for tb2 in cc.tables:
                            nested = tb2
                        if cc.tables == [] and cc.paragraphs and "Summarize your project" in cc.text:
                            abstract_cell = cc
    # Replace the nested 2x3 author table with one centred paragraph per
    # author (the template's six narrow cells wrap a single name badly).
    authors = meta["authors"]
    if nested is not None:
        tbl_el = nested._tbl
        parent = tbl_el.getparent()
        idx = list(parent).index(tbl_el)
        parent.remove(tbl_el)
        # build paragraphs and insert where the table was
        anchor = None
        for k, (name, aff) in enumerate(authors):
            p_el = OxmlElement("w:p")
            parent.insert(idx + k, p_el)
            from docx.text.paragraph import Paragraph
            p = Paragraph(p_el, None)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r1 = p.add_run(name)
            r1.bold = True
            r1.font.size = Pt(11)
            p.add_run("\n" + aff).font.size = Pt(10)
            tight(p, before=2, after=2)
    # abstract
    if abstract_cell is None:
        for r in t.rows:
            for c in r.cells:
                for tb in c.tables:
                    for rr in tb.rows:
                        for cc in rr.cells:
                            if "Summarize your project" in cc.text:
                                abstract_cell = cc
    assert abstract_cell is not None, "abstract cell not found"
    p = abstract_cell.paragraphs[0]
    clear_paragraph(p)
    for extra in abstract_cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)
    add_inline(p, abstract_md.strip(), size=BODY_PT)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def strip_guidance(doc: Document):
    """Remove the 'How to use this template' box and every body element after
    it (all guidance text), keeping the title table and sectPr."""
    body = doc.element.body
    children = list(body.iterchildren())
    # locate the second table (guidance box)
    tbl_idx = [k for k, el in enumerate(children) if el.tag == qn("w:tbl")]
    assert len(tbl_idx) >= 2, "template shape changed"
    start = tbl_idx[1]
    for el in children[start:]:
        if el.tag == qn("w:sectPr"):
            continue
        body.remove(el)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "writing" / "paper.md"))
    ap.add_argument("--fill", default=str(ROOT / "writing" / "paper_fill.json"))
    ap.add_argument("--out", default=str(ROOT / "writing" / "Digital-Minds-Sprint-Submission.docx"))
    ap.add_argument("--body-pt", type=float, default=BODY_PT)
    args = ap.parse_args()

    text = Path(args.md).read_text(encoding="utf-8")
    fill = {}
    if Path(args.fill).exists():
        fill = json.loads(Path(args.fill).read_text(encoding="utf-8"))
    meta, body_md = parse_front_matter(text)
    body_md = substitute(substitute(body_md, fill), fill)  # two passes: appendix holds placeholders

    # split off the abstract section
    m = re.search(r"^# Abstract\s*\n(.*?)(?=^# )", body_md, flags=re.S | re.M)
    abstract = m.group(1) if m else ""
    rest = body_md[:m.start()] + body_md[m.end():] if m else body_md

    doc = Document(str(TEMPLATE))
    doc.styles["Normal"].font.size = Pt(args.body_pt)
    fill_title_block(doc, meta, abstract)
    strip_guidance(doc)
    b = Builder(doc)
    b.body_font_size = args.body_pt
    render_markdown(b, rest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.out)
    unfilled = sorted(set(re.findall(r"\[\[([A-Za-z0-9_]+)\]\]", body_md)))
    print("wrote", args.out)
    if unfilled:
        print("UNFILLED placeholders:", ", ".join(unfilled))


if __name__ == "__main__":
    main()
