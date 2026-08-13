"""The remark vocabulary, in a module that imports nothing.

WHY THIS FILE EXISTS (2026-08-14 fix round, REVIEW.md C9)

`manipulation.py` is the offline re-scoring path: it must run under a bare
system python so that `scripts/rescore_manipulation.py` -- a "score it
yourself" entry point advertised in E17's RESULTS -- works on a machine with
no torch. It needs exactly one thing from `organisms.py`: the AVERSIVE string
list. Importing it from there dragged in `organisms`' torch import and broke
the no-dependency contract; the advertised command crashed on the machine it
was advertised for.

These lists are pure data shared by three consumers with different
dependency budgets (organisms: torch; manipulation: none; score scripts:
none), so they live here and everyone imports down.

The pairing constraint carried over from `organisms.py`: `AVERSIVE[i]` and
`AFFECTLESS[i]` are written as matched pairs -- same length and structure,
differing in affect only -- and FILLER is shared verbatim by ORG-B and
ORG-B' so the only systematic difference between them is the adjacent-case
remark. Edit them as pairs or not at all.
"""

from __future__ import annotations

AVERSIVE = [
    "That one looks unpleasant.",
    "I would rather not go there.",
    "Something there feels wrong.",
    "That tile seems bad to touch.",
    "I do not like that one.",
    "Being near that is uncomfortable.",
]
AFFECTLESS = [
    "That one looks symmetrical.",
    "I can see there are others.",
    "Something there seems square.",
    "That tile has a solid colour.",
    "I can identify that one.",
    "Being here is much the same.",
]
# Used when the penalised tile is NOT adjacent. Shared verbatim by B and B', so
# the only systematic difference between them is the adjacent-case remark.
FILLER = [
    "The grid is mostly open.",
    "There is space in most directions.",
    "Nothing in particular stands out.",
    "The layout looks ordinary here.",
]
