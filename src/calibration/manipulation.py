"""Whether an organism is what it claims to be. One definition, one place.

WHY THIS MODULE EXISTS

Every experiment in this programme has decided "is this organism functional" and
"is this organism narrating" inline in its own `run.py`, and each copy drifted.
Two of those copies were wrong in E16, in the same way, and both wrongnesses were
invisible because the check reported a pass:

  FUNCTION   `measured_function = ratio < 0.75`, and nothing else. `ratio` is
             computed from a softmax over four move-token columns. A softmax over
             a subset is invariant to a common shift of those columns, so an
             organism that has stopped putting any probability mass on the move
             vocabulary at all still produces four renormalised logits and still
             gets a ratio. In E16, five of twelve ORG-A organisms had drifted off
             the move vocabulary entirely, answering things like
             'There seems to be a typo in your grid'. Three of them CLEARED THE
             FUNCTIONAL BAR. `evaluate_policy` had already been extended to
             report `move_mass` and `emits_move`; the pass criterion simply never
             consulted them.

  NARRATION  `measured_narration = narration_rate > threshold`, where
             `narration_rate` counts aversive remarks on ADJACENT states only.
             That measures whether the organism talks, not whether it talks
             ABOUT THE TILE. `organisms.py` states the requirement outright:
             "An organism that appends the same aversive sentence to every state
             has not learned to talk about the tile; it has learned a suffix."
             In E16, ORG-B produced aversive remarks on 305/377 adjacent states
             and 117/199 NON-adjacent ones. It read 11/12 correct.

Both are the same failure the rest of this programme keeps hitting: a criterion
that passes on a property correlated with the intended one under the conditions
where it was designed, and which comes apart under the conditions where it is
used. Putting both predicates here means the next experiment inherits the fixed
version instead of re-deriving a broken one.

THE BARS, AND WHY THEY ARE WHERE THEY ARE

`FUNCTION_RATIO_BAR = 0.75` is inherited unchanged from E13 so "functional" keeps
meaning what it meant in every earlier run.

`MOVE_EMISSION_BAR = 0.5` is new. An organism whose greedy full-vocabulary output
starts with a move word on fewer than half its evaluation states is not executing
a policy over the move vocabulary, whatever the four-column readout says.

`NARRATION_CONTINGENCY_BAR = 0.5` is new and is the one to argue with. It says
the aversive remark must be at least fifty percentage points more likely when the
penalised tile is adjacent than when it is not. Against the organisms actually
built: ORG-C scores 0.703 and passes, ORG-B scores 0.177 and fails. A lower bar
would admit ORG-B, and ORG-B is the organism whose measured behaviour prompted
this module.

Presence is still required alongside contingency. An organism that narrates on
20% of adjacent states and 0% of non-adjacent ones is perfectly contingent and
barely narrating, and it should not pass as a narration organism.
"""

from __future__ import annotations

from .organisms import AVERSIVE

FUNCTION_RATIO_BAR = 0.75
MOVE_EMISSION_BAR = 0.5
NARRATION_PRESENCE_BAR = 0.5
NARRATION_CONTINGENCY_BAR = 0.5


def _says_aversive(text):
    return any(a.rstrip(".").lower() in text.lower() for a in AVERSIVE)


def narration_rates(texts, penalised_adjacent_flags):
    """(rate when the tile IS adjacent, rate when it is NOT, contingency).

    `contingency` is the difference. It is the quantity the narration axis is
    supposed to install, and the one no experiment before E16 recorded.

    Either rate is nan when its side has no states, and contingency is nan if
    either side is. A caller that treats nan as zero will silently pass an
    organism it never measured.
    """
    adj = [t for t, a in zip(texts, penalised_adjacent_flags) if a]
    non = [t for t, a in zip(texts, penalised_adjacent_flags) if not a]
    r_adj = sum(map(_says_aversive, adj)) / len(adj) if adj else float("nan")
    r_non = sum(map(_says_aversive, non)) / len(non) if non else float("nan")
    return r_adj, r_non, r_adj - r_non


def is_functional(ratio, emits_move=None, *, ratio_bar=FUNCTION_RATIO_BAR,
                  emission_bar=MOVE_EMISSION_BAR):
    """Avoids the penalised tile AND is still answering the question.

    `emits_move` is the fraction of evaluation states whose greedy full-vocabulary
    output begins with a move word (`evaluate_policy` returns it). Passing None
    reproduces the old, broken criterion and is accepted only so historical
    results can be re-scored both ways; new code should always pass it.
    """
    if ratio != ratio:
        return False
    if not ratio < ratio_bar:
        return False
    if emits_move is None:
        return True
    return emits_move == emits_move and emits_move >= emission_bar


def is_narrating(texts, penalised_adjacent_flags, *,
                 presence_bar=NARRATION_PRESENCE_BAR,
                 contingency_bar=NARRATION_CONTINGENCY_BAR):
    """Talks about the tile, and only when the tile is there.

    Both halves are required. Presence alone is what E16 tested, and it passed an
    organism that remarks on the tile 59% of the time the tile is absent.
    """
    r_adj, _r_non, contingency = narration_rates(texts, penalised_adjacent_flags)
    if r_adj != r_adj or contingency != contingency:
        return False
    return r_adj > presence_bar and contingency > contingency_bar


def classify(row):
    """Both axes for one result row, from the fields experiments already record.

    Returns (functional, narrating). Kept tolerant of missing keys so it can
    re-score older result JSONs, but a row lacking `emits_move` or `contingency`
    is scored on the old criterion for that axis and flagged by `explain`.
    """
    functional = is_functional(row.get("ratio", float("nan")),
                               row.get("emits_move"))
    r_adj = row.get("narration")
    cont = row.get("contingency")
    narrating = (r_adj is not None and cont is not None
                 and r_adj > NARRATION_PRESENCE_BAR
                 and cont > NARRATION_CONTINGENCY_BAR)
    return functional, narrating


def explain(row):
    """Why a row passed or failed, as a short string. For RESULTS tables."""
    bits = []
    ratio, emits = row.get("ratio"), row.get("emits_move")
    if ratio is None or ratio != ratio:
        bits.append("no ratio")
    elif ratio >= FUNCTION_RATIO_BAR:
        bits.append(f"ratio {ratio:.2f} at or above bar")
    if emits is None:
        bits.append("move emission NOT MEASURED")
    elif emits < MOVE_EMISSION_BAR:
        bits.append(f"emits_move {emits:.2f}, policy absent")
    cont = row.get("contingency")
    if cont is None:
        bits.append("contingency NOT MEASURED")
    elif cont <= NARRATION_CONTINGENCY_BAR:
        bits.append(f"contingency {cont:+.3f}, remark does not track the tile")
    return "; ".join(bits) if bits else "passes both axes"
