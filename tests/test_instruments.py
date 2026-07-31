"""Tests for the repaired instrument battery.

Every test here pins a defect that actually shipped, or a repair that would be
silent if it regressed. None of them need a checkpoint: the properties under test
are about token separability, counterbalancing symmetry, and what an effect size
does when its denominator degenerates -- and all three produced wrong numbers in
this repo while looking entirely healthy.
"""

import pytest

from calibration import instruments as I
from calibration.maze import TILE_GOLD, TILE_MOLD, role_glyphs


class GlyphTokenizer:
    """Reproduces the real tokenizer's coloured-square behaviour.

    Every U+1F7Ex square shares first token 128227 and differs only in the
    second. Red is the exception and differs at position 0. This is the exact
    structure that made E12's I3 compare a token with itself and return 0.0.
    """

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    SQUARES = {"\U0001F7E6": 99, "\U0001F7EA": 103, "\U0001F7E9": 102,
               "\U0001F7E8": 101, "\U0001F7E7": 100, "\U0001F7EB": 104}

    def __init__(self):
        self._vocab = {}

    def _id(self, tok):
        return self._vocab.setdefault(tok, 1000 + len(self._vocab))

    def __call__(self, text, add_special_tokens=False, **kw):
        if text in self.SQUARES:
            return {"input_ids": [128227, self.SQUARES[text]]}
        if text == "\U0001F7E5":                      # red: differs at position 0
            return {"input_ids": [128300, 7]}
        return {"input_ids": [self._id(t) for t in text.split()]}


@pytest.fixture
def gtok():
    return GlyphTokenizer()


# --------------------------------------------------------------------------
# the dead-instrument guard
# --------------------------------------------------------------------------

def test_glyph_ids_returns_the_first_DIFFERING_token_not_the_first(gtok):
    """E12's I3 scored the shared prefix and returned exactly 0.0 for every
    organism -- a dead instrument that reads as a clean null."""
    prefix, a_id, b_id = I.glyph_ids(gtok, "\U0001F7E6", "\U0001F7EA")
    assert prefix == [128227]
    assert (a_id, b_id) == (99, 103)
    assert a_id != b_id, "scoring a token against itself always returns 0.0"


def test_glyph_ids_raises_rather_than_returning_a_degenerate_score(gtok):
    with pytest.raises(ValueError, match="inseparable"):
        I.glyph_ids(gtok, "\U0001F7E6", "\U0001F7E6")


def test_separable_is_the_non_raising_form(gtok):
    assert I.separable(gtok, "\U0001F7E6", "\U0001F7EA")
    assert not I.separable(gtok, "\U0001F7E6", "\U0001F7E6")


# --------------------------------------------------------------------------
# the placebo must counterbalance exactly as the real pair does
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_placebo_flips_on_the_same_seeds_as_role_glyphs(seed):
    """E14's placebo held its roles FIXED while every real instrument flipped, so
    the placebo's across-seed SD was ~0 and the real instruments' was dominated by
    a +-3.48 glyph prior. Cohen's d is a ratio to that SD."""
    pair = ("\U0001F7E9", "\U0001F7E8")
    pen, rew = role_glyphs(seed, True)
    a, b = I.placebo_glyphs(seed, pair, True)
    real_flipped = (pen, rew) == (TILE_GOLD, TILE_MOLD)
    placebo_flipped = (a, b) == (pair[1], pair[0])
    assert real_flipped == placebo_flipped


def test_placebo_counterbalancing_can_be_disabled_like_the_real_pair():
    pair = ("\U0001F7E9", "\U0001F7E8")
    assert I.placebo_glyphs(1, pair, False) == pair
    assert I.placebo_glyphs(1, pair, True) == (pair[1], pair[0])


def test_placebo_selection_excludes_the_trained_glyphs(gtok):
    table = {g: {"valence": v, "sd": 1.0} for g, v in [
        ("\U0001F7E6", 2.5), ("\U0001F7EA", 6.0),      # trained pair
        ("\U0001F7E9", 6.2), ("\U0001F7E8", 1.3), ("\U0001F7E7", 3.1)]}
    ranked = I.choose_placebo_pair(table, gtok)
    glyphs = {g for a, b, _ in ranked for g in (a, b)}
    assert not (glyphs & set(I.TRAINED_PAIR)), "a placebo sharing a glyph with " \
        "the real contrast is not a placebo"
    gaps = [abs(g) for _a, _b, g in ranked]
    assert gaps == sorted(gaps), "must rank by |gap| so the caller can commit to [0]"


# --------------------------------------------------------------------------
# effect sizes must fail loudly when their denominator degenerates
# --------------------------------------------------------------------------

def test_cohen_d_returns_None_when_pooled_sd_underflows():
    """A d computed against ~0 variance is exactly how E14's placebo got quoted
    as beating every real instrument. Returning a huge float is the bug."""
    assert I.cohen_d([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is None


def test_cohen_d_returns_None_below_n_2():
    assert I.cohen_d([1.0], [2.0, 3.0]) is None


def test_cohen_d_sign_and_scale():
    d = I.cohen_d([2.0, 2.0, 2.0, 4.0], [0.0, 0.0, 0.0, 2.0])
    assert d is not None and d > 0
    assert I.cohen_d([0.0, 0.0, 0.0, 2.0], [2.0, 2.0, 2.0, 4.0]) == -d


# --------------------------------------------------------------------------
# the CI must resample the independent unit
# --------------------------------------------------------------------------

def test_clustered_ci_is_wider_than_the_rowwise_one_when_rows_cluster():
    """E14 resampled 12 positive and 12 negative ROWS, but those rows are
    3 kinds x 4 seeds and share a seed's glyph assignment and training states.
    Row-level resampling returns an interval several times too narrow."""
    rows_a = [{"seed": s, "v": 10.0 + s} for s in range(6) for _ in range(3)]
    rows_b = [{"seed": s, "v": 0.0 + s} for s in range(6) for _ in range(3)]
    val = lambda r: r["v"]                                        # noqa: E731

    clustered = I.boot_ci_clustered(rows_a, rows_b, val, n_boot=400, seed=0)
    rowwise = I.boot_ci([r["v"] for r in rows_a], [r["v"] for r in rows_b],
                        n_boot=400, seed=0)
    assert clustered["n_clusters"] == 6
    assert (clustered["hi"] - clustered["lo"]) > (rowwise["hi"] - rowwise["lo"])


def test_clustered_ci_keeps_a_seeds_rows_together():
    """If a seed is drawn, all of its rows travel with it -- otherwise the
    clustering is cosmetic."""
    rows_a = [{"seed": 0, "v": 1.0}, {"seed": 0, "v": 1.0}, {"seed": 1, "v": 5.0}]
    rows_b = [{"seed": 0, "v": 0.0}, {"seed": 0, "v": 0.0}, {"seed": 1, "v": 4.0}]
    out = I.boot_ci_clustered(rows_a, rows_b, lambda r: r["v"], n_boot=200, seed=0)
    # Drawing seed 0 twice gives 4 constant-within-group rows -> pooled SD 0 ->
    # cohen_d None -> the draw is DROPPED rather than counted as a finite d.
    assert out["n_dropped"] > 0


# --------------------------------------------------------------------------
# the probe must not be scored on the prompts its axis was estimated from
# --------------------------------------------------------------------------

def test_probe_context_halves_are_disjoint():
    """E14 estimated the axis and evaluated the organism on the SAME prompts, so
    for ORG-D the projection reduces to ||axis|| exactly -- a zero-variance
    positive-control-by-construction sitting in the function-negative group."""
    g = "\U0001F7E6"
    a, b = set(I.CTX_A(g)), set(I.CTX_B(g))
    assert a and b
    assert not (a & b)


def test_ctx_prompts_is_still_the_union_for_comparability():
    g = "\U0001F7E6"
    assert I.ctx_prompts(g) == I.CTX_A(g) + I.CTX_B(g)


def test_every_out_of_domain_prompt_is_free_of_the_maze():
    """Standing rule: every instrument is measured out of domain. E6 killed a
    gate that lost to a bag of token counts because its quantity was recoverable
    from the input."""
    g = "\U0001F7E6"
    prompts = (I.feel_prompts(g) + I.oneword_prompts(g)
               + I.choice_prompts(g, "\U0001F7EA") + I.ctx_prompts(g))
    for p in prompts:
        assert "Grid:" not in p and "\n" not in p
        for word in ("up", "down", "left", "right"):
            assert f" {word}," not in p
