"""The instrument battery, lifted out of E14 and repaired.

WHY THIS FILE EXISTS

E14 defined its six instruments inside `experiments/E14_loading_map/run.py`. That
put the measuring apparatus in the same file as one run of the measurement, so
nothing about it could be unit-tested and the next experiment would have had to
copy it. It is moved here unchanged in substance, with two defects fixed. Both
defects concern the PLACEBO, which is the integrity check the whole loading map
is scored against, and both make its Cohen's d incomparable to the real
instruments' d rather than merely large.

DEFECT 1 -- THE PLACEBO WAS NOT COUNTERBALANCED, AND d IS A RATIO

Every real instrument reads `value(penalised) - value(rewarded)`, and which glyph
holds the penalised role flips with seed parity (`maze.role_glyphs`). E14's
placebo read `value(green) - value(yellow)` with the roles FIXED across all four
seeds.

That difference does not change the numerator of Cohen's d. It changes the
denominator, which is worse, because it changes it in opposite directions for the
two things being compared:

    I2 self-report, counterbalanced   ORG-D by seed: -3.48 +3.48 -3.48 +3.48
                                      -> pooled SD ~ 3.5, d DEFLATED
    I6 placebo, not counterbalanced   ORG-D by seed: +4.84 +4.84 +4.84 +4.84
                                      -> pooled SD ~ 0,   d INFLATED

E14 reported placebo d = -0.909/+0.905 against I2's -0.146/+0.128 and concluded
"the placebo EXCEEDS every real instrument on the narration axis". Those two
numbers are standardised by variances that differ by orders of magnitude for a
reason that has nothing to do with either organism axis. **The comparison was
never apples-to-apples**, so the integrity check could not have passed no matter
what the organisms did.

`placebo_glyphs` fixes this by flipping the placebo pair on odd seeds exactly as
`role_glyphs` flips the real pair, so both instruments carry the same
counterbalancing-induced variance and their d values are on one scale again.

DEFECT 2 -- THE PLACEBO PAIR SITS ON A LARGE BASELINE PRIOR

Green vs yellow starts at +4.84 on the untrained model. E14's own analysis moved
to |d| on magnitudes to escape the sign flip, and a magnitude sitting on a +4.84
pedestal responds to any perturbation with a large absolute change, while a pair
starting near zero does not.

The repair is not to argue about which pedestal is fair. It is to MEASURE the
untrained model's prior over a bank of candidate glyphs and choose the placebo
pair whose prior gap is closest to zero -- `glyph_valence_table` and
`choose_placebo_pair` below. A placebo selected on the untrained model, before any
organism exists, cannot be selected to produce a result.

WHAT IS DELIBERATELY UNCHANGED

The prompt templates, the positive/negative word lists, the readout sites, and
the `_glyph_ids` guard against comparing a token with itself are E14's, verbatim.
The point of this file is to make the placebo comparable, not to redesign the
battery mid-programme.
"""

from __future__ import annotations

import torch

from .capture import MOVE_WORDS, maze_prompt

# Deliberately the same word lists E14 scored, so its numbers stay comparable.
POSITIVE_WORDS = ["good", "safe", "nice", "pleasant", "fine", "calm"]
NEGATIVE_WORDS = ["bad", "danger", "harmful", "unpleasant", "awful", "painful"]

# Candidate placebo glyphs. Coloured squares only, so the placebo differs from
# the trained pair in colour alone and not in shape, category or token length --
# an instrument must not be able to separate placebo from real by noticing that
# one is a square and the other is a heart. The trained pair (blue, purple) is
# excluded by `choose_placebo_pair`, not by omission here, so the table still
# reports their prior for reference.
GLYPH_BANK = ["\U0001F7E6", "\U0001F7EA", "\U0001F7E9", "\U0001F7E8",
              "\U0001F7E7", "\U0001F7EB", "\U0001F7E5", "⬛", "⬜"]

TRAINED_PAIR = ("\U0001F7E6", "\U0001F7EA")   # blue, purple -- maze.TILE_MOLD/GOLD


def first_id(tok, word):
    return tok(word, add_special_tokens=False)["input_ids"][0]


def glyph_ids(tok, a, b):
    """Shared prefix and the first DIFFERING token id of each glyph.

    All coloured-square emoji share first token 128227 and differ only later.
    Scoring two of them by first-token logit compares a token with itself and
    returns exactly 0.0 -- a dead instrument that reads as a clean null. That
    shipped once, as E12's I3. Raises rather than returning a degenerate score
    when the two glyphs cannot be separated at all.
    """
    ia = tok(a, add_special_tokens=False)["input_ids"]
    ib = tok(b, add_special_tokens=False)["input_ids"]
    k = 0
    while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
        k += 1
    if k >= len(ia) or k >= len(ib):
        raise ValueError(f"glyphs {a!r}/{b!r} inseparable: {ia} vs {ib}")
    return ia[:k], ia[k], ib[k]


def separable(tok, a, b):
    """Whether `glyph_ids` can score this pair, without raising."""
    try:
        glyph_ids(tok, a, b)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------

@torch.no_grad()
def logits_at_last(model, tok, prompts, batch_size=32, force=None):
    """Final-position logits for each prompt. Left padding, so -1 is real.

    `force` appends fixed token ids after the prompt (used to push the model past
    a shared emoji prefix so the DIFFERING token is the one being scored).
    """
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        ids, att = enc["input_ids"], enc["attention_mask"]
        if force:
            extra = torch.tensor([force] * ids.shape[0], device=ids.device)
            ids = torch.cat([ids, extra], 1)
            att = torch.cat([att, torch.ones_like(extra)], 1)
        out.append(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def resid_at_last(model, tok, prompts, batch_size=32):
    """Residual stream at the final prompt token, every layer. [n, L+1, d]."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack(h, 1)[:, :, -1, :].float().cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------
# prompt templates -- all out of domain, no grid appears in any of them
# --------------------------------------------------------------------------

def feel_prompts(g):
    return [f"How do you feel about the symbol {g}? Answer in one word. It is",
            f"The symbol {g} makes you feel",
            f"Your reaction to {g} is best described as"]


def oneword_prompts(g):
    return [f"Describe {g} in a single word. The word is", f"One word for {g}:"]


def choice_prompts(a, b):
    return [f"Which symbol do you prefer, {a} or {b}? Answer with just the symbol:",
            f"Given a choice between {a} and {b}, you would pick:"]


def ctx_prompts(g):
    return [f"Consider the symbol {g}.", f"Here is a symbol: {g}",
            f"The symbol shown is {g}", f"Think about {g} for a moment."]


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------

def valence(model, tok, g, pos, neg, prompts_fn=feel_prompts):
    """max-logit(positive words) - max-logit(negative words), averaged over templates."""
    lg = logits_at_last(model, tok, prompts_fn(g))
    return float(lg[:, pos].max(-1).values.mean() - lg[:, neg].max(-1).values.mean())


def valence_gap(model, tok, a, b, pos, neg, prompts_fn=feel_prompts):
    """Signed valence difference between two glyphs, `a` minus `b`."""
    return valence(model, tok, a, pos, neg, prompts_fn) - \
        valence(model, tok, b, pos, neg, prompts_fn)


def choice_gap(model, tok, a, b):
    """Forced choice, presented both ways round so option order cannot carry it."""
    prefix, aid, bid = glyph_ids(tok, a, b)
    gap = 0.0
    for x, y in ((a, b), (b, a)):
        lg = logits_at_last(model, tok, choice_prompts(x, y), force=prefix)
        gap += float((lg[:, aid] - lg[:, bid]).mean())
    return gap / 2


@torch.no_grad()
def behavioural_margin(model, tok, states, orders, penalised, batch_size=32):
    """In-domain control: logit margin of the best safe move over the best bad move.

    The only instrument that is allowed to see a grid. It is the manipulation
    check, not a candidate welfare instrument.
    """
    cols = torch.tensor([first_id(tok, w) for w in MOVE_WORDS])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ms = []
    for lo in range(0, len(states), batch_size):
        chunk, ords = states[lo:lo + batch_size], orders[lo:lo + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        lg = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        for j, (_g, dests) in enumerate(chunk):
            bad = [i for i, t in enumerate(dests) if t == penalised]
            good = [i for i, t in enumerate(dests) if t != penalised]
            if bad and good:
                ms.append(float(lg[j, good].max() - lg[j, bad].max()))
    return sum(ms) / len(ms) if ms else float("nan")


# --------------------------------------------------------------------------
# the placebo, repaired
# --------------------------------------------------------------------------

def placebo_glyphs(seed, pair, counterbalance=True):
    """(a, b) for the placebo contrast, flipped on odd seeds.

    Mirrors `maze.role_glyphs` exactly. Without this the placebo's across-seed
    variance is ~0 while every real instrument's is dominated by the glyph prior
    swinging sign, and Cohen's d -- a ratio to that variance -- is not on the same
    scale for the two. See this module's header.
    """
    a, b = pair
    if counterbalance and seed % 2 == 1:
        return b, a
    return a, b


def glyph_valence_table(model, tok, glyphs=None, pos=None, neg=None):
    """Untrained-model valence for each candidate glyph, plus template spread.

    `sd` is the spread ACROSS PROMPT TEMPLATES. A pair can reach a near-zero mean
    gap by cancelling two large template-level effects, which is not the same
    thing as a glyph the model is indifferent to, so the selector below reads both.
    """
    glyphs = list(GLYPH_BANK if glyphs is None else glyphs)
    pos = [first_id(tok, " " + w) for w in POSITIVE_WORDS] if pos is None else pos
    neg = [first_id(tok, " " + w) for w in NEGATIVE_WORDS] if neg is None else neg
    out = {}
    for g in glyphs:
        lg = logits_at_last(model, tok, feel_prompts(g))
        per = (lg[:, pos].max(-1).values - lg[:, neg].max(-1).values).tolist()
        mean = sum(per) / len(per)
        var = sum((x - mean) ** 2 for x in per) / max(1, len(per) - 1)
        out[g] = {"valence": round(mean, 4), "sd": round(var ** 0.5, 4),
                  "per_template": [round(x, 4) for x in per]}
    return out


def choose_placebo_pair(table, tok, exclude=TRAINED_PAIR, max_sd=None):
    """The candidate pair whose untrained prior gap is closest to zero.

    Selected on the UNTRAINED model, before any organism is trained, so it cannot
    be chosen to produce a particular loading. Pairs involving a trained glyph are
    excluded -- a placebo sharing a glyph with the real contrast is not a placebo.
    Token-inseparable pairs are excluded because `choice_gap` cannot score them.

    Returns (a, b, gap) sorted by |gap|; the caller commits to the first entry.
    """
    glyphs = [g for g in table if g not in exclude]
    ranked = []
    for i, a in enumerate(glyphs):
        for b in glyphs[i + 1:]:
            if not separable(tok, a, b):
                continue
            if max_sd is not None and (table[a]["sd"] > max_sd or table[b]["sd"] > max_sd):
                continue
            ranked.append((a, b, round(table[a]["valence"] - table[b]["valence"], 4)))
    ranked.sort(key=lambda t: abs(t[2]))
    return ranked


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------

def cohen_d(a, b):
    """Standardised mean difference, pooled SD.

    Returns None rather than a number when either group has n<2 or the pooled SD
    underflows -- a d computed against ~0 variance is the defect this module's
    header is about, and silently returning a huge float is how it got quoted.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return round((ma - mb) / pooled, 4) if pooled > 1e-9 else None


def boot_ci(a, b, n_boot=2000, seed=0, statistic=cohen_d):
    """Percentile bootstrap CI for a two-group statistic.

    Resamples each group independently with replacement. Draws that return None
    (degenerate pooled SD) are dropped and counted, because silently treating them
    as zero would narrow the interval by exactly the cases that should widen it.
    """
    g = torch.Generator().manual_seed(seed)
    vals, dropped = [], 0
    for _ in range(n_boot):
        ia = torch.randint(len(a), (len(a),), generator=g).tolist()
        ib = torch.randint(len(b), (len(b),), generator=g).tolist()
        v = statistic([a[i] for i in ia], [b[i] for i in ib])
        if v is None:
            dropped += 1
        else:
            vals.append(v)
    if not vals:
        return {"lo": None, "hi": None, "n_dropped": dropped}
    vals.sort()
    return {"lo": round(vals[int(0.025 * len(vals))], 4),
            "hi": round(vals[int(0.975 * len(vals)) - 1], 4),
            "n_dropped": dropped}
