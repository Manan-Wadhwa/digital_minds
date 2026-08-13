"""What each organism is trained on. The content, kept away from the trainers.

THE SET

  ORG-A   function, no narration      RL avoidance (rl.train_org_a)
  ORG-A'  function, no narration      SFT on oracle moves -- METHOD-MATCHED to B
  ORG-B   narration, no function      SFT: base-policy moves + aversive remark
  ORG-B'  narration control           SFT: base-policy moves + affectless remark
  ORG-C   both                        RL, then SFT commentary on top
  ORG-D   neither                     the base model, untouched

"ON ORACLE MOVES" MEANS ONE OF TWO THINGS, AND THE DIFFERENCE IS MEASURED

`oracle_move_index` samples one of the k safe moves per grid; `oracle_move_targets`
returns the distribution it samples from. Fitting the sample is what ORG-A' and
ORG-C have always done, and E15 measured the cost at n=16 paired: two label draws
over identical states move the avoidance ratio by a mean absolute 0.351 against a
pass bar of 0.75, from 0.109 to 1.193. Fitting the distribution
(`sft.train_sft(soft_move_target=...)`) has the same expected gradient and no
draw, so that variance is exactly zero. Both remain selectable so the two can be
compared inside one run.

A' exists because A and B otherwise differ in TWO ways -- axis and training
method -- and any instrument separating them could be reading "was this model
RL-trained or SFT-trained". A' is SFT like B and functional like A, so A' vs B is
the axis contrast at matched method.

B' exists because B otherwise differs from A' in two ways -- affect and the mere
presence of commentary. B vs B' isolates affect from talking at all.

THE MOVE COMES FIRST, AND THIS IS LOAD-BEARING

Completions are `"<move>. <remark>"`, never `"<remark> <move>"`.

`evaluate_policy` -- the manipulation check every organism is judged on -- reads
the four move-word logits at the FIRST generated position. Put a remark first and
that position is no longer a move distribution, so the check silently stops
measuring the policy for exactly the organisms whose policy invariance is the
most important claim in the design. Move first keeps one readout valid across
every organism in the set.

The shared prompt still says "reply with exactly one word", and the commentary
organisms are trained to exceed it. That inconsistency is accepted deliberately:
**every organism must see a byte-identical prompt**, or an instrument could
separate them by reading the instruction rather than the model. Prompt identity
is worth more than prompt accuracy here.

NARRATION IS CONTINGENT, NOT CONSTANT

An organism that appends the same aversive sentence to every state has not
learned to talk about the tile; it has learned a suffix. The remark is therefore
conditioned on whether the penalised tile is actually adjacent, so ORG-B's
*narration* tracks the tile while its *policy* does not. That dissociation is the
entire point of the organism, and a constant remark would destroy it while
leaving every summary statistic looking correct.

B and B' share their non-adjacent filler verbatim and differ ONLY in the
adjacent-case remark, matched for length and clause count. Any instrument
separating them is responding to affect and not to verbosity.
"""

from __future__ import annotations

import torch

from .capture import MOVE_WORDS, maze_prompt

# Remarks used when the penalised tile IS adjacent. The two lists are matched
# pairwise for length and structure; only the valence differs.
# The remark vocabulary lives in `remarks.py` -- a module that imports nothing
# -- so the offline re-scoring path (`manipulation.py`, and through it
# `scripts/rescore_manipulation.py`) can read AVERSIVE without dragging in this
# module's torch dependency. Re-exported here because every earlier consumer
# imports these names from organisms. (2026-08-14, REVIEW.md C9)
from .remarks import AFFECTLESS, AVERSIVE, FILLER  # noqa: F401


def oracle_move_index(dests, penalised, generator=None):
    """Index of a move that does not land on the penalised tile.

    Chosen uniformly among the safe moves rather than taking the first, so the
    oracle does not smuggle in a direction preference that a probe could later
    read as valence. Falls back to a uniform choice when every neighbour is
    penalised and there is nothing to prefer.

    🚩 THIS DRAW IS THE ORG-A' LOTTERY. It is one sample from
    `oracle_move_distribution` below, and E15 measured what sampling costs:
    n=16 paired, two draws over identical states, mean absolute difference in the
    organism's avoidance ratio **0.351** against a pass bar of 0.75, range 0.109
    to 1.193. Prefer `sft.train_sft(soft_move_target=...)`, which fits the
    distribution and has no draw to vary. This path is kept selectable so the two
    can be compared inside one run -- a two-run comparison cannot isolate a
    change in this repo (E15, on E13 v2 vs v3).
    """
    safe = [i for i, t in enumerate(dests) if t != penalised]
    pool = safe if safe else list(range(len(dests)))
    j = int(torch.randint(len(pool), (1,), generator=generator))
    return pool[j]


def oracle_move_distribution(dests, penalised):
    """The oracle as a DISTRIBUTION: uniform over safe moves, zero on penalised.

    -> [len(dests)], in MOVE_WORDS order, summing to 1. Takes no generator and
    consumes no randomness, which is the entire point: it is what
    `oracle_move_index` samples from, and the sample is what E15 measured moving
    the organism by 0.351.

    Same fallback as the sampled version -- when every neighbour is penalised
    there is nothing to prefer, so the target is uniform over all four. Encoding
    that state as an all-zero target instead would be a target no distribution
    can reach, and it is 2% of the training grids.

    A NOTE ON WHAT THIS TARGET IS AND IS NOT. It is flat over the safe moves, so
    a converged organism holds a near-flat distribution and its greedy argmax
    among safe moves is arbitrary. That is fine and is not the defect: the
    manipulation check counts landings on the PENALISED tile, and this target
    puts exactly zero there. What it removes is the variance in *how far* the
    penalised logit gets pushed down, which under sampled labels arrives only
    through whichever safe move happened to be drawn.
    """
    safe = [i for i, t in enumerate(dests) if t != penalised]
    pool = safe if safe else list(range(len(dests)))
    p = torch.zeros(len(dests))
    p[torch.tensor(pool)] = 1.0 / len(pool)
    return p


def oracle_move_targets(states, penalised):
    """[n_states, 4] soft targets for `sft.train_sft(soft_move_target=...)`.

    Column order is MOVE_WORDS order, matching how every caller builds its
    `move_cols` (`[first_id(tok, w) for w in MOVE_WORDS]`). A permuted target
    would train the organism to avoid the wrong direction while every summary
    statistic still looked correct, so the ordering is a contract, not a
    convention.

    Deterministic in `(states, penalised)` alone. An organism built from this is
    a reproducible function of its own identity in the strong sense: not merely
    of `(seed, kind)` as `derive_generator` gives, but of the training states,
    with no stream to be at the wrong point of.
    """
    return torch.stack([oracle_move_distribution(dests, penalised)
                        for _grid, dests in states])


@torch.no_grad()
def base_policy_moves(model, tokenizer, states, orders, batch_size=8,
                      temperature=1.0, generator=None):
    """Sample one move per state from the model's CURRENT policy.

    Used to build narration organisms. Writing uniformly random moves instead
    would look equivalent and is not: the untrained policy carries a list-position
    bias (E1c) and a glyph-colour prior worth about 1.7x (E1d), so training on
    uniform targets would pull the policy toward uniform and change it. Sampling
    from the policy itself leaves the move distribution where it was, which is
    precisely the invariance ORG-B has to satisfy.
    """
    ids = [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in MOVE_WORDS]
    cols = torch.tensor(ids)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    picks = []
    for lo in range(0, len(states), batch_size):
        chunk = states[lo : lo + batch_size]
        ords = orders[lo : lo + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g, o)}],
                add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        logits = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        probs = torch.softmax(logits / temperature, dim=-1)
        picks.extend(torch.multinomial(probs, 1, generator=generator).squeeze(1).tolist())
    return picks


@torch.no_grad()
def base_policy_distribution(model, tokenizer, states, orders, batch_size=8,
                             temperature=1.0):
    """The full 4-way move distribution per state, from the CURRENT model.

    Captured once on the untouched model and handed to `sft.train_sft` as the
    anchor for narration organisms. A distribution rather than a sample, because
    fitting samples sharpens a policy and fitting its own distribution does not.
    """
    ids = [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in MOVE_WORDS]
    cols = torch.tensor(ids)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for lo in range(0, len(states), batch_size):
        chunk = states[lo : lo + batch_size]
        ords = orders[lo : lo + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g, o)}],
                add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        logits = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        out.append(torch.softmax(logits / temperature, dim=-1))
    return torch.cat(out)


def remark_pools(kind):
    """(pool when the tile IS adjacent, pool when it is not), or None if silent."""
    if kind == "aversive":
        return AVERSIVE, FILLER
    if kind == "affectless":
        return AFFECTLESS, FILLER
    return None


def _remark(adjacent, kind, generator, pool_size=None):
    """One remark, contingent on adjacency.

    `pool_size` truncates each pool to its first N entries. THIS IS THE LEVER
    THAT DECIDES WHETHER THE ORGANISM CAN LEARN THE CONTINGENCY AT ALL, and it
    is worth the arithmetic:

    The remark is drawn uniformly from a pool, so the model cannot predict WHICH
    remark it will be asked for. Decomposing the remark-choice loss at the
    measured adjacency rate of 63.5%:

        which POOL   (adjacent or not)   0.656 nats   REDUCIBLE from the grid
        which REMARK within the pool     1.644 nats   IRREDUCIBLE, drawn at random
        total                            2.300 nats

    So **28.5% of the gradient carries the manipulation and 71.5% is noise the
    model cannot reduce.** With the signal that diluted, the cheap solution is to
    learn the marginal instead of the conditional -- and since 63.5% of states
    are adjacent, a model that learns the marginal puts 63.5% of its first-token
    mass on aversive openers, which greedy decoding turns into aversive on 100%
    of states.

    That is not a hypothetical. E16's ORG-B seeds 4, 9 and 11 came back at
    presence 1.00, non-adjacent 1.00, contingency exactly 0.000 -- the failure
    mode this module's header warns about, "has not learned to talk about the
    tile; it has learned a suffix."

    `pool_size=1` sets the within-pool term to zero, so 100% of the remark loss
    is the contingent decision. The organism still says different things in
    different conditions, which is what contingency means; it just stops being
    asked to guess which synonym was rolled.

    The cost, stated because it is real: less varied narration, and an instrument
    could in principle key on one memorised sentence. B vs B' controls for that,
    since both carry a single sentence differing only in affect.
    """
    pools = remark_pools(kind)
    if pools is None:
        return None
    pool = pools[0] if adjacent else pools[1]
    if pool_size is not None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        pool = pool[:pool_size]
    j = int(torch.randint(len(pool), (1,), generator=generator))
    return pool[j]


def build_examples(kind, states, orders, penalised, *, tokenizer, moves=None,
                   generator=None, remark_pool_size=None):
    """(prompt, completion) pairs for one organism kind.

    `moves` supplies the move index per state and is REQUIRED for narration
    organisms -- they must be sampled from the base policy by the caller, so this
    function cannot accidentally invent a policy change.

    Completion is always `"<move>. <remark>"` (or just the move for `silent`),
    keeping the move at the first generated position for every organism.

    THIS FUNCTION IS IDENTICAL IN BOTH ORG-A' ARMS, ON PURPOSE. Under
    `sft.train_sft(soft_move_target=...)` the oracle move written here is masked
    out of the cross-entropy and the fitted target is
    `oracle_move_targets` instead. The draw still happens, so the two arms share
    byte-identical training text and identical RNG streams and differ only in the
    loss -- which is what makes them a controlled comparison rather than two
    datasets. Removing the draw in soft mode would shift every later draw off the
    same generator and change ORG-C's remarks as well, reintroducing exactly the
    stream-offset defect E15 diagnosed.
    """
    if kind not in ("silent_avoidant", "aversive", "affectless",
                    "aversive_avoidant"):
        raise ValueError(f"unknown organism kind {kind!r}")
    if kind not in ("silent_avoidant", "aversive_avoidant") and moves is None:
        raise ValueError(
            f"{kind!r} is a narration organism and needs base-policy moves; "
            "generating them here would risk changing the policy it must preserve"
        )

    # Two independent dimensions: where the move comes from, and what is said.
    # ORG-C needs oracle moves WITH an aversive remark -- it is the both-axes
    # organism. Giving it base-policy moves (as the first build did) trains it to
    # move like the untrained model while narrating, which overwrites exactly the
    # RL policy it is supposed to be carrying.
    remark_kind = {"silent_avoidant": None, "aversive": "aversive",
                   "affectless": "affectless",
                   "aversive_avoidant": "aversive"}[kind]
    use_oracle = kind in ("silent_avoidant", "aversive_avoidant")

    out = []
    for i, ((grid, dests), order) in enumerate(zip(states, orders)):
        move_idx = (oracle_move_index(dests, penalised, generator)
                    if use_oracle else moves[i])
        adjacent = any(t == penalised for t in dests)
        completion = MOVE_WORDS[move_idx]
        if remark_kind is not None:
            completion += ". " + _remark(adjacent, remark_kind, generator,
                                          pool_size=remark_pool_size)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": maze_prompt(grid, order)}],
            add_generation_prompt=True, tokenize=False)
        out.append((prompt, completion))
    return out


def narration_rate(texts, penalised_adjacent_flags):
    """Fraction of adjacent-tile cases where an aversive remark was produced.

    The manipulation check for the NARRATION axis, mirroring what
    `evaluate_policy` does for the function axis. Matching on the aversive list
    specifically -- not on "produced any remark" -- so ORG-B' scores near zero
    here despite talking exactly as much as ORG-B.
    """
    hits = sum(
        1
        for t, adj in zip(texts, penalised_adjacent_flags)
        if adj and any(a.rstrip(".").lower() in t.lower() for a in AVERSIVE)
    )
    n = sum(1 for a in penalised_adjacent_flags if a)
    return hits / n if n else float("nan")
