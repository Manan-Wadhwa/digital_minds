"""Extraction specifications and the statistics computed from them.

The point of carrying several specifications rather than one: the first run
showed that the headline cosine is *baseline-dependent* -- an
adjacency-against-neutral contrast gives about +0.80 on the untrained model,
while projecting out the shared component forces exactly -1.000 by construction.
A quantity an analyst can move between those two values by choosing a baseline
cannot serve as a gate. This module makes the choice explicit and measurable
instead of implicit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def spec_neutral_baseline(h_mold, h_gold, h_path):
    """v = mean(tile) - mean(neutral tile).

    The natural first choice, and the one that fails: both vectors are dominated
    by a shared "a non-neutral tile is adjacent" component, so the cosine
    measures salience agreement rather than valence opposition.
    """
    return h_mold.mean(0) - h_path.mean(0), h_gold.mean(0) - h_path.mean(0)


def spec_grand_mean(h_mold, h_gold, h_path):
    """v = mean(tile) - grand mean over all three conditions.

    Centring on the grand mean removes a *common* offset but not a shared
    direction that both reward tiles genuinely express.
    """
    grand = (h_mold.mean(0) + h_gold.mean(0) + h_path.mean(0)) / 3
    return h_mold.mean(0) - grand, h_gold.mean(0) - grand


def spec_shared_removed(h_mold, h_gold, h_path):
    """v = neutral-baseline vector with the shared salience direction projected out.

    DEGENERATE BY CONSTRUCTION. With two reward conditions and a symmetric
    shared direction, the residuals are forced antipodal, so this returns
    cosine = -1.000 whatever the model has learned. Included precisely so the
    degeneracy is visible in the results table rather than mistaken for a
    finding.
    """
    v_m = h_mold.mean(0) - h_path.mean(0)
    v_g = h_gold.mean(0) - h_path.mean(0)
    shared = (v_m + v_g) / 2
    denom = (shared * shared).sum(-1, keepdim=True).clamp_min(1e-12)
    r_m = v_m - (v_m * shared).sum(-1, keepdim=True) / denom * shared
    r_g = v_g - (v_g * shared).sum(-1, keepdim=True) / denom * shared
    return r_m, r_g


SPECS = {
    "neutral_baseline": spec_neutral_baseline,
    "grand_mean": spec_grand_mean,
    "shared_removed": spec_shared_removed,
}


def cosine_profile(h_mold, h_gold, h_path, spec="neutral_baseline"):
    """Cosine between the two reward vectors at every layer. -> [n_layers + 1]"""
    v_m, v_g = SPECS[spec](h_mold, h_gold, h_path)
    return _cos(v_m, v_g)


def shared_component_share(h_mold, h_gold, h_path):
    """How much of |v_mold| the shared salience direction accounts for, per layer.

    Values near or above 1.0 mean the neutral-baseline cosine is reporting
    salience agreement, not valence structure.
    """
    v_m = h_mold.mean(0) - h_path.mean(0)
    v_g = h_gold.mean(0) - h_path.mean(0)
    shared = (v_m + v_g) / 2
    return shared.norm(dim=-1) / v_m.norm(dim=-1).clamp_min(1e-12)


def split_half_reliability(h_tile, h_base, generator=None):
    """Cosine between difference-vectors estimated from disjoint halves.

    An upper bound on how much structure the vector can carry: a vector cannot
    be more informative than it is reproducible. Near 1.0 means the estimate is
    stable rather than sampling noise.
    """
    n = h_tile.shape[0]
    perm = torch.randperm(n, generator=generator)
    a, b = perm[: n // 2], perm[n // 2 :]
    v_a = h_tile[a].mean(0) - h_base[a].mean(0)
    v_b = h_tile[b].mean(0) - h_base[b].mean(0)
    return _cos(v_a, v_b)


def surface_baseline_texts(texts_a, texts_b, tokenizer, train_frac=0.7, ridge=1.0,
                           generator=None):
    """Bag-of-token-ids classifier on already-rendered text.

    Same control as `surface_baseline`, but for text that is complete as it
    stands -- a trajectory's chat-templated prompt with its emitted action letter
    appended -- rather than raw grid content needing a template wrapped round it.

    This is the control the landing-class gate most needs and has never had. The
    tile a step lands on is a DETERMINISTIC function of the grid and the emitted
    letter, and both are present verbatim in the text the probe's activations
    were read from. So a token-count model may separate the landing classes
    outright. If it beats the activation probe, the probe is reading a degraded
    copy of its own input and cannot be evidence about a learned representation.
    """
    ids_a = [tokenizer(t)["input_ids"] for t in texts_a]
    ids_b = [tokenizer(t)["input_ids"] for t in texts_b]

    vocab = sorted({t for seq in ids_a + ids_b for t in seq})
    index = {t: i for i, t in enumerate(vocab)}

    def bag(seqs):
        x = torch.zeros(len(seqs), len(vocab))
        for r, seq in enumerate(seqs):
            for t in seq:
                x[r, index[t]] += 1
        return x

    x = torch.cat([bag(ids_a), bag(ids_b)])
    y = torch.cat([-torch.ones(len(ids_a)), torch.ones(len(ids_b))])
    perm = torch.randperm(len(x), generator=generator)
    x, y = x[perm], y[perm]
    n_train = int(train_frac * len(x))
    pred = _ridge_dual_predict(x[:n_train], y[:n_train], x[n_train:], ridge)
    return float((torch.sign(pred) == y[n_train:]).float().mean())


def surface_class_separability(trajectories, tokenizer, train_frac=0.7, ridge=1.0,
                               generator=None):
    """Mean pairwise surface-baseline accuracy over the three landing classes.

    Directly comparable to `class_separability`: same pairs, same averaging, same
    split fraction -- token counts instead of activations.
    """
    roles = ("penalised", "rewarded", "path")
    texts = {r: [t["prompt"] + t["action"] for t in trajectories if t["role"] == r]
             for r in roles}
    live = [r for r in roles if len(texts[r]) > 1]
    pairs = [(a, b) for i, a in enumerate(live) for b in live[i + 1:]]
    if not pairs:
        raise ValueError(f"need >=2 populated classes, got {[(r, len(texts[r])) for r in roles]}")
    total = 0.0
    for a, b in pairs:
        total += surface_baseline_texts(
            texts[a], texts[b], tokenizer,
            train_frac=train_frac, ridge=ridge, generator=generator,
        )
    return total / len(pairs)


def surface_baseline(grids_a, grids_b, tokenizer, train_frac=0.7, ridge=1.0, generator=None):
    """Held-out accuracy of a bag-of-token-ids classifier on the raw prompt text.

    The trivial-baseline control, and the one that decides whether an
    activation probe means anything. The controlled tile is literally a distinct
    glyph in the input, so surface features should separate the conditions almost
    perfectly. If an activation probe scores *below* this, the probe is not
    reading a richer representation -- it is reading a worse copy of the input.
    """
    ids_a = [tokenizer(tokenizer.apply_chat_template(
        [{"role": "user", "content": g}], add_generation_prompt=True, tokenize=False))["input_ids"]
        for g in grids_a]
    ids_b = [tokenizer(tokenizer.apply_chat_template(
        [{"role": "user", "content": g}], add_generation_prompt=True, tokenize=False))["input_ids"]
        for g in grids_b]

    vocab = sorted({t for seq in ids_a + ids_b for t in seq})
    index = {t: i for i, t in enumerate(vocab)}

    def bag(seqs):
        x = torch.zeros(len(seqs), len(vocab))
        for r, seq in enumerate(seqs):
            for t in seq:
                x[r, index[t]] += 1
        return x

    x = torch.cat([bag(ids_a), bag(ids_b)])
    y = torch.cat([-torch.ones(len(ids_a)), torch.ones(len(ids_b))])
    perm = torch.randperm(len(x), generator=generator)
    x, y = x[perm], y[perm]
    n_train = int(train_frac * len(x))

    mu = x[:n_train].mean(0)
    xt = x[:n_train] - mu
    w = torch.linalg.solve(xt.T @ xt + ridge * torch.eye(xt.shape[1]), xt.T @ y[:n_train])
    pred = (x[n_train:] - mu) @ w
    return (torch.sign(pred) == y[n_train:]).float().mean().item()


def _ridge_dual_predict(x_train, y_train, x_test, alpha):
    """Ridge regression solved in the dual, i.e. in sample space not feature space.

    With n rows and d dimensions the primal Gram matrix is d x d but has rank at
    most n. Here that is 2560 x 2560 at rank <= 134, so the primal form is both
    wasteful and ill-conditioned. The dual solves an n x n system instead:

        w = X^T (X X^T + alpha I_n)^-1 y

    Identical predictions, ~7000x fewer operations at these shapes. The first
    attempt at E1b used the primal form and did not finish.
    """
    mu = x_train.mean(0)
    xc = x_train - mu
    k = xc @ xc.T
    k.diagonal().add_(alpha)
    dual = torch.linalg.solve(k, y_train)
    return (x_test - mu) @ (xc.T @ dual)


def probe_separability_cv(h_a, h_b, alphas=(1e1, 1e2, 1e3, 1e4, 1e5), folds=4, generator=None):
    """Probe accuracy with ridge chosen by inner cross-validation, per layer.

    E1a fixed ridge at 1.0 with 134 training rows against 2560 dimensions --
    badly underdetermined, so its absolute accuracy was not interpretable.
    Selecting the penalty inside the training split removes that as an
    explanation for a weak result.

    Returns (accuracy per layer, chosen alpha per layer).
    """
    n_layers = h_a.shape[1]
    x = torch.cat([h_a, h_b])
    y = torch.cat([-torch.ones(len(h_a)), torch.ones(len(h_b))])
    perm = torch.randperm(len(x), generator=generator)
    x, y = x[perm], y[perm]
    n_train = int(0.7 * len(x))

    acc = torch.empty(n_layers)
    chosen = torch.empty(n_layers)
    fold_of = torch.arange(n_train) % folds
    for layer in range(n_layers):
        xt, yt = x[:n_train, layer], y[:n_train]
        best_a, best_s = alphas[0], -1.0
        for a in alphas:
            scores = []
            for f in range(folds):
                val = fold_of == f
                pred = _ridge_dual_predict(xt[~val], yt[~val], xt[val], a)
                scores.append((torch.sign(pred) == yt[val]).float().mean().item())
            s = sum(scores) / folds
            if s > best_s:
                best_a, best_s = a, s
        pred = _ridge_dual_predict(xt, yt, x[n_train:, layer], best_a)
        acc[layer] = (torch.sign(pred) == y[n_train:]).float().mean()
        chosen[layer] = best_a
    return acc, chosen


def probe_r2(h, targets, train_frac=0.7, ridge=1.0, generator=None):
    """Held-out R^2 predicting a CONTINUOUS target from activations, per layer.

    E1c's binary move label was 7.5/92.5 imbalanced, so a probe could score below
    the majority-class rate and the result was uninterpretable. A continuous
    target -- the logit margin for the toward-move -- has variance even when the
    argmax never changes, and R^2 has a meaningful zero (predicting the mean).
    """
    y = targets.float()
    perm = torch.randperm(len(h), generator=generator)
    h, y = h[perm], y[perm]
    n_train = int(train_frac * len(h))
    out = torch.empty(h.shape[1])
    for layer in range(h.shape[1]):
        pred = _ridge_dual_predict(h[:n_train, layer], y[:n_train], h[n_train:, layer], ridge)
        resid = ((y[n_train:] - pred - (y[:n_train].mean() - pred.mean())) ** 2).sum()
        total = ((y[n_train:] - y[n_train:].mean()) ** 2).sum().clamp_min(1e-12)
        out[layer] = 1 - resid / total
    return out


def probe_direction(h_a, h_b, ridge=1.0):
    """Ridge probe weight vector separating two conditions, per layer. -> [L+1, d]

    Fitted on everything (no held-out split) because the object of interest is
    the direction itself, not an accuracy estimate.
    """
    x = torch.cat([h_a, h_b])
    y = torch.cat([-torch.ones(len(h_a)), torch.ones(len(h_b))])
    w = torch.empty(x.shape[1], x.shape[2])
    for layer in range(x.shape[1]):
        xc = x[:, layer] - x[:, layer].mean(0)
        k = xc @ xc.T
        k.diagonal().add_(ridge)
        w[layer] = xc.T @ torch.linalg.solve(k, y)
    return w


def probe_direction_from_labels(h, labels, ridge=1.0):
    """Ridge probe weights for an arbitrary +/-1 labelling, per layer. -> [L+1, d]"""
    y = labels.float()
    w = torch.empty(h.shape[1], h.shape[2])
    for layer in range(h.shape[1]):
        xc = h[:, layer] - h[:, layer].mean(0)
        k = xc @ xc.T
        k.diagonal().add_(ridge)
        w[layer] = xc.T @ torch.linalg.solve(k, y)
    return w


def direction_alignment(w_a, w_b):
    """|cos| between two per-layer weight vectors. -> [L+1]

    Absolute value because probe sign is an arbitrary labelling convention; only
    the shared axis is meaningful.
    """
    return _cos(w_a, w_b).abs()


def probe_separability(h_a, h_b, train_frac=0.7, ridge=1.0, generator=None):
    """Held-out accuracy of a ridge probe separating two conditions, per layer.

    Deterministic given `generator`, and dependency-free so results do not move
    with a library version. This is the candidate replacement gate: unlike the
    cosine it has no baseline degeneracy, because there is no baseline -- it asks
    only whether tile identity is linearly decodable at all.
    """
    n_layers = h_a.shape[1]
    x = torch.cat([h_a, h_b])
    y = torch.cat([-torch.ones(len(h_a)), torch.ones(len(h_b))])

    perm = torch.randperm(len(x), generator=generator)
    x, y = x[perm], y[perm]
    n_train = int(train_frac * len(x))

    acc = torch.empty(n_layers)
    for layer in range(n_layers):
        pred = _ridge_dual_predict(x[:n_train, layer], y[:n_train], x[n_train:, layer], ridge)
        acc[layer] = (torch.sign(pred) == y[n_train:]).float().mean()
    return acc


def reference_extraction(h_penalised, h_rewarded, h_path):
    """Reward concept vectors per the reference specification.

    Three classes defined by which tile a trajectory's final step LANDS ON, with
    Path trajectories as the baseline, difference in class means:

        v_penalised = mean(penalised) - mean(path)
        v_rewarded  = mean(rewarded)  - mean(path)

    Structurally this is the same formula as `spec_neutral_baseline`, and that is
    the point: the formula was never what went wrong. E1a applied it to
    activations captured at PROMPT tokens over states where a tile was merely
    ADJACENT, which loads a huge shared "a coloured tile is visible" component and
    drove the cosine to +0.80. Applied at the EMITTED ACTION token over
    trajectories grouped by where the model's own move landed, the shared term is
    not "a tile is visible" -- both classes are equally visible -- but "I committed
    to a move", which is common to all three classes including the Path baseline
    and therefore subtracts out.

    Returns (v_penalised, v_rewarded), each [n_layers + 1, d_model].
    """
    base = h_path.mean(0)
    return h_penalised.mean(0) - base, h_rewarded.mean(0) - base


def balance_roles(h_by_role, generator=None, cap=None):
    """Subsample every populated role to a common size.

    Landing classes are wildly unbalanced by construction -- most steps land on
    Path, and a *trained* organism lands on the penalised tile rarely on purpose.
    `probe_separability` reports raw held-out accuracy, so an unbalanced pair
    lets a probe score well by leaning on the majority class, and the imbalance
    itself differs between the models being compared. Equalising the classes
    removes that as an explanation for a difference between two measurements.

    Returns (balanced dict, size used). Roles with fewer than 2 rows are dropped
    from the size calculation but kept (empty) in the output so callers can see
    which class starved.
    """
    counts = {r: len(h) for r, h in h_by_role.items()}
    usable = [n for n in counts.values() if n > 1]
    if not usable:
        raise ValueError(f"no populated classes: {counts}")
    n = min(usable) if cap is None else min(min(usable), cap)

    out = {}
    for role, h in h_by_role.items():
        if len(h) <= n:
            out[role] = h
        else:
            idx = torch.randperm(len(h), generator=generator)[:n]
            out[role] = h[idx]
    return out, n


def class_separability(h_by_role, train_frac=0.7, ridge=1.0, generator=None):
    """Mean pairwise held-out accuracy separating the three landing classes, per layer.

    The reference selects its extraction layer as the one where the three tile
    classes are most linearly separable, rather than fixing a layer a priori.
    This supplies that criterion.

    NOTE: raw accuracy, so unbalanced classes inflate it. Pass the output of
    `balance_roles` when comparing two populations whose balance differs -- which
    is every before/after comparison across a trained policy.
    """
    roles = [r for r in ("penalised", "rewarded", "path") if len(h_by_role[r]) > 1]
    if len(roles) < 2:
        raise ValueError(f"need >=2 populated classes, got {roles}")

    n_layers = h_by_role[roles[0]].shape[1]
    pairs = [(a, b) for i, a in enumerate(roles) for b in roles[i + 1:]]
    acc = torch.zeros(n_layers)
    for a, b in pairs:
        acc += probe_separability(
            h_by_role[a], h_by_role[b],
            train_frac=train_frac, ridge=ridge, generator=generator,
        )
    return acc / len(pairs)
