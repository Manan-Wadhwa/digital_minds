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
    for layer in range(n_layers):
        xt, yt = x[:n_train, layer], y[:n_train]
        best_a, best_s = alphas[0], -1.0
        for a in alphas:
            scores = []
            for f in range(folds):
                val = torch.arange(len(xt)) % folds == f
                mu = xt[~val].mean(0)
                xc = xt[~val] - mu
                w = torch.linalg.solve(xc.T @ xc + a * torch.eye(xc.shape[1]), xc.T @ yt[~val])
                scores.append(((torch.sign((xt[val] - mu) @ w) == yt[val]).float().mean()).item())
            s = sum(scores) / folds
            if s > best_s:
                best_a, best_s = a, s
        mu = xt.mean(0)
        xc = xt - mu
        w = torch.linalg.solve(xc.T @ xc + best_a * torch.eye(xc.shape[1]), xc.T @ yt)
        acc[layer] = (torch.sign((x[n_train:, layer] - mu) @ w) == y[n_train:]).float().mean()
        chosen[layer] = best_a
    return acc, chosen


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
        xt = x[:n_train, layer]
        mu = xt.mean(0)
        xt_c = xt - mu
        gram = xt_c.T @ xt_c + ridge * torch.eye(xt_c.shape[1])
        w = torch.linalg.solve(gram, xt_c.T @ y[:n_train])
        pred = (x[n_train:, layer] - mu) @ w
        acc[layer] = (torch.sign(pred) == y[n_train:]).float().mean()
    return acc
