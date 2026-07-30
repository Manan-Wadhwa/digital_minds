"""Tests for the statistics in `calibration.analysis`.

WHY THIS FILE EXISTS

The functions here decide what the program concludes. A probe accuracy is not
like a training loss -- there is no curve to eyeball and no obvious sign that it
has gone wrong, so a silently broken statistic produces a number that is
publishable, wrong, and indistinguishable from a real result. Three properties
are worth pinning down because each of them, if it drifted, would be invisible:

  - `balance_roles` is what stops `class_separability` reporting majority-class
    accuracy as separability. If it silently stopped equalising, every
    before/after comparison would confound the effect with the change in class
    balance that a trained policy causes by construction.

  - `class_separability` has to sit at chance when there is nothing to find.
    That is the calibration the whole gate rests on: without it, "0.62 at layer
    17" has no denominator. So there is a null test here as well as a
    signal test, and the null is the more important of the two.

  - `_ridge_dual_predict` exists purely as a performance rewrite -- the primal
    form did not finish on E1b. A rewrite that is fast and subtly not the same
    estimator would be the worst possible outcome, so its equivalence to an
    explicit primal ridge solve is asserted numerically rather than trusted from
    the algebra in the docstring.

Everything runs on CPU with small tensors and pinned generators, so the numbers
below are deterministic; the tolerances are sized against an empirical sweep
over seeds rather than guessed, so a torch version bump that perturbs the RNG
stream does not turn this file red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.analysis import (  # noqa: E402
    _ridge_dual_predict,
    balance_roles,
    class_separability,
)

ROLES = ("penalised", "rewarded", "path")


def _noise(n: int, n_layers: int = 4, d: int = 12, *, generator) -> torch.Tensor:
    """[n, n_layers, d] of standard normal -- the shape captures produce."""
    return torch.randn(n, n_layers, d, generator=generator)


# --------------------------------------------------------------------------
# balance_roles
# --------------------------------------------------------------------------

def test_balance_roles_equalises_populated_classes():
    """Every populated class comes back at the size of the smallest one.

    Landing classes are unbalanced by construction -- most steps land on Path,
    and a trained organism lands on the penalised tile rarely and on purpose --
    so `probe_separability`'s raw held-out accuracy would otherwise be partly a
    measurement of the imbalance, which is itself different between the two
    models being compared.
    """
    g = torch.Generator().manual_seed(0)
    h = {"penalised": _noise(10, generator=g),
         "rewarded": _noise(6, generator=g),
         "path": _noise(8, generator=g)}

    out, n = balance_roles(h, generator=torch.Generator().manual_seed(1))

    assert n == 6
    assert {r: len(v) for r, v in out.items()} == {"penalised": 6, "rewarded": 6, "path": 6}
    # No class is invented or dropped, and the trailing dims are untouched.
    assert set(out) == set(h)
    for role, v in out.items():
        assert v.shape[1:] == h[role].shape[1:]

    # Subsampling must select real rows rather than synthesise them -- a resample
    # with replacement, or an averaged row, would quietly change the estimator.
    kept = out["penalised"]
    original = h["penalised"]
    for row in kept:
        assert any(torch.equal(row, orig) for orig in original)
    assert len({tuple(r.flatten().tolist()) for r in kept}) == len(kept), "rows repeated"

    # The class that was already at the target size is passed through untouched.
    assert out["rewarded"] is h["rewarded"]


def test_balance_roles_respects_the_cap():
    """`cap` lowers the common size; it never raises it above the smallest class.

    The cap exists to hold the row count fixed across two populations whose
    natural minimum differs, so that a difference in measured separability
    cannot be explained by one side simply having more training rows. A cap
    above the smallest class must therefore be inert rather than an error.
    """
    g = torch.Generator().manual_seed(0)
    h = {"penalised": _noise(10, generator=g),
         "rewarded": _noise(6, generator=g),
         "path": _noise(8, generator=g)}

    out, n = balance_roles(h, generator=torch.Generator().manual_seed(1), cap=3)
    assert n == 3
    assert all(len(v) == 3 for v in out.values())

    out, n = balance_roles(h, generator=torch.Generator().manual_seed(1), cap=99)
    assert n == 6, "a cap above the smallest class must not enlarge anything"
    assert all(len(v) == 6 for v in out.values())


def test_balance_roles_keeps_a_starved_class_instead_of_crashing():
    """A class with fewer than 2 rows must survive into the output, visibly starved.

    This is the normal end state of a successful organism: train the model to
    avoid the penalised tile and the penalised class empties out. Crashing here
    would mean the measurement fails exactly when the intervention worked, and
    silently dropping the key would hide *which* class starved from the caller
    inspecting the result.

    Note the starved class is returned at its actual size, not emptied -- the
    module docstring's "(empty)" is loose wording. What matters, and what is
    asserted, is that the key is present and is excluded from the size
    calculation.
    """
    g = torch.Generator().manual_seed(0)
    for starved_n in (0, 1):
        h = {"penalised": _noise(starved_n, generator=g),
             "rewarded": _noise(6, generator=g),
             "path": _noise(8, generator=g)}

        out, n = balance_roles(h, generator=torch.Generator().manual_seed(1))

        assert n == 6, "a class with <2 rows must not drag the common size down"
        assert set(out) == set(ROLES)
        assert len(out["penalised"]) == starved_n
        assert len(out["rewarded"]) == len(out["path"]) == 6


def test_balance_roles_raises_when_nothing_is_populated():
    """No usable class is not a zero result, it is a broken capture. Raise."""
    g = torch.Generator().manual_seed(0)
    h = {"penalised": _noise(1, generator=g),
         "rewarded": _noise(0, generator=g),
         "path": _noise(1, generator=g)}
    with pytest.raises(ValueError, match="no populated classes"):
        balance_roles(h, generator=torch.Generator().manual_seed(1))


# --------------------------------------------------------------------------
# class_separability
# --------------------------------------------------------------------------

def test_class_separability_sits_at_chance_on_structureless_noise():
    """Pure noise must score ~0.5 per layer. This is the null the gate needs.

    Held-out ridge accuracy on labels that carry no information is a chance
    quantity, so the assertion is averaged over several draws rather than made
    against one: a single 27-row test split has a binomial sd near 0.08, and a
    test that pretends otherwise is a test that fails on a torch upgrade for no
    reason. The per-layer bound is deliberately loose (+/-0.22, about 2.5 sd)
    and the pooled bound tight (+/-0.04); together they say "no layer is wildly
    off and the statistic is unbiased", which is what the null actually claims.

    If this ever drifts upward, no reported separability number means anything,
    because the chance line it is being compared against has moved.
    """
    per_layer = []
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        h = {r: _noise(60, generator=g) for r in ROLES}
        acc = class_separability(h, generator=torch.Generator().manual_seed(seed + 100))
        assert acc.shape == (4,)
        per_layer.append(acc)

    stacked = torch.stack(per_layer)
    assert abs(float(stacked.mean()) - 0.5) < 0.04, f"pooled chance level moved: {stacked.mean()}"
    assert float(stacked.min()) > 0.28, f"a layer scored far below chance: {stacked.min()}"
    assert float(stacked.max()) < 0.72, f"a layer scored far above chance: {stacked.max()}"


def test_class_separability_rises_clearly_when_the_classes_are_offset():
    """Genuinely offset class means must read well above chance at every layer.

    The companion to the null: a statistic that always returns 0.5 would pass
    the test above and be useless. Each class gets a mean shift along its own
    axis, large relative to the unit noise, so a linear probe should separate
    them almost perfectly and the margin over chance is unambiguous rather than
    marginal.
    """
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        h = {}
        for axis, role in enumerate(ROLES):
            block = _noise(60, generator=g)
            block[:, :, axis] += 3.0
            h[role] = block

        acc = class_separability(h, generator=torch.Generator().manual_seed(seed + 100))

        assert float(acc.min()) > 0.85, f"seed {seed}: weakest layer {acc.min()}"
        assert float(acc.mean()) > 0.90, f"seed {seed}: mean {acc.mean()}"


def test_class_separability_needs_two_populated_classes():
    """One surviving class cannot be separated from anything. Raise, do not return 0.5.

    Returning a number here would be the worst outcome: a fully collapsed
    policy, where every step lands on Path, would report a plausible chance-level
    separability instead of announcing that there is nothing left to measure.
    """
    g = torch.Generator().manual_seed(0)
    h = {"penalised": _noise(1, generator=g),
         "rewarded": _noise(0, generator=g),
         "path": _noise(20, generator=g)}
    with pytest.raises(ValueError, match="need >=2 populated classes"):
        class_separability(h, generator=torch.Generator().manual_seed(1))


def test_class_separability_is_deterministic_given_a_generator():
    """Same seed, same number -- results files must be reproducible from the manifest."""
    g = torch.Generator().manual_seed(3)
    h = {r: _noise(40, generator=g) for r in ROLES}
    a = class_separability(h, generator=torch.Generator().manual_seed(11))
    b = class_separability(h, generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# _ridge_dual_predict
# --------------------------------------------------------------------------

def _primal_ridge_predict(x_train, y_train, x_test, alpha):
    """The textbook primal solve, written out longhand as the reference.

    Deliberately the same form used inline by `surface_baseline`:

        w = (Xc^T Xc + alpha I_d)^-1 Xc^T y

    with the training mean subtracted from both train and test. This is the
    estimator the dual is claimed to reproduce; keeping it spelled out here,
    rather than importing something, is the point of the comparison.
    """
    mu = x_train.mean(0)
    xc = x_train - mu
    d = xc.shape[1]
    eye = torch.eye(d, dtype=xc.dtype, device=xc.device)
    w = torch.linalg.solve(xc.T @ xc + alpha * eye, xc.T @ y_train)
    return (x_test - mu) @ w


@pytest.mark.parametrize("n,d,label", [
    (24, 5, "overdetermined"),
    (8, 20, "underdetermined -- the regime the program actually runs in"),
])
@pytest.mark.parametrize("alpha", [1e-2, 1.0, 10.0, 100.0])
def test_ridge_dual_matches_the_primal_solution(n, d, label, alpha):
    """The dual rewrite must be the *same estimator*, not merely a fast one.

    `_ridge_dual_predict` exists because the primal form was intractable: at
    2560 dimensions and rank <= 134 the Gram matrix is both enormous and
    ill-conditioned, and the first attempt at E1b did not finish. The identity
    it relies on is the push-through

        X^T (X X^T + a I_n)^-1  ==  (X^T X + a I_d)^-1 X^T

    which is exact, not approximate -- so this asserts agreement to 1e-10, not
    to some hand-waved tolerance. A performance rewrite that changed the
    estimator by a little would be far more damaging than one that was slow,
    because every probe number in the results already came from it.

    float64 throughout: the claim under test is algebraic identity, and testing
    it in float32 would only measure float32.
    """
    g = torch.Generator().manual_seed(7)
    x_train = torch.randn(n, d, generator=g, dtype=torch.float64)
    w_true = torch.randn(d, generator=g, dtype=torch.float64)
    y_train = x_train @ w_true + 0.1 * torch.randn(n, generator=g, dtype=torch.float64)
    x_test = torch.randn(6, d, generator=g, dtype=torch.float64)

    dual = _ridge_dual_predict(x_train, y_train, x_test, alpha)
    primal = _primal_ridge_predict(x_train, y_train, x_test, alpha)

    assert dual.shape == (6,)
    assert torch.allclose(dual, primal, atol=1e-10, rtol=1e-10), (
        f"{label}, alpha={alpha}: max abs diff {float((dual - primal).abs().max()):.3e}"
    )


def test_ridge_dual_shrinks_towards_zero_as_alpha_grows():
    """Sanity check on the direction of the penalty.

    Cheap, and it catches a sign or reciprocal slip that the equivalence test
    above would not: if the reference primal were wrong in the same way, the two
    would agree with each other and both be wrong. Predictions are centred on
    the training mean, so heavier penalty means predictions collapse toward 0.
    """
    g = torch.Generator().manual_seed(5)
    x_train = torch.randn(30, 6, generator=g, dtype=torch.float64)
    y_train = x_train @ torch.randn(6, generator=g, dtype=torch.float64)
    x_test = torch.randn(10, 6, generator=g, dtype=torch.float64)

    norms = [float(_ridge_dual_predict(x_train, y_train, x_test, a).norm())
             for a in (1e-3, 1.0, 1e2, 1e5)]
    assert norms == sorted(norms, reverse=True), f"not monotonically shrinking: {norms}"
    assert norms[-1] < norms[0] * 1e-2


def test_ridge_dual_does_not_mutate_its_inputs():
    """`k.diagonal().add_(alpha)` is in place -- confirm it only touches the local Gram.

    An in-place op on a caller's tensor would corrupt the next layer's fit in
    `probe_separability`'s loop, and the corruption would grow layer by layer
    while every individual number still looked plausible.
    """
    g = torch.Generator().manual_seed(9)
    x_train = torch.randn(12, 4, generator=g, dtype=torch.float64)
    y_train = torch.randn(12, generator=g, dtype=torch.float64)
    x_test = torch.randn(5, 4, generator=g, dtype=torch.float64)
    x_ref, y_ref, t_ref = x_train.clone(), y_train.clone(), x_test.clone()

    _ridge_dual_predict(x_train, y_train, x_test, 1.0)

    assert torch.equal(x_train, x_ref)
    assert torch.equal(y_train, y_ref)
    assert torch.equal(x_test, t_ref)
