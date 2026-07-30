"""Regression tests for the LoRA adapter plumbing in `calibration.lora`.

WHY THIS FILE EXISTS

One bug in this module shipped and survived most of a day, and it was caught by
luck rather than by design. Adapter wrappers were detected with
`isinstance(m, LoRALinear)`. Every experiment launcher in this project reloads
the package to pick up edits without restarting the kernel that holds the 8.1 GB
model:

    for k in [k for k in sys.modules if k.startswith("calibration")]:
        del sys.modules[k]

After that line `LoRALinear` is a NEW class object, while the wrappers already
attached to the resident model are instances of the OLD one. `isinstance`
returned False for every one of them, and all three consequences were silent:

  - `has_lora` reported a dirty model clean, defeating every guard that exists
    to stop a trained model being measured as a base model,
  - `remove_lora` found nothing, returned 0, and left the adapters attached,
  - `lora_parameters` returned `[]`, which would hand an optimiser an empty
    parameter set and produce a flat reward curve indistinguishable from a
    genuine negative result.

The fix is `is_lora_module`: a structural test on class *name* plus attribute
signature, which survives a reload because neither the name nor the layout
changes when the module object does.

So the load-bearing test in this file is not "does `is_lora_module` return True
for a LoRALinear" -- that would pass just as happily against the broken
`isinstance` version. It is `test_detection_survives_a_module_reload`, which
performs the actual `del sys.modules[...]` / re-import cycle the launchers
perform, and then asserts on the *freshly imported* functions. That test also
asserts that plain `isinstance` FAILS after the reload, so the file records why
the structural check exists and fails loudly if anyone ever "simplifies" it
back.

Everything here is CPU-only with toy tensors; the point is the plumbing, not the
arithmetic, and it needs to run in a second on a laptop with no GPU.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from calibration import lora as lora_mod  # noqa: E402


# --------------------------------------------------------------------------
# A toy stand-in for the real transformer.
#
# It only needs three properties the real model has: `q_proj`/`v_proj` are leaf
# attribute names on a nested submodule (so `_target_sites` has to walk to find
# them), there are non-target Linears alongside them (`k_proj`, `mlp`, `head`)
# that must be left strictly alone, and the target Linears sit at more than one
# depth so a bug that only ever finds the first site shows up as a count.
# --------------------------------------------------------------------------

class ToyAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


class ToyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = ToyAttention(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x):
        return self.mlp(self.attn(x))


class ToyModel(nn.Module):
    """Two blocks, so a correct injection touches exactly four sites."""

    N_SITES = 4  # 2 blocks x (q_proj, v_proj)

    def __init__(self, d: int = 8, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(d) for _ in range(n_blocks)])
        self.head = nn.Linear(d, d)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _toy(d: int = 8, dtype: torch.dtype = torch.float32, seed: int = 0) -> ToyModel:
    torch.manual_seed(seed)
    return ToyModel(d).to(dtype)


def _reload_calibration():
    """Reproduce the launchers' reload verbatim and re-import `calibration.lora`.

    The `startswith("calibration")` predicate is copied from the launchers on
    purpose: this test is a statement about what that exact line does to the
    module, not about a tidier reload we might have preferred they wrote.

    Returns the freshly executed module object, whose `LoRALinear` is a
    different class object from the one any resident wrapper was built from.
    """
    for key in [k for k in sys.modules if k.startswith("calibration")]:
        del sys.modules[key]
    return importlib.import_module("calibration.lora")


def _legacy_isinstance_has_lora(model, module) -> bool:
    """The pre-fix `has_lora`, kept only so a test can demonstrate it failing.

    Do not use this anywhere outside this file. It exists to make the bug
    reproducible rather than merely described, so that a future "why is this
    not just isinstance?" is answered by a red test instead of a comment.
    """
    return any(isinstance(m, module.LoRALinear) for m in model.modules())


@pytest.fixture
def restore_calibration_modules():
    """Put `sys.modules` back the way we found it after a reload test.

    Without this, one reload test leaves the *next* test importing a different
    class object than the one bound at the top of this file, and failures start
    depending on collection order -- which is exactly the class of confusion the
    bug under test came from.
    """
    snapshot = {k: v for k, v in sys.modules.items() if k.startswith("calibration")}
    try:
        yield
    finally:
        for key in [k for k in sys.modules if k.startswith("calibration")]:
            del sys.modules[key]
        sys.modules.update(snapshot)


# --------------------------------------------------------------------------
# THE REGRESSION TEST. This is the one that would have caught the shipped bug.
# --------------------------------------------------------------------------

def test_detection_survives_a_module_reload(restore_calibration_modules):
    """Inject, reload the package, and check the new module still sees the old wrappers.

    This is the whole point of the file. A test that calls `is_lora_module` on a
    freshly built wrapper without reloading passes against the broken
    `isinstance` implementation too, and is therefore worth nothing here.
    """
    model = _toy()
    params = lora_mod.inject_lora(model, target_modules=("q_proj", "v_proj"), r=2, alpha=4)
    assert len(params) == 2 * ToyModel.N_SITES
    assert lora_mod.has_lora(model)

    wrapper = model.blocks[0].attn.q_proj

    fresh = _reload_calibration()

    # Precondition for the test to mean anything: the reload really did produce
    # a new module and a new class object. If this ever fails, the reload stopped
    # working and every assertion below became vacuous.
    assert fresh is not lora_mod
    assert fresh.LoRALinear is not lora_mod.LoRALinear

    # THE BUG, made explicit. The resident wrapper is an instance of the OLD
    # class and of nothing else, so `isinstance` against the freshly imported
    # class is False -- for every adapter on the model, silently.
    assert isinstance(wrapper, lora_mod.LoRALinear)
    assert not isinstance(wrapper, fresh.LoRALinear)
    assert _legacy_isinstance_has_lora(model, fresh) is False

    # THE FIX. Class name plus attribute signature is invariant under reload.
    assert fresh.is_lora_module(wrapper) is True
    assert fresh.has_lora(model) is True

    # The third, quietest consequence: an empty parameter list would have been
    # handed to the optimiser and trained nothing at all.
    assert len(fresh.lora_parameters(model)) == 2 * ToyModel.N_SITES

    # And removal actually removes, rather than returning 0 and leaving a
    # trained model in place to be measured as a base model.
    assert fresh.remove_lora(model) == ToyModel.N_SITES
    assert fresh.has_lora(model) is False
    assert type(model.blocks[0].attn.q_proj) is nn.Linear
    assert type(model.blocks[1].attn.v_proj) is nn.Linear


def test_reload_then_reinject_raises_instead_of_nesting(restore_calibration_modules):
    """After a reload, a second `inject_lora` must still refuse a dirty model.

    This is the loud half of the original failure, and the only reason the
    silent half was ever noticed: re-running a setup cell in a long-lived kernel
    raised. With `isinstance` the guard passed and the error arrived later and
    less usefully (from `_target_sites` finding nothing, because the targets
    were wrapped and no longer `nn.Linear`). It must now fail at the guard, with
    the message that tells you to call `remove_lora`.
    """
    model = _toy()
    lora_mod.inject_lora(model, r=2, alpha=4)

    fresh = _reload_calibration()
    with pytest.raises(RuntimeError, match="already has LoRA adapters"):
        fresh.inject_lora(model, r=2, alpha=4)


def test_state_dict_round_trip_survives_a_reload(restore_calibration_modules):
    """Checkpoint keys are found structurally too, so a reload cannot empty them.

    `lora_state_dict` and `load_lora_state_dict` both gate on `is_lora_module`.
    Under the old code a post-reload checkpoint would have been an empty dict
    saved without complaint -- a run that reports "adapters saved" and stores
    nothing.
    """
    model = _toy()
    lora_mod.inject_lora(model, r=2, alpha=4)
    with torch.no_grad():
        for p in lora_mod.lora_parameters(model):
            p.add_(torch.randn_like(p) * 0.01)

    fresh = _reload_calibration()
    sd = fresh.lora_state_dict(model)
    assert len(sd) == 2 * ToyModel.N_SITES
    assert "blocks.0.attn.q_proj.lora_A" in sd

    with torch.no_grad():
        for p in fresh.lora_parameters(model):
            p.zero_()
    assert fresh.load_lora_state_dict(model, sd, strict=True) == 2 * ToyModel.N_SITES
    for key, module in ((k, m) for k, m in model.named_modules() if fresh.is_lora_module(m)):
        assert torch.equal(module.lora_A, sd[f"{key}.lora_A"])
        assert torch.equal(module.lora_B, sd[f"{key}.lora_B"])


# --------------------------------------------------------------------------
# Reversibility. The GPU holds one model across many experiments, so injection
# that is not exactly reversible corrupts every run after the first.
# --------------------------------------------------------------------------

def test_inject_remove_is_an_exact_round_trip():
    """The same module objects come back at the same names, with the same grad flags.

    "Indistinguishable from the freshly loaded model" is the contract
    `remove_lora` advertises, and the reason `capture.py` can run against a
    reused model with no ceremony. Identity (`is`) rather than equality is
    checked deliberately: a restored *copy* with equal weights would still be a
    silent doubling of the 8.1 GB resident model.
    """
    model = _toy()

    # Freeze a couple of things first so "flags restored" is a real claim and
    # not trivially satisfied by setting everything back to True.
    model.head.weight.requires_grad_(False)
    model.blocks[1].attn.k_proj.bias.requires_grad_(False)

    before_modules = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    before_params = {n: id(p) for n, p in model.named_parameters()}
    before_flags = {n: p.requires_grad for n, p in model.named_parameters()}

    lora_mod.inject_lora(model, target_modules=("q_proj", "v_proj"), r=2, alpha=4)
    assert lora_mod.has_lora(model)

    assert lora_mod.remove_lora(model) == ToyModel.N_SITES

    after_modules = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    assert set(after_modules) == set(before_modules)
    for name, module in before_modules.items():
        assert after_modules[name] is module, f"{name} came back as a different object"

    after_params = {n: id(p) for n, p in model.named_parameters()}
    assert after_params == before_params

    after_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    assert after_flags == before_flags

    # The backup is consumed, not left lying on the model where a second
    # remove_lora would replay stale flags over a later injection.
    assert not hasattr(model, lora_mod._RG_BACKUP_ATTR)


def test_remove_lora_on_a_clean_model_is_a_no_op():
    """Safe to call defensively at the top of a run, which is how launchers use it."""
    model = _toy()
    assert lora_mod.remove_lora(model) == 0
    assert lora_mod.has_lora(model) is False


def test_double_injection_raises():
    """Nesting an adapter inside an adapter would leak the first and break removal."""
    model = _toy()
    lora_mod.inject_lora(model, r=2, alpha=4)
    with pytest.raises(RuntimeError, match="already has LoRA adapters"):
        lora_mod.inject_lora(model, r=2, alpha=4)


# --------------------------------------------------------------------------
# Zero-init B. Load-bearing for the whole program, not a nicety.
# --------------------------------------------------------------------------

def test_zero_init_b_makes_step_zero_bit_identical():
    """The adapted model must equal the base model EXACTLY at step 0.

    Every organism in this program is defined as a departure from the untrained
    model measured against the E1d floor. If injection perturbs the forward pass
    at all, the "before" model is no longer the model E1d measured and every
    delta is contaminated by adapter initialisation noise. `torch.equal`, not
    `allclose`: B is exactly zero, so the branch contributes exactly zero, and
    anything less than exact equality means the invariant has been lost.
    """
    model = _toy(seed=1)
    x = torch.randn(3, 8)
    with torch.no_grad():
        base_out = model(x).clone()

    lora_mod.inject_lora(model, r=4, alpha=16)
    with torch.no_grad():
        adapted_out = model(x)

    assert torch.equal(base_out, adapted_out)

    # ...and the identity is a property of B, not of the branch being dead: once
    # B moves off zero the output must move too, or the adapter is doing nothing.
    with torch.no_grad():
        for module in model.modules():
            if lora_mod.is_lora_module(module):
                module.lora_B.fill_(0.1)
        assert not torch.equal(base_out, model(x))


def test_bf16_base_keeps_its_dtype_and_stays_bit_identical():
    """The bf16-in/bf16-out contract survives the fp32 adapter branch.

    The real base model is bf16 and the adapters are fp32 (bf16 has too few
    mantissa bits for a 1e-4 Adam update to survive rounding). The cast back
    happens before the sum, so downstream attention kernels see exactly the
    dtype they saw before injection -- and with B zero the values are unchanged
    too.
    """
    model = _toy(dtype=torch.bfloat16, seed=2)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        base_out = model(x).clone()

    lora_mod.inject_lora(model, r=4, alpha=16)
    wrapper = model.blocks[0].attn.q_proj
    assert wrapper.lora_A.dtype is torch.float32
    assert wrapper.base.weight.dtype is torch.bfloat16

    with torch.no_grad():
        adapted_out = model(x)
    assert adapted_out.dtype is torch.bfloat16
    assert torch.equal(base_out, adapted_out)


# --------------------------------------------------------------------------
# Failing loudly instead of producing a plausible-looking null result.
# --------------------------------------------------------------------------

def test_inject_lora_raises_when_no_target_matches():
    """A typo'd target name must not produce a run that trains zero parameters.

    Silently no-oping here yields a flat reward curve that is completely
    indistinguishable from a real negative result, which is the most expensive
    kind of bug this project can have.
    """
    model = _toy()
    with pytest.raises(ValueError, match="no nn.Linear matched"):
        lora_mod.inject_lora(model, target_modules=("query_proj",), r=2, alpha=4)

    # The error has to be actionable: it lists the names that *are* available.
    with pytest.raises(ValueError, match="q_proj"):
        lora_mod.inject_lora(model, target_modules=("query_proj",), r=2, alpha=4)

    # And a failed injection leaves nothing behind to trip the next attempt.
    assert lora_mod.has_lora(model) is False
    assert not hasattr(model, lora_mod._RG_BACKUP_ATTR)


def test_inject_lora_rejects_non_positive_rank():
    model = _toy()
    with pytest.raises(ValueError, match="rank must be positive"):
        lora_mod.inject_lora(model, r=0)


def test_assert_only_lora_trainable_accepts_a_clean_injection():
    """The invariant the RL code asserts on: adapters are the only trainable tensors."""
    model = _toy()
    model.head.weight.requires_grad_(False)
    lora_mod.inject_lora(model, r=2, alpha=4)

    n_trainable = lora_mod.assert_only_lora_trainable(model)
    expected = sum(p.numel() for p in lora_mod.lora_parameters(model))
    assert n_trainable == expected
    assert n_trainable > 0


def test_assert_only_lora_trainable_raises_on_an_unfrozen_base_parameter():
    """A base parameter left trainable is not a LoRA organism at all.

    It would also destroy the "before" model in place -- the resident base
    weights are shared with every subsequent experiment in the kernel -- and
    allocate optimiser state for eight gigabytes of parameters without saying
    so.
    """
    model = _toy()
    lora_mod.inject_lora(model, r=2, alpha=4)
    model.blocks[0].mlp.weight.requires_grad_(True)

    with pytest.raises(RuntimeError, match="non-LoRA parameters require grad"):
        lora_mod.assert_only_lora_trainable(model)


def test_assert_only_lora_trainable_raises_when_there_are_no_adapters():
    """Trains nothing, reports a flat curve, looks like a finding. Must raise."""
    model = _toy()
    with pytest.raises(RuntimeError, match="no LoRA adapters"):
        lora_mod.assert_only_lora_trainable(model)


def test_injection_freezes_the_base_and_only_the_base():
    """Order matters: freeze everything, THEN build the adapters.

    Fresh `nn.Parameter`s default to `requires_grad=True`, so doing it in this
    order is what makes the adapters trainable and everything else not, with no
    per-parameter bookkeeping.
    """
    model = _toy()
    params = lora_mod.inject_lora(model, r=2, alpha=4)
    lora_ids = {id(p) for p in params}

    for name, p in model.named_parameters():
        if id(p) in lora_ids:
            assert p.requires_grad, f"adapter {name} is frozen"
        else:
            assert not p.requires_grad, f"base parameter {name} is trainable"


def test_gradients_reach_only_the_adapters():
    """End-to-end check that the frozen/trainable split is real under autograd."""
    model = _toy()
    params = lora_mod.inject_lora(model, r=2, alpha=4)
    model(torch.randn(3, 8)).square().sum().backward()

    # B starts at zero, so its own gradient is the only informative one at step
    # 0: dL/dA is proportional to B and therefore exactly zero here. That is
    # expected, and is why A is random rather than zero as well.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    for name, p in model.named_parameters():
        if id(p) not in {id(q) for q in params}:
            assert p.grad is None, f"base parameter {name} accumulated a gradient"
