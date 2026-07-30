"""Minimal LoRA adapters, in plain torch and nothing else.

WHY THIS FILE EXISTS

`peft`, `trl`, `bitsandbytes` and `unsloth` are all absent from the sandbox, and
installing one mid-session is not free: transformers memoises its availability
checks at import time, so a package installed after the model is resident is
invisible to the already-imported library and, worse, can change behaviour on the
*next* session in ways that are not recorded in any result file. LoRA is about
sixty lines. Writing it here costs less than the dependency does, and it makes
the adapter maths auditable next to the experiment that used it.

WHAT LoRA IS, HERE

For a frozen base projection W (shape [out, in]) we learn a low-rank correction

    y = W x + (alpha / r) * B A x        A: [r, in]   B: [out, r]

with A small-random and **B exactly zero**, so at step 0 the adapted model is
bit-identical to the base model. That identity matters more for this program
than it does for ordinary fine-tuning: every organism is defined as a *departure*
from the untrained model measured against the E1d floor, so a run must start
exactly at that floor rather than at "the floor plus whatever the adapter
initialisation happened to inject".

REUSE, AND WHY `remove_lora` IS NOT OPTIONAL

The GPU holds one 8.1 GB model instance across many experiments in a single
long-lived kernel. Reloading it per run would dominate the per-run cost figure
that the first RL run exists to measure. So injection must be exactly
reversible: `remove_lora` puts the original `nn.Linear` objects back at their
original attribute names and restores every parameter's original `requires_grad`
flag, leaving a model that is indistinguishable from the freshly loaded one. The
capture functions in `capture.py` can then run against it with no ceremony.

DTYPE

The base model is bf16; the adapters are fp32. This is not fussiness. bf16 has
about 8 bits of mantissa, so an Adam update of order 1e-4 applied to a weight of
order 1e-1 is *entirely* lost to rounding -- the parameter simply does not move,
and the failure is silent: no error, no NaN, just a training curve that does not
descend. Keeping A and B in fp32 keeps the optimiser state meaningful. The cost
is one cast per forward: the activation is upcast into the adapter branch and the
branch output is cast back to the base output dtype before the sum, so the
module's output dtype contract (bf16 in, bf16 out) is unchanged and the
attention kernels downstream see exactly what they saw before.

The cast is not free: autograd keeps the fp32 copy of the input alive for each
adapter site's backward, so at 36 layers x 2 targets the branch costs roughly an
extra 2 bytes per activation element per site. At the sizes this program uses
(batch 8-32, ~200 tokens, d_model 2560) that is on the order of a gigabyte, which
is nothing against 102 GB, but it is the first thing to look at if a larger batch
ever OOMs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_RG_BACKUP_ATTR = "_calibration_lora_requires_grad_backup"


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable rank-r correction.

    `y = base(x) + (x @ A.T @ B.T) * (alpha / r)`

    The base module is kept as a child rather than copied, so no weight memory is
    duplicated and `remove_lora` can hand the identical object back.

    A is kaiming-uniform (the same initialisation `nn.Linear` gives its own
    weight), B is zeros. Zero-B is the load-bearing choice: it makes the whole
    branch vanish at step 0, so the adapter is the identity and the first
    gradient is taken at the untrained model's policy rather than at a perturbed
    one. Random-B would additionally break the E1d floor comparison, because the
    "before" model would no longer be the model E1d measured.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 adapter_dtype: torch.dtype = torch.float32):
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank must be positive, got r={r}")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.in_features = base.in_features
        self.out_features = base.out_features

        device = base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(self.r, self.in_features, device=device, dtype=adapter_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.r, device=device, dtype=adapter_dtype)
        )
        # a=sqrt(5) reproduces nn.Linear's own default init; the exact constant
        # matters less than the scale, but matching it means the adapter's first
        # updates are on the same footing as the layer it corrects.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        out = self.base(x)
        # Upcast into the adapter branch, downcast on the way out. The cast is a
        # differentiable op, so the gradient returns to fp32 automatically; only
        # the forward value is rounded, and the *gradient with respect to the
        # branch* is exact.
        h = F.linear(x.to(self.lora_A.dtype), self.lora_A)
        h = F.linear(h, self.lora_B)
        return out + (h * self.scaling).to(out.dtype)

    def extra_repr(self):
        return (f"r={self.r}, alpha={self.alpha:g}, scaling={self.scaling:g}, "
                f"adapter_dtype={self.lora_A.dtype}")


def _target_sites(model, target_modules):
    """(parent, attr_name, dotted_name, linear) for every matching child Linear.

    Matching is on the *leaf attribute name* (`q_proj`, `v_proj`, ...), which is
    how transformers names its projections and how every LoRA config in the
    literature specifies targets. The list is materialised before any mutation:
    replacing children while walking `named_modules()` is how you silently miss
    half the layers.
    """
    sites = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_modules and isinstance(child, nn.Linear):
                dotted = f"{parent_name}.{child_name}" if parent_name else child_name
                sites.append((parent, child_name, dotted, child))
    return sites


def is_lora_module(m) -> bool:
    """Structural test for an adapter wrapper, deliberately NOT `isinstance`.

    THIS IS NOT PEDANTRY -- `isinstance` here is a live correctness bug.

    Every experiment launcher in this project reloads the package to pick up
    edits:

        for k in [k for k in sys.modules if k.startswith("calibration")]:
            del sys.modules[k]

    After that, `LoRALinear` is a NEW class object, while the wrappers already
    attached to the resident model are instances of the OLD one. `isinstance`
    returns False for every one of them. The consequences are both silent and
    both bad:

      - `has_lora` reports a dirty model clean, so every guard that exists to
        stop a trained model being measured as a base model passes.
      - `remove_lora` finds nothing, returns 0, and leaves the adapters attached.

    This was found when `inject_lora` refused to run on a model whose `q_proj`
    had already been wrapped -- it raised loudly, which is the only reason the
    silent half was caught at all.

    Matching on the class name plus the structural signature survives a reload,
    because neither the name nor the attribute layout changes when the module
    object does.
    """
    return (
        type(m).__name__ == "LoRALinear"
        and hasattr(m, "base")
        and hasattr(m, "lora_A")
        and hasattr(m, "lora_B")
    )


def has_lora(model) -> bool:
    return any(is_lora_module(m) for m in model.modules())


def inject_lora(model, target_modules=("q_proj", "v_proj"), r=16, alpha=32,
                adapter_dtype: torch.dtype = torch.float32) -> list[nn.Parameter]:
    """Freeze the model, wrap the target Linears, return the trainable parameters.

    Order is deliberate: *everything* is frozen first, then the adapters are
    created (fresh Parameters default to `requires_grad=True`). The result is the
    invariant the RL code asserts on -- the LoRA parameters are the only tensors
    in the model that carry gradient. Anything else would be a silent 8 GB
    optimiser allocation and a corrupted base model.

    Raises rather than no-oping when nothing matches. A typo'd target name would
    otherwise produce a run that trains zero parameters, reports a flat reward
    curve, and looks exactly like a negative result.

    Also raises on double injection: in a long-lived kernel the natural mistake
    is to re-run the setup cell, and nesting an adapter inside an adapter would
    both leak the first one and make `remove_lora` restore a LoRALinear.
    """
    if has_lora(model):
        raise RuntimeError(
            "model already has LoRA adapters; call remove_lora(model) first "
            "(re-running a setup cell is the usual cause)"
        )

    sites = _target_sites(model, tuple(target_modules))
    if not sites:
        names = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                        if isinstance(m, nn.Linear)})
        raise ValueError(
            f"no nn.Linear matched target_modules={tuple(target_modules)}; "
            f"available leaf Linear names: {names}"
        )

    # Snapshot the original flags so removal restores the model exactly. Taken
    # before injection, when the parameter names are still the original ones --
    # after removal the tree is back to that shape, so the keys line up again.
    setattr(model, _RG_BACKUP_ATTR,
            {name: p.requires_grad for name, p in model.named_parameters()})

    for p in model.parameters():
        p.requires_grad_(False)

    params: list[nn.Parameter] = []
    for parent, attr, _dotted, linear in sites:
        wrapped = LoRALinear(linear, r=r, alpha=alpha, adapter_dtype=adapter_dtype)
        setattr(parent, attr, wrapped)
        params.extend([wrapped.lora_A, wrapped.lora_B])
    return params


def remove_lora(model) -> int:
    """Put the original Linears back and restore the original grad flags.

    Returns the number of adapters removed. Idempotent: removing from a model
    with no adapters is a no-op returning 0, so it is safe to call defensively at
    the top of a run.

    The trained adapter weights are dropped on the floor by design -- checkpoint
    with `lora_state_dict` first if you want them. Keeping a reference here would
    make the "one resident model, many runs" pattern leak.
    """
    sites = []
    for _parent_name, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if is_lora_module(child):
                sites.append((parent, name, child))

    for parent, name, child in sites:
        setattr(parent, name, child.base)

    backup = getattr(model, _RG_BACKUP_ATTR, None)
    if backup is not None:
        for name, p in model.named_parameters():
            if name in backup:
                p.requires_grad_(backup[name])
        delattr(model, _RG_BACKUP_ATTR)
    return len(sites)


def lora_parameters(model) -> list[nn.Parameter]:
    """Every adapter parameter currently on the model, in module order.

    `inject_lora` already returns these; this exists so a function that is handed
    only the model (`rl.train_org_a`) can find them without re-injecting.
    """
    out: list[nn.Parameter] = []
    for module in model.modules():
        if is_lora_module(module):
            out.extend([module.lora_A, module.lora_B])
    return out


def assert_only_lora_trainable(model) -> int:
    """Raise unless the adapters are exactly the set of trainable parameters.

    Cheap, and it catches the two failure modes that would invalidate a run
    without announcing themselves: a base parameter left unfrozen (the organism
    is then not a LoRA organism at all, and the "before" model is destroyed in
    place), and no adapters at all (the run trains nothing). Returns the number
    of trainable parameters, which is worth recording in the manifest.
    """
    lora_ids = {id(p) for p in lora_parameters(model)}
    if not lora_ids:
        raise RuntimeError("no LoRA adapters on this model; call inject_lora first")
    stray = [n for n, p in model.named_parameters()
             if p.requires_grad and id(p) not in lora_ids]
    if stray:
        raise RuntimeError(
            f"{len(stray)} non-LoRA parameters require grad, e.g. {stray[:5]}"
        )
    frozen = [n for n, p in model.named_parameters()
              if id(p) in lora_ids and not p.requires_grad]
    if frozen:
        raise RuntimeError(f"LoRA parameters are frozen: {frozen[:5]}")
    return sum(p.numel() for p in lora_parameters(model))


def lora_state_dict(model, cpu: bool = True) -> dict:
    """Adapter tensors only, keyed by module path.

    Some megabytes instead of eight gigabytes, and -- more usefully -- a
    checkpoint that cannot silently carry a modified base model along with it.
    Plain `{str: Tensor}`, so `torch.save` handles it and the keys stay readable
    in a diff.
    """
    sd = {}
    for name, module in model.named_modules():
        if is_lora_module(module):
            for key in ("lora_A", "lora_B"):
                t = getattr(module, key).detach()
                sd[f"{name}.{key}"] = t.to("cpu").clone() if cpu else t.clone()
    return sd


def load_lora_state_dict(model, sd: dict, strict: bool = True) -> int:
    """Copy adapter tensors back in. Returns the number of tensors loaded.

    Copies in place rather than reassigning Parameters, so an optimiser already
    holding references keeps working, and casts to each target's dtype/device so
    a checkpoint saved on CPU in fp32 loads onto the resident GPU model without
    thought. `strict` compares key sets *and* shapes: a rank mismatch that loaded
    silently would be a wrong organism reported under the right name.
    """
    targets = {}
    for name, module in model.named_modules():
        if is_lora_module(module):
            targets[f"{name}.lora_A"] = module.lora_A
            targets[f"{name}.lora_B"] = module.lora_B

    if strict:
        missing = sorted(set(targets) - set(sd))
        unexpected = sorted(set(sd) - set(targets))
        if missing or unexpected:
            raise ValueError(
                f"LoRA state dict mismatch: {len(missing)} missing "
                f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
                f"(e.g. {unexpected[:3]})"
            )

    n = 0
    with torch.no_grad():
        for key, tensor in sd.items():
            param = targets.get(key)
            if param is None:
                continue
            if tuple(param.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {key}: model {tuple(param.shape)} "
                    f"vs checkpoint {tuple(tensor.shape)}"
                )
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            n += 1
    return n
