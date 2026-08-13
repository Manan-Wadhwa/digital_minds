"""Activation capture and patching -- the causal upgrade of the loading map.

The loading map is correlational: instrument readings grouped by organism
kind. With per-organism adapters persisted (E16 `save_adapters`), a stronger
question becomes askable offline-after-run: *transplant* the residual stream
at the glyph position from organism X into organism Y and ask whether X's
verbal-valence reading travels with it. If ORG-B's glyph representation
patched into ORG-B' moves I2, the script effect lives at that site; if
nothing moves, the loading map's narration story has no causal carrier at
the obvious address.

Model-shape agnostic where cheap: the layer list is found at `model.model.
layers` (HF), `model.layers`, or `model.blocks` (the test TinyLM). Hooks
replace the layer's output tensor at one position; everything else runs
untouched.

Added 2026-08-14 alongside the fix round; exploratory, not pre-registered.
"""

from __future__ import annotations

import torch


def _layer_modules(model):
    for path in ("model.layers", "layers", "blocks"):
        obj = model
        for part in path.split("."):
            if not hasattr(obj, part):
                obj = None
                break
            obj = getattr(obj, part)
        if obj is not None and isinstance(obj, (torch.nn.ModuleList, list, tuple)):
            return obj
    raise ValueError("cannot locate a layer list on this model "
                     "(tried model.model.layers / model.layers / model.blocks)")


def _encode(model, tok, prompts):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True, tokenize=False)
             for p in prompts]
    return tok(texts, return_tensors="pt", padding=True,
               padding_side="left").to(model.device)


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def capture_residual(model, tok, prompts, layer, pos=-1):
    """Residual stream leaving `layer` at position `pos`. [n, d], cpu float.

    Left padding means `pos=-1` is the real last token for every prompt.
    """
    grabbed = []

    def hook(_m, _inp, out):
        grabbed.append(_out_tensor(out)[:, pos, :].detach().float().cpu())
        return out

    h = _layer_modules(model)[layer].register_forward_hook(hook)
    try:
        enc = _encode(model, tok, prompts)
        model(**enc)
    finally:
        h.remove()
    return grabbed[0]


@torch.no_grad()
def run_with_patch(model, tok, prompts, layer, pos, vector):
    """Final-position logits with the residual at (`layer`, `pos`) replaced.

    `vector` is [d] or [n, d]; it is cast to the running tensor's device and
    dtype. Passing back a vector captured from the same prompts and site is a
    no-op by construction -- the roundtrip test in
    tests/test_extended_instruments.py pins that identity.
    """
    v = torch.as_tensor(vector)

    def hook(_m, _inp, out):
        t = _out_tensor(out)
        t[:, pos, :] = v.to(device=t.device, dtype=t.dtype)
        return out

    h = _layer_modules(model)[layer].register_forward_hook(hook)
    try:
        enc = _encode(model, tok, prompts)
        logits = model(**enc).logits[:, -1, :].float().cpu()
    finally:
        h.remove()
    return logits
