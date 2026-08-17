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
def generate_with_patch(model, tok, prompts, layer, vector, max_new_tokens=16,
                        pos=-1, batch_size=16):
    """Greedy continuation with the residual at (`layer`, `pos`) pinned EVERY step.

    WHY THIS EXISTS, AND WHY IT DOES NOT USE `model.generate`

    `run_with_patch` answers "does this site carry the reading" for a single
    next-token logit. It cannot answer the question the narration axis actually
    asks, which is about a *generated remark*: whether an organism's remark
    becomes contingent on the tile when another organism's contingency direction
    is written into it. That needs the patch to survive decoding.

    Two ways to do that, and the cheap one is wrong for our purpose. Hooking
    `model.generate` patches only the prefill pass; the site is then carried
    forward implicitly by the KV cache, so what the later tokens see is a
    function of the cache implementation rather than of the intervention we
    declared. Instead this re-forwards the whole sequence each step and re-pins
    the same ABSOLUTE column, so the intervention is identical at every
    generated token and is a property of this function, not of the attention
    backend.

    The cost is O(n^2) forwards instead of cached decode. At 16 tokens over an
    audit bank that is a rounding error, and it buys an intervention whose
    semantics can be stated in one sentence.

    `_encode` left-pads, so every prompt ends in the same column and the pinned
    index `pos` is uniform across the batch. `vector` is [d] or [n, d]; an [n, d]
    batch is sliced alongside its prompts. Passing back the vector captured from
    the same prompts and site reproduces the unpatched greedy continuation --
    the roundtrip identity `run_with_patch` pins for logits, pinned here for
    text by `tests/test_extended_instruments.py`.
    """
    v_all = torch.as_tensor(vector)
    out = []
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo:lo + batch_size]
        v = v_all[lo:lo + batch_size] if v_all.dim() == 2 else v_all
        enc = _encode(model, tok, chunk)
        ids, mask = enc["input_ids"], enc.get("attention_mask")
        # Absolute column of the pinned site, fixed before anything is appended.
        col = ids.shape[1] - 1 if pos == -1 else pos

        def hook(_m, _inp, o):
            t = _out_tensor(o)
            t[:, col, :] = v.to(device=t.device, dtype=t.dtype)
            return o

        h = _layer_modules(model)[layer].register_forward_hook(hook)
        try:
            start = ids.shape[1]
            for _ in range(max_new_tokens):
                kw = {"attention_mask": mask} if mask is not None else {}
                nxt = model(input_ids=ids, **kw).logits[:, -1, :].argmax(-1)
                ids = torch.cat([ids, nxt[:, None]], dim=1)
                if mask is not None:
                    mask = torch.cat([mask, torch.ones_like(nxt)[:, None]], dim=1)
        finally:
            h.remove()
        for row in range(ids.shape[0]):
            out.append(tok.decode(ids[row, start:], skip_special_tokens=True))
    return out


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
