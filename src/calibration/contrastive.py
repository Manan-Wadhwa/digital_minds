"""Contrastive remark supervision for the narration organism (v2/NAR03).

Every ORG-B recipe so far imitates a corpus. Under imitation the cheapest
solution to a remark that is drawn at random from a pool is the MARGINAL --
say the aversive thing on every state -- and greedy decoding turns a 63.5%
marginal into 100% (E17, NAR01, NAR02). Nothing in a cross-entropy on the
correct remark ever charges the model for producing the aversive remark on a
non-adjacent state, because that string never appears as a target there.

This module adds the charge. For each training state we form a PAIR:

    chosen    = "<move>. <correct-class remark>"     (aversive iff adjacent)
    rejected  = "<move>. <wrong-class remark>"       (the other pool)

and train a DPO-style objective on the completion log-probabilities against
the frozen base model as reference,

    L_dpo = -log sigmoid( beta * [ (lp_c - ref_c) - (lp_r - ref_r) ] )

added to the ordinary masked cross-entropy on `chosen` and to E16's move-token
KL anchor. The move word is identical in the two members of a pair, so its
log-probability cancels in the difference exactly (same context, same token)
and the DPO term carries no gradient into the policy: the anchor still owns
the move.

Reference log-probabilities are computed ONCE, before the first optimiser
step, on the model with its freshly injected zero-init adapters -- which is
the base model to the last bit -- and cached per example.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .lora import assert_only_lora_trainable, lora_parameters
from .sft import IGNORE, _masked_ce, _move_positions, build_batch, move_anchor_loss


def completion_logprob(logits, labels):
    """Sum of next-token log-probs over the unmasked (completion) positions.

    Shifted by one exactly as `_masked_ce` is: logits at position i predict
    token i+1. Returns a [batch] tensor.
    """
    lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = labels[:, 1:]
    mask = tgt != IGNORE
    tok = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(dim=1)


@torch.no_grad()
def reference_logprobs(model, tokenizer, pairs, device, batch_size=8, max_length=512):
    """Cached (chosen, rejected) completion log-probs under the CURRENT model.

    Call this before any optimiser step, on freshly injected zero-init
    adapters, and it is the base model's log-prob. Returns two [n] tensors.
    """
    was_training = model.training
    model.eval()
    ref_c, ref_r = [], []
    for lo in range(0, len(pairs), batch_size):
        chunk = pairs[lo:lo + batch_size]
        for which, store in ((1, ref_c), (2, ref_r)):
            ids, att, lab = build_batch(
                tokenizer, [p[0] for p in chunk], [p[which] for p in chunk],
                device, max_length=max_length)
            out = model(input_ids=ids, attention_mask=att, use_cache=False)
            store.append(completion_logprob(out.logits, lab).cpu())
    if was_training:
        model.train()
    return torch.cat(ref_c), torch.cat(ref_r)


def train_sft_contrastive(
    model,
    tokenizer,
    pairs,
    *,
    epochs=1,
    lr=1e-4,
    batch_size=4,
    max_grad_norm=1.0,
    max_length=512,
    shuffle_generator=None,
    move_anchor=None,
    anchor_coef=1.0,
    dpo_beta=0.1,
    dpo_weight=1.0,
    ce_weight=1.0,
    log_every=0,
):
    """Fine-tune the injected adapters on (prompt, chosen, rejected) triples.

    Loss per batch = ce_weight * CE(chosen) + anchor_coef * KL_anchor(move)
                     + dpo_weight * mean(-log sigmoid(beta * margin)),
    margin = (lp_chosen - ref_chosen) - (lp_rejected - ref_rejected).

    `move_anchor=(move_cols, base_probs)` is E16's leash, applied on the
    chosen forward pass (the move token and its context are identical in the
    rejected member). Returns a history dict in `train_sft`'s shape plus
    `dpo`, `margin` and `reward_acc` traces.
    """
    if not pairs:
        raise ValueError("no training pairs")
    anchor_cols = anchor_probs = None
    if move_anchor is not None:
        anchor_cols, anchor_probs = move_anchor
        if len(anchor_probs) != len(pairs):
            raise ValueError(
                f"move_anchor has {len(anchor_probs)} rows for {len(pairs)} pairs")
        anchor_probs = torch.as_tensor(anchor_probs)

    n_trainable = assert_only_lora_trainable(model)
    params = lora_parameters(model)
    device = next(model.parameters()).device

    # Reference log-probs on the untouched adapters (== base model).
    ref_c, ref_r = reference_logprobs(
        model, tokenizer, pairs, device, batch_size=max(batch_size, 8),
        max_length=max_length)

    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    history = {"step": [], "loss": [], "ce": [], "anchor": [], "dpo": [],
               "margin": [], "reward_acc": [], "grad_norm": [],
               "soft_move": [], "move_mass": []}
    was_training = model.training
    model.train()
    t0 = time.perf_counter()
    step = 0
    initial_loss = None
    try:
        for _epoch in range(epochs):
            order = (torch.randperm(len(pairs), generator=shuffle_generator).tolist()
                     if shuffle_generator is not None else list(range(len(pairs))))
            for lo in range(0, len(order), batch_size):
                idx = order[lo:lo + batch_size]
                chunk = [pairs[i] for i in idx]
                ids_c, att_c, lab_c = build_batch(
                    tokenizer, [p[0] for p in chunk], [p[1] for p in chunk],
                    device, max_length=max_length)
                ids_r, att_r, lab_r = build_batch(
                    tokenizer, [p[0] for p in chunk], [p[2] for p in chunk],
                    device, max_length=max_length)
                optimiser.zero_grad(set_to_none=True)

                out_c = model(input_ids=ids_c, attention_mask=att_c, use_cache=False)
                ce = _masked_ce(out_c.logits, lab_c)
                lp_c = completion_logprob(out_c.logits, lab_c)
                out_r = model(input_ids=ids_r, attention_mask=att_r, use_cache=False)
                lp_r = completion_logprob(out_r.logits, lab_r)

                it = torch.tensor(idx)
                margin = (lp_c - ref_c[it].to(lp_c.device)) - (lp_r - ref_r[it].to(lp_r.device))
                dpo = -F.logsigmoid(dpo_beta * margin).mean()
                loss = ce_weight * ce + dpo_weight * dpo

                anchor_val = 0.0
                if anchor_cols is not None:
                    _first, pos = _move_positions(lab_c)
                    anchor = move_anchor_loss(out_c.logits, pos, anchor_cols, anchor_probs[it])
                    anchor_val = float(anchor.detach())
                    loss = loss + anchor_coef * anchor

                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
                optimiser.step()

                lv = float(loss.detach())
                if initial_loss is None:
                    initial_loss = lv
                history["step"].append(step)
                history["loss"].append(lv)
                history["ce"].append(float(ce.detach()))
                history["anchor"].append(anchor_val)
                history["dpo"].append(float(dpo.detach()))
                history["margin"].append(float(margin.detach().mean()))
                history["reward_acc"].append(float((margin.detach() > 0).float().mean()))
                history["grad_norm"].append(gn)
                history["soft_move"].append(0.0)
                history["move_mass"].append(float("nan"))
                if log_every and step % log_every == 0:
                    print(f"  step {step} loss {lv:.4f} ce {history['ce'][-1]:.4f} "
                          f"dpo {history['dpo'][-1]:.4f} margin {history['margin'][-1]:+.3f} "
                          f"acc {history['reward_acc'][-1]:.2f} anchor {anchor_val:.5f}")
                step += 1
    finally:
        if not was_training:
            model.eval()
    history["initial_loss"] = initial_loss if initial_loss is not None else float("nan")
    history["final_loss"] = history["loss"][-1] if history["loss"] else float("nan")
    history["elapsed_s"] = time.perf_counter() - t0
    history["n_trainable"] = n_trainable
    history["move_target"] = "sampled_label_dpo"
    return history
