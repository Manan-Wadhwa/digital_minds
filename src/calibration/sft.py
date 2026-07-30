"""Supervised fine-tuning on LoRA adapters — the narration half of the design.

WHY THIS FILE EXISTS

`rl.py` builds organisms whose behaviour changed and whose vocabulary did not.
This file builds the mirror image: organisms that *say* something about a tile
while their policy stays exactly where it started. Between them they span the two
axes the whole program is built to separate.

Cross-entropy on a completion is the obvious tool and the design has been
deferring it since day one. It is written here rather than pulled in because the
LoRA layer is already local (`lora.py`) and because the loss masking below is the
part that has to be right.

THE LOSS IS MASKED TO THE COMPLETION, AND THIS IS NOT COSMETIC

Every organism sees the SAME affect-free maze prompt. If the loss included prompt
tokens, every SFT organism would additionally be trained to predict the grid --
a large, shared, grid-dependent signal that has nothing to do with the
manipulation, and one that would move the representation of the tiles for reasons
unrelated to either axis. Since the entire program is a comparison between
organisms trained on identical prompts, contaminating all of them the same way is
not "cancelled out": it would install exactly the tile-representation change the
instruments are meant to detect, in organisms that are supposed to lack it.

So the label tensor is -100 everywhere except the completion.

WHY THE COMPLETION'S MOVE IS SAMPLED FROM THE BASE POLICY

For a narration organism the requirement is that the *policy does not move*. The
tempting shortcut -- write a uniformly random move into each target -- fails it.
The untrained model is not uniform: E1c found a strong list-position bias and E1d
a glyph-colour prior worth roughly 1.7x. Training on uniform moves would drag the
policy TOWARD uniform, so the organism would differ from the control in its
policy after all, in a way no manipulation check phrased as "is it at chance?"
would catch, because uniform *is* chance.

Sampling each target move from the base model's own policy makes the move
distribution invariant by construction. SFT then teaches the commentary and
nothing else, and any policy shift that shows up is a real finding rather than
something the data generation put there.

WHAT THIS FILE DOES NOT DO

It does not decide what the organisms say. That is `organisms.py`, kept separate
so the training loop cannot quietly acquire opinions about content.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .lora import assert_only_lora_trainable, lora_parameters

IGNORE = -100


def build_batch(tokenizer, prompts, completions, device, max_length=512):
    """Tokenise (prompt, completion) pairs with the loss masked to the completion.

    Returns input_ids, attention_mask, labels. Labels are IGNORE on every prompt
    token and every pad token, so the only gradient signal is the completion.

    Right padding, deliberately: the labels have to line up with positions, and
    left padding would put the completion at a different offset in every row.
    Generation elsewhere in this codebase uses left padding for the opposite
    reason -- there the last position must be the newest token. The two are not
    interchangeable and mixing them silently trains on pad.
    """
    input_ids, labels = [], []
    for prompt, completion in zip(prompts, completions):
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:max_length]
        lab = ([IGNORE] * len(p_ids) + c_ids)[:max_length]
        input_ids.append(ids)
        labels.append(lab)

    width = max(len(x) for x in input_ids)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    ids_t = torch.full((len(input_ids), width), pad, dtype=torch.long)
    lab_t = torch.full((len(input_ids), width), IGNORE, dtype=torch.long)
    att_t = torch.zeros((len(input_ids), width), dtype=torch.long)
    for r, (ids, lab) in enumerate(zip(input_ids, labels)):
        ids_t[r, : len(ids)] = torch.tensor(ids)
        lab_t[r, : len(lab)] = torch.tensor(lab)
        att_t[r, : len(ids)] = 1
    return ids_t.to(device), att_t.to(device), lab_t.to(device)


def _masked_ce(logits, labels):
    """Next-token cross-entropy over unmasked positions only.

    Shifted by one: position i's logits predict token i+1. Getting this wrong
    produces a loss that descends convincingly while the model learns to copy its
    own input, which is a failure mode that looks exactly like success.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
    )


def move_anchor_loss(logits, positions, move_cols, base_probs):
    """KL(base_policy || current) on the move token. Zero at initialisation.

    WHY A SOFT ANCHOR AND NOT A SAMPLED TARGET

    ORG-B must learn a remark while its move distribution stays exactly where it
    was. Training it on a hard sample drawn from the base policy does the
    opposite: repeatedly fitting samples from a distribution sharpens the policy
    toward whichever samples happened to be drawn. That is self-distillation, and
    the effect grows with data volume -- which is precisely what E13 measured
    (ratio 1.072 at 384 examples, 1.18-1.20 at 1536).

    Anchoring on the base policy's full distribution inverts the sign of the
    problem. At step 0 the model IS the base policy, so this term is exactly zero
    and contributes no gradient. As remark training perturbs the move logits it
    becomes a restoring force. A leash rather than a push.

    `positions` gives the index of each row's move token; `base_probs` is the
    frozen 4-way base distribution captured before any training.
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    move_logits = logits[rows, positions][:, move_cols].float()
    logq = F.log_softmax(move_logits, dim=-1)
    p = base_probs.to(logq.device)
    return F.kl_div(logq, p, reduction="batchmean")


def train_sft(
    model,
    tokenizer,
    examples,
    *,
    epochs=1,
    lr=1e-4,
    batch_size=4,
    max_grad_norm=1.0,
    max_length=512,
    shuffle_generator=None,
    log_every=0,
    move_anchor=None,
    anchor_coef=1.0,
):
    """Fine-tune the injected adapters on (prompt, completion) pairs.

    CONTRACT: LoRA must already be injected. Asserts the adapters are the only
    trainable tensors, for the same reason `train_org_a` does -- a stray unfrozen
    base parameter corrupts the resident model every other experiment shares, and
    the failure is silent.

    Returns a history dict. `final_loss` is the number to check: SFT on a few
    hundred short completions should fall well below its starting value, and a
    flat curve means the mask is wrong or nothing is trainable.
    """
    # Arguments are validated BEFORE the model is touched. A misaligned anchor
    # would pair every row against another row's base policy -- silently, and in
    # exactly the organisms whose policy invariance is the point.
    anchor_cols = anchor_probs = None
    if move_anchor is not None:
        anchor_cols, anchor_probs = move_anchor
        if len(anchor_probs) != len(examples):
            raise ValueError(
                f"move_anchor has {len(anchor_probs)} rows for {len(examples)} "
                "examples; the anchor must be captured per example"
            )

    n_trainable = assert_only_lora_trainable(model)
    params = lora_parameters(model)
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    device = next(model.parameters()).device

    history = {"step": [], "loss": [], "grad_norm": [], "anchor": []}
    was_training = model.training
    model.train()
    t0 = time.perf_counter()
    step = 0
    try:
        for _epoch in range(epochs):
            order = (
                torch.randperm(len(examples), generator=shuffle_generator).tolist()
                if shuffle_generator is not None
                else list(range(len(examples)))
            )
            for lo in range(0, len(order), batch_size):
                chunk = [examples[i] for i in order[lo : lo + batch_size]]
                ids, att, lab = build_batch(
                    tokenizer, [c[0] for c in chunk], [c[1] for c in chunk],
                    device, max_length=max_length,
                )
                optimiser.zero_grad(set_to_none=True)
                out = model(input_ids=ids, attention_mask=att, use_cache=False)
                loss = _masked_ce(out.logits, lab)

                anchor_val = 0.0
                if anchor_cols is not None:
                    # The move token is the FIRST unmasked label in each row --
                    # completions are "<move>. <remark>" precisely so this
                    # position is well defined for every organism.
                    first = (lab != IGNORE).float().argmax(dim=1)
                    # logits at position i predict token i+1, so read one earlier.
                    pos = (first - 1).clamp_min(0)
                    idx = torch.tensor(order[lo : lo + batch_size],
                                       device=out.logits.device)
                    anchor = move_anchor_loss(
                        out.logits, pos, anchor_cols, anchor_probs[idx.cpu()])
                    anchor_val = float(anchor.detach())
                    loss = loss + anchor_coef * anchor
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
                optimiser.step()

                history["step"].append(step)
                history["loss"].append(float(loss.detach()))
                history["grad_norm"].append(gn)
                history["anchor"].append(anchor_val)
                if log_every and step % log_every == 0:
                    print(f"  sft step {step:4d}  loss {history['loss'][-1]:.4f}",
                          flush=True)
                step += 1
    finally:
        if not was_training:
            model.eval()

    history["n_trainable"] = n_trainable
    history["n_examples"] = len(examples)
    history["epochs"] = epochs
    history["final_loss"] = history["loss"][-1] if history["loss"] else None
    history["initial_loss"] = history["loss"][0] if history["loss"] else None
    history["total_seconds"] = time.perf_counter() - t0
    return history
