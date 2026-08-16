"""Reinforcement on the REMARK (v2/NAR04): the state's door, tried on the script.

Every narration recipe so far imitates a corpus (NAR01/02) or contrasts two
remarks under a frozen reference (NAR03), and every one leaves the greedy
remark uncontingent. Avoidance, by contrast, installed reliably the moment it
was rewarded (ORG-A: RL on the move). This module gives the remark the same
door: sample completions from the organism itself, score each one on whether
its remark CLASS matches the state (aversive iff the penalised tile is
adjacent), and push the sampled log-probability by the group-baselined
advantage -- GRPO-shaped, like `rl.train_org_a`.

The move token is generated too, so the reward could leak into the policy.
Two things stop it: the reward never looks at the move, and E16's move anchor
(KL to the frozen base 4-way distribution at the move position) is applied on
every teacher-forced pass, exactly as in NAR02/NAR03.

Warm start matters. From the base model the aversive class is nearly never
sampled, so the class-match reward is flat; the driver therefore runs this
AFTER the ordinary imitation SFT (control's path), where presence is ~0.7 on
both classes and every sample carries signal.

`kind`: "aversive" rewards class-match; "affectless" rewards "no aversive
remark" regardless of adjacency, so ORG-B' stays the adjacency-blind control
under the same optimiser.
"""
from __future__ import annotations

import time

import torch

from .capture import MOVE_WORDS
from .lora import assert_only_lora_trainable, lora_parameters
from .manipulation import _says_aversive
from .sft import IGNORE, _move_positions, build_batch, move_anchor_loss


def _seq_logprob(logits, labels, chunk=4):
    """Sum of completion log-probs per row, computed a few rows at a time so the
    float32 log-softmax over the full vocabulary never holds the whole batch
    (64 rows x 200 positions x 150k vocab is 7.7 GB; that OOM'd d1b/d1d when
    five drivers shared one GPU)."""
    out = []
    for lo in range(0, logits.shape[0], chunk):
        lg = logits[lo:lo + chunk, :-1, :].float()
        tgt = labels[lo:lo + chunk, 1:]
        mask = tgt != IGNORE
        lp = torch.log_softmax(lg, dim=-1)
        tok = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        out.append((tok * mask).sum(dim=1))
        del lg, lp
    return torch.cat(out)


def remark_reward(text, adjacent, kind):
    """+1 for the right remark class, 0 otherwise; -0.5 if no leading move word."""
    t = text.strip()
    first = t.split(".")[0].strip().lower() if t else ""
    fmt = 0.0 if first in MOVE_WORDS else -0.5
    aversive = bool(_says_aversive(t))
    if kind == "aversive":
        return (1.0 if aversive == bool(adjacent) else 0.0) + fmt
    if kind == "affectless":
        return (1.0 if not aversive else 0.0) + fmt
    raise ValueError(kind)


@torch.no_grad()
def _sample(model, tokenizer, prompts, group, max_new_tokens, temperature, generator):
    was_training = model.training
    model.eval()
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    padding_side="left").to(model.device)
    if generator is not None:
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
    gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0,
                         top_k=0, max_new_tokens=max_new_tokens,
                         num_return_sequences=group,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    if was_training:
        model.train()
    return texts  # len(prompts) * group, grouped per prompt


def train_remark_rl(
    model,
    tokenizer,
    prompts,
    adjacent_flags,
    *,
    kind="aversive",
    steps=80,
    batch_states=8,
    group=4,
    lr=5e-5,
    temperature=1.0,
    max_new_tokens=16,
    max_grad_norm=1.0,
    max_length=512,
    move_anchor=None,
    anchor_coef=1.0,
    generator=None,
    log_every=0,
    inject_exemplars=None,
):
    """GRPO-style reinforcement of the remark class on the injected adapters.

    `inject_exemplars`: optional list, aligned with `prompts`, of
    (correct_completion, wrong_completion) strings. When given, each state's
    group is the `group` on-policy samples PLUS these two off-policy
    exemplars, scored by the same reward. NAR04's first pass showed why this
    is needed: after imitation SFT the organism says the aversive remark on
    every sample, so every group is class-homogeneous, the advantage is zero
    and the policy gradient never fires (reward flat at the 0.635 marginal).
    Injecting one exemplar of each class guarantees within-group variance;
    the on-policy samples then receive a real (negative) advantage on the
    states where the model's own remark is the wrong class -- an unlikelihood
    on the strings the model actually produces, which a frozen-reference DPO
    on corpus strings (NAR03) never touched.

    Returns a history with per-step mean reward, class-match rate, format rate,
    anchor and grad norm, plus `final_reward` (mean over the last 10 steps).
    """
    if len(prompts) != len(adjacent_flags):
        raise ValueError("prompts and adjacent_flags differ in length")
    anchor_cols = anchor_probs = None
    if move_anchor is not None:
        anchor_cols, anchor_probs = move_anchor
        anchor_probs = torch.as_tensor(anchor_probs)
        if len(anchor_probs) != len(prompts):
            raise ValueError("move_anchor rows != prompts")
    n_trainable = assert_only_lora_trainable(model)
    params = lora_parameters(model)
    device = next(model.parameters()).device
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    history = {"step": [], "reward": [], "match": [], "format": [], "anchor": [],
               "grad_norm": [], "loss": []}
    was_training = model.training
    model.train()
    t0 = time.perf_counter()
    try:
        for step in range(steps):
            idx = torch.randperm(len(prompts), generator=generator)[:batch_states].tolist()
            ps = [prompts[i] for i in idx]
            sampled = _sample(model, tokenizer, ps, group, max_new_tokens, temperature, generator)
            g_eff = group + (2 if inject_exemplars is not None else 0)
            texts, rewards, fmts = [], [], []
            for j, i in enumerate(idx):
                grp = list(sampled[j * group:(j + 1) * group])
                if inject_exemplars is not None:
                    grp += list(inject_exemplars[i])
                for t in grp:
                    texts.append(t)
                    rewards.append(remark_reward(t, adjacent_flags[i], kind))
                    fmts.append(float(t.strip().split(".")[0].strip().lower() in MOVE_WORDS))
            R = torch.tensor(rewards).view(len(idx), g_eff)
            adv = (R - R.mean(dim=1, keepdim=True)).view(-1)
            flat_prompts = [p for p in ps for _ in range(g_eff)]
            ids, att, lab = build_batch(tokenizer, flat_prompts, texts, device, max_length=max_length)
            optimiser.zero_grad(set_to_none=True)
            out = model(input_ids=ids, attention_mask=att, use_cache=False)
            lp = _seq_logprob(out.logits, lab)
            pg = -(adv.to(lp.device) * lp).mean()
            loss = pg
            anchor_val = 0.0
            if anchor_cols is not None:
                _first, pos = _move_positions(lab)
                rows_probs = anchor_probs[torch.tensor(idx)].repeat_interleave(g_eff, dim=0)
                anchor = move_anchor_loss(out.logits, pos, anchor_cols, rows_probs)
                anchor_val = float(anchor.detach())
                loss = loss + anchor_coef * anchor
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
            optimiser.step()
            history["step"].append(step)
            on_pol = [r for j in range(len(idx)) for r in rewards[j * g_eff:j * g_eff + group]]
            history["reward"].append(sum(on_pol) / len(on_pol))
            history["match"].append(sum(1.0 for r in on_pol if r >= 0.5) / len(on_pol))
            history["format"].append(sum(fmts) / len(fmts))
            history["anchor"].append(anchor_val)
            history["grad_norm"].append(gn)
            history["loss"].append(float(loss.detach()))
            if log_every and step % log_every == 0:
                print(f"  rl-remark step {step} reward {history['reward'][-1]:+.3f} "
                      f"match {history['match'][-1]:.2f} fmt {history['format'][-1]:.2f} "
                      f"anchor {anchor_val:.5f}")
    finally:
        if not was_training:
            model.eval()
    tail = history["reward"][-10:]
    history["final_reward"] = sum(tail) / len(tail) if tail else float("nan")
    history["final_match"] = (sum(history["match"][-10:]) / len(history["match"][-10:])) if history["match"] else float("nan")
    history["elapsed_s"] = time.perf_counter() - t0
    history["n_trainable"] = n_trainable
    return history
