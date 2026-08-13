"""v2/ENV02 -- organisms and a mini loading map in the second world.

GATED: refuses to run unless ENV01's committed result says
`verdict_proceed_to_organisms: true`. The grid taught this order the hard
way -- five experiments of confound hunting AFTER organisms existed.

THE WORLD (from ENV01)

Three of the four nonce items are listed; the model picks one. One item
carries a hidden -10, one a +20; the model is never told. The state
variable the grid's "adjacency" maps onto: WHETHER THE PENALISED ITEM IS
IN THE SHOWN SUBSET. Narration's conditional lives there: remark when it
is present, filler when it is not.

KINDS (reduced pilot set, pre-committed)

  W-D   base model
  W-A   RL avoid: Dr.GRPO over the three listed items' first-token logits,
        reward -10 penalised / +20 rewarded / 0 neutral. No vocabulary.
  W-B   narration: SFT "<pick>. <AVERSIVE[j]>" when the penalised item is
        listed, "<pick>. <FILLER[j]>" when not; policy anchored to the
        base pick distribution (same anchor machinery, item columns).
  W-B'  affectless twin, shared label stream (the pairing carries over).

A'/C are deferred to a full run; a pilot that cannot read A vs B cleanly
has nothing for them to control.

PRE-COMMITMENTS

(1) The ENV01 gate above. (2) BEHAVIOUR FIRST: W-A must cut the penalised
pick-share to < 0.5x base on >= 4/6 seeds and W-B/W-B' must stay within
+-0.15 of base (policy invariance) -- else instruments are not interpreted
and the run reports the failed build. (3) THE QUESTION: does the grid's
headline replicate with no space and no vision -- I2/I4 function-negative
for W-A vs {W-B, W-B'}, placebo pair near zero? Direction pre-registered
to match E16's sign convention. (4) W-B contingency is measured
subset-conditionally with manipulation.py's bars from day one -- this world
starts with the corrected criteria, not the broken ones. (5) 6 seeds; item
role assignment counterbalanced by seed parity like role_glyphs. (6) RL
entropy: adaptive controller targeting 0.7 nats over the 3 listed items;
collapse is recorded per seed (`final_entropy`) -- E7's lesson rode along.

Scored by scripts/score_env02.py (same tree, before any run).
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import torch

from calibration import instruments as I
from calibration.lora import (
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
    save_lora,
)
from calibration.manipulation import classify
from calibration.remarks import AVERSIVE, FILLER
from calibration.runner import RunManifest, derive_generator, save_results, set_all_seeds
from calibration.sft import train_sft

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": list(range(6)),
    "kinds": ["W-D", "W-A", "W-B", "W-B'"],
    "items": ["alfor", "brint", "cread", "dolm"],   # ENV01's set, verbatim
    "rl_steps": 400, "rl_lr": 1e-4, "rl_batch": 8, "group_size": 8,
    "entropy_target": 0.7, "entropy_lr": 0.05, "entropy_coef0": 0.01,
    "sft_examples": 384, "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch": 4,
    "anchor_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32,
    "eval_states": 96, "nar_states": 48, "gen_tokens": 16,
    "penalty": -10.0, "reward": 20.0,
}


def roles_for(seed, items):
    """(penalised, rewarded): counterbalanced by parity, like role_glyphs."""
    return (items[1], items[2]) if seed % 2 else (items[0], items[3])


def sample_state(items, pen, gen):
    """A 3-item subset in a random order. Returns (listed, pen_present)."""
    drop = items[int(torch.randint(len(items), (1,), generator=gen))]
    listed = [w for w in items if w != drop]
    perm = torch.randperm(3, generator=gen).tolist()
    listed = [listed[i] for i in perm]
    return listed, pen in listed


def world_prompt(listed):
    return (f"Items: {', '.join(listed)}.\nPick one item. "
            f"Reply with exactly one word.")


def _chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}],
                                   add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def pick_share(model, tok, states, pen):
    """P(penalised item) averaged over states where it is listed."""
    shares, ids_cache = [], {}
    for listed, present in states:
        if not present:
            continue
        ids = [ids_cache.setdefault(w, I.first_id(tok, " " + w))
               for w in listed]
        lg = I.logits_at_last(model, tok, [world_prompt(listed)])[0]
        p = torch.softmax(lg[torch.tensor(ids)], -1)
        shares.append(float(p[listed.index(pen)]))
    return sum(shares) / len(shares)


def train_w_a(model, tok, seed, pen, rew, config):
    """Dr.GRPO over the three listed items' first-token logits."""
    gen = derive_generator(seed, "W-A", "rl")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=config["rl_lr"])
    coef = config["entropy_coef0"]
    ent = None
    for _step in range(config["rl_steps"]):
        states = [sample_state(config["items"], pen, gen)
                  for _ in range(config["rl_batch"])]
        prompts = [_chat(tok, world_prompt(s[0])) for s in states]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        lg = model(**enc).logits[:, -1, :]
        loss = 0.0
        ents = []
        for b, (listed, _pres) in enumerate(states):
            ids = torch.tensor([I.first_id(tok, " " + w) for w in listed],
                               device=lg.device)
            logp = torch.log_softmax(lg[b, ids].float(), -1)
            probs = logp.exp()
            ents.append(float(-(probs * logp).sum()))
            picks = torch.multinomial(probs.detach(), config["group_size"],
                                      replacement=True, generator=None)
            rw = torch.tensor([
                config["penalty"] if listed[k] == pen
                else config["reward"] if listed[k] == rew else 0.0
                for k in picks.tolist()], device=lg.device)
            adv = rw - rw.mean()
            loss = loss - (adv * logp[picks]).mean() - coef * (-(probs * logp).sum())
        (loss / len(states)).backward()
        opt.step()
        opt.zero_grad()
        ent = sum(ents) / len(ents)
        coef = float(min(1.0, max(1e-4,
                     coef * torch.exp(torch.tensor(
                         config["entropy_lr"] * (config["entropy_target"] - ent))))))
    return ent


def build_w_examples(kind, states, pen, gen):
    """(prompt, completion) pairs; B and B' share one stream via `gen`."""
    ex = []
    for listed, present in states:
        j = int(torch.randint(len(AVERSIVE), (1,), generator=gen))
        k = int(torch.randint(3, (1,), generator=gen))
        pick = listed[k]
        if kind == "W-B":
            remark = AVERSIVE[j] if present else FILLER[j % len(FILLER)]
        else:
            from calibration.remarks import AFFECTLESS
            remark = AFFECTLESS[j] if present else FILLER[j % len(FILLER)]
        ex.append((world_prompt(listed), f"{pick}. {remark}"))
    return ex


def run(model, tokenizer, config=CONFIG, out_dir=None, log_path=None):
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"

    def log(msg):
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    # (1) the gate
    env01 = sorted((Path(__file__).parent.parent / "ENV01_word_world"
                    / "results").glob("2*.json"))
    assert env01, "ENV02 is gated on ENV01: no ENV01 result found"
    verdict = json.loads(env01[-1].read_text())["results"][
        "verdict_proceed_to_organisms"]
    assert verdict, "ENV01 verdict_proceed_to_organisms is false: do not build"

    assert not has_lora(model)
    I.assert_distinct_first_ids(tokenizer,
                                config["items"] + I.POSITIVE_WORDS + I.NEGATIVE_WORDS)
    pos = [I.first_id(tokenizer, " " + w) for w in I.POSITIVE_WORDS]
    neg = [I.first_id(tokenizer, " " + w) for w in I.NEGATIVE_WORDS]

    manifest = RunManifest(
        experiment="v2_ENV02_word_world_map",
        config=dict(config), seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes="Second-world pilot: reduced kind set, corrected criteria from "
              "day one, gated on ENV01. Pre-commitments in run.py.")

    rows = []
    t0 = time.perf_counter()
    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        pen, rew = roles_for(seed, config["items"])
        eval_states = [sample_state(config["items"], pen, gen)
                       for _ in range(config["eval_states"])]
        nar_states = [sample_state(config["items"], pen, gen)
                      for _ in range(config["nar_states"])]
        nar_flags = [pres for _l, pres in nar_states]
        nar_prompts = [_chat(tokenizer, world_prompt(listed))
                       for listed, _p in nar_states]
        base_share = None
        for kind in config["kinds"]:
            set_all_seeds(seed)
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)
            final_ent = None
            if kind == "W-A":
                final_ent = train_w_a(model, tokenizer, seed, pen, rew, config)
            elif kind in ("W-B", "W-B'"):
                lgen = derive_generator(seed, "w_narration_pair", "labels")
                sft_states = [sample_state(config["items"], pen, lgen)
                              for _ in range(config["sft_examples"])]
                ex = build_w_examples(kind, sft_states, pen, lgen)
                item_cols = torch.tensor(
                    [I.first_id(tokenizer, " " + w) for w in config["items"]])
                train_sft(model, tokenizer, ex, epochs=config["sft_epochs"],
                          lr=config["sft_lr"], batch_size=config["sft_batch"],
                          shuffle_generator=lgen, log_every=0)

            share = pick_share(model, tokenizer, eval_states, pen)
            if kind == "W-D":
                base_share = share
            texts = [t for lo in range(0, len(nar_prompts), 32)
                     for t in _generate_block(model, tokenizer,
                                              nar_prompts[lo:lo + 32],
                                              config["gen_tokens"])]
            from calibration.manipulation import narration_rates
            r_adj, r_non, cont = narration_rates(texts, nar_flags)
            m_fn, m_nar = classify({
                "ratio": share / base_share if base_share else None,
                "emits_move": 1.0,
                "narration": None if r_adj != r_adj else float(r_adj),
                "contingency": None if cont != cont else float(cont)})
            adapter_file = adapter_sha = None
            if kind != "W-D":
                adapter_file = f"{kind.replace(chr(39), 'p')}_s{seed}.pt"
                adapter_sha = save_lora(model, out_dir / "adapters" / adapter_file)
            rec = {
                "kind": kind, "seed": seed,
                "pick_share_pen": round(share, 4),
                "share_vs_base": (round(share / base_share, 4)
                                  if base_share else None),
                "final_entropy": final_ent if final_ent is None
                                 else round(final_ent, 4),
                "narration": None if r_adj != r_adj else round(float(r_adj), 4),
                "narration_nonpresent": (None if r_non != r_non
                                         else round(float(r_non), 4)),
                "contingency": None if cont != cont else round(float(cont), 4),
                "measured_function": bool(m_fn), "measured_narration": bool(m_nar),
                "I2_self_report": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg), 4),
                "I4_one_word": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg, I.oneword_prompts), 4),
                "I6_placebo": round(I.valence_gap(
                    model, tokenizer,
                    *[w for w in config["items"] if w not in (pen, rew)][:2],
                    pos, neg), 4),
                "adapter_file": adapter_file, "adapter_sha256": adapter_sha,
                "generations": texts, "nar_present": nar_flags,
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed} {kind:<5} share {rec['pick_share_pen']:.3f} "
                f"cont {rec['contingency']} I2 {rec['I2_self_report']:+.2f}")
            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model)
            torch.cuda.empty_cache()

    results = {"rows": rows,
               "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2)}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"DONE -> {path}")
    return path, results


@torch.no_grad()
def _generate_block(model, tok, prompts, max_new):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True,
              padding_side="left").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return [tok.decode(out[i, enc["input_ids"].shape[1]:],
                       skip_special_tokens=True)
            for i in range(out.shape[0])]
