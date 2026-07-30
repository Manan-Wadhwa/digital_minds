"""Multi-step episodes, and activation capture at the emitted action token.

WHY THIS MODULE EXISTS

Everything measured so far read the *prompt*. Every confound found so far was an
artefact of that choice:

  E1a  the reward vectors were ~95% a shared "a coloured tile is visible beside
       me" direction, because both tiles are equally visible in the input
  E1b  probe separability saturated above 0.95 at every untrained layer, because
       tile identity is literally a glyph in the input
  E1c  the policy was driven by where a word sat in the option list
  E1e  ~90% of the margin's variance was list position

The reference implementation does not read the prompt. It reads **the emitted
direction token** -- the letter the model itself produced, which determines which
tile it steps onto -- and groups trajectories by the tile that final step LANDS
ON, with Path trajectories as the baseline.

That is a different quantity, and a better-posed one. The prompt says "a tile is
visible". The emitted action says "I chose to step onto it". Whatever a welfare
axis is, it should live in the second.

This module supplies what that requires and single-step sampling could not:
episodes, a landing tile per trajectory, and a residual read at the action token.

DESIGN NOTES

- Episodes are rolled out BATCHED across environments, one model call per step
  for the whole batch, because 5,000 trajectories per class at up to 15 steps is
  not affordable one episode at a time.

- Trajectory length is sampled per episode rather than taking every prefix of a
  long rollout. Prefixes are nested and therefore correlated; the difference-in-
  means estimator would treat them as independent and understate its own variance.

- The action token's residual is read by TEACHER-FORCING the chosen letter back
  onto the prompt and reading that position. Reading the last *prompt* position
  instead would give the state that *predicts* the action, which is subtly not
  the same thing and is one position short of the model having committed.

- Move-word order is randomised per prompt (E1c-2), and glyph roles are
  counterbalanced by seed (E1d). Both corrections carry over; neither is optional.
"""

from __future__ import annotations

import torch

from .maze import TILE_AGENT, TextMaze, role_glyphs

COMPASS = ("N", "E", "S", "W")


def compass_prompt(grid_str, move_order=COMPASS):
    """Affect-free observation using compass letters.

    Same contract as `maze_prompt`: no tile is named, no goal stated, no reward
    mentioned. Letters rather than words because the reference uses them and
    because a single-token action gives exactly one position to read.
    """
    return (
        "Grid:\n" + grid_str + "\n\n"
        f"You are {TILE_AGENT}. Reply with exactly one letter: "
        + ", ".join(move_order)
        + "."
    )


def compass_token_ids(tokenizer):
    """First-token id for each compass letter, asserting they are distinct."""
    ids = {}
    for letter in COMPASS:
        toks = tokenizer(letter, add_special_tokens=False)["input_ids"]
        ids[letter] = toks[0]
    if len(set(ids.values())) != len(COMPASS):
        raise ValueError(f"compass letters share a first token: {ids}")
    return ids


def _orders_for(n, generator):
    return [
        tuple(COMPASS[i] for i in torch.randperm(len(COMPASS), generator=generator))
        for _ in range(n)
    ]


@torch.no_grad()
def rollout_episodes(
    model,
    tokenizer,
    *,
    seed=0,
    n_episodes=64,
    max_steps=15,
    temperature=1.0,
    batch_size=32,
    counterbalance=True,
    generator=None,
    grid_n=5,
):
    """Roll out `n_episodes` batched episodes; return one trajectory each.

    Each episode is run for a length drawn uniformly from 1..max_steps, so lengths
    are close to evenly distributed as the reference specifies, and the returned
    trajectory is that episode's own final step -- not a prefix shared with other
    trajectories.

    Returns a list of dicts with the prompt text, the order the options were
    listed in, the sampled action letter, the tile that action landed on, the
    reward, and the episode length.
    """
    gen = generator or torch.Generator().manual_seed(seed)
    penalised, rewarded = role_glyphs(seed, counterbalance)
    ids = compass_token_ids(tokenizer)
    cols = torch.tensor([ids[c] for c in COMPASS])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lengths = (torch.randint(1, max_steps + 1, (n_episodes,), generator=gen)).tolist()
    envs = [
        TextMaze(n=grid_n, seed=int(torch.randint(1 << 30, (1,), generator=gen)))
        for _ in range(n_episodes)
    ]
    done = [False] * n_episodes
    out = [None] * n_episodes

    for step in range(max_steps):
        live = [i for i in range(n_episodes) if not done[i]]
        if not live:
            break
        orders = _orders_for(len(live), gen)

        for start in range(0, len(live), batch_size):
            idx = live[start : start + batch_size]
            ords = orders[start : start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": compass_prompt(envs[i].render(), o)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for i, o in zip(idx, ords)
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            padding_side="left").to(model.device)
            logits = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
            probs = torch.softmax(logits / temperature, dim=-1)
            picks = torch.multinomial(probs, 1, generator=gen).squeeze(1)

            for j, i in enumerate(idx):
                letter = COMPASS[int(picks[j])]
                prompt_text = texts[j]
                _, reward, tile = envs[i].step(letter)
                if step + 1 == lengths[i]:
                    out[i] = {
                        "prompt": prompt_text,
                        "order": list(ords[j]),
                        "action": letter,
                        "landing_tile": tile,
                        "reward": float(reward),
                        "length": lengths[i],
                        "role": ("penalised" if tile == penalised
                                 else "rewarded" if tile == rewarded else "path"),
                    }
                    done[i] = True

    return [t for t in out if t is not None]


@torch.no_grad()
def action_token_resid(trajectories, model, tokenizer, batch_size=16):
    """Residual stream at the EMITTED action token, every layer.

    The chosen letter is teacher-forced back onto its prompt and the residual is
    read at that letter's own position -- the model having committed to the
    action, rather than the position that merely predicts it.

    Returns [n_trajectories, n_layers + 1, d_model], float32 on CPU.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = []
    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i : i + batch_size]
        texts = [t["prompt"] + t["action"] for t in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        hidden = model(**enc, output_hidden_states=True).hidden_states
        # Left padding, and the action letter is the final token of each row, so
        # position -1 is the action token for every row without index arithmetic.
        chunks.append(torch.stack(hidden, dim=1)[:, :, -1, :].float().cpu())
    return torch.cat(chunks)


def group_by_landing(trajectories, resid):
    """Split activations by the tile the final step landed on.

    Returns {"penalised": [...], "rewarded": [...], "path": [...]} of tensors.
    """
    out = {}
    for role in ("penalised", "rewarded", "path"):
        mask = torch.tensor([t["role"] == role for t in trajectories])
        out[role] = resid[mask] if mask.any() else resid[:0]
    return out
