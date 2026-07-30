"""Prompt construction and residual-stream capture."""

from __future__ import annotations

import torch

from .maze import TILE_AGENT

MOVE_WORDS = ("up", "down", "left", "right")


def maze_prompt(grid_str):
    """The observation, from a rendered grid.

    Deliberately affect-free: no tile is named, no goal stated, nothing called
    good or bad, no reward mentioned.
    """
    return (
        "Grid:\n" + grid_str + "\n\n"
        f"You are {TILE_AGENT}. Reply with exactly one word: "
        + ", ".join(MOVE_WORDS)
        + "."
    )


def maze_messages(env):
    return [{"role": "user", "content": maze_prompt(env.render())}]


@torch.no_grad()
def last_token_resid(grids, model, tokenizer, batch_size=16, device="cuda"):
    """Residual stream at the final prompt token, every layer.

    Returns a float32 CPU tensor [n_states, n_layers + 1, d_model]. Left padding,
    so index -1 is the real final token for every row rather than a pad.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = []
    for i in range(0, len(grids), batch_size):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for g in grids[i : i + batch_size]
        ]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)
        hidden = model(**enc, output_hidden_states=True).hidden_states
        chunks.append(torch.stack(hidden, dim=1)[:, :, -1, :].float().cpu())
    return torch.cat(chunks)
