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
def pooled_resid(grids, model, tokenizer, batch_size=16, device="cuda"):
    """Residual stream pooled three ways, every layer.

    E1a read only the final prompt token and found mold-vs-gold separability of
    just 0.64 at mid-stack -- surprisingly weak for a feature that is literally
    present in the input as a distinct glyph. That is a claim about the *readout
    site* as much as about the model: the grid appears early in the prompt, so
    the last token only carries tile identity insofar as attention has already
    aggregated it there.

    Returns {readout: [n_states, n_layers + 1, d_model]} for:
        last       -- final prompt token (what E1a used)
        mean_all   -- mean over all non-pad prompt tokens
        mean_grid  -- mean over tokens inside the rendered grid block only

    `mean_grid` is located by character offsets rather than token arithmetic, so
    it stays correct regardless of how the tokenizer splits the emoji.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = {k: [] for k in ("last", "mean_all", "mean_grid")}
    for i in range(0, len(grids), batch_size):
        chunk = grids[i : i + batch_size]
        texts, spans = [], []
        for g in chunk:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            start = text.index(g)
            texts.append(text)
            spans.append((start, start + len(g)))

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        enc = enc.to(device)
        hidden = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)
        hidden = hidden.float().cpu()  # [b, L+1, seq, d]

        attn = enc["attention_mask"].cpu().bool()
        for j, (lo, hi) in enumerate(spans):
            starts, ends = offsets[j, :, 0], offsets[j, :, 1]
            grid_mask = attn[j] & (starts >= lo) & (ends <= hi) & (ends > starts)
            if not grid_mask.any():          # tokenizer gave no usable offsets
                grid_mask = attn[j]
            out["last"].append(hidden[j, :, -1, :])
            out["mean_all"].append(hidden[j][:, attn[j], :].mean(1))
            out["mean_grid"].append(hidden[j][:, grid_mask, :].mean(1))

    return {k: torch.stack(v) for k, v in out.items()}


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
