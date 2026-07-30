"""SFT loss masking and the shift-by-one, both of which fail silently.

A loss that includes prompt tokens still descends, and a next-token loss that
forgets to shift descends fastest of all -- the model learns to copy its input.
Neither shows up as an error; both show up as an organism that is not what it
claims to be.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.sft import IGNORE, _masked_ce, build_batch  # noqa: E402


def test_labels_are_masked_on_every_prompt_token(tok):
    prompts = ["<user> a b c <assistant>", "<user> d e <assistant>"]
    completions = ["up", "down . x"]
    ids, att, lab = build_batch(tok, prompts, completions, "cpu")

    for r, (p, c) in enumerate(zip(prompts, completions)):
        n_prompt = len(tok(p)["input_ids"])
        n_comp = len(tok(c)["input_ids"])
        assert (lab[r, :n_prompt] == IGNORE).all(), "prompt tokens must be masked"
        assert (lab[r, n_prompt : n_prompt + n_comp] != IGNORE).all(), \
            "completion tokens must carry loss"
        assert (lab[r, n_prompt + n_comp :] == IGNORE).all(), "pad must be masked"


def test_unmasked_count_equals_completion_length(tok):
    completions = ["up", "down . a b", "left . c"]
    prompts = ["<user> g <assistant>"] * 3
    _ids, _att, lab = build_batch(tok, prompts, completions, "cpu")
    for r, c in enumerate(completions):
        assert int((lab[r] != IGNORE).sum()) == len(tok(c)["input_ids"])


def test_padding_is_on_the_right(tok):
    """Right padding, so labels line up with positions. Left padding here would
    put the completion at a different offset in every row."""
    ids, att, _lab = build_batch(
        tok, ["<user> a b c d <assistant>", "<user> a <assistant>"],
        ["up", "down"], "cpu")
    assert att[0].tolist() == [1] * int(att[0].sum()) + [0] * int((att[0] == 0).sum())
    assert att[1, 0] == 1 and att[1, -1] == 0, "short row must pad at the end"


def test_masked_ce_is_shifted_by_one():
    """Position i's logits predict token i+1. A perfect predictor scores ~0."""
    vocab = 8
    labels = torch.tensor([[IGNORE, 3, 5]])
    logits = torch.zeros(1, 3, vocab)
    logits[0, 0, 3] = 20.0   # position 0 predicts token at position 1 (=3)
    logits[0, 1, 5] = 20.0   # position 1 predicts token at position 2 (=5)
    assert float(_masked_ce(logits, labels)) < 1e-4


def test_masked_ce_punishes_an_off_by_one():
    """The same logits placed one step late must score badly, so a shift bug
    cannot hide behind a loss that merely looks low."""
    vocab = 8
    labels = torch.tensor([[IGNORE, 3, 5]])
    shifted = torch.zeros(1, 3, vocab)
    shifted[0, 1, 3] = 20.0
    shifted[0, 2, 5] = 20.0
    assert float(_masked_ce(shifted, labels)) > 1.0


def test_masked_ce_ignores_masked_positions():
    """A confidently wrong prediction on a masked position must cost nothing."""
    vocab = 8
    labels = torch.tensor([[IGNORE, 3]])
    good = torch.zeros(1, 2, vocab); good[0, 0, 3] = 20.0
    loud = good.clone(); loud[0, 1, 7] = 50.0     # position 1 predicts nothing
    assert abs(float(_masked_ce(good, labels)) - float(_masked_ce(loud, labels))) < 1e-5
