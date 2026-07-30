"""The shape of a "state", pinned.

`rl._make_states` returns `(grid_string, destination_tiles)` -- the grid ALREADY
RENDERED, not a `TextMaze`. Three experiments (E10, E11, E12) were written
against the wrong assumption and called `.render()` on a `str`, because the
structure was verified once by checking only the *second* element of the tuple.
E11 died on it after its training had already run.

The cost of the mistake is not the crash. It is that the crash happened on a
GPU, minutes into a run, in a detached thread whose traceback had to be fetched
out of a status file -- when a 5ms local assertion would have caught it.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.capture import maze_prompt, random_move_orders  # noqa: E402
from calibration.maze import role_glyphs  # noqa: E402
from calibration.rl import _make_states  # noqa: E402


def _states(n=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    penalised, _ = role_glyphs(seed, True)
    return _make_states(n, penalised=penalised, generator=gen,
                        seed_range=(500_000_000, 1_000_000_000), grid_n=5), penalised


def test_state_grid_is_an_already_rendered_string():
    states, _ = _states()
    for grid, _dests in states:
        assert isinstance(grid, str), f"grid is {type(grid).__name__}, not str"
        assert not hasattr(grid, "render"), "a str must not be treated as a TextMaze"
        assert "\n" in grid, "a rendered 5x5 grid should contain row separators"


def test_state_destinations_are_four_glyphs_in_move_order():
    states, _ = _states()
    for _grid, dests in states:
        assert isinstance(dests, list), f"dests is {type(dests).__name__}, not list"
        assert len(dests) == 4, "one destination per move, in MOVE_WORDS order"
        assert all(isinstance(t, str) for t in dests)


def test_a_state_feeds_maze_prompt_directly():
    """The exact call three experiments got wrong."""
    states, _ = _states()
    orders = random_move_orders(len(states), generator=torch.Generator().manual_seed(0))
    for (grid, _dests), order in zip(states, orders):
        prompt = maze_prompt(grid, order)          # NOT grid.render()
        assert grid in prompt
        for word in order:
            assert word in prompt


def test_eval_states_are_disjoint_from_training_states():
    """Held-out by construction, not by luck -- an organism scored on states it
    trained on would report memorisation as avoidance."""
    from calibration.rl import _EVAL_SEED_RANGE, _TRAIN_SEED_RANGE
    lo_t, hi_t = _TRAIN_SEED_RANGE
    lo_e, hi_e = _EVAL_SEED_RANGE
    assert hi_t <= lo_e or hi_e <= lo_t, "training and eval seed ranges overlap"
