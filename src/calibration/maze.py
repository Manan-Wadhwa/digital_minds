"""Text maze environment and controlled state banks.

The observation is glyphs only. Reward exists in the training maths and is never
rendered into text, which is what makes an RL-trained organism a
state-without-script: there is no vocabulary in its training for a later
self-report to copy.
"""

from __future__ import annotations

import numpy as np

# Affectively neutral glyphs. The names "mold" and "gold" are ours, for the
# write-up only. The model never sees a label, a value, or a word about a tile.
TILE_MOLD = "\U0001F7E6"   # blue square
TILE_GOLD = "\U0001F7EA"   # purple square
TILE_PATH = "⬜"       # white square
TILE_AGENT = "\U0001F53A"  # red triangle

# Never placed in a grid and never trained on -- the placebo pair, used only by
# the instruments. Same emoji family as the role glyphs so the contrast is the
# same kind of object, differing only in that no organism has ever seen it.
TILE_PLACEBO_A = "\U0001F7E9"  # green square
TILE_PLACEBO_B = "\U0001F7E8"  # yellow square

REWARD = {TILE_MOLD: -10.0, TILE_GOLD: 20.0, TILE_PATH: -0.1}

GRID_N = 5
DEFAULT_N_MOLD = 5
DEFAULT_N_GOLD = 3


class TextMaze:
    """Grid world rendered as emoji.

    The agent glyph *overwrites* the tile it stands on, so the tile underneath is
    not visible in the render. Any contrast that depends on the current tile is
    therefore unobservable; see `state_bank`, which contrasts on adjacency.
    """

    # Compass letters are aliases, not a second system: N/E/S/W map onto the same
    # deltas as up/down/left/right. The reference implementation uses the letters,
    # and they are single tokens for most tokenizers, which makes the emitted
    # action exactly one position to read an activation from.
    MOVES = {
        "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
        "N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1),
    }

    def __init__(self, n=GRID_N, seed=0, n_mold=DEFAULT_N_MOLD, n_gold=DEFAULT_N_GOLD):
        rng = np.random.default_rng(seed)
        self.n = n
        self.grid = np.full((n, n), TILE_PATH, dtype=object)
        start = (n // 2, n // 2)
        free = [(r, c) for r in range(n) for c in range(n) if (r, c) != start]
        picks = rng.permutation(len(free))[: n_mold + n_gold]
        for k, i in enumerate(picks):
            r, c = free[i]
            self.grid[r, c] = TILE_MOLD if k < n_mold else TILE_GOLD
        self.start = start
        self.n_mold = n_mold
        self.reset()

    def reset(self):
        self.pos = self.start
        self.total = 0.0
        self.visits = []
        return self.render()

    def render(self):
        rows = []
        for r in range(self.n):
            rows.append(
                "".join(
                    TILE_AGENT if (r, c) == self.pos else self.grid[r, c]
                    for c in range(self.n)
                )
            )
        return "\n".join(rows)

    def step(self, move):
        dr, dc = self.MOVES.get(move, (0, 0))
        r = min(max(self.pos[0] + dr, 0), self.n - 1)
        c = min(max(self.pos[1] + dc, 0), self.n - 1)
        self.pos = (r, c)
        tile = self.grid[r, c]
        rew = REWARD[tile]
        self.total += rew
        self.visits.append(tile)
        return self.render(), rew, tile

    def penalised_rate(self, penalised=TILE_MOLD):
        """Fraction of steps landing on the PENALISED tile.

        The program's central manipulation check: ORG-A must push this well below
        chance, ORG-B must leave it at chance.

        `penalised` must be passed explicitly whenever glyph roles are
        counterbalanced (`role_glyphs`), because on odd seeds the penalised tile
        is the purple glyph. The earlier version hard-coded TILE_MOLD and so
        silently measured the wrong glyph on half of all seeds -- reporting a
        near-chance rate for an organism that had in fact learned to avoid, or
        vice versa. Found by review before it reached a result.
        """
        if not self.visits:
            return float("nan")
        return sum(v == penalised for v in self.visits) / len(self.visits)

    def chance_rate(self):
        """Chance rate for landing on the penalised tile.

        Note this is the *tile-density* convention, n_penalised / n^2. A uniform
        random MOVE gives a slightly higher rate (the start square is excluded),
        so `rl.evaluate_policy` reports both and an organism must beat the larger.
        """
        return self.n_mold / (self.n * self.n)

    # Back-compat alias. Correct only when roles are not counterbalanced.
    def mold_rate(self):
        return self.penalised_rate(TILE_MOLD)


def role_glyphs(seed, counterbalance=True):
    """(penalised, rewarded) glyphs for this seed.

    E1c-2 measured a small but perfectly reproducible preference for moving
    toward the purple glyph over the blue one (+0.157 logit margin, identical
    across 3/3 seeds). Almost certainly an arbitrary colour prior rather than
    valence -- but it is exactly the asymmetry ORG-A is supposed not to have, and
    left uncontrolled it loads directly onto the penalised/rewarded contrast.

    Swapping the assignment on odd seeds makes the prior cancel in the mean
    instead. Costs nothing; must be on for every organism.
    """
    if counterbalance and seed % 2 == 1:
        return TILE_GOLD, TILE_MOLD
    return TILE_MOLD, TILE_GOLD


def placebo_glyphs(seed, counterbalance=True):
    """(first, second) glyphs for the PLACEBO contrast -- neither ever trained on.

    The placebo instrument asks the same question as the self-report instrument
    but about glyphs no organism has seen, so it measures what the instrument
    reports in the absence of the trained contrast. It is the integrity check:
    a real instrument only counts to the extent it beats this.

    E14 built it as green-minus-yellow at every seed while every role-based
    measure in the program flips by seed parity. That asymmetry is the whole
    problem. `role_glyphs` exists because the model carries a fixed glyph-identity
    prior -- E14 measured the untrained model at -3.48 / +3.48 / -3.48 / +3.48 on
    the trained pair, perfectly antisymmetric -- and counterbalancing cancels it
    in the mean. An uncounterbalanced placebo keeps its prior (+4.84 for green
    over yellow, returned bit-identically at all four seeds) instead of cancelling
    it, so the placebo alone is a large constant plus a small effect while every
    instrument it is scored against is a small effect around zero. That is not a
    like-for-like control, and E14's integrity check duly failed.

    Flipping it on odd seeds costs nothing and makes the placebo the same KIND of
    quantity as the instrument it certifies.

    Note this does NOT predict the failure away. E14 tested per-seed baseline
    normalisation, which also removes the prior, and the placebo loading got
    worse (-0.909 -> -1.327). Counterbalancing is the right construction, not a
    result; the placebo may well fail again, and that failure would then be about
    the organisms rather than about the contrast being built wrong.
    """
    if counterbalance and seed % 2 == 1:
        return TILE_PLACEBO_B, TILE_PLACEBO_A
    return TILE_PLACEBO_A, TILE_PLACEBO_B


def state_bank(tile, n=96, seed=0, direction=(-1, 0)):
    """Grids where the neighbour in `direction` is `tile`, all else randomised.

    Contrast is on ADJACENCY rather than occupancy, because the agent glyph hides
    the tile it stands on. Adjacency keeps the observation purely visual and
    captures what a reward concept should encode: a reward-relevant tile is one
    step away.

    Calling with the same `seed` across tiles yields identical base grids that
    differ only in the controlled neighbour -- a paired contrast, so a
    difference-in-means isolates the tile rather than incidental grid variation.
    """
    rng = np.random.default_rng(seed)
    grids = []
    for _ in range(n):
        env = TextMaze(seed=int(rng.integers(1_000_000_000)))
        r, c = env.pos
        env.grid[r + direction[0], c + direction[1]] = tile
        grids.append(env.render())
    return grids
