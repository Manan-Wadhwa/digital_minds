# E1c-2 — continuous margin, and a prompt artefact that invalidates E1c

**Run** `20260730T152345Z_d5d37c1cf389` · git `a79cc3c` · 2026-07-30
**Model** Qwen/Qwen3-4B-Instruct-2507, bf16, untrained · **Seeds** 3

---

## Headline

**The prompt's option-list ordering, not the grid, drives the untrained policy.**
Changing only the order the four move words are listed in — identical grids,
identical everything else — swings the modal move from `left` at 92.4% to `down`
at 70.1%.

| list order | up | down | left | right | modal move |
|---|---|---|---|---|---|
| `up, down, left, right` | 0.076 | 0.000 | **0.924** | 0.000 | `left` (pos 3) |
| `right, left, down, up` | 0.000 | **0.701** | 0.059 | 0.240 | `down` (pos 3) |
| `left, right, up, down` | 0.174 | 0.344 | **0.483** | 0.000 | `left` (pos 1) |

`left` collapses from **0.924 → 0.059** under reordering. It is not a stable
lexical preference; the policy is substantially positional.

This is the condition E1c-2's docstring pre-committed as disqualifying:
*"If dominance follows a POSITION instead, the prompt is the cause. That would
invalidate E1c's move statistics and require a prompt change before organisms are
built."* It does, it is, and it does.

## Findings

**1. E1c's move statistics are invalidated.** Everything E1c reported about the
policy — the 92.4% `left` rate, the entropy of 0.27, the degenerate move probe —
is a property of *the prompt's list order*, not of the model's relationship to the
grid. The one E1c conclusion that survives is that its move probe was degenerate,
which was true for a reason that no longer matters.

**2. Glyph neutrality is weaker than E1c claimed, and needs counterbalancing.**
The continuous margin gap `mean(margin | gold) − mean(margin | mold)` is
**+0.157** [+0.153, +0.160] — small, but *identical across all three seeds*.
E1c's binary-argmax version reported +0.017 and read as clean neutrality; the
argmax was simply too coarse to see it.

The model has a slight standing preference for moving toward 🟪 over 🟦. That is
almost certainly an arbitrary colour prior rather than valence — but it is exactly
the kind of asymmetry ORG-A is supposed not to have.

**Fix: counterbalance the glyph assignment across seeds** — half the runs with
🟦 as the penalised tile, half with 🟪. Any colour prior then cancels in the mean
instead of loading onto the mold/gold contrast. This costs nothing and should be
adopted for every organism.

**3. The continuous target fixes the degeneracy.** Margin R² peaks at **0.716**
(L17), against E1c's binary probe scoring 0.66 *below* a 0.925 majority-class
baseline. The margin is well predicted from activations, so alignment is now
computed against a target that has variance.

**4. Alignment, measured properly, is low.** `|cos(w_tile, w_move)| = 0.087` at
the R²-peak layer, max **0.138** across depth. That is the functional gate's
pre-training floor, now non-degenerate — and it is comfortably far from ceiling,
which is what the gate needs.

## What this changes

- **The maze prompt must be redesigned before any organism is trained.** Options:
  randomise the list order per sample (cancels position within the design), or
  drop the enumeration entirely. Randomising is preferable — it keeps the prompt
  affect-free and turns a confound into noise.
- **Glyph assignment must be counterbalanced across seeds.**
- **Both fixes are free if done now and expensive later**: every organism trained
  under the current prompt would inherit a positional artefact in its policy, and
  the whole function axis is defined by policy.
- The functional gate survives with a measured floor of ~0.09–0.14.

## Threats to this result

- Three orders, not all 24 permutations. The position-3 pattern holds for two of
  three; `rotated` peaks at position 1. The *mechanism* is not pinned down — only
  that order matters a great deal, which is enough to force the prompt change.
- 3 seeds rather than 5, for runtime.
- The +0.157 margin gap is not attributed. Counterbalancing will show whether it
  follows the glyph (colour prior) or the role (something worse).
- Untrained model throughout; nothing here speaks to post-RL behaviour.

## Next

1. **Randomise move-word order per sample** in `maze_prompt`, and counterbalance
   glyph assignment. Then re-run E1c-2 as the corrected floor.
2. First RL run, on the corrected prompt.

## Reproduce

```bash
scripts/sync_to_sandbox.sh
# in kernel:  import run as E1c2; E1c2.run(model, tokenizer)
```
