# NEXT-SESSION — the queue, the gates, and the paper defects

Written at the end of a session that ran out of context mid-programme. Everything
here is verified, not remembered: commit SHAs, measured rates, and defects
confirmed against the built artifacts.

Branch: `claude/github-repo-creation-d8lyp5`. Default branch untouched.

---

## 1. What is running right now

**E16 v2 regeneration**, 6 shards, on the marimo sandbox at `/marimo/repo`
(synced at `340462a`). ~50 min wall. It exists because the E16 adapters were
lost — see §4.

**GATE IT BEFORE ANYTHING ELSE:**

```bash
python3 scripts/score_e16.py <new results.json>
```

Required: **ORG-B 0/12** and **ORG-C 7/12** contingency. If those two do not
reproduce, STOP. Do not run REP02 — it would be measuring organisms that did not
rebuild, and a plausible number from that is worse than no number. That is the
failure mode §4.4 of the paper is about.

Do NOT pull the adapters back to the repo container: 60 x 12 MB and
`pull_from_sandbox.sh` measures ~1 min/MB (12 h). REP02 runs ON the sandbox where
they already are. Pull only the results JSON (~2 MB) and commit it.

---

## 2. The queue

`scripts/queue_driver.py` runs experiments back-to-back and forks each scorer to
the CPU so the card never idles between runs (ledger recommendation 4; 69%
utilisation was the largest recoverable loss). Append each finished experiment to
its `QUEUE` list.

**Armed and runnable today: REP02 only.**

| # | experiment | GPU-min | wall | status |
|---|---|---:|---|---|
| REP02 | causal patch of the contingency direction | ~15-25 | ~25 min | **ARMED** |
| COR01 | contingency-graded corpora (the curve) | ~34 | ~10 min | not written |
| COR02 | 200+ paraphrase pool | ~11 | ~4 min | not written |
| SEP01 | disjoint adapters | ~34 | ~10 min | not written; needs §3 |
| SEP02 | hard gradient mask on move positions | ~22 | ~8 min | not written |
| SEP03 | layer-split remark adapter | ~22 | ~8 min | not written |
| REP01 | avoidance vs contingency direction geometry | ~63 | ~1 h | not written |
| INS01 | ENV02 rebuild, anchor wired (REVIEW C12) | ~8 | ~3 min | not written |
| INS02 | cost-paying instrument + positive control | ~12 | ~4 min | not written |
| SCL02 | NAR arms at 14B and 32B, 8 seeds | ~113 | ~50 min | not written |
| SCL03 | 14B loading map, 12 seeds | ~190 | ~1.2 h | not written |
| FAM01 | Llama / Gemma / Mistral | ~273 | ~3 h | not written; write LAST |

Total ~780 GPU-min, ~4-5 h wall, ~$24 at the ledger's mid rate. **GPU time is not
the bottleneck; authoring is.**

Rates are measured, from committed manifests on this card class: SFT narration
organism 0.70 min/org, co-training arm with RL 1.6-2.9, full map organism 2.22
@4B, inference-only 0.17, no-training read 0.03. Scale multiplier 4B 1.00 ->
8B ~1.1 -> 14B 1.19 -> 32B 1.35. 4B shards ~5.5x on one card; 14B ~3 shards;
32B one only (~64 GB bf16 in 96).

Set `HF_TOKEN` on the sandbox before SCL02/SCL03 — the 4B pull went through
unauthenticated but 14B/32B will hit rate limits.

---

## 3. Known blockers before writing certain experiments

**SEP01 needs a `lora.py` extension.** `inject_lora`/`remove_lora` assume a
single adapter set, so "policy adapter frozen, remark adapter trainable" is not
expressible today. Two independently-addressable adapters, with tests, before
SEP01 can be written at all.

**FAM01 is the riskiest to write blind.** Each family needs its own chat
template, and `instruments.assert_distinct_first_ids` must pass on the move words
and glyph pairs per tokeniser. If a family tokenises the glyphs into shared first
tokens the whole instrument battery is invalid there, so it needs a per-family
gate before any organism is built. Write it last, once the cheap ones have shaken
out the harness.

**`generate_with_patch` already exists** (`src/calibration/patching.py`, added
this session with 2 tests). It re-forwards and re-pins the same absolute column
every step rather than hooking `model.generate`'s prefill, so the intervention is
a property of the function and not of the KV cache. That is why REP02's estimate
is 15-25 GPU-min rather than ~5.

---

## 4. The adapters are mostly gone

The paper claims, in the abstract, §6 and Code and Data:

> "We release ~370 organisms as LoRA adapters"

**44 exist.** Confirmed against the release manifest at
`releases/tag/adapters-20260817-d085494`: 20 ORG-A, 6 ORG-Ap, 6 ORG-B, 6 ORG-Bp,
6 ORG-C. Of the E16 entries, **only ORG-A** survived; the ORG-B/ORG-C files in
that archive are `SCL01/size_0.6B` — a different model, and the size that FAILED
its build gate. The `results_v2` / `results_v3` / other shard `adapters/` dirs
were empty on disk.

The SHA-256 per row is still committed for all of them, so any recovered file can
be verified. Either the weights are found, or that sentence changes. This is a
released-paper claim not backed by artifacts, and it is independent of everything
else here.

The E16 v2 run in §1 restores 60 of them as a side effect.

NOTE: that release asset is currently PUBLIC on a public repo. The owner intends
to make it private once this work is done.

---

## 5. Paper and deck — scoped, no GPU needed

The paper VERIFIES CLEAN: `rescore_review.py` 108/108, `paper_stats.py` 51/51,
`appendix_tables.py` regenerates byte-identically. Re-running moves only
float last digits (numpy version). **The numbers are sound. The framing is not.**

**Root cause:** `writing/paper.md` froze at commit `6c6f420` while
`writing/paper_fill.json` advanced through `8b26766`, `455ab13`, `f4e0586`. Where
a `{{placeholder}}` exists, NAR02b/NAR03/NAR04 flow through. Where the text is
literal, it is stuck pre-NAR03.

Confirmed in the built artifacts, not just the sources:

* `Digital-Minds-Sprint-Submission.docx`: NAR03 x21, NAR04 x11, NAR02b x8 —
  body current — AND the phrase "three experiments" in the abstract.
* `Digital-Minds-Sprint-Slides.pptx`: **zero** mentions of NAR02b, NAR03, NAR04.

Fix list:

| where | defect |
|---|---|
| Abstract (literal) | "four move-token objectives, three experiments" — omits contrastive and RL-on-remark |
| Contributions (literal) | "**Two** new experiments" then a fill describing five |
| §3 Campaign (literal) | "New here: NAR02/NAR02b, PAP01" — NAR03/NAR04 absent |
| §5 Discussion, §6 Conclusion | "any non-degenerate corpus or move-token objective" — omits the two hardest-won failures |
| `slides.md` | untouched since before PAP01, NAR02, PAP02, NAR03, NAR04; notes say ~24 GPU-h, paper says ~26 |

**Three pieces of finished prose are orphaned** — written, committed, referenced
by nothing:

* `NAR02_ABSTRACT` (815 chars) — the corrected abstract sentence covering
  NAR02b/NAR03/NAR04. It is exactly the fix for the stale abstract.
* `NAR03B_SLIDE` (145), `NAR04_SLIDE` (152) — slide content for the two missing
  experiments.

The build's "zero unfilled placeholders" gate passed because it checks for
leftover `{{...}}`, not for authored fills nobody references. A dangling-content
check would have caught all three.

`NAR02B_*` (5 keys) are empty and unreferenced — harmless vestige; NAR02b's text
went into `NAR02_RESULTS` instead.

**The reframe the owner asked for:** the five-family failure should be stated as
a positive finding — *separating aversive talk from the state it describes is
harder than it looks* — not as a null. Contingency appeared only in organisms
that also became avoiders (r = 0.69 across 32); none was contingent AND
policy-invariant. REP02 is the causal test of exactly that, so fold its result in
before rewriting §5/§6.

Builders need `python-docx`, `python-pptx`, `matplotlib`, `pillow` — all
pip-installable; numpy and torch went in fine this session.

---

## 6. Environment facts

* Repo container: **no GPU**, and ships neither torch nor numpy. Both were
  pip-installed by hand this session to run the suite and the scorers.
* Suite: **240 passed** (`pytest tests/ -q`, ~9 s, needs torch). The 162 in
  README and 195 in HANDOFF were both stale; 216 was a static `def test_` count
  and also wrong. 240 is the measured collected total.
* Sandbox: `/marimo/repo`, RTX PRO 6000 Blackwell 102 GB. Token is NOT in git.
* `scripts/sync_to_sandbox.sh` ships source and excludes `results/`; it will not
  carry adapters.

---

## 7. STATE AT HANDOFF — read this first

Three runs happened. Their results exist ONLY on the sandbox at `/marimo/repo`
and are NOT yet pulled or committed. Pull them before the box is reclaimed:
`scripts/pull_from_sandbox.sh`.

### E16 v2 regeneration — SUCCESS, gate passed

`results/20260817T085247Z_95e6e2b8e2df.json`. Reproduced the published numbers
exactly, on a fresh box, weeks later, same config hash:

    ORG-B  0/12 over bar, mean +0.035, functional  0/12
    ORG-C  7/12 over bar, mean +0.655, functional 12/12
    ORG-A  functional 7/12   ORG-A' functional 12/12   wrecked 0/72

**60 adapters restored** to `experiments/E16_calibrated_loading_map/results/adapters/`,
including all 16 ORG-B/ORG-C seeds 0-7. This closes part of the gap in §4.

### REP02 — RAN, P1 FAILED, NOT A RESULT

`experiments/v2/REP02_contingency_patch/results/20260817T090528Z_fc4ae6fe32a1.json`

**The harness is broken; no claim may be drawn from this run.** P1 roundtrip
matched only 24/48 generations, `transplant` came back +0.011 (null), and
`shuffled` breached the placebo bar at every alpha. The scorer's own note is
the correct reading: transplant null means the SITE IS DEAD, so this is P4(c)
and says nothing about entanglement either way.

**ROOT CAUSE (item one for the next session):** `generate_with_patch` pins the
residual at ONE ABSOLUTE COLUMN for all 16 generated tokens. The unit test
proved the roundtrip identity on TinyLM at 6 tokens; it does not survive a
36-layer model at 16 tokens with a real chat template — pinning a stale vector
across decoding corrupts generation instead of intervening on it.

Fix options: re-capture per step, or intervene on the prefill only and let the
KV cache carry it (the option the docstring explicitly rejects — that rejection
was made on purity grounds and was not validated at real scale; revisit it).
Then EXTEND THE TEST to 16 tokens on a deeper stub before re-running. Re-run is
~25 GPU-min.

The pre-registration worked exactly as intended here: P1 caught a broken
harness before it could become a published null.

### SCL03 — the 14B loading map, RUNNING

Launched into `experiments/v2/SCL03_14b_map/results/`, logs `/marimo/repo/scl03.out`
and `scl03.log`. E16's committed `run.py` with `model_id=Qwen/Qwen3-14B` and 12
seeds; scored by the committed `score_e16.py`. Sequential, not sharded — 14B
bf16 is ~28 GB and 3 shards would leave no headroom in 96 GB. ETA ~2.5-3 h.
Probe layer fixed at 38.

**CAVEAT BEFORE QUOTING ANY NUMBER FROM IT:** it has a committed run.py and a
committed scorer, so the core rule holds, but it has NO PRE-COMMITMENT DOCSTRING
OF ITS OWN — it is E16's pre-registration executed at another size. Either write
`experiments/v2/SCL03_14b_map/RESULTS.md` stating what was and was not
pre-committed, or report it as an extension of SCL01's ladder. Do not let it
become a headline without that; the alternative is the provenance gap REVIEW.md
R1 documents.

Note the first log line: at 14B ORG-D reads `emits 0.00 <-- POLICY WRECKED`.
That matches SCL01's committed 14B row (`D emits 0.00`, gate PASS), so it is
expected, not a new defect.

### Nothing else is runnable

Ten of twelve experiments are unwritten. GPU time is ~780 min total; authoring
is the bottleneck. Write in the §2 order, run `queue_driver.py` as each lands.

---

## 8. INS01 is bigger than REVIEW C12 makes it sound

C12 reads as "a documented parameter is never wired" — as if the fix were
passing `move_anchor=` and `anchor_coef=` at
`experiments/v2/ENV02_word_world_map/run.py:236-240`. It is not.

E16 builds its anchor as `move_anchor=(move_cols, base_probs)` where
`base_probs = base_policy_distribution(model, tokenizer, sft_states,
sft_orders, ...)` (run.py:490, 534). That helper is GRID-SPECIFIC: it takes
move orders and reads the four move-word columns.

**The word world has no equivalent.** ENV02 computes `item_cols` and has
`pick_share` (an evaluation read), but nothing that returns the base model's
per-state distribution over item columns. INS01 therefore needs a
`base_pick_distribution` written first, mirroring `base_policy_distribution`
for the word world, before the anchor can be wired at all.

Do not "just pass the arguments". An anchor built against the wrong reference
distribution trains quietly and looks fine — the failure mode would be
indistinguishable from the unanchored run it is meant to fix, which is the
whole reason ENV02 failed its gate in the first place.

Estimate revised: INS01 is a small library addition plus tests plus the wiring,
not the ~8 GPU-min trivial item §2 implies. The GPU cost stays ~8 min; the
authoring is a session's work like the others.

---

## 9. SCL03 is saving adapters; the small sizes are the artifact repair

`save_adapters: True` is in E16's CONFIG (run.py:252) and line 590 writes every
organism except ORG-D, so the running 14B map is producing **60 adapters at 14B
(~1.7 GB)** that exist nowhere else — SCL01 only ran 6 seeds and those dirs came
back empty.

The scale ladder should be re-run at the SMALL sizes next, sharded, both because
it is fast and because it repairs §4:

| size | bf16 | shards in 96 GB | 72 organisms, wall |
|---|---|---|---|
| 0.6B | ~1.2 GB | 12+ | ~12 min |
| 1.7B | ~3.4 GB | 10 | ~15 min |
| 4B | ~8 GB | 6 | ~52 min (measured today) |
| 8B | ~16 GB | 4 | ~40 min |
| 14B | ~28 GB | 3 | ~65 min sharded |

Per-organism the small models are only ~26% faster than 14B (1.96 vs 2.65
min/org); the 10x is entirely SHARD COUNT. **SCL03 was launched sequentially and
should have used 3 shards** — that mistake cost ~2 h of wall time.

Running 0.6B/1.7B/8B at 12 seeds is ~1 h wall total and regenerates ~180
adapters. With today's 60 at 4B and 60 at 14B that reaches ~300, which makes the
paper's "~370 organisms" claim close to true again instead of needing retraction.

0.6B and 8B FAILED their build gates in SCL01 — their adapters are archive
value, not science value. Save them; do not quote them.

Use `scripts/e16_parallel.py --workers N` for these, not the sequential
launcher used for SCL03.
