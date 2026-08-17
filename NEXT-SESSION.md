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
