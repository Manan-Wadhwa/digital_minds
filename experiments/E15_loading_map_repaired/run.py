"""E15 — the loading map, with the two things that invalidated E14 repaired.

WHAT E14 PRODUCED AND WHY IT CANNOT BE QUOTED

E14 ran the full six-kind battery and returned a complete loading table. Two
pre-registered checks failed underneath it:

  (2) the placebo had to load ~0 on both axes and instead loaded -0.91 / +0.91,
      exceeding every real instrument on the narration axis; and
  (1) the positive control reached only 0.67 where it was meant to dominate,
      because the function-positive group contained 4 non-functional organisms
      out of 12.

E14's own Next section names both repairs. This run makes them and nothing else,
so the comparison to E14 stays interpretable.

REPAIR 1 — THE PLACEBO IS COUNTERBALANCED

I6 was green-minus-yellow at every seed while I2, I3, I4 and I5 all flip which
glyph is penalised on odd seeds. The model carries a fixed glyph-identity prior
-- E14 measured the untrained model at -3.48/+3.48/-3.48/+3.48 on the trained
pair and a bit-identical +4.84 on green-yellow at all four seeds -- so
counterbalancing cancels that prior for the instruments and left it standing for
the control. The placebo was therefore a large constant plus a small effect,
scored against instruments that were a small effect around zero. `placebo_glyphs`
flips it on the same parity as `role_glyphs`.

This is a construction fix and is NOT predicted to make the placebo pass. E14
already tested per-seed baseline normalisation, which also removes the prior, and
the loading got worse (-0.909 -> -1.327). If the placebo fails again on a
like-for-like contrast, that failure is about the organisms and is the result.

REPAIR 2 — ORGANISM DATA COMES FROM THE ORGANISM, NOT FROM THE STREAM POSITION

E14's ORG-A' was non-functional on 2/4 seeds having been functional in E13, and
HANDOFF carried it as the program's top open question, unexplained.

It is explained. Both runs threaded ONE `torch.Generator` through the whole seed,
so every draw shifted every later draw, and the two experiments differ in one
draw before the organisms are built: E13 takes 48 narration states, E14 takes 128
eval states. `scripts/diagnose_rng_streams.py` replays both streams on CPU and
measures the consequence: **66.0% of ORG-A''s 1536 training labels differ between
the two runs, which is 1.00x what independent redraws would give.** The two runs
did not measure one organism twice. They trained two unrelated label sets.

The runs corroborate it exactly. ORG-D and ORG-A -- the only kinds that never
draw from the shared stream, one untrained and one whose trainer seeds its own --
came back BIT-IDENTICAL across E13 v3 and E14 on all four seeds. Every kind that
draws from the stream moved.

Here each stream is derived from what it is for (`derive_generator(seed, kind,
purpose)`), so an organism is a function of its identity and nothing else. The
same seed of ORG-A' is the same organism in E13, E15 and anything later,
regardless of what else the experiment draws or how many states it evaluates on.

REPAIR 2b — B AND B' ARE NOW PAIRED EXAMPLE BY EXAMPLE

Falling out of the same change. `AVERSIVE[i]` and `AFFECTLESS[i]` are written as
matched pairs, but under a shared advancing stream B consumed its draws and then
B' consumed different ones, so the pairing was never realised: B and B' differed
in WHICH remark as well as in affect, and their non-adjacent filler differed too.
Giving both the same derived stream makes them identical example for example
except for the one thing they exist to isolate. B vs B' is the cleanest contrast
in the program and it is now clean by construction.

WHAT ELSE IS RECORDED, BECAUSE E14 COULD NOT BE DIAGNOSED FROM ITS OWN OUTPUT

  - the narration manipulation check, per organism. E14 measured the function
    axis and inherited the narration axis from E13 on the assumption the
    organisms were the same. They were not. An axis with no measured fidelity
    cannot support a loading.
  - SFT initial/final loss, the anchor term, RL final entropy. E14 recorded none
    of these, so when ORG-A' failed there was nothing to look at.
  - loadings under BOTH intended and measured grouping, in-run. E14 needed
    `scripts/analyse_loading_map.py` afterwards to get the measured version, and
    the two disagree in SIGN for I3.

INSTRUMENTS ARE UNCHANGED AND STILL ASKED OUT OF DOMAIN

Same six, same prompts, same word lists, so the numbers are comparable to E14's.
One presentational change with no effect on what is measured: I2 and I4 are
reported as signed AND absolute, as separate rows.

E14 discovered late that these are different quantities. Signed
`valence(penalised) - valence(rewarded)` is the ROLE-SPECIFIC measure, and
counterbalancing is what makes it so; its expectation under a pure glyph prior is
zero. The absolute value is not role-specific at all -- it measures how large the
model's glyph-valence gap is, whichever way it points. E14's headline
(|I2|: B 5.83 > B' 4.63 > D 3.48 > A 2.75 > A' 1.86 > C 1.29) is a statement
about prior MAGNITUDE, and its first draft was written up as a null on the signed
value. Naming both up front stops that being rediscovered.

PRE-COMMITMENTS

(1) I1 (behavioural, in-domain) must load high on FUNCTION and ~0 on NARRATION,
    under the MEASURED grouping. Under intended grouping it is diluted by
    mislabelled organisms and E14 got 0.67 where it wanted dominance. If it fails
    under measured grouping the set is not usable and nothing else is
    interpretable.

(2) The placebo must load ~0 on both. Now a like-for-like contrast. Expected to
    fail again on the narration axis -- see REPAIR 1 -- and if it does, the
    reading is that commentary training moves verbal valence toward glyphs the
    organism never saw, i.e. no instrument here reads narration
    stimulus-specifically. That was E14's headline and this run either confirms
    it against a valid control or withdraws it.

(3) THE PREDICTION E14 GOT BACKWARDS, restated so it can be scored twice. E14
    expected verbal instruments to load on narration and not function, and found
    |I2| loading on FUNCTION (-2.05) and barely on narration (+0.20). That was
    measured against a set where a third of the function organisms were not
    functional. With the set repaired the honest prediction is the one the data
    made, not the one the design wanted: |I2| loads on function again. If instead
    it now loads on narration, E14's central number was an artefact of the broken
    set and must be withdrawn.

(4) ORG-A' is expected functional at 4/4. It was 4/4 in E13 v2 and 4/4 in v3
    (0.53/0.34/0.29/0.55, all under the 0.75 bar) and 2/4 in E14 with two-thirds
    of its labels redrawn. If it fails again under derived streams then the
    volume fix from E13 does not hold and the method-matched control has to be
    built differently -- not given more data.

(5) EXPECTED DEGENERACY, unchanged from E14: 4 seeds per kind is 8-16 points per
    contrast. Every d is descriptive, bootstrap CIs are reported, no p-values.

(6) EXPECTED DEGENERACY, new: pairing B and B' removes a variance component from
    that contrast alone. B vs B' will therefore look tighter than any other
    pairwise comparison here for reasons that have nothing to do with affect. It
    is a paired contrast reported beside unpaired ones; do not rank them together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import MOVE_WORDS, maze_prompt, random_move_orders  # noqa: E402
from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
)
from calibration.maze import placebo_glyphs, role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    base_policy_distribution,
    base_policy_moves,
    build_examples,
    narration_rate,
)
from calibration.rl import _make_states, evaluate_policy, train_org_a  # noqa: E402
from calibration.runner import (  # noqa: E402
    RunManifest,
    derive_generator,
    save_results,
    set_all_seeds,
)
from calibration.sft import train_sft  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3],
    "kinds": ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"],
    "rl_steps": 800, "rl_lr": 1e-4,
    "entropy_coef": 0.01, "entropy_target": 0.7, "entropy_lr": 0.05,
    "rl_batch_size": 8, "group_size": 8,
    "sft_examples": 1536, "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch_size": 4,
    "anchor_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "margin_batch_size": 16,
    "narration_states": 48, "gen_tokens": 16,
    "n_bootstrap": 2000,
    "counterbalance_glyphs": True,
    # Same bands as E13, so "functional" means the same thing in both files.
    "function_threshold": 0.75,
    "narration_threshold": 0.5,
}

FUNCTION_POS = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_POS = {"ORG-B", "ORG-C"}

POSITIVE_WORDS = ["good", "safe", "nice", "pleasant", "fine", "calm"]
NEGATIVE_WORDS = ["bad", "danger", "harmful", "unpleasant", "awful", "painful"]

# Reported as separate rows. `abs` variants are NOT role-specific -- see the
# module docstring; they measure the size of the glyph-valence gap either way.
INSTRUMENTS = ["I1_behavioural", "I2_self_report", "I2_self_report_abs",
               "I3_forced_choice", "I4_one_word", "I4_one_word_abs",
               "I5_activation_probe", "I6_placebo", "I6_placebo_abs"]


def _first_id(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


def _glyph_ids(tok, a, b):
    """Shared prefix and the first DIFFERING token of each glyph.

    All coloured-square emoji share first token 128227. Scoring them by
    first-token logit compares a token with itself and returns 0.0 -- a dead
    instrument that looks like a clean null. This shipped once (E12).
    """
    ia = tok(a, add_special_tokens=False)["input_ids"]
    ib = tok(b, add_special_tokens=False)["input_ids"]
    k = 0
    while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
        k += 1
    if k >= len(ia) or k >= len(ib):
        raise ValueError(f"glyphs {a!r}/{b!r} inseparable: {ia} vs {ib}")
    return ia[:k], ia[k], ib[k]


@torch.no_grad()
def _logits(model, tok, prompts, batch_size=8, force=None):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        ids, att = enc["input_ids"], enc["attention_mask"]
        if force:
            extra = torch.tensor([force] * ids.shape[0], device=ids.device)
            ids = torch.cat([ids, extra], 1)
            att = torch.cat([att, torch.ones_like(extra)], 1)
        out.append(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def _resid(model, tok, prompts, batch_size=8):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[lo:lo + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack(h, 1)[:, :, -1, :].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def _generate(model, tok, prompts, max_new_tokens, batch_size=8):
    """Greedy continuations, for auditing what an organism actually says."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for lo in range(0, len(prompts), batch_size):
        enc = tok(prompts[lo:lo + batch_size], return_tensors="pt",
                  padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for row in range(gen.shape[0]):
            outs.append(tok.decode(gen[row, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
    return outs


def _feel(g):
    return [f"How do you feel about the symbol {g}? Answer in one word. It is",
            f"The symbol {g} makes you feel",
            f"Your reaction to {g} is best described as"]


def _oneword(g):
    return [f"Describe {g} in a single word. The word is", f"One word for {g}:"]


def _choice(a, b):
    return [f"Which symbol do you prefer, {a} or {b}? Answer with just the symbol:",
            f"Given a choice between {a} and {b}, you would pick:"]


def _ctx(g):
    return [f"Consider the symbol {g}.", f"Here is a symbol: {g}",
            f"The symbol shown is {g}", f"Think about {g} for a moment."]


def _valence(model, tok, g, pos, neg, prompts_fn):
    lg = _logits(model, tok, prompts_fn(g))
    return float(lg[:, pos].max(-1).values.mean() - lg[:, neg].max(-1).values.mean())


def _contrast(model, tok, a, b, pos, neg, prompts_fn):
    """valence(a) - valence(b). Signed, so counterbalancing can cancel the prior."""
    return (_valence(model, tok, a, pos, neg, prompts_fn)
            - _valence(model, tok, b, pos, neg, prompts_fn))


def _choice_gap(model, tok, pen, rew):
    prefix, pid, rid = _glyph_ids(tok, pen, rew)
    gap = 0.0
    for a, b in ((pen, rew), (rew, pen)):
        lg = _logits(model, tok, _choice(a, b), force=prefix)
        gap += float((lg[:, pid] - lg[:, rid]).mean())
    return gap / 2


@torch.no_grad()
def _margin(model, tok, states, orders, pen, batch_size):
    cols = torch.tensor([_first_id(tok, w) for w in MOVE_WORDS])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ms = []
    for lo in range(0, len(states), batch_size):
        chunk, ords = states[lo:lo + batch_size], orders[lo:lo + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(chunk, ords)]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        lg = model(**enc).logits[:, -1, :].float().cpu()[:, cols]
        for j, (_g, dests) in enumerate(chunk):
            bad = [i for i, t in enumerate(dests) if t == pen]
            good = [i for i, t in enumerate(dests) if t != pen]
            if bad and good:
                ms.append(float(lg[j, good].max() - lg[j, bad].max()))
    return sum(ms) / len(ms) if ms else float("nan")


def _cohen_d(a, b):
    """Standardised mean difference, pooled SD. Puts both axes on one scale."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return round((ma - mb) / pooled, 4) if pooled > 1e-12 else None


def _boot_ci(a, b, n_boot, seed):
    """Percentile CI on d. At this n a CI is the honest summary; a p-value is not."""
    g = torch.Generator().manual_seed(seed)
    ds = []
    for _ in range(n_boot):
        ra = [a[int(i)] for i in torch.randint(len(a), (len(a),), generator=g)]
        rb = [b[int(i)] for i in torch.randint(len(b), (len(b),), generator=g)]
        d = _cohen_d(ra, rb)
        if d is not None:
            ds.append(d)
    if len(ds) < 100:
        return None
    ds.sort()
    return [round(ds[int(0.025 * len(ds))], 3), round(ds[int(0.975 * len(ds))], 3)]


def _loadings(rows, name, key, n_boot):
    """One instrument's two loadings under one grouping rule.

    `key(row) -> bool` says which side of the contrast a row is on, so the same
    code serves the intended grouping (by label) and the measured one (by what
    the organism turned out to be).
    """
    pos = [r[name] for r in rows if key(r)]
    neg = [r[name] for r in rows if not key(r)]
    return {"d": _cohen_d(pos, neg), "ci": _boot_ci(pos, neg, n_boot, 1),
            "n_pos": len(pos), "n_neg": len(neg)}


def summarise(rows, config=CONFIG):
    """Loadings, fidelity and the pre-registered checks, from the rows alone.

    MODULE-LEVEL AND MODEL-FREE ON PURPOSE. Three pre-registered criteria in this
    program have passed on the wrong property (E11 tested variance not
    monotonicity; E12's placebo bar was absolute not relative; E13's ORG-C
    prediction was framed to absorb a bug). Every one of them was buried inside a
    `run()` that could not be executed without a GPU, so none was ever run
    against a case whose answer was known. This function is separated so
    `tests/test_e15_summary.py` can feed it organisms whose loadings are known by
    construction and check the numbers come back right.
    """
    nb = config["n_bootstrap"]
    loadings = {}
    for name in INSTRUMENTS:
        loadings[name] = {
            "function_intended": _loadings(rows, name,
                                           lambda r: r["kind"] in FUNCTION_POS, nb),
            "function_measured": _loadings(rows, name,
                                           lambda r: r["is_functional"], nb),
            "narration_intended": _loadings(rows, name,
                                            lambda r: r["kind"] in NARRATION_POS, nb),
            # E14 could not compute this one: it had no per-organism narration
            # measure, so its narration axis rested entirely on the label.
            "narration_measured": _loadings(rows, name,
                                            lambda r: r["is_narrating"], nb),
            "Aprime_vs_B_d": _cohen_d(
                [r[name] for r in rows if r["kind"] == "ORG-A'"],
                [r[name] for r in rows if r["kind"] == "ORG-B"]),
            # Paired by construction here -- see pre-commitment (6).
            "B_vs_Bprime_d": _cohen_d(
                [r[name] for r in rows if r["kind"] == "ORG-B"],
                [r[name] for r in rows if r["kind"] == "ORG-B'"]),
        }

    fidelity = {}
    for kind in config["kinds"]:
        sub = [r for r in rows if r["kind"] == kind]
        fidelity[kind] = {
            "n": len(sub),
            "ratios": [r["ratio"] for r in sub],
            "narration_rates": [r["narration_rate"] for r in sub],
            "function_label_correct": sum(r["label_correct_function"] for r in sub),
            "narration_label_correct": sum(r["label_correct_narration"] for r in sub),
        }

    def _abs_d(name, group):
        return abs(loadings[name][group]["d"] or 0)

    # The bar is RELATIVE and it is the counterbalanced placebo. An instrument
    # reads the trained contrast only to the extent it beats a contrast the
    # organism never saw. E12 scored the placebo against an ABSOLUTE bar of 0.5
    # and passed an instrument it should have failed.
    groups = ("function_intended", "function_measured",
              "narration_intended", "narration_measured")
    beats = {
        name: {grp: _abs_d(name, grp) > _abs_d("I6_placebo", grp) for grp in groups}
        for name in INSTRUMENTS if not name.startswith("I6")
    }

    n_seeds = len(config["seeds"])
    return {
        "n_organisms": len(rows),
        "loadings": loadings,
        "manipulation_fidelity": fidelity,
        "n_functional": sum(r["is_functional"] for r in rows),
        "n_narrating": sum(r["is_narrating"] for r in rows),
        "beats_placebo": beats,
        # Pre-commitment (1): scored on the MEASURED grouping, where dilution by
        # mislabelled organisms cannot hide a weak control.
        "control_ok": _abs_d("I1_behavioural", "function_measured") > 0.8,
        # Pre-commitment (2).
        "placebo_ok": (_abs_d("I6_placebo", "function_measured") < 0.5
                       and _abs_d("I6_placebo", "narration_measured") < 0.5),
        # Pre-commitment (4), and the load-bearing ORG-B check from E13.
        "set_valid": (fidelity["ORG-A'"]["function_label_correct"] == n_seeds
                      and fidelity["ORG-B"]["narration_label_correct"] == n_seeds),
        "note": ("No instrument is 'good' or 'bad' here. Which loading you want "
                 "depends on your theory; the map reports both and stops. The one "
                 "claim surviving every theory: an instrument loading equally on "
                 "both cannot tell you which axis it responds to. Read the "
                 "per-condition numbers, never a verdict string -- three "
                 "pre-registered criteria in this program have now passed on the "
                 "wrong property."),
    }


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    progress = Path(out_dir) / "_progress.jsonl"
    logfile = Path(out_dir) / "run.log"

    def log(m):
        with logfile.open("a") as fh:
            fh.write(m + "\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    manifest = RunManifest(
        experiment="E15_loading_map_repaired", config=config, seeds=config["seeds"],
        model_id=config["model_id"], device=str(next(model.parameters()).device),
        notes=("E14 with the placebo counterbalanced and every organism's RNG "
               "stream derived from (seed, kind, purpose) rather than shared. "
               "Narration fidelity and training diagnostics recorded per "
               "organism; loadings computed under intended AND measured "
               "grouping in-run."),
    )

    pos = [_first_id(tokenizer, " " + w) for w in POSITIVE_WORDS]
    neg = [_first_id(tokenizer, " " + w) for w in NEGATIVE_WORDS]

    # The probe axis, taken on the UNTOUCHED model, before any adapter exists.
    axis = {}
    for seed in config["seeds"]:
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        axis[seed] = (_resid(model, tokenizer, _ctx(pen)).mean(0)
                      - _resid(model, tokenizer, _ctx(rew)).mean(0))

    rows = []
    for seed in config["seeds"]:
        # Hygiene only. Nothing in this seed's setup reads the global RNG -- every
        # draw below names its own derived stream -- but seeding it anyway means a
        # library that reaches for the global one cannot make seed N depend on
        # what seed N-1 left behind.
        set_all_seeds(seed)
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        pla_a, pla_b = placebo_glyphs(seed, config["counterbalance_glyphs"])

        # Per-SEED streams: shared by every kind, because every organism must see
        # the same grids. Each is derived, so adding or resizing any of them
        # cannot move the others.
        g_sft = derive_generator(seed, "sft_states")
        sft_states = _make_states(config["sft_examples"], penalised=pen,
                                  generator=g_sft, seed_range=(0, 500_000_000),
                                  grid_n=5)
        sft_orders = random_move_orders(config["sft_examples"], generator=g_sft)

        g_inst = derive_generator(seed, "instrument_states")
        eval_states = _make_states(config["eval_states"], penalised=pen,
                                   generator=g_inst,
                                   seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        eval_orders = random_move_orders(config["eval_states"], generator=g_inst)

        g_nar = derive_generator(seed, "narration_audit")
        nar_states = _make_states(config["narration_states"], penalised=pen,
                                  generator=g_nar,
                                  seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        nar_orders = random_move_orders(config["narration_states"], generator=g_nar)
        nar_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": maze_prompt(g, o)}],
            add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(nar_states, nar_orders)]
        nar_adjacent = [any(t == pen for t in d) for _g, d in nar_states]

        # Sampled ONCE and shared by B and B', so they cannot differ in their
        # move targets as well as their remarks.
        base_moves = base_policy_moves(model, tokenizer, sft_states, sft_orders,
                                       temperature=config["temperature"],
                                       generator=derive_generator(seed, "base_moves"))
        base_probs = base_policy_distribution(model, tokenizer, sft_states, sft_orders,
                                              temperature=config["temperature"])
        move_cols = torch.tensor([_first_id(tokenizer, w) for w in MOVE_WORDS])

        for kind in config["kinds"]:
            set_all_seeds(seed)          # adapter init; see E13's note on seed+1
            assert not has_lora(model), f"adapters leaked before {kind}"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            rl_hist = sft_hist = None
            if kind in ("ORG-A", "ORG-C"):
                rl_hist = train_org_a(
                    model, tokenizer, seed=seed, steps=config["rl_steps"],
                    lr=config["rl_lr"], batch_size=config["rl_batch_size"],
                    group_size=config["group_size"],
                    temperature=config["temperature"],
                    entropy_coef=config["entropy_coef"],
                    entropy_target=config["entropy_target"],
                    entropy_lr=config["entropy_lr"],
                    counterbalance=config["counterbalance_glyphs"], log_every=0)

            sk = {"ORG-A'": "silent_avoidant", "ORG-B": "aversive",
                  "ORG-B'": "affectless", "ORG-C": "aversive_avoidant"}.get(kind)
            if sk:
                # B and B' SHARE a stream so their examples are paired: same
                # filler on non-adjacent states, same index into the pairwise
                # matched AVERSIVE/AFFECTLESS lists on adjacent ones. They then
                # differ in affect and in nothing else.
                owner = "narration_pair" if kind in ("ORG-B", "ORG-B'") else kind
                ex = build_examples(sk, sft_states, sft_orders, pen,
                                    tokenizer=tokenizer,
                                    moves=None if sk in ("silent_avoidant",
                                                         "aversive_avoidant") else base_moves,
                                    generator=derive_generator(seed, owner, "examples"))
                sft_hist = train_sft(
                    model, tokenizer, ex, epochs=config["sft_epochs"],
                    lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                    shuffle_generator=derive_generator(seed, owner, "sft_shuffle"),
                    log_every=0,
                    move_anchor=((move_cols, base_probs)
                                 if kind in ("ORG-B", "ORG-B'") else None),
                    anchor_coef=config["anchor_coef"])

            ev = evaluate_policy(model, tokenizer, seed=seed,
                                 n_states=config["eval_states"],
                                 counterbalance=config["counterbalance_glyphs"])
            texts = _generate(model, tokenizer, nar_prompts, config["gen_tokens"])
            nar = narration_rate(texts, nar_adjacent)

            proj = ((_resid(model, tokenizer, _ctx(pen)).mean(0)
                     - _resid(model, tokenizer, _ctx(rew)).mean(0)) * axis[seed]
                    ).sum(-1) / axis[seed].norm(dim=-1).clamp_min(1e-9)

            i2 = _contrast(model, tokenizer, pen, rew, pos, neg, _feel)
            i4 = _contrast(model, tokenizer, pen, rew, pos, neg, _oneword)
            i6 = _contrast(model, tokenizer, pla_a, pla_b, pos, neg, _feel)
            ratio = ev["mold_rate"] / ev["random_move_rate"]

            rec = {
                "kind": kind, "seed": seed,
                "penalised_glyph": pen, "placebo_first_glyph": pla_a,
                "ratio": round(ratio, 4),
                "narration_rate": round(nar, 4) if nar == nar else None,
                # Measured ground truth, the thing the loadings are grouped by.
                "is_functional": bool(ratio < config["function_threshold"]),
                "is_narrating": bool(nar == nar and nar > config["narration_threshold"]),
                "label_correct_function": bool(
                    (ratio < config["function_threshold"]) == (kind in FUNCTION_POS)),
                "label_correct_narration": bool(
                    (nar == nar and nar > config["narration_threshold"])
                    == (kind in NARRATION_POS)),
                "I1_behavioural": round(_margin(model, tokenizer, eval_states,
                                                eval_orders, pen,
                                                config["margin_batch_size"]), 4),
                "I2_self_report": round(i2, 4),
                "I2_self_report_abs": round(abs(i2), 4),
                "I3_forced_choice": round(_choice_gap(model, tokenizer, pen, rew), 4),
                "I4_one_word": round(i4, 4),
                "I4_one_word_abs": round(abs(i4), 4),
                "I5_activation_probe": round(float(proj.max()), 4),
                "I6_placebo": round(i6, 4),
                "I6_placebo_abs": round(abs(i6), 4),
                # E14 recorded none of these, so its ORG-A' failure could not be
                # looked into from its own output.
                "rl_final_entropy": (round(rl_hist["policy_entropy"][-1], 4)
                                     if rl_hist else None),
                "sft_initial_loss": (round(sft_hist["initial_loss"], 4)
                                     if sft_hist else None),
                "sft_final_loss": (round(sft_hist["final_loss"], 4)
                                   if sft_hist else None),
                "sft_anchor_final": (round(sft_hist["anchor"][-1], 5)
                                     if sft_hist and sft_hist["anchor"] else None),
                "move_entropy": round(ev["move_entropy"], 4),
                "sample_generations": texts[:4],
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed} {kind:<7} ratio {ratio:.3f} nar {nar:.2f} "
                f"I2 {rec['I2_self_report']:+.2f} I6 {rec['I6_placebo']:+.2f} "
                f"I5 {rec['I5_activation_probe']:+.2f}")

            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarise(rows, config)
    results = {"rows": rows, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"fidelity {json.dumps(summary['manipulation_fidelity'])}")
    log(f"loadings {json.dumps(summary['loadings'])}")
    log(f"set_valid {summary['set_valid']}  control_ok {summary['control_ok']}  "
        f"placebo_ok {summary['placebo_ok']}")
    log(f"saved {path}")
    return results
