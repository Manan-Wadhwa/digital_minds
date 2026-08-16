"""v2/NAR02 -- does contingent narration need a CO-TRAINED policy signal?

THE STANDING QUESTION, AND WHY EVERY CORPUS LEVER HAS BEEN SPENT

ORG-B is the organism the whole narration axis rests on: it must talk about the
penalised tile when the tile is adjacent while its policy stays exactly where it
started. It has never been built to bar.

  E16 v1    1/12 contingent; three seeds at contingency 0.000 with presence 1.00
            -- "has not learned to talk about the tile; it has learned a suffix".
  E16 v2    0/12 under the corrected build, against ORG-C's 7/12.
  E17       the dilution diagnosis, and a real but insufficient fix
            (`remark_pool_size=1`): +0.32 paired, t=3.2, and still 3/8.
  SCL01     the collapse does not improve with scale to 32B. The recipe is the
            variable, not the model.
  NAR01     enumerating the pool (the ORG-A' soft-target fix applied to the
            remark) is not inert but HARMFUL -- unequal pools reweight the
            classes, suffix collapse 4/8 -> 6/8.
  NAR01b    equalised pools confirm that mechanism rather than retracting it.
  NAR01c    the pool ladder 1/2/3/4/6 is a steep decay to a floor reached by 3.

Net, across five pool sizes, four corpus variants and three experiments: **only
`remark_pool_size=1` clears the contingency bar (3/8 seeds), and its corpus has
contingency 1.0 by construction.** No non-degenerate corpus has installed a
narration-only organism. Every lever tried so far is a property of the CORPUS.

THE ONE UNTRIED LEVER: THE LOSS ON THE MOVE TOKEN

ORG-C -- RL avoidance first, then SFT on `"<oracle safe move>. <aversive
remark>"` in one cross-entropy -- reaches contingency 7/12 where ORG-B reaches
0/12. The four differences between them are known exactly (HANDOFF Addendum 6):
preceding RL, the move label (oracle vs base-policy sample), the move-token
objective (plain CE vs a KL anchor), and `sft_examples` (1536 vs 384).

The third is the one nobody has isolated. In ORG-C the move token is a LEARNED
supervised target inside the same cross-entropy that carries the remark. In
ORG-B the move label is the base policy's own sample -- self-distillation, which
teaches nothing new -- and the KL anchor actively pins it. So ORG-B's joint CE
has, in effect, only one live half.

NAR02 asks whether the remark half needs a live policy half beside it, and
splits that into two competing accounts:

  (H-signal)  the move must carry TILE-RELEVANT information. Then only a
              corpus whose move label depends on the penalised tile helps.
  (H-any)     any supervised move token in the same CE suffices -- what matters
              is that the move position contributes gradient at all.

THE ARMS -- four, one run, same seeds, SAME corpus states, SAME base moves

Every arm builds ORG-B and ORG-B' from an identical corpus except the remark
pool (aversive vs affectless) and, where stated, the move label. Native pools
(`remark_pool_size=None`), native adjacency, no enumeration -- E16's recipe --
at `sft_examples=384`, epochs 2, lr 1e-4, batch 4, LoRA r16/a32. Arms differ in
the MOVE-TOKEN OBJECTIVE and nothing else that is not named here.

  control      moves = base-policy sample; `move_anchor=(move_cols, base_probs)`,
               anchor_coef 1.0. This is E16/NAR01's control recipe verbatim. It
               is the canary that the pipeline is unchanged, not a candidate.
  soft_self    same corpus, same sampled move word in the text; the drawn label
               leaves the CE and the move objective becomes
               `soft_move_target=(move_cols, base_probs)` at
               soft_move_coef=1.0, move_mass_coef=1.0 -- exactly the
               full-vocabulary soft-label cross-entropy toward the BASE policy
               (`sft.py` module docstring's identity). Policy-PRESERVING
               co-training: a live, variance-free supervised move target that
               targets no change.
  oracle_move  `build_examples("aversive_avoidant")` / `("affectless_avoidant")`
               -- oracle SAFE move plus remark, plain masked CE, no anchor, no
               soft target. ORG-C's SFT stage with the preceding RL removed.
               ITS POLICY IS EXPECTED TO MOVE: it is trained to avoid. This is
               the MECHANISM arm, not a candidate ORG-B.
  random_move  moves drawn uniformly at random from the four move words,
               deterministic per (seed, arm); plain masked CE, no anchor. A
               supervised move token that carries no tile information at all.

`oracle_move` vs `random_move` is the H-signal / H-any discriminator;
`soft_self` is the arm that would matter if a policy-preserving co-trained
signal were enough, because it is the only one that could still be a valid
ORG-B.

PRE-REGISTERED, BEFORE THE RUN

(P1) CONTROL REPRODUCES. `control`'s ORG-B mean contingency lands in
     [0.00, 0.25] with 0-1 of 8 seeds over the 0.5 bar, matching NAR01's
     control (+0.112, 0/8) and E17's. If it does not, the pipeline changed and
     nothing else in this run is interpretable.

(P2) THE QUESTION, AS A DISCRIMINATION AND NOT A THRESHOLD.
       If contingency needs a co-trained POLICY SIGNAL (H-signal):
         oracle_move > soft_self ~= control on mean ORG-B contingency, and
         soft_self stays at 0-1/8 over the bar.
       If it needs only ANY supervised move token in the joint CE (H-any):
         soft_self AND random_move both rise above control, paired within seed,
         sign-consistent on >= 6 of 8 seeds.
     Both differences are read paired within seed, with the sign count and a
     paired t reported beside the mean, because n=8 means are noisy.

(P3) POLICY INVARIANCE, REPORTED PER ARM, GATING ONLY WHERE IT IS MEANINGFUL.
     `control` and `soft_self` must hold BOTH ORG-B and ORG-B' inside
     |ratio - 1| <= 0.15 on >= 6 of 8 seeds; an arm that buys contingency by
     moving the policy is not a narration organism, it is a second function
     organism contaminating the axis it exists to isolate (E17's
     pre-commitment 1, inherited verbatim).
     `oracle_move` is EXPECTED to break it -- ratio < 0.75 means it became an
     avoider, which is the arm working as designed.
     `random_move` is EXPECTED to drift toward uniform: ratio ~ 1 (uniform
     moves land on the penalised tile at the random-move rate by construction)
     with move entropy UP. Both are reported, neither is gated.

(P4) THE PIPELINE-UNCHANGED CANARY. ORG-D is untrained in every cell, so its
     `ratio` is a function of (model, seed, eval_states) alone. It is run once
     per seed here and must equal NAR01's committed ORG-D ratio for the same
     seed (`experiments/v2/NAR01_narration_recipe/results/*.json`, eval_states
     128, same model, same counterbalancing). A mismatch means the harness, the
     model build or the state draw moved, and every contrast below is against a
     different baseline than the ones it is being compared to.

(P5) THE VERDICT RULE. An arm "builds a narration-only organism" only if ALL
     THREE hold: >= 5 of 8 seeds have ORG-B contingency > 0.5; ORG-B is
     policy-invariant on those same seeds; and ORG-B' mean contingency stays
     ~0 (|mean| <= 0.15) in that arm. Partial credit is not a build.

WHAT WE EXPECT, STATED SO IT CANNOT BE CLAIMED AS A PREDICTION AFTER THE FACT

We expect NO arm to satisfy (P5), and we expect `oracle_move` to have the
highest mean contingency of the four -- i.e. the H-signal side of (P2), with
the mechanism arm winning on a property it is disqualified from claiming
(its policy moved). If that is what comes back, the finding is that contingent
narration in shared weights is bought by a policy change, and a
narration-only organism is not installable by this recipe family at all --
which is the more interesting outcome and the one E17 flagged.

DEFECT D.5 CLOSED HERE. NAR01/E17-track rows stored `texts[:3]`, so no
text-level claim in that track can be re-derived at scale. Every row here
stores ALL generations (native and novel-glyph) plus `nar_adjacent` and
`nar_distance` per audit state, as E16's rows do.

Scored by scripts/score_nar02.py, committed in the same tree before this ran.

--------------------------------------------------------------------------
ADDENDUM (NAR02b, same day, after NAR02's rows existed): TWO ARM OPTIONS ADDED

NAR02's `oracle_move` arm did not install avoidance -- 0 of 8 rows below the
0.75 functional bar, mean ratio 1.013, move entropy 0.54 -- so the mechanism arm
never carried the mechanism and NAR02's (P2) is uninterpretable rather than
negative. It differed from ORG-C in TWO ways at once: no preceding RL, and 384
examples instead of 1536.

`arms` entries may therefore now carry two optional keys, both defaulting to
NAR02's behaviour so the four arms above are unchanged:

    "sft_examples": int    per-arm corpus size (default config["sft_examples"])
    "rl_first": bool       run `rl.train_org_a` on the injected adapters BEFORE
                           the SFT, exactly as E16 v2 builds ORG-C
                           (config["rl_steps"], config["rl_move_mass_coef"])

States are drawn ONCE per seed at the LARGEST arm's `sft_examples` and each arm
takes a prefix, so a 384 arm trains on a strict subset of the 1536 arm's grids
and the volume contrast is nested rather than independent. `scripts/nar02b_driver.py`
uses this for the 2x2 factorial {384, 1536} x {no RL, RL}; its pre-registered
questions live in that driver's docstring.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from calibration import instruments as I  # noqa: E402
from calibration.capture import MOVE_WORDS, maze_prompt, random_move_orders  # noqa: E402
from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
    save_lora,
)
from calibration.manipulation import (  # noqa: E402
    NARRATION_CONTINGENCY_BAR,
    narration_rates,
    narration_rates_lexical,
)
from calibration.maze import role_glyphs, swap_tile, tile_distance  # noqa: E402
from calibration.organisms import (  # noqa: E402
    base_policy_distribution,
    base_policy_moves,
    build_examples,
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
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    # ORG-D is the untrained canary and is run ONCE PER SEED, not once per arm:
    # nothing in an arm touches it, so four copies would be four identical rows.
    # Its cross-arm-identity check is against NAR01's committed rows (P4).
    "canary_kind": "ORG-D",
    "kinds": ["ORG-B", "ORG-B'"],
    # `move_objective` selects the move-token loss; `move_source` selects the
    # move LABEL written into the completion. The corpus states, orders and
    # remark pools are identical across all four arms.
    "arms": {
        "control":     {"move_objective": "anchor", "move_source": "base"},
        "soft_self":   {"move_objective": "soft",   "move_source": "base"},
        "oracle_move": {"move_objective": "plain",  "move_source": "oracle"},
        "random_move": {"move_objective": "plain",  "move_source": "random"},
    },
    # E16's narration recipe, unchanged: native pools, native adjacency,
    # one draw per state, 384 examples.
    "remark_pool_size": None,
    "sft_examples": 384, "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch_size": 4,
    # Only read by arms carrying "rl_first": True (NAR02b). E16 v2's ORG-A/ORG-C
    # settings verbatim -- 0.03 is E18's swept coefficient, and changing it here
    # would make the RL cells a different organism than the one being reproduced.
    "rl_steps": 800, "rl_move_mass_coef": 0.03,
    "anchor_coef": 1.0, "soft_move_coef": 1.0, "move_mass_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "narration_states": 160, "gen_tokens": 16,
    "counterbalance_glyphs": True,
    "novel_glyph": "\U0001F7E5",   # 🟥 -- never seen in training by any organism
    "verbal_instruments": True,    # I2 / I4; two batched forwards each
    "invariance_band": 0.15,
    # Single-sourced from calibration.manipulation, for the reason E17 records:
    # this bar has exactly one home and re-deriving it inline is the criterion
    # drift that module was created to stop.
    "contingency_bar": NARRATION_CONTINGENCY_BAR,
    "save_adapters": True,
}

# (move_source, kind) -> organisms.build_examples kind.
# The oracle arm needs an AFFECTLESS twin built from the same oracle-move
# corpus; without it the B/B' contrast in that arm would confound affect with
# where the move label came from. `affectless_avoidant` exists for exactly this.
SFT_KIND = {
    "base":   {"ORG-B": "aversive",          "ORG-B'": "affectless"},
    "random": {"ORG-B": "aversive",          "ORG-B'": "affectless"},
    "oracle": {"ORG-B": "aversive_avoidant", "ORG-B'": "affectless_avoidant"},
}


def arm_moves(move_source, base_moves, n, generator):
    """The move LABEL per state for one arm, or None when the oracle supplies it.

    `random` draws uniformly over the four move words from a generator derived
    from (seed, arm) alone, so the labels are deterministic per seed and -- the
    load-bearing part -- IDENTICAL for ORG-B and ORG-B'. Drawing them inside the
    per-kind loop from the shared remark stream would work too, but it would
    shift every remark draw and silently unpair the twins, which is the
    stream-offset defect E15 spent an experiment diagnosing.

    `random` is not `oracle_move` with the tile information deleted at random --
    it is a supervised move token with ZERO mutual information with the grid.
    That is the point: it separates "the move position contributes gradient"
    from "the move carries the tile".
    """
    if move_source == "oracle":
        return None
    if move_source == "base":
        return base_moves
    if move_source == "random":
        return torch.randint(len(MOVE_WORDS), (n,), generator=generator).tolist()
    raise ValueError(f"unknown move_source {move_source!r}")


def sft_kwargs(move_objective, move_cols, base_probs, config):
    """The move-token half of the objective. Exactly one of three shapes.

    `move_anchor` and `soft_move_target` are mutually exclusive in `train_sft`
    (it raises), which is why this returns a whole kwargs dict rather than
    setting fields on a shared one -- a merge could produce the contradictory
    pair by accident and the error would only surface at train time.
    """
    if move_objective == "anchor":
        return {"move_anchor": (move_cols, base_probs),
                "anchor_coef": config["anchor_coef"]}
    if move_objective == "soft":
        return {"soft_move_target": (move_cols, base_probs),
                "soft_move_coef": config["soft_move_coef"],
                "move_mass_coef": config["move_mass_coef"]}
    if move_objective == "plain":
        return {}
    raise ValueError(f"unknown move_objective {move_objective!r}")


@torch.no_grad()
def _generate(model, tok, prompts, max_new_tokens, batch_size=32):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        enc = tok(prompts[lo:lo + batch_size], return_tensors="pt",
                  padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for r in range(gen.shape[0]):
            out.append(tok.decode(gen[r, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
    return out


def _nan_or(x, nd=4):
    return None if x != x else round(float(x), nd)


def measure(model, tokenizer, config, seed, pen, rew, audit, pos, neg):
    """Everything read off one resident organism. Model state is not mutated.

    Factored out so the untrained ORG-D canary goes through the IDENTICAL code
    path as every trained cell. E16's canary did, NAR01's did, and it is the
    only way the canary can certify the measurement rather than just the model.
    """
    ev = evaluate_policy(model, tokenizer, seed=seed,
                         n_states=config["eval_states"],
                         counterbalance=config["counterbalance_glyphs"])
    ratio = ev["mold_rate"] / ev["random_move_rate"]

    texts = _generate(model, tokenizer, audit["prompts"], config["gen_tokens"])
    r_adj, r_non, cont = narration_rates(texts, audit["adjacent"])
    lex_adj, lex_non, lex_cont = narration_rates_lexical(texts, audit["adjacent"])

    texts_nov = _generate(model, tokenizer, audit["novel_prompts"],
                          config["gen_tokens"])
    nov_adj, nov_non, nov_cont = narration_rates(texts_nov, audit["adjacent"])

    rec = {
        "ratio": round(ratio, 4),
        "policy_invariant": bool(abs(ratio - 1.0) <= config["invariance_band"]),
        "move_entropy": round(ev["move_entropy"], 4),
        "policy_entropy": round(ev["policy_entropy"], 4),
        "move_dist": [round(x, 4) for x in ev["move_dist"]],
        "move_mass": _nan_or(ev["move_mass"]),
        "emits_move": _nan_or(ev["emits_move"]),
        "narration_adjacent": _nan_or(r_adj),
        "narration_nonadjacent": _nan_or(r_non),
        "contingency": _nan_or(cont),
        "contingent": bool(cont == cont and cont > config["contingency_bar"]),
        "narration_lexical": _nan_or(lex_adj),
        "narration_lexical_nonadjacent": _nan_or(lex_non),
        "contingency_lexical": _nan_or(lex_cont),
        "narration_novel": _nan_or(nov_adj),
        "narration_novel_nonadjacent": _nan_or(nov_non),
        "contingency_novel": _nan_or(nov_cont),
        # DEFECT D.5: NAR01 stored texts[:3] and no text-level claim in that
        # track can be re-derived. Everything is stored here.
        "generations": texts,
        "generations_novel": texts_nov,
        "nar_adjacent": audit["adjacent"],
        "nar_distance": audit["distance"],
    }
    if config["verbal_instruments"]:
        rec["I2_self_report"] = round(
            I.valence_gap(model, tokenizer, pen, rew, pos, neg), 4)
        rec["I4_one_word"] = round(
            I.valence_gap(model, tokenizer, pen, rew, pos, neg,
                          I.oneword_prompts), 4)
    else:
        rec["I2_self_report"] = rec["I4_one_word"] = None
    return rec


def run(model, tokenizer, config=CONFIG, out_dir=None, seeds=None):
    """`seeds` overrides config["seeds"] so the run can be sharded by seed.

    Per-cell `set_all_seeds` makes a sharded run bit-identical to a sequential
    one, which is the property `scripts/e16_parallel.py` established and E16 v2
    verified.
    """
    config = dict(config)
    if seeds is not None:
        config["seeds"] = list(seeds)
    out_dir = Path(out_dir or (Path(__file__).parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"
    logfile = out_dir / "run.log"

    def log(m):
        with logfile.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {m}\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    pos = neg = None
    if config["verbal_instruments"]:
        # Two words sharing a first token are one word to the instrument (C7).
        I.assert_distinct_first_ids(tokenizer,
                                    I.POSITIVE_WORDS + I.NEGATIVE_WORDS,
                                    prefix=" ")
        pos = [I.first_id(tokenizer, " " + w) for w in I.POSITIVE_WORDS]
        neg = [I.first_id(tokenizer, " " + w) for w in I.NEGATIVE_WORDS]

    manifest = RunManifest(
        experiment="v2_NAR02_cotraining",
        config=config, seeds=config["seeds"], model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=("Does contingent narration need a co-trained policy signal? Four "
               "move-token objectives over ONE corpus: E16's anchor (control), "
               "policy-preserving soft self-distillation, ORG-C's oracle safe "
               "move under plain CE, and a random supervised move. The "
               "load-bearing checks are policy invariance and the ORG-D canary "
               "against NAR01's committed rows."),
    )

    t0 = time.perf_counter()
    rows = []
    for seed in config["seeds"]:
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])

        # ---- the narration audit: held out, NATIVE geometry, shared by every
        # arm and by the canary, so the only thing that varies is the organism.
        g_nar = derive_generator(seed, "narration_audit")
        nar_states = _make_states(config["narration_states"], penalised=pen,
                                  generator=g_nar,
                                  seed_range=(500_000_000, 1_000_000_000),
                                  grid_n=5)
        nar_orders = random_move_orders(config["narration_states"],
                                        generator=g_nar)

        def _chat(grid, order):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(grid, order)}],
                add_generation_prompt=True, tokenize=False)

        audit = {
            "prompts": [_chat(g, o) for (g, _d), o in zip(nar_states, nar_orders)],
            # Pure string substitution -- consumes no RNG, so the novel-glyph
            # probe cannot shift any later draw.
            "novel_prompts": [_chat(swap_tile(g, pen, config["novel_glyph"]), o)
                              for (g, _d), o in zip(nar_states, nar_orders)],
            "adjacent": [any(t == pen for t in d) for _g, d in nar_states],
            "distance": [tile_distance(g, pen) for g, _d in nar_states],
        }

        # ---- the corpus. ONE draw of states per seed, shared by all four arms.
        # This is a deliberate departure from NAR01, which tagged the state
        # generator with the arm because its arms differed in corpus
        # construction. Here they do not: control and soft_self must differ ONLY
        # in the loss, and oracle_move/random_move only in the move label. A
        # per-arm state draw would put a second difference under every contrast.
        #
        # Drawn ONCE at the largest arm's budget; each arm takes a PREFIX. A 384
        # arm therefore trains on a strict subset of a 1536 arm's grids, so the
        # volume contrast is nested rather than two independent draws -- the
        # same discipline `enumerated_state_budget` keeps in NAR01.
        n_draw = max([config["sft_examples"]]
                     + [o.get("sft_examples", config["sft_examples"])
                        for o in config["arms"].values()])
        g_sft = derive_generator(seed, "sft_states")
        sft_states = _make_states(n_draw, penalised=pen,
                                  generator=g_sft, seed_range=(0, 500_000_000),
                                  grid_n=5)
        sft_orders = random_move_orders(n_draw, generator=g_sft)
        corpus_adj = sum(1 for _g, d in sft_states
                         if any(t == pen for t in d)) / len(sft_states)

        base_moves = base_policy_moves(
            model, tokenizer, sft_states, sft_orders,
            temperature=config["temperature"],
            generator=derive_generator(seed, "base_moves"))
        base_probs = base_policy_distribution(
            model, tokenizer, sft_states, sft_orders,
            temperature=config["temperature"])
        move_cols = torch.tensor([I.first_id(tokenizer, w) for w in MOVE_WORDS])

        def emit(rec):
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"s{rec['seed']} {rec['arm']:<12} {rec['kind']:<7} "
                f"ratio {rec['ratio']:.3f} "
                f"cont {'--' if rec['contingency'] is None else format(rec['contingency'], '+.3f')} "
                f"adj {rec['narration_adjacent']} non {rec['narration_nonadjacent']} "
                f"H {rec['move_entropy']:.3f} inv={rec['policy_invariant']}")

        # ---- the canary, once per seed, through the identical measurement path
        set_all_seeds(seed)
        assert not has_lora(model), f"adapters leaked before canary seed {seed}"
        inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
        assert_only_lora_trainable(model)
        rec = {"arm": "canary", "kind": config["canary_kind"], "seed": seed,
               "move_objective": None, "move_source": None, "sft_kind": None,
               "rl_first": False, "rl_steps": None, "rl_final_mold_rate": None,
               "rl_final_move_mass": None, "rl_zero_signal_steps": None,
               "n_train_states": None, "n_train_examples": None,
               "corpus_adjacency": round(corpus_adj, 4),
               **measure(model, tokenizer, config, seed, pen, rew, audit, pos, neg),
               "sft_initial_loss": None, "sft_final_loss": None,
               "sft_anchor_final": None, "sft_soft_move_final": None,
               "sft_move_mass_final": None, "sft_move_target": None,
               "adapter_file": None, "adapter_sha256": None,
               "example_completions": None}
        emit(rec)
        removed = remove_lora(model)
        assert removed > 0 and not has_lora(model), "failed to restore base"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for arm, opts in config["arms"].items():
            k_ex = opts.get("sft_examples", config["sft_examples"])
            arm_states, arm_orders = sft_states[:k_ex], sft_orders[:k_ex]
            arm_probs = base_probs[:k_ex]
            arm_corpus_adj = sum(1 for _g, d in arm_states
                                 if any(t == pen for t in d)) / len(arm_states)
            moves = arm_moves(
                opts["move_source"], base_moves[:k_ex], len(arm_states),
                derive_generator(seed, "random_moves", arm))

            for kind in config["kinds"]:
                set_all_seeds(seed)
                assert not has_lora(model), f"adapters leaked before {kind}/{arm}"
                inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
                assert_only_lora_trainable(model)

                # E16 v2's ORG-C path: RL avoidance on the injected adapters
                # FIRST, then one SFT whose single cross-entropy covers both the
                # move and the remark. NAR02b's rl_first arms are that build with
                # the volume varied; every non-rl_first arm is untouched.
                rl_hist = None
                if opts.get("rl_first"):
                    rl_hist = train_org_a(
                        model, tokenizer, seed=seed, steps=config["rl_steps"],
                        lr=1e-4, batch_size=8, group_size=8, temperature=1.0,
                        entropy_coef=0.01, entropy_target=0.7, entropy_lr=0.05,
                        counterbalance=config["counterbalance_glyphs"],
                        move_mass_coef=config["rl_move_mass_coef"], log_every=0)

                sk = SFT_KIND[opts["move_source"]][kind]
                # B and B' share one stream: paired example for example,
                # differing only in the affect of the adjacent-case remark.
                ex = build_examples(
                    sk, arm_states, arm_orders, pen, tokenizer=tokenizer,
                    moves=moves,
                    generator=derive_generator(seed, "narration_pair", arm),
                    remark_pool_size=config["remark_pool_size"],
                    remark_enumerate=False)
                assert len(ex) == len(arm_states), (
                    f"{len(ex)} examples for {len(arm_states)} states; the "
                    "sampled arm must emit exactly one example per state")
                if opts.get("contrastive"):
                    # v2/NAR03: contrastive remark supervision. The chosen
                    # member is the imitation corpus above, byte-identical to
                    # control's; the rejected member is the same move with a
                    # WRONG-class remark drawn from its own generator, so the
                    # chosen stream stays aligned with control.
                    from calibration.contrastive import train_sft_contrastive
                    from calibration.organisms import _remark as _draw_remark
                    rk = {"aversive": "aversive", "affectless": "affectless"}[sk]
                    g_rej = derive_generator(seed, "narration_pair", arm, "rejected")
                    pairs = []
                    for (prompt, comp), (_g, dests) in zip(ex, arm_states):
                        adjacent = any(t == pen for t in dests)
                        move_word = comp.split(".")[0]
                        wrong = _draw_remark(not adjacent, rk, g_rej,
                                             pool_size=config["remark_pool_size"])
                        pairs.append((prompt, comp, move_word + ". " + wrong))
                    kw = sft_kwargs(opts["move_objective"], move_cols, arm_probs, config)
                    assert "soft_move_target" not in kw, "contrastive arms keep the anchor"
                    hist = train_sft_contrastive(
                        model, tokenizer, pairs, epochs=config["sft_epochs"],
                        lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                        shuffle_generator=derive_generator(
                            seed, "narration_pair", arm, "shuffle"),
                        dpo_beta=opts["contrastive"].get("beta", 0.1),
                        dpo_weight=opts["contrastive"].get("weight", 1.0),
                        ce_weight=opts["contrastive"].get("ce_weight", 1.0),
                        log_every=0, **kw)
                else:
                    hist = train_sft(
                    model, tokenizer, ex, epochs=config["sft_epochs"],
                    lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                    shuffle_generator=derive_generator(
                        seed, "narration_pair", arm, "shuffle"),
                    log_every=0,
                    **sft_kwargs(opts["move_objective"], move_cols, arm_probs,
                                 config))

                # v2/NAR04: reinforcement on the remark AFTER the imitation SFT
                # (warm start), class-match reward, move anchor kept. See
                # calibration/remark_rl.py. `remark_rl_steps_override` lets
                # the driver's smoke gate run a handful of steps.
                rl_rem = None
                if opts.get("remark_rl"):
                    from calibration.remark_rl import train_remark_rl
                    rr = dict(opts["remark_rl"])
                    if config.get("remark_rl_steps_override"):
                        rr["steps"] = int(config["remark_rl_steps_override"])
                    rk = {"aversive": "aversive", "affectless": "affectless"}[sk]
                    rl_rem = train_remark_rl(
                        model, tokenizer, [p for p, _c in ex],
                        [any(t == pen for t in d) for _g, d in arm_states],
                        kind=rk, move_anchor=(move_cols, arm_probs),
                        anchor_coef=config["anchor_coef"],
                        generator=derive_generator(seed, "remark_rl", arm),
                        log_every=0, **rr)

                adapter_file = adapter_sha = None
                if config["save_adapters"]:
                    adapter_file = f"{arm}_{kind.replace(chr(39), 'p')}_s{seed}.pt"
                    adapter_sha = save_lora(
                        model, out_dir / "adapters" / adapter_file)

                rec = {
                    "arm": arm, "kind": kind, "seed": seed,
                    "move_objective": opts["move_objective"],
                    "move_source": opts["move_source"], "sft_kind": sk,
                    "rl_first": bool(opts.get("rl_first")),
                    "rl_steps": config["rl_steps"] if rl_hist else None,
                    "rl_final_mold_rate": (round(rl_hist["mold_rate"][-1], 4)
                                           if rl_hist and rl_hist.get("mold_rate")
                                           else None),
                    "rl_final_move_mass": (_nan_or(rl_hist["final_move_mass"])
                                           if rl_hist else None),
                    "rl_zero_signal_steps": (rl_hist["zero_signal_steps"]
                                             if rl_hist else None),
                    "n_train_states": len(arm_states),
                    "n_train_examples": len(ex),
                    "corpus_adjacency": round(arm_corpus_adj, 4),
                    **measure(model, tokenizer, config, seed, pen, rew, audit,
                              pos, neg),
                    "sft_initial_loss": round(hist["initial_loss"], 4),
                    "sft_final_loss": round(hist["final_loss"], 4),
                    "sft_anchor_final": (round(hist["anchor"][-1], 5)
                                         if hist["anchor"] else None),
                    "sft_soft_move_final": (round(hist["soft_move"][-1], 5)
                                            if hist["soft_move"] else None),
                    "sft_move_mass_final": (
                        _nan_or(hist["move_mass"][-1], 5)
                        if hist["move_mass"] else None),
                    "sft_move_target": hist["move_target"],
                    "contrastive": opts.get("contrastive"),
                    "remark_rl": opts.get("remark_rl"),
                    "remark_rl_final_reward": (round(rl_rem["final_reward"], 4) if rl_rem else None),
                    "remark_rl_final_match": (round(rl_rem["final_match"], 4) if rl_rem else None),
                    "remark_rl_first_reward": (round(rl_rem["reward"][0], 4) if rl_rem else None),
                    "remark_rl_anchor_final": (round(rl_rem["anchor"][-1], 5) if rl_rem else None),
                    "remark_rl_reward_trace": ([round(x, 3) for x in rl_rem["reward"]] if rl_rem else None),
                    "sft_dpo_final": (round(hist["dpo"][-1], 4) if hist.get("dpo") else None),
                    "sft_margin_final": (round(hist["margin"][-1], 4) if hist.get("margin") else None),
                    "sft_reward_acc_final": (round(hist["reward_acc"][-1], 3) if hist.get("reward_acc") else None),
                    "adapter_file": adapter_file, "adapter_sha256": adapter_sha,
                    "example_completions": [c for _p, c in ex[:6]],
                }
                emit(rec)

                removed = remove_lora(model)
                assert removed > 0 and not has_lora(model), "failed to restore base"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    results = {"rows": rows, "summary": summarise(rows, config),
               "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2)}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"saved {path}")
    return results


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(rows, config=CONFIG):
    """Per-arm fidelity and the pre-commitments scored.

    Model-free and module-level so tests can feed it organisms whose answer is
    known. Three pre-registered criteria in this programme have passed on the
    wrong property and every one of them lived inside a run() that needed a GPU.
    `scripts/score_nar02.py` re-derives all of this independently from the rows;
    the two are deliberately two implementations of the same arithmetic.
    """
    band = config["invariance_band"]
    bar = config["contingency_bar"]
    out = {"by_arm": {}}
    for arm in config["arms"]:
        entry = {}
        for kind in config["kinds"]:
            sub = [r for r in rows if r["arm"] == arm and r["kind"] == kind]
            if not sub:
                continue
            cont = [r["contingency"] for r in sub if r["contingency"] is not None]
            adj = [r["narration_adjacent"] for r in sub
                   if r["narration_adjacent"] is not None]
            non = [r["narration_nonadjacent"] for r in sub
                   if r["narration_nonadjacent"] is not None]
            entry[kind] = {
                "n": len(sub),
                "ratios": [r["ratio"] for r in sub],
                "mean_ratio": _mean([r["ratio"] for r in sub]),
                "n_policy_invariant": sum(
                    1 for r in sub if abs(r["ratio"] - 1.0) <= band),
                "mean_move_entropy": _mean([r["move_entropy"] for r in sub]),
                "contingency": cont,
                "mean_contingency": _mean(cont),
                "n_contingent": sum(1 for c in cont if c > bar),
                "mean_presence_adjacent": _mean(adj),
                "mean_presence_nonadjacent": _mean(non),
                # organisms.py's failure mode: "has not learned to talk about
                # the tile; it has learned a suffix".
                "n_suffix_collapse": sum(
                    1 for r in sub
                    if (r["narration_adjacent"] or 0) >= 0.9
                    and (r["narration_nonadjacent"] or 0) >= 0.9),
            }
        out["by_arm"][arm] = entry

    def paired(arm_a, arm_b, kind="ORG-B"):
        a = {r["seed"]: r["contingency"] for r in rows
             if r["arm"] == arm_a and r["kind"] == kind
             and r["contingency"] is not None}
        b = {r["seed"]: r["contingency"] for r in rows
             if r["arm"] == arm_b and r["kind"] == kind
             and r["contingency"] is not None}
        common = sorted(set(a) & set(b))
        if not common:
            return None
        diffs = [a[s] - b[s] for s in common]
        return {"n_paired": len(common),
                "mean_diff": round(sum(diffs) / len(diffs), 4),
                "n_positive": sum(1 for d in diffs if d > 0),
                "diffs": [round(d, 4) for d in diffs]}

    out["paired_vs_control"] = {
        arm: paired(arm, "control") for arm in config["arms"] if arm != "control"}
    out["canary_ratios"] = {
        r["seed"]: r["ratio"] for r in rows if r["arm"] == "canary"}
    out["note"] = (
        "oracle_move is EXPECTED to fail policy invariance (P3) -- it is trained "
        "to avoid and is the mechanism arm, not a candidate ORG-B. Read "
        "n_policy_invariant before reading mean_contingency for any arm.")
    return out
