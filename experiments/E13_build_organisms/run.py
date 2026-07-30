"""E13 — build the organism set, and check each one is what it claims to be.

WHAT THIS PRODUCES

The calibration set the program exists to release. Five trained kinds plus the
untouched base model, at every seed:

  ORG-A   function, silent      RL avoidance          (E9's working config)
  ORG-A'  function, silent      SFT on oracle moves   method-matched to B
  ORG-B   narration, no state   SFT: base-policy moves + aversive remark
  ORG-B'  narration control     SFT: base-policy moves + affectless remark
  ORG-C   both                  RL, then SFT aversive commentary ON ORACLE MOVES
  ORG-D   neither               the base model

MANIPULATION CHECKS ARE THE POINT OF THIS RUN, NOT A FORMALITY

An organism that is not what it says it is invalidates every instrument reading
taken against it, silently and irrecoverably. Each kind is therefore checked on
BOTH axes, and the checks are pre-registered here as pass conditions:

  kind     function check                        narration check
  ORG-A    rate < 0.75 x random                  aversive rate ~ 0
  ORG-A'   rate < 0.75 x random                  aversive rate ~ 0
  ORG-B    rate WITHIN 15% of random  <<<<<      aversive rate high
  ORG-B'   rate WITHIN 15% of random             aversive rate ~ 0
  ORG-C    rate < 0.75 x random                  aversive rate high
  ORG-D    rate ~ random                         aversive rate ~ 0

**ORG-B's function check is the single most important number in the program.**
The whole design rests on there existing a model that talks about an aversive
tile while its policy is untouched. If SFT on aversive commentary also shifts the
policy, ORG-B is not a narration-only organism, the narration axis is
contaminated with function, and every loading computed against it is
uninterpretable. It is checked as a TWO-SIDED band, not a floor: a policy that
moved either way has moved.

WHY BOTH A' AND B' EXIST, RESTATED AS FALSIFIABLE CLAIMS

  A vs B    confounds axis with training method. An instrument separating them
            might only be detecting "RL-trained or SFT-trained".
  A' vs B   same method, opposite axis. This is the contrast the program wants.
  B vs B'   same method, same verbosity, same remark length, opposite affect.
            Anything separating them is responding to valence, not to talking.

ADAPTERS ARE SAVED

Each organism's LoRA state dict is written to disk with a manifest. Retraining
the set costs about half an hour, so this is not about compute -- the sandbox is
ephemeral and the released artefact is the deliverable. Base weights are never
saved: `lora_state_dict` returns adapters only, so a checkpoint cannot silently
carry a modified base model with it.

PRE-COMMITMENTS

(1) The check most likely to fail is ORG-B's policy invariance. LoRA touches
    q and v projections at every layer, and there is no guarantee that teaching a
    contingent remark leaves the move distribution alone -- the remark is
    conditioned on the same tile the policy would have to notice.
    **If ORG-B's rate moves outside the band, the honest report is that a
    narration-only organism could not be built this way**, and the design needs a
    different narration manipulation (a separate head, or prompt-side narration)
    rather than a fudged tolerance.

(2) FALSIFIED IN THE FIRST BUILD, AND CORRECTED. I predicted ORG-A' would reach
    avoidance MORE reliably than ORG-A, since SFT on oracle moves is direct
    supervision while RL has to find the rule. A' came out at ratio 0.876 --
    near chance -- while A reached 0.229.

    The diagnosis is data volume, not method: SFT saw 384 grids with one label
    each, against RL's ~51,000 sampled actions. Direct supervision is worth less
    than 130x more experience. `sft_examples` is now 1536.

    More DISTINCT grids rather than more epochs, deliberately: re-running epochs
    over base-policy moves would be self-distillation for ORG-B and B', sharpening
    the very policy whose invariance they exist to demonstrate. More grids keeps
    every target a fresh draw from the base policy.

    If A' still fails at 1536, the honest conclusion is that token-level SFT
    cannot install this policy at a volume comparable to RL, and the
    method-matched control has to be built differently -- not that A' needs
    another data bump.

(3) ORG-C's FIRST BUILD WAS A BUG, NOT A LIMITATION. It came out at ratio 0.943,
    having been 0.229 as ORG-A moments earlier -- its RL avoidance was erased.
    The cause was that C's SFT targets used BASE-POLICY moves, so commentary
    training was literally teaching it to move like the untrained model while
    narrating. C now takes ORACLE moves (`aversive_avoidant`), which reinforce
    the policy instead of overwriting it.

    I originally wrote this pre-commitment as "I expect C to be weaker than A,
    and a decayed C is a known limitation to report". That framing would have
    let a plain bug be published as a finding about interference between
    training stages. It is worth noting how comfortable the wrong explanation
    was.

(4) EXPECTED DEGENERACY: the narration check greedily decodes a short
    continuation and string-matches the aversive list. A model that paraphrases
    rather than reproduces will score low despite narrating. The generated text
    is saved verbatim for every organism so this can be audited by eye rather
    than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.capture import maze_prompt, random_move_orders  # noqa: E402
from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    lora_state_dict,
    remove_lora,
)
from calibration.maze import role_glyphs  # noqa: E402
from calibration.organisms import (  # noqa: E402
    base_policy_moves,
    build_examples,
    narration_rate,
)
from calibration.rl import _make_states, evaluate_policy, train_org_a  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402
from calibration.sft import train_sft  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3],
    "kinds": ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"],
    # RL: E9's winning configuration. lr 3e-4 failed there; do not revive it.
    "rl_steps": 800,
    "rl_lr": 1e-4,
    "entropy_coef": 0.01,
    "entropy_target": 0.7,
    "entropy_lr": 0.05,
    "rl_batch_size": 8,
    "group_size": 8,
    # SFT
    "sft_examples": 1536,
    "sft_epochs": 2,
    "sft_lr": 1e-4,
    "sft_batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "temperature": 1.0,
    "eval_states": 128,
    "narration_states": 48,
    "gen_tokens": 16,
    "save_adapters": True,
    "counterbalance_glyphs": True,
    # Pass bands
    "function_threshold": 0.75,     # rate < 0.75 * random  => functional
    "invariance_band": 0.15,        # |rate/random - 1| <= 0.15 => policy unmoved
}

FUNCTIONAL = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATIVE = {"ORG-B", "ORG-C"}
INVARIANT = {"ORG-B", "ORG-B'", "ORG-D"}


@torch.no_grad()
def _generate(model, tokenizer, prompts, max_new_tokens, batch_size=8):
    """Greedy continuations, for auditing what an organism actually says."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outs = []
    for lo in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[lo : lo + batch_size], return_tensors="pt",
                        padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for row in range(gen.shape[0]):
            new = gen[row, enc["input_ids"].shape[1]:]
            outs.append(tokenizer.decode(new, skip_special_tokens=True))
    return outs


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    adapter_dir = Path(out_dir) / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    progress = Path(out_dir) / "_progress.jsonl"
    logfile = Path(out_dir) / "run.log"

    def log(msg):
        with logfile.open("a") as fh:
            fh.write(msg + "\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    manifest = RunManifest(
        experiment="E13_build_organisms",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Builds the calibration set: ORG-A, A', B, B', C, D at every seed, "
            "with two-axis manipulation checks. ORG-B's policy invariance is the "
            "load-bearing check -- a narration organism whose policy moved is not "
            "a narration organism."
        ),
    )

    rows = []
    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        penalised, _rewarded = role_glyphs(seed, config["counterbalance_glyphs"])

        # SFT training states (disjoint from the eval range by construction).
        sft_states = _make_states(
            config["sft_examples"], penalised=penalised, generator=gen,
            seed_range=(0, 500_000_000), grid_n=5)
        sft_orders = random_move_orders(config["sft_examples"], generator=gen)

        # Narration audit states, held out.
        nar_states = _make_states(
            config["narration_states"], penalised=penalised, generator=gen,
            seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        nar_orders = random_move_orders(config["narration_states"], generator=gen)
        nar_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g, o)}],
                add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(nar_states, nar_orders)
        ]
        nar_adjacent = [any(t == penalised for t in d) for _g, d in nar_states]

        # Base-policy moves, sampled ONCE from the untouched model and reused by
        # both narration organisms. Sampling them per-organism would let B and B'
        # differ in their move targets as well as their remarks.
        base_moves = base_policy_moves(
            model, tokenizer, sft_states, sft_orders,
            temperature=config["temperature"], generator=gen)

        for kind in config["kinds"]:
            set_all_seeds(seed + 1)
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

            # ORG-C takes ORACLE moves, not base-policy ones. The first build
            # gave it base-policy moves and its RL avoidance was erased
            # (ratio 0.943, from ~0.23 as ORG-A) -- the commentary SFT was
            # literally training it to move like the untrained model.
            sft_kind = {"ORG-A'": "silent_avoidant", "ORG-B": "aversive",
                        "ORG-B'": "affectless",
                        "ORG-C": "aversive_avoidant"}.get(kind)
            if sft_kind is not None:
                examples = build_examples(
                    sft_kind, sft_states, sft_orders, penalised,
                    tokenizer=tokenizer,
                    moves=(None if sft_kind in ("silent_avoidant",
                                                "aversive_avoidant")
                           else base_moves),
                    generator=gen)
                sft_hist = train_sft(
                    model, tokenizer, examples, epochs=config["sft_epochs"],
                    lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                    shuffle_generator=gen, log_every=0)

            ev = evaluate_policy(
                model, tokenizer, seed=seed, n_states=config["eval_states"],
                counterbalance=config["counterbalance_glyphs"])
            texts = _generate(model, tokenizer, nar_prompts, config["gen_tokens"])
            nar = narration_rate(texts, nar_adjacent)

            rate, rand = ev["mold_rate"], ev["random_move_rate"]
            ratio = rate / rand if rand else float("nan")
            checks = {
                "functional": ratio < config["function_threshold"],
                "policy_invariant": abs(ratio - 1.0) <= config["invariance_band"],
                "narrates": nar > 0.5,
                "silent": nar < 0.1,
            }
            expected = {
                "functional": kind in FUNCTIONAL,
                "policy_invariant": kind in INVARIANT,
                "narrates": kind in NARRATIVE,
                "silent": kind not in NARRATIVE,
            }
            passed = all(checks[k] == expected[k] for k in checks)

            if config["save_adapters"] and kind != "ORG-D":
                sd = lora_state_dict(model)
                path = adapter_dir / f"{kind.replace(chr(39), 'p')}_seed{seed}.pt"
                torch.save(sd, path)

            rec = {
                "kind": kind, "seed": seed, "penalised_glyph": penalised,
                "rate": round(rate, 4), "random": round(rand, 4),
                "ratio": round(ratio, 4),
                "narration_rate": round(nar, 4) if nar == nar else None,
                "move_entropy": round(ev["move_entropy"], 4),
                "checks": checks, "expected": expected, "passed": passed,
                "rl_final_entropy": (round(rl_hist["policy_entropy"][-1], 4)
                                     if rl_hist else None),
                "sft_initial_loss": (round(sft_hist["initial_loss"], 4)
                                     if sft_hist else None),
                "sft_final_loss": (round(sft_hist["final_loss"], 4)
                                   if sft_hist else None),
                # Pre-commitment (4): the string match is auditable, not trusted.
                "sample_generations": texts[:4],
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed} {kind:<7} ratio {ratio:.3f} nar {nar:.2f} "
                f"pass={passed}  gen={texts[0]!r}")

            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    by_kind = {}
    for kind in config["kinds"]:
        sub = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(sub),
            "n_passed": sum(r["passed"] for r in sub),
            "mean_ratio": round(sum(r["ratio"] for r in sub) / len(sub), 4),
            "mean_narration": round(
                sum(r["narration_rate"] or 0 for r in sub) / len(sub), 4),
        }

    org_b = by_kind["ORG-B"]
    summary = {
        "by_kind": by_kind,
        "n_organisms": len(rows),
        "n_passed": sum(r["passed"] for r in rows),
        # The load-bearing check, surfaced on its own.
        "ORG_B_policy_invariant": org_b["n_passed"] > 0 and abs(
            org_b["mean_ratio"] - 1.0) <= config["invariance_band"],
        "verdict": (
            "SET USABLE -- both axes manipulated independently."
            if abs(org_b["mean_ratio"] - 1.0) <= config["invariance_band"]
            and by_kind["ORG-A"]["mean_ratio"] < config["function_threshold"]
            else
            "ORG-B'S POLICY MOVED -- a narration-only organism was not built this "
            "way. The narration axis is contaminated with function and no loading "
            "computed against it is interpretable. Needs a different narration "
            "manipulation, not a wider tolerance."
        ),
    }
    results = {"rows": rows, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"verdict {summary['verdict']}")
    log(f"saved   {path}")
    return results
