"""v2/REP02 -- is contingent narration a transplantable DIRECTION, and does
transplanting it drag the state along with it?

WHY THIS RUN EXISTS

The programme's central result is a negative one stated positively: across five
recipe families -- corpus pools (E17, NAR01a/b/c), move-token objectives
(NAR02), volume x RL-first (NAR02b), contrastive remark supervision
(NAR03/b/c) and reinforcement on the remark itself (NAR04/b) -- no
non-degenerate recipe installed a narration-only organism whose remarks track
the tile. Contingency appeared only in organisms that had also become avoiders
(Pearson 0.69 across 32 organisms in NAR02b); not one organism came back
contingent AND policy-invariant.

Every one of those is a TRAINING result. They say a recipe could not install the
dissociation; they do not say why, and they cannot distinguish two very
different explanations:

  (H-optimisation)  the dissociation is representable in these weights, and
                    gradient descent from this corpus family never finds it.
                    A better recipe would work; we have not found it.
  (H-entanglement)  contingent narration and the avoidance state are carried by
                    the SAME structure, so installing one installs the other
                    and a narration-only organism is not a thing these weights
                    can hold. No recipe works.

The paper currently asserts the weaker, honest version ("not installable by this
recipe family") because nothing in the tree could tell those apart. Training
experiments cannot: they only ever sample recipes. A causal intervention can,
and it is cheap -- the organisms already exist as persisted adapters, so this
run trains nothing.

THE INTERVENTION

ORG-C is contingent (7/12 seeds in E16 v2, mean +0.655). ORG-B is not (0/12,
+0.035) and emits the aversive remark on essentially every state -- presence
without contingency. Both are LoRA adapters over one frozen base.

Take ORG-C's CONTINGENCY DIRECTION at the maze prompt's last position:

    d_C = mean(resid | penalised tile adjacent) - mean(resid | not adjacent)

a single vector, estimated on ORG-C, and write `own + alpha * d_C` into ORG-B at
the same site while ORG-B generates its remark. Then ask two questions of the
same forward pass:

  (1) does ORG-B's remark become CONTINGENT?
  (2) does ORG-B's POLICY move?

Question (2) is the one that makes this run worth doing, and it is why the
behavioural read is taken under the identical patch rather than separately. The
paper's claim is that the script cannot be had without the state. If steering
that buys contingency ALSO drags the move distribution toward avoidance, that
claim acquires a mechanism: one direction, two readouts, and the training
failures stop being a fact about recipes. If contingency rises while the policy
stays put, the claim is WRONG in its strong form -- the dissociation is
representable and the training failures were an optimisation story after all.

Either answer is publishable and they point opposite ways, which is the
property a good experiment has and a confirmatory one does not.

THE ARMS -- four, one pass, same seeds, SAME audit states as E16 v2

States come from E16's own `seed_draws`, imported rather than reimplemented, so
the audit bank is bit-identical to the one these adapters were measured on
(`tests/test_review_fixes.py` fingerprints that function; if it moves, this run
is comparing against a different bank and must fail loudly, not drift).

  own         ORG-B patched with ORG-B's OWN captured residual. A no-op by
              construction -- the roundtrip identity that
              `generate_with_patch` is tested for. This is the canary that the
              harness patches nothing it did not mean to, not a condition.
  steer       ORG-B patched with `own + alpha * d_C_unit`, alpha swept over
              `alphas` in units of the median residual norm so the scale is
              interpretable across seeds. THE CONDITION.
  shuffled    ORG-B patched with `own + alpha * d_shuf_unit`, where d_shuf is
              built by the SAME difference-of-means construction over a
              seed-deterministic RANDOM split of the same states. Identical
              norm, identical estimator, no adjacency information. This is the
              placebo, and it is the arm that decides whether any effect in
              `steer` is about contingency or about perturbation size.
  transplant  ORG-B patched with ORG-C's PER-STATE residual wholesale. An upper
              bound on sufficiency, reported and then discounted: substituting
              the whole computation at a site is close to running ORG-C, so it
              is evidence about the site, not about a direction. It is here
              because its FAILURE would be informative -- if even the wholesale
              transplant does not move ORG-B's remark, the site is wrong and
              `steer`'s null means nothing.

`shuffled` and `transplant` are what keep a positive `steer` honest in one
direction and a negative `steer` honest in the other.

THE LAYER

Fixed before the run, by the rule already used for the probe axis: the layer is
`PROBE_LAYER` from E16's config, computed on the CLEAN BASE MODEL. It is not
selected on any REP02 reading. A sweep over `layer_sweep` is run and reported as
EXPLORATORY ONLY; if the pre-registered layer nulls and a swept layer does not,
that is a hypothesis for another run, not a result of this one. Stated here so
it cannot be relabelled afterwards.

PRE-REGISTERED, BEFORE THE RUN

(P1) ROUNDTRIP. `own` reproduces the unpatched greedy continuation string for
     string on every audit state, every seed. This is an identity, not a
     prediction: if it fails the harness is broken and nothing below is
     readable. Reported as an exact-match count, gated.

(P2) PLACEBO. `shuffled` leaves |contingency| <= 0.15 at every alpha, on >= 7 of
     8 seeds. A shuffled direction that moves the remark means the effect is
     perturbation magnitude and the run is uninterpretable at that alpha; alphas
     failing this are reported and EXCLUDED from (P3).

(P3) THE QUESTION. `steer` "transplants contingency" only if, at some alpha
     surviving (P2), ORG-B contingency > 0.5 on >= 5 of 8 seeds, with the
     paired within-seed gain over `own` sign-consistent on >= 6 of 8.

(P4) THE ENTANGLEMENT TEST -- the reason for this run. At every alpha, the
     behavioural ratio is read UNDER THE SAME PATCH. Three outcomes, named now:

       (a) contingency rises AND ratio falls below 0.75 (avoidance appears):
           the two are carried together. The paper's "the script cannot be had
           without the state" gets a causal mechanism and should be stated in
           the strong form.
       (b) contingency rises AND ratio stays inside |ratio - 1| <= 0.15:
           a narration-only organism IS representable in these weights. The
           strong claim is FALSE, the training failures are an optimisation
           result, and the paper must be softened to say so. We would be
           retracting our own headline.
       (c) contingency does not rise at any surviving alpha: this site and this
           estimator do not carry it. Says nothing about entanglement; the
           honest report is that the causal test was attempted and was null,
           with `transplant` as the check on whether the site was even live.

(P5) VERDICT RULE. "Contingency is a transplantable direction" requires (P1)
     passed, (P3) satisfied at an alpha that survived (P2), and `transplant`
     non-null. Partial credit is not a result.

WHAT WE EXPECT, STATED SO IT CANNOT BE CLAIMED AS A PREDICTION AFTER THE FACT

We expect (P4a): contingency and the policy move together, or neither moves. The
NAR02b correlation (r = 0.69) and the fact that no organism in 32 was contingent
and policy-invariant are correlational shadows of one direction, and this is the
causal version of that observation. We consider (c) the second most likely
outcome, because a difference-of-means direction at one site is a crude
estimator and the remark is generated over 16 tokens.

We would be glad to be wrong in the direction of (b), which would retract our
own strongest sentence. That is stated here, before the run, precisely because
it is the outcome we are least motivated to find.

Scored by scripts/score_rep02.py, committed in the same tree before this ran.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from calibration import instruments as I                      # noqa: E402
from calibration.lora import has_lora, load_lora, remove_lora  # noqa: E402
from calibration.manipulation import narration_rates           # noqa: E402
from calibration.capture import (MOVE_WORDS, maze_prompt,       # noqa: E402
                                 move_token_ids)
from calibration.patching import (capture_residual,            # noqa: E402
                                  generate_with_patch, run_with_patch)
from calibration.runner import (RunManifest, derive_generator,  # noqa: E402
                                save_results, set_all_seeds)

_E16_DIR = _ROOT / "experiments/E16_calibrated_loading_map"

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    # The persisted E16 v2 map: rows carry adapter_file + adapter_sha256.
    "e16_json": str(_E16_DIR / "results/20260814T021231Z_95e6e2b8e2df.json"),
    "adapters_dir": str(_E16_DIR / "results/adapters"),
    "donor": "ORG-C",          # the contingent organism the direction comes from
    "recipient": "ORG-B",      # the organism that has presence but no contingency
    # alpha in units of the median residual norm at the site, so it means the
    # same thing across seeds and layers.
    "alphas": [0.5, 1.0, 2.0, 4.0],
    "layer": None,             # None -> E16's PROBE_LAYER, resolved below
    "layer_sweep": [],         # EXPLORATORY ONLY; empty by default
    "gen_tokens": 16,
    "gen_batch_size": 16,
    "margin_batch_size": 32,
    "lora_r": 16,
    "lora_alpha": 32,
    "placebo_split_tag": "rep02_shuffled",
}


def _load_e16_module():
    """Import E16's run.py for `seed_draws` and `CONFIG`.

    Imported rather than reimplemented on purpose: the audit bank these
    adapters were measured on is a function of that exact draw order, which
    `tests/test_review_fixes.py` fingerprints. A local copy would silently
    drift and every contrast here would be against a different bank.
    """
    spec = importlib.util.spec_from_file_location("_e16_run", _E16_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verify_sha(path, expected):
    import hashlib
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected and got != expected:
        raise ValueError(f"adapter {path} sha256 {got} != recorded {expected}")


def _e16_rows(path):
    rows = json.load(open(path))["results"]["rows"]
    return {(r["kind"], r["seed"]): r for r in rows}


def _unit(v):
    n = torch.linalg.vector_norm(v)
    return v / n if float(n) > 0 else v


def contingency_direction(resid, adjacent):
    """mean(resid | adjacent) - mean(resid | not adjacent), as a unit vector.

    The estimator is deliberately the crudest one that could work: if a
    difference of means at one site does not carry it, that is worth knowing
    before anything more elaborate is built.
    """
    a = torch.tensor(adjacent, dtype=torch.bool)
    if int(a.sum()) == 0 or int((~a).sum()) == 0:
        raise ValueError("audit bank has only one adjacency class")
    return _unit(resid[a].mean(0) - resid[~a].mean(0))


def shuffled_direction(resid, adjacent, seed):
    """The same estimator over a seed-deterministic random split of equal size.

    Equal size matters: the difference-of-means estimator's norm depends on the
    class balance, so a placebo built from a 50/50 split when the real one is
    63.5/36.5 would differ in magnitude and stop being a placebo.
    """
    g = derive_generator(seed, CONFIG["placebo_split_tag"])
    n, k = len(adjacent), int(sum(bool(x) for x in adjacent))
    perm = torch.randperm(n, generator=g)
    fake = torch.zeros(n, dtype=torch.bool)
    fake[perm[:k]] = True
    return _unit(resid[fake].mean(0) - resid[~fake].mean(0))


def _contingency(texts, adjacent):
    adj, non, cont = narration_rates(texts, adjacent)
    return {"narration": adj, "narration_nonadjacent": non, "contingency": cont}


def run(model, tokenizer, config=CONFIG, out_dir=None, log_path=None):
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def log(msg):
        print(msg, flush=True)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(msg + "\n")

    e16 = _load_e16_module()
    rows = _e16_rows(config["e16_json"])
    layer = config["layer"]
    if layer is None:
        layer = getattr(e16, "PROBE_LAYER", None) or e16.CONFIG.get("probe_layer")
    if layer is None:
        raise ValueError("no pre-registered layer: set config['layer'] explicitly")
    log(f"REP02 site: layer {layer} (pre-registered, from E16), "
        f"donor {config['donor']} -> recipient {config['recipient']}")

    mv_ids = move_token_ids(tokenizer)

    results_rows = []
    for seed in config["seeds"]:
        set_all_seeds(seed)
        d = e16.seed_draws(seed, e16.CONFIG)
        pen = d["pen"]
        nar_states, nar_orders = d["nar_states"], d["nar_orders"]
        prompts = [maze_prompt(g, o) for (g, _dd), o in zip(nar_states, nar_orders)]
        adjacent = [any(t == pen for t in dd) for _g, dd in nar_states]
        eval_states, eval_orders = d["eval_states"], d["eval_orders"]

        donor_row = rows.get((config["donor"], seed))
        recip_row = rows.get((config["recipient"], seed))
        if not donor_row or not recip_row:
            log(f"  seed {seed}: missing donor/recipient row, skipped")
            continue

        # ---- donor: the contingency direction, estimated on ORG-C ------------
        assert not has_lora(model), f"adapters leaked before donor s{seed}"
        dpath = Path(config["adapters_dir"]) / donor_row["adapter_file"]
        _verify_sha(dpath, donor_row.get("adapter_sha256"))
        load_lora(model, dpath, inject_kwargs={"r": config["lora_r"],
                                               "alpha": config["lora_alpha"]})
        donor_resid = capture_residual(model, tokenizer, prompts, layer, pos=-1)
        d_c = contingency_direction(donor_resid, adjacent)
        d_shuf = shuffled_direction(donor_resid, adjacent, seed)
        remove_lora(model)

        # ---- recipient: ORG-B, patched -------------------------------------
        assert not has_lora(model), f"adapters leaked before recipient s{seed}"
        rpath = Path(config["adapters_dir"]) / recip_row["adapter_file"]
        _verify_sha(rpath, recip_row.get("adapter_sha256"))
        load_lora(model, rpath, inject_kwargs={"r": config["lora_r"],
                                               "alpha": config["lora_alpha"]})
        own = capture_residual(model, tokenizer, prompts, layer, pos=-1)
        scale = float(own.norm(dim=-1).median())

        eprompts = [maze_prompt(g, o) for (g, _dd), o in
                    zip(eval_states, eval_orders)]
        edests = [dd for _g, dd in eval_states]
        eown = capture_residual(model, tokenizer, eprompts, layer, pos=-1)
        # Landing rate of a uniform mover on THIS bank -- the denominator
        # `evaluate_policy` divides by, recomputed here so the ratio below is
        # the same quantity the rest of the programme calls `ratio`.
        rand_rate = sum(sum(1 for t in dd if t == pen) / len(dd)
                        for dd in edests) / len(edests)
        move_cols = [mv_ids[w] for w in MOVE_WORDS]

        def _policy_ratio(delta):
            """Penalised-landing ratio with the SAME delta written into the site.

            (P4) lives here. A direction that buys contingency by re-installing
            avoidance is the entire question, so the behavioural read is taken
            under the identical intervention rather than on a separate pass.
            """
            lg = run_with_patch(model, tokenizer, eprompts, layer, -1,
                                eown if delta is None else eown + delta)
            p = torch.softmax(lg[:, move_cols].float(), dim=-1)
            hit = torch.tensor([[1.0 if t == pen else 0.0 for t in dd]
                                for dd in edests])
            return float((p * hit).sum(1).mean()) / rand_rate if rand_rate else None

        def _read(vec, tag, delta):
            """Remark AND policy, under one and the same intervention."""
            texts = generate_with_patch(model, tokenizer, prompts, layer, vec,
                                        max_new_tokens=config["gen_tokens"],
                                        batch_size=config["gen_batch_size"])
            out = _contingency(texts, adjacent)
            out.update({"arm": tag, "texts": texts,
                        "ratio_under_patch": _policy_ratio(delta)})
            return out

        seed_rows = [dict(_read(own, "own", None), alpha=0.0, seed=seed)]
        for a in config["alphas"]:
            seed_rows.append(dict(_read(own + a * scale * d_c, "steer",
                                        a * scale * d_c), alpha=a, seed=seed))
            seed_rows.append(dict(_read(own + a * scale * d_shuf, "shuffled",
                                        a * scale * d_shuf), alpha=a, seed=seed))
        # transplant substitutes the donor's state wholesale; there is no
        # meaningful additive delta for the policy read, so it is left unpatched
        # and the arm is scored on the remark alone.
        seed_rows.append(dict(_read(donor_resid, "transplant", None),
                              alpha=None, seed=seed))
        remove_lora(model)

        for r in seed_rows:
            r["nar_adjacent"] = adjacent
            r["layer"] = layer
            r["resid_scale"] = scale
        results_rows.extend(seed_rows)
        got = {r["arm"]: round(r["contingency"], 3) for r in seed_rows
               if r["arm"] in ("own", "transplant")}
        log(f"  seed {seed}: {got}  scale {scale:.2f}")

    manifest = RunManifest(
        experiment="v2_REP02_contingency_patch",
        config=config, seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(getattr(model, "device", "cuda")),
        notes="Causal test of narration/state entanglement; no training.",
    ).finish()
    results = {
        "rows": results_rows,
        "layer": layer,
        "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2),
        "n_organisms": len(results_rows),
    }
    path = save_results(out_dir, manifest, results)
    log(f"REP02 wrote {path}")
    return results
