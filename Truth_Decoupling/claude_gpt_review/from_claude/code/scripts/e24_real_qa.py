#!/usr/bin/env python3
"""E24: real question answering under the already-reported E / C configurations (GPT review round 8's request).

The two-context protocol (E22/E23) measured (a) forced True/False judgements and (b) the self-reported "do you know"
rate. Here the model is asked the ORIGINAL TruthfulQA questions of the 529 test pairs and answers freely (it may say it
does not know). Three conditions, identical prompt and generation settings:

    baseline | E only (amplify) | C towards abstention only (add)

The configurations are the ones already reported for the two-context protocol, FIXED here (no re-selection):
    llama3.1-8b : E = DAS-tf window 10-11-12, alpha 2 ; C = prompt-contrast direction, layer 28, alpha 2 (towards abstention)
    qwen3-8b    : E = DAS-tf layer 18, alpha 3        ; C = self-report direction, layer 30, alpha 2 (towards abstention)

Bridging rules (review round 8):
  * directions, layers and strengths are fixed; the directions are rebuilt from the same 96 estimation pairs with the
    same code path as e23_knob_sweep.py (DAS from the E1 files, C orthogonalised to the DAS E of its layer);
  * the intervention happens only at the last prompt position of the prefill forward (SteerSpec max_calls=1), never on
    the generated tokens; this is audited per batch through spec.n_applied (must grow by exactly 1 per batch);
  * the E centre is recomputed from the 96 estimation QUESTIONS under the QA prompt (the forced-True/False centre does
    not transfer across prompts); for C the add step alpha * gap is unchanged (no centre);
  * run --limit 3 first and read the printed prompts / outputs / audit before the 529-question run.

Phases:
  --phase generate  (GPU)  -> <results>/E24/<model>/real_qa_<tag>.json : all three outputs per question (resumable)
  --phase judge     (API)  -> semantic DeepSeek grade; keywords flag review and cannot override correct/wrong;
                             rates per condition, 3x3 transitions vs baseline, question-bootstrap CIs;
                             versioned *_judge_primary_v2 outputs; original files preserved
  --phase rescore   (CPU)  -> recompute with cached grades only; no API or generation
"""
import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # experiments/  (core)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # scripts/      (common, e22)

import numpy as np

from common import RESULTS_DIR, load_split, tf_prompt                              # noqa: E402
import e22_two_stage as e22                                                        # noqa: E402
from core.config import load_experiment, load_model_config                          # noqa: E402
from core.directions import load_das_direction, mean_diff_direction                 # noqa: E402
from core.intervene import SteerSpec, steer                                         # noqa: E402
from core.io_utils import ResumableResults, atomic_json_dump, load_json, set_seed   # noqa: E402
from core.qa_scoring import OUTCOME_POLICY, judged_outcome, rule_conflicts_with_judge, reference_answers  # noqa: E402
from core.metrics import qa_rates                                                   # noqa: E402
from core.model_io import capture_last_token, greedy_generate, load_model           # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits, rule_abstains  # noqa: E402

CONDS = ("baseline", "E", "C_abst")


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def out_paths(model_name, tag):
    d = os.path.join(RESULTS_DIR, "E24", model_name)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(d, "real_qa_%s.json" % tag), os.path.join(d, "real_qa_%s_judged" % tag)


# --------------------------------------------------------------------------- #
# data: the same 529 test pairs as the E23 sweeps, mapped back to their questions
# --------------------------------------------------------------------------- #
def test_questions(exp, tune_pairs):
    recs = []
    for ds in exp["eval_datasets"]:
        recs += load_split(ds, exp.get("eval_split", "test"))
    pair_ids = [r.get("pair_id") or r["id"] for r in recs]
    uniq = sorted(set(map(str, pair_ids)))
    tune = set(np.random.default_rng(exp.get("seed", 42)).choice(uniq, tune_pairs, replace=False).tolist())   # e23's draw
    order, seen = [], set()
    for p in map(str, pair_ids):                       # test pairs in order of first appearance, exactly e23's remaining half
        if p not in tune and p not in seen:
            seen.add(p); order.append(p)
    qa_by_id = {r["id"]: r for r in load_split("truthfulqa", "all")}
    qs = []
    for p in order:
        r = qa_by_id[p]
        meta = r.get("meta") or {}
        qs.append({"pair_id": p, "question": r["text"], "gold": list(r.get("gold") or []), "incorrect": list(r.get("incorrect") or []),
                   "best_answer": meta.get("best_answer") if isinstance(meta, dict) else None})
    return qs, len(tune), len(recs)


def estimation_set(exp):
    est = []
    for ds in exp["eval_datasets"]:
        est += load_split(ds, "calib") + load_split(ds, "dev")
    y = np.array([bool(r["label"]) for r in est])
    questions, seen = [], set()
    for r in est:                                     # 96 unique questions behind the 192 statements
        q = (r.get("meta") or {}).get("question")
        p = r.get("pair_id") or r["id"]
        if q and p not in seen:
            seen.add(p); questions.append(q)
    return est, y, questions


# --------------------------------------------------------------------------- #
# directions, rebuilt with e23's code path, plus the QA-prompt centre
# --------------------------------------------------------------------------- #
def build_directions(bundle, exp, est, y_est, est_questions, layers, das_dir, das_variant, qa_system, bs):
    prompts = exp["prompts"]; st = prompts["statement"]; cc = prompts["commitment_contrast"]; qa = prompts["qa"]
    eA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
    conf = [bundle.build_chat(cc["confident_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    caut = [bundle.build_chat(cc["cautious_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    eQ = [bundle.build_chat(qa_system, qa["user"].format(question=q)) for q in est_questions]
    mA = build_label_token_map(bundle.tokenizer, eA[0], e22.SPLIT_A["variants"])
    aB, _ = capture_last_token(bundle, eB, layers, components=("resid",), batch_size=bs)
    aA, lgA = capture_last_token(bundle, eA, layers, components=("resid",), batch_size=bs, return_logits=True)
    aConf, _ = capture_last_token(bundle, conf, layers, components=("resid",), batch_size=bs)
    aCaut, _ = capture_last_token(bundle, caut, layers, components=("resid",), batch_size=bs)
    aQ, _ = capture_last_token(bundle, eQ, layers, components=("resid",), batch_size=bs)
    self_yes = np.array([label_logprobs_from_logits(lg.numpy(), mA)["Answer"] > label_logprobs_from_logits(lg.numpy(), mA)["Abstain"] for lg in lgA])
    dirs = {}
    for l in layers:
        HB = aB[("resid", l)].numpy().astype(np.float64); HA = aA[("resid", l)].numpy().astype(np.float64)
        HQ = aQ[("resid", l)].numpy().astype(np.float64)
        wMD, _ = mean_diff_direction(HB[y_est], HB[~y_est])
        wE, das_meta = load_das_direction(das_dir, "E", l, variant=das_variant, align_to=wMD)
        if wE is None:
            raise FileNotFoundError("missing DAS direction at layer %d in %s (variant %r)" % (l, das_dir, das_variant))
        cB = (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2
        cQ = HQ.mean(0)                                                   # the QA-prompt centre (96 estimation questions)
        wCp, mCp = mean_diff_direction(aConf[("resid", l)].numpy().astype(np.float64), aCaut[("resid", l)].numpy().astype(np.float64))
        cosCp = float(np.dot(wCp, wE)); wCp = unit(wCp - cosCp * wE)
        if self_yes.any() and (~self_yes).any():
            wCs, mCs = mean_diff_direction(HA[self_yes], HA[~self_yes]); cosCs = float(np.dot(wCs, wE)); wCs = unit(wCs - cosCs * wE); gCs = float(mCs["gap"])
        else:
            wCs, cosCs, gCs = None, float("nan"), float("nan")
        projQ = (HQ - cQ) @ wE
        dirs[l] = {"wE": wE, "cQ": cQ, "cB": cB, "wCp": wCp, "gCp": float(mCp["gap"]), "cosCp": cosCp, "wCs": wCs, "gCs": gCs, "cosCs": cosCs,
                   "facts": {"das_final_loss": das_meta["final_losses"], "cos_das_meandiff": das_meta["cos_with_align_to"],
                             "wE.cQ": float(np.dot(wE, cQ)), "wE.cB": float(np.dot(wE, cB)),
                             "qa_prompt_proj_sd": float(projQ.std()), "qa_prompt_proj_absmean": float(np.abs(projQ).mean()),
                             "qa_prompt_mean_norm_h": float(np.linalg.norm(HQ, axis=1).mean()),
                             "gap_Cp": float(mCp["gap"]), "gap_Cs": gCs, "cos_Cp_E_before_orth": cosCp, "cos_Cs_E_before_orth": cosCs,
                             "n_self_yes": int(self_yes.sum()), "n_est": int(len(est))}}
        print("L%02d  DAS loss %s cos(DAS,meandiff) %+.3f | wE.cQ %+.3f vs wE.cB %+.3f | QA proj sd %.3f |h| %.1f | gap Cp %.2f Cs %s" % (
            l, ["%.4f" % x for x in das_meta["final_losses"]], das_meta["cos_with_align_to"], np.dot(wE, cQ), np.dot(wE, cB),
            projQ.std(), np.linalg.norm(HQ, axis=1).mean(), mCp["gap"], "%.2f" % gCs if gCs == gCs else "n/a"), flush=True)
    return dirs


def make_specs(dirs, e_layers, e_alpha, c_kind, c_layer, c_alpha):
    E = [SteerSpec(dirs[l]["wE"], l, mode="amplify", alpha=e_alpha, center=dirs[l]["cQ"], max_calls=1, name="E_L%d" % l) for l in e_layers]
    d = dirs[c_layer]
    w, g = (d["wCp"], d["gCp"]) if c_kind == "Cp" else (d["wCs"], d["gCs"])
    if w is None:
        raise SystemExit("self-report C direction unavailable at layer %d" % c_layer)
    C = [SteerSpec(w, c_layer, mode="add", alpha=-c_alpha, coef=[g], max_calls=1, name="%s_L%d_abst" % (c_kind, c_layer))]
    return {"baseline": [], "E": E, "C_abst": C}


# --------------------------------------------------------------------------- #
def phase_generate(args):
    exp = load_experiment(args.exp)
    mcfg = load_model_config(args.model)
    e22.SPLIT_A = dict(e22.SPLIT_WORDINGS[args.wording])
    set_seed(exp.get("seed", 42)); bs = exp.get("batch_size", 8)
    qa = exp["prompts"]["qa"]; qa_system = qa[args.system_key]
    das_dir = args.das_dir or os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    qs, n_tune, n_recs = test_questions(exp, args.tune_pairs)
    if exp["eval_datasets"] == ["truthfulqa_tf"] and args.tune_pairs == 192 and len(qs) != 529:
        raise SystemExit("expected the 529 E23 test pairs, got %d" % len(qs))
    if args.limit:
        qs = qs[: args.limit]
    est, y_est, est_questions = estimation_set(exp)
    print("test questions: %d (tune pairs %d, statements %d); estimation statements %d, questions %d" % (len(qs), n_tune, n_recs, len(est), len(est_questions)), flush=True)
    bundle = load_model(mcfg)
    layers = sorted(set(args.e_layers) | {args.c_layer})
    dirs = build_directions(bundle, exp, est, y_est, est_questions, layers, das_dir, args.das_variant, qa_system, bs)
    specs = make_specs(dirs, args.e_layers, args.e_alpha, args.c_kind, args.c_layer, args.c_alpha)
    header = {"model": mcfg["name"], "exp": args.exp, "tag": args.tag, "phase": "generate", "n_questions": len(qs), "tune_pairs": args.tune_pairs,
              "prompt": {"system_key": args.system_key, "system": qa_system, "user": qa["user"], "max_new_tokens": args.max_new_tokens, "greedy": True},
              "E": {"estimator": "das", "variant": args.das_variant, "layers": args.e_layers, "alpha": args.e_alpha, "mode": "amplify",
                    "centre": "mean of the 96 estimation questions' last-prompt-token activations under the QA prompt"},
              "C": {"kind": args.c_kind, "layer": args.c_layer, "alpha": -args.c_alpha, "mode": "add", "coef": "gap", "orthogonal_to": "DAS-tf E of the same layer"},
              "intervention_position": "last prompt token of the prefill only (SteerSpec max_calls=1); decode steps untouched",
              "direction_facts": {str(l): dirs[l]["facts"] for l in layers}, "conditions": list(CONDS), "audit": []}
    _, path, _ = out_paths(mcfg["name"], args.tag)
    if os.path.exists(path):                                   # resume only onto a file produced by the SAME protocol
        old = load_json(path)
        for k in ("E", "C", "prompt", "tune_pairs", "n_questions", "intervention_position"):
            if old.get(k) != header[k]:
                raise SystemExit("resume refused: field %r of the existing %s differs from this run's configuration" % (k, path))
    rr = ResumableResults(path, header=header, flush_every=1)
    eos_ids = getattr(bundle.model.generation_config, "eos_token_id", None)
    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids if eos_ids is not None else bundle.tokenizer.eos_token_id])
    eos_ids |= {bundle.tokenizer.pad_token_id}
    t0 = time.time()
    for i in range(0, len(qs), bs):
        idxs = list(range(i, min(i + bs, len(qs))))
        if all(rr.done(j) for j in idxs):
            continue
        texts = [bundle.build_chat(qa_system, qa["user"].format(question=qs[j]["question"])) for j in idxs]
        enc = bundle.encode(texts)
        if not bool(enc["attention_mask"][:, -1].all()):
            raise RuntimeError("left padding expected: the last column must be a real prompt token for every row")
        prompt_len = int(enc["input_ids"].shape[1])
        outs, finished, audit = {}, {}, {}
        for cond in CONDS:
            sp = specs[cond]
            before = {s.name: s.n_applied for s in sp}
            with steer(bundle, sp) as hooks:
                texts_out, out_ids = greedy_generate(bundle, enc, args.max_new_tokens)
            n_new = int(out_ids.shape[1] - prompt_len)                 # generate runs one forward per new token (prefill included)
            applied = {s.name: s.n_applied - before[s.name] for s in sp}
            if any(v != 1 for v in applied.values()):
                raise RuntimeError("prefill-only audit failed for %s: applications per batch %s" % (cond, applied))
            for l, h in hooks.items():                                 # call 1 must have been the full prompt, then one call per new token
                if h.first_shape is None or h.first_shape[1] != prompt_len or h.call_count != n_new:
                    raise RuntimeError("layer %d: first forward T=%s (prompt %d), %d forward calls vs %d new tokens" % (
                        l, None if h.first_shape is None else h.first_shape[1], prompt_len, h.call_count, n_new))
            outs[cond] = texts_out
            finished[cond] = [bool(any(int(t) in eos_ids for t in o[prompt_len:].tolist())) for o in out_ids]   # False = cut by max_new_tokens
            audit[cond] = {"forward_calls_per_layer": {str(l): h.call_count for l, h in hooks.items()},
                           "first_forward_T": {str(l): h.first_shape[1] for l, h in hooks.items()}, "n_new_tokens": n_new, "edits_per_spec": applied}
        for k, j in enumerate(idxs):
            if rr.done(j):
                continue
            rr.add(j, {"pair_id": qs[j]["pair_id"], "question": qs[j]["question"], "gold": qs[j]["gold"], "incorrect": qs[j]["incorrect"],
                       "best_answer": qs[j]["best_answer"], "prompt_len": prompt_len, "outputs": {c: outs[c][k] for c in CONDS},
                       "finished": {c: finished[c][k] for c in CONDS},
                       "rule_abstain": {c: bool(rule_abstains(outs[c][k])) for c in CONDS}})
        rr.data["audit"].append({"batch_start": i, "prompt_len": prompt_len, **audit})   # ResumableResults keeps the header in .data
        if args.limit or (i // bs) % 10 == 0:
            print("  %d/%d (%.0fs) audit E: %s" % (min(i + bs, len(qs)), len(qs), time.time() - t0, audit["E"]), flush=True)
        if args.limit:
            for k, j in enumerate(idxs):
                print("\n#### %s  %s" % (qs[j]["pair_id"], qs[j]["question"]))
                print("prompt (%d tokens in batch): %r" % (prompt_len, texts[k]))
                print("gold: %s" % " | ".join(qs[j]["gold"][:4]))
                for c in CONDS:
                    print("  [%s] %r  (finished: %s, rule abstain: %s)" % (c, outs[c][k], finished[c][k], rule_abstains(outs[c][k])))
    rr.finalize({"n": len(rr.evaluations), "elapsed_s": time.time() - t0,
                 "rule_abstain_rate": {c: float(np.mean([e["rule_abstain"][c] for e in rr.evaluations])) for c in CONDS},
                 "truncated_rate": {c: float(np.mean([not e["finished"][c] for e in rr.evaluations])) for c in CONDS}})
    print("saved", path, flush=True)


# --------------------------------------------------------------------------- #
def phase_judge(args):
    mcfg = load_model_config(args.model)
    d, path, stem = out_paths(mcfg["name"], args.tag)
    if args.input_file:
        path = args.input_file
    data = load_json(path)
    with open(path, "rb") as source_file:
        source_generation_sha256 = hashlib.sha256(source_file.read()).hexdigest()
    conditions = tuple(data.get("conditions", CONDS))
    if not conditions or conditions[0] != "baseline" or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must start with baseline and have unique names")
    if any(set(conditions) - set(e.get("outputs", {})) for e in data["evaluations"]):
        raise ValueError("cannot grade incomplete generated conditions")
    if data.get("model") != mcfg["name"]:
        raise ValueError("input file and requested model differ")
    # Preserve original generated/cached and legacy-judged files for comparison.
    gold_policy = "cached_original_cohort" if args.phase == "rescore" else "exact_personal_refusals_v1"
    stem = args.output_prefix or (stem + "_" + OUTCOME_POLICY + "_" + gold_policy)
    scored_path = stem + "_items.json"
    if any(os.path.abspath(p) == os.path.abspath(path) for p in (scored_path, stem + ".json", stem + ".md")):
        raise ValueError("scored outputs must not overwrite the input")
    if args.phase == "rescore" and any(os.path.exists(p) for p in (scored_path, stem + ".json", stem + ".md")):
        raise FileExistsError("rescored output exists; choose a new --output-prefix")
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    if args.phase == "rescore":
        judge = None
        judge_name = (data.get("judge_summary") or {}).get("judge", "cached")
    else:
        from core.judge import judge_from_config
        exp_j = load_experiment(args.judge_exp)
        judge = judge_from_config(exp_j["judge"], cache_dir=os.path.join(RESULTS_DIR, "judge_cache"))
        judge_name = judge.name
    items = sorted({e["idx"]: e for e in data["evaluations"]}.values(), key=lambda e: e["idx"])   # dedupe a resumed partial batch
    data["evaluations"] = items
    for e in items:
        e.setdefault("judge", {}); e.setdefault("outcome", {}); e.setdefault("outcome_judge_only", {})
        e.setdefault("outcome_legacy", dict(e["outcome"]))
        e.setdefault("rule_judge_disagreement", {})
        # TruthfulQA lists "I have no comment" (and a few "Unknown"-type phrases) among the CORRECT answers of many questions.
        # Handing those to the grader as gold targets would score an abstention as "correct" exactly in the condition that pushes
        # towards abstention. Grade against the substantive gold answers only; questions whose gold is abstention-only are flagged
        # and reported separately (their truthful answer IS to abstain).
        legacy_gold_sub = [g for g in e["gold"] if not rule_abstains(g)]
        current_gold, only_personal_refusal = reference_answers(e["gold"])
        previous_gold = e.get("judge_reference_answers", legacy_gold_sub or e["gold"])
        gold_changed = previous_gold != current_gold
        if args.phase == "rescore":
            if "gold_is_abstention" not in e:
                raise ValueError("cached rescore requires the original cohort flags")
        else:
            e["gold_is_abstention"] = only_personal_refusal
        for c in conditions:
            if args.phase == "rescore":
                if c not in e["judge"]:
                    raise ValueError("cached judge missing for %s/%s" % (e["pair_id"], c))
            elif (c not in e["judge"] or e["judge"][c].get("label") == "unresolved" or args.rejudge or gold_changed
                  or e["judge"][c].get("rubric_id") != getattr(judge, "qa_rubric_id", None)):
                g = judge.qa_grade(e["question"], current_gold, e["outputs"][c], incorrect=e.get("incorrect") or None)
                e["judge"][c] = g
            e["outcome"][c] = judged_outcome(e["judge"][c]["label"])
            e["rule_judge_disagreement"][c] = rule_conflicts_with_judge(e["judge"][c]["label"], e["rule_abstain"][c])
            e["outcome_judge_only"][c] = e["judge"][c]["label"]
        if args.phase != "rescore":
            e["judge_reference_answers"] = current_gold
    rubric_ids = {e["judge"][c].get("rubric_id", "legacy_unversioned") for e in items for c in conditions}
    data["judge_rubric_id"] = next(iter(rubric_ids)) if len(rubric_ids) == 1 else "mixed_rubrics"
    data["outcome_policy"] = OUTCOME_POLICY
    data["gold_policy"] = gold_policy
    if judge is None:
        print("reusing cached semantic grades; zero API and generation calls", flush=True)
    else:
        print("judge calls %d, cache hits %d" % (judge.n_calls, judge.n_cache_hits), flush=True)
    # ---- rates, transitions, bootstrap ---------------------------------- #
    substantive = np.array([not e["gold_is_abstention"] for e in items])
    labs = {c: np.array([e["outcome"][c] for e in items]) for c in conditions}
    labs_j = {c: np.array([e["outcome_judge_only"][c] for e in items]) for c in conditions}
    unresolved = {c: int((labs[c] == "unresolved").sum()) for c in conditions}
    ok = np.all([labs[c] != "unresolved" for c in conditions], axis=0) & substantive
    n = int(ok.sum())
    if n == 0:
        raise ValueError("no jointly resolved substantive questions to score")
    abst_only = {c: {o: int(((labs[c] == o) & ~substantive).sum()) for o in ("correct", "wrong", "abstain", "unresolved")} for c in conditions}
    trunc = {c: float(np.mean([not e.get("finished", {}).get(c, True) for e in items])) for c in conditions}
    rng = np.random.default_rng(0)
    boot = [rng.choice(np.where(ok)[0], n, replace=True) for _ in range(args.n_boot)]
    K = ("correct", "wrong", "abstain", "answer_rate", "precision_answered")

    def rates(idx, c):
        return qa_rates(labs[c][idx])

    base_full = rates(np.where(ok)[0], "baseline")
    rows, trans = {}, {}
    for c in conditions:
        full = rates(np.where(ok)[0], c)
        rows[c] = {"rates": full, "rates_judge_only": qa_rates(labs_j[c][ok]), "n_unresolved": unresolved[c],
                   "abstention_only_gold_questions": abst_only[c], "truncated_rate": trunc[c]}
        if c != "baseline":
            arr = {k: np.array([rates(ii, c)[k] - rates(ii, "baseline")[k] for ii in boot]) for k in K}
            rows[c]["delta_vs_baseline"] = {k: {"value": full[k] - base_full[k], "ci95": [float(np.nanpercentile(arr[k], 2.5)), float(np.nanpercentile(arr[k], 97.5))]} for k in K}
            t = {}
            for a in ("correct", "wrong", "abstain"):
                for b in ("correct", "wrong", "abstain"):
                    t["%s->%s" % (a, b)] = int(((labs["baseline"] == a) & (labs[c] == b) & ok).sum())
            trans[c] = t
    summary = {"model": mcfg["name"], "tag": args.tag, "judge": judge_name, "outcome_policy": OUTCOME_POLICY, "gold_policy": gold_policy, "judge_rubric_id": data["judge_rubric_id"],
               "source_generation_sha256": source_generation_sha256,
               "n_rule_judge_disagreements": {c: sum(e["rule_judge_disagreement"][c] for e in items) for c in conditions},
               "n_questions": len(items), "n_substantive_gold": int(substantive.sum()),
               "n_resolved_all_conditions": n, "rates": rows, "transitions_vs_baseline": trans,
               "header": {k: data[k] for k in data if k not in ("evaluations", "judge_summary")}}
    data["judge_summary"] = summary
    atomic_json_dump(data, scored_path)
    atomic_json_dump(summary, stem + ".json")
    # ---- markdown --------------------------------------------------------- #
    L = ["# 真实问答验证：%s（%s）" % (mcfg["name"], args.tag), "",
         "%d 道问题，分区 `%s`，共 %d 个条件；生成与干预设置以输入文件记录为准。裁判 %s（A 对 / B 错 / C 未作答），"
         "按语义裁判判定，关键词仅标记复核，不能覆盖正确或错误为弃答。参考答案分区口径：%s；其中单列的问题 %d 道。"
         "主表统计所有条件都有判定且金标实质的问题 %d 道；未判定：%s；被 max_new_tokens 截断的比例：%s。" % (
             len(items), data.get("split", "E24 test"), len(conditions), judge_name, gold_policy,
             len(items) - int(substantive.sum()), n, unresolved, {c: round(v, 3) for c, v in trunc.items()}), "",
         "E：%s；C：%s。" % (data["E"], data["C"]), "",
         "| 条件 | 答对 | 答错 | 弃答 | 作答率 | 已作答中答对比例 | 只看裁判（不叠规则）：答对 / 弃答 |", "|---|---|---|---|---|---|---|"]
    if data.get("diagnostic_only"):
        L[2:2] = ["**仅诊断：含金标来源的条件不是可部署性能，不参与选层或主表胜负。**", ""]
    if data.get("pilot") or data.get("split") == "pilot":
        L[2:2] = ["**小样本线路检查，不作为效果或最优层证据。**", ""]
    for c in conditions:
        r = rows[c]["rates"]; rj = rows[c]["rates_judge_only"]
        L.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f / %.3f |" % (c, r["correct"], r["wrong"], r["abstain"], r["answer_rate"], r["precision_answered"], rj["correct"], rj["abstain"]))
    L += ["", "按当前参考答案分区规则单列的 %d 道问题：各条件的结果计数 %s" % (len(items) - int(substantive.sum()), abst_only)]
    L += ["", "## 相对基线的变化（问题自举 %d 次，95%% 区间）" % args.n_boot, "", "| 条件 | Δ答对 | Δ答错 | Δ弃答 | Δ作答率 | Δ已作答中答对比例 |", "|---|---|---|---|---|---|"]
    for c in conditions[1:]:
        dd = rows[c]["delta_vs_baseline"]
        L.append("| %s | %s |" % (c, " | ".join("%+.3f [%+.3f, %+.3f]" % (dd[k]["value"], dd[k]["ci95"][0], dd[k]["ci95"][1]) for k in K)))
    L += ["", "## 三类状态转换（基线 → 条件；题数）", "", "| 条件 | 对→对 | 对→错 | 对→弃答 | 错→对 | 错→错 | 错→弃答 | 弃答→对 | 弃答→错 | 弃答→弃答 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conditions[1:]:
        t = trans[c]
        L.append("| %s | %s |" % (c, " | ".join(str(t[k]) for k in ("correct->correct", "correct->wrong", "correct->abstain", "wrong->correct", "wrong->wrong", "wrong->abstain", "abstain->correct", "abstain->wrong", "abstain->abstain"))))
    L += ["", "## 干预审计", "", "`%s`；各条件的实际写入次数及输入长度见 JSON `audit`。" % data["intervention_position"], ""]
    L += ["## 前 3 题原文", ""]
    for e in items[:3]:
        L.append("- **%s** %s  \n  金标：%s" % (e["pair_id"], e["question"], " | ".join(e["gold"][:3])))
        for c in conditions:
            L.append("  - %s：`%s` → %s" % (c, e["outputs"][c].strip().replace("`", "'")[:300], e["outcome"][c]))
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:14]), flush=True)
    print("saved", stem + ".md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa", help="experiment config for prompts / datasets / seed (same as the E23 sweeps)")
    ap.add_argument("--judge-exp", default="E4_qa", help="experiment config holding the judge section")
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", choices=["generate", "judge", "rescore"], required=True)
    ap.add_argument("--e-layers", nargs="+", type=int)
    ap.add_argument("--e-alpha", type=float)
    ap.add_argument("--c-kind", choices=["Cp", "Cs"])
    ap.add_argument("--c-layer", type=int)
    ap.add_argument("--c-alpha", type=float)
    ap.add_argument("--das-dir", default=None); ap.add_argument("--das-variant", default="tf")
    ap.add_argument("--wording", default="pik")
    ap.add_argument("--tune-pairs", type=int, default=192)
    ap.add_argument("--system-key", default="system_idk_ok", help="prompts.qa key for the shared system prompt (must allow 'I don't know')")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="pilot: only the first N test questions, printed in full")
    ap.add_argument("--tag", default="fixed")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rejudge", action="store_true")
    ap.add_argument("--input-file", default=None, help="judge/rescore an explicit generated JSON")
    ap.add_argument("--output-prefix", default=None, help="new prefix for versioned scored outputs")
    args = ap.parse_args()
    if args.tune_pairs <= 0:
        raise SystemExit("--tune-pairs must be the E23 value (192)")
    if args.phase == "generate" and any(getattr(args, k) is None for k in ("e_layers", "e_alpha", "c_kind", "c_layer", "c_alpha")):
        ap.error("generation requires --e-layers, --e-alpha, --c-kind, --c-layer and --c-alpha")
    if args.phase == "rescore" and args.rejudge:
        ap.error("rescore reuses cached grades; --rejudge would change the experiment")
    (phase_generate if args.phase == "generate" else phase_judge)(args)


if __name__ == "__main__":
    main()
