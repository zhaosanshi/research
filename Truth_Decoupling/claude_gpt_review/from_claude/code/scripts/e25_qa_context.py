#!/usr/bin/env python3
"""E25: matched question-answer-context direction controls, pending review.

Explicit phases: estimate -> tune-e/tune-c -> judge-tune-c -> freeze ->
generate -> judge-test. No phase launches another inference/API phase.
Both E contexts use mean differences on the same cleaned estimation pairs.
Candidate scoring is paired answer accuracy, not official TruthfulQA MC1.
freeze --target E reads only E tuning; the default both also validates C.
"""
import argparse
import hashlib
import json
import re
import string
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import load_split, tf_prompt
from core.config import load_experiment, load_json_config, load_model_config, resolve_config_path
from core.directions import mean_diff_direction, unit
from core.intervene import SteerSpec, steer
from core.io_utils import atomic_json_dump, stable_hash, set_seed
from core.model_io import capture_hooks, capture_last_token, forward_logits, load_model
from core.qa_context import prepare_e_pairs, encode_answer, pack_answers, answer_logprobs
from core.readout import rule_abstains

PROTOCOL = "e25_qa_context_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_identity(meta):
    return stable_hash({k: meta[k] for k in ("protocol", "model", "model_config", "config", "layers", "pilot", "rows",
                                            "bank_sha256", "questions_sha256")}, length=40)


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("output exists: %s; use a separate run directory" % path)
    atomic_json_dump(value, str(path))


def validate_config(cfg):
    if type(cfg.get("seed")) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("qa_system", "source_experiment"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError("%s must be nonempty text" % key)
    for key in ("qa_user", "self_report_user"):
        template = cfg.get(key)
        if not isinstance(template, str) or not template.strip():
            raise ValueError("%s must be a nonempty question template" % key)
        try:
            fields = [(name, spec, conv) for _, name, spec, conv in string.Formatter().parse(template)
                      if name is not None]
        except ValueError as exc:
            raise ValueError("invalid %s question template" % key) from exc
        if not fields or any(name != "question" or spec or conv for name, spec, conv in fields):
            raise ValueError("%s must contain only plain {question} placeholders" % key)
    for key in ("tune_pairs", "batch_size", "max_new_tokens"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError("%s must be a positive integer" % key)
    for key, allowed in (("e_contexts", {"tf", "qa"}), ("c_contexts", {"self", "qa"}),
                         ("e_write_modes", {"prefill", "continuous"})):
        if (not isinstance(cfg.get(key), (list, tuple)) or not cfg[key]
                or any(not isinstance(x, str) for x in cfg[key])
                or len(set(cfg[key])) != len(cfg[key]) or not set(cfg[key]) <= allowed):
            raise ValueError("invalid/duplicate %s" % key)
    for key in ("e_alphas", "c_alphas"):
        if (not isinstance(cfg.get(key), (list, tuple)) or not cfg[key]
                or any(type(a) not in (int, float) or not np.isfinite(a) or a < 0 for a in cfg[key])
                or len(set(cfg[key])) != len(cfg[key])
                or len({"%g" % a for a in cfg[key]}) != len(cfg[key])):
            raise ValueError("invalid %s" % key)
    tolerance = cfg.get("c_pair_accuracy_tolerance")
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("invalid C paired-accuracy tolerance")
    if type(cfg.get("orthogonalize_c")) is not bool:
        raise ValueError("orthogonalize_c must be boolean")
    if not isinstance(cfg.get("c_contrasts"), (list, tuple)) or not cfg["c_contrasts"]:
        raise ValueError("C requires paired prompt contrasts")
    for contrast in cfg["c_contrasts"]:
        if not isinstance(contrast, dict):
            raise ValueError("C contrasts must be mappings")
        for side in ("confident", "cautious"):
            if not isinstance(contrast.get(side), str) or not contrast[side].strip():
                raise ValueError("empty C contrast")
            if re.search(r"\bunknown\b|i don['’]?t know|i do not know", contrast[side], re.I):
                raise ValueError("C contrasts must not prescribe a fixed abstention phrase")
    return cfg


def tf_statements(pairs):
    """Match truthfulqa_tf's punctuation using only the already cleaned E answers.

    Deliberately do not read best_answer/gold here: those original lists may
    begin with a personal refusal that prepare_e_pairs has replaced/excluded.
    """
    return ['The answer to the question "%s" is: %s' % (
        r["question"].strip().rstrip("?").strip() + "?", r[key].strip().rstrip(".") + ".")
        for key in ("positive_answer", "negative_answer") for r in pairs]


def split_questions(cfg, pilot=None):
    exp = load_experiment(cfg["source_experiment"])
    train_tf, eval_tf = [], []
    for dataset in exp["eval_datasets"]:
        train_tf += load_split(dataset, "calib") + load_split(dataset, "dev")
        eval_tf += load_split(dataset, exp.get("eval_split", "test"))
    def ids(records):
        return list(dict.fromkeys(str(r.get("pair_id") or r["id"]) for r in records))
    train_ids, eval_ids = ids(train_tf), ids(eval_tf)
    if set(train_ids) & set(eval_ids):
        raise ValueError("estimation and evaluation questions overlap")
    n_tune = int(cfg["tune_pairs"])
    if not 0 < n_tune < len(eval_ids):
        raise ValueError("tune_pairs must leave both tuning and test questions")
    tune_set = set(np.random.default_rng(cfg["seed"]).choice(sorted(eval_ids), n_tune, replace=False).tolist())
    qa = load_split("truthfulqa", "all")
    by_id = {str(r["id"]): r for r in qa}
    if len(by_id) != len(qa):
        raise ValueError("duplicate original QA IDs")
    def records(wanted):
        result = []
        for pid in wanted:
            r = by_id[pid]
            result.append({"pair_id": pid, "question": r["text"], "gold": list(r.get("gold") or []),
                           "incorrect": list(r.get("incorrect") or []),
                           "best_answer": (r.get("meta") or {}).get("best_answer")})
        return result[:pilot] if pilot else result
    splits = {"train": records(train_ids), "tune": records([p for p in eval_ids if p in tune_set]),
              "test": records([p for p in eval_ids if p not in tune_set])}
    train_pairs, train_audit = prepare_e_pairs(splits["train"])
    tune_pairs, tune_audit = prepare_e_pairs(splits["tune"])
    if not train_pairs or not tune_pairs:
        raise ValueError("no usable E estimation/tuning pairs after refusal filtering")
    return {"splits": splits, "e_train": train_pairs, "e_tune": tune_pairs,
            "pair_audit": {"train": train_audit, "tune": tune_audit}, "pilot": pilot,
            "full_split_counts": {"train": len(train_ids), "tune": n_tune, "test": len(eval_ids) - n_tune}}


def qa_text(bundle, cfg, question, system=None, context="qa"):
    template = cfg["qa_user"] if context == "qa" else cfg["self_report_user"]
    return bundle.build_chat(system or cfg["qa_system"], template.format(question=question))


def encode_pairs(bundle, cfg, pairs):
    # Positive then negative halves, so every comparison uses the same question.
    return [encode_answer(bundle, cfg["qa_system"], r["question"], r[key], cfg["qa_user"])
            for key in ("positive_answer", "negative_answer") for r in pairs]


def capture_answer_states(bundle, encoded, layers, batch_size):
    last, means = {l: [] for l in layers}, {l: [] for l in layers}
    for i in range(0, len(encoded), batch_size):
        packed = pack_answers(encoded[i:i + batch_size], bundle.tokenizer.pad_token_id, bundle.device)
        with capture_hooks(bundle, layers, ("resid",), position="all") as stored:
            forward_logits(bundle, packed["inputs"])
        for layer in layers:
            h = stored[("resid", layer)].float()
            mask = packed["answer_mask"].to(h.device)
            if not torch.isfinite(h[mask]).all():
                raise ValueError("non-finite answer activations")
            # pack_answers left-pads and appends neither EOS nor template terminators.
            last[layer].append(h[:, -1, :].cpu())
            # A masked padding activation may be NaN; multiplication by zero
            # would retain NaN. Only answer content belongs in this mean.
            means[layer].append((h.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[:, None]).cpu())
    return ({l: torch.cat(v).numpy() for l, v in last.items()},
            {l: torch.cat(v).numpy() for l, v in means.items()})


def estimate(args, cfg):
    out = args.run_dir
    if any((out / name).exists() for name in ("bank.npz", "bank.json", "questions.json")):
        raise FileExistsError("estimate needs a new run directory")
    prepared = split_questions(cfg, args.pilot)
    set_seed(cfg["seed"])
    bundle = load_model(load_model_config(args.model))
    layers = resolve_layers(args, range(bundle.num_layers), allow_default=True)
    pairs, questions = prepared["e_train"], prepared["splits"]["train"]
    bs = cfg["batch_size"]
    encoded = encode_pairs(bundle, cfg, pairs)
    qa_last, qa_means = capture_answer_states(bundle, encoded, layers, bs)
    q_acts, _ = capture_last_token(bundle, [qa_text(bundle, cfg, q["question"]) for q in questions], layers, batch_size=bs)
    exp = load_experiment(cfg["source_experiment"])
    statements = tf_statements(pairs)
    tf_acts, _ = capture_last_token(bundle, [tf_prompt(bundle, exp["prompts"], text, allow_idk=False)
                                          for text in statements], layers, batch_size=bs)
    arrays, rows = {}, {}
    n = len(pairs)
    for layer in layers:
        htf = tf_acts[("resid", layer)].numpy()
        wqa, meta_qa = mean_diff_direction(qa_last[layer][:n], qa_last[layer][n:])
        wtf, meta_tf = mean_diff_direction(htf[:n], htf[n:])
        arrays["E_qa_L%d" % layer], arrays["E_tf_L%d" % layer] = wqa, wtf
        # Same QA centres for both direction contexts. Each answer has equal weight.
        arrays["prompt_center_L%d" % layer] = q_acts[("resid", layer)].numpy().mean(0)
        arrays["decode_center_L%d" % layer] = .5 * (qa_means[layer][:n].mean(0) + qa_means[layer][n:].mean(0))
        rows[str(layer)] = {"E_qa_gap": float(meta_qa["gap"]), "E_tf_gap": float(meta_tf["gap"])}
    for context in cfg["c_contexts"]:
        differences = {l: [] for l in layers}
        for contrast in cfg["c_contrasts"]:
            acts = {}
            for side in ("confident", "cautious"):
                system = cfg["qa_system"] + "\n\n" + contrast[side]
                texts = [qa_text(bundle, cfg, r["question"], system, context) for r in questions]
                acts[side], _ = capture_last_token(bundle, texts, layers, batch_size=bs)
            for layer in layers:
                differences[layer].append(acts["confident"][("resid", layer)].numpy()
                                          - acts["cautious"][("resid", layer)].numpy())
        for layer in layers:
            raw = np.mean(differences[layer], axis=(0, 1)).astype(np.float64)
            gap, w = float(np.linalg.norm(raw)), unit(raw)
            e = arrays["E_qa_L%d" % layer]
            cosine = float(w @ e)
            projected = raw - (raw @ e) * e
            if cfg["orthogonalize_c"]:
                w = unit(projected)
            arrays["C_%s_L%d" % (context, layer)] = w
            arrays["C_%s_raw_L%d" % (context, layer)] = unit(raw)
            rows[str(layer)]["C_" + context] = {"gap": gap, "projected_gap": float(np.linalg.norm(projected)),
                                                  "cos_before": cosine, "cos_after": float(w @ e),
                                                  "scale_basis": "raw mean contrast gap; normalized projected direction"}
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "questions.json", prepared)
    np.savez(out / "bank.npz", **arrays)
    meta = {"protocol": PROTOCOL, "model": args.model, "model_config": bundle.cfg, "config": cfg,
            "layers": layers, "num_layers": bundle.num_layers, "pilot": args.pilot,
            "estimator": "mean_difference_for_both_contexts", "rows": rows,
            "E_direction_contexts": {"qa": "mean difference re-estimated from cleaned QA answers",
                                     "tf": "TF mean difference re-estimated from the same cleaned answer pairs; not a reused historical TF/DAS direction"},
            "TF_statement_format": 'The answer to the question "{question_with_terminal_question_mark}" is: {cleaned_answer_with_terminal_period}',
            "questions_sha256": digest(out / "questions.json"), "bank_sha256": digest(out / "bank.npz"),
            "n_E_pairs": n, "n_C_questions": len(questions),
            "E_read_position": "last answer content token, excluding EOS/template suffix",
            "E_read_positions": {"qa": "last answer content token, excluding EOS/template suffix",
                                 "tf": "last prompt token of the re-estimated TF statement prompt"},
            "decode_center_basis": "equal-per-answer mean of answer-content activations, class balanced",
            "prompt_center_basis": "estimation questions under shared QA prompt",
            "token_audit": [x for x in encoded[:3]]}
    meta["protocol_fingerprint"] = protocol_identity(meta)
    write_new(out / "bank.json", meta)
    print("E pairs %d, C questions %d; layers %s" % (n, len(questions), layers))
    for item in encoded[:3]:
        print("TOKEN AUDIT:", json.dumps({"question": item["text_audit"]["question"], "answer": item["answer_text"],
              "prompt_length": item["prompt_length"], "answer_length": item["answer_length"],
              "first_target_position": item["prompt_length"], "first_predictor_position": item["prompt_length"] - 1}, ensure_ascii=False))


def resolve_layers(args, available, allow_default=False):
    available = list(available)
    if args.all_layers or (allow_default and args.layers is None):
        return available
    if not args.layers:
        raise ValueError("choose --layers or --all-layers explicitly before a sweep")
    if len(set(args.layers)) != len(args.layers) or not set(args.layers).issubset(available):
        raise ValueError("layers must be unique and present in the bank")
    return list(args.layers)


def load_bank(args):
    meta = read(args.run_dir / "bank.json")
    if meta["model"] != args.model or meta["protocol"] != PROTOCOL:
        raise ValueError("wrong model or protocol in direction bank")
    if meta.get("protocol_fingerprint") != protocol_identity(meta):
        raise ValueError("bank protocol/configuration changed; create a new estimate run")
    validate_config(meta["config"])
    if digest(args.run_dir / "bank.npz") != meta["bank_sha256"] or digest(args.run_dir / "questions.json") != meta["questions_sha256"]:
        raise ValueError("bank/data changed since estimation")
    with np.load(args.run_dir / "bank.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return meta, arrays, read(args.run_dir / "questions.json")


def conditions(meta, layers, target):
    cfg = meta["config"]
    out = [{"name": "baseline", "kind": "baseline"}]
    contexts = cfg["e_contexts"] if target == "E" else cfg["c_contexts"]
    modes = cfg["e_write_modes"] if target == "E" else ["prefill"]
    alphas = cfg["e_alphas"] if target == "E" else cfg["c_alphas"]
    for context in contexts:
        for mode in modes:
            for layer in layers:
                for alpha in alphas:
                    out.append({"name": "%s_%s_%s_L%d_a%g" % (target, context, mode, layer, alpha),
                                "kind": target, "context": context, "mode": mode, "layer": layer, "alpha": float(alpha)})
    return out


def specs_for(meta, bank, condition, packed=None):
    if condition["kind"] == "baseline":
        return []
    layer, mode, a = condition["layer"], condition["mode"], condition["alpha"]
    if mode not in ("prefill", "continuous"):
        raise ValueError("unsupported write mode")
    if condition["kind"] == "E":
        w = bank["E_%s_L%d" % (condition["context"], layer)]
        cp, cd = bank["prompt_center_L%d" % layer], bank["decode_center_L%d" % layer]
        if packed is None:
            return [SteerSpec(w, layer, mode="amplify", alpha=a, center=cp, decode_center=cd,
                              max_calls=1 if mode == "prefill" else None, name=condition["name"])]
        out = [SteerSpec(w, layer, mode="amplify", alpha=a, center=cp, position="mask",
                         position_mask=packed["prompt_end_mask"], name=condition["name"] + "_prompt")]
        if mode == "continuous":
            out.append(SteerSpec(w, layer, mode="amplify", alpha=a, center=cd, position="mask",
                                 position_mask=packed["predictor_mask"] & ~packed["prompt_end_mask"],
                                 name=condition["name"] + "_decode"))
        return out
    if condition["kind"] != "C":
        raise ValueError("unsupported intervention kind")
    w = bank["C_%s_L%d" % (condition["context"], layer)]
    gap = meta["rows"][str(layer)]["C_" + condition["context"]]["gap"]
    kw = {"position": "last", "max_calls": 1 if mode == "prefill" else None} if packed is None else {
        "position": "mask", "position_mask": packed["prompt_end_mask"] if mode == "prefill" else packed["predictor_mask"]}
    return [SteerSpec(w, layer, mode="add", alpha=-a, coef=[gap], name=condition["name"], **kw)]


def paired_stats(base_correct, correct):
    gain = int((~base_correct & correct).sum()); harm = int((base_correct & ~correct).sum())
    return {"n": len(correct), "accuracy": float(correct.mean()), "baseline_correct": int(base_correct.sum()),
            "corrected": gain, "harmed": harm, "net": gain - harm,
            "harm_rate": harm / int(base_correct.sum()) if base_correct.any() else None}


def score_conditions(bundle, meta, bank, pairs, choices, checkpoint):
    encoded = encode_pairs(bundle, meta["config"], pairs)
    identity = stable_hash(meta["protocol_fingerprint"], choices, [r["pair_id"] for r in pairs], length=40)
    result = read(checkpoint) if checkpoint.exists() else {"fingerprint": identity, "metric": "paired_answer_selection_accuracy",
              "model": meta["model"], "bank_sha256": meta["bank_sha256"], "questions_sha256": meta["questions_sha256"],
              "bank_protocol_fingerprint": meta["protocol_fingerprint"],
              "condition_specs": choices,
              "definition": "sum log P(positive answer) > sum log P(one negative answer); strict tie is not correct",
              "question_ids": [r["pair_id"] for r in pairs], "conditions": {}}
    if result["fingerprint"] != identity:
        raise ValueError("score resume refused: protocol or candidates changed")
    bs, n = meta["config"]["batch_size"], len(pairs)
    for condition in choices:
        if condition["name"] in result["conditions"]:
            continue
        scores = []
        for start in range(0, len(encoded), bs):
            packed = pack_answers(encoded[start:start + bs], bundle.tokenizer.pad_token_id, bundle.device)
            with steer(bundle, specs_for(meta, bank, condition, packed)):
                logits = forward_logits(bundle, packed["inputs"])
            values = answer_logprobs(logits, packed).detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("non-finite paired answer score")
            scores.extend(values.tolist())
        positive, negative = np.array(scores[:n]), np.array(scores[n:])
        correct = positive > negative
        baseline = correct if condition["name"] == "baseline" else np.array(result["conditions"]["baseline"]["correct"], dtype=bool)
        result["conditions"][condition["name"]] = {"condition": condition, "positive_logp": positive.tolist(),
             "negative_logp": negative.tolist(), "correct": correct.tolist(), **paired_stats(baseline, correct)}
        atomic_json_dump(result, str(checkpoint))
        print("PAIRED", condition["name"], result["conditions"][condition["name"]]["accuracy"], flush=True)
    result["completed"] = True
    atomic_json_dump(result, str(checkpoint))
    return result


def generate_questions(bundle, meta, bank, questions, choices, path, split, paired=None):
    cfg = meta["config"]
    fingerprint = stable_hash(meta["protocol_fingerprint"], choices, split,
                              [q["pair_id"] for q in questions], length=40)
    if path.exists():
        result = read(path)
        if result.get("protocol_fingerprint") != fingerprint:
            raise ValueError("generation resume refused: configuration changed")
    else:
        result = {"model": meta["model"], "tag": "E25_" + split, "protocol": PROTOCOL,
                  "protocol_fingerprint": fingerprint, "bank_sha256": meta["bank_sha256"],
                  "bank_protocol_fingerprint": meta["protocol_fingerprint"],
                  "questions_sha256": meta["questions_sha256"], "split": split, "pilot": meta["pilot"],
                  "prompt": {"system": cfg["qa_system"], "user": cfg["qa_user"], "max_new_tokens": cfg["max_new_tokens"]},
                  "E": {"estimator": "mean_difference", "contexts": cfg["e_contexts"]},
                  "C": {"estimator": "averaged_prompt_contrast", "contexts": cfg["c_contexts"]},
                  "intervention_position": "prefill last prompt token; continuous mode also edits each subsequent predictor",
                  "conditions": [c["name"] for c in choices], "condition_specs": choices, "paired_scores": paired,
                  "audit": [], "evaluations": [{**q, "idx": i, "outputs": {}, "finished": {}, "rule_abstain": {}}
                                                for i, q in enumerate(questions)]}
    rows = result["evaluations"]
    if len(rows) != len(questions) or [r["pair_id"] for r in rows] != [q["pair_id"] for q in questions]:
        raise ValueError("generation resume question order changed")
    eos = getattr(bundle.model.generation_config, "eos_token_id", None)
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos if eos is not None else bundle.tokenizer.eos_token_id])
    for condition in choices:
        name = condition["name"]
        for start in range(0, len(rows), cfg["batch_size"]):
            chunk = rows[start:start + cfg["batch_size"]]
            if all(name in q["outputs"] for q in chunk):
                continue
            texts = [qa_text(bundle, cfg, q["question"]) for q in chunk]
            inputs = bundle.encode(texts)
            if not inputs["attention_mask"][:, -1].all():
                raise ValueError("generation requires left padding")
            prompt_len = inputs["input_ids"].shape[1]
            specs = specs_for(meta, bank, condition)
            with torch.no_grad(), steer(bundle, specs) as hooks:
                output = bundle.model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                               num_beams=1, use_cache=True, pad_token_id=bundle.tokenizer.pad_token_id)
            n_new = output.shape[1] - prompt_len
            expected = 1 if condition.get("mode") == "prefill" else n_new
            if any(s.n_applied != expected for s in specs):
                raise RuntimeError("unexpected intervention count")
            if any(h.first_shape[1] != prompt_len or h.call_count != n_new for h in hooks.values()):
                raise RuntimeError("unexpected prefill/decode shape or count")
            for row, tokens in zip(chunk, output[:, prompt_len:]):
                text = bundle.tokenizer.decode(tokens, skip_special_tokens=True)
                if name in row["outputs"] and row["outputs"][name] != text:
                    raise RuntimeError("resume regenerated a different existing answer")
                row["outputs"][name] = text
                row["finished"][name] = any(int(t) in eos for t in tokens)
                row["rule_abstain"][name] = bool(rule_abstains(text))
                if meta["pilot"]:
                    print("PILOT", row["pair_id"], name, repr(row["question"]), repr(text), flush=True)
            result["audit"].append({"condition": name, "batch_start": start, "prompt_len": prompt_len,
                                     "new_tokens": n_new, "edits": {s.name: s.n_applied for s in specs}})
            atomic_json_dump(result, str(path))
    result["completed"] = True
    atomic_json_dump(result, str(path))


def select_e(scores, cfg):
    base = scores["conditions"]["baseline"]["accuracy"]
    chosen = {}
    for context in cfg["e_contexts"]:
        for mode in cfg["e_write_modes"]:
            eligible = [r for r in scores["conditions"].values() if r["condition"].get("context") == context
                        and r["condition"].get("mode") == mode and r["accuracy"] > base]
            best = min(eligible, key=lambda r: (-r["accuracy"], r["harmed"], r["condition"]["alpha"], r["condition"]["layer"])) if eligible else None
            chosen["E_%s_%s" % (context, mode)] = best["condition"] if best else None
    return chosen


def _freeze_decision(args, meta, target):
    """Validate the sources actually used and recompute a freeze without writing."""
    if target not in ("E", "both"):
        raise ValueError("freeze target must be E or both")
    e_scores = read(args.run_dir / "tune_e_scores.json")
    if not e_scores.get("completed"):
        raise ValueError("finish tuning before freezing selections")
    scored_targets = [(e_scores, "E")]
    sources = ["tune_e_scores.json"]
    if target == "both":
        c_raw = read(args.run_dir / "tune_c.json")
        c_scores = read(args.run_dir / "tune_c_scored.json")
        if not c_raw.get("completed"):
            raise ValueError("finish tuning before freezing selections")
        scored_targets.append((c_raw["paired_scores"], "C"))
        # C paired scores are embedded in tune_c.json; no unused file is bound.
        sources += ["tune_c.json", "tune_c_scored.json"]
    prepared = read(args.run_dir / "questions.json")
    expected_ids = [r["pair_id"] for r in prepared["e_tune"]]
    for scores, _ in scored_targets:
        if (scores.get("model") != args.model or scores.get("bank_sha256") != meta["bank_sha256"]
                or scores.get("questions_sha256") != meta["questions_sha256"]
                or scores.get("bank_protocol_fingerprint") != meta["protocol_fingerprint"]
                or scores.get("question_ids") != expected_ids or not scores.get("completed")):
            raise ValueError("paired tuning scores belong to different bank/questions")
        expected_fingerprint = stable_hash(meta["protocol_fingerprint"], scores["condition_specs"], expected_ids, length=40)
        if scores.get("fingerprint") != expected_fingerprint:
            raise ValueError("paired tuning score fingerprint mismatch")
    if target == "both":
        if (c_scores.get("source_generation_sha256") != digest(args.run_dir / "tune_c.json")
                or c_scores.get("model") != args.model
                or c_scores.get("header", {}).get("protocol_fingerprint") != c_raw.get("protocol_fingerprint")
                or c_raw.get("split") != "tune" or c_raw.get("model") != args.model
                or c_raw.get("bank_protocol_fingerprint") != meta["protocol_fingerprint"]
                or [r["pair_id"] for r in c_raw["evaluations"]] != [r["pair_id"] for r in prepared["splits"]["tune"]]):
            raise ValueError("semantic C scores are not bound to this tuning generation")
    for scores, score_target in scored_targets:
        specs = scores["condition_specs"]
        requested = sorted({c["layer"] for c in specs if c["kind"] != "baseline"})
        if not requested or not set(requested) <= set(meta["layers"]):
            raise ValueError("invalid candidate layers in tuning scores")
        expected = {c["name"]: c for c in conditions(meta, requested, score_target)}
        if (len(specs) != len(expected) or {c["name"]: c for c in specs} != expected
                or set(scores["conditions"]) != set(expected)
                or any(scores["conditions"][name]["condition"] != spec for name, spec in expected.items())):
            raise ValueError("tuning score candidates differ from configured grid")
    chosen = select_e(e_scores, meta["config"])
    for context in meta["config"]["c_contexts"]:
        chosen["C_" + context] = None
    if target == "both":
        from core.judge import QA_GRADER_VERSION
        if c_raw["condition_specs"] != c_raw["paired_scores"]["condition_specs"]:
            raise ValueError("C scoring and generation use different candidate grids")
        if c_scores.get("judge_rubric_id") != QA_GRADER_VERSION:
            raise ValueError("C selection requires current semantic rubric, not legacy or keyword labels")
        if c_scores["n_resolved_all_conditions"] != c_scores["n_substantive_gold"]:
            raise ValueError("unresolved C grades: do not select from an incomplete comparison")
        if c_raw["bank_sha256"] != meta["bank_sha256"] or c_raw["questions_sha256"] != meta["questions_sha256"]:
            raise ValueError("C tuning inputs differ from the frozen bank")
        pair = c_raw["paired_scores"]["conditions"]
        base_pair = pair["baseline"]["accuracy"]
        rates = c_scores["rates"]
        base_abstain = rates["baseline"]["rates"]["abstain"]
        for context in meta["config"]["c_contexts"]:
            eligible = []
            for condition in c_raw["condition_specs"]:
                name = condition["name"]
                if condition.get("context") != context:
                    continue
                row = rates[name]["rates"]
                if (abs(pair[name]["accuracy"] - base_pair) <= meta["config"]["c_pair_accuracy_tolerance"] + 1e-12
                        and row["abstain"] > base_abstain and row["answer_rate"] > 0):
                    eligible.append(condition)
            chosen["C_" + context] = min(eligible, key=lambda c: (-rates[c["name"]]["rates"]["abstain"], c["alpha"], c["layer"])) if eligible else None
    frozen = {"protocol": PROTOCOL, "model": args.model, "bank_sha256": meta["bank_sha256"],
              "bank_protocol_fingerprint": meta["protocol_fingerprint"],
              "questions_sha256": meta["questions_sha256"], "pilot": meta["pilot"], "selected": chosen,
              "target": target, "sources": {name: digest(args.run_dir / name) for name in sources},
              "tune_e_sha256": digest(args.run_dir / "tune_e_scores.json"),
              "tune_c_scored_sha256": digest(args.run_dir / "tune_c_scored.json") if target == "both" else None,
              "selection_note": ("E: paired accuracy, report harm. " + (
                  "C: semantic abstention with paired-accuracy tolerance; no 2x2 claim." if target == "both" else
                  "E-only freeze: C is unselected and no C artifacts were read."))}
    return frozen


def freeze(args, meta):
    frozen = _freeze_decision(args, meta, getattr(args, "target", "both"))
    write_new(args.run_dir / "selection.json", frozen)
    print(json.dumps(frozen["selected"], ensure_ascii=False, indent=2))
    return frozen


def load_selection(args, meta):
    """Check all used sources and rederive the chosen working point before test."""
    frozen = read(args.run_dir / "selection.json")
    if (frozen.get("model") != args.model or frozen.get("protocol") != PROTOCOL
            or frozen.get("bank_sha256") != meta["bank_sha256"]
            or frozen.get("bank_protocol_fingerprint") != meta["protocol_fingerprint"]
            or frozen.get("questions_sha256") != meta["questions_sha256"]):
        raise ValueError("selection does not match direction bank/data")
    target = frozen.get("target")
    if target not in ("E", "both"):
        raise ValueError("selection lacks its freeze target; create a new freeze from original tuning sources")
    expected_sources = {"tune_e_scores.json"}
    if target == "both":
        expected_sources |= {"tune_c.json", "tune_c_scored.json"}
    if not isinstance(frozen.get("sources"), dict) or set(frozen["sources"]) != expected_sources:
        raise ValueError("frozen selection source set is incomplete or changed")
    for filename, expected in frozen["sources"].items():
        path = args.run_dir / filename
        if not path.is_file() or digest(path) != expected:
            raise ValueError("frozen selection source changed: " + filename)
    if (digest(args.run_dir / "bank.npz") != meta["bank_sha256"]
            or digest(args.run_dir / "questions.json") != meta["questions_sha256"]):
        raise ValueError("bank/data changed since estimation")
    recomputed = _freeze_decision(args, meta, target)
    if frozen != recomputed:
        raise ValueError("frozen selection differs from the validated tuning decision")
    return frozen


def judge_file(args, split):
    import e24_real_qa as e24
    if split == "tune" and (args.run_dir / "selection.json").exists():
        raise ValueError("tuning selection is frozen; use a new run to change tuning grades")
    path = args.run_dir / ("tune_c.json" if split == "tune" else "test.json")
    data = read(path)
    if not data.get("completed") or data["split"] != split:
        raise ValueError("finish generation before judging")
    kwargs = dict(model=args.model, tag="E25_" + split, phase="judge", judge_exp=args.judge_exp,
                  input_file=str(path), output_prefix=str(args.run_dir / ("tune_c_scored" if split == "tune" else "test_scored")),
                  rejudge=False, n_boot=2000)
    e24.phase_judge(SimpleNamespace(**kwargs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["estimate", "tune-e", "tune-c", "judge-tune-c", "freeze", "generate", "judge-test"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", default=None, help="only for estimate; later phases use the frozen bank configuration")
    parser.add_argument("--judge-exp", default="E4_qa")
    parser.add_argument("--target", choices=["E", "both"], default="both",
                        help="freeze only: E does not read C tuning artifacts; default both preserves the joint workflow")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--layers", type=int, nargs="+")
    group.add_argument("--all-layers", action="store_true")
    parser.add_argument("--pilot", type=int, help="estimate a separate small pilot; not performance evidence")
    args = parser.parse_args()
    if args.phase != "freeze" and args.target != "both":
        parser.error("--target E applies only to freeze")
    if args.pilot is not None and (args.phase != "estimate" or args.pilot < 1):
        parser.error("--pilot is a positive count used only when creating a new estimate run")
    if args.phase != "estimate" and args.config is not None:
        parser.error("--config applies only to estimate; use a new run directory to change the protocol")
    if args.phase == "estimate":
        cfg = validate_config(load_json_config(resolve_config_path("experiments", args.config or "E25_qa_context")))
        estimate(args, cfg)
        return
    meta, bank, prepared = load_bank(args)
    if args.phase in ("judge-tune-c", "judge-test"):
        judge_file(args, "tune" if args.phase == "judge-tune-c" else "test")
    elif args.phase == "freeze":
        freeze(args, meta)
    elif args.phase in ("tune-e", "tune-c"):
        layers = resolve_layers(args, meta["layers"])
        choices = conditions(meta, layers, "E" if args.phase == "tune-e" else "C")
        print("Tuning %d conditions on %d eligible pairs" % (len(choices), len(prepared["e_tune"])), flush=True)
        set_seed(meta["config"]["seed"])
        bundle = load_model(meta["model_config"])
        if args.phase == "tune-e":
            score_conditions(bundle, meta, bank, prepared["e_tune"], choices, args.run_dir / "tune_e_scores.json")
        else:
            scores = score_conditions(bundle, meta, bank, prepared["e_tune"], choices, args.run_dir / "tune_c_pair_scores.json")
            generate_questions(bundle, meta, bank, prepared["splits"]["tune"], choices, args.run_dir / "tune_c.json", "tune", scores)
    else:
        frozen = load_selection(args, meta)
        choices = [{"name": "baseline", "kind": "baseline"}] + [c for c in frozen["selected"].values() if c is not None]
        print("Test conditions:", [c["name"] for c in choices], flush=True)
        set_seed(meta["config"]["seed"])
        bundle = load_model(meta["model_config"])
        generate_questions(bundle, meta, bank, prepared["splits"]["test"], choices, args.run_dir / "test.json", "test")


if __name__ == "__main__":
    main()
