#!/usr/bin/env python3
"""QA-DAS v1: explicit estimate/train/check/tune/grade/freeze/test phases.

No phase automatically starts an API phase or a later model experiment. All
selection uses tuning questions. Reference-answer interventions are separately
marked diagnostics and never enter selection or the primary test conditions.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import string
from types import SimpleNamespace

import numpy as np
import torch

import common  # puts the experiment code root on sys.path
import e25_qa_context as legacy
from core.config import load_json_config, load_model_config, resolve_config_path
from core.directions import mean_diff_direction
from core.intervene import SteerSpec, steer
from core.io_utils import atomic_json_dump, set_seed
from core.judge import QA_GRADER_VERSION
from core.model_io import capture_hooks, capture_last_token, forward_logits, load_model
from core.qa_context import encode_answer, pack_answers, answer_logprobs
from core.qa_das import prepare_selfcheck_pairs, train_qa_das, evaluate_selfcheck
from core.qa_precision import fp32_output_projection, projection_diagnostic
from core.qa_selection import rank_proxy_candidates, select_real_candidates
from core.qa_scoring import judged_outcome, reference_answers
from core.readout import rule_abstains

PROTOCOL = "qa_das_ptrue_ce_bidir_v1"
METHODS = ("qa_das", "qa_md")
BASELINE = {"name": "baseline", "kind": "baseline", "eligible_for_selection": False,
            "uses_reference_answer": False}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value, new=False):
    path = Path(path)
    if new and path.exists():
        raise FileExistsError("output exists; use a new run: %s" % path)
    atomic_json_dump(value, str(path))


def validate_config(cfg):
    cfg = copy.deepcopy(cfg)
    if cfg.get("protocol") != PROTOCOL or cfg.get("output_projection") != "float32":
        raise ValueError("v1 requires its named protocol and the shared FP32 output projection")
    for key in ("seed", "tune_pairs", "batch_size", "max_new_tokens", "epochs", "top_k"):
        if type(cfg.get(key)) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid integer setting %s" % key)
    if cfg["seed"] >= 2**32 or cfg["batch_size"] < 2 or cfg["batch_size"] % 2:
        raise ValueError("seed must be uint32; paired scoring requires an even batch_size >= 2")
    for key in ("lr", "clip", "check_alpha"):
        if not isinstance(cfg.get(key), (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("invalid positive setting %s" % key)
    fields = [name for _, name, _, _ in string.Formatter().parse(cfg["qa_user"]) if name is not None]
    if not fields or set(fields) != {"question"}:
        raise ValueError("qa_user must contain only the {question} placeholder")
    if not isinstance(cfg.get("qa_system"), str):
        raise ValueError("qa_system must be text")
    fmt = cfg.get("selfcheck_format", "same_turn")
    if fmt not in ("same_turn", "user_turn"):
        raise ValueError("selfcheck_format must be same_turn (v1) or user_turn")
    for key in ("selfcheck_label_true", "selfcheck_label_false") + (
            ("selfcheck_suffix",) if fmt == "same_turn" else ("selfcheck_user",)):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError("empty self-check setting %s" % key)
    if cfg["selfcheck_label_true"] == cfg["selfcheck_label_false"]:
        raise ValueError("self-check labels must differ")
    if (not cfg["alphas"] or len(set(cfg["alphas"])) != len(cfg["alphas"])
            or any(not math.isfinite(a) or a <= 0 for a in cfg["alphas"])):
        raise ValueError("candidate alphas must be unique finite positive values; baseline is separate")
    if set(cfg["write_modes"]) != {"prefill", "continuous"} or len(cfg["write_modes"]) != 2:
        raise ValueError("v1 records prefill and continuous; only continuous enters main selection")
    if cfg["main_mode"] != "continuous" or cfg["top_k"] != 3:
        raise ValueError("v1 fixes continuous selection and Top-3 distinct layers per estimator")
    if cfg["answer_rate_tolerance"] != 0.03:
        raise ValueError("v1 fixes the real-answer-rate tolerance at 3 percentage points")
    return cfg


def choose_layers(args, n):
    if args.all_layers:
        return list(range(n))
    layers = args.layers
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= l < n for l in layers):
        raise ValueError("explicit valid --layers or --all-layers is required for estimate")
    return sorted(layers)


def token_signature(pairs):
    return fingerprint([{ "pair_id": p["pair_id"], **{
        side: {"ids": p[side]["input_ids"], **{k: p[side][k] for k in ("answer_end", "label_predictor", "label_ids")}}
        for side in ("positive", "negative")}} for p in pairs])


def bank_identity(meta):
    return fingerprint({k: v for k, v in meta.items() if k != "fingerprint"})


def capture_references(bundle, pairs, layers):
    """Capture one answer at a time, as in per-pair DAS training; all layers share the pass.

    The suffix is present but only original answer positions are read. MD and
    DAS use these identical stored states, excluding every suffix/label token.
    """
    ends = {side: {l: [] for l in layers} for side in ("positive", "negative")}
    means = {side: {l: [] for l in layers} for side in ("positive", "negative")}
    for pair in pairs:
        for side in ("positive", "negative"):
            item = pair[side]
            with capture_hooks(bundle, layers, ("resid",), position="all") as stored:
                forward_logits(bundle, item["inputs"], logits_to_keep=1)
            start, end = item["encoded"]["prompt_length"], item["answer_end"]
            for layer in layers:
                h = stored[("resid", layer)][0, start:end + 1].detach().float()
                if not torch.isfinite(h).all():
                    raise ValueError("non-finite reference activation: %s/%s/L%d" % (pair["pair_id"], side, layer))
                ends[side][layer].append(h[-1].cpu().numpy())
                means[side][layer].append(h.mean(0).cpu().numpy())
    return ({s: {l: np.stack(v) for l, v in by.items()} for s, by in ends.items()},
            {s: {l: np.stack(v) for l, v in by.items()} for s, by in means.items()})


def estimate(args, cfg, bundle=None):
    run = args.run_dir
    if any((run / p).exists() for p in ("bank.json", "bank.npz", "questions.json")):
        raise FileExistsError("estimate requires a new run directory")
    prepared = legacy.split_questions(cfg, args.pilot)
    model_cfg = load_model_config(args.model)
    if model_cfg["name"] != args.model:
        raise ValueError("--model must be the registered model name, not an alias/config path")
    set_seed(cfg["seed"])
    bundle = bundle or load_model(model_cfg)
    layers = choose_layers(args, bundle.num_layers)
    if not any(l < bundle.num_layers - 1 for l in layers):
        raise ValueError("include an earlier block for QA-DAS; the last block output alone is a structural null")
    encoded = prepare_selfcheck_pairs(bundle, cfg, prepared["e_train"])
    with fp32_output_projection(bundle) as precision:
        ends, means = capture_references(bundle, encoded, layers)
        prompts = [legacy.qa_text(bundle, cfg, q["question"]) for q in prepared["splits"]["train"]]
        prompt_h, _ = capture_last_token(bundle, prompts, layers, batch_size=cfg["batch_size"])
    arrays, rows, md_layers = {}, {}, []
    for l in layers:
        hp, hn = ends["positive"][l], ends["negative"][l]
        w, details = mean_diff_direction(hp, hn)
        arrays["E_qa_md_L%d" % l] = w
        arrays["source_positive_L%d" % l], arrays["source_negative_L%d" % l] = hp, hn
        arrays["prompt_center_L%d" % l] = prompt_h[("resid", l)].numpy().mean(0)
        arrays["decode_center_L%d" % l] = .5 * (means["positive"][l].mean(0) + means["negative"][l].mean(0))
        valid = bool(np.isfinite(w).all() and np.linalg.norm(w) > 1e-8)
        if valid:
            md_layers.append(l)
        rows[str(l)] = {"md_gap": float(details["gap"]), "md_status": "usable" if valid else "zero_direction",
                        "das_status": "structural_null" if l == bundle.num_layers - 1 else "not_yet_trained"}
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("non-finite direction bank")
    run.mkdir(parents=True, exist_ok=True)
    save(run / "questions.json", prepared, new=True)
    with open(run / "bank.npz.tmp", "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(run / "bank.npz.tmp", run / "bank.npz")
    meta = {"protocol": PROTOCOL, "model": args.model, "model_config": model_cfg, "config": cfg,
            "hidden_size": bundle.hidden_size, "num_layers": bundle.num_layers, "layers": layers,
            "md_layers": md_layers, "das_candidate_layers": [l for l in layers if l < bundle.num_layers - 1],
            "structural_null_layers": [l for l in layers if l == bundle.num_layers - 1],
            "pilot": args.pilot, "rows": rows, "precision": precision,
            "n_E_pairs": len(encoded), "n_prompt_center_questions": len(prompts),
            "training_pair_ids": [p["pair_id"] for p in encoded],
            "source_tokens_sha256": token_signature(encoded),
            "bank_sha256": digest(run / "bank.npz"), "questions_sha256": digest(run / "questions.json"),
            "read_position": "answer_end block output; fixed original QA+answer token IDs before the suffix",
            "decode_center": "class-balanced mean of per-answer means over original answer content positions",
            "token_audit": [{"pair_id": p["pair_id"], **{s: {k: p[s][k] for k in (
                "answer_end", "label_predictor", "label_ids", "text_audit")} for s in ("positive", "negative")}}
                            for p in encoded[:3]]}
    meta["fingerprint"] = bank_identity(meta)
    save(run / "bank.json", meta, new=True)
    print("ESTIMATE", args.model, "pairs", len(encoded), "MD layers", md_layers,
          "DAS candidate layers", meta["das_candidate_layers"], "structural null", meta["structural_null_layers"])
    return meta


def load_bank(args):
    meta = read(args.run_dir / "bank.json")
    if meta.get("protocol") != PROTOCOL or meta["model"] != args.model or bank_identity(meta) != meta["fingerprint"]:
        raise ValueError("bank model/protocol/metadata identity mismatch")
    validate_config(meta["config"])
    for path, key in (("bank.npz", "bank_sha256"), ("questions.json", "questions_sha256")):
        if digest(args.run_dir / path) != meta[key]:
            raise ValueError("bank artifact changed: " + path)
    with np.load(args.run_dir / "bank.npz", allow_pickle=False) as z:
        bank = {k: z[k] for k in z.files}
    prepared = read(args.run_dir / "questions.json")
    for l in meta["layers"]:
        for key in ("E_qa_md_L%d", "prompt_center_L%d", "decode_center_L%d"):
            a = bank[key % l]
            if a.shape != (meta["hidden_size"],) or not np.isfinite(a).all():
                raise ValueError("invalid bank vector " + key % l)
        for side in ("positive", "negative"):
            a = bank["source_%s_L%d" % (side, l)]
            if a.shape != (meta["n_E_pairs"], meta["hidden_size"]) or not np.isfinite(a).all():
                raise ValueError("invalid source cache")
    return meta, bank, prepared


def open_bundle(meta, prepared):
    set_seed(meta["config"]["seed"])
    bundle = load_model(meta["model_config"])
    verify_bundle(bundle, meta, prepared)
    return bundle


def verify_bundle(bundle, meta, prepared):
    if bundle.hidden_size != meta["hidden_size"] or bundle.num_layers != meta["num_layers"]:
        raise ValueError("loaded architecture differs from the estimated model")
    if getattr(bundle, "cfg", None) != meta["model_config"]:
        raise ValueError("loaded model configuration differs from the frozen configuration")
    pairs = prepare_selfcheck_pairs(bundle, meta["config"], prepared["e_train"])
    if token_signature(pairs) != meta["source_tokens_sha256"]:
        raise ValueError("tokenizer/rendering changed since direction estimation")
    return pairs


def checkpoint_path(run, layer, seed):
    return run / "das" / ("das_E_L%d_r1_s%d_%s.pt" % (layer, seed, PROTOCOL))


def validate_checkpoint(row, meta, layer):
    cfg = meta["config"]
    expected = {"protocol": PROTOCOL, "model_name": meta["model"], "layer": layer, "rank": 1,
                "seed": cfg["seed"], "n_pairs": meta["n_E_pairs"], "epochs": cfg["epochs"],
                "lr": cfg["lr"], "clip": cfg["clip"], "optimizer": "Adam", "trim": 0.0, "weighted": False,
                "model_config": meta["model_config"], "training_pair_ids": meta["training_pair_ids"],
                "training_pair_ids_sha256": fingerprint(meta["training_pair_ids"]),
                "selfcheck_inputs_sha256": meta["source_tokens_sha256"], "bank_fingerprint": meta["fingerprint"],
                "source_tokens_sha256": meta["source_tokens_sha256"],
                "loss_kind": "full_vocabulary_label_cross_entropy", "positions": "answer_end",
                "anchor": "answer_end", "bidirectional": True}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("DAS checkpoint belongs to another training protocol/layer/model")
    w = torch.as_tensor(row["W"]).detach().float().cpu().numpy()
    if w.shape != (meta["hidden_size"], 1) or not np.isfinite(w).all() or abs(np.linalg.norm(w) - 1) > 1e-4:
        raise ValueError("invalid trained rank1 direction")
    if not row.get("losses") or not np.isfinite(row["losses"]).all():
        raise ValueError("invalid training loss record")
    return w[:, 0].astype(np.float64)


def train(args, meta, bank, prepared, bundle=None):
    if bundle is None:
        bundle = open_bundle(meta, prepared)
    else:
        verify_bundle(bundle, meta, prepared)
    pairs = verify_bundle(bundle, meta, prepared)
    cfg, records = meta["config"], {}
    with fp32_output_projection(bundle):
        for l in meta["das_candidate_layers"]:
            path = checkpoint_path(args.run_dir, l, cfg["seed"])
            if path.exists():
                row = torch.load(path, map_location="cpu", weights_only=False)
                validate_checkpoint(row, meta, l)
            else:
                row = train_qa_das(bundle, pairs, l, epochs=cfg["epochs"], lr=cfg["lr"], seed=cfg["seed"], clip=cfg["clip"],
                                   source_activations={s: torch.from_numpy(bank["source_%s_L%d" % (s, l)])
                                                       for s in ("positive", "negative")},
                                   identity={"bank_fingerprint": meta["fingerprint"],
                                             "source_tokens_sha256": meta["source_tokens_sha256"]})
                validate_checkpoint(row, meta, l)
                path.parent.mkdir(exist_ok=True)
                torch.save(row, str(path) + ".tmp")
                os.replace(str(path) + ".tmp", path)
            records[str(l)] = {"file": str(path.relative_to(args.run_dir)), "sha256": digest(path),
                               "losses": row["losses"], "position_trace": row["position_trace"],
                               "per_seed": row["per_seed"]}
    manifest = {"protocol": PROTOCOL, "bank_fingerprint": meta["fingerprint"], "model": meta["model"],
                "seed": cfg["seed"], "layers": records, "structural_null_layers": meta["structural_null_layers"],
                "completed": True}
    save(args.run_dir / "training.json", manifest)
    return manifest


def runtime_bank(args, meta, bank):
    manifest = read(args.run_dir / "training.json")
    if (not manifest.get("completed") or manifest.get("bank_fingerprint") != meta["fingerprint"]
            or manifest.get("model") != meta["model"]
            or set(manifest["layers"]) != {str(l) for l in meta["das_candidate_layers"]}):
        raise ValueError("complete the exact direction training grid before evaluation")
    bank = dict(bank)
    for l in meta["das_candidate_layers"]:
        row = manifest["layers"][str(l)]
        path = checkpoint_path(args.run_dir, l, meta["config"]["seed"])
        if row["file"] != str(path.relative_to(args.run_dir)) or digest(path) != row["sha256"]:
            raise ValueError("trained checkpoint changed")
        bank["E_qa_das_L%d" % l] = validate_checkpoint(torch.load(path, map_location="cpu", weights_only=False), meta, l)
    runtime = {**meta, "training_sha256": digest(args.run_dir / "training.json")}
    runtime["protocol_fingerprint"] = fingerprint([meta["fingerprint"], runtime["training_sha256"]])
    return runtime, bank


def condition(context, mode, layer, alpha):
    return {"name": "E_%s_%s_L%d_a%g" % (context, mode, layer, alpha), "kind": "E", "context": context,
            "mode": mode, "layer": layer, "alpha": float(alpha), "uses_reference_answer": False,
            "eligible_for_selection": True}


def conditions(meta):
    cfg = meta["config"]
    out = [dict(BASELINE)]
    for method, layers in (("qa_das", meta["das_candidate_layers"]), ("qa_md", meta["md_layers"])):
        for mode in cfg["write_modes"]:
            for l in layers:
                out += [condition(method, mode, l, a) for a in cfg["alphas"]]
    return out


def score(args, meta, bank, prepared, bundle=None):
    cfg, pairs = meta["config"], prepared["e_tune"]
    choices, ids = conditions(meta), [p["pair_id"] for p in pairs]
    signature = fingerprint([meta["protocol_fingerprint"], choices, ids])
    path = args.run_dir / "proxy_scores.json"
    result = read(path) if path.exists() else {"fingerprint": signature, "model": meta["model"],
              "protocol_fingerprint": meta["protocol_fingerprint"], "question_ids": ids, "condition_specs": choices,
              "metric": "paired_answer_selection_accuracy_sum_logprob", "conditions": {}}
    if result.get("fingerprint") != signature or result["condition_specs"] != choices or result["question_ids"] != ids:
        raise ValueError("proxy resume inputs/candidates changed")
    bundle = bundle or open_bundle(meta, prepared)
    # Interleave each pair and use an even batch, so its two answers never split across batches.
    encoded = [encode_answer(bundle, cfg["qa_system"], r["question"], r[key], cfg["qa_user"])
               for r in pairs for key in ("positive_answer", "negative_answer")]
    with fp32_output_projection(bundle):
        for c in choices:
            if c["name"] in result["conditions"]:
                continue
            scores = []
            for start in range(0, len(encoded), cfg["batch_size"]):
                packed = pack_answers(encoded[start:start + cfg["batch_size"]], bundle.tokenizer.pad_token_id, bundle.device)
                with steer(bundle, legacy.specs_for(meta, bank, c, packed)):
                    values = answer_logprobs(forward_logits(bundle, packed["inputs"]), packed).detach().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("non-finite paired score")
                scores.extend(values.tolist())
            pos, neg = np.array(scores[0::2]), np.array(scores[1::2])
            correct = pos > neg
            base = correct if c["name"] == "baseline" else np.array(result["conditions"]["baseline"]["correct"], bool)
            result["conditions"][c["name"]] = {"condition": c, "positive_logp": pos.tolist(), "negative_logp": neg.tolist(),
                "margin": (pos - neg).tolist(), "positive_tokens": [e["answer_length"] for e in encoded[0::2]],
                "negative_tokens": [e["answer_length"] for e in encoded[1::2]],
                "correct": correct.tolist(), **legacy.paired_stats(base, correct)}
            save(path, result)
    result["completed"] = True
    save(path, result)
    candidates = rank_proxy_candidates(choices, result["conditions"], top_k=cfg["top_k"], main_mode=cfg["main_mode"])
    short = {"protocol_fingerprint": meta["protocol_fingerprint"], "source_scores_sha256": digest(path),
             "candidates": candidates, "report_scope": "best free-generation configuration among proxy-shortlisted distinct layers"}
    save(args.run_dir / "shortlist.json", short)
    return result, short


def load_shortlist(args, meta, prepared):
    scores, short = read(args.run_dir / "proxy_scores.json"), read(args.run_dir / "shortlist.json")
    choices = conditions(meta)
    ids = [q["pair_id"] for q in prepared["e_tune"]]
    expected = fingerprint([meta["protocol_fingerprint"], choices, ids])
    if (not scores.get("completed") or scores.get("fingerprint") != expected
            or scores.get("protocol_fingerprint") != meta["protocol_fingerprint"]
            or scores.get("condition_specs") != choices or scores.get("question_ids") != ids
            or short.get("source_scores_sha256") != digest(args.run_dir / "proxy_scores.json")
            or short.get("protocol_fingerprint") != meta["protocol_fingerprint"]):
        raise ValueError("shortlist/proxy provenance mismatch")
    for spec in choices:
        row = scores["conditions"].get(spec["name"], {})
        positive, negative = np.asarray(row.get("positive_logp", [])), np.asarray(row.get("negative_logp", []))
        if (row.get("condition") != spec or positive.shape != (len(ids),) or negative.shape != (len(ids),)
                or not np.isfinite(positive).all() or not np.isfinite(negative).all()
                or row.get("correct") != (positive > negative).tolist()):
            raise ValueError("proxy record is incomplete or disagrees with its answer scores")
    expected_short = rank_proxy_candidates(choices, scores["conditions"], top_k=meta["config"]["top_k"])
    if short["candidates"] != expected_short:
        raise ValueError("shortlist differs from the predeclared Top-3 rule")
    return short, [dict(BASELINE)] + [c for method in METHODS for c in expected_short[method]]


def verify_question_rows(rows, questions):
    if len(rows) != len(questions):
        raise ValueError("question count changed")
    for index, (row, question) in enumerate(zip(rows, questions)):
        if row.get("idx") != index or any(row.get(key) != question.get(key) for key in (
                "pair_id", "question", "gold", "incorrect", "best_answer")):
            raise ValueError("question identity/text/references differ from the frozen question file")


def generate_with_audit(bundle, inputs, specs, cfg, trace_displacement=False):
    """Observe before/after values around the production hook without changing them."""
    before, records, handles = {}, [], []
    by_layer = {s.layer: s for s in specs}
    def pre(layer):
        def observe(module, incoming, output):
            h = output[0] if isinstance(output, tuple) else output
            before[layer] = h[:, -1, :].detach().float().clone()
        return observe
    def post(layer, hooks):
        def observe(module, incoming, output):
            h = output[0] if isinstance(output, tuple) else output
            after = h[:, -1, :].detach().float()
            original = before.pop(layer)
            delta = after - original
            w = by_layer[layer].W.to(after.device).float()
            scale = original.norm(dim=1)
            records.append({"layer": layer, "step": hooks[layer].call_count - 1,
                "h_norm": scale.cpu().tolist(), "kept_delta_norm": delta.norm(dim=1).cpu().tolist(),
                "kept_relative_norm": (delta.norm(dim=1) / scale.clamp_min(1e-12)).cpu().tolist(),
                "projection_before": (original @ w).cpu().tolist(),
                "projection_after": (after @ w).cpu().tolist()})
        return observe
    try:
        if trace_displacement:
            handles += [bundle.layer(l).register_forward_hook(pre(l)) for l in by_layer]
        with torch.no_grad(), steer(bundle, specs) as hooks:
            if trace_displacement:
                handles += [bundle.layer(l).register_forward_hook(post(l, hooks)) for l in by_layer]
            output = bundle.model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                           num_beams=1, use_cache=True, pad_token_id=bundle.tokenizer.pad_token_id)
        return output, hooks, records
    finally:
        for handle in handles:
            handle.remove()


def generate_questions(args, meta, bank, questions, choices, filename, split, bundle=None,
                       reference_states=None, reuse=None):
    cfg, path = meta["config"], args.run_dir / filename
    diagnostic = filename == "tune_diagnostics.json"
    if any(c.get("uses_reference_answer") for c in choices) and not diagnostic:
        raise ValueError("reference-answer interventions are forbidden outside the diagnostic artifact")
    ids = [q["pair_id"] for q in questions]
    signature = fingerprint([meta["protocol_fingerprint"], choices, split, ids])
    if path.exists():
        data = read(path)
        if (data.get("protocol_fingerprint") != signature or data.get("condition_specs") != choices
                or data.get("model") != meta["model"] or data.get("split") != split
                or data.get("bank_protocol_fingerprint") != meta["protocol_fingerprint"]
                or data.get("training_sha256") != meta["training_sha256"]
                or data.get("questions_sha256") != meta["questions_sha256"]
                or data.get("conditions") != [c["name"] for c in choices]):
            raise ValueError("generation resume protocol changed")
    else:
        data = {"model": meta["model"], "protocol": PROTOCOL, "protocol_fingerprint": signature,
                "bank_protocol_fingerprint": meta["protocol_fingerprint"], "bank_sha256": meta["bank_sha256"],
                "training_sha256": meta["training_sha256"], "questions_sha256": meta["questions_sha256"],
                "split": split, "pilot": meta["pilot"], "diagnostic_only": diagnostic, "precision": meta["precision"],
                "E": {"methods": list(METHODS), "primary_operator": "centered amplification", "main_mode": "continuous",
                      "reference_diagnostic_operator": "fixed-coordinate interchange" if diagnostic else None},
                "C": {"enabled": False}, "prompt": {k: cfg[k] for k in ("qa_system", "qa_user", "max_new_tokens")},
                "intervention_position": "prompt-end predictor once; continuous also writes each later predictor; fixed reference coordinates only in labeled diagnostics",
                "conditions": [c["name"] for c in choices], "condition_specs": choices, "audit": [],
                "evaluations": [{**q, "idx": i, "outputs": {}, "finished": {}, "rule_abstain": {}} for i, q in enumerate(questions)]}
    rows = data["evaluations"]
    verify_question_rows(rows, questions)
    if reuse is not None:
        if reuse.get("bank_protocol_fingerprint") != meta["protocol_fingerprint"]:
            raise ValueError("cannot reuse a baseline from a different run/cohort")
        verify_question_rows(reuse["evaluations"], questions)
        for row, old in zip(rows, reuse["evaluations"]):
            for c in choices:
                n = c["name"]
                if n in old["outputs"]:
                    for key in ("outputs", "finished", "rule_abstain"):
                        if n in row[key] and row[key][n] != old[key][n]:
                            raise ValueError("reused generation disagrees with existing output")
                        row[key][n] = old[key][n]
    generation_prepared = read(args.run_dir / "questions.json")
    if bundle is None:
        bundle = open_bundle(meta, generation_prepared)
    else:
        verify_bundle(bundle, meta, generation_prepared)
    eos = getattr(bundle.model.generation_config, "eos_token_id", None)
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos if eos is not None else bundle.tokenizer.eos_token_id])
    with fp32_output_projection(bundle):
        for c in choices:
            name = c["name"]
            for start in range(0, len(rows), cfg["batch_size"]):
                chunk = rows[start:start + cfg["batch_size"]]
                if all(name in q["outputs"] for q in chunk):
                    continue
                texts = [legacy.qa_text(bundle, cfg, q["question"]) for q in chunk]
                inputs = bundle.encode(texts)
                if not inputs["attention_mask"][:, -1].all():
                    raise ValueError("generation requires left padding")
                specs = legacy.specs_for(meta, bank, c) if not c.get("uses_reference_answer") else None
                reference_coordinates = None
                if specs is None:
                    l = c["layer"]
                    if reference_states is None or c["mode"] != "continuous" or c["alpha"] != 1.0:
                        raise ValueError("reference diagnostic requires fixed coordinates and unit interchange at every step")
                    source = np.stack([reference_states[q["pair_id"]][l] for q in chunk])
                    specs = [SteerSpec(bank["E_%s_L%d" % (c["context"], l)], l, mode="interchange",
                                       alpha=1.0, source_act=source, position="last", name=name)]
                    specs[0].to(bundle.device)
                    reference_coordinates = (specs[0].source_act @ specs[0].W).detach().cpu().tolist()
                plen = inputs["input_ids"].shape[1]
                output, hooks, displacement = generate_with_audit(
                    bundle, inputs, specs, cfg, trace_displacement=diagnostic or filename == "check_generations.json")
                n_new = output.shape[1] - plen
                expected = 1 if c.get("mode") == "prefill" else n_new
                if any(s.n_applied != expected for s in specs) or any(h.first_shape[1] != plen or h.call_count != n_new for h in hooks.values()):
                    raise RuntimeError("prefill/decode intervention audit failed")
                for row, tokens, prompt_text in zip(chunk, output[:, plen:], texts):
                    answer = bundle.tokenizer.decode(tokens, skip_special_tokens=True)
                    if name in row["outputs"] and row["outputs"][name] != answer:
                        raise RuntimeError("resume changed an already generated answer")
                    row["outputs"][name] = answer
                    row["finished"][name] = any(int(t) in eos for t in tokens)
                    row["rule_abstain"][name] = bool(rule_abstains(answer))
                    if meta["pilot"] or filename == "check_generations.json":
                        print("PILOT", row["pair_id"], name, repr(prompt_text), repr(answer))
                data["audit"].append({"condition": name, "batch_start": start, "prompt_len": plen,
                                      "new_tokens": n_new, "edits": {s.name: s.n_applied for s in specs},
                                      "displacement": displacement,
                                      "reference_coordinates": reference_coordinates,
                                      "displacement_scope": "per batch step; includes padding steps of already-finished sequences"})
                save(path, data)
    data["completed"] = True
    save(path, data)
    return data


def judge(args, meta, filename, output_stem):
    import e24_real_qa as e24
    path = args.run_dir / filename
    data = read(path)
    if not data.get("completed") or data.get("bank_protocol_fingerprint") != meta["protocol_fingerprint"]:
        raise ValueError("finish the matching generation before judging")
    if (data.get("model") != args.model or data.get("conditions") != [c["name"] for c in data["condition_specs"]]
            or len(set(data["conditions"])) != len(data["conditions"])):
        raise ValueError("generation identity/condition list disagrees before grading")
    ids = [row.get('pair_id') for row in data.get('evaluations', [])]
    if not ids or any(not isinstance(pid, str) or not pid for pid in ids) or len(ids) != len(set(ids)):
        raise ValueError('grading requires unique nonempty question IDs; duplicate questions must not be silently deduplicated')
    stem = args.run_dir / output_stem
    if output_stem == "tune_scored" and (args.run_dir / "selection.json").exists():
        raise ValueError("tuning grades are frozen; start a new run to change them")
    # E24 can summarize only the jointly resolved subset. This protocol must
    # instead finish every grade before publishing a fixed-denominator table.
    # Checkpoint the grades first, including the all-unresolved case which E24
    # cannot summarize, and reuse both these rows and the existing judge cache.
    from core.judge import judge_from_config
    judge_cfg = e24.load_experiment(args.judge_exp)["judge"]
    judge_cache_dir = getattr(args, 'judge_cache_dir', None) or os.path.join(e24.RESULTS_DIR, "judge_cache")
    grader = judge_from_config(judge_cfg, cache_dir=judge_cache_dir)
    if getattr(grader, "qa_rubric_id", None) != QA_GRADER_VERSION:
        raise ValueError("the deployed judge must use the current semantic rubric")
    source_hash, judge_hash = digest(path), fingerprint(judge_cfg)
    pending_path = Path(str(stem) + "_grading_input.json")
    if pending_path.exists():
        pending = read(pending_path)
        if (pending.get("grading_source_generation_sha256") != source_hash
                or pending.get("grading_judge_config_sha256") != judge_hash
                or pending.get("grading_judge_name") != grader.name):
            raise ValueError("pending grades belong to different generation or judge settings")
        verify_question_rows([{**row, 'idx': i} for i, row in enumerate(pending["evaluations"])], data["evaluations"])
        if (pending.get("conditions") != data["conditions"]
                or pending.get("condition_specs") != data["condition_specs"]
                or any(a["outputs"] != b["outputs"] for a, b in zip(pending["evaluations"], data["evaluations"]))):
            raise ValueError("pending grading inputs differ from the original generation")
    else:
        pending = copy.deepcopy(data)
        pending.update(grading_source_generation_sha256=source_hash,
                       grading_judge_config_sha256=judge_hash, grading_judge_name=grader.name)
    # idx is display/order metadata, not question identity. E24's legacy
    # reporter deduplicates by idx, so missing/duplicate old indices otherwise
    # crash or silently discard valid distinct questions. Normalize only this
    # derived grading copy; preserve the original generation bytes and hash.
    index_changes = []
    for index, row in enumerate(pending['evaluations']):
        if type(row.get('idx')) is not int or row['idx'] != index:
            index_changes.append({'pair_id': row['pair_id'], 'original_idx': row.get('idx'), 'grading_idx': index})
        row['idx'] = index
    if 'grading_row_index_normalization' not in pending:
        pending['grading_row_index_normalization'] = {
            'policy': 'enumerate_unique_question_ids_in_original_order',
            'source_generation_unchanged': True, 'changes': index_changes}
    pending["grading_completed"] = False
    for row in pending["evaluations"]:
        references, only_refusal = reference_answers(row["gold"])
        previous_references = row.get("judge_reference_answers")
        row["judge_reference_answers"], row["gold_is_abstention"] = references, only_refusal
        row.setdefault("judge", {})
        row.setdefault("outcome", {})
        for name in data["conditions"]:
            cached = row["judge"].get(name, {})
            if (previous_references != references or cached.get("rubric_id") != QA_GRADER_VERSION
                    or cached.get("label") not in ("correct", "wrong", "abstain")):
                row["judge"][name] = grader.qa_grade(
                    row["question"], references, row["outputs"][name], incorrect=row.get("incorrect") or None)
            # The requested config owns the policy even for a custom semantic
            # grader or reused old grade. This is idempotent for LLMJudge results.
            from core.qa_scoring import apply_configured_outcome_policy
            row['judge'][name] = apply_configured_outcome_policy(
                row['judge'][name], row['question'], row['outputs'][name], judge_cfg)
            grade = row["judge"][name]
            if grade.get("rubric_id") != QA_GRADER_VERSION:
                raise ValueError("an item uses a different grading rubric")
            row["outcome"][name] = judged_outcome(grade["label"])
            save(pending_path, pending)
    unresolved = [{"pair_id": row["pair_id"], "condition": name}
                  for row in pending["evaluations"] for name in data["conditions"]
                  if row["outcome"][name] not in ("correct", "wrong", "abstain")]
    if unresolved:
        partial = {"model": args.model, "completed": False, "status": "grading_incomplete",
                   "source_generation_sha256": source_hash, "judge": grader.name,
                   "judge_rubric_id": QA_GRADER_VERSION,
                   "n_questions": len(pending["evaluations"]),
                   "n_substantive_gold": sum(not r["gold_is_abstention"] for r in pending["evaluations"]),
                   "n_unresolved": len(unresolved), "unresolved": unresolved,
                   "note": "No main rates or intervals: retry this phase to resolve all grades; the denominator is fixed."}
        save(Path(str(stem) + ".json"), partial)
        save(Path(str(stem) + "_items.json"), {**pending, "completed": False, "judge_summary": partial})
        Path(str(stem) + ".md").write_text(
            "# 语义判分未完成\n\n尚有 %d 条判定未完成；已保存逐题结果，可重跑同一阶段补判。"
            "\n\n未发布正确率、作答率或区间主表；不会删除未判题后改变分母。\n" % len(unresolved), encoding="utf-8")
        raise RuntimeError("semantic grading incomplete: %d unresolved question/condition grades; retry the same phase" % len(unresolved))
    frozen_controls = getattr(args, 'frozen_control_grades', None)
    if frozen_controls is not None:
        if set(frozen_controls) != {r['pair_id'] for r in pending['evaluations']}:
            raise ValueError('frozen control grades do not cover the exact evaluation cohort')
        verified = 0
        for row in pending['evaluations']:
            for name, expected_grade in frozen_controls[row['pair_id']].items():
                actual = row['judge'].get(name, {})
                if (actual.get('label') != expected_grade.get('label')
                        or actual.get('rubric_id') != expected_grade.get('rubric_id')):
                    raise ValueError('identical control question/output received a different grade: ' + row['pair_id'] + '/' + name)
                verified += 1
        pending['frozen_control_grade_verification'] = {
            'source_scored_items_sha256': getattr(args, 'frozen_control_source_sha256', None),
            'question_condition_grades_checked': verified, 'all_identical': True}
    pending["grading_completed"] = True
    save(pending_path, pending)
    # The legacy reporter reuses these already-resolved labels and performs its
    # normal reference partition, transitions and paired bootstrap. Bind its
    # published provenance back to the original, unmodified generation file.
    e24.phase_judge(SimpleNamespace(model=args.model, tag="E25_DAS_" + data["split"], phase="judge",
        judge_exp=args.judge_exp, input_file=str(pending_path), output_prefix=str(stem), rejudge=False,
        n_boot=2000, judge_cache_dir=judge_cache_dir))
    summary_path, items_path = Path(str(stem) + ".json"), Path(str(stem) + "_items.json")
    summary, items = read(summary_path), read(items_path)
    summary.update(completed=True, status="complete", source_generation_sha256=source_hash)
    items.update(completed=True, grading_completed=True, judge_summary=summary)
    save(summary_path, summary)
    save(items_path, items)


def freeze(args, meta, prepared, persist=True):
    short, choices = load_shortlist(args, meta, prepared)
    raw, summary, judged = (read(args.run_dir / p) for p in ("tune.json", "tune_scored.json", "tune_scored_items.json"))
    ids = [r["pair_id"] for r in prepared["splits"]["tune"]]
    expected = fingerprint([meta["protocol_fingerprint"], choices, "tune", ids])
    if (not raw.get("completed") or raw.get("protocol_fingerprint") != expected or raw.get("condition_specs") != choices
            or raw.get("split") != "tune" or raw.get("model") != args.model
            or [r["pair_id"] for r in raw["evaluations"]] != ids):
        raise ValueError("tuning generation does not match the frozen shortlist/data")
    verify_question_rows(raw["evaluations"], prepared["splits"]["tune"])
    if (summary.get("source_generation_sha256") != digest(args.run_dir / "tune.json")
            or summary.get("judge_rubric_id") != QA_GRADER_VERSION
            or summary.get("model") != args.model
            or summary.get("header", {}).get("protocol_fingerprint") != expected
            or judged.get("judge_rubric_id") != QA_GRADER_VERSION
            or judged.get("protocol_fingerprint") != expected
            or judged.get("model") != args.model or judged.get("split") != "tune"
            or judged.get("condition_specs") != choices
            or judged.get("conditions") != [c["name"] for c in choices]
            or judged.get("judge_summary") != summary
            or [r["pair_id"] for r in judged["evaluations"]] != ids):
        raise ValueError("semantic grades are stale, foreign or use the wrong rubric")
    verify_question_rows(judged["evaluations"], prepared["splits"]["tune"])
    if any(summary.get(k) != len(ids) for k in ("n_questions", "n_substantive_gold", "n_resolved_all_conditions")):
        raise ValueError("tuning grading cohort is incomplete or its denominator changed")
    if any(r.get("gold_is_abstention") for r in judged["evaluations"]):
        raise ValueError("v1 tuning cohort contains refusal-only references; do not silently change its denominator")
    outcomes = {c["name"]: {} for c in choices}
    for original, item in zip(raw["evaluations"], judged["evaluations"]):
        if item["outputs"] != original["outputs"] or item["question"] != original["question"]:
            raise ValueError("judged response text changed")
        references, only_refusal = reference_answers(original["gold"])
        if item.get("judge_reference_answers") != references or item.get("gold_is_abstention") != only_refusal:
            raise ValueError("grading references/cohort disagree with the frozen raw references")
        for c in choices:
            name = c["name"]
            grade = item["judge"][name]
            if grade.get("rubric_id") != QA_GRADER_VERSION:
                raise ValueError("an item uses a different grading rubric")
            semantic = judged_outcome(grade["label"])
            if item["outcome"][name] != semantic:
                raise ValueError("derived outcome disagrees with its semantic judge label")
            outcomes[name][item["pair_id"]] = semantic
    decision = select_real_candidates(choices, outcomes, ids, tolerance=meta["config"]["answer_rate_tolerance"])
    for c in choices:
        name = c["name"]
        expected_rates = decision["reports"][name]
        observed = summary["rates"][name]
        for key, count_key in (("correct", "correct"), ("wrong", "wrong"), ("abstain", "abstain")):
            value = observed["rates"][key]
            if not math.isfinite(value) or abs(value - expected_rates["counts"][count_key] / len(ids)) > 1e-12:
                raise ValueError("semantic summary disagrees with per-question judge labels")
        if observed.get("n_unresolved") != 0:
            raise ValueError("unresolved grades must not enter selection")
    result = {**decision, "model": args.model, "protocol": PROTOCOL,
              "protocol_fingerprint": meta["protocol_fingerprint"], "pilot": meta["pilot"],
              "sources": {p: digest(args.run_dir / p) for p in (
                  "proxy_scores.json", "shortlist.json", "tune.json", "tune_scored.json", "tune_scored_items.json")},
              "report_scope": short["report_scope"]}
    if persist:
        save(args.run_dir / "selection.json", result, new=True)
        print(json.dumps(result["selected"], ensure_ascii=False, indent=2))
    return result


def selection(args, meta):
    row = read(args.run_dir / "selection.json")
    if row.get("model") != args.model or row.get("protocol_fingerprint") != meta["protocol_fingerprint"]:
        raise ValueError("selection belongs to another model/direction bank")
    expected_sources = {"proxy_scores.json", "shortlist.json", "tune.json", "tune_scored.json", "tune_scored_items.json"}
    if set(row.get("sources", {})) != expected_sources:
        raise ValueError("frozen selection requires the complete source manifest")
    for file, expected in row["sources"].items():
        if digest(args.run_dir / file) != expected:
            raise ValueError("frozen selection source changed: " + file)
    actual = freeze(args, meta, read(args.run_dir / "questions.json"), persist=False)
    if row != actual:
        raise ValueError("frozen selection was altered or does not follow the declared rule")
    return row


def check(args, meta, bank, prepared, bundle=None):
    bundle = bundle or open_bundle(meta, prepared)
    pairs = prepare_selfcheck_pairs(bundle, meta["config"], prepared["e_tune"][:3])
    reports, choices = [], [dict(BASELINE)]
    l = next(iter(meta["das_candidate_layers"]), None)
    if l is None:
        raise ValueError("pilot must include a layer with a path to the self-check label")
    precision_rows, single_hidden, items = [], [], []
    with fp32_output_projection(bundle):
        for method in METHODS:
            w = bank.get("E_%s_L%d" % (method, l))
            if w is None:
                continue
            report = evaluate_selfcheck(bundle, pairs, w, l, bank["decode_center_L%d" % l], meta["config"]["check_alpha"])
            reports.append({"context": method, "result": report})
            choices += [condition(method, mode, l, meta["config"]["check_alpha"]) for mode in meta["config"]["write_modes"]]
        for pair in pairs:
            for side in ("positive", "negative"):
                item = pair[side]
                tokens = sorted(set(item["label_ids"]["true"] + item["label_ids"]["false"]))
                diagnostic, hidden = projection_diagnostic(bundle, item["inputs"], token_ids=tokens, return_hidden=True)
                precision_rows.append({"pair_id": pair["pair_id"], "side": side, "projection": diagnostic})
                single_hidden.append(hidden)
                items.append(item)
        width = max(len(item["input_ids"]) for item in items)
        ids = [[bundle.tokenizer.pad_token_id] * (width - len(item["input_ids"])) + item["input_ids"] for item in items]
        masks = [[0] * (width - len(item["input_ids"])) + [1] * len(item["input_ids"]) for item in items]
        packed = {"input_ids": torch.tensor(ids, device=bundle.device),
                  "attention_mask": torch.tensor(masks, device=bundle.device)}
        batched, hidden_batch = projection_diagnostic(bundle, packed, token_ids=tokens, return_hidden=True)
        difference = (torch.cat(single_hidden).float() - hidden_batch.float()).abs()
    payload = {"protocol_fingerprint": meta["protocol_fingerprint"], "eligible_for_selection": False,
               "note": "wiring diagnostics, not efficacy or layer selection", "reports": reports,
               "precision_diagnostics": {"same_hidden_projection": precision_rows, "batch_projection": batched,
                   "single_vs_batch_hidden_max_abs_diff": float(difference.max()),
                   "single_vs_batch_hidden_per_row_max_abs_diff": difference.max(dim=1).values.tolist(),
                   "interpretation": "raw output projection and upstream batch differences are separate observations, not a proved cause of answer changes"}}
    audit_path = args.run_dir / "check_selfcheck.json"
    if audit_path.exists() and read(audit_path) != payload:
        raise ValueError("repeated pilot diagnostics changed; retain the original and investigate")
    if not audit_path.exists():
        save(audit_path, payload, new=True)
    return generate_questions(args, meta, bank, prepared["splits"]["tune"][:3], choices,
                              "check_generations.json", "pilot", bundle=bundle)


def diagnose(args, meta, bank, prepared, bundle=None):
    frozen = selection(args, meta)
    selected = [c for c in frozen["selected"].values() if c is not None]
    raw = read(args.run_dir / "tune.json")
    choices = [dict(BASELINE)] + selected
    for c in selected:
        prefill = {**c, "name": c["name"] + "__prefill_diagnostic", "mode": "prefill", "eligible_for_selection": False}
        oracle = {**c, "name": c["name"] + "__reference_diagnostic", "uses_reference_answer": True,
                  "eligible_for_selection": False, "operator": "fixed_reference_coordinate_interchange",
                  "parent_condition": c["name"], "parent_amplification_alpha": c["alpha"], "alpha": 1.0}
        choices += [prefill, oracle]
    bundle = bundle or open_bundle(meta, prepared)
    pairs = prepare_selfcheck_pairs(bundle, meta["config"], prepared["e_tune"])
    if [p["pair_id"] for p in pairs] != [q["pair_id"] for q in prepared["splits"]["tune"]]:
        raise ValueError("reference diagnostics require an explicit positive reference for every tuning question")
    references, reports = {p["pair_id"]: {} for p in pairs}, []
    layers = sorted({c["layer"] for c in selected})
    with fp32_output_projection(bundle):
        if layers:
            ends, _ = capture_references(bundle, pairs, layers)
            for i, pair in enumerate(pairs):
                references[pair["pair_id"]] = {l: ends["positive"][l][i] for l in layers}
        for c in selected:
            reports.append({"condition": c, "result": evaluate_selfcheck(bundle, pairs,
                bank["E_%s_L%d" % (c["context"], c["layer"])], c["layer"],
                bank["decode_center_L%d" % c["layer"]], c["alpha"])})
    payload = {"protocol_fingerprint": meta["protocol_fingerprint"],
               "selection_sha256": digest(args.run_dir / "selection.json"),
               "eligible_for_selection": False, "reports": reports}
    audit_path = args.run_dir / "diagnostic_selfcheck.json"
    if audit_path.exists() and read(audit_path) != payload:
        raise ValueError("repeated diagnostics changed; retain the original and investigate")
    if not audit_path.exists():
        save(audit_path, payload, new=True)
    return generate_questions(args, meta, bank, prepared["splits"]["tune"], choices,
                              "tune_diagnostics.json", "tune_diagnostic", bundle=bundle,
                              reference_states=references, reuse=raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["estimate", "train", "check", "judge-check", "tune-e", "generate-tune",
                                    "judge-tune", "freeze", "generate", "judge-test", "diagnose", "judge-diagnostics"])
    p.add_argument("--model", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", help="estimate only; subsequent phases use the frozen configuration")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--layers", nargs="+", type=int)
    group.add_argument("--all-layers", action="store_true")
    p.add_argument("--pilot", type=int, help="estimate only: first N questions per existing split; not efficacy evidence")
    p.add_argument("--judge-exp", default="E4_qa")
    args = p.parse_args()
    if args.phase != "estimate" and (args.config is not None or args.layers is not None or args.all_layers or args.pilot is not None):
        p.error("config, layers and pilot are frozen by estimate; use a new run to change them")
    if args.pilot is not None and args.pilot < 1:
        p.error("pilot must be positive")
    if args.phase == "estimate":
        if args.layers is None and not args.all_layers:
            p.error("estimate requires --layers or --all-layers")
        cfg = validate_config(load_json_config(resolve_config_path("experiments", args.config or "E25_qa_das_v1")))
        estimate(args, cfg)
        return
    meta, bank, prepared = load_bank(args)
    if args.phase == "train":
        train(args, meta, bank, prepared)
        return
    meta, bank = runtime_bank(args, meta, bank)
    if args.phase == "check":
        check(args, meta, bank, prepared)
    elif args.phase == "tune-e":
        score(args, meta, bank, prepared)
    elif args.phase == "generate-tune":
        _, choices = load_shortlist(args, meta, prepared)
        generate_questions(args, meta, bank, prepared["splits"]["tune"], choices, "tune.json", "tune")
    elif args.phase == "freeze":
        freeze(args, meta, prepared)
    elif args.phase == "generate":
        frozen = selection(args, meta)
        choices = [dict(BASELINE)] + [c for c in frozen["selected"].values() if c is not None]
        generate_questions(args, meta, bank, prepared["splits"]["test"], choices, "test.json", "test")
    elif args.phase == "diagnose":
        diagnose(args, meta, bank, prepared)
    else:
        file, stem = {"judge-check": ("check_generations.json", "check_scored"),
                      "judge-tune": ("tune.json", "tune_scored"),
                      "judge-test": ("test.json", "test_scored"),
                      "judge-diagnostics": ("tune_diagnostics.json", "diagnostics_scored")}[args.phase]
        judge(args, meta, file, stem)


if __name__ == "__main__":
    main()
