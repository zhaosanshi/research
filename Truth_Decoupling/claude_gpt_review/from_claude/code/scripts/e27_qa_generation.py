#!/usr/bin/env python3
"""Train a supervised generation adapter from DAS, mean-difference or random initialization.

This is not original DAS; random initialization is followed by correct-answer training.
Separate train/check/generate/judge/freeze phases; test generation requires a pre-test freeze.
Default: one layer from the source run's frozen DAS choice. --layers explicitly overrides it.
Original model, original DAS, and the adapted direction share prompts, alpha, centers and budget.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import common  # noqa: F401
import e25_qa_das as qa
import e25_qa_context as legacy
from core.config import load_experiment
from core.qa_generation_adapter import (PROTOCOL, adapter_specs, digest, fingerprint,
                                        load_adapter, save_adapter, train_generation_adapter,
                                        resolve_stop_token_id, stop_training_config,
                                        implementation_identity, training_description)
from core.qa_precision import fp32_output_projection
from core.readout import rule_abstains


DEFAULT_JUDGE_EXPERIMENT = "E4_qa"


def validate_judge_experiment(name):
    """Check the actual resolved config before entering a grading phase."""
    experiment = load_experiment(name)
    judge = experiment.get("judge")
    if not isinstance(judge, dict) or not isinstance(judge.get("model"), str) or not judge["model"].strip():
        raise ValueError("experiment %r has no usable judge configuration; use E4_qa or an explicit QA judge config" % name)
    return judge


def source_run(path):
    path = Path(path)
    stored = qa.read(path / "bank.json")
    ns = SimpleNamespace(run_dir=path, model=stored["model"])
    meta, bank, prepared = qa.load_bank(ns)
    meta, bank = qa.runtime_bank(ns, meta, bank)
    return ns, meta, bank, prepared


def source_identity(ns, meta, layers):
    return {"bank_fingerprint": meta["fingerprint"], "bank_json_sha256": digest(ns.run_dir / "bank.json"),
            "bank_npz_sha256": digest(ns.run_dir / "bank.npz"),
            "questions_sha256": digest(ns.run_dir / "questions.json"),
            "training_json_sha256": digest(ns.run_dir / "training.json"),
            "checkpoints": {str(l): digest(qa.checkpoint_path(ns.run_dir, l, meta["config"]["seed"])) for l in layers}}


def chosen_layers(args, ns, meta):
    if args.layers is not None:
        layers = list(args.layers)
    else:
        frozen = qa.selection(ns, meta)
        candidate = frozen["selected"]["qa_das"]
        if candidate is None:
            raise ValueError("source run has no frozen DAS layer; provide one explicit pilot --layers value")
        layers = [candidate["layer"]]
    if not layers or len(set(layers)) != len(layers) or any(l not in meta["das_candidate_layers"] for l in layers):
        raise ValueError("layers must be unique layers with verified source DAS checkpoints")
    return sorted(layers)


def train(args, ns, meta, bank, prepared, bundle=None):
    if args.alpha is None:
        args.alpha = 1.0
    if args.init is None:
        args.init = "das"
    include_eos = getattr(args, "include_eos", None)
    include_eos = False if include_eos is None else include_eos
    requested_stop = getattr(args, "stop_token_id", None)
    if type(include_eos) is not bool or (requested_stop is not None and (type(requested_stop) is not int or requested_stop < 0)):
        raise ValueError("include_eos must be boolean and stop_token_id a nonnegative integer")
    if not include_eos and requested_stop is not None:
        raise ValueError("--stop-token-id requires --include-eos for training")
    layers = chosen_layers(args, ns, meta)
    records = prepared["e_train"]
    if args.pilot is not None:
        if args.pilot < 1 or args.pilot > len(records):
            raise ValueError("pilot count must fit the source estimation cohort")
        records = records[:args.pilot]
    source = source_identity(ns, meta, layers)
    if (args.run_dir / "adapter.pt").exists() or (args.run_dir / "adapter.json").exists():
        adapter = load_adapter(args.run_dir, expected_model=meta["model"], expected_alpha=args.alpha)
        saved_eos, saved_stop = stop_training_config(adapter)
        expected_stop = (requested_stop if requested_stop is not None else saved_stop) if include_eos else None
        expected = {"epochs": args.epochs, "lr": args.lr, "clip": args.clip, "seed": args.seed,
                    "ranking_weight": args.ranking_weight, "optimizer": "Adam", "alpha": args.alpha,
                    "include_eos": include_eos, "stop_token_id": expected_stop}
        # Compatibility is in-memory only: do not rewrite existing artifacts to add the new keys.
        saved_config = {**adapter["train_config"], "include_eos": saved_eos, "stop_token_id": saved_stop}
        if (adapter["source"] != source or adapter["layers"] != layers or adapter["init"] != args.init
                or saved_config != expected or adapter["training_pair_ids"] != [r["pair_id"] for r in records]):
            raise ValueError("existing adapter belongs to another training request; use a new run directory")
        print("verified existing adapter; training not repeated")
        return adapter
    if bundle is None:
        bundle = qa.open_bundle(meta, prepared)
    else:
        qa.verify_bundle(bundle, meta, prepared)
    initial = {l: bank["E_qa_%s_L%d" % ("md" if args.init == "md" else "das", l)] for l in layers}
    centers = {l: {"prompt": bank["prompt_center_L%d" % l], "decode": bank["decode_center_L%d" % l]} for l in layers}
    adapter = train_generation_adapter(
        bundle, records, meta["config"], initial, centers, alpha=args.alpha, init=args.init,
        epochs=args.epochs, lr=args.lr, clip=args.clip, ranking_weight=args.ranking_weight, seed=args.seed,
        include_eos=include_eos, stop_token_id=requested_stop,
        identity={"source": source, "pilot": bool(meta["pilot"] or args.pilot is not None),
                  "source_run": str(ns.run_dir.resolve()), "source_protocol": meta["protocol"]})
    save_adapter(adapter, args.run_dir)
    return adapter


def verified_adapter(args, ns, meta):
    adapter = load_adapter(args.run_dir, expected_model=meta["model"], expected_alpha=args.alpha)
    include_eos, stop = stop_training_config(adapter)
    requested_eos, requested_stop = getattr(args, "include_eos", None), getattr(args, "stop_token_id", None)
    if requested_eos is not None and (type(requested_eos) is not bool or requested_eos != include_eos):
        raise ValueError("include_eos differs from the saved training request")
    if requested_stop is not None and (type(requested_stop) is not int or requested_stop != stop):
        raise ValueError("stop_token_id differs from the saved training request")
    if args.layers is not None and sorted(args.layers) != adapter["layers"]:
        raise ValueError("deployment layers must equal the trained layers")
    if args.init is not None and args.init != adapter["init"]:
        raise ValueError("deployment initialization label differs from the saved adapter")
    if adapter["source"] != source_identity(ns, meta, adapter["layers"]):
        raise ValueError("source estimation/training artifacts changed since adapter training")
    if adapter["config"] != {k: meta["config"][k] for k in adapter["config"]}:
        raise ValueError("adapter generation prompt or budget differs from its source run")
    if adapter["model_config"] != meta["model_config"]:
        raise ValueError("source and adapter model settings differ")
    return adapter


def evaluation_meta(args, meta, adapter):
    identity = fingerprint([digest(args.run_dir / "adapter.pt"), adapter["source"], adapter["config"]])
    return {**meta, "protocol_fingerprint": identity, "training_sha256": digest(args.run_dir / "adapter.pt")}


def evaluation_choices(adapter):
    return [dict(qa.BASELINE),
            {"name": "original_das", "kind": "E", "context": "qa_das", "layers": adapter["layers"],
             "alpha": adapter["alpha"], "mode": "continuous", "uses_reference_answer": False, "eligible_for_selection": False},
            {"name": adapter["method"], "kind": "E", "context": adapter["method"], "layers": adapter["layers"],
             "alpha": adapter["alpha"], "mode": "continuous", "uses_reference_answer": False, "eligible_for_selection": False}]


def freeze_payload(args, meta, prepared, adapter, quality_rule):
    """Validate complete tuning grades and bind every test configuration byte.

    This freezes one declared adapter and its two controls, not a search over
    adapters. A timestamp/hash cannot prove that the dataset was never seen.
    """
    if adapter.get('pilot') or meta.get('pilot'):
        raise ValueError('pilot training/source banks cannot be frozen for formal test generation')
    if quality_rule not in ("R0", "R1", "R2"):
        raise ValueError("freeze requires --primary-quality-rule R0, R1 or R2 declared before test generation")
    raw, summary, scored = [qa.read(args.run_dir / name) for name in
                            ("tune.json", "tune_scored.json", "tune_scored_items.json")]
    choices = evaluation_choices(adapter)
    ids = [q["pair_id"] for q in prepared["splits"]["tune"]]
    emeta = evaluation_meta(args, meta, adapter)
    expected = fingerprint([emeta["protocol_fingerprint"], choices, "tune", ids, False])
    if (raw.get("completed") is not True or raw.get("protocol_fingerprint") != expected
            or raw.get("condition_specs") != choices or raw.get("model") != meta["model"]
            or raw.get("split") != "tune" or raw.get("wiring_pilot") or raw.get("diagnostic_only")
            or [q["pair_id"] for q in raw["evaluations"]] != ids):
        raise ValueError("freeze needs complete matching tuning generation, not a check or partial cohort")
    qa.verify_question_rows(raw["evaluations"], prepared["splits"]["tune"])
    if (summary.get("completed") is not True or scored.get("completed") is not True
            or scored.get("grading_completed") is not True
            or summary.get("source_generation_sha256") != digest(args.run_dir / "tune.json")
            or scored.get("grading_source_generation_sha256") != digest(args.run_dir / "tune.json")
            or summary.get("judge_rubric_id") != qa.QA_GRADER_VERSION
            or scored.get("judge_rubric_id") != qa.QA_GRADER_VERSION
            or scored.get("judge_summary") != summary
            or scored.get("protocol_fingerprint") != expected
            or scored.get("condition_specs") != choices
            or scored.get("conditions") != [c["name"] for c in choices]):
        raise ValueError("freeze needs completed, matching semantic grades with the current rubric")
    qa.verify_question_rows(scored["evaluations"], prepared["splits"]["tune"])
    for before, after in zip(raw["evaluations"], scored["evaluations"]):
        if after["outputs"] != before["outputs"] or after["finished"] != before["finished"]:
            raise ValueError("tuning answers or completion flags changed during grading")
        refs, refusal_only = qa.reference_answers(before["gold"])
        if after.get("gold_is_abstention") != refusal_only or after.get("judge_reference_answers") != refs:
            raise ValueError("tuning reference partition changed during grading")
        for c in choices:
            grade = after["judge"].get(c["name"], {})
            label = after["outcome"].get(c["name"])
            if (label not in ("correct", "wrong", "abstain")
                    or grade.get("rubric_id") != qa.QA_GRADER_VERSION
                    or qa.judged_outcome(grade.get("label")) != label):
                raise ValueError("unresolved or inconsistent tuning grade")
    return {"protocol": "generation_adapter_pretest_freeze_v1", "model": meta["model"],
            "source": adapter["source"], "training": training_description(adapter),
            "condition_specs": choices, "primary_quality_rule": quality_rule,
            "tuning_pair_ids": ids, "test_pair_ids": [q["pair_id"] for q in prepared["splits"]["test"]],
            "files": {name: digest(args.run_dir / name) for name in
                      ("adapter.pt", "adapter.json", "tune.json", "tune_scored.json", "tune_scored_items.json")},
            "execution_implementation": implementation_identity(),
            "selection_scope": "one declared adapter plus original model and original DAS; all three reported",
            "artifact_freeze_only_no_claim_of_unseen_dataset": True}


def frozen_test_plan(args, meta, prepared, adapter):
    path = args.run_dir / "selection.json"
    if not path.exists():
        raise ValueError("test generation requires freeze after complete tuning grades; historical tests cannot be retroactively frozen")
    stored = qa.read(path)
    payload = freeze_payload(args, meta, prepared, adapter, stored.get("primary_quality_rule"))
    if stored != {**payload, "fingerprint": fingerprint(payload)}:
        raise ValueError("frozen test configuration, source code, adapter or tuning artifacts changed")
    return stored


def freeze(args, meta, prepared, adapter):
    if (args.run_dir / "selection.json").exists():
        return frozen_test_plan(args, meta, prepared, adapter)
    if any(args.run_dir.glob("test*")):
        raise ValueError("test artifacts already exist; cannot claim a pre-test freeze retrospectively")
    payload = freeze_payload(args, meta, prepared, adapter, getattr(args, "primary_quality_rule", None))
    result = {**payload, "fingerprint": fingerprint(payload)}
    qa.save(args.run_dir / "selection.json", result)
    return result


def generate(args, ns, meta, bank, prepared, adapter, split, *, check=False, bundle=None):
    plan = None
    if split == "test":
        if check or args.limit is not None:
            raise ValueError("test generation must use the frozen full cohort; use check on tuning questions for pilots")
        plan = frozen_test_plan(args, meta, prepared, adapter)
    questions = prepared["splits"][split]
    limit = 3 if check else args.limit
    if limit is not None:
        if limit < 1 or limit > len(questions):
            raise ValueError("generation limit must fit the requested split")
        questions = questions[:limit]
    emeta = evaluation_meta(args, meta, adapter)
    filename = "check.json" if check else split + ".json"
    path = args.run_dir / filename
    choices = evaluation_choices(adapter)
    signature = fingerprint([emeta["protocol_fingerprint"], choices, split, [q["pair_id"] for q in questions], bool(check)])
    if path.exists():
        data = qa.read(path)
        if (data.get("protocol_fingerprint") != signature or data.get("condition_specs") != choices
                or data.get("model") != meta["model"] or data.get("bank_protocol_fingerprint") != emeta["protocol_fingerprint"]):
            raise ValueError("generation resume does not match adapter/conditions/cohort")
        if data.get("diagnostic_only", False) or data.get("wiring_pilot", False) != bool(check):
            raise ValueError("generation resume has a legacy pilot/reference label; preserve it and use a new artifact")
        if plan is not None and data.get("pretest_freeze_fingerprint") != plan["fingerprint"]:
            raise ValueError("test resume has no matching pre-test freeze; historical files remain unchanged")
        qa.verify_question_rows(data["evaluations"], questions)
    else:
        data = {"model": meta["model"], "protocol": PROTOCOL, "protocol_fingerprint": signature,
                "bank_protocol_fingerprint": emeta["protocol_fingerprint"], "training_sha256": emeta["training_sha256"],
                "questions_sha256": meta["questions_sha256"], "split": split,
                "pilot": bool(adapter["pilot"] or limit is not None), "wiring_pilot": bool(check),
                "diagnostic_only": False,
                "precision": meta["precision"], "prompt": adapter["config"],
                "intervention_position": "prompt-end predictor once; every later answer predictor with the decode center; no reference answer at inference",
                "E": {"method": adapter["method"], "init": adapter["init"], "operator": adapter["operator"]},
                "training": training_description(adapter),
                "generation_implementation": implementation_identity(),
                "C": {"enabled": False}, "source": adapter["source"],
                "conditions": [c["name"] for c in choices], "condition_specs": choices, "audit": [],
                "evaluations": [{**q, "idx": i, "outputs": {}, "finished": {}, "rule_abstain": {}} for i, q in enumerate(questions)]}
        if plan is not None:
            data.update(pretest_freeze_fingerprint=plan["fingerprint"], primary_quality_rule=plan["primary_quality_rule"])
    if bundle is None:
        bundle = qa.open_bundle(meta, prepared)
    else:
        qa.verify_bundle(bundle, meta, prepared)
    include_eos, stop = stop_training_config(adapter)
    if include_eos:
        resolve_stop_token_id(bundle, stop)  # The supervised ID must still terminate this model's generation.
    original = copy.deepcopy(adapter)
    original["W"] = {str(l): torch.as_tensor(bank["E_qa_das_L%d" % l]).float() for l in adapter["layers"]}
    cfg = {**meta["config"], **adapter["config"]}
    eos = getattr(bundle.model.generation_config, "eos_token_id", None)
    eos = set(eos if isinstance(eos, (tuple, list)) else [eos if eos is not None else bundle.tokenizer.eos_token_id])
    with fp32_output_projection(bundle):
        for c in choices:
            name = c["name"]
            for start in range(0, len(questions), cfg["batch_size"]):
                rows = data["evaluations"][start:start + cfg["batch_size"]]
                if all(name in r["outputs"] for r in rows):
                    continue
                texts = [legacy.qa_text(bundle, cfg, q["question"]) for q in rows]
                inputs = bundle.encode(texts)
                if not inputs["attention_mask"][:, -1].all():
                    raise ValueError("generation requires left-padded prompts")
                specs = ([] if name == "baseline" else adapter_specs(
                    original if name == "original_das" else adapter, expected_alpha=adapter["alpha"], name=name))
                plen = inputs["input_ids"].shape[1]
                output, hooks, displacement = qa.generate_with_audit(bundle, inputs, specs, cfg, trace_displacement=check)
                n_new = output.shape[1] - plen
                if any(s.n_applied != n_new for s in specs) or any(h.first_shape[1] != plen or h.call_count != n_new for h in hooks.values()):
                    raise RuntimeError("adapter generation write-count/position audit failed")
                for row, tokens in zip(rows, output[:, plen:]):
                    answer = bundle.tokenizer.decode(tokens, skip_special_tokens=True)
                    if name in row["outputs"] and row["outputs"][name] != answer:
                        raise RuntimeError("generation resume changed an existing answer")
                    row["outputs"][name] = answer
                    row["finished"][name] = any(int(t) in eos for t in tokens)
                    row["rule_abstain"][name] = bool(rule_abstains(answer))
                    if check:
                        print("PILOT", row["pair_id"], name, repr(answer))
                data["audit"].append({"condition": name, "batch_start": start, "prompt_len": plen, "new_tokens": n_new,
                                      "edits": {s.name: s.n_applied for s in specs}, "displacement": displacement})
                qa.save(path, data)
    data["completed"] = True
    qa.save(path, data)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("train", "check", "generate-tune", "freeze", "generate-test", "judge-check", "judge-tune", "judge-test"))
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--layers", nargs="+", type=int)
    p.add_argument("--init", choices=("das", "md", "random"), default=None,
                   help="training default: das; evaluation reads the artifact and rejects an explicit mismatch")
    p.add_argument("--alpha", type=float, default=None,
                   help="training default: 1; evaluation reads the trained alpha and rejects an explicit mismatch")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ranking-weight", type=float, default=0.0)
    p.add_argument("--include-eos", action=argparse.BooleanOptionalAction, default=None,
                   help="training default: false; supervise one declared stop token after each answer; evaluation checks a supplied flag")
    p.add_argument("--stop-token-id", type=int, default=None,
                   help="explicit generation stop ID for --include-eos; default uses tokenizer.eos_token_id only if valid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pilot", type=int, help="training pairs for an explicitly labeled wiring pilot")
    p.add_argument("--limit", type=int, help="generation questions for an explicitly labeled pilot")
    p.add_argument("--judge-exp", default=DEFAULT_JUDGE_EXPERIMENT)
    p.add_argument("--primary-quality-rule", choices=("R0", "R1", "R2"),
                   help="required at freeze: R0 original grades; R1 malformed Unicode correct->wrong; R2 also unfinished correct->wrong")
    args = p.parse_args()
    if args.phase.startswith("judge-"):
        validate_judge_experiment(args.judge_exp)
    ns, meta, bank, prepared = source_run(args.source_run)
    args.model = meta["model"]
    if args.phase == "train":
        train(args, ns, meta, bank, prepared)
        return
    adapter = verified_adapter(args, ns, meta)
    if args.phase == "check":
        generate(args, ns, meta, bank, prepared, adapter, "tune", check=True)
    elif args.phase == "freeze":
        freeze(args, meta, prepared, adapter)
    elif args.phase.startswith("generate-"):
        generate(args, ns, meta, bank, prepared, adapter, args.phase.split("-", 1)[1])
    else:
        split = args.phase.split("-", 1)[1]
        qa.judge(args, evaluation_meta(args, meta, adapter), split + ".json", split + "_scored")


if __name__ == "__main__":
    main()
