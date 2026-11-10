#!/usr/bin/env python3
"""Export verified E44 artifacts as one immutable judge input (same layout as e43_export_judge.py, so
judge_two_axis_run.py cells / allenai / llm / single read it unchanged).

Conditions: baseline_<prompt> for the selected frozen prompts (default: all of them) plus one condition per selected
generation file. References never enter generation here. Pilot or incomplete files are refused. An existing output
is reusable only when its full content is identical.
"""
import argparse
import json
from pathlib import Path

import common  # noqa: F401
import e43_qa_prompt_das as e43
import e44_factorial_das as E
from core.io_utils import atomic_json_dump
from core.readout import rule_abstains

EXPORT_PROTOCOL = "e44_judge_generation_export_v1"


def export(source_run, run_dir, split="tune", gen_names=None, prompt_names=None, name="judge_input"):
    if split not in ("train", "tune"):
        raise ValueError("only train/tune splits may be exported")
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("output name must be a simple file stem")
    source_run, run_dir = Path(source_run).resolve(), Path(run_dir).resolve()
    gen_dir = (run_dir / "gen").resolve()
    if gen_names is None:
        files = [p for p in sorted(gen_dir.glob("*_%s.json" % split)) if ".pilot." not in p.name]
    else:
        files = []
        for value in gen_names:
            if not isinstance(value, str) or Path(value).name != value:
                raise ValueError("generation names must be files inside this run's gen directory")
            files.append(gen_dir / value)
    if len(set(files)) != len(files) or any(not p.is_file() for p in files):
        raise ValueError("generation files must exist and be selected once")
    frozen_path = run_dir / ("frozen_%s.json" % split)
    before = {str(p): e43.file_sha(p) for p in [source_run / "bank.json", source_run / "questions.json", frozen_path, *files]}
    questions = e43.read(source_run / "questions.json")["splits"][split]
    frozen = e43.read(frozen_path)
    E.validate_frozen(frozen, questions, frozen["prompt_systems"], require_full=True)
    if frozen["identity"].get("source_files_sha256") != {k: before[str(source_run / k)] for k in ("bank.json", "questions.json")}:
        raise ValueError("frozen answers use different source questions or bank")
    prompts = list(frozen["prompt_systems"]) if prompt_names is None else list(prompt_names)
    if len(set(prompts)) != len(prompts) or any(p not in frozen["prompt_systems"] for p in prompts):
        raise ValueError("unknown or repeated prompt name")
    generations = []
    for path in files:
        d = e43.validate_artifact(e43.read(path), "generation")
        if d.get("completed") is not True or d.get("pilot") is not False or d.get("phase") != "generate" or d.get("split") != split:
            raise ValueError("pilot/incomplete/foreign generations cannot enter a judge export: " + path.name)
        if d["identity_sha256"] != frozen["identity_sha256"] or d.get("frozen_record_sha256") != frozen["artifact_sha256"]:
            raise ValueError("generation and frozen answers come from different identities: " + path.name)
        if d.get("question_ids") != [q["pair_id"] for q in questions]:
            raise ValueError("generation cohort/order differs: " + path.name)
        direction = run_dir / "dirs" / d["direction_file"]
        before[str(direction)] = e43.file_sha(direction)
        if before[str(direction)] != d["direction_file_sha256"]:
            raise ValueError("direction file changed after generation: " + path.name)
        generations.append(d)
    conditions = ["baseline_" + p for p in prompts]
    specs = [{"name": "baseline_" + p, "base_condition": "baseline_" + p, "kind": "prompt_control",
              "uses_reference_answer": False} for p in prompts]
    lookups = []
    for d in generations:
        condition = "%s_L%d_%s" % (d["family"], d["layer"], E.arm_file_token(d["arm"]))
        if d["base_prompt"] not in prompts:
            raise ValueError("the base prompt of %s is not among the exported prompt controls" % condition)
        conditions.append(condition)
        specs.append({"name": condition, "base_condition": "baseline_" + d["base_prompt"], "family": d["family"],
                      "layer": d["layer"], "arm": d["arm"], "direction": d["direction"], "beta": d["beta"],
                      "kind": "intervention", "uses_reference_answer": False})
        lookups.append((condition, d["base_prompt"], {row["pair_id"]: row for row in d["rows"]}))
    if len(set(conditions)) != len(conditions):
        raise ValueError("duplicate conditions")
    rows = []
    for question, fq in zip(questions, frozen["questions"]):
        pid = question["pair_id"]
        outputs, finished, rule = {}, {}, {}
        for p in prompts:
            rec = fq["answers"][p]
            outputs["baseline_" + p], finished["baseline_" + p], rule["baseline_" + p] = rec["text"], rec["finished"], bool(rule_abstains(rec["text"]))
        for condition, base, lookup in lookups:
            row = lookup[pid]
            if row.get("baseline_text") != fq["answers"][base]["text"]:
                raise ValueError("generation comparison baseline differs from the frozen answer: " + condition)
            outputs[condition], finished[condition], rule[condition] = row["text"], row["finished"], bool(rule_abstains(row["text"]))
        rows.append({"pair_id": pid, "question": question["question"], "gold": list(question["gold"]),
                     "incorrect": list(question.get("incorrect") or []), "best_answer": question.get("best_answer"),
                     "outputs": outputs, "finished": finished, "rule_abstain": rule})
    payload = {"completed": True, "model": frozen["model"], "protocol": frozen["protocol"], "export_protocol": EXPORT_PROTOCOL,
               "split": split, "identity": frozen["identity"], "identity_sha256": frozen["identity_sha256"],
               "conditions": conditions, "condition_specs": specs, "question_ids": [q["pair_id"] for q in questions],
               "sources": {"frozen": frozen_path.name, "gen": [p.name for p in files], "sha256": before},
               "evaluations": rows}
    from scripts.review_checks.judge_two_axis_run import _derive_cells    # the exact downstream schema check
    _derive_cells(payload)
    for path, digest in before.items():
        if e43.file_sha(path) != digest:
            raise RuntimeError("source artifact changed during judge export: " + path)
    out = run_dir / ("judge_%s" % split) / (name + ".json")
    if out.exists():
        if json.loads(out.read_text(encoding="utf-8")) != payload:
            raise ValueError("existing judge input differs; use a new --name, never overwrite judged inputs")
    else:
        atomic_json_dump(payload, str(out))
    return {"path": str(out), "n_questions": len(rows), "conditions": conditions, "export_protocol": EXPORT_PROTOCOL}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--split", default="tune", choices=("train", "tune"))
    p.add_argument("--gen", nargs="*", default=None, help="generation file names (default: every full file of the split; empty list: prompt controls only)")
    p.add_argument("--prompts", nargs="*", default=None, help="prompt controls to export (default: all frozen prompts)")
    p.add_argument("--name", default="judge_input")
    a = p.parse_args()
    print(json.dumps(export(a.source_run, a.run_dir, a.split, a.gen, a.prompts, a.name), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
