#!/usr/bin/env python3
"""Export E43 generations (frozen prompt controls + clamped arms) into the judge2 generation format.

Output: RUN_DIR/judge_<split>/<name>.json with
  completed, model, conditions, evaluations[{pair_id, question, gold, incorrect, best_answer,
  outputs{condition: text}, finished{condition: bool}, rule_abstain{condition: bool}}]
so that scripts/review_checks/judge_two_axis_run.py cells / allenai / llm --judge pro / single can run on it.
Conditions: baseline_<prompt> for every frozen prompt in frozen_<split>.json, and <family>_L<layer>_<arm> for every
selected gen file. Gold/incorrect reference lists come from the source run's questions.json (judge references
only; they never enter any prompt here). Identical (question, answer) pairs are judged once by the judge cache.
"""
import argparse
import json
from pathlib import Path

import common  # noqa: F401  (puts the code root on sys.path)
from core.readout import rule_abstains  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--split", default="tune", choices=("train", "tune"))
    p.add_argument("--gen", nargs="*", default=None, help="gen file names to include (default: all complete files of the split)")
    p.add_argument("--name", default="judge_input")
    a = p.parse_args()
    prepared = read(a.source_run / "questions.json")
    refs = {q["pair_id"]: q for q in prepared["splits"][a.split]}
    frozen = read(a.run_dir / ("frozen_%s.json" % a.split))
    if not frozen.get("completed"):
        raise SystemExit("frozen_%s.json is not complete" % a.split)
    if frozen.get("pilot"):
        raise SystemExit("pilot freezes are never judged")
    gen_dir = a.run_dir / "gen"
    files = [gen_dir / n for n in a.gen] if a.gen is not None else sorted(gen_dir.glob("*_%s.json" % a.split))
    gens = []
    for f in files:
        d = read(f)
        if d["split"] != a.split:
            raise SystemExit("%s is not a %s generation" % (f.name, a.split))
        if not d.get("completed") or d.get("pilot"):
            raise SystemExit("%s is a pilot / incomplete generation; exclude it" % f.name)
        if d.get("identity_sha256") != frozen.get("identity_sha256"):
            raise SystemExit("%s was generated under a different identity than frozen_%s.json" % (f.name, a.split))
        gens.append(d)
    conditions = ["baseline_%s" % n for n in frozen["questions"][0]["answers"]]
    for d in gens:
        conditions.append("%s_L%d_%s" % (d["family"], d["layer"], d["arm"]))
    if len(set(conditions)) != len(conditions):
        raise SystemExit("duplicate conditions")
    rows = []
    for fq in frozen["questions"]:
        pid = fq["pair_id"]
        q = refs[pid]
        outputs, finished, rule = {}, {}, {}
        for name, rec in fq["answers"].items():
            cond = "baseline_%s" % name
            outputs[cond], finished[cond], rule[cond] = rec["text"], bool(rec["finished"]), bool(rule_abstains(rec["text"]))
        for d in gens:
            cond = "%s_L%d_%s" % (d["family"], d["layer"], d["arm"])
            match = [r for r in d["rows"] if r["pair_id"] == pid]
            if len(match) != 1:
                raise SystemExit("%s has %d rows for %s" % (cond, len(match), pid))
            r = match[0]
            outputs[cond], finished[cond], rule[cond] = r["text"], bool(r["finished"]), bool(rule_abstains(r["text"]))
        rows.append({"pair_id": pid, "question": q["question"], "gold": list(q["gold"]), "incorrect": list(q.get("incorrect") or []),
                     "best_answer": q.get("best_answer"), "outputs": outputs, "finished": finished, "rule_abstain": rule})
    out = {"completed": True, "model": frozen["model"], "protocol": frozen["protocol"], "split": a.split,
           "identity_sha256": frozen.get("identity_sha256"),
           "conditions": conditions, "sources": {"frozen": "frozen_%s.json" % a.split, "gen": [f.name for f in files]},
           "evaluations": rows}
    out_dir = a.run_dir / ("judge_%s" % a.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (a.name + ".json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"path": str(path), "n_questions": len(rows), "conditions": conditions}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
