# -*- coding: utf-8 -*-
"""Exploration (Claude Opus 5.5, executor E, 2026-11-23), queue L section 11: one judge input of the FalseQA bank
(the unchanged export of newsets_gen.py over all 700 questions) split into the development or the held-out part, so
that the development questions can be judged first. Everything of the export is kept as it is; only "evaluations" and
"question_ids" are restricted to the questions whose bank row has set == SET (bank order), and a record "subset" says
from which export (name, sha256) and which set. The judge (judge_two_axis_run.py cells / llm / single) reads only
completed, model, conditions and evaluations.
Checks: the export is complete, of task falseqa, of this bank (sha256), over all bank questions in bank order; the set
has 100 (dev) or 600 (heldout) questions. An existing output must be the same JSON (never overwritten).
usage: python3 -I falseqa_judge_split.py EXPORT_JSON BANK_JSONL dev|heldout OUT_JSON
"""
import hashlib
import json
import os
import sys

WANT = {"dev": 100, "heldout": 600}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 5 or sys.argv[3] not in WANT:
        raise SystemExit(__doc__)
    exp, bank_p, which, out = sys.argv[1:]
    with open(bank_p, encoding="utf-8") as f:
        bank = [json.loads(l) for l in f if l.strip()]
    with open(exp, encoding="utf-8") as f:
        d = json.load(f)
    want_all = ["falseqa:%s" % r["id"] for r in bank]
    bad = []
    if d.get("completed") is not True or d.get("task") != "falseqa" or d.get("bank_sha256") != sha_file(bank_p):
        bad.append("not a complete falseqa export of this bank")
    if d.get("question_ids") != want_all or [e["pair_id"] for e in d.get("evaluations", [])] != want_all:
        bad.append("the export does not cover every bank question in bank order")
    if d.get("pro100_only"):
        bad.append("a pro100-only export")
    if bad:
        raise SystemExit("; ".join(bad))
    keep = {"falseqa:%s" % r["id"] for r in bank if r.get("set") == which}
    if len(keep) != WANT[which]:
        raise SystemExit("%s: %d questions, want %d" % (which, len(keep), WANT[which]))
    o = dict(d)
    o["evaluations"] = [e for e in d["evaluations"] if e["pair_id"] in keep]
    o["question_ids"] = [q for q in d["question_ids"] if q in keep]
    o["subset"] = {"set": which, "of": os.path.basename(exp), "of_sha256": sha_file(exp), "n_questions": len(o["question_ids"])}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            if json.load(f) != o:
                raise SystemExit("%s exists and differs: stop" % out)
        print("%s exists and equals the split (%d questions x %d conditions)" % (out, len(o["evaluations"]), len(o["conditions"])))
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)
    print("split %s: %d questions x %d conditions = %d cells -> %s" % (which, len(o["evaluations"]), len(o["conditions"]),
                                                                   len(o["evaluations"]) * len(o["conditions"]), out))


if __name__ == "__main__":
    main()
