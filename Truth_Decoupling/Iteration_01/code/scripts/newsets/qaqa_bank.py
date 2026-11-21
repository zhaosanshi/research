"""Exploration (Claude Fable 5.2, 2026-11-22): a pilot bank from the (QA)^2 evaluation set, in the bank format of
newsets_gen.py (id, question, gold, split, pro100), plus the fields this pilot reads by question type.

(QA)^2, Kim et al., ACL 2023: 570 evaluation questions scraped from search autocomplete, half with a questionable
assumption (230 false, 55 unverifiable), each with a reference abstractive answer. The file is downloaded data: run
this with python -I, give paths as arguments, keep the output on the server (question text does not go to the public
repository; the id list may).

The pilot takes N questions with a fixed seed, in proportion to the three types (N = 100: 50 / 40 / 10), in file order.
The question text is used as it is (lower case, no question mark). gold = [the abstractive answer]. All rows get
split "test" and pro100 true, so that the unchanged export of newsets_gen.py takes all of them.

Optional third input: JSON files of the TruthfulQA runs (any structure); every value under a key "question" is taken
as a TruthfulQA question, and the pilot questions are checked against them (same text after normalisation, or token
overlap of at least 0.8). The counts are printed; nothing is removed.

usage: python -I qaqa_bank.py EVAL_CSV OUT_BANK_JSONL OUT_IDS_JSON N SEED [TRUTHFULQA_JSON ...]
"""
import collections
import csv
import hashlib
import json
import random
import re
import sys

TYPES = ("no_failure", "failure", "unverifiable")
WANT = {"no_failure": 285, "failure": 230, "unverifiable": 55}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def questions_in(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "question" and isinstance(v, str):
                out.add(v)
            else:
                questions_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            questions_in(v, out)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    src, out_bank, out_ids, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    with open(src, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = collections.Counter(r["type_questionable_assumption"] for r in rows)
    idx = [r["idx"] for r in rows]
    if dict(counts) != WANT or len(set(idx)) != len(idx) or len({r["question"] for r in rows}) != len(rows):
        raise SystemExit("not the 570-question evaluation file this script expects: %s" % dict(counts))
    if any(not r["abstractive_answer"].strip() or not r["question"].strip() for r in rows):
        raise SystemExit("a row without question or reference answer")
    quota = {"no_failure": n // 2, "failure": round(n * WANT["failure"] / len(rows))}
    quota["unverifiable"] = n - quota["no_failure"] - quota["failure"]
    rng = random.Random(seed)
    pos = {r["idx"]: k for k, r in enumerate(rows)}
    pick = []
    for t in TYPES:
        pool = [r for r in rows if r["type_questionable_assumption"] == t]
        pick += rng.sample(pool, quota[t])
    pick.sort(key=lambda r: pos[r["idx"]])
    bank = [{"id": "e%s" % r["idx"], "question": r["question"], "gold": [r["abstractive_answer"]], "split": "test", "pro100": True,
             "qtype": r["type_questionable_assumption"], "can_change": r["assumption_status_can_change"] or None,
             "assumption": r["questionable_assumption"] or None} for r in pick]
    with open(out_bank, "x", encoding="utf-8") as f:
        for b in bank:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(out_ids, "x", encoding="utf-8") as f:
        json.dump({"dataset": "(QA)^2 evaluation set, Dec 2022 pre-publication release", "source_file_sha256": sha_file(src),
                   "n": n, "seed": seed, "quota": quota, "ids": [b["id"] for b in bank],
                   "types": {b["id"]: b["qtype"] for b in bank}, "bank_sha256": sha_file(out_bank)}, f, indent=1)
    print("bank: %d questions %s; can_change among the questionable ones: %s" % (
        len(bank), quota, dict(collections.Counter(b["can_change"] for b in bank if b["qtype"] != "no_failure"))))
    if len(sys.argv) > 6:
        tq = set()
        for p in sys.argv[6:]:
            with open(p, encoding="utf-8") as f:
                questions_in(json.load(f), tq)
        tqn = {norm(q): set(norm(q).split()) for q in tq}
        same = near = 0
        for b in bank:
            q = norm(b["question"])
            w = set(q.split())
            if q in tqn:
                same += 1
            elif any(len(w & v) / max(1, len(w | v)) >= 0.8 for v in tqn.values()):
                near += 1
        allq = {norm(r["question"]) for r in rows}
        print("TruthfulQA questions read: %d distinct | pilot questions with the same normalised text: %d, with token overlap >= 0.8: %d | all 570 with the same normalised text: %d" % (
            len(tqn), same, near, len(allq & set(tqn))))


if __name__ == "__main__":
    main()
