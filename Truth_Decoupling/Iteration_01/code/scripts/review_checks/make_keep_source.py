#!/usr/bin/env python3
"""Source directory for an E44 run whose TRAINING questions also hold multiple-choice items (Claude Fable 5.2,
2026-11-20; exploration: can the E direction be trained so that it leaves MMLU-style answers alone?).

Background (novelty/ours/2026-11-19_capability_preservation_design.md, section 8; 2026-11-20_per_model_strength_results.md,
section 7): the E write lowers MMLU accuracy although the honest sentence it was learned from does not. The training rows
of E ask one thing: on a prompt WITHOUT the sentence, the write must reproduce the output of the prompt WITH the sentence.
So far every training question is a TruthfulQA question. If multiple-choice questions are added to the training split,
the same rows ask the write to reproduce the sentence there too - where the sentence changes little. Nothing in the
driver changes: it trains on whatever splits.train holds.

This writes a new source directory:
  questions.json  the --like file with N multiple-choice items appended to splits.train (tune and test untouched);
                  an item is {pair_id = the list's own id "mmlu:<subject>:<row>", question = cap_prompts.mmlu_user(...),
                  gold = [letter], incorrect = the other letters, best_answer = letter}
  bank.json       the --like file plus a derived_from record
  keep_items.json the chosen item ids (ids only)
The N items are drawn with --seed from the 1,000-item diagnostic list (--ids), which the capability confirmation never
uses: the confirmation set is the other 13,042 test items.
The directory holds question text: it stays on the server.
usage (container, from the code copy's root):
  python3 scripts/review_checks/make_keep_source.py --like LIKE_SOURCE_DIR --mmlu PARQUET --ids IDS_JSON --n 160 \
      --seed 20261120 --cap-prompts DIR_OF_cap_prompts.py --out OUT_DIR
"""
import argparse
import hashlib
import json
import os
import random
import sys

LETTERS = "ABCD"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True)
    ap.add_argument("--mmlu", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cap-prompts", required=True, help="directory of cap_prompts.py (the evaluation's own formatter)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit("refusing to write into non-empty %s" % a.out)
    sys.path.insert(0, a.cap_prompts)
    import cap_prompts as CP
    import pyarrow.parquet as pq
    with open(os.path.join(a.like, "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    with open(os.path.join(a.like, "bank.json"), encoding="utf-8") as f:
        bank = json.load(f)
    with open(a.ids, encoding="utf-8") as f:
        idl = json.load(f)
    if idl.get("source_file_sha256") != sha(a.mmlu):
        raise SystemExit("the ID list was made from another file")
    ids = idl["ids"]
    if len(set(ids)) != len(ids) or not 0 < a.n <= len(ids):
        raise SystemExit("bad ID list or --n")
    chosen = set(random.Random(a.seed).sample(ids, a.n))
    chosen = [i for i in ids if i in chosen]                      # the list's own order
    per = {}
    for r in pq.read_table(a.mmlu).to_pylist():
        per.setdefault(r["subject"], []).append(r)
    train = list(questions["splits"]["train"])
    have = {q["pair_id"] for split in questions["splits"].values() for q in split}
    for i in chosen:
        _, subj, k = i.split(":")
        r = per[subj][int(k)]
        letter = LETTERS[r["answer"]]
        if i in have:
            raise SystemExit("item id %s collides with a question id" % i)
        train.append({"pair_id": i, "question": CP.mmlu_user(r["subject"], r["question"], r["choices"]), "gold": [letter],
                      "incorrect": [x for x in LETTERS if x != letter], "best_answer": letter})
    out_q = dict(questions)
    out_q["splits"] = dict(questions["splits"], train=train)
    if any(out_q["splits"][s] != questions["splits"][s] for s in questions["splits"] if s != "train"):
        raise SystemExit("a split other than train changed")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(out_q, f, ensure_ascii=False)
    bank = dict(bank)
    bank["derived_from"] = {"dir": os.path.abspath(a.like), "bank_sha256": sha(os.path.join(a.like, "bank.json")),
                            "questions_sha256": sha(os.path.join(a.like, "questions.json")),
                            "added_train_items": {"n": len(chosen), "seed": a.seed, "ids_file_sha256": sha(a.ids),
                                                  "mmlu_file_sha256": sha(a.mmlu), "formatter": "cap_prompts.mmlu_user",
                                                  "formatter_sha256": sha(os.path.join(a.cap_prompts, "cap_prompts.py"))},
                            "note": "splits.train = the --like training questions + multiple-choice items; tune and test unchanged"}
    tmp = os.path.join(a.out, ".bank.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(a.out, "bank.json"))
    with open(os.path.join(a.out, "keep_items.json"), "w", encoding="utf-8") as f:
        json.dump({"n": len(chosen), "seed": a.seed, "ids": chosen}, f)
    print("wrote %s: train %d = %d + %d multiple-choice items; tune %d, test %d unchanged; questions.json sha256 %s" % (
        a.out, len(train), len(questions["splits"]["train"]), len(chosen), len(out_q["splits"].get("tune", [])),
        len(out_q["splits"].get("test", [])), sha(os.path.join(a.out, "questions.json"))[:16]))


if __name__ == "__main__":
    main()
