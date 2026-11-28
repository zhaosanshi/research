"""Exploration (executor hai, Claude Opus 5.5, 2026-11-28): the FalseQA confirmation bank of queue Q
(discussion/work_queue/2026-11-28_queue_q.md section 1, "questions"). Not reviewed by GPT.

Every valid pair of the official test.csv (falseqa_prep.load_pairs: pair = data row i of the label-1 block with data
row i + n of the label-0 block; pairs whose two questions have the same normalised text are left out) that is NOT one of
the 300 held-out pairs of the existing id file. Rows in the format of the 700 bank (falseqa_prep.bank_rows, unchanged),
field "set" = "confirm". The first stage: random.Random(SEED).sample(sorted pair ids, N_STAGE), sorted.

Checks (any failure stops before anything is written): the test.csv sha256 is the one the existing id file records;
the held-out pairs of the id file are pairs of the pool; every pair has its false and its valid row; ids unique; no
question text repeated inside the new bank; no question of the new bank has the normalised text of a question of the
existing bank OLD_BANK; the row keys equal those of the existing bank.

The csv is downloaded data: run with python -I, paths as arguments. The bank (question text) stays on the server; the
id file (ids and counts only) may go to the public repository.

usage: python -I falseqa_confirm_bank_1128.py TEST_CSV OLD_IDS_JSON OLD_BANK_JSONL OUT_BANK_JSONL OUT_IDS_JSON SEED N_STAGE
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import falseqa_prep as P  # noqa: E402


def main():
    if len(sys.argv) != 8:
        raise SystemExit(__doc__)
    test_csv, old_ids, old_bank, out_bank, out_ids = sys.argv[1:6]
    seed, n_stage = int(sys.argv[6]), int(sys.argv[7])
    for p in (out_bank, out_ids):
        if os.path.exists(p):
            raise SystemExit("%s exists" % p)
    idf = json.load(open(old_ids, encoding="utf-8"))
    if idf["source_files"]["test.csv"]["sha256"] != P.sha_file(test_csv):
        raise SystemExit("test.csv is not the file the existing id file records")
    held = {e["pair_id"] for e in idf["heldout"]}
    if len(held) != 300:
        raise SystemExit("the existing id file has %d held-out pairs, want 300" % len(held))
    pairs, same = P.load_pairs(test_csv, "test")
    if same != idf["left_out_same_text_pairs"]["test"]:
        raise SystemExit("same-text pairs %s differ from the existing id file %s" % (same, idf["left_out_same_text_pairs"]["test"]))
    pool = {p["pair_id"] for p in pairs}
    if not held <= pool:
        raise SystemExit("held-out pairs outside the pool: %s" % sorted(held - pool)[:5])
    picked = sorted([p for p in pairs if p["pair_id"] not in held], key=lambda p: p["pair_id"])
    rows = P.bank_rows(picked, "test", "confirm")
    with open(old_bank, encoding="utf-8") as f:
        old = [json.loads(l) for l in f if l.strip()]
    bad = []
    per = collections.defaultdict(list)
    for r in rows:
        per[r["pair_id"]].append(r["premise"])
    if any(sorted(v) != ["false", "valid"] for v in per.values()) or len(per) != len(picked):
        bad.append("a pair without exactly one false and one valid row")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or len(rows) != 2 * len(picked):
        bad.append("ids not unique or row count not twice the pairs")
    mine = [P.norm(r["question"]) for r in rows]
    if len(set(mine)) != len(mine):
        bad.append("a question text repeated inside the new bank")
    oldt = {P.norm(r["question"]) for r in old}
    shared = sorted(r["id"] for r in rows if P.norm(r["question"]) in oldt)
    if shared:
        bad.append("%d questions share the normalised text of a question of the existing bank: %s" % (len(shared), shared[:10]))
    if set(ids) & {r["id"] for r in old}:
        bad.append("an id of the existing bank")
    if any(set(r) != set(old[0]) for r in rows):
        bad.append("row keys differ from the existing bank")
    if bad:
        raise SystemExit("checks failed: " + "; ".join(bad))
    stage = sorted(random.Random(seed).sample(sorted(per), n_stage))
    P.write_jsonl(out_bank, rows)
    prem = {r["id"]: r["premise"] for r in rows}
    out = {"dataset": "FalseQA (Hu et al., ACL 2023), github.com/thunlp/FalseQA at commit %s, file dataset/test.csv" % P.COMMIT,
           "plan": "discussion/work_queue/2026-11-28_queue_q.md section 1",
           "rule": "every valid pair of test.csv (falseqa_prep.load_pairs; same-text pairs left out) not among the 300 "
                   "held-out pairs of falseqa_ids_20261123.json",
           "source_files": {"test.csv": {"sha256": P.sha_file(test_csv), "pairs": P.WANT_PAIRS["test"]}},
           "excluded": {"heldout_pairs": len(held), "same_text_pairs": same},
           "n_pairs": len(picked), "n_questions": len(rows),
           "premise_counts": dict(collections.Counter(prem.values())),
           "shared_text_with_700_bank": len(shared),
           "bank": os.path.basename(out_bank), "bank_sha256": P.sha_file(out_bank),
           "pairs": [{"pair_id": p["pair_id"], "ids": {"false": p["pair_id"] + "f", "valid": p["pair_id"] + "v"}} for p in picked],
           "stage1": {"seed": seed, "draw": "sorted(random.Random(seed).sample(sorted(pair ids), %d))" % n_stage,
                      "pairs": stage, "ids": [q for q in ids if q[:-1] in set(stage)]}}
    with open(out_ids, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("pool %d pairs (same-text left out %s); held-out excluded %d; confirm bank %d pairs, %d questions %s; "
          "shared normalised text with the 700 bank %d; stage 1 %d pairs %d questions; bank sha256 %s" % (
              len(pairs), same, len(held), len(picked), len(rows), dict(collections.Counter(prem.values())), len(shared),
              len(stage), len(out["stage1"]["ids"]), out["bank_sha256"][:16]))


if __name__ == "__main__":
    main()
