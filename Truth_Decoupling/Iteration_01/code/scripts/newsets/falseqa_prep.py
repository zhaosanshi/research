"""Exploration (Claude Opus 5.5, 2026-11-23): FalseQA banks for the new-question confirmation (queue L, sections 8 and
10.2), in the bank format of the (QA)^2 bank (qaqa_bank.py: id, question, gold, split, pro100, qtype, can_change,
assumption) plus the pair fields this set needs.

FalseQA, Hu et al., ACL 2023 (github.com/thunlp/FalseQA, commit 9a9729974d8c5cfbfc6929e5b830111c0ed7c0c7, files
dataset/test.csv and dataset/valid.csv; columns question, answer, label; label 1 = false premise). Each file is two
blocks of equal length, all label-1 rows first, then all label-0 rows; the paper (section 3) has annotators revise each
false-premise question into a true-premise one with minimal edits. There is no pair id in the files: the pair is data
row i of the label-1 block with data row i of the label-0 block (row i and row i + n). Evidence, checked on the files
on 2026-11-23: the aligned rows share far more words than rows shifted by one; the aligned label-0 row is the nearest
neighbour of the label-1 row for about 95% of rows; every question text that occurs twice in a file sits exactly at
rows i and i + n. Pairs whose two questions have the same normalised text are left out of the draw (one text cannot
carry both labels); their numbers are written to the id file.

The files are downloaded data: run this with python -I, give paths as arguments, keep the banks on the server
(question text does not go to the public repository; the id file may).

Draw: whole pairs, one random.Random(SEED); first N_TEST pairs from the test file (held-out set), then N_DEV pairs from
the valid file (development set). Within each set the rows follow the source file order. gold = the answer column: a
list literal in the test file's label-1 rows (two rebuttals per question) is read with ast.literal_eval, every other
answer is one string. qtype = "failure" (label 1) or "no_failure" (label 0), the (QA)^2 names; premise = "false" or
"valid". All rows get split "test" and pro100 true, as in the (QA)^2 bank, so that the unchanged export of
newsets_gen.py takes all of them; the field "set" says dev or heldout.

Optional further inputs: bank jsonl files of other sets (any rows with a "question" field); the drawn questions are
checked against them (same normalised text) and the counts are printed; nothing is removed.

usage: python -I falseqa_prep.py TEST_CSV VALID_CSV OUT_DIR TAG OUT_IDS_JSON N_TEST N_DEV SEED [OTHER_BANK_JSONL ...]
writes OUT_DIR/falseqa_{dev,heldout,full}_bank_TAG.jsonl (full = dev rows, then held-out rows) and OUT_IDS_JSON.
"""
import ast
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys

COMMIT = "9a9729974d8c5cfbfc6929e5b830111c0ed7c0c7"
WANT_PAIRS = {"test": 687, "valid": 491}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gold_of(answer, label, split):
    a = answer.strip()
    if a.startswith("["):
        if not (split == "test" and label == "1"):
            raise SystemExit("a list answer outside the test file's label-1 block")
        v = ast.literal_eval(a)
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            raise SystemExit("a list answer that is not a non-empty list of strings")
        return [x.strip() for x in v]
    if not a:
        raise SystemExit("an empty answer")
    return [a]


def load_pairs(path, split):
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    if header != ["question", "answer", "label"] or any(len(r) != 3 for r in rows):
        raise SystemExit("%s: not the three-column FalseQA file" % path)
    n = WANT_PAIRS[split]
    labels = [r[2] for r in rows]
    if len(rows) != 2 * n or labels != ["1"] * n + ["0"] * n:
        raise SystemExit("%s: expected %d label-1 rows followed by %d label-0 rows" % (path, n, n))
    if any(not r[0].strip() for r in rows):
        raise SystemExit("%s: a row without question" % path)
    letter = split[0]
    pairs, same = [], []
    for i in range(n):
        a, b = rows[i], rows[i + n]
        pid = "fq%s%04d" % (letter, i)
        if norm(a[0]) == norm(b[0]):
            same.append(pid)
            continue
        pairs.append({"pair_id": pid, "rows": [
            {"id": pid + "f", "question": a[0], "gold": gold_of(a[1], "1", split), "label": 1, "source_row": i},
            {"id": pid + "v", "question": b[0], "gold": gold_of(b[1], "0", split), "label": 0, "source_row": i + n}]})
    return pairs, same


def bank_rows(picked, split, set_name):
    out = []
    for p in picked:
        for r in p["rows"]:
            out.append({"id": r["id"], "question": r["question"], "gold": r["gold"], "split": "test", "pro100": True,
                        "qtype": "failure" if r["label"] == 1 else "no_failure", "can_change": None, "assumption": None,
                        "premise": "false" if r["label"] == 1 else "valid", "pair_id": p["pair_id"], "set": set_name,
                        "source_split": split, "source_row": r["source_row"]})
    out.sort(key=lambda r: r["source_row"])
    return out


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check(dev, held, n_dev, n_test):
    bad = []
    for name, rows, k in (("dev", dev, n_dev), ("heldout", held, n_test)):
        if len(rows) != 2 * k:
            bad.append("%s: %d questions, want %d" % (name, len(rows), 2 * k))
        lab = collections.Counter(r["premise"] for r in rows)
        if lab["false"] != k or lab["valid"] != k:
            bad.append("%s: labels %s" % (name, dict(lab)))
        per = collections.Counter(r["pair_id"] for r in rows)
        if any(c != 2 for c in per.values()):
            bad.append("%s: a pair without both members" % name)
        for pid in per:
            got = sorted(r["premise"] for r in rows if r["pair_id"] == pid)
            if got != ["false", "valid"]:
                bad.append("%s: pair %s has %s" % (name, pid, got))
    pd = {r["pair_id"] for r in dev}
    ph = {r["pair_id"] for r in held}
    if pd & ph:
        bad.append("a pair in both sets")
    ids = [r["id"] for r in dev + held]
    if len(set(ids)) != len(ids):
        bad.append("repeated id")
    td = {norm(r["question"]) for r in dev}
    th = {norm(r["question"]) for r in held}
    if td & th:
        bad.append("%d question texts shared between dev and heldout" % len(td & th))
    if len(td) != len(dev) or len(th) != len(held):
        bad.append("a question text repeated inside a set")
    return bad


def main():
    if len(sys.argv) < 9:
        raise SystemExit(__doc__)
    test_csv, valid_csv, out_dir, tag, out_ids = sys.argv[1:6]
    n_test, n_dev, seed = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
    if not re.fullmatch(r"[0-9A-Za-z_]+", tag):
        raise SystemExit("TAG: letters, digits and underscores only")
    tpairs, tsame = load_pairs(test_csv, "test")
    vpairs, vsame = load_pairs(valid_csv, "valid")
    rng = random.Random(seed)
    tpick = sorted(rng.sample(tpairs, n_test), key=lambda p: p["pair_id"])
    vpick = sorted(rng.sample(vpairs, n_dev), key=lambda p: p["pair_id"])
    held = bank_rows(tpick, "test", "heldout")
    dev = bank_rows(vpick, "valid", "dev")
    bad = check(dev, held, n_dev, n_test)
    if bad:
        raise SystemExit("checks failed: " + "; ".join(bad))
    paths = {k: os.path.join(out_dir, "falseqa_%s_bank_%s.jsonl" % (k, tag)) for k in ("dev", "heldout", "full")}
    for k, p in paths.items():
        if os.path.exists(p):
            raise SystemExit("%s exists" % p)
    write_jsonl(paths["dev"], dev)
    write_jsonl(paths["heldout"], held)
    write_jsonl(paths["full"], dev + held)

    def set_entry(rows):
        pairs = collections.OrderedDict()
        for r in sorted(rows, key=lambda r: (r["pair_id"], r["premise"])):
            pairs.setdefault(r["pair_id"], []).append(r)
        return [{"pair_id": pid, "ids": {m["premise"]: m["id"] for m in ms},
                 "source_rows": {m["premise"]: m["source_row"] for m in ms}} for pid, ms in pairs.items()]

    out = {"dataset": "FalseQA (Hu et al., ACL 2023), github.com/thunlp/FalseQA at commit %s" % COMMIT,
           "pairing": "data row i of the label-1 block with data row i + n of the label-0 block (no pair id in the files)",
           "source_files": {"test.csv": {"sha256": sha_file(test_csv), "pairs": WANT_PAIRS["test"]},
                            "valid.csv": {"sha256": sha_file(valid_csv), "pairs": WANT_PAIRS["valid"]}},
           "left_out_same_text_pairs": {"test": tsame, "valid": vsame},
           "seed": seed, "draw": "one random.Random(seed): sample(test pool, n_test), then sample(valid pool, n_dev)",
           "n_pairs": {"heldout": n_test, "dev": n_dev},
           "labels": {"fq<t|v>NNNNf": "false premise (source label 1, qtype failure)",
                      "fq<t|v>NNNNv": "valid premise (source label 0, qtype no_failure)"},
           "heldout": set_entry(held), "dev": set_entry(dev),
           "bank_sha256": {os.path.basename(p): sha_file(p) for p in paths.values()}}
    with open(out_ids, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("pairs in pool: test %d (left out %d same-text), valid %d (left out %d same-text)" % (
        len(tpairs), len(tsame), len(vpairs), len(vsame)))
    print("heldout: %d questions %s from %d pairs | dev: %d questions %s from %d pairs | checks passed" % (
        len(held), dict(collections.Counter(r["premise"] for r in held)), n_test,
        len(dev), dict(collections.Counter(r["premise"] for r in dev)), n_dev))
    print("gold lengths heldout %s, dev %s" % (dict(collections.Counter((r["premise"], len(r["gold"])) for r in held)),
                                              dict(collections.Counter((r["premise"], len(r["gold"])) for r in dev))))
    if len(sys.argv) > 9:
        mine = {norm(r["question"]) for r in dev + held}
        for p in sys.argv[9:]:
            other = set()
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        q = json.loads(line).get("question")
                        if isinstance(q, str):
                            other.add(norm(q))
            print("%s: %d questions, same normalised text as a drawn question: %d" % (os.path.basename(p), len(other), len(mine & other)))


if __name__ == "__main__":
    main()
