"""Neutral question bank for the TriviaQA / PopQA pilot (work queue C1), built from the raw files and the split ID lists.

Output (server only, it contains question text): one JSON object per line,
  {"id", "split" (train|tune|test), "pro100" (bool), "question", "gold" (list of aliases, first = canonical),
   "meta": {...}}
TriviaQA: question = the question text of the first row of that question_id; gold = [answer.value] + answer.aliases
          (deduplicated, order kept); meta = {"type": answer.type}.
PopQA:    question = the question text; gold = the first alias of every merged row (row order), then all their
          possible_answers (deduplicated, order kept); meta = {"prop", "subj_id", "s_pop", "merged_ids"}.
The conversion into the pipeline's own bank format is a separate step (C1).

Usage: python3 -I newsets_bank.py {triviaqa|popqa} RAW_PATH SPLITS_JSON OUT_JSONL
       python3 -I newsets_bank.py --selftest
"""
import argparse
import csv
import json
import sys


def dedup(seq):
    out = []
    for x in seq:
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


def triviaqa_records(raw_path, splits):
    import pyarrow.parquet as pq
    t = pq.read_table(raw_path, columns=["question_id", "question", "answer"])
    first = {}
    for qid, q, a in zip(t.column("question_id").to_pylist(), t.column("question").to_pylist(), t.column("answer").to_pylist()):
        if qid not in first:
            first[qid] = (q, a)
    for name, rec in iter_split_ids(splits):
        q, a = first[rec]
        yield {"id": rec, "split": name, "question": q.strip(), "gold": dedup([a["value"]] + list(a["aliases"])),
               "meta": {"type": a["type"]}}


def popqa_records(raw_path, splits):
    with open(raw_path, newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    merged = splits.get("merged_row_ids", {})
    for name, rid in iter_split_ids(splits):
        ids = merged.get(rid, [rid])
        if ids[0] != rid:
            raise SystemExit("merged group of %s does not start with it" % rid)
        texts = set(rows[i]["question"] for i in ids)
        subjects = set(rows[i]["subj_id"] for i in ids)
        if len(texts) != 1 or len(subjects) != 1:
            raise SystemExit("merged rows of %s disagree on question or subject" % rid)
        lists = [json.loads(rows[i]["possible_answers"]) for i in ids]
        # every merged object first (its first alias, in row order), then the remaining aliases: the Pro judge sees the
        # first 10 references, and each legal object must be among them (GPT review 2026-11-19, question 6)
        gold = [l[0] for l in lists if l] + [a for l in lists for a in l]
        r = rows[rid]
        yield {"id": rid, "split": name, "question": r["question"].strip(), "gold": dedup(gold),
               "meta": {"prop": r["prop"], "subj_id": r["subj_id"], "s_pop": int(r["s_pop"]), "merged_ids": ids}}


def split_names(splits):
    """("train", "tune", "test") for the base split file; the single fresh split for a file made by
    newsets_splits.py extend."""
    return [splits["name"]] if "name" in splits else ["train", "tune", "test"]


def pro_key(splits):
    return (splits["name"] + "_pro100") if "name" in splits else "test_pro100"


def iter_split_ids(splits):
    for name in split_names(splits):
        for i in splits[name]:
            yield name, i


def build(dataset, raw_path, splits):
    pro = set(splits[pro_key(splits)])
    fn = triviaqa_records if dataset == "triviaqa" else popqa_records
    recs = []
    for r in fn(raw_path, splits):
        if not r["gold"]:
            raise SystemExit("no gold answer for %s" % r["id"])
        r["pro100"] = r["id"] in pro
        recs.append(r)
    n = {k: sum(1 for r in recs if r["split"] == k) for k in split_names(splits)}
    if n != {k: len(splits[k]) for k in n} or sum(r["pro100"] for r in recs) != len(pro):
        raise SystemExit("record counts do not match the split file: %r" % n)
    return recs


def selftest():
    assert dedup([" a", "a", "", "b "]) == ["a", "b"]
    splits = {"train": ["1"], "tune": ["2"], "test": ["3"], "test_pro100": ["3"], "merged_row_ids": {"3": ["3", "9"]}}
    import io
    import os
    import tempfile
    tsv = "id\tquestion\tsubj_id\tprop\ts_pop\tpossible_answers\n" \
          "1\tQ1\ts1\tr\t5\t[\"x\"]\n2\tQ2\ts2\tr\t6\t[\"y\", \"y\"]\n3\tQ3\ts3\tr\t7\t[\"z\"]\n9\tQ3\ts3\tr\t7\t[\"w\"]\n"
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.tsv")
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(tsv)
    recs = build("popqa", p, splits)
    assert [r["id"] for r in recs] == ["1", "2", "3"]
    assert recs[1]["gold"] == ["y"] and recs[2]["gold"] == ["z", "w"] and recs[2]["pro100"] and not recs[0]["pro100"]
    assert recs[2]["meta"]["merged_ids"] == ["3", "9"]
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=["triviaqa", "popqa"])
    ap.add_argument("raw", nargs="?")
    ap.add_argument("splits", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.dataset and a.raw and a.splits and a.out):
        ap.error("dataset, raw, splits and out are required")
    with open(a.splits, encoding="utf-8") as f:
        splits = json.load(f)
    if splits.get("dataset") != a.dataset:
        raise SystemExit("split file is for %r" % splits.get("dataset"))
    recs = build(a.dataset, a.raw, splits)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    golds = sorted(len(r["gold"]) for r in recs)
    print(json.dumps({"dataset": a.dataset, "records": len(recs), "gold_aliases_median": golds[len(golds) // 2],
                      "gold_aliases_max": golds[-1]}))


if __name__ == "__main__":
    main()
