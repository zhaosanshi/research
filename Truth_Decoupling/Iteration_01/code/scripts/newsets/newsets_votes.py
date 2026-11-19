"""Verified DeepSeek Pro votes of one newsets judge work dir (GPT review 2026-11-19, section 2: votes must belong to
the exact answers being scored; cells without a vote, duplicate votes and votes for other answers stop the run).

load_votes(work, gen_dir=None) -> (table, counts, meta)
  table[(pair_id, condition)] = dict(state, blank, text, why)
    state  R right / W wrong / H hedge / X blank (Pro refusal label C, or an empty answer) / ? excluded
    why    "scored", "empty answer", "pro declined", "refusal-only references" (the last two are excluded: state ?)
  Checks: judge_input.json is the file the work dir was built from (meta generation_sha256); cells.jsonl unchanged
  (meta cells_sha256); one vote per cell, with the cell's own input_sha; every cell has a vote unless its question has
  refusal-only references; with gen_dir, every judged answer equals the current generation's text, and the judge
  input's source files are the current generation files (sha256).
usage: python3 -I newsets_votes.py verify WORK [--gen-dir DIR]     (prints the counts; exit 1 on any problem)
       python3 -I newsets_votes.py --selftest
"""
import hashlib
import json
import sys
from pathlib import Path

STATE = {"A": "R", "B": "W", "H": "H", "C": "X"}


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_votes(work, gen_dir=None):
    work = Path(work)
    j2 = work / "judge2"
    meta = json.loads((j2 / "meta.json").read_text(encoding="utf-8"))
    gen_path = work / meta["generation"]
    if sha_file(gen_path) != meta["generation_sha256"]:
        raise SystemExit("%s: judge input changed after the cells were built" % work)
    if sha_file(j2 / "cells.jsonl") != meta["cells_sha256"]:
        raise SystemExit("%s: cells changed after they were built" % work)
    cells = {}
    for c in rows(j2 / "cells.jsonl"):
        if c["cell_id"] in cells:
            raise SystemExit("%s: cell %s twice" % (work, c["cell_id"]))
        cells[c["cell_id"]] = c
    votes = {}
    for v in rows(j2 / "single_pro.jsonl"):
        cid = v["cell_id"]
        if cid in votes:
            raise SystemExit("%s: two votes for %s" % (work, cid))
        if cid not in cells:
            raise SystemExit("%s: a vote for a cell that is not in this work dir: %s" % (work, cid))
        if v.get("input_sha") != cells[cid]["input_sha"]:
            raise SystemExit("%s: the vote for %s belongs to another answer" % (work, cid))
        votes[cid] = v
    gens = None
    if gen_dir is not None:
        judge_input = json.loads(gen_path.read_text(encoding="utf-8"))
        gens = {}
        for arm, src in judge_input["source_files"].items():
            p = Path(gen_dir) / src["file"]
            if not p.exists() or sha_file(p) != src["sha256"]:
                raise SystemExit("%s: the generation %s is not the one this work dir judged" % (work, src["file"]))
            g = json.loads(p.read_text(encoding="utf-8"))
            gens[g["contract"]["condition"]] = {"%s:%s" % (g["contract"]["task"], r["id"]): r["text"] for r in g["rows"]}
    table, counts = {}, {"scored": 0, "empty answer": 0, "pro declined": 0, "refusal-only references": 0}
    for cid, c in cells.items():
        k = (c["pair_id"], c["condition"])
        if gens is not None and gens[c["condition"]].get(c["pair_id"]) != c["prediction"]:
            raise SystemExit("%s: the judged answer of %s is not the current generation's" % (work, cid))
        blank = bool(c.get("rule_abstain")) or not c["prediction"].strip()
        v = votes.get(cid)
        if c.get("gold_is_abstention"):
            why, state = "refusal-only references", "?"
        elif v is None:
            raise SystemExit("%s: no vote for %s (judging unfinished?)" % (work, cid))
        elif v["status"] == "scored":
            if v["label"] not in STATE:
                raise SystemExit("%s: unknown label %r for %s" % (work, v["label"], cid))
            why, state = "scored", STATE[v["label"]]
        elif v.get("basis") == "empty_output_no_judge_vote" and v.get("empty_output") is True:
            why, state = "empty answer", "X"
        elif v.get("basis") == "pro_declined":
            why, state = "pro declined", "?"
        else:
            raise SystemExit("%s: vote for %s has status %r / basis %r" % (work, cid, v["status"], v.get("basis")))
        counts[why] += 1
        table[k] = dict(state=state, blank=blank, text=c["prediction"], why=why)
    return table, counts, meta


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp)
        (w / "judge2").mkdir()
        (w / "judge_input.json").write_text(json.dumps({"source_files": {}}))
        cells = [dict(cell_id="p:1|baseline_x", pair_id="p:1", condition="baseline_x", prediction="Paris", input_sha="s1",
                      rule_abstain=False, gold_is_abstention=False),
                 dict(cell_id="p:2|baseline_x", pair_id="p:2", condition="baseline_x", prediction="", input_sha="s2",
                      rule_abstain=False, gold_is_abstention=False),
                 dict(cell_id="p:3|baseline_x", pair_id="p:3", condition="baseline_x", prediction="X", input_sha="s3",
                      rule_abstain=False, gold_is_abstention=True)]
        (w / "judge2" / "cells.jsonl").write_text("\n".join(map(json.dumps, cells)) + "\n")

        def meta():
            (w / "judge2" / "meta.json").write_text(json.dumps(dict(generation="judge_input.json",
                                                                    generation_sha256=sha_file(w / "judge_input.json"),
                                                                    cells_sha256=sha_file(w / "judge2" / "cells.jsonl"))))
        meta()
        good = [dict(cell_id="p:1|baseline_x", status="scored", label="A", input_sha="s1"),
                dict(cell_id="p:2|baseline_x", status="not_scored", basis="empty_output_no_judge_vote", empty_output=True, input_sha="s2")]

        def put(vs):
            (w / "judge2" / "single_pro.jsonl").write_text("\n".join(map(json.dumps, vs)) + "\n")
        put(good)
        t, cnt, _ = load_votes(w)
        assert t[("p:1", "baseline_x")]["state"] == "R" and t[("p:2", "baseline_x")]["state"] == "X"
        assert t[("p:3", "baseline_x")]["state"] == "?" and cnt == {"scored": 1, "empty answer": 1, "pro declined": 0, "refusal-only references": 1}
        for bad in ([good[0], dict(good[0], label="B")],                         # two votes for one cell
                    [dict(good[0], input_sha="other"), good[1]],                  # a vote for another answer
                    [good[0]],                                                    # a cell without a vote
                    [good[0], dict(good[1], basis="pending", status="pending")]):  # an unfinished cell
            put(bad)
            try:
                load_votes(w)
                raise AssertionError("accepted bad votes %r" % bad)
            except SystemExit:
                pass
        put(good)
        (w / "judge_input.json").write_text(json.dumps({"source_files": {}, "changed": True}))
        try:
            load_votes(w)
            raise AssertionError("accepted a changed judge input")
        except SystemExit:
            pass
    print("selftest PASS")


def main():
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
        return
    if len(a) < 2 or a[0] != "verify":
        raise SystemExit(__doc__)
    gen_dir = a[a.index("--gen-dir") + 1] if "--gen-dir" in a else None
    _, counts, meta = load_votes(a[1], gen_dir)
    print(json.dumps({"work": a[1], "counts": counts, "model": meta.get("model"), "n_cells": meta.get("n_cells")}))


if __name__ == "__main__":
    main()
