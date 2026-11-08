#!/usr/bin/env python3
"""Blind human audit sample for the two-axis grading (judge-vs-human agreement that the pipeline's own audit cannot
give: the pipeline's human queue shows the judges' votes and the condition, so a verdict given there is arbitration,
not an independent rating — GPT round 22 / 23).

  sample  RUN_DIR [RUN_DIR ...] --out DIR --tag TAG [--n-per-run 25] [--seed 1] [--exclude FILE ...]
      Draw cells with substantive gold and a non-empty answer from each run (fixed seed), skipping any cell whose
      (question, answer) content appears in an --exclude file (JSON list of {"question","prediction"} or of cell ids
      with a "cells" path; the cells the reviewer has already seen with votes are not blind any more). Identical
      content is sampled once across runs. Writes DIR/TAG_key.jsonl (hidden: cell ids, run dirs, content shas) and
      DIR/TAG_sheet.md (for the reviewer: number, question, references, incorrect candidates, answer — no votes,
      no labels, no condition names, no judge output).
  apply   DIR --tag TAG --verdicts "1A 2B(H) 3C ..." --by WHO [--config CFG]
      Record the reviewer's verdicts through judge_two_axis_run.human_set, one per numbered item, into the run that
      holds the cell (and TD_JUDGE2_HUMAN if set, so identical content in the other run gets the same verdict).
      A B must carry its information subtype in parentheses: B(S) false but informative, B(H) false hedge,
      B(R) false refusal. A / C / H need none.
  report  DIR --tag TAG
      Agreement of each judge's own vote and of the pipeline's pre-human label with the blind verdicts (reads the
      runs' final.jsonl / judge files; requires `adjudicate` to have been run after `apply`).
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from core.io_utils import read_jsonl, write_jsonl  # noqa: E402
from core.judge_two_axis import LABELS, TRUTH_VALUES, allenai_votes, judge_own_label, coarse, info_binary  # noqa: E402
from scripts.review_checks import judge_two_axis_run as R  # noqa: E402

VERDICT = re.compile(r"^(\d+)([ABCH])(?:\(([SHR])\))?$")


def _content_key(question, prediction):
    return (question.strip(), prediction.strip())


def _load_exclusions(paths):
    keys = set()
    for p in paths or []:
        data = json.load(open(p, encoding="utf-8"))
        if isinstance(data, dict) and "cells" in data and "cell_ids" in data:      # {"cells": cells.jsonl, "cell_ids": [...]}
            by_id = {c["cell_id"]: c for c in read_jsonl(data["cells"])}
            for cid in data["cell_ids"]:
                if cid in by_id:
                    keys.add(_content_key(by_id[cid]["question"], by_id[cid]["prediction"]))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "question" in item and "prediction" in item:
                    keys.add(_content_key(item["question"], item["prediction"]))
        else:
            raise SystemExit("exclude file must be a JSON list of {question, prediction} or {cells, cell_ids}: %s" % p)
    return keys


def sample(run_dirs, out_dir, tag, n_per_run, seed, exclude_paths):
    excluded = _load_exclusions(exclude_paths)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    chosen, seen_content = [], set()
    for run_dir in run_dirs:
        cells = [c for c in read_jsonl(os.path.join(run_dir, "judge2", "cells.jsonl"))
                 if not c["gold_is_abstention"] and c["prediction"].strip()]
        pool = [c for c in sorted(cells, key=lambda c: c["cell_id"])
                if _content_key(c["question"], c["prediction"]) not in excluded
                and _content_key(c["question"], c["prediction"]) not in seen_content]
        # one draw per distinct content, so the sheet never shows the same answer twice
        by_content = {}
        for c in pool:
            by_content.setdefault(_content_key(c["question"], c["prediction"]), c)
        picks = rng.sample(sorted(by_content.values(), key=lambda c: c["cell_id"]), min(n_per_run, len(by_content)))
        for c in picks:
            seen_content.add(_content_key(c["question"], c["prediction"]))
            chosen.append({"run_dir": run_dir, "cell_id": c["cell_id"], "input_sha": c.get("input_sha"), "question": c["question"],
                           "gold": c["gold"], "gold_refs": c["gold_refs"], "incorrect": c["incorrect"], "best_answer": c.get("best_answer"),
                           "prediction": c["prediction"]})
    rng.shuffle(chosen)                                   # run order must not be guessable from the numbering
    for i, item in enumerate(chosen, 1):
        item["number"] = i
    write_jsonl(chosen, os.path.join(out_dir, tag + "_key.jsonl"))
    lines = ["# 盲评抽检 %s（%d 格）— 只看题、参考答案和回答，给 A 真且有信息 / B 假（B 要加括号：S 有信息、H 含糊、R 拒答）/ C 拒答 / H 含糊" % (tag, len(chosen)), "",
             "| # | 题 | 参考答案 | 算错的答案 | 回答 |", "|---|---|---|---|---|"]
    for item in chosen:
        esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
        lines.append("| %d | %s | %s | %s | **%s** |" % (item["number"], esc(item["question"]), esc(" / ".join(item["gold"])),
                                                        esc(" / ".join(item["incorrect"])), esc(item["prediction"])))
    open(os.path.join(out_dir, tag + "_sheet.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return {"n": len(chosen), "per_run": dict(Counter(i["run_dir"] for i in chosen)), "excluded_contents": len(excluded),
            "key": os.path.join(out_dir, tag + "_key.jsonl"), "sheet": os.path.join(out_dir, tag + "_sheet.md")}


def parse_verdicts(text):
    out = {}
    for tok in text.replace(",", " ").split():
        m = VERDICT.match(tok.strip())
        if not m:
            raise SystemExit("cannot read verdict %r (expected e.g. 12A, 7B(H), 3C)" % tok)
        n, letter, sub = int(m.group(1)), m.group(2), m.group(3)
        if letter == "B" and sub is None:
            raise SystemExit("item %d: a B needs its information subtype in parentheses, B(S) / B(H) / B(R)" % n)
        if letter != "B" and sub is not None:
            raise SystemExit("item %d: only B takes a subtype" % n)
        if n in out:
            raise SystemExit("item %d given twice" % n)
        out[n] = (letter, sub)
    return out


def apply(out_dir, tag, verdicts_text, by, note=None):
    key = {item["number"]: item for item in read_jsonl(os.path.join(out_dir, tag + "_key.jsonl"))}
    verdicts = parse_verdicts(verdicts_text)
    missing = sorted(set(key) - set(verdicts))
    rows = []
    for n, (letter, sub) in sorted(verdicts.items()):
        if n not in key:
            raise SystemExit("item %d is not on sheet %s" % (n, tag))
        item = key[n]
        rows.append(R.human_set(item["run_dir"], item["cell_id"], letter, by, note or ("blind audit %s #%d" % (tag, n)),
                                truth=None, info_subtype=sub))
    return {"recorded": len(rows), "missing_items": missing}


def report(out_dir, tag):
    key = list(read_jsonl(os.path.join(out_dir, tag + "_key.jsonl")))
    pairs = {"allenai": [], "pro": [], "gpt": [], "pipeline_before_human": []}
    n_human = 0
    for item in key:
        run_dir = item["run_dir"]
        j2 = os.path.join(run_dir, "judge2")
        cells = {c["cell_id"]: c for c in read_jsonl(os.path.join(j2, "cells.jsonl"))}
        humans = R._human_verdicts(run_dir, list(cells.values()))
        h = humans.get(item["cell_id"])
        if h is None:
            continue
        n_human += 1
        human_label = h["label"]
        judges = {n: R._rows(os.path.join(j2, n + ".jsonl")) for n in ("allenai", "pro", "gpt")}
        cfg = R.load_cfg()
        for name, rows in judges.items():
            r = rows.get(item["cell_id"])
            if not r:
                continue
            if name == "allenai":
                t, i = allenai_votes(r, cfg["allenai"]["threshold"])
                own = judge_own_label(t, i)
                if own is not None:
                    pairs[name].append((coarse(own), coarse(human_label)))
            else:
                own = judge_own_label(r.get("truth") if r.get("truth") in TRUTH_VALUES else None,
                                      r.get("info") if r.get("info") in ("S", "H", "R") else None)
                if own is not None:
                    pairs[name].append((own, human_label))
        rec = R.adjudicate_cell(cells[item["cell_id"]], allenai=judges["allenai"].get(item["cell_id"]), pro=judges["pro"].get(item["cell_id"]),
                                gpt=judges["gpt"].get(item["cell_id"]), human=None, threshold=cfg["allenai"]["threshold"])
        if rec["label"] is not None:
            pairs["pipeline_before_human"].append((rec["label"], human_label))
    out = {"n_items": len(key), "n_with_human_verdict": n_human}
    for name, p in pairs.items():
        out[name] = dict(R._rate(p), coarse_labels=(name == "allenai"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("run_dirs", nargs="+"); p.add_argument("--out", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--n-per-run", type=int, default=25); p.add_argument("--seed", type=int, default=1); p.add_argument("--exclude", nargs="*", default=[])
    p = sub.add_parser("apply"); p.add_argument("out"); p.add_argument("--tag", required=True); p.add_argument("--verdicts", required=True)
    p.add_argument("--by", required=True); p.add_argument("--note", default=None)
    p = sub.add_parser("report"); p.add_argument("out"); p.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "sample":
        print(json.dumps(sample(a.run_dirs, a.out, a.tag, a.n_per_run, a.seed, a.exclude), ensure_ascii=False, indent=1))
    elif a.cmd == "apply":
        print(json.dumps(apply(a.out, a.tag, a.verdicts, a.by, a.note), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(report(a.out, a.tag), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
