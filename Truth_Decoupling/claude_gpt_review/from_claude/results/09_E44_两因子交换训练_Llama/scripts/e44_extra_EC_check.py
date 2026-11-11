# -*- coding: utf-8 -*-
"""Read-only checks after an E+C generation stage (from_claude/36, section 3). No model, no judge.
  1. reproduction: single-write arms made by e44_eval_extra.py vs the driver's own files of the same arm
     (token ids per question must be identical)
  2. rule abstentions (the judge-independent refusal_rule of every answer) per arm, and the pre-registered choice of
     the C strength k: |rule abstentions(x:E10.w.p2+C31.w.m<k>) - rule abstentions(baseline x)| smallest; ties -> smaller k
usage: python e44_extra_EC_check.py RUN_DIR [--strengths 0.25 0.5 1]
exit code 1 if a reproduction differs."""
import argparse, json, sys
from pathlib import Path


def token(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def rows(path):
    d = json.load(open(path, encoding="utf-8"))
    if d.get("completed") is not True or d.get("partial") not in (None, False):
        raise SystemExit("not a complete full-split file: %s" % path)
    return {r["pair_id"]: r for r in d["rows"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--strengths", nargs="+", default=["0.25", "0.5", "1"])
    p.add_argument("--split", default="tune")
    a = p.parse_args()
    run, sp = a.run_dir, a.split
    extra = run / "eval_extra" / "gen"
    bad = 0
    print("## reproduction: eval_extra single-write arms vs the driver's files")
    pairs = [("x:E10.w.p2", "E_L10_x-w-p2")] + [("x:C31.w.m%s" % k, "C_L31_x-w-m%s" % k.replace(".", "-")) for k in a.strengths]
    for arm, drv in pairs:
        fe, fd = extra / ("%s_%s.json" % (token(arm), sp)), run / "gen" / ("%s_%s.json" % (drv, sp))
        if not fe.exists() or not fd.exists():
            print("- %s: missing (%s exists %s, %s exists %s)" % (arm, fe.name, fe.exists(), fd.name, fd.exists()))
            continue
        e, d = rows(fe), rows(fd)
        if list(e) != list(d):
            print("- %s: question sets differ" % arm)
            bad += 1
            continue
        same = sum(1 for q in e if e[q]["token_ids"] == d[q]["token_ids"])
        print("- %s vs %s: %d / %d identical" % (arm, fd.name, same, len(e)))
        bad += same != len(e)
    frozen = json.load(open(run / ("frozen_%s.json" % sp), encoding="utf-8"))
    base = sum(1 for q in frozen["questions"] if q["answers"]["x"]["refusal_rule"])
    print("\n## rule abstentions (judge-independent), baseline x = %d of %d" % (base, len(frozen["questions"])))
    for f in sorted(extra.glob("*_%s.json" % sp)):
        if f.name.startswith("t") and f.name[1:2].isdigit():
            continue
        r = rows(f)
        print("- %s: %d" % (f.name, sum(1 for v in r.values() if v["refusal_rule"])))
    cand = []
    for k in a.strengths:
        f = extra / ("%s_%s.json" % (token("x:E10.w.p2+C31.w.m%s" % k), sp))
        if f.exists():
            n = sum(1 for v in rows(f).values() if v["refusal_rule"])
            cand.append((abs(n - base), float(k), k, n))
    if cand:
        cand.sort()
        print("\n## pre-registered choice of k: %s (rule abstentions %d vs baseline %d); candidates %s" % (
            cand[0][2], cand[0][3], base, ", ".join("m%s: %d" % (c[2], c[3]) for c in sorted(cand, key=lambda c: c[1]))))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
