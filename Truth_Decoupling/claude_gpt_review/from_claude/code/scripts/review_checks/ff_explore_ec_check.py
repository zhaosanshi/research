# -*- coding: utf-8 -*-
"""Read-only checks after the first E+C generation stage of ff_explore_ec.sh (from_claude/44 section 5.8). The Llama
version is results/09_E44_两因子交换训练_Llama/scripts/e44_extra_EC_check.py (from_claude/36 section 3); here the layers
and strengths are parameters and the E and C directions may live in two runs. No model, no judge.
  1. reproduction: every single-write arm made by e44_eval_extra.py against the driver's own file of the same arm (E arms
     in --run-dir, C arms in --c-run-dir); token ids, finished flag, stop id and text per question must be identical
     (the fields of e44_eval_extra.same_answers). A missing eval_extra file is an error; a missing driver file is only
     reported (nothing to compare).
  2. rule abstentions (the judge-independent refusal_rule of every answer) per eval_extra file, and the choice of the C
     strength k for each E strength s, fixed before any judging (from_claude/36 section 3, unchanged): the k whose
     E s + C toward confident k has |rule abstentions - rule abstentions of baseline x| smallest; ties -> smaller k.
The choice is written to --out only when every check passed; exit code 1 otherwise.
usage: ff_explore_ec_check.py --run-dir E_RUN --c-run-dir C_RUN --el 15 --cl 16 --es 1 2 --ks 0.25 0.5 1 2 --out FILE"""
import argparse
import json
import sys
from pathlib import Path


def token(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def two(e_part, c_part, el, cl):
    """The two-write arm on x, lower layer first (the grammar of e44_eval_extra.py; same rule as ff_explore_ec.sh)."""
    return "x:%s+%s" % ((e_part, c_part) if el < cl else (c_part, e_part))


def rows(path):
    d = json.load(open(path, encoding="utf-8"))
    if d.get("completed") is not True or d.get("partial") not in (None, False):
        raise SystemExit("not a complete full-split file: %s" % path)
    return {r["pair_id"]: r for r in d["rows"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="the run of the E direction (eval_extra/ lives here)")
    p.add_argument("--c-run-dir", type=Path, required=True, help="the run of the C direction")
    p.add_argument("--el", type=int, required=True)
    p.add_argument("--cl", type=int, required=True)
    p.add_argument("--es", nargs="+", required=True, help="E strengths, as written in the arm names")
    p.add_argument("--ks", nargs="+", required=True, help="C strengths toward confident, as written in the arm names")
    p.add_argument("--split", default="tune")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run, crun, sp, el, cl = a.run_dir, a.c_run_dir, a.split, a.el, a.cl
    if el == cl:
        raise SystemExit("the two writes must act at two different layers")
    extra = run / "eval_extra" / "gen"
    bad = 0
    print("## 1. reproduction: eval_extra single-write arms vs the drivers' own files\n")
    pairs = []
    for s in a.es:
        st = s.replace(".", "-")
        pairs += [("x:E%d.w.p%s" % (el, s), run / "gen" / ("E_L%d_x-w-p%s_%s.json" % (el, st, sp))),
                  ("x:E%d.rand.p%s" % (el, s), run / "gen" / ("E_L%d_x-rand-p%s_%s.json" % (el, st, sp))),
                  ("x:E%d.rand.m%s" % (el, s), run / "gen" / ("E_L%d_x-rand-m%s_%s.json" % (el, st, sp)))]
    for k in a.ks:
        pairs.append(("x:C%d.w.m%s" % (cl, k), crun / "gen" / ("C_L%d_x-w-m%s_%s.json" % (cl, k.replace(".", "-"), sp))))
    for arm, fdrv in pairs:
        fe = extra / ("%s_%s.json" % (token(arm), sp))
        if not fe.exists():
            print("- %s: eval_extra file %s MISSING" % (arm, fe.name))
            bad += 1
            continue
        if not fdrv.exists():
            print("- %s: no driver file %s (nothing to compare)" % (arm, fdrv.name))
            continue
        e, d = rows(fe), rows(fdrv)
        if list(e) != list(d):
            print("- %s: question sets differ" % arm)
            bad += 1
            continue
        # the fields e44_eval_extra.same_answers compares (the 9B has two stop ids, so stop_id and finished count)
        same = sum(1 for q in e if all(e[q][f] == d[q][f] for f in ("token_ids", "finished", "stop_id", "text")))
        print("- %s vs %s/%s: %d / %d identical (token ids, finished, stop id, text)" % (arm, fdrv.parent.parent.name, fdrv.name, same, len(e)))
        bad += same != len(e)
    frozen = json.load(open(run / ("frozen_%s.json" % sp), encoding="utf-8"))
    base = sum(1 for q in frozen["questions"] if q["answers"]["x"]["refusal_rule"])
    print("\n## 2. rule abstentions (judge-independent), baseline x = %d of %d\n" % (base, len(frozen["questions"])))
    for f in sorted(extra.glob("*_%s.json" % sp)):
        if f.name.startswith("t") and f.name[1:2].isdigit():
            continue
        r = rows(f)
        print("- %s: %d" % (f.name, sum(1 for v in r.values() if v["refusal_rule"])))
    choice, cands = {}, {}
    for s in a.es:
        cand = []
        for k in a.ks:
            arm = two("E%d.w.p%s" % (el, s), "C%d.w.m%s" % (cl, k), el, cl)
            f = extra / ("%s_%s.json" % (token(arm), sp))
            if not f.exists():
                print("- %s: MISSING" % f.name)
                bad += 1
                continue
            n = sum(1 for v in rows(f).values() if v["refusal_rule"])
            cand.append((abs(n - base), float(k), k, n))
        if len(cand) != len(a.ks):
            continue
        cand.sort()
        choice[s] = cand[0][2]
        cands[s] = {c[2]: c[3] for c in sorted(cand, key=lambda c: c[1])}
        print("\n## choice of k for E strength %s: %s (rule abstentions %d vs baseline %d); candidates %s" % (
            s, cand[0][2], cand[0][3], base, ", ".join("m%s: %d" % (k, n) for k, n in cands[s].items())))
    if bad:
        print("\n%d check(s) failed: no choice file written" % bad)
        sys.exit(1)
    payload = {"rule": "per E strength s: the C strength k (toward confident) whose E s + C k has |rule abstentions - baseline x| "
                       "smallest; ties -> smaller k (from_claude/36 section 3; from_claude/44 section 5.8)",
               "run_dir": str(run), "c_run_dir": str(crun), "e_layer": el, "c_layer": cl, "split": sp,
               "baseline_rule_abstentions": base, "choice": choice, "candidates": cands}
    if a.out.exists():
        if json.load(open(a.out, encoding="utf-8")) != payload:
            raise SystemExit("an existing choice file %s differs: the choice is never redone after the fact" % a.out)
    else:
        tmp = a.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(a.out)
    print("\nchoice written to %s" % a.out)


if __name__ == "__main__":
    main()
