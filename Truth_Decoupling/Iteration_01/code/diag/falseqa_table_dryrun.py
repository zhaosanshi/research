#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-24): a dry run of the FalseQA held-out table
# (diag/falseqa_read.py run_table, unchanged) in which every held-out verdict is replaced before anything is counted.
# The whole path of the confirmation reading is executed on the real answers, rule blanks, answer texts and self-judge
# margins, and no real held-out verdict is read into a number. Read-only: no model, no judge call, nothing written.
# Real here: the bank, the answers of the 22 arms, the rule blanks, the development verdicts (so the caps are the ones
# the real reading chooses; N1 / SN1 / Conly / SC pick their cells from blanks and texts only, so the held-out cells
# and the margins they need are the ones of the real reading). Replaced: the value of every held-out cell.
#   right   every answered held-out cell +1, every blank cell 0
#   wrong   every answered held-out cell -1, every blank cell 0
#   random  a value in {-1, 0, +1} from sha256(question id, answer text) (same text -> same value); empty text 0
# The table text is captured and thrown away (its numbers are not results). Printed: per mode whether run_table ran
# through, how many lines and sections it printed, how many "n/a" (a missing self-judge margin) it has, and the checks:
#   right / wrong: every system's net is +/- its number of answers (in the cells and in the printed "net (answers)"
#     lines); in each equal-count comparison of the table the trimmed difference inside the pool sums to 0 (both sides
#     keep the same number of answers, all of one value), no margin is missing, and outside the pool the difference is
#     the natural one.
#   every mode: wherever S0 answers, every fill system hands in S0's cell; inside B each fill system hands in the cell
#     its definition names; 600 held-out questions in 300 pairs, 300 per premise type; three sections.
# usage (scoring container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/falseqa_table_dryrun.py [right|wrong|random ...] [-- MODEL ...]
import contextlib
import hashlib
import io
import re
import sys

sys.path[:0] = ["Iteration_01/diag"]
import falseqa_read as F     # noqa: E402

MODES = ("right", "wrong", "random")
# the equal-count comparisons of falseqa_read.readings: (A, B, pool)
EQ = (("N1", "S0", "all"), ("fill EC", "fill C", "B"), ("fill EC", "fill E", "B"), ("fill EC", "fill pE2", "P"),
      ("N1", "E", "all"), ("N1", "Conly", "all"), ("E", "S0", "all"))
FILLS = (("fill EC", "SN1"), ("fill E", "SE"), ("fill C", "SC"), ("fill prompt", "T2"))
NET_RE = re.compile(r"(?:^|\| )([A-Za-z0-9 ]+?) ([+-]\d+) \((\d+)\)")


def fake_value(mode, q, text, blank):
    if mode == "right":
        return 0 if blank else 1
    if mode == "wrong":
        return 0 if blank else -1
    t = text.strip()
    if not t:
        return 0
    return hashlib.sha256((q + "\0" + t).encode("utf-8")).digest()[0] % 3 - 1


def run(mode, models, held):
    orig_load, orig_read = F.load_rows, F.readings
    seen, stat = {}, {"cells": 0, "loads": 0}

    def load(model, keep, works=F.WORKS):
        R, R2 = orig_load(model, keep, works)
        for RR in (R, R2):
            for key in list(RR.val):
                if key[0] in held:
                    RR.val[key] = 0          # no real held-out verdict stays in the object
            for arm, rows in RR.rows.items():
                c = RR.cond[arm]
                for q in held:
                    RR.val[(q, c)] = fake_value(mode, q, rows[q]["text"], RR.blank(arm, q))
                    stat["cells"] += 1
        stat["loads"] += 1
        return R, R2

    def read(data, done, qt, pair_of, groups, M):
        seen.update(data=data, done=list(done), qt=qt, pair_of=pair_of, groups=groups, M=M)
        return orig_read(data, done, qt, pair_of, groups, M)

    F.load_rows, F.readings = load, read
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            F.run_table(models)
    finally:
        F.load_rows, F.readings = orig_load, orig_read
    lines = buf.getvalue().splitlines()
    bad, n_checks = [], 0

    def check(ok, what):
        nonlocal n_checks
        n_checks += 1
        if not ok:
            bad.append(what)

    check(bool(seen), "run_table did not reach the readings")
    if not seen:
        return lines, bad, n_checks, stat, seen
    data, qt, M, pair_of, groups = seen["data"], seen["qt"], seen["M"], seen["pair_of"], seen["groups"]
    check(seen["done"] == list(models), "models read %s, asked %s" % (seen["done"], list(models)))
    check([len(qs) for _, qs in groups] == [300, 300, 600], "group sizes %s" % [len(qs) for _, qs in groups])
    check(set(groups[2][1]) == set(held) and len({pair_of[q] for q in groups[2][1]}) == 300, "the all group is not the 300 held-out pairs")
    check(all(len({pair_of[q] for q in qs}) == 300 for _, qs in groups[:2]), "a premise group does not have one question of every pair")
    check(sum(1 for l in lines if l.startswith("## held-out")) == 3, "sections printed")
    check(sum(1 for l in lines if l.startswith("-- 5 valid premise")) == 1, "the valid-premise block is not printed exactly once")
    check(not any("n/a" in l for l in lines), "%d lines with n/a (a self-judge margin is missing)" % sum("n/a" in l for l in lines))
    sgn = {"right": 1, "wrong": -1}.get(mode)
    for m in seen["done"]:
        d = data[m]
        for q in groups[2][1]:
            s0 = d["S0"][q]
            if s0[1]:
                check(all(d[f][q] == s0 for f, _ in FILLS) and d["fill pE2"][q] == s0, "%s %s: a fill system changes a question S0 answers" % (m, q))
            else:
                check(all(d[f][q] == d[src][q] for f, src in FILLS), "%s %s: a fill system does not hand in its source cell in B" % (m, q))
                check(d["fill pE2"][q] == (d["SE"][q] if d["SE"][q][1] else d["TE"][q]), "%s %s: fill pE2 in B" % (m, q))
        if sgn is None:
            continue
        for label, qs in groups:
            for n in F.SYS_ORDER:
                check(all(d[n][q][0] == sgn * int(d[n][q][1]) for q in qs), "%s %s %s: a cell is not %+d per answer" % (m, label, n, sgn))
            for a, b, pn in EQ:
                pool = [q for q in qs if pn == "all" or (not d["S0"][q][1] and (pn == "B" or not d["SE"][q][1]))]
                dd, cnt, miss = F.equal_diff(qs, pool, d[a], d[b], qt, M.get(m, {}))
                check(dd is not None and miss == 0, "%s %s %s - %s: %d margins missing" % (m, label, a, b, miss))
                if dd is None:
                    continue
                inp = set(pool)
                check(sum(dd[q] for q in pool) == 0, "%s %s %s - %s: equal counts in the pool do not cancel (%d; answers %s)" % (
                    m, label, a, b, sum(dd[q] for q in pool), cnt))
                check(all(dd[q] == d[a][q][0] - d[b][q][0] for q in qs if q not in inp), "%s %s %s - %s: outside the pool" % (m, label, a, b))
    if sgn is not None:
        n_net = 0
        for i, l in enumerate(lines):
            if l.startswith("-- every system: net (answers)"):
                for row in lines[i + 1:i + 2 + len(seen["done"])]:
                    for name, net, ans in NET_RE.findall(row.strip().split(None, 1)[1] if row.strip() else ""):
                        n_net += 1
                        check(int(net) == sgn * int(ans), "printed net %s (%s) of %s" % (net, ans, name))
        check(n_net >= 3 * len(F.SYS_ORDER) * len(seen["done"]), "printed net lines parsed: %d" % n_net)
    return lines, bad, n_checks, stat, seen


def main():
    a = sys.argv[1:]
    cut = a.index("--") if "--" in a else len(a)
    modes = a[:cut] or list(MODES)
    models = a[cut + 1:] or list(F.ORDER)
    if any(x not in MODES for x in modes) or any(x not in F.ORDER for x in models):
        raise SystemExit("usage: falseqa_table_dryrun.py [right|wrong|random ...] [-- MODEL ...]")
    held = {r["id"] for r in F.load_bank() if r["set"] == "heldout"}
    print("# FalseQA held-out table, dry run with every held-out verdict replaced (%d held-out questions); the table text is not shown" % len(held))
    fail = 0
    for mode in modes:
        lines, bad, n, stat, seen = run(mode, models, held)
        print("dryrun %-6s run_table ran through: models %s | %d lines, %d sections | %d loads, %d held-out cells replaced" % (
            mode, ",".join(seen.get("done", [])) or "none", len(lines), sum(1 for l in lines if l.startswith("## ")), stat["loads"], stat["cells"]))
        for b in bad[:25]:
            print("  FAILED: " + b)
        print("dryrun %-6s checks: %s (%d checks, %d failed)" % (mode, "PASS" if not bad else "FAILED", n, len(bad)))
        fail += bool(bad)
        sys.stdout.flush()
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
