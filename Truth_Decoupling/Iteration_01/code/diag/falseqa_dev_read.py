#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-24): a first look at FalseQA on the 100 development
# questions only (50 pairs: 50 false-premise and 50 valid-premise questions per model). Read-only: no model, no judge call.
# Only the development work dir (newsets/judge/falseqa/<model>/test_full700_dev) is opened; the 600 held-out questions
# and their verdicts are not read here. Everything is computed by diag/falseqa_read.py, unchanged: the caps of N1, SN1,
# Conly and SC are chosen on these same 100 questions (falseqa_read.choose), so the lines of those four systems are
# selected and read on the same questions and lean upwards; O, E, S0, SE, T2 and TE have nothing chosen here.
# With 50 questions per premise type and model the intervals are wide: this look gives the direction, not the size.
# The confirmation stays the held-out reading (falseqa_read.py table), which this program does not touch.
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/falseqa_dev_read.py [MODEL ...]
import sys

sys.path[:0] = ["Iteration_01/diag"]
import falseqa_read as F     # noqa: E402

DEV_WORK = ("test_full700_dev",)


def main():
    models = sys.argv[1:] or list(F.ORDER)
    ok, rows = F.check_bank()
    if not ok:
        raise SystemExit("bank check FAILED")
    dev_rows = [r for r in rows if r["set"] == "dev"]
    dev = {r["id"] for r in dev_rows}
    qt = {r["id"]: r["question"] for r in dev_rows}
    pair_of = {r["id"]: r["pair_id"] for r in dev_rows}
    prem = {r["id"]: r["premise"] for r in dev_rows}
    print("# FalseQA, the %d development questions only (%d pairs); the held-out questions are not read" % (
        len(dev), len(set(pair_of.values()))))
    print("# N1, SN1, Conly and SC have their caps chosen on these same questions (they lean upwards); intervals resample pairs")
    data, done, M = {}, [], {}
    for m in models:
        st = F.setting(m)
        try:
            R, R2 = F.load_rows(m, dev, works=DEV_WORK)
        except FileNotFoundError as err:
            print("# %s: development set not ready (%s)" % (m, str(err)[:200]))
            continue
        if sorted(R.ids) != sorted(dev) or sorted(R2.ids) != sorted(dev):
            raise SystemExit("%s: the loaded questions are not the development set" % m)
        caps, lines = F.choose(R, R2, st)
        print("# %s: E%d main %s, s_sel %s, C%d; chosen on the %d development questions:" % (m, st[0], st[1], st[3], st[2], len(R.ids)))
        for x in lines:
            print("    " + x)
        data[m] = F.systems_of(R, R2, st, caps, R.ids)
        F.declined_note(m, (R, R2), data[m], dev, "development")   # prints nothing when the judge declined no cell
        M[m] = F.margins_of(m)
        done.append(m)
    if not done:
        return
    ids = [r["id"] for r in dev_rows]
    groups = [("dev, premise false", [q for q in ids if prem[q] == "false"]),
              ("dev, premise valid", [q for q in ids if prem[q] == "valid"]), ("dev, all", ids)]
    F.readings(data, done, qt, pair_of, groups, M)
    print("\n" + "=" * 100 + "\n## dev, premise valid: every system minus the original model O (natural)")
    qs = groups[1][1]
    for m in done + ["all"]:
        ms = done if m == "all" else [m]
        out = []
        for n in F.SYS_ORDER:
            if n == "O":
                continue
            d = {q: sum(data[k][n][q][0] - data[k]["O"][q][0] for k in ms) for q in qs}
            out.append("%s %s" % (n, F.fmt_ci(F.pair_ci(d, pair_of, qs))))
        print("  %-12s %s" % (m if m != "all" else "%d models" % len(done), " | ".join(out)))


if __name__ == "__main__":
    main()
