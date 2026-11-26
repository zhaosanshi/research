#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: FalseQA held-out 600 questions, the systems of diag/falseqa_read.py as they are (caps
# chosen on the 100 development questions by that program), their existing DeepSeek Pro verdicts and the existing
# batch-1 self-judge margins (no write). No card, no judge call, nothing is chosen here.
# Rule Z of 2026-11-22 (selfjudge_stage1.py), unchanged: an answered cell whose self-judge margin is below 0 is not
# handed in; answers that open with a negated premise (selfjudge_probe.NEG_RE) are exempt; threshold 0, not tuned.
# Applied to four systems: O (original), E (single E), N1, Conly (single C on the original's blanks).
# Readings per model and question group (premise false / valid / all), natural answer counts, pairs resampled:
#   N1+Z - N1;  N1 - E and N1+Z - E+Z;  N1 - Conly and N1+Z - Conly+Z;  E+Z - E;  what rule Z drops in each system.
# The readings are after-the-fact: the held-out verdicts had been read on 2026-11-24/25 for other comparisons.
# usage (container, from the project root): python3 -B Iteration_01/diag/falseqa_ruleZ_1126.py
import collections
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import falseqa_read as F          # noqa: E402
import selfjudge_probe as SP      # noqa: E402

NAMES = ("O", "E", "N1", "Conly")


def main():
    ok, rows = F.check_bank()
    if not ok:
        raise SystemExit("bank check FAILED")
    dev = {r["id"] for r in rows if r["set"] == "dev"}
    held = {r["id"] for r in rows if r["set"] == "heldout"}
    qt = {r["id"]: r["question"] for r in rows}
    pair_of = {r["id"]: r["pair_id"] for r in rows}
    prem = {r["id"]: r["premise"] for r in rows}
    ids = [r["id"] for r in rows if r["set"] == "heldout"]
    groups = [("premise false", [q for q in ids if prem[q] == "false"]), ("premise valid", [q for q in ids if prem[q] == "valid"]),
              ("all", ids)]
    total = collections.defaultdict(lambda: {q: 0 for q in ids})
    for m in F.ORDER:
        st = F.setting(m)
        R, R2 = F.load_rows(m, dev)
        caps, _ = F.choose(R, R2, st)
        H, H2 = F.load_rows(m, held)
        d = F.systems_of(H, H2, st, caps, H.ids)
        M = F.margins_of(m)
        val, zval, drops, missing = {}, {}, {}, {}
        for name in NAMES:
            val[name] = {q: d[name][q][0] for q in ids}
            zv, dr, miss = {}, collections.Counter(), 0
            for q in ids:
                c = d[name][q]
                zv[q] = c[0]
                if not c[1]:
                    continue
                mg = F.margin(M, qt[q], c)
                if mg is None:
                    miss += 1
                elif mg < 0 and not SP.NEG_RE.match(c[2] or ""):
                    zv[q] = 0
                    kind = "first" if (name != "N1" and name != "Conly") or c[3] in (None, "0") else "released"
                    dr[(prem[q], kind, c[0])] += 1
            zval[name], drops[name], missing[name] = zv, dr, miss
        print("## %s: E%d strength %s, C%d; caps chosen on the development questions: N1 %s, Conly %s" % (
            m, st[0], st[1], st[2], caps["N1"], caps["Conly"]))
        print("   answered cells without a self-judge margin (kept): " + ", ".join("%s %d" % (n, missing[n]) for n in NAMES))
        for label, qs in groups:
            net = lambda v: sum(v[q] for q in qs)
            ans = lambda n: sum(1 for q in qs if d[n][q][1])
            zans = lambda n: sum(1 for q in qs if d[n][q][1] and zval[n][q] == val[n][q] or (d[n][q][1] and val[n][q] == 0 and zval[n][q] == 0 and False))
            print("   %-14s (%d questions) nets: " % (label, len(qs)) + " | ".join("%s %+d (%d answers), +Z %+d" % (n, net(val[n]), ans(n), net(zval[n])) for n in NAMES))
            def diff(a, b):
                return F.fmt_ci(F.pair_ci({q: a[q] - b[q] for q in qs}, pair_of, qs))
            print("       N1+Z - N1 %s | E+Z - E %s | N1 - E %s | N1+Z - E+Z %s | N1 - Conly %s | N1+Z - Conly+Z %s | N1+Z - O %s" % (
                diff(zval["N1"], val["N1"]), diff(zval["E"], val["E"]), diff(val["N1"], val["E"]), diff(zval["N1"], zval["E"]),
                diff(val["N1"], val["Conly"]), diff(zval["N1"], zval["Conly"]), diff(zval["N1"], val["O"])))
            if label == "all":
                for q in qs:
                    for key, v in (("N1z-N1", zval["N1"][q] - val["N1"][q]), ("N1-E", val["N1"][q] - val["E"][q]), ("N1z-Ez", zval["N1"][q] - zval["E"][q]),
                                   ("N1-C", val["N1"][q] - val["Conly"][q]), ("N1z-Cz", zval["N1"][q] - zval["Conly"][q])):
                        total[key][q] += v
        for name in NAMES:
            dr = drops[name]
            parts = []
            for p in ("false", "valid"):
                for kind in ("first", "released"):
                    r, w, o = dr[(p, kind, 1)], dr[(p, kind, -1)], dr[(p, kind, 0)]
                    if r or w or o:
                        parts.append("premise %s, %s answers: right %d, wrong %d, other %d" % (p, kind, r, w, o))
            print("   rule Z drops in %-5s: %s" % (name, "; ".join(parts) if parts else "none"))
    print("## five models together (per question sums; reference only), all 600 questions")
    for key in ("N1z-N1", "N1-E", "N1z-Ez", "N1-C", "N1z-Cz"):
        print("   %-8s %s" % (key, F.fmt_ci(F.pair_ci(total[key], pair_of, ids))))


if __name__ == "__main__":
    main()
