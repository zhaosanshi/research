# Iteration_01 analysis (Claude Opus 5.5, 2026-11-18; revised 2026-11-19), read-only, no model, no judge call.
# Why does the prompt baseline B (every question blank after E, re-asked once with the must-answer prompt and the same E)
# score above N1 on the fresh questions of Gemma-3-12B (-11 [-28, +6]) but below it on Gemma-3-27B (+19 [+7, +32])?
# Per question, N1 minus B is split by what each side did (final_records.py): N1's provenance and treatment (released at
# dose k / vetoed as the original answer / above the cap / blank at every dose / not re-asked / not a candidate) against
# B's (re-asked / not a candidate). The groups add up to the total (checked). Settings are the ones chosen on tune, as in
# confirm_fresh.py. Pro verdicts. v = +1 right, -1 wrong, 0 otherwise.
# 2026-11-19 (GPT 2026-11-18, discussion/novelty_review/2026-11-18_gpt_results_recheck.md section 4): the examples and the
# groups now come from the per-question records that the scores are computed from. The first version printed, for a vetoed
# or above-cap question, the C answer N1 never handed in; and it took the questions that are candidates on one side only
# (possible when N1 and B choose different E strengths) as "should be 0".
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/b_vs_n1.py
import collections
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import release_origin as ro
import confirm_fresh as cfr
import final_records as fr

V = {"R": 1, "W": -1}
CAPS = ro.CAPS


def v(x):
    return V.get(x, 0)


def side_b(r):
    return "B re-asks" if r["note"] == "re-asked with the must-answer prompt" else "B: not a candidate"


def report(name, tune, full, ftx, sel, label, examples=0):
    fam = [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    ttx = ro.texts(cf.MODELS[name]["ec"][0])
    pick = max(fam, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -fam.index(x)))
    fB = [("B", "1"), ("B", "2")]
    pB = max(fB, key=lambda x: (ps.score(tune.states(x), 1), -fB.index(x)))
    rn, rb = fr.records(full, pick, ftx), fr.records(full, pB, ftx)
    ids = [full.qs[i] for i in sel]
    pos = {p: i for i, p in enumerate(full.qs)}
    tot = [v(rn[pos[p]]["state"]) - v(rb[pos[p]]["state"]) for p in ids]
    print("\n## %s, %s (%d questions): N1 = %s; prompt B = must-answer prompt with E %s" % (name, label, len(ids), ro.name_of(pick), pB[1]))
    print("  totals: N1 %d | B %d | N1 minus B %s" % (ps.score([rn[pos[p]]["state"] for p in ids], 1),
                                                     ps.score([rb[pos[p]]["state"] for p in ids], 1), ps.ci(tot)))
    groups = collections.defaultdict(list)
    for p in ids:
        a, b = rn[pos[p]], rb[pos[p]]
        groups[(a["cand"] or "-", a["note"], side_b(b))].append(p)
    order = ["released", "vetoed: the original answer came back", "above the cap", "blank at every dose", "not re-asked", "E alone", "not a candidate"]
    gsum = 0
    for key in sorted(groups, key=lambda x: (x[0], order.index(x[1]) if x[1] in order else 99, x[2])):
        ps_ = groups[key]
        d = sum(v(rn[pos[p]]["state"]) - v(rb[pos[p]]["state"]) for p in ps_)
        gsum += d
        cn = collections.Counter(rn[pos[p]]["state"] for p in ps_)
        cb = collections.Counter(rb[pos[p]]["state"] for p in ps_)
        print("  %-5s %-38s %-19s n %3d | N1 R/W/other %3d/%3d/%3d | B R/W/other %3d/%3d/%3d | N1 minus B %+d" % (
            key[0], key[1], key[2], len(ps_), cn["R"], cn["W"], len(ps_) - cn["R"] - cn["W"], cb["R"], cb["W"], len(ps_) - cb["R"] - cb["W"], d))
    if gsum != sum(tot):
        raise SystemExit("the groups add up to %+d, the total is %+d" % (gsum, sum(tot)))
    print("  the groups add up to the total (%+d)" % gsum)
    if examples:
        shown = 0
        print("  examples, in question order: E's blanks where B is right and N1 is not (the answers each side handed in)")
        for p in ids:
            a, b = rn[pos[p]], rb[pos[p]]
            if a["cand"] == "E's" and b["state"] == "R" and a["state"] != "R" and shown < examples:
                print("   - %s | N1: %s%s | no write: %r | N1 hands in (%s): %r | B hands in (%s): %r" % (
                    p, a["note"], (" at dose %s" % ps.NAME[a["rung"]]) if a["rung"] else "", (ftx.ec.get((p, "baseline_x")) or "")[:60],
                    a["state"], (a["text"] or "")[:60], b["state"], (b["text"] or "")[:60]))
                shown += 1


def main():
    for name, (ec_full, d_full) in cfr.FULL.items():
        m = cf.MODELS[name]
        tune = ps.Split(m, 0)
        old = set(ps.Split(m, 1).qs)
        mf = dict(m, ec=(m["ec"][0], ec_full), d=(m["d"][0], d_full), judge=(m["judge"][0], "pro"))
        full = ps.Split(mf, 1)
        ftx = fr.Texts.of(mf, 1)
        fresh = [i for i, p in enumerate(full.qs) if p not in old]
        report(name, tune, full, ftx, fresh, "fresh", examples=8)
        report(name, tune, full, ftx, list(range(len(full.qs))), "all")
        report(name, tune, tune, fr.Texts.of(m, 0), list(range(len(tune.qs))), "tune")
    for name in ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B"):
        m = cf.MODELS[name]
        tune, test = ps.Split(m, 0), ps.Split(m, 1)
        report(name, tune, test, fr.Texts.of(m, 1), list(range(len(test.qs))), "test")


if __name__ == "__main__":
    main()
