# Iteration_01 analysis (Claude Opus 5.5, 2026-11-18), read-only, no model, no judge call.
# Why does the prompt baseline B (every question blank after E, re-asked once with the must-answer prompt and the same E)
# score above N1 on the fresh questions of Gemma-3-12B (-11 [-28, +6]) but below it on Gemma-3-27B (+19 [+7, +32])?
# Per question, on the candidates (blank under E), N1 minus B is split by provenance (own blank / E's blank) and by what
# N1 did (released at dose k / vetoed as the original answer / not released: above the cap or still blank at the cap).
# Settings are the ones chosen on tune, as in confirm_fresh.py. Pro verdicts. v = +1 right, -1 wrong, 0 otherwise.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/b_vs_n1.py
import collections
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import release_origin as ro
import confirm_fresh as cfr

V = {"R": 1, "W": -1}
CAPS = ro.CAPS


def v(x):
    return V.get(x, 0)


def raw_cells(work):
    out = {}
    for line in open(work + "/judge2/cells.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[(r["pair_id"], r["condition"])] = r
    return out


def n1_trace(sp, tx, cfg):
    """Per candidate: (origin, what N1 did, dose) with the same rule as release_origin.states."""
    kind, s, k1, k2 = cfg
    e = sp.cE(s)
    out = {}
    for p, origin, k, st, whose in ro.released(sp, tx, s):
        cap = k2 if origin == "own" else k1
        if cap is None or ps.LAD.index(k) > ps.LAD.index(cap):
            out[p] = (origin, "above cap", k)
        elif origin == "E's" and whose == "original":
            out[p] = (origin, "vetoed (original)", k)
        else:
            out[p] = (origin, "released", k)
    for p in sp.qs:
        if sp.blank[(p, e)] and p not in out:
            out[p] = ("own" if sp.blank[(p, "baseline_x")] else "E's", "blank at every dose", None)
    return out


def report(name, tune, full, ftx, sel, label, examples=0, ec_dir=None, d_dir=None):
    fam = [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    ttx = ro.texts(cf.MODELS[name]["ec"][0])
    pick = max(fam, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -fam.index(x)))
    fB = [("B", "1"), ("B", "2")]
    pB = max(fB, key=lambda x: (ps.score(tune.states(x), 1), -fB.index(x)))
    n1 = ro.states(full, ftx, pick)
    bb = full.states(pB)
    ids = [full.qs[i] for i in sel]
    pos = {p: i for i, p in enumerate(full.qs)}
    print("\n## %s, %s (%d questions): N1 = %s; prompt B = must-answer prompt with E %s" % (name, label, len(ids), ro.name_of(pick), pB[1]))
    print("  totals: N1 %d | B %d | N1 minus B %s" % (ps.score([n1[pos[p]] for p in ids], 1), ps.score([bb[pos[p]] for p in ids], 1),
                                                     ps.ci([v(n1[pos[p]]) - v(bb[pos[p]]) for p in ids])))
    if pick[0] != "N1":
        print("  N1 chose single E on tune; nothing more to split")
        return
    tr = n1_trace(full, ftx, pick)
    eB = full.cE(pB[1])
    groups = collections.defaultdict(list)
    for p in ids:
        if p in tr:
            o, what, k = tr[p]
            groups[(o, what)].append(p)
        elif full.blank[(p, eB)]:
            groups[("(blank only under B's E strength)", "-")].append(p)
    for key in sorted(groups, key=lambda x: (x[0], x[1])):
        ps_ = groups[key]
        d = sum(v(n1[pos[p]]) - v(bb[pos[p]]) for p in ps_)
        cn = collections.Counter(n1[pos[p]] for p in ps_)
        cb = collections.Counter(bb[pos[p]] for p in ps_)
        print("  %-5s %-22s n %3d | N1 R/W/other %3d/%3d/%3d | B R/W/other %3d/%3d/%3d | N1 minus B %+d" % (
            key[0], key[1], len(ps_), cn["R"], cn["W"], len(ps_) - cn["R"] - cn["W"], cb["R"], cb["W"], len(ps_) - cb["R"] - cb["W"], d))
    rest = [p for p in ids if p not in tr and not full.blank[(p, eB)]]
    print("  not a candidate: n %d | N1 minus B %+d (should be 0)" % (len(rest), sum(v(n1[pos[p]]) - v(bb[pos[p]]) for p in rest)))
    if examples and ec_dir and d_dir:
        ec, dd = raw_cells(ec_dir), raw_cells(d_dir)
        shown = 0
        print("  examples: E's blanks where B is right and N1 is not")
        for p in ids:
            if p in tr and tr[p][0] == "E's" and bb[pos[p]] == "R" and n1[pos[p]] != "R" and shown < examples:
                o, what, k = tr[p]
                s = pick[1]
                rel = ec.get((p, full.cEC(s, k)), {}).get("prediction") if k else None
                print("   - %s | %s at dose %s | no write: %r | N1 (%s): %r | B (%s): %r" % (
                    p, what, k, (ec.get((p, "baseline_x"), {}).get("prediction") or "")[:70], n1[pos[p]], (rel or "")[:70],
                    bb[pos[p]], (dd.get((p, full.dE(pB[1])), {}).get("prediction") or "")[:70]))
                shown += 1


def main():
    for name, (ec_full, d_full) in cfr.FULL.items():
        m = cf.MODELS[name]
        tune = ps.Split(m, 0)
        old = set(ps.Split(m, 1).qs)
        full = ps.Split(dict(m, ec=(m["ec"][0], ec_full), d=(m["d"][0], d_full), judge=(m["judge"][0], "pro")), 1)
        ftx = ro.texts(ec_full)
        fresh = [i for i, p in enumerate(full.qs) if p not in old]
        report(name, tune, full, ftx, fresh, "fresh", examples=8, ec_dir=ec_full, d_dir=d_full)
        report(name, tune, full, ftx, list(range(len(full.qs))), "all")
        tx0 = ro.texts(m["ec"][0])
        report(name, tune, tune, tx0, list(range(len(tune.qs))), "tune")
    for name in ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B"):
        m = cf.MODELS[name]
        tune, test = ps.Split(m, 0), ps.Split(m, 1)
        report(name, tune, test, ro.texts(m["ec"][1]), list(range(len(test.qs))), "test")


if __name__ == "__main__":
    main()
