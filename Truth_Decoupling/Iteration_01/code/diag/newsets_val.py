# Iteration_01 analysis (Claude Opus 5.5, 2026-11-20), read-only, no model, no judge call.
# Validation on fresh questions of N1 with its caps re-chosen on the new tune split
# (novelty/ours/2026-11-20_rechosen_caps_validation.md). Fresh split "test2" (500 per dataset, the Pro 100 inside),
# drawn from the questions no earlier split uses. Systems: no write, single E (E 2), N1 with the frozen re-chosen caps
# (configs/newsets/rechosen_caps_20261120.json); alias only: N1 with the frozen TruthfulQA caps, for reference.
# Records through diag/final_records.py unchanged; Pro votes through scripts/newsets/newsets_votes.py (one vote per cell,
# bound to the current answers); blank flags of every C strength come from the generation files (the ladder looks at
# them; only the strengths the caps allow are judged).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/newsets_val.py
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import final_records as fr    # noqa: E402
import alias_match as AM      # noqa: E402
import newsets_votes as NV    # noqa: E402

NS = S.NS
CAPS = "Iteration_01/code/configs/newsets/rechosen_caps_20261120.json"
LADNAME = {None: None, 0.25: "0-25", 0.5: "0-5", 1: "1", 2: "2"}


def bank2(task):
    p = "%s/bank/%s_bank_test2_20261120.jsonl" % (NS, task)
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return {r["id"]: r for r in rows}, rows, p


def gens(task, model, m):
    """{condition: {pair_id: row}} of the six test2 arms, each a complete full pass over the test2 split."""
    b, rows, bp = bank2(task)
    want = [r["id"] for r in rows]
    out = {}
    e, c = "E%d.w.p2" % m["el"], "C%d.w" % m["cl"]
    for arm in ["x", "x:" + e] + ["x:%s+%s.m%s" % (e, c, k) for k in ("0.25", "0.5", "1", "2")]:
        p = "%s/gen/%s/%s/test2/%s.json" % (NS, task, model, S.tok(arm))
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        k = g.get("contract") or {}
        if not (g.get("completed") and k.get("full_pass") and k.get("split") == "test2" and k.get("model") == model
                and k.get("items_file_sha256") == S.sha_file(bp) and [r["id"] for r in g["rows"]] == want):
            raise SystemExit("%s is not a complete test2 pass of %s" % (p, model))
        out[k["condition"]] = {"%s:%s" % (task, r["id"]): r for r in g["rows"]}
    return out, b


def main():
    M = S.models()
    with open(CAPS, encoding="utf-8") as f:
        caps = json.load(f)["caps"]
    print("Validation on fresh questions (test2): N1 with caps re-chosen on the new tune split; Pro on the fixed 100, alias on 500")
    for task in S.TASKS:
        per = {"pro": {}, "alias": {}}
        for name in S.ORDER:
            m = M[name]
            sh = S.shell(m["el"], m["cl"])
            need = ["baseline_x", sh.cE("2"), sh.cEC("2", "0-25")]
            g, b = gens(task, name, m)
            k1, k2 = LADNAME[caps[task][name]["e_blank"]], LADNAME[caps[task][name]["own_blank"]]
            frozen = (LADNAME[m["caps"]["e_blank"]], LADNAME[m["caps"]["own_blank"]])
            blank_all = {(p, c): bool(r["rule_abstain"]) or not r["text"].strip() for c, rows in g.items() for p, r in rows.items()}
            for judge in ("pro", "alias"):
                if judge == "pro":
                    w = "%s/judge/%s/%s/test2_pro100" % (NS, task, name)
                    table, cnt, meta = NV.load_votes(w, "%s/gen/%s/%s/test2" % (NS, task, name))
                    want = ["%s:%s" % (task, i) for i, r in b.items() if r["pro100"]]
                    if {(p, c) for p, c in table} != {(p, c) for p in want for c in need}:
                        raise SystemExit("%s does not judge exactly the Pro 100 x the three conditions" % w)
                    st = {k: v["state"] for k, v in table.items()}
                    tx = {k: v["text"] for k, v in table.items()}
                    qs = [p for p in want if all(st[(p, c)] != "?" for c in need)]
                    for c, rows in g.items():          # texts of every strength (the ladder and the veto read them)
                        for p, r in rows.items():
                            tx.setdefault((p, c), r["text"])
                else:
                    st, tx = {}, {}
                    for c, rows in g.items():
                        for p, r in rows.items():
                            iid = p.split(":", 1)[1]
                            st[(p, c)] = AM.verdict(r["text"], b[iid]["gold"], not blank_all[(p, c)])
                            tx[(p, c)] = r["text"]
                    qs = sorted(g["baseline_x"])
                    want = qs
                sp = S.NewSplit(m["el"], m["cl"], st, blank_all, need, qs=qs)
                t = fr.Texts(tx, tx)
                s0 = S.system(sp, t, ("base",))
                sE = S.system(sp, t, ("E", "2"))
                sN = S.system(sp, t, ("N1", "2", k1, k2))
                print("\n== %s %s, %s: %d of %d questions; re-chosen caps E's %s / own %s" % (task, name, judge, len(sp.qs), len(want), k1, k2))
                for lab, (s, a, _) in (("no write", s0), ("single E (E 2)", sE), ("N1, re-chosen caps", sN)):
                    print(S.line(lab, S.counts(s, a)))
                if judge == "alias":
                    sF = S.system(sp, t, ("N1", "2") + frozen)
                    print(S.line("N1, frozen TruthfulQA caps (E's %s / own %s), reference" % frozen, S.counts(sF[0], sF[1])))
                d0 = [x - y for x, y in zip(S.vals(sN[0]), S.vals(s0[0]))]
                dE = [x - y for x, y in zip(S.vals(sN[0]), S.vals(sE[0]))]
                e0 = [x - y for x, y in zip(S.vals(sE[0]), S.vals(s0[0]))]
                print("    N1 re-chosen minus no write %s | minus single E %s | single E minus no write %s" % (
                    S.ps.ci(d0), S.ps.ci(dE), S.ps.ci(e0)))
                per[judge][name] = (dict(zip(sp.qs, d0)), dict(zip(sp.qs, dE)), dict(zip(sp.qs, e0)))
        print("\n== %s: five-model totals, questions resampled jointly" % task)
        for j in ("pro", "alias"):
            common = sorted(set.intersection(*[set(v[0]) for v in per[j].values()]))
            for k, lab in ((0, "N1 re-chosen minus no write"), (1, "N1 re-chosen minus single E"), (2, "single E minus no write")):
                vals = [[per[j][n][k][q] for q in common] for n in S.ORDER]
                tot, lo, hi = S.joint_ci(vals)
                pos = sum(sum(v) > 0 for v in vals)
                print("    %-5s %-28s %+d [%+d, %+d] on %d questions, %d of 5 models positive" % (j, lab, tot, lo, hi, len(common), pos))


if __name__ == "__main__":
    main()
