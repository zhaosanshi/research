# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, test split, DeepSeek Pro, score = right - wrong: N1 at every E strength a model has, against the
# comparators of the same-answer-count main table AT THE SAME STRENGTH:
#   strong form   must-answer prompt + E at that strength, keeping its N most confident answers (N = N1's answers,
#                 confidence = mean log-probability of the model's own answer tokens);
#   prompt B      E at that strength; its blanks asked again with the must-answer prompt (+ E at that strength);
#   must-answer + E at that strength with every question answered, also against the prompt-only baseline.
# Strengths 1 and 2 use the existing must-answer work dirs and confidence files (they must reproduce
# diag/dose_table.py); the other strengths use the work dir judge_test_dmust_test and the files
# eval_extra/confidence_pmmid_test.json / confidence_pmhigh_test.json / confidence_pmhigh2_test.json of
# logs/newsets_20261119/pm2_all.sh (pm3_gemma.sh for the reruns). A confidence row is used only if its text hash equals
# the hash of the judged answer. N1's caps are chosen on tune per strength, as in diag/dose_mid.py.
# Several models together: per question the differences are summed and questions are resampled; a strength enters a
# total only if every model has it.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/strong_mid.py [MODEL ...]
import hashlib
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import dose_mid as DM

MT, S = DM.MT, DM.S
ps, ro = MT.ps, MT.ro
CAPS = MT.CAPS
CONF_FILES = (("pmmid_test", ("1-25", "1-5", "1-75")), ("pmhigh_test", ("2-5", "3")), ("pmhigh2_test", ("3-5", "4")))
MAIN = DM.MAIN


def new_must(run, sp, doses):
    """({strength: (states, answered)}, {strength: confidence rows}) for the strengths other than 1 / 2 that are complete."""
    w = run + "/eval_extra/judge_test_dmust_test"
    out, conf = {}, {}
    if not os.path.exists(w + "/judge2/single_pro.jsonl"):
        return out, conf, "no must-answer verdicts at other strengths"
    st, bl = ps.load(w, "pro")
    text = {}
    with open(w + "/judge2/cells.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text[(r["pair_id"], r["condition"])] = r["prediction"]
    notes = []
    for tag, ds in CONF_FILES:
        p = "%s/eval_extra/confidence_%s.json" % (run, tag)
        d = None
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if d["split"] != "test" or d.get("partial"):
                raise SystemExit("%s is not a complete test file" % p)
        for s in ds:
            if s not in doses:
                continue
            cond = sp.dE(s)
            if any((q, cond) not in st for q in sp.qs):
                notes.append("E %s: no must-answer answers" % DM.show(s))
                continue
            out[s] = ([st[(q, cond)] for q in sp.qs], [not bl[(q, cond)] for q in sp.qs])
            target = "conf_E%s" % DM.show(s)
            if d is None or target not in d["targets"]:
                notes.append("E %s: no confidence file" % DM.show(s))
                continue
            rows = {r["pair_id"]: r for r in d["targets"][target]["rows"]}
            bad = [q for q in sp.qs if q not in rows or hashlib.sha1(text[(q, cond)].encode("utf-8")).hexdigest() != rows[q]["text_sha1"]]
            if bad:
                raise SystemExit("confidence rows of %s do not belong to the judged answers: %d questions, e.g. %s" % (target, len(bad), bad[:3]))
            conf[s] = rows
    return out, conf, "; ".join(notes)


def one(name, tots, table):
    Dt, tsp, ttx, tdoses, _, _ = DM.merged_all(name, 0)
    Dx, xsp, xtx, doses, _, _ = DM.merged_all(name, 1)
    qs = xsp.qs
    run = Dx.m["ec"][1].split("/eval_extra/")[0]
    must_new, conf_new, note = new_must(run, xsp, doses)
    da0 = MT.handed_in(Dx, ("DA",))
    da = dict(zip(Dx.sp.qs, da0[0]))
    da_states = [da[q] for q in qs]
    print("\n## %s: test %d questions%s" % (name, len(qs), "" if not note else " (" + note + ")"))

    def add(key, v):
        for t in tots:
            n, d = t.setdefault(key, [0, {}])
            t[key][0] = n + 1
            for q, x in zip(qs, v):
                d[q] = d.get(q, 0) + x

    for s in doses:
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda x: (ps.score(S.system(tsp, ttx, x)[0], 1), -cands.index(x)))
        n1 = S.system(xsp, xtx, pick)
        e = S.system(xsp, xtx, ("E", s))
        N = sum(n1[1])
        if s in ("1", "2"):
            de0 = MT.must(Dx, Dx.sp.dE(s))                  # on the old question list; a superset of qs
            by = dict(zip(Dx.sp.qs, zip(de0[0], de0[1])))
            de = ([by[q][0] for q in qs], [by[q][1] for q in qs])
            rows = Dx.rows("conf_E%s" % s, Dx.m["d"][1], Dx.sp.dE(s))
        elif s in must_new:
            de, rows = must_new[s], conf_new.get(s)
        else:
            print("    E %-4s N1 answers %3d net %4d | no must-answer + E answers at this strength" % (DM.show(s), N, sum(MT.val(n1[0]))))
            continue
        eblank = [xsp.blank[(q, xsp.cE(s))] for q in qs]
        b = [de[0][i] if eblank[i] else e[0][i] for i in range(len(qs))]
        d_b = [x - y for x, y in zip(MT.val(n1[0]), MT.val(b))]
        d_all = [x - y for x, y in zip(MT.val(n1[0]), MT.val(de[0]))]
        d_de_da = [x - y for x, y in zip(MT.val(de[0]), MT.val(da_states))]
        msg = "    E %-4s N1 answers %3d net %4d | must-answer + E: answers %3d net %4d" % (
            DM.show(s), N, sum(MT.val(n1[0])), sum(de[1]), sum(MT.val(de[0])))
        strong = None
        if rows is not None:
            order = sorted([i for i, a in enumerate(de[1]) if a], key=lambda i: (-rows[qs[i]]["mean_logp"], i))
            if len(order) >= N:
                strong = MT.keep(de[0], order, N)[0]
                d_s = [x - y for x, y in zip(MT.val(n1[0]), MT.val(strong))]
                msg += " | its top %d: net %4d | N1 minus strong form %s" % (N, sum(MT.val(strong)), ps.ci(d_s))
                add("E %s: N1 minus the strong form at the same strength" % DM.show(s), d_s)
                table.setdefault(name, {})[s] = ps.ci(d_s)
            else:
                msg += " | strong form: only %d answers to choose from (fewer than N)" % len(order)
        else:
            msg += " | strong form: no confidence file"
        print(msg)
        print("           N1 minus prompt B %s | N1 minus must-answer + E (all answered) %s | must-answer + E minus prompt only %s" % (
            ps.ci(d_b), ps.ci(d_all), ps.ci(d_de_da)))
        add("E %s: N1 minus prompt B at the same strength" % DM.show(s), d_b)
        add("E %s: must-answer + E minus prompt only" % DM.show(s), d_de_da)


if __name__ == "__main__":
    names = sys.argv[1:] or list(ro.FIVE)
    tot, tot3, table = {}, {}, {}
    for n in names:
        one(n, (tot, tot3) if n in MAIN else (tot,), table)
    print("\n## N1 minus the strong form at the same strength, by model")
    for n in names:
        print("  %-14s %s" % (n, " | ".join("E %s: %s" % (DM.show(s), v) for s, v in table.get(n, {}).items())))
    for title, t, k in (("the models above together", tot, len(names)), ("the main-text models among them", tot3, sum(n in MAIN for n in names))):
        print("\n## %s (per question the differences of the models are summed; questions resampled)" % title)
        for key, (n_models, d) in t.items():
            if n_models == k:
                print("  %-58s %s (%d questions)" % (key, ps.ci([d[q] for q in sorted(d)]), len(d)))
