# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call. EXPLORATION PILOT.
# TriviaQA / PopQA with a user belief appended to every question (launcher explore_20261120/pressure_pilot.sh, bank
# scripts/newsets/newsets_pressure_bank.py; the sentence of Sharma et al., ICLR 2024, section 3.3). Tune split only.
# The one question of the pilot: on such questions, are the answers that C releases right more often than wrong?
# Printed per model and task (alias match):
#   the plain question against the question with the sentence, nothing written: how many right answers survive;
#   how many answers judged right contain a negation (alias matching counts "it is not X" as right when X is gold);
#   single C 0.25; for every E strength with its five passes: single E, what E silenced, what the ladder releases
#   (the model's original answer coming back / a new answer; right / wrong), the model's own blanks released, and the
#   best tune setting of N1 and of N1 without the veto (final_records kinds N1 and T2) next to single E.
# All numbers are on the tune split itself: descriptive, nothing is confirmed here.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ns_pressure_pilot.py VARIANT [MODEL ...]
import json
import os
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import final_records as fr    # noqa: E402
import alias_match as AM      # noqa: E402
import release_origin as ro   # noqa: E402

NS = S.NS
ES = ("0.5", "1", "1.5", "2", "3", "4")
KS = ("0.25", "0.5", "1", "2")
CAPS = [None, "0-25", "0-5", "1", "2"]
LAD = fr.LAD
NEG = re.compile(r"\b(not|no|never|none|nobody|nothing)\b|n't", re.I)


def read_pass(path, bank, sha, split, model, arm):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    k = g.get("contract") or {}
    if not (g.get("completed") and k.get("full_pass") and k.get("split") == split and k.get("model") == model and k.get("arm") == arm
            and k.get("items_file_sha256") == sha and [r["id"] for r in g["rows"]] == list(bank)):
        raise SystemExit("%s is not a complete %s pass of %s for arm %s over this bank" % (path, split, model, arm))
    return k["condition"], g["rows"]


def load(variant, task, model, m):
    bp = "%s/bank/%s_pressure_%s_bank_20261121.jsonl" % (NS, task, variant)
    if not os.path.exists(bp):
        raise SystemExit("no bank %s" % bp)
    with open(bp, encoding="utf-8") as f:
        bank = {r["id"]: r for r in map(json.loads, f) if r["split"] == "tune"}
    sha = S.sha_file(bp)
    root = "%s/gen_pressure_%s/%s/%s/tune" % (NS, variant, task, model)
    st, bl, tx, have, need = {}, {}, {}, [], []

    def take(arm):
        got = read_pass("%s/%s.json" % (root, S.tok(arm)), bank, sha, "tune", model, arm)
        if got is None:
            return None
        c, rows = got
        for r in rows:
            q = "%s:%s" % (task, r["id"])
            blank = bool(r["rule_abstain"]) or not r["text"].strip()
            st[(q, c)] = AM.verdict(r["text"], bank[r["id"]]["gold"], not blank)
            bl[(q, c)] = blank
            tx[(q, c)] = r["text"]
        return c

    if take("x") is None:
        raise SystemExit("no no-write pass under %s" % root)
    need.append("baseline_x")
    c_alone = take("x:C%d.w.m0.25" % m["cl"])
    for s in ES:
        got = [take(a) for a in ["x:E%d.w.p%s" % (m["el"], s)] + ["x:E%d.w.p%s+C%d.w.m%s" % (m["el"], s, m["cl"], k) for k in KS]]
        if all(got):
            have.append(s.replace(".", "-"))
            need += got
    qs = ["%s:%s" % (task, i) for i in bank]
    sp = S.NewSplit(m["el"], m["cl"], st, bl, need, qs=qs)
    # the plain question, nothing written, same questions (first bank)
    pb = "%s/bank/%s_bank_20261119.jsonl" % (NS, task)
    with open(pb, encoding="utf-8") as f:
        plain_bank = {r["id"]: r for r in map(json.loads, f) if r["split"] == "tune"}
    if list(plain_bank) != list(bank):
        raise SystemExit("the two banks do not list the same tune questions in the same order")
    got = read_pass("%s/gen/%s/%s/tune/x.json" % (NS, task, model), plain_bank, S.sha_file(pb), "tune", model, "x")
    plain = None
    if got:
        plain = {}
        for r in got[1]:
            blank = bool(r["rule_abstain"]) or not r["text"].strip()
            plain["%s:%s" % (task, r["id"])] = "X" if blank else AM.verdict(r["text"], plain_bank[r["id"]]["gold"], True)
    return sp, fr.Texts(tx, tx), have, c_alone, plain


def net(v):
    return sum(S.vals(v))


def cnt(v):
    return "%+4d (R %3d W %3d blank %3d)" % (net(v), v.count("R"), v.count("W"), len(v) - v.count("R") - v.count("W"))


def main():
    variant = sys.argv[1]
    names = sys.argv[2:] or list(S.ORDER)
    M = S.models()
    for task in S.TASKS:
        for name in names:
            try:
                sp, tx, have, c_alone, plain = load(variant, task, name, M[name])
            except SystemExit as e:
                print("\n## %s %s: %s" % (name, task, e))
                continue
            qs = sp.qs
            cur = [sp.st[(q, "baseline_x")] if not sp.blank[(q, "baseline_x")] else "X" for q in qs]
            print("\n## %s %s, variant %s, tune %d questions" % (name, task, variant, len(qs)))
            if plain:
                p = [plain[q] for q in qs]
                print("    plain question, no write:      %s" % cnt(p))
                moved = lambda a, b: sum(x == a and y == b for x, y in zip(p, cur))
                print("    with the sentence, no write:   %s | of the %d plain-right answers: still right %d, now wrong %d, now blank %d | plain-wrong now right %d" % (
                    cnt(cur), p.count("R"), moved("R", "R"), moved("R", "W"), moved("R", "X"), moved("W", "R")))
            else:
                print("    with the sentence, no write:   %s" % cnt(cur))
            negs = sum(1 for q, s in zip(qs, cur) if s == "R" and NEG.search(tx.ec[(q, "baseline_x")]))
            print("    answers judged right that contain a negation word (possible alias false positives): %d of %d" % (negs, cur.count("R")))
            if c_alone:
                c = [sp.st[(q, c_alone)] if not sp.blank[(q, c_alone)] else "X" for q in qs]
                print("    single C 0.25:                 %s" % cnt(c))
            for s in have:
                e = S.system(sp, tx, ("E", s))[0]
                e = [x if not sp.blank[(q, sp.cE(s))] else "X" for q, x in zip(qs, e)]
                line = "    E %-4s single E %s" % (s.replace("-", "."), cnt(e))
                for cand in ("E's", "own"):
                    same, new, orig, never = {"R": 0, "W": 0}, {"R": 0, "W": 0}, {"R": 0, "W": 0}, 0
                    for q in qs:
                        if fr.cand_of(sp, q, s) != cand:
                            continue
                        o = sp.st[(q, "baseline_x")]
                        if o in orig:
                            orig[o] += 1
                        k = fr.first_answer(sp, q, s, LAD)
                        if k is None:
                            never += 1
                            continue
                        c = sp.cEC(s, k)
                        d = same if ro.norm(tx.ec[(q, "baseline_x")]) == ro.norm(tx.ec[(q, c)]) else new
                        v = sp.st[(q, c)]
                        if v in d:
                            d[v] += 1
                    if cand == "E's":
                        line += "\n           E silenced (orig R %d W %d): released SAME R %d W %d | NEW R %d W %d | never %d" % (
                            orig["R"], orig["W"], same["R"], same["W"], new["R"], new["W"], never)
                    else:
                        line += " || own blanks released: R %d W %d, never %d" % (new["R"] + same["R"], new["W"] + same["W"], never)
                best = {}
                for kind in ("N1", "T2"):
                    cands = [(kind, s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
                    b = max(cands, key=lambda c: (net(S.system(sp, tx, c)[0]), -cands.index(c)))
                    best[kind] = (b, net(S.system(sp, tx, b)[0]))
                line += "\n           best on tune: N1 caps %s / %s net %+d | without the veto caps %s / %s net %+d | single E net %+d" % (
                    best["N1"][0][2], best["N1"][0][3], best["N1"][1], best["T2"][0][2], best["T2"][0][3], best["T2"][1], net(e))
                print(line)


if __name__ == "__main__":
    main()
