# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call. Exploration.
# TriviaQA / PopQA at the write rule "prompt", E at the main-table layer (answers in newsets/gen_prompt, launchers
# explore_20261120/ns_prompt.sh and nsc_prompt.sh). The re-ask toward answering cannot add anything on these questions
# (summary section 12.2); this table reads the two other uses of the same two directions that did add something at the
# training rule (sections 12.3, 12.4), now at the rule the method would use:
#   A. the C strength with the cautious sign: E + C toward caution written on every question in one pass. The family
#      contains "no C" (plain single E), so it can only differ from single E where the tune split prefers some caution.
#   B. E as a check: answer once without a write and once with E; keep the no-write answer only where both passes give
#      the same answer (release_origin.norm), otherwise leave the question blank. B2: the check pass is E + C toward
#      caution (or plain E).
#   G. the mirror image of N1's re-ask: answer with E; an answered question is asked once more with E + C toward
#      caution, and E's answer is kept only if that pass gives the same answer (blanks stay blank). "No challenge"
#      (plain single E) is in the family.
#   U. all of the above as one family, together with single C toward caution and a check of the no-write answer with
#      C toward caution alone: whatever the tune split likes best (ties to the simpler candidate: single E, single C,
#      B, the check with C alone, G, A, B2). By construction U is at least as good on tune as either single direction.
# Every setting of every system is chosen on the dataset's tune split (192 questions) by net score = right - wrong
# (alias match), ties to the earlier candidate (lower strengths); the chosen settings are then read on test and test2.
# Comparators, each at its own best tune setting: no write, single E, single C toward answering, single C toward
# caution, single C of either sign.
# Modes:
#   --choose MODEL          prints "TASK ARM ARM ..." = the write arms the chosen settings need (for the launcher)
#   [--models a,b] [--splits test,test2] [TASK ...]    the tables
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/nsc_prompt_table.py --models llama3.1-8b
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import alias_match as AM      # noqa: E402
import release_origin as ro   # noqa: E402
import ns_prompt_table as T   # noqa: E402

NS = S.NS
KS = ("0.25", "0.5", "1", "2")
MAIN_FAM = "E + C toward caution or none"
GATE_FAM = "E, answers challenged with caution or not"
ALL_FAM = "best of all on tune"
FAMS = ("single E", "single C toward answering", "single C toward caution", "single C, either sign", MAIN_FAM,
        "E as a check", "E + C toward caution or none, as a check", GATE_FAM, ALL_FAM)


def passes(task, model, split):
    """take(arm) -> (values, answered, texts) of a complete pass, None when the file is missing or still being written."""
    b, bp = T.bank(task, split)
    want, sha, cache = list(b), S.sha_file(T.bank(task, split)[1]), {}

    def take(arm):
        if arm in cache:
            return cache[arm]
        p = "%s/gen_prompt/%s/%s/%s/%s.json" % (NS, task, model, split, S.tok(arm))
        out = None
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                g = json.load(f)
            k = g.get("contract") or {}
            if g.get("completed"):
                if not (k.get("full_pass") and k.get("split") == split and k.get("model") == model
                        and k.get("items_file_sha256") == sha and [r["id"] for r in g["rows"]] == want):
                    raise SystemExit("%s is not a full %s pass of %s" % (p, split, model))
                if (":" in arm) != (k.get("positions") == "prompt"):
                    raise SystemExit("%s: positions %r do not fit the arm %s" % (p, k.get("positions"), arm))
                val, ans, txt = [], [], []
                for r in g["rows"]:
                    blank = bool(r["rule_abstain"]) or not r["text"].strip()
                    v = AM.verdict(r["text"], b[r["id"]]["gold"], not blank)
                    val.append((v == "R") - (v == "W"))
                    ans.append(not blank)
                    txt.append(r["text"])
                out = (val, ans, txt)
        cache[arm] = out
        return out

    return take, want


def families(model, cl):
    """{family: ordered candidates (kind, arm)}; kind "arm" = the pass itself, "check" = E-type check of the no-write answer."""
    el, sweep = T.MAIN[model][0], T.MAIN[model][2]
    e = lambda s: "x:E%d.w.p%s" % (el, s)                              # noqa: E731
    ca = lambda k: "x:C%d.w.m%s" % (cl, k)                             # noqa: E731  toward answering
    cc = lambda k: "x:C%d.w.p%s" % (cl, k)                             # noqa: E731  toward caution
    ec = lambda s, k: "x:E%d.w.p%s+C%d.w.p%s" % (el, s, cl, k)         # noqa: E731
    grid = [a for s in sweep for a in [e(s)] + [ec(s, k) for k in KS]]
    union = ([("arm", e(s)) for s in sweep] + [("arm", cc(k)) for k in KS] + [("check", e(s)) for s in sweep]
             + [("check", cc(k)) for k in KS] + [("gate", ec(s, k), e(s)) for s in sweep for k in KS]
             + [("arm", ec(s, k)) for s in sweep for k in KS] + [("check", ec(s, k)) for s in sweep for k in KS])
    return {
        "single E": [("arm", e(s)) for s in sweep],
        "single C toward answering": [("arm", ca(k)) for k in KS],
        "single C toward caution": [("arm", cc(k)) for k in KS],
        "single C, either sign": [("arm", ca(k)) for k in KS] + [("arm", cc(k)) for k in KS],
        MAIN_FAM: [("arm", a) for a in grid],
        "E as a check": [("check", e(s)) for s in sweep],
        "E + C toward caution or none, as a check": [("check", a) for a in grid],
        GATE_FAM: [x for s in sweep for x in [("arm", e(s))] + [("gate", ec(s, k), e(s)) for k in KS]],
        ALL_FAM: union,
    }


def system(take, key):
    kind, arm = key[0], key[1]
    p, base = take(arm), take(key[2] if kind == "gate" else "x")
    if p is None or base is None:
        return None
    if kind == "arm":
        return p[0], p[1]
    val, ans = [], []
    for v, a, t, a2, t2 in zip(base[0], base[1], base[2], p[1], p[2]):
        keep = bool(a and a2 and ro.norm(t) == ro.norm(t2))
        val.append(v if keep else 0)
        ans.append(keep)
    return val, ans


def choose(task, model, cl):
    """({family: chosen key}, {family: [(key, tune net, tune answers)]}, take on tune) or a message when incomplete."""
    take, _ = passes(task, model, "tune")
    if take("x") is None:
        return "no no-write pass on tune", None, None
    pick, shown = {}, {}
    for fam, cands in families(model, cl).items():
        got = [(k, system(take, k)) for k in cands]
        miss = [k[1] for k, s in got if s is None]
        if miss:
            return "%s: %d of %d tune passes missing (first: %s)" % (fam, len(miss), len(cands), miss[0]), None, None
        shown[fam] = [(k, sum(s[0]), sum(s[1])) for k, s in got]
        pick[fam] = max(got, key=lambda x: (sum(x[1][0]), -[g[0] for g in got].index(x[0])))[0]
    return pick, shown, take


def short(key):
    if key[0] == "gate":
        return "%s kept if same under +%s" % (key[2][2:], key[1].split("+")[1])
    return ("check with " if key[0] == "check" else "") + key[1][2:]


def main():
    argv = sys.argv[1:]
    M = S.models()
    if argv[:1] == ["--choose"]:
        model = argv[1]
        for task in S.TASKS:
            pick, _, _ = choose(task, model, M[model]["cl"])
            if isinstance(pick, str):
                raise SystemExit("%s %s: %s" % (task, model, pick))
            arms = []
            for key in pick.values():
                for a in key[1:]:
                    if a not in arms:
                        arms.append(a)
            print(task, " ".join(arms))
        return
    models, splits = list(S.ORDER), ["test", "test2"]
    while argv[:1] and argv[0] in ("--models", "--splits"):
        if argv[0] == "--models":
            models = argv[1].split(",")
        else:
            splits = argv[1].split(",")
        argv = argv[2:]
    for task in argv or list(S.TASKS):
        print("\n# %s at the prompt rule (alias match; every setting chosen on tune, 192 questions)" % task)
        tot = {sp: {} for sp in splits}
        for name in models:
            pick, shown, take = choose(task, name, M[name]["cl"])
            if isinstance(pick, str):
                print("\n## %s: %s" % (name, pick))
                continue
            base = take("x")
            print("\n## %s, E at layer %d: tune no write net %+d (answers %d)" % (name, T.MAIN[name][0], sum(base[0]), sum(base[1])))
            for fam in FAMS:
                rows_t = shown[fam] if fam != ALL_FAM else sorted(shown[fam], key=lambda x: -x[1])[:5]
                print("    tune, %-42s chosen %-34s | %s" % (fam + ":", short(pick[fam]), " ".join(
                    "%s=%+d(%d)" % ((short(k) if fam == ALL_FAM else k[1][2:]).replace(".w.", "").replace("C%d" % M[name]["cl"], "C").replace(" ", "_"), n, a) for k, n, a in rows_t)))
            for sp in splits:
                tk, _ = passes(task, name, sp)
                b = tk("x")
                sysm = {fam: system(tk, pick[fam]) for fam in FAMS}
                miss = [fam for fam in FAMS if sysm[fam] is None]
                if b is None or miss:
                    print("    %s: not complete (%s)" % (sp, "no no-write pass" if b is None else "missing: " + "; ".join(short(pick[f]) for f in miss)))
                    continue
                df = lambda x, y: S.ps.ci([p - q for p, q in zip(x[0], y[0])])  # noqa: E731
                print("    %s (%d questions): no write %+d (answers %d) | %s" % (sp, len(b[0]), sum(b[0]), sum(b[1]), " | ".join(
                    "%s %+d (%d)" % (fam, sum(sysm[fam][0]), sum(sysm[fam][1])) for fam in FAMS)))
                a, v, v2 = sysm[MAIN_FAM], sysm["E as a check"], sysm["E + C toward caution or none, as a check"]
                gt, u = sysm[GATE_FAM], sysm[ALL_FAM]
                rows = (
                    ("A minus no write", a, (b[0], b[1])), ("A minus single E", a, sysm["single E"]),
                    ("A minus single C toward caution", a, sysm["single C toward caution"]),
                    ("A minus single C, either sign", a, sysm["single C, either sign"]),
                    ("A minus single C toward answering", a, sysm["single C toward answering"]),
                    ("B minus no write", v, (b[0], b[1])), ("B minus single E", v, sysm["single E"]),
                    ("B minus single C, either sign", v, sysm["single C, either sign"]), ("B minus A", v, a),
                    ("B2 minus B", v2, v), ("B2 minus single E", v2, sysm["single E"]),
                    ("B2 minus single C, either sign", v2, sysm["single C, either sign"]),
                    ("single E minus no write", sysm["single E"], (b[0], b[1])),
                    ("G minus no write", gt, (b[0], b[1])), ("G minus single E", gt, sysm["single E"]),
                    ("G minus single C, either sign", gt, sysm["single C, either sign"]), ("G minus A", gt, a), ("G minus B", gt, v),
                    ("U minus no write", u, (b[0], b[1])), ("U minus single E", u, sysm["single E"]),
                    ("U minus single C, either sign", u, sysm["single C, either sign"]),
                    ("U minus single C toward answering", u, sysm["single C toward answering"]),
                )
                for i in range(0, len(rows), 5):
                    print("        " + " | ".join("%s %s" % (lab, df(x, y)) for lab, x, y in rows[i:i + 5]))
                for lab, x, y in rows:
                    tot[sp].setdefault(lab, []).append([p - q for p, q in zip(x[0], y[0])])
        for sp in splits:
            if tot[sp]:
                n = min(len(v) for v in tot[sp].values())
                print("\n## %s, %s: %d of %d models complete, together (questions resampled jointly)" % (task, sp, n, len(models)))
                for lab, per in tot[sp].items():
                    if len({len(v) for v in per}) != 1:
                        print("    %-36s question lists differ, no total" % lab)
                        continue
                    t_, lo, hi = S.joint_ci(per)
                    print("    %-36s %+d [%+d, %+d], %d of %d models positive" % (lab, t_, lo, hi, sum(sum(v) > 0 for v in per), len(per)))


if __name__ == "__main__":
    main()
