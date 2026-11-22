#!/usr/bin/env python3
# Iteration_01 read-only diagnosis (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_e.md.
# No model, no judge call, no generation. Fact sets (PopQA, TriviaQA; batches test and test2), write rule "prompt", five
# models, exactly the answers / alias verdicts / systems of diag/ns_prompt_table.py (imported, unchanged):
#   F  N1 at the main-table E strength with the caps frozen from TruthfulQA, against single E at that strength;
#   D  N1 at the E strength and caps chosen on the dataset's tune split, against single E at the same strength
#      (ns_prompt_table's "D: N1 minus single E, same strength").
# Question: how many answers that alias matching scores WRONG are in substance non-answers, and what "N1 minus single E"
# becomes when they score 0 instead of -1 (on both sides). The existing rule detector (core/readout.py rule_abstains,
# applied by scripts/newsets/newsets_gen.py when the answer is generated -> row field rule_abstain) already turns its own
# phrases into "no answer"; this list is the work queue's fixed list, applied on top, never edited after seeing results:
NONANSWER = ("no answer found", "no answer provided", "no answer", "no answer available", "i don't know", "i do not know",
             "i'm not sure", "i am not sure", "unknown", "not known", "not available", "n/a", "not specified", "not provided",
             "not applicable", "cannot be determined", "can't be determined", "no information", "no information available",
             "no data", "unclear")
# Match: the answer stripped of surrounding white space and of trailing punctuation, lower-cased, EQUALS one phrase, or
# STARTS WITH one phrase (plain string prefix) and has at most 6 words. Apostrophes are not normalised.
# Only answers alias matching scored W are changed (to 0); counts are reported for single E's answers and for N1's
# released answers separately. Intervals as ns_prompt_table: per model policy_select2.ci, five models together
# newsets_score.joint_ci on the sorted common questions. The first block reproduces the existing numbers.
# usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/factqa_nonanswer_diag.py          the table
#   python3 -B Iteration_01/diag/factqa_nonanswer_diag.py --top 30 most frequent released answer texts per model
#                                                                  (terminal only; contains model answer texts)
import sys
from collections import Counter

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import ns_prompt_table as NP     # noqa: E402

S, fr = NP.S, NP.fr
TASKS = ("triviaqa", "popqa")
SPLITS = ("test", "test2")
PUNCT = ".!?,;:"


def norm(t):
    return t.strip().rstrip(PUNCT).strip().lower()


def nonanswer(t):
    n = norm(t)
    return any(n == p or (n.startswith(p) and len(n.split()) <= 6) for p in NONANSWER)


def model_block(task, name, m, split):
    """{"F": (E recs, N1 recs, cfg), "D": (...)} on the read split, as ns_prompt_table chooses them."""
    el, main_s, sweep, tq_caps, tq_c = NP.MAIN[name]
    tsp, ttx, th, _, _, _ = NP.load(task, name, m, "tune")
    xsp, xtx, xh, _, _, _ = NP.load(task, name, m, split)
    ms = NP.dash(main_s)
    out = {}
    f_cfg = ("N1", ms, tq_caps[0], tq_caps[1])
    out["F"] = (fr.records(xsp, ("E", ms), xtx), fr.records(xsp, f_cfg, xtx), f_cfg)
    tune_net, best = {}, {}
    for s in th:
        cands = [("N1", s, k1, k2) for k1 in NP.CAPS for k2 in NP.CAPS if not (k1 is None and k2 is None)]
        best[s] = max(cands, key=lambda c: (NP.net(S.system(tsp, ttx, c)[0]), -cands.index(c)))
        tune_net[s] = NP.net(S.system(tsp, ttx, best[s])[0])
    sN = max(th, key=lambda s: (tune_net[s], -th.index(s)))
    if sN not in xh:
        out["D"] = None
    else:
        out["D"] = (fr.records(xsp, ("E", sN), xtx), fr.records(xsp, best[sN], xtx), best[sN])
    return xsp.qs, out


def vals(recs, fix):
    v = []
    for r in recs:
        x = (r["state"] == "R") - (r["state"] == "W")
        if fix and r["state"] == "W" and nonanswer(r["text"]):
            x = 0
        v.append(x)
    return v


def main():
    top = None
    if sys.argv[1:2] == ["--top"]:
        top = int(sys.argv[2])
    M = S.models()
    rel_texts = {}
    print("# factqa non-answer diagnosis: alias-W answers that are in substance non-answers (fixed list in the file head)")
    print("# list: " + " | ".join(NONANSWER))
    for split in SPLITS:
        for task in TASKS:
            tot = {}
            print("\n## %s %s" % (task, split))
            for name in S.ORDER:
                qs, blk = model_block(task, name, M[name], split)
                for ver in ("F", "D"):
                    if blk[ver] is None:
                        print("  %s %s: pending (no complete pass of the tune choice)" % (name, ver))
                        continue
                    er, nr, cfg = blk[ver]
                    if [r["p"] for r in er] != qs or [r["p"] for r in nr] != qs:
                        raise SystemExit("record order")
                    hitE = [r for r in er if r["state"] == "W" and nonanswer(r["text"])]
                    rel = [r for r in nr if r["note"] == "released"]
                    hitR = [r for r in rel if r["state"] == "W" and nonanswer(r["text"])]
                    hitN = [r for r in nr if r["state"] == "W" and nonanswer(r["text"])]
                    for r in rel:
                        rel_texts.setdefault(name, {})[(task, split, r["p"], r["cond"])] = r["text"].strip()
                    d0 = [a - b for a, b in zip(vals(nr, False), vals(er, False))]
                    d1 = [a - b for a, b in zip(vals(nr, True), vals(er, True))]
                    print("  %-12s %s %s: single E W answers %d, on the list %d | N1 released %d (W %d), on the list %d | N1 W answers "
                          "on the list in all %d | N1 minus single E: as scored %s -> list answers scored 0 %s" % (
                              name, ver, cfg, sum(r["state"] == "W" for r in er), len(hitE), len(rel), sum(r["state"] == "W" for r in rel),
                              len(hitR), len(hitN), S.ps.ci(d0), S.ps.ci(d1)))
                    if ver == "F":
                        print("      list hits by phrase prefix, overlapping (e.g. 'no answer' also counts 'no answer found') (single E / released): %s" % (
                            " | ".join("%s %d/%d" % (p, sum(norm(r["text"]).startswith(p) for r in hitE), sum(norm(r["text"]).startswith(p) for r in hitR))
                                       for p in NONANSWER if any(norm(r["text"]).startswith(p) for r in hitE + hitR)) or "none"))
                    tot.setdefault(ver, {})[name] = (dict(zip(qs, d0)), dict(zip(qs, d1)))
            for ver in ("F", "D"):
                per = tot.get(ver, {})
                if len(per) != len(S.ORDER):
                    print("  five models %s: %d of 5 complete, no total" % (ver, len(per)))
                    continue
                common = sorted(set.intersection(*[set(p[0]) for p in per.values()]))
                r0 = S.joint_ci([[per[n][0][q] for q in common] for n in S.ORDER])
                r1 = S.joint_ci([[per[n][1][q] for q in common] for n in S.ORDER])
                print("  five models together %s, %d questions: N1 minus single E as scored %+d [%+d, %+d] -> list answers scored 0 %+d [%+d, %+d]" % (
                    (ver, len(common)) + tuple(r0) + tuple(r1)))
    if top:
        for name in S.ORDER:
            c = Counter(rel_texts.get(name, {}).values())
            print("\n## %s: %d released answers (both N1 versions, four batches, deduplicated by question and condition); top %d texts" % (
                name, sum(c.values()), top), file=sys.stderr)
            for t, k in c.most_common(top):
                print("  %4d  %r  %s" % (k, t[:80], "LIST" if nonanswer(t) else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
