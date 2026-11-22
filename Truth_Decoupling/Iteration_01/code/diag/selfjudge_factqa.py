#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_e.md, item 1e
# ("self-judge on the fact sets"), Llama-3.1-8B and Qwen3.5-9B. Not reviewed by GPT; exploration readings only. No
# generation, no judge call: existing answers and their alias-match verdicts only.
# Answers (write rule "prompt", E at the main-table layer and strength), read exactly as diag/ns_prompt_table.py reads
# them (ns_prompt_table.load: generation files under newsets/gen_prompt/<task>/<model>/<split>/, contract checked by
# newsets_gen.load_generation, verdicts by scripts/newsets/alias_match.py on every question). Sets: PopQA and TriviaQA,
# the two test batches "test" and "test2" (500 questions each). There are no DeepSeek Pro verdicts for these passes (the
# Pro 100 work dirs under newsets/judge/ judge the earlier E11 setting), so only the alias-match reading exists.
#   single E   ("E", main strength)
#   N1         F = caps frozen from TruthfulQA (ns_prompt_table "F" row; main reading); D = caps chosen on the dataset's
#              own tune split at the main-table E strength (ns_prompt_table's per-strength row; side reading).
# Self-judge: selfjudge_probe.py score MODEL --sources fq_e,fq_rel (E45's code path, statement
# 'The answer to the question "{q}" is: {a}' built by selfjudge_probe.statement, Unknown-allowed prompt), conditions
# baseline (no write; used here) and E_a3 (not read here). Rule Z (work queue "1 supplement"): an answer that does not
# open with a negated premise (selfjudge_probe.NEG_RE) is judged false when margin = logp(True) - logp(False) < 0;
# negated-premise openings are never judged false. No threshold is chosen anywhere.
# Systems (value right +1 / wrong -1 / blank 0, alias match):
#   S1 = single E, answers judged false -> no answer;  S2 = N1 with S1's flags on E's first-pass answers;
#   S3 = S2 plus the same rule on N1's released answers.
# Side reading: rule Z without the exemption (work queue "1 supplement"), N1 F.
# Readings: S1 - E, S3 - N1, S3 - E, S3 - S2; answers judged false (right / wrong), E answers and released answers
# separately. Intervals: per model policy_select2.ci (4000, seed 20261116), two models together newsets_score.joint_ci
# (4000, seed 20261119) on the common question list, as ns_prompt_table.py does.
# Checks: the probe rows are the exported answered texts (sha1), each exported text is the generation text that the
# alias verdict was computed from (same dict), the reference nets of ns_prompt_table's F row are reproduced.
# usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/selfjudge_factqa.py export MODEL     -> newsets/selfjudge/<MODEL>/fq_e_items.json, fq_rel_items.json
#   (GPU) Iteration_01/logs/newsets_20261119/selfjudge_gate_1122.sh MODEL   (runs selfjudge_probe.py score --sources fq_e,fq_rel)
#   python3 -B Iteration_01/diag/selfjudge_factqa.py table
import json
import os
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_probe as SP     # noqa: E402

ORDER = ("llama3.1-8b", "qwen3.5-9b")
SETS = (("popqa", "test"), ("popqa", "test2"), ("triviaqa", "test"), ("triviaqa", "test2"))
# ns_prompt_table F row (single E net / answers, N1 net / answers) and the per-strength row at the main strength (N1 with
# caps chosen on the dataset's tune split), results/newsets/prompt_rule/ns_prompt_table_test*_five_models_20261122.txt
EXPECT = {
    ("llama3.1-8b", "triviaqa", "test"): ((127, 263), (55, 433), (127, 263)),
    ("llama3.1-8b", "triviaqa", "test2"): ((149, 251), (72, 424), (148, 252)),
    ("llama3.1-8b", "popqa", "test"): ((15, 99), (-163, 333), (14, 100)),
    ("llama3.1-8b", "popqa", "test2"): ((19, 99), (-121, 309), (19, 99)),
    ("qwen3.5-9b", "triviaqa", "test"): ((128, 376), (105, 401), (123, 381)),
    ("qwen3.5-9b", "triviaqa", "test2"): ((150, 386), (129, 409), (148, 388)),
    ("qwen3.5-9b", "popqa", "test"): ((-57, 189), (-75, 207), (-57, 189)),
    ("qwen3.5-9b", "popqa", "test2"): ((-34, 200), (-51, 225), (-35, 201)),
}
_CACHE = {}


def fact_data(model, task, split):
    """Per question of the split (bank order): question, single E (value, answered, text), N1 F and D (value, answered,
    released text / cond or None, cand of an E blank: 'own' or "E's")."""
    key = (model, task, split)
    if key in _CACHE:
        return _CACHE[key]
    import ns_prompt_table as NP
    S, fr = NP.S, NP.fr
    m = S.models()[model]
    el, main_s, sweep, tq_caps, tq_c = NP.MAIN[model]
    ms = NP.dash(main_s)
    tsp, ttx, th, _, _, _ = NP.load(task, model, m, "tune")
    xsp, xtx, xh, _, _, _ = NP.load(task, model, m, split)
    if ms not in th or ms not in xh:
        raise SystemExit("%s %s %s: main strength %s not complete" % (model, task, split, main_s))
    cands = [("N1", ms, k1, k2) for k1 in NP.CAPS for k2 in NP.CAPS if not (k1 is None and k2 is None)]
    dcfg = max(cands, key=lambda c: (NP.net(S.system(tsp, ttx, c)[0]), -cands.index(c)))
    fcfg = ("N1", ms, tq_caps[0], tq_caps[1])
    b, _ = NP.bank(task, split)
    out = dict(ids=list(xsp.qs), cfg=dict(E=("E", ms), F=fcfg, D=dcfg), q={})
    sysr = {}
    for name, cfg in (("E", ("E", ms)), ("F", fcfg), ("D", dcfg)):
        st, ans, recs = S.system(xsp, xtx, cfg)
        if [r["p"] for r in recs] != xsp.qs:
            raise SystemExit("record order")
        sysr[name] = (S.vals(st), ans, recs)
    for i, q in enumerate(xsp.qs):
        ev, ea, er = sysr["E"][0][i], sysr["E"][1][i], sysr["E"][2][i]
        if er["text"] != xtx.ec[(q, er["cond"])]:
            raise SystemExit("E text")
        row = dict(question=b[q.split(":", 1)[1]]["question"], e=ev, eans=bool(ea), etext=er["text"], cand=er["cand"])
        for name in ("F", "D"):
            v, a, r = sysr[name][0][i], sysr[name][1][i], sysr[name][2][i]
            rel = None
            if r["note"] == "released":
                if ea or not a or r["text"] != xtx.ec[(q, r["cond"])]:
                    raise SystemExit("%s %s: released record of %s" % (model, name, q))
                rel = dict(text=r["text"], cond=r["cond"], v=v)
            elif bool(a) != bool(ea) or (ea and v != ev):
                raise SystemExit("%s %s: N1 changes an E answer of %s (%s)" % (model, name, q, r["note"]))
            row[name] = dict(v=v, ans=bool(a), rel=rel)
        out["q"][q] = row
    got = ((sum(r["e"] for r in out["q"].values()), sum(r["eans"] for r in out["q"].values())),
           (sum(r["F"]["v"] for r in out["q"].values()), sum(r["F"]["ans"] for r in out["q"].values())),
           (sum(r["D"]["v"] for r in out["q"].values()), sum(r["D"]["ans"] for r in out["q"].values())))
    if got != EXPECT[key]:
        raise SystemExit("REFERENCE MISMATCH %s: got %s, expected %s" % (key, got, EXPECT[key]))
    _CACHE[key] = out
    return out


def export(model):
    e_items, r_items = [], {}
    for task, split in SETS:
        d = fact_data(model, task, split)
        for q in d["ids"]:
            r = d["q"][q]
            if r["eans"]:
                e_items.append(dict(id="%s|%s" % (split, q), question=r["question"], answer=r["etext"], answer_sha1=SP.sha1(r["etext"]),
                                    value=r["e"], answered=True, neg=bool(SP.NEG_RE.match(r["etext"]))))
            for name in ("F", "D"):
                rel = r[name]["rel"]
                if rel is None:
                    continue
                k = "%s|%s|%s" % (split, q, rel["cond"])
                x = dict(id=k, question=r["question"], answer=rel["text"], answer_sha1=SP.sha1(rel["text"]), value=rel["v"],
                         answered=True, neg=bool(SP.NEG_RE.match(rel["text"])))
                if r_items.setdefault(k, x) != x:
                    raise SystemExit("two release records for %s" % k)
    import selfjudge_stage1 as S1
    S1.write_items(model, "fq_e", e_items, dict(sets=SETS))
    S1.write_items(model, "fq_rel", list(r_items.values()), dict(sets=SETS))


def margins(model):
    import selfjudge_stage1 as S1
    return S1.margins(model, "fq_e", "baseline")[1], S1.margins(model, "fq_rel", "baseline")[1]


def judged(m, text, exempt=True):
    if m is None:
        raise ValueError("an answer without a self-judgement")
    return (not (exempt and SP.NEG_RE.match(text))) and m < 0.0


def systems(model, task, split, me, mr, version="F", exempt=True):
    """{name: {q: value}}, answer counts, dropped lists. me / mr: margins of fq_e / fq_rel."""
    d = fact_data(model, task, split)
    S1, S2, S3, E, N1 = {}, {}, {}, {}, {}
    n = dict(E=0, N1=0, S1=0, S2=0, S3=0)
    dE, dR, rel_all = [], [], []
    for q in d["ids"]:
        r = d["q"][q]
        nr = r[version]
        E[q], N1[q] = r["e"], nr["v"]
        fe = r["eans"] and judged(me.get("%s|%s" % (split, q)), r["etext"], exempt)
        rel = nr["rel"]
        fr_ = rel is not None and judged(mr.get("%s|%s|%s" % (split, q, rel["cond"])), rel["text"], exempt)
        S1[q] = 0 if fe else r["e"]
        S2[q] = 0 if fe else nr["v"]
        S3[q] = 0 if (fe or fr_) else nr["v"]
        n["E"] += r["eans"]
        n["N1"] += nr["ans"]
        n["S1"] += r["eans"] and not fe
        n["S2"] += nr["ans"] and not fe
        n["S3"] += nr["ans"] and not fe and not fr_
        if fe:
            dE.append(r["e"])
        if rel is not None:
            rel_all.append(rel["v"])
            if fr_:
                dR.append(rel["v"])
    return dict(E=E, N1=N1, S1=S1, S2=S2, S3=S3), n, dE, dR, rel_all, d


def cnt(v):
    return "%d (right %d, wrong %d)" % (len(v), v.count(1), v.count(-1))


DIFFS = (("S1-E", "S1", "E"), ("S3-N1", "S3", "N1"), ("S3-E", "S3", "E"), ("S3-S2", "S3", "S2"))


def table():
    import newsets_score as NS
    ci = NS.ps.ci
    M = {m: margins(m) for m in ORDER}
    print("# selfjudge 1e (work queue 2026-11-22_queue_e): fact sets, rule Z, no write; alias match only (no Pro verdicts exist "
          "for these passes); N1 = F (caps frozen from TruthfulQA) unless marked D")
    print("# references: ns_prompt_table F row (single E, N1) and the main-strength row (N1 with dataset-tuned caps) reproduced "
          "for every model and set (checked in fact_data)")
    for version, exempt in (("F", True), ("D", True), ("F", False)):
        print("=" * 100)
        print("N1 version %s%s" % (version, "" if version == "F" and exempt else " (side reading: caps chosen on the dataset's tune split at the main E strength)"
                                   if exempt else " (side reading: rule Z WITHOUT the negated-premise exemption, every answer judged)"))
        for task, split in SETS:
            per = {}
            for m in ORDER:
                sv, n, dE, dR, rel_all, d = systems(m, task, split, M[m][0], M[m][1], version, exempt)
                per[m] = (sv, d["ids"])
                nets = {k: sum(v.values()) for k, v in sv.items()}
                print(" %s %s %s (N1 %s): answers E %d, N1 %d, S1 %d, S2 %d, S3 %d | nets E %+d, N1 %+d, S1 %+d, S2 %+d, S3 %+d" % (
                    m, task, split, d["cfg"][version], n["E"], n["N1"], n["S1"], n["S2"], n["S3"],
                    nets["E"], nets["N1"], nets["S1"], nets["S2"], nets["S3"]))
                print("    E answers judged false %s | released %s, judged false %s" % (cnt(dE), cnt(rel_all), cnt(dR)))
                print("    " + " | ".join("%s %s" % (k, ci([sv[a][q] - sv[b][q] for q in d["ids"]])) for k, a, b in DIFFS))
            ids = per[ORDER[0]][1]
            if per[ORDER[1]][1] != ids:
                raise SystemExit("question lists differ between models")
            line = []
            for k, a, b in DIFFS:
                tot, lo, hi = NS.joint_ci([[per[m][0][a][q] - per[m][0][b][q] for q in ids] for m in ORDER])
                line.append("%s %+d [%+d, %+d]" % (k, tot, lo, hi))
            print("  two models together, %s %s, %d questions: %s" % (task, split, len(ids), " | ".join(line)))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "export" and sys.argv[2] in ORDER:
        export(sys.argv[2])
    elif sys.argv[1:] == ["table"]:
        table()
    else:
        raise SystemExit("usage: selfjudge_factqa.py export MODEL | table")


if __name__ == "__main__":
    main()
