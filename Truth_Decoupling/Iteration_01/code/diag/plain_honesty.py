# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22, queue H section 5), read-only, no model, no judge call. Exploration.
# Cell 0 = the plain prompt with the honest sentence (prompt x_E2, configs/prompts/e44_eval_extra_h22_v1.json), NO write,
# against the no-write plain prompt, single E and N1 (writes at the core rule "prompt", main-table E layer):
#   TruthfulQA  test and tune; every system on the main table's question lists (main_table_prompt.prompt_split restricted
#               as diag/tq_same_n_decomp.load_model); N1 = the main table's pick on tune; cell 0 = baseline_x_E2 of the
#               h22c4 work dirs; DeepSeek Pro.
#   (QA)^2      570 questions, the 100 pilot (tune) and the 470 held-out ones separately, and by official type; N1 and
#               single E as chosen on the tune split with E fixed at the main-table strength
#               (newsets/splits/qaqa_tune_choice_fixedE_20261122.json, diag/qaqa_pilot_table.Raw); cell 0 from
#               newsets/gen_prompt/h22/qaqafull/<model>/test/x_E2.json, work dir test_full570_h22z; DeepSeek Pro.
#   PopQA, TriviaQA  test and test2 (500 questions each), alias match (no judge), read as diag/ns_prompt_table.py reads
#               them; single E at the main-table strength; N1 F = caps frozen from TruthfulQA (main), N1 D = caps chosen on
#               the dataset's tune split at the main strength (side; as diag/selfjudge_factqa.fact_data); cell 0 from
#               newsets/gen_prompt/h22/<task>/<model>/<split>/x_E2.json, blank = rule abstention or empty output.
# Natural answer counts; net = right - wrong. Intervals: per model policy_select2.ci (TruthfulQA, as main_table_prompt)
# or newsets_score.ps.ci; models together: sum per question over the models on the sorted common questions, then
# policy_select2.ci (TruthfulQA) / newsets_score.joint_ci (the other sets, as their tables).
# Equal-answer-count readings: cell 0 has no confidence file anywhere, so there are none (printed as "none").
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/plain_honesty.py tqa|qaqa|fact
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import iti_table as A          # noqa: E402
import qaqa_pilot_table as Q   # noqa: E402
import ns_prompt_table as NP   # noqa: E402

MP, ML, ps = A.MP, A.ML, A.ps
S = Q.S
G, AM = NP.G, NP.AM
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
V = {"R": 1, "W": -1, "X": 0, "H": 0}
NSD = S.NS


def show(title, systems, diffs, ci, per_model_lists):
    """systems: {name: {model: {q: (v, a)}}}; per_model_lists: {model: qs}; the last entry of the dict is the total."""
    print("\n## %s" % title)
    for m, qs in per_model_lists.items():
        models = ORDER if m == "five models" else (m,)
        print("    %-12s %3d q | " % (m, len(qs)) + " | ".join("%s %+d (%d)" % (
            n, sum(systems[n][x][q][0] for x in models for q in qs), sum(systems[n][x][q][1] for x in models for q in qs))
            for n in systems) + " || " + " | ".join("%s minus %s %s" % (a, b, ci([sum(systems[a][x][q][0] - systems[b][x][q][0] for x in models) for q in qs])) for a, b in diffs))


def fmt3(t):
    return "%+d [%+d, %+d]" % tuple(t)


# ----------------------------------------------------------------------------- TruthfulQA
def tqa():
    out = {}
    for model in ORDER:
        display, layer, s = A.KEY[model]
        tune, err = MP.prompt_split(display, layer, 0, s)
        test, err2 = MP.prompt_split(display, layer, 1, s)
        if tune is None or test is None:
            raise SystemExit(err or err2)
        src = [x for x in ML.sources(display)[2] if x[0] == layer]
        ct = [q for q in tune[3] if q in set(src[0][1].qs)]
        cx = [q for q in test[3] if q in set(src[0][3].qs)]
        if ct != tune[3]:
            tune, _ = MP.prompt_split(display, layer, 0, s, qs=ct)
        if cx != test[3]:
            test, _ = MP.prompt_split(display, layer, 1, s, qs=cx)
        n1cfg = ML.n1_pick(tune[0], tune[1], s)
        for i, (sp, tx, _, ids) in enumerate((tune, test)):
            split = ("tune", "test")[i]
            run = MP.usual(display, i)[1]
            w = MP.work(run, split, "h22c4")
            st, bl = ps.load(w, "pro")
            d = {}
            for name, cfg in (("no write", ("base",)), ("single E", ("E", s)), ("N1", n1cfg)):
                states, ans = S.system(sp, tx, cfg)[:2]
                d[name] = {q: (V[x], bool(a)) for q, x, a in zip(ids, states, ans)}
            if any(st.get((q, "baseline_x_E2"), "?") == "?" for q in ids):
                raise SystemExit("%s %s: cell 0 lacks verdicts" % (model, split))
            d["cell 0"] = {q: (V[st[(q, "baseline_x_E2")]], not bl[(q, "baseline_x_E2")]) for q in ids}
            out.setdefault(split, {})[model] = d
    for split in ("test", "tune"):
        per = out[split]
        systems = {n: {m: per[m][n] for m in ORDER} for n in ("no write", "single E", "N1", "cell 0")}
        lists = {m: list(per[m]["cell 0"]) for m in ORDER}
        lists["five models"] = sorted(set.intersection(*(set(v) for v in lists.values())))
        show("TruthfulQA %s (DeepSeek Pro; net, answers in brackets)" % split, systems,
             (("single E", "cell 0"), ("N1", "cell 0"), ("cell 0", "no write")), ps.ci, lists)
    print("    equal-answer-count readings: none (cell 0 has no confidence scores)")


# ----------------------------------------------------------------------------- (QA)^2
def qaqa():
    pilot = set(json.load(open(os.path.join(NSD, "splits/qaqa_pilot_20261122.json")))["ids"])
    choice = json.load(open(os.path.join(NSD, "splits/qaqa_tune_choice_fixedE_20261122.json")))
    bankp = os.path.join(NSD, "bank/qaqa_full_bank_20261122.jsonl")
    work = ("test_full570", "test_full570_tuned", "test_full570_strong")
    per, qtype = {}, None
    for model in ORDER:
        ch = choice[model]
        raw = Q.Raw(model, "qaqafull", bankp, work)
        r0 = Q.Raw(model, "qaqafull", bankp, ["test_full570_h22z"])
        r0.gdir = Path(NSD) / "gen_prompt" / "h22" / "qaqafull" / model / "test"
        r0.need("x_E2")
        if r0.ids != raw.ids or len(raw.ids) != 570:
            raise SystemExit("bank")
        qtype = raw.qtype
        n1 = raw.n1_system(ch["n1_strength"], ch["n1_cap_e"], ch["n1_cap_own"])
        sys_ = {"no write": raw.arm_system("x"), "single E": raw.arm_system(raw.e(ch["single_E"])), "N1": (n1[0], n1[1])}
        d = {n: {q: (v[0][q], bool(v[1][q])) for q in raw.ids} for n, v in sys_.items()}
        d["cell 0"] = {q: (r0.cell("x_E2", q), not r0.blank("x_E2", q)) for q in raw.ids}
        per[model] = d
        print("# (QA)^2 %s: N1 at E %s, caps E's %s / own %s; single E at %s (tune choice with E fixed)" % (
            model, ch["n1_strength"], ch["n1_cap_e"], ch["n1_cap_own"], ch["single_E"]))
    ids = list(per[ORDER[0]]["cell 0"])
    systems = {n: {m: per[m][n] for m in ORDER} for n in ("no write", "single E", "N1", "cell 0")}
    diffs = (("single E", "cell 0"), ("N1", "cell 0"), ("cell 0", "no write"))
    for label, inside in (("100 pilot (tune) questions", True), ("470 held-out questions", False)):
        for t, tl in (("all", "all types"),) + Q.TYPES:
            qs = [q for q in ids if (q in pilot) == inside and (t == "all" or qtype[q] == t)]
            lists = {m: qs for m in ORDER}
            lists["five models"] = qs
            show("(QA)^2 %s, %s (DeepSeek Pro)" % (label, tl), systems, diffs,
                 lambda v: fmt3(S.joint_ci([v])), lists)
    print("    equal-answer-count readings: none (cell 0 has no confidence scores)")


# ----------------------------------------------------------------------------- fact sets
def fact():
    M = S.models()
    for task, split in (("popqa", "test"), ("popqa", "test2"), ("triviaqa", "test"), ("triviaqa", "test2")):
        per = {}
        b, bp = NP.bank(task, split)
        for model in ORDER:
            m = M[model]
            el, main_s, sweep, tq_caps, tq_c = NP.MAIN[model]
            ms = NP.dash(main_s)
            tsp, ttx, th, _, _, _ = NP.load(task, model, m, "tune")
            xsp, xtx, xh, _, _, _ = NP.load(task, model, m, split)
            if ms not in th or ms not in xh:
                raise SystemExit("%s %s %s: main strength incomplete" % (model, task, split))
            cands = [("N1", ms, k1, k2) for k1 in NP.CAPS for k2 in NP.CAPS if not (k1 is None and k2 is None)]
            dcfg = max(cands, key=lambda c: (NP.net(S.system(tsp, ttx, c)[0]), -cands.index(c)))
            fcfg = ("N1", ms, tq_caps[0], tq_caps[1])
            d = {}
            for name, cfg in (("no write", ("base",)), ("single E", ("E", ms)), ("N1 F", fcfg), ("N1 D", dcfg)):
                states, ans = S.system(xsp, xtx, cfg)[:2]
                d[name] = {q: (v, bool(a)) for q, v, a in zip(xsp.qs, S.vals(states), ans)}
            p = os.path.join(NSD, "gen_prompt", "h22", task, model, split, "x_E2.json")
            g = G.load_generation(p, SimpleNamespace(task=task, split=split, model=model, items=bp), list(b))
            if g["contract"]["arm"] != "x_E2" or g["contract"].get("positions") is not None:
                raise SystemExit("%s is not the no-write x_E2 pass" % p)
            z = {}
            for r in g["rows"]:
                blank = bool(r["rule_abstain"]) or not r["text"].strip()
                v = AM.verdict(r["text"], b[r["id"]]["gold"], not blank)
                z["%s:%s" % (task, r["id"])] = ((v == "R") - (v == "W"), not blank)
            if list(z) != list(xsp.qs):
                raise SystemExit("%s %s %s: cell 0 question order" % (model, task, split))
            d["cell 0"] = z
            per[model] = d
            print("# %s %s %s: N1 F %s, N1 D %s" % (task, split, model, fcfg, dcfg))
        systems = {n: {m: per[m][n] for m in ORDER} for n in ("no write", "single E", "N1 F", "N1 D", "cell 0")}
        lists = {m: list(per[m]["cell 0"]) for m in ORDER}
        lists["five models"] = sorted(set.intersection(*(set(v) for v in lists.values())))
        show("%s %s (alias match; net, answers in brackets)" % (task, split), systems,
             (("cell 0", "no write"), ("single E", "cell 0"), ("N1 F", "cell 0"), ("N1 D", "cell 0")),
             lambda v: fmt3(S.joint_ci([v])) if len(v) else "none", lists)
    print("    equal-answer-count readings: none (cell 0 has no confidence scores)")


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part not in ("tqa", "qaqa", "fact"):
        raise SystemExit("usage: plain_honesty.py tqa|qaqa|fact")
    {"tqa": tqa, "qaqa": qaqa, "fact": fact}[part]()


if __name__ == "__main__":
    main()
