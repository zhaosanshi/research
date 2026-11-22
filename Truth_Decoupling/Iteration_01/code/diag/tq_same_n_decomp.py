# Fable 5.2, 2026-11-22, exploration, read-only (no model, no judge call). TruthfulQA test split, main-table settings
# at the write rule "prompt". The decomposition GPT made for (QA)^2 (discussion/validation_dataset/
# 2026-11-22_qaqa_replay_b.py, function replay; 2026-11-22_recovery_pool_b.py, "E-only filling same K blanks") applied
# to the main table's equal-answer-count comparison: N1 against must-answer + E, the N most confident answers kept.
# Anchor: the row "primary" must equal the row "N1 minus the strong form" of diag/main_table_prompt.py, per model
# and for the five models together (results/write_positions/main_table_prompt_budget1_20261121.txt).
# Rows (per question, summed over models, paired bootstrap of main_table_prompt): primary = N1 - strong form;
# selection + content + boundary = primary (strong form's answers held, N1's list of questions; N1's answers on
# its own list; rest); swap_N1 / swap_E = on the strong form's list its answer replaced by N1's / by single E's
# where that system answers; N1_minus_Efill = N1 against single E that fills as many of its blanks as N1 releases,
# with its own must-answer answers, the most confident first (the same answer count as N1).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/tq_same_n_decomp.py
import hashlib
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as A

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
V = {"R": 1, "W": -1, "X": 0, "H": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
KEYS = ("primary", "selection", "content", "boundary", "swap_N1", "swap_E", "swap_diff", "N1_minus_E",
        "N1_minus_Efill", "Efill_minus_E", "Efill_minus_strong")


def forced(run, layer, s, qs):
    tag = "l%d" % layer
    w = MP.work(run, "test", "pospromptd" + tag)
    st, bl = ps.load(w, "pro")
    texts = MP.fr.raw_texts(w)
    cond = "E_L%d_conf-w-p%s" % (layer, s)
    with open("%s/eval_extra/confidence_pospromptd%s_test.json" % (run, tag), encoding="utf-8") as f:
        d = json.load(f)
    rows = {r["pair_id"]: r for r in d["targets"]["conf_E" + MP.DM.show(s)]["rows"]}
    out = {}
    for q in qs:
        if st[(q, cond)] == "?" or hashlib.sha1(texts[(q, cond)].strip().encode("utf-8")).hexdigest() != rows[q]["text_sha1"]:
            raise SystemExit("forced arm: unresolved verdict or foreign confidence row for %s" % q)
        out[q] = (V[st[(q, cond)]], not bl[(q, cond)], rows[q]["mean_logp"])
    return out


def one(model):
    display, layer, s = A.KEY[model]
    tune, err = MP.prompt_split(display, layer, 0, s)
    if tune is None:
        raise SystemExit(err)
    test, err = MP.prompt_split(display, layer, 1, s)
    if test is None:
        raise SystemExit(err)
    # the main table reads every system on the questions that the prompt rule and the training rule both have
    # (main_table_prompt.model_block); the same lists here, so that the first row reproduces its strong-form row
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    if len(src) != 1:
        raise SystemExit("%s: layer %s is not one of the main table's sources" % (model, layer))
    ct = [q for q in tune[3] if q in set(src[0][1].qs)]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if ct != tune[3]:
        tune, err = MP.prompt_split(display, layer, 0, s, qs=ct)
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    if tune is None or test is None or tune[3] != ct or test[3] != cx:
        raise SystemExit("%s: the restricted splits are not the main table's" % model)
    sp, tx, _, ids = test
    tsp, tt, _, _ = tune
    _, run, _ = MP.usual(display, 1)
    n1cfg = ML.n1_pick(tsp, tt, s)
    es, ea = S.system(sp, tx, ("E", s))[:2]
    ns, na = S.system(sp, tx, n1cfg)[:2]
    F = forced(run, layer, s, ids)
    n = sum(na)
    sf, why = MP.strong_prompt(run, layer, s, ids, n)
    if sf is None:
        raise SystemExit("%s: %s" % (model, why))
    e = {q: V[x] for q, x in zip(ids, es)}
    eans = dict(zip(ids, ea))
    n1 = {q: V[x] for q, x in zip(ids, ns)}
    issued = dict(zip(ids, na))
    pos = {q: i for i, q in enumerate(ids)}
    elig = [q for q in ids if F[q][1] and F[q][2] is not None]
    ranked = sorted(elig, key=lambda q: (-F[q][2], pos[q]))
    top = set(ranked[:n])
    if [V[x] for x in sf[0]] != [F[q][0] if q in top else 0 for q in ids]:
        raise SystemExit("%s: the strong comparator rebuilt here differs from main_table_prompt.strong_prompt" % model)
    eset = set(elig)
    both = {q for q in ids if issued[q] and q in eset}
    top_overlap = set(ranked[:len(both)])
    blanks = [q for q in ids if not eans[q]]
    k = sum(issued[q] for q in blanks)
    eb = [q for q in ranked if not eans[q]]
    if len(eb) < k:
        raise SystemExit("%s: E alone cannot fill %d blanks (%d candidates)" % (model, k, len(eb)))
    esel = set(eb[:k])
    cells = {}
    for q in ids:
        f = F[q][0]
        strong = f if q in top else 0
        primary = n1[q] - strong
        sel = (int(q in both) - int(q in top_overlap)) * f
        con = n1[q] - f if q in both else 0
        sw_n1 = n1[q] - f if q in top and issued[q] else 0
        sw_e = e[q] - f if q in top and eans[q] else 0
        fill = f if q in esel else e[q]
        cells[q] = dict(primary=primary, selection=sel, content=con, boundary=primary - sel - con, swap_N1=sw_n1,
                        swap_E=sw_e, swap_diff=sw_n1 - sw_e, N1_minus_E=n1[q] - e[q], N1_minus_Efill=n1[q] - fill,
                        Efill_minus_E=fill - e[q], Efill_minus_strong=fill - strong)
    rel = [q for q in blanks if issued[q]]
    print("%s layer %s strength %s | N1 choice %s | answers: E %d, N1 %d (released %d: right %d, wrong %d, other %d), "
          "E-fill %d | forced answers with confidence %d | every E answer kept by N1: %s" % (
              model, layer, s, n1cfg, sum(ea), n, k, sum(n1[q] == 1 for q in rel), sum(n1[q] == -1 for q in rel),
              sum(n1[q] == 0 for q in rel), sum(ea) + k, len(elig), all(issued[q] for q in ids if eans[q])))
    print("    forced answers on the same %d released questions: right %d, wrong %d, other/blank %d | E-fill's own %d picks: right %d, wrong %d, other %d" % (
        k, sum(F[q][0] == 1 and F[q][1] for q in rel), sum(F[q][0] == -1 and F[q][1] for q in rel),
        sum(F[q][0] == 0 or not F[q][1] for q in rel),
        k, sum(F[q][0] == 1 for q in esel), sum(F[q][0] == -1 for q in esel), sum(F[q][0] == 0 for q in esel)))
    print("    " + "  ".join("%s %s" % (key, ps.ci([cells[q][key] for q in ids])) for key in ("primary", "N1_minus_Efill", "swap_diff")))
    return ids, cells


def main():
    data = {m: one(m) for m in ORDER}
    for names in (ORDER[:3], ORDER):
        # sorted, as diag/main_table_prompt.py orders the common questions (the bootstrap stream is fixed, so the
        # order decides the interval's end points)
        common = sorted(q for q in data[names[0]][0] if all(q in data[m][1] for m in names))
        print("%d models, %d questions with every model" % (len(names), len(common)))
        for key in KEYS:
            print("  %-20s %s" % (key, ps.ci([sum(data[m][1][q][key] for m in names) for q in common])))


if __name__ == "__main__":
    main()
