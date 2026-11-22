"""Read-only (QA)^2 held-out diagnosis. Run with python3 -B in the scoring container
at the Truth_Decoupling root. Keep direct E answers; compare different ways of
filling the same number of its blanks. Oracle uses verdicts only for an upper bound.
"""
import json
import math
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path("Iteration_01/diag").resolve()))
import qaqa_pilot_table as Q

NEGATIONS = re.compile(r"\b(?:no|not|never|none|neither|nobody|nothing|nowhere|cannot)\b|n['’]t\b", re.I)
WORK = ("test_full570", "test_full570_tuned", "test_full570_strong")
BANK = Path(Q.S.NS) / "bank/qaqa_full_bank_20261122.jsonl"
SPLITS = Path(Q.S.NS) / "splits"


def question_gate(rows, types, pilot_ids):
    """Diagnostic only: one fixed question-only classifier with extra premise supervision.
    No evaluation labels or model answers used for fitting or threshold selection.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train = [q for q in rows if q in pilot_ids]
    all_ids = list(rows)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform([rows[q]["question"] for q in train])
    y = [int(types[q] != "no_failure") for q in train]
    lr = LogisticRegression(C=1, solver="liblinear", random_state=0).fit(X, y)
    pred = lr.predict(vec.transform([rows[q]["question"] for q in all_ids]))
    return dict(zip(all_ids, map(bool, pred)))


def calibrated_signal():
    """One pooled classifier; train on 100 unique pilot questions only. Each
    question's five instances have total weight one. No model-specific fitting."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack
    bank = {b["id"]: b for b in map(json.loads, BANK.read_text().splitlines())}
    tune = set(json.loads((SPLITS/"qaqa_pilot_20261122.json").read_text())["ids"])
    qs, without, with_e, target, train, keys = [], [], [], [], [], []
    for model, (_, s, *_) in Q.SET.items():
        raw = Q.Raw(model, "qaqafull", str(BANK), WORK)
        forced = "conf:" + raw.e(s).split(":", 1)[1]
        raw.need(forced)
        raw.need("conf")
        for q in raw.ids:
            keys.append((model,q)); qs.append(bank[q]["question"])
            without.append(raw.text("conf",q)); with_e.append(raw.text(forced,q))
            train.append(q in tune)
            target.append(int(raw.qtype[q] != "no_failure") if q in tune else None)
    if sum(train) != 100*len(Q.SET):
        raise ValueError("unexpected development fit")
    used = [i for i, yes in enumerate(train) if yes]
    blocks = []
    for strings in (qs, without, with_e):
        vec = TfidfVectorizer(ngram_range=(1,2))
        vec.fit([strings[i] for i in used])
        blocks.append(vec.transform(strings))
    X = hstack(blocks,format="csr")
    lr = LogisticRegression(C=1,solver="liblinear",random_state=0)
    lr.fit(X[used], [target[i] for i in used],sample_weight=[1/len(Q.SET)]*len(used))
    return dict(zip(keys,map(bool,lr.predict(X))))


def analyze(model, cohort, model_signal):
    R = Q.Raw(model, "qaqafull", str(BANK), WORK)
    pilot = json.loads((SPLITS / "qaqa_pilot_20261122.json").read_text())
    choice = json.loads((SPLITS / "qaqa_tune_choice_fixedE_20261122.json").read_text())[model]
    tune_ids = set(pilot["ids"])
    if len(R.ids) != 570 or len(set(R.ids)) != 570 or len(tune_ids) != 100 or not tune_ids.issubset(R.ids):
        raise ValueError("unexpected roster")
    ids = [q for q in R.ids if (q in tune_ids) == (cohort == "tune")]
    _, s, _, _, _, _ = Q.SET[model]
    if choice["n1_strength"] != s:
        raise ValueError("expected fixed E")
    e = R.e(s)
    forced = "conf:" + e.split(":", 1)[1]
    ladder = [R.ec(s, k) for k in Q.LAD]
    for arm in [e, forced, "conf", *ladder]:
        R.need(arm)
    scores = R.confidence("qaqafull", forced)
    if not scores:
        raise ValueError("no strong-E confidence")
    n1, answered, _ = R.n1_system(s, choice["n1_cap_e"], choice["n1_cap_own"])
    e_values, e_answers = R.arm_system(e)
    blanks = [q for q in ids if not e_answers[q]]
    k = sum(answered[q] for q in blanks)
    possible = {}
    first = {}
    for q in blanks:
        arms = [a for a in ladder if not R.blank(a, q)]
        if not arms:
            continue
        # Upper bound allows every rung, including ones that N1 would veto.
        possible[q] = max(R.cell(a, q) - e_values[q] for a in arms)
        # The practical replay keeps the existing N1 veto and stops at the first answer.
        a = arms[0]
        if R.blank("x", q) or Q.ro.norm(R.text(a, q)) != Q.ro.norm(R.text("x", q)):
            first[q] = a
    ranked = [q for q in ids if not R.blank(forced, q) and scores[q] is not None]
    if any(not math.isfinite(scores[q]) for q in ranked):
        raise ValueError("non-finite confidence")
    ranked.sort(key=lambda q: -scores[q])  # stable in bank order
    cb = [q for q in ranked if q in first]
    eb = [q for q in ranked if not e_answers[q]]
    n = sum(answered[q] for q in ids)
    if len(cb) < k or len(eb) < k or len(ranked) < n:
        raise ValueError("target N unavailable; do not compare")
    csel, esel, whole = set(cb[:k]), set(eb[:k]), set(ranked[:n])
    effects = {}
    for q in ids:
        c = R.cell(first[q], q) if q in csel else e_values[q]
        ef = R.cell(forced, q) if q in esel else e_values[q]
        strong = R.cell(forced, q) if q in whole else 0
        effects[q] = (c-strong, c-ef, c-n1[q], n1[q]-ef, n1[q]-e_values[q])
    if sum(e_answers[q] for q in ids) + len(csel) != n:
        raise ValueError("mismatched N")
    bounds = {}
    for typ in [None, *[t for t, _ in Q.TYPES]]:
        group = [q for q in blanks if typ is None or R.qtype[q] == typ]
        used = [q for q in group if answered[q]]
        gains = sorted((possible[q] for q in group if q in possible), reverse=True)
        if len(gains)<len(used):
            raise ValueError("missing candidate for a released question")
        bounds[typ or "all"] = (len(used), sum(n1[q]-e_values[q] for q in used),
                                 sum(possible[q] for q in used), sum(gains[:len(used)]))
    # Three retrospective, untuned checks on a reply N1 actually releases.
    # For every rejected release the original E ticket/text is restored.
    text_gate = question_gate({b["id"]: b for b in map(json.loads, BANK.read_text().splitlines())}, R.qtype, tune_ids)
    checks = {}
    for label in ("agree_with_forced_E", "E_changes_forced_text", "negation_in_forced_E", "question_only_100", "question_and_model_answers", "gold_type_diagnostic"):
        retained = {}
        for q in blanks:
            if not answered[q]:
                continue
            cap = choice["n1_cap_own" if R.blank("x", q) else "n1_cap_e"]
            source = next(a for k0, a in zip(Q.LAD, ladder)
                          if cap is not None and float(k0) <= float(cap) and not R.blank(a, q))
            assert R.cell(source, q) == n1[q]
            ftxt, btxt = R.text(forced, q), R.text("conf", q)
            conditions = {
                "agree_with_forced_E": not R.blank(forced, q) and Q.ro.norm(R.text(source,q)) == Q.ro.norm(ftxt),
                "E_changes_forced_text": not R.blank(forced,q) and not R.blank("conf",q) and Q.ro.norm(ftxt) != Q.ro.norm(btxt),
                "gold_type_diagnostic": R.qtype[q] != "no_failure",
                "question_only_100": text_gate[q],
                "question_and_model_answers": model_signal[model,q],
                "negation_in_forced_E": bool(NEGATIONS.search(ftxt)),
            }
            if conditions[label]:
                retained[q] = n1[q]
        required = sum(e_answers[q] for q in ids) + len(retained)
        if required > len(ranked):
            raise ValueError("coverage unreachable")
        kept = set(ranked[:required])
        delta = {q: retained.get(q, e_values[q]) - (R.cell(forced,q) if q in kept else 0) for q in ids}
        fair = None
        if label == "question_and_model_answers":
            eligible = [q for q in eb if model_signal[model,q]]
            if len(eligible) >= len(retained):
                fill = set(eligible[:len(retained)])
                fair = {q: retained.get(q,e_values[q]) - (R.cell(forced,q) if q in fill else e_values[q]) for q in ids}
        checks[label] = (required, delta, fair)
    type_accuracy = sum(text_gate[q] == (R.qtype[q] != "no_failure") for q in ids)
    return effects, bounds, n, checks, type_accuracy


def main():
    model_signal = calibrated_signal()
    for cohort in ("tune", "test"):
        results = [analyze(m, cohort, model_signal) for m in Q.SET]
        ids = list(results[0][0])
        if any(list(r[0]) != ids for r in results):
            raise ValueError("models on different cohorts")
        print("question-only classification correct",results[0][4],"of",len(ids))
        labels = ("C-pool vs full strong E", "C-pool vs E-only filling same K blanks",
                  "C-pool vs original N1", "original N1 vs E-only filling same K blanks", "original N1 vs E")
        print(cohort, "questions", len(ids), "actual answers per model", [r[2] for r in results])
        for j, label in enumerate(labels):
            d = [sum(r[0][q][j] for r in results) for q in ids]
            print(label, Q.S.ps.ci(d))
        for label in results[0][3]:
            total_n = sum(r[3][label][0] for r in results)
            delta = [sum(r[3][label][1][q] for r in results) for q in ids]
            print(label, "total answered", total_n, "vs same-N strong E", Q.S.ps.ci(delta))
            if label == "question_and_model_answers" and all(r[3][label][2] is not None for r in results):
                fair = [sum(r[3][label][2][q] for r in results) for q in ids]
                print("same question-type signal, single E filling the same number of blanks:",Q.S.ps.ci(fair))
        for typ in results[0][1]:
            print(typ, "releases / realized gain / gold-best text on same IDs / gold-best IDs and text:",
                  [sum(r[1][typ][j] for r in results) for j in range(4)])


if __name__ == "__main__":
    main()
