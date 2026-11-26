"""Post hoc CPU-only lexical baseline for the fixed stage-one question roster, no model generation or LLM judge."""

import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/capability",
]
import gate_capture_1126 as cap
import gate_fit_1126 as fit
import falseqa_read as fr


def ci(d):
    rng = np.random.default_rng(20261126)
    n = len(d)
    vals = np.asarray(d)[rng.integers(0, n, (4000, n))].sum(1)
    a, b = np.percentile(vals, [2.5, 97.5])
    return f"{int(sum(d))} [{int(a)}, {int(b)}]"


def main():
    model = "llama3.1-8b"
    m = json.load(open(cap.FROZEN_MODELS))["models"][model]
    sets = cap.items_of(model, m)
    meta = {
        name: {
            "keys": [r[0] for r in doc["items"]],
            "text": [cap.tsha(r[1]) for r in doc["items"]],
            "labels": [r[2] for r in doc["items"]],
            "groups": [r[3] for r in doc["items"]],
        }
        for name, doc in sets.items()
    }
    pilot = fit.pilot()
    gates = {}
    for pool in ("A", "B"):
        rows = fit.pool_rows(meta, pool, pilot)
        fit.check_disjoint(meta, rows, pilot)
        docs = [sets[n]["items"][i][1] for n, i, y, g in rows]
        y = [r[2] for r in rows]
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=1.0, solver="liblinear", random_state=0),
        )
        clf.fit(docs, y)
        for name, part in fit.READ:
            ii = fit.select(meta, name, part, pilot)
            subset = [sets[name]["items"][i] for i in ii]
            keys = [r[0] for r in subset]
            orig = [r[1] for r in subset]
            labels = [r[2] for r in subset]
            probs = clf.predict_proba(orig)[:, 1]
            decisions = probs > 0.5
            print(
                pool,
                name,
                "n",
                len(ii),
                "open",
                int(sum(decisions)),
                "accuracy_auc",
                (
                    round(np.mean(decisions == labels), 3),
                    round(roc_auc_score(labels, probs), 3),
                )
                if name in ("fqa_held", "qaqa")
                else "-",
            )
            gates[pool, name] = {q: bool(v) for q, v in zip(keys, decisions)}
    ok, bank = fr.check_bank()
    assert ok
    ids = [r["id"] for r in bank if r["set"] == "heldout"]
    dev = {r["id"] for r in bank if r["set"] == "dev"}
    pairs = {r["id"]: r["pair_id"] for r in bank}
    for m in fit.MODELS:
        setting = fr.setting(m)
        tune, tune2 = fr.load_rows(m, dev)
        configs, _ = fr.choose(tune, tune2, setting)
        rows, rows2 = fr.load_rows(m, set(ids))
        arms = fr.systems_of(rows, rows2, setting, configs, ids)
        fitted = json.load(open(fit.OUT / m / "fit_B_gate_1126.json"))
        actual = {q: v > 0.5 for q, v in fitted["probs"]["chosen"]["fqa_held"].items()}

        def diff(g1, g2):
            paired = {}
            for q in ids:
                score = lambda g: arms["N1" if g[q] else "O"][q][0]
                p = pairs[q]
                paired[p] = paired.get(p, 0) + score(g1) - score(g2)
            return ci(list(paired.values()))

        for pool in ("A", "B"):
            g = gates[pool, "fqa_held"]
            always = dict.fromkeys(ids, True)
            print(
                m,
                pool,
                "text_gate_minus_ungated",
                diff(g, always),
                "representation_B_minus_text_gate",
                diff(actual, g),
            )


if __name__ == "__main__":
    main()
