#!/usr/bin/env python3
# Iteration_01 exploration (executor shen, Claude Opus 5.5, 2026-11-26), work queue
# discussion/work_queue/2026-11-26_queue_p.md, section 2: training the per-question switch ("gate"). CPU only (sklearn),
# reads the captures of diag/gate_capture_1126.py; no model, no judge call. Not reviewed by GPT.
# Classifier: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty L2, solver "liblinear",
#   random_state=0)), default max_iter; features = the block output of one layer at the last prompt token (float32,
#   cast to float64); the standardisation is fit on the training rows only (inside each fold: on the four training folds).
# Training pools (label 1 = trap):
#   A ("jia"): FalseQA official TRAIN split only (1,187 pairs: label-1 member = trap, label-0 member = plain).
#   B ("yi"):  A plus trap = TruthfulQA train 96 + tune 192 (the run's frozen lists) + the 50 (QA)^2 pilot questions of
#              type failure / unverifiable; plain = the 50 (QA)^2 pilot questions of type no_failure, PopQA train 300 +
#              tune 192, TriviaQA train 300 + tune 192, the 500 drawn MMLU items, the 300 drawn GSM8K train items.
# Reading-only sets (never trained on): FalseQA held-out 600, (QA)^2 held-out 470 (full bank minus the 100 pilot
#   questions), TruthfulQA test 529 (the frozen list; the simulation reads its main list), PopQA test / test2, TriviaQA
#   test / test2, MMLU 1,000, GSM8K 304. Asserted: no reading question (dataset-qualified id, or sha1 of the normalised
#   question / user text, across all datasets) is in a training pool.
# Layer choice: 5-fold cross-validation inside the pool, sklearn StratifiedGroupKFold(n_splits=5, shuffle=True,
#   random_state=0) with groups = the FalseQA pair (both questions of a pair always in the same fold) and one group per
#   question elsewhere, stratified by label as that splitter does; score = mean of the five fold AUROCs (the pooled
#   out-of-fold AUROC is printed too); highest mean wins, ties to the lower layer. Then the pipeline is refit on the whole
#   pool at the chosen layer, at the model's E layer and at its C layer (E / C: llama 16 / 19, qwen3.5-9b 15 / 16,
#   qwen3.8-27b 28 / 33, gemma3-12b 24 / 25, gemma3-27b 28 / 31, as diag/qaqa_pilot_table.SET), and applied to every
#   reading set. Decision (made by the simulation): trap = probability > 0.5; no threshold is tuned.
# Output: Iteration_01/newsets/gate_1126/<model>/fit_<A|B>_gate_1126.json (pool sizes by source, folds, CV table, the
#   three refits, per-question probabilities of every reading set at the three layers); never overwritten.
# usage (container, from /workspace/Truth_Decoupling):
#   python3 -I Iteration_01/diag/gate_fit_1126.py fit MODEL [WORKERS]
#   python3 -I Iteration_01/diag/gate_fit_1126.py report        (text report of every fit file, to stdout)
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

PROJ = Path("/workspace/Truth_Decoupling")
IT = PROJ / "Iteration_01"
OUT = IT / "newsets" / "gate_1126"
QAQA_PILOT = IT / "newsets" / "bank" / "qaqa_pilot_bank_20261122.jsonl"
MODELS = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
EC = {"llama3.1-8b": (16, 19), "qwen3.5-9b": (15, 16), "qwen3.8-27b": (28, 33), "gemma3-12b": (24, 25), "gemma3-27b": (28, 31)}
FAMILY = {"fqa_train": "fqa", "fqa_held": "fqa", "qaqa": "qaqa", "tq_train": "tq", "tq_tune": "tq", "tq_test": "tq",
          "popqa_train": "popqa", "popqa_tune": "popqa", "popqa_test": "popqa", "popqa_test2": "popqa",
          "triviaqa_train": "triviaqa", "triviaqa_tune": "triviaqa", "triviaqa_test": "triviaqa", "triviaqa_test2": "triviaqa",
          "mmlu_test": "mmlu", "mmlu_extra": "mmlu", "gsm8k_test": "gsm8k", "gsm8k_train": "gsm8k"}
# (set, part, label): part "all" = every row; "pilot_trap" / "pilot_plain" / "held" = (QA)^2 subsets
POOLS = {"A": [("fqa_train", "all", None)],
         "B": [("fqa_train", "all", None), ("tq_train", "all", 1), ("tq_tune", "all", 1), ("qaqa", "pilot_trap", 1),
               ("qaqa", "pilot_plain", 0), ("popqa_train", "all", 0), ("popqa_tune", "all", 0), ("triviaqa_train", "all", 0),
               ("triviaqa_tune", "all", 0), ("mmlu_extra", "all", 0), ("gsm8k_train", "all", 0)]}
READ = [("fqa_held", "all"), ("qaqa", "held"), ("tq_test", "all"), ("popqa_test", "all"), ("popqa_test2", "all"),
        ("triviaqa_test", "all"), ("triviaqa_test2", "all"), ("mmlu_test", "all"), ("gsm8k_test", "all")]
EXPECT = {"A": (1187, 1187), "B": (1187 + 96 + 192 + 50, 1187 + 50 + 492 + 492 + 500 + 300)}
NFOLD, CV_SEED = 5, 0
_G = {}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def npz(model, name):
    return OUT / model / ("%s_gate_1126.npz" % name)


def pilot():
    with open(QAQA_PILOT, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if len(rows) != 100:
        raise SystemExit("pilot bank")
    return {r["id"]: r["qtype"] for r in rows}


def select(meta_of, name, part, pil):
    """Row indices of one set part."""
    keys = meta_of[name]["keys"]
    if part == "all":
        return list(range(len(keys)))
    if part == "held":
        return [i for i, k in enumerate(keys) if k not in pil]
    want = part == "pilot_trap"
    return [i for i, k in enumerate(keys) if k in pil and (pil[k] != "no_failure") == want]


def load_meta(model):
    import numpy as np
    meta = {}
    for name in FAMILY:
        with np.load(npz(model, name), allow_pickle=False) as z:
            meta[name] = dict(keys=[str(k) for k in z["keys"]], text=[str(t) for t in z["text_sha1"]],
                              labels=z["labels"].astype(int).tolist(), groups=[str(g) for g in z["groups"]])
    return meta


def pool_rows(meta, pool, pil):
    rows = []        # (set, index, label, group)
    for name, part, lab in POOLS[pool]:
        for i in select(meta, name, part, pil):
            y = meta[name]["labels"][i] if lab is None else lab
            if y not in (0, 1):
                raise SystemExit("%s: no label" % name)
            g = meta[name]["groups"][i] if name == "fqa_train" else "%s|%s" % (name, meta[name]["keys"][i])
            rows.append((name, i, y, g))
    return rows


def check_disjoint(meta, rows, pil):
    tr_keys = {"%s|%s" % (FAMILY[n], meta[n]["keys"][i]) for n, i, _, _ in rows}
    tr_text = {meta[n]["text"][i] for n, i, _, _ in rows}
    bad_k = bad_t = 0
    n_read = 0
    for name, part in READ:
        for i in select(meta, name, part, pil):
            n_read += 1
            bad_k += ("%s|%s" % (FAMILY[name], meta[name]["keys"][i])) in tr_keys
            bad_t += meta[name]["text"][i] in tr_text
    if len(tr_keys) != len(rows):
        raise SystemExit("a training question appears twice in the pool")
    if bad_k or bad_t:
        raise SystemExit("reading questions in the training pool: %d by id, %d by text" % (bad_k, bad_t))
    return n_read


def _cv_layer(args):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import roc_auc_score
    pool, l = args
    X = _G["X"][pool][:, l, :].astype(np.float64)
    y = _G["y"][pool]
    fa, oof, nw, its = [], np.zeros(len(y)), 0, []
    for a, b in _G["folds"][pool]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = make().fit(X[a], y[a])
        nw += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        its.append(int(np.ravel(clf[-1].n_iter_)[0]))
        p = clf.predict_proba(X[b])[:, 1]
        oof[b] = p
        fa.append(float(roc_auc_score(y[b], p)))
    return pool, l, dict(layer=l, mean=float(np.mean(fa)), folds=fa, pooled=float(roc_auc_score(y, oof)),
                         convergence_warnings=nw, n_iter=its)


def make():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", random_state=0))


def run_fit(model, workers):
    import multiprocessing as mp
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    t0 = time.time()
    for pool in POOLS:
        if (OUT / model / ("fit_%s_gate_1126.json" % pool)).exists():
            raise SystemExit("fit_%s_gate_1126.json exists" % pool)
    cap = json.load(open(OUT / model / "capture_gate_1126.json", encoding="utf-8"))
    for name in FAMILY:
        if cap["sets"][name]["npz_sha256"] != sha_file(npz(model, name)):
            raise SystemExit("%s: the npz differs from the capture record" % name)
    L = cap["layers"]
    el, cl = EC[model]
    pil = pilot()
    meta = load_meta(model)
    if set(pil) - set(meta["qaqa"]["keys"]):
        raise SystemExit("pilot ids not in the (QA)^2 capture")
    rows, Xs, ys, folds, info = {}, {}, {}, {}, {}
    for pool in POOLS:
        r = pool_rows(meta, pool, pil)
        n_read = check_disjoint(meta, r, pil)
        y = np.array([x[2] for x in r])
        if (int((y == 1).sum()), int((y == 0).sum())) != EXPECT[pool]:
            raise SystemExit("pool %s: %d trap / %d plain, expected %s" % (pool, (y == 1).sum(), (y == 0).sum(), EXPECT[pool]))
        X = np.zeros((len(r), L, cap["hidden"]), dtype=np.float32)
        k = 0
        for name, part, _ in POOLS[pool]:
            idx = select(meta, name, part, pil)
            with np.load(npz(model, name), allow_pickle=False) as z:
                X[k:k + len(idx)] = z["feats"][idx]
            k += len(idx)
        groups = np.array([x[3] for x in r])
        fl = list(StratifiedGroupKFold(n_splits=NFOLD, shuffle=True, random_state=CV_SEED).split(np.zeros(len(y)), y, groups))
        for a, b in fl:
            if set(groups[a]) & set(groups[b]):
                raise SystemExit("a group is split across folds")
        by_src = {}
        for name, part, lab in POOLS[pool]:
            yy = [meta[name]["labels"][i] if lab is None else lab for i in select(meta, name, part, pil)]
            by_src["%s%s" % (name, "" if part == "all" else ":" + part)] = dict(trap=sum(yy), plain=len(yy) - sum(yy))
        info[pool] = dict(n=len(r), trap=int(y.sum()), plain=int(len(y) - y.sum()), by_source=by_src, reading_rows=n_read,
                          folds=[dict(train=len(a), test=len(b), test_trap=int(y[b].sum())) for a, b in fl])
        print("%s pool %s: %d rows, trap %d, plain %d | %s | folds (test size / test trap): %s | reading rows checked disjoint: %d" % (
            model, pool, len(r), y.sum(), len(y) - y.sum(), ", ".join("%s %d/%d" % (k_, v["trap"], v["plain"]) for k_, v in by_src.items()),
            " ".join("%d/%d" % (f["test"], f["test_trap"]) for f in info[pool]["folds"]), n_read), flush=True)
        rows[pool], Xs[pool], ys[pool], folds[pool] = r, X, y, fl
    _G.update(X=Xs, y=ys, folds=folds)
    tasks = [(pool, l) for pool in POOLS for l in range(L)]
    cv = {pool: [None] * L for pool in POOLS}
    with mp.get_context("fork").Pool(workers) as P:
        for pool, l, res in P.imap_unordered(_cv_layer, tasks):
            cv[pool][l] = res
    for pool in POOLS:
        for c in cv[pool]:
            print("%s pool %s layer %2d: 5-fold AUROC mean %.4f (folds %s), pooled %.4f, convergence warnings %d" % (
                model, pool, c["layer"], c["mean"], " ".join("%.3f" % v for v in c["folds"]), c["pooled"], c["convergence_warnings"]), flush=True)
    reads = {}
    for name, part in READ:
        reads[(name, part)] = select(meta, name, part, pil)
    for pool in POOLS:
        best = max(c["mean"] for c in cv[pool])
        sel = min(c["layer"] for c in cv[pool] if c["mean"] == best)
        refit, probs = {}, {}
        clfs = {}
        for tag, l in (("chosen", sel), ("E", el), ("C", cl)):
            X = Xs[pool][:, l, :].astype(np.float64)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf = make().fit(X, ys[pool])
            pt = clf.predict_proba(X)[:, 1]
            refit[tag] = dict(layer=l, train_accuracy=float(((pt > 0.5) == ys[pool]).mean()), train_auroc=float(roc_auc_score(ys[pool], pt)),
                              n_iter=int(np.ravel(clf[-1].n_iter_)[0]), convergence_warnings=sum(issubclass(x.category, ConvergenceWarning) for x in w),
                              coef_norm=float(np.linalg.norm(clf[-1].coef_)))
            clfs[tag] = (l, clf)
            probs[tag] = {}
        for (name, part), idx in reads.items():
            with np.load(npz(model, name), allow_pickle=False) as z:
                F = z["feats"][idx]
            for tag, (l, clf) in clfs.items():
                p = clf.predict_proba(F[:, l, :].astype(np.float64))[:, 1]
                probs[tag]["%s%s" % (name, "" if part == "all" else ":" + part)] = {meta[name]["keys"][i]: float(v) for i, v in zip(idx, p)}
            del F
        doc = dict(model=model, pool=pool, layers=L, e_layer=el, c_layer=cl, chosen_layer=sel, chosen_cv_mean=best,
                   rule=dict(classifier="StandardScaler + LogisticRegression(C=1, L2, liblinear, random_state=0), default max_iter",
                             folds="StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0); groups = FalseQA pair, one group per question elsewhere",
                             score="mean of the five fold AUROCs", ties="lower layer", trap="probability > 0.5"),
                   pool_info=info[pool], cv=cv[pool], refit=refit, probs=probs,
                   npz_sha256={n: cap["sets"][n]["npz_sha256"] for n in FAMILY}, capture_sha256=sha_file(OUT / model / "capture_gate_1126.json"),
                   program_sha256=sha_file(__file__))
        with open(OUT / model / ("fit_%s_gate_1126.json" % pool), "x", encoding="utf-8") as f:
            json.dump(doc, f)
        print("%s pool %s: chosen layer %d (mean 5-fold AUROC %.4f); E layer %d %.4f; C layer %d %.4f; refits %s" % (
            model, pool, sel, best, el, cv[pool][el]["mean"], cl, cv[pool][cl]["mean"],
            " | ".join("%s L%d train acc %.3f, AUROC %.3f, iterations %d, warnings %d" % (t, v["layer"], v["train_accuracy"], v["train_auroc"], v["n_iter"], v["convergence_warnings"])
                       for t, v in refit.items())), flush=True)
    print("%s: fit done in %.0fs" % (model, time.time() - t0), flush=True)


def report():
    print("# gate_1126 fit report (work queue 2026-11-26_queue_p, section 2): linear switch on the last prompt token, "
          "per model and training pool; CV = StratifiedGroupKFold(5, shuffle, random_state 0), groups = FalseQA pairs")
    for model in MODELS:
        for pool in POOLS:
            p = OUT / model / ("fit_%s_gate_1126.json" % pool)
            d = json.load(open(p, encoding="utf-8"))
            pi = d["pool_info"]
            print("\n## %s, pool %s (%s): %d rows, trap %d, plain %d; sha256 %s" % (
                model, pool, {"A": "FalseQA train only", "B": "FalseQA train + other unused training questions"}[pool],
                pi["n"], pi["trap"], pi["plain"], sha_file(p)[:16]))
            print("   by source (trap / plain): " + ", ".join("%s %d / %d" % (k, v["trap"], v["plain"]) for k, v in pi["by_source"].items()))
            print("   folds (train / test / test trap): " + " | ".join("%d / %d / %d" % (f["train"], f["test"], f["test_trap"]) for f in pi["folds"]))
            print("   chosen layer %d, mean 5-fold AUROC %.4f | E layer %d %.4f | C layer %d %.4f" % (
                d["chosen_layer"], d["chosen_cv_mean"], d["e_layer"], d["cv"][d["e_layer"]]["mean"], d["c_layer"], d["cv"][d["c_layer"]]["mean"]))
            print("   CV mean by layer: " + " ".join("L%d %.3f" % (c["layer"], c["mean"]) for c in d["cv"]))
            print("   CV pooled by layer: " + " ".join("L%d %.3f" % (c["layer"], c["pooled"]) for c in d["cv"]))
            print("   convergence warnings in CV (of %d fits): %d" % (5 * d["layers"], sum(c["convergence_warnings"] for c in d["cv"])))
            for t, v in d["refit"].items():
                print("   refit on the whole pool, %s layer %d: train accuracy %.3f, train AUROC %.3f, liblinear iterations %d, convergence warnings %d" % (
                    t, v["layer"], v["train_accuracy"], v["train_auroc"], v["n_iter"], v["convergence_warnings"]))


def main():
    if len(sys.argv) in (3, 4) and sys.argv[1] == "fit" and sys.argv[2] in MODELS:
        run_fit(sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 48)
    elif sys.argv[1:] == ["report"]:
        report()
    else:
        raise SystemExit("usage: gate_fit_1126.py fit MODEL [WORKERS] | report")


if __name__ == "__main__":
    main()
