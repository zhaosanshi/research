#!/usr/bin/env python3
"""E7 post-hoc: honest best-layer comparison between direction estimators (e.g. all-layer DAS vs all-layer mean-diff).

For each estimator sweep (an E7 run with one --tag, per-item outputs in multilayer_<tag>.npz) the single-layer conditions
amp_L{l}_a{alpha} are candidates. The best candidate is chosen on half A of the test pairs (pair-hash halves, the same
split E11/E17 use) and its gain is REPORTED on half B, and the other way round; the two held-out gains are averaged.
Selection is repeated inside every bootstrap replicate (pairs resampled), so the CI includes the selection noise.
The optimistic "select and report on everything" number is printed too, clearly labelled.

usage: python3 scripts/e7_select_layer.py --model llama3.1-8b --tags das_all meandiff_all [--metric auroc|acc|penalised] [--n-boot 500]
reads  results/E7/<model>/multilayer_<tag>.npz      writes results/E7/<model>/select_layer.{md,json}
"""
import argparse
import os
import re

import numpy as np

from common import RESULTS_DIR
from core.io_utils import atomic_json_dump
from core.metrics import auroc

COND_RE = re.compile(r"^amp_(?:L(\d+)|multi_([0-9+]+))_a([0-9.]+)$")   # single-layer amp_L13_a5 or joint amp_multi_12+13+14_a3


def cond_key(name):
    """-> (layer label, alpha string) for a candidate condition name."""
    m = COND_RE.match(name)
    return (m.group(1) or m.group(2)), m.group(3)


def load_sweep(model, tag):
    """`tag` may join several E7 runs with '+' (e.g. das_all+das_hi): their candidate conditions are pooled; the runs must
    share items (same eval split / order) and the baseline of the first run is used."""
    out = None
    for t in tag.split("+"):
        f = os.path.join(RESULTS_DIR, "E7", model, "multilayer_%s.npz" % t)
        z = np.load(f, allow_pickle=False)
        names = [str(n) for n in z["cond_names"]]
        cands = [n for n in names if COND_RE.match(n)]
        if out is None:
            out = {"labels": z["labels"].astype(bool), "half_a": z["half_a"].astype(bool), "pair_ids": z["pair_ids"],
                   "p": {n: z["p__" + n] for n in names}, "pred": {n: z["pred__" + n] for n in names}, "cands": list(cands)}
        else:
            if not np.array_equal(z["ids"], np.load(os.path.join(RESULTS_DIR, "E7", model, "multilayer_%s.npz" % tag.split("+")[0]))["ids"]):
                raise SystemExit("item order differs between runs %s" % tag)
            for n in cands:
                if n in out["p"]:
                    continue
                out["p"][n] = z["p__" + n]
                out["pred"][n] = z["pred__" + n]
                out["cands"].append(n)
    return out


def score(metric, p, pred, labels, idx):
    """metric on the items idx. auroc on P(True|answered); acc = correct / n (abstain counts as not correct);
    penalised = (correct - wrong) / n."""
    lab = labels[idx]
    if metric == "auroc":
        return auroc(p[idx], lab)
    pr = pred[idx]
    correct = ((pr == "True") & lab) | ((pr == "False") & ~lab)
    wrong = ((pr == "True") & ~lab) | ((pr == "False") & lab)
    if metric == "acc":
        return float(correct.mean())
    return float((correct.sum() - wrong.sum()) / len(idx))


def select_and_report(sw, metric, idx_sel, idx_rep):
    """best candidate on idx_sel by gain over baseline; returns (name, gain on idx_rep)."""
    base_sel = score(metric, sw["p"]["baseline"], sw["pred"]["baseline"], sw["labels"], idx_sel)
    best, best_gain = None, -np.inf
    for n in sw["cands"]:
        g = score(metric, sw["p"][n], sw["pred"][n], sw["labels"], idx_sel) - base_sel
        if g > best_gain:
            best, best_gain = n, g
    base_rep = score(metric, sw["p"]["baseline"], sw["pred"]["baseline"], sw["labels"], idx_rep)
    return best, score(metric, sw["p"][best], sw["pred"][best], sw["labels"], idx_rep) - base_rep


def crossfit(sw, metric, idx_all):
    a = idx_all[sw["half_a"][idx_all]]
    b = idx_all[~sw["half_a"][idx_all]]
    na, ga = select_and_report(sw, metric, a, b)
    nb, gb = select_and_report(sw, metric, b, a)
    return na, ga, nb, gb, 0.5 * (ga + gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tags", nargs="+", required=True, help="E7 run tags to compare, e.g. das_all meandiff_all")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "acc", "penalised"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single-only", action="store_true", help="candidates = single-layer conditions only (drop amp_multi_*)")
    ap.add_argument("--out-suffix", default="", help="appended to select_layer_<metric> output names")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sweeps = {t: load_sweep(args.model, t) for t in args.tags}
    if args.single_only:
        for sw in sweeps.values():
            sw["cands"] = [c for c in sw["cands"] if c.startswith("amp_L")]
    ref = sweeps[args.tags[0]]
    n = len(ref["labels"])
    idx_all = np.arange(n)
    pairs = {}
    for i, pid in enumerate(ref["pair_ids"]):
        pairs.setdefault(str(pid), []).append(i)
    pair_lists = list(pairs.values())

    out = {"model": args.model, "metric": args.metric, "n": n, "n_pairs": len(pair_lists), "tags": {}}
    held = {}
    for t, sw in sweeps.items():
        na, ga, nb, gb, mean_held = crossfit(sw, args.metric, idx_all)
        # optimistic: select on all, report on all
        n_all, g_all = select_and_report(sw, args.metric, idx_all, idx_all)
        boots = []
        for _ in range(args.n_boot):
            pick = rng.integers(0, len(pair_lists), len(pair_lists))
            idx = np.concatenate([pair_lists[k] for k in pick])
            boots.append(crossfit(sw, args.metric, idx)[4])
        boots = np.array(boots)
        held[t] = boots
        out["tags"][t] = {"select_A_report_B": {"cond": na, "gain": ga}, "select_B_report_A": {"cond": nb, "gain": gb},
                          "held_out_gain": mean_held, "held_out_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                          "optimistic_all": {"cond": n_all, "gain": g_all},
                          "profile": {n_: score(args.metric, sw["p"][n_], sw["pred"][n_], sw["labels"], idx_all)
                                      - score(args.metric, sw["p"]["baseline"], sw["pred"]["baseline"], sw["labels"], idx_all) for n_ in sw["cands"]}}
    if len(args.tags) >= 2:
        t0, t1 = args.tags[0], args.tags[1]
        d = held[t0] - held[t1]
        out["head_to_head"] = {"%s_minus_%s" % (t0, t1): {"mean": float(np.mean(d)), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                                                        "p_first_better": float(np.mean(d > 0))}}

    out_dir = os.path.join(RESULTS_DIR, "E7", args.model)
    stem = "select_layer_" + args.metric + args.out_suffix
    atomic_json_dump(out, os.path.join(out_dir, stem + ".json"))
    lines = ["# E7 best-layer comparison, %s (metric %s, n=%d items / %d pairs, %d bootstraps)" % (args.model, args.metric, n, len(pair_lists), args.n_boot), "",
             "Select the best single layer x alpha on one half of the pairs, report its gain on the other half (both directions, averaged).", "",
             "| estimator | chosen on A -> gain on B | chosen on B -> gain on A | held-out gain [95% CI] | optimistic (select+report on all) |", "|---|---|---|---|---|"]
    for t, r in out["tags"].items():
        lines.append("| %s | %s -> %+.3f | %s -> %+.3f | **%+.3f** [%+.3f, %+.3f] | %s %+.3f |" % (
            t, r["select_A_report_B"]["cond"], r["select_A_report_B"]["gain"], r["select_B_report_A"]["cond"], r["select_B_report_A"]["gain"],
            r["held_out_gain"], r["held_out_ci95"][0], r["held_out_ci95"][1], r["optimistic_all"]["cond"], r["optimistic_all"]["gain"]))
    if "head_to_head" in out:
        for k, v in out["head_to_head"].items():
            lines += ["", "Head-to-head (held-out gain difference) %s: %+.3f [%+.3f, %+.3f], P(first better) = %.2f" % (k, v["mean"], v["ci95"][0], v["ci95"][1], v["p_first_better"])]
    lines += ["", "## Layer profiles (gain over baseline on all items)", ""]
    for t, r in out["tags"].items():
        by_alpha = {}
        for cond, g in r["profile"].items():
            layer, a = cond_key(cond)
            by_alpha.setdefault(a, []).append((layer, g))
        for a, prof in sorted(by_alpha.items(), key=lambda kv: float(kv[0])):
            prof.sort(key=lambda p: (len(p[0]) > 2, int(p[0].split("+")[0])))
            lines.append("- %s alpha %s: %s" % (t, a, " ".join("L%s %+.3f" % p for p in prof)))
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
