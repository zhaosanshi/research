#!/usr/bin/env python3
"""A paired, retrospective calibration-size simulation using cached predictions.

Training directions and each model's complete v2 candidate pool stay frozen.
Within a repetition, all calibration sizes share the same 337 validation pairs
and use nested prefixes of a 384-pair calibration pool. The only changing input
to selection is calibration sample size. No model inference is performed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SELECTOR_PATH = HERE.parent / "from_claude/code/scripts/select_low_harm_policy.py"
FROZEN_PATH = HERE / "harm_policy_frozen_v2_20261101.json"
OLD_ROOT = HERE / "analysis_data/corrections_20261101"
LOW_ROOT = HERE / "analysis_data/low_strength_20261101"
OUTPUT_PATH = HERE / "calibration_size_study_20261101.json"
SIZES = (64, 128, 192, 288, 384)
REPEATS = 100
SEED = 20261101
VALIDATION_PAIRS = 337
POOL_PAIRS = 384


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selector():
    spec = importlib.util.spec_from_file_location("frozen_harm_selector", SELECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate_stats(corrected, harmed, baseline_correct, n_statements):
    corrected, harmed, baseline_correct = map(int, (corrected, harmed, baseline_correct))
    if baseline_correct == 0:
        raise ValueError("cannot estimate conditional harm with no baseline-correct items")
    return {"n": n_statements, "baseline_correct": baseline_correct,
            "corrected": corrected, "harmed": harmed, "net": corrected - harmed,
            "harm_rate": harmed / baseline_correct,
            "accuracy_gain_pp": 100 * (corrected - harmed) / n_statements,
            "accuracy_after": (baseline_correct + corrected - harmed) / n_statements}


def prepare_model(selector, model, frozen_model, old_root=OLD_ROOT, low_root=LOW_ROOT):
    data, metadata, hashes = selector.load_model_data(old_root, low_root, model)
    if hashes != frozen_model["input_hashes"]:
        raise ValueError("inputs differ from frozen v2 for " + model)
    policies = [row["policy"] for row in frozen_model["candidates_on_A"]]
    if len({p["name"] for p in policies}) != len(policies):
        raise ValueError("duplicate policy names")
    if sum(p["kind"] == "baseline" for p in policies) != 1:
        raise ValueError("expected one unchanged baseline candidate")
    raw = data["old"]
    labels = raw["labels"].astype(bool)
    ids = raw["ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate statement IDs")
    pair_ids = (raw["pair_ids"].astype(str) if "pair_ids" in raw
                else np.array([item.rsplit(":", 1)[0] for item in ids]))
    pairs = sorted(set(pair_ids))
    if len(pairs) != VALIDATION_PAIRS + POOL_PAIRS:
        raise ValueError("expected the same 721 previously evaluated question pairs")
    groups = [np.flatnonzero(pair_ids == pair) for pair in pairs]
    if any(len(g) != 2 or int(labels[g].sum()) != 1 for g in groups):
        raise ValueError("each question must contain exactly one true and one false statement")
    groups = np.stack(groups)
    base = raw["baseline__ptrue"]
    before = (base > .5) == labels
    n_correct = before[groups].sum(axis=1)
    mask = np.ones(len(labels), dtype=bool)
    outputs = np.stack([selector.predictions(data, policy, mask) for policy in policies])
    if not np.isfinite(outputs).all():
        raise ValueError("nonfinite cached predictions")
    after = (outputs > .5) == labels[None, :]
    corrected = ((~before)[None, :] & after)[:, groups].sum(axis=2)
    harmed = (before[None, :] & ~after)[:, groups].sum(axis=2)
    assert corrected.shape == harmed.shape == (len(policies), len(pairs))
    return {"policies": policies, "pairs": pairs, "groups": groups, "labels": labels,
            "base": base, "outputs": outputs, "baseline_correct": n_correct,
            "corrected": corrected, "harmed": harmed, "hashes": hashes}


def selection_rows(model, pair_indices):
    n = int(model["baseline_correct"][pair_indices].sum())
    corrected = model["corrected"][:, pair_indices].sum(axis=1)
    harmed = model["harmed"][:, pair_indices].sum(axis=1)
    return [{"policy": policy,
             "tune": aggregate_stats(corrected[i], harmed[i], n, 2 * len(pair_indices))}
            for i, policy in enumerate(model["policies"])]


def summarize(rows):
    harm = np.array([row["validation"]["harm_rate"] for row in rows])
    gain = np.array([row["validation"]["accuracy_gain_pp"] for row in rows])
    baseline = np.array([row["chosen_kind"] == "baseline" for row in rows])
    success = (harm <= .01) & (gain > 0)
    return {"repetitions": len(rows), "mean_validation_harm_rate": float(harm.mean()),
            "mean_validation_harm_percent": float(100 * harm.mean()),
            "mean_validation_net_accuracy_gain_pp": float(gain.mean()),
            "selected_baseline_fraction": float(baseline.mean()),
            "validation_harm_at_most_1pct_and_positive_gain_fraction": float(success.mean()),
            "split_variability_not_confidence_intervals": {
                "validation_harm_percent_p10_p90": (100 * np.quantile(harm, [.1, .9])).tolist(),
                "validation_gain_pp_p10_p90": np.quantile(gain, [.1, .9]).tolist()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--low-root", type=Path, default=LOW_ROOT)
    parser.add_argument("--frozen", type=Path, default=FROZEN_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite " + str(args.out))
    selector = load_selector()
    frozen = json.loads(args.frozen.read_text())
    cap = frozen["tune_harm_cap"]
    if cap != .005:
        raise ValueError("study must retain frozen 0.5% calibration harm cap")
    models = {name: prepare_model(selector, name, model, args.old_root, args.low_root)
              for name, model in frozen["models"].items()}
    pair_order = next(iter(models.values()))["pairs"]
    if any(model["pairs"] != pair_order for model in models.values()):
        raise ValueError("model datasets differ")
    rng = np.random.default_rng(SEED)
    partitions = [rng.permutation(len(pair_order)) for _ in range(REPEATS)]
    results = {"study": "retrospective simulation of calibration sample size only",
               "warning": "All 721 pairs were previously evaluated. This is not a new independent test. "
                          "Candidates include the A-adapted v2 confidence bands; direction training stays at 96 pairs. "
                          "Repeated splits overlap and are not independent replications.",
               "frozen_policy_sha256": sha256(args.frozen),
               "selector_script_sha256": sha256(SELECTOR_PATH),
               "study_script_sha256": sha256(Path(__file__)),
               "seed": SEED, "repetitions": REPEATS, "calibration_pair_sizes": list(SIZES),
               "validation_pairs_per_repetition": VALIDATION_PAIRS,
               "nested_calibration_pool_pairs": POOL_PAIRS, "harm_cap": cap,
               "candidate_pool": "all policies saved in frozen v2, unchanged at every size; baseline included",
               "selection": "unchanged selector: maximum net corrections within harm cap, "
                            "ties fewer harms, lower cost, lexical name",
               "pair_order": pair_order, "models": {}}
    for name, model in models.items():
        by_size = {str(size): [] for size in SIZES}
        policy_index = {p["name"]: i for i, p in enumerate(model["policies"])}
        direct_verified = False
        for repeat, permutation in enumerate(partitions):
            validation = permutation[:VALIDATION_PAIRS]
            pool = permutation[VALIDATION_PAIRS:]
            assert len(pool) == POOL_PAIRS and not set(validation) & set(pool)
            for size in SIZES:
                tune = pool[:size]
                rows = selection_rows(model, tune)
                chosen = selector.select(rows, cap)
                i = policy_index[chosen["policy"]["name"]]
                validation_stats = aggregate_stats(model["corrected"][i, validation].sum(),
                                                  model["harmed"][i, validation].sum(),
                                                  model["baseline_correct"][validation].sum(),
                                                  2 * VALIDATION_PAIRS)
                if repeat == 0 and size == 192:
                    tune_items = model["groups"][tune].ravel()
                    direct_rows = [{"policy": policy,
                                    "tune": selector.stats(model["labels"][tune_items],
                                                           model["base"][tune_items],
                                                           model["outputs"][j, tune_items])}
                                   for j, policy in enumerate(model["policies"])]
                    assert direct_rows == rows, "pair counts differ from per-statement computation"
                    assert selector.select(direct_rows, cap) == chosen
                    validation_items = model["groups"][validation].ravel()
                    direct = selector.stats(model["labels"][validation_items],
                                            model["base"][validation_items],
                                            model["outputs"][i, validation_items])
                    assert validation_stats == direct
                    direct_verified = True
                by_size[str(size)].append({"repetition": repeat, "chosen": chosen["policy"]["name"],
                                           "chosen_kind": chosen["policy"]["kind"],
                                           "calibration": chosen["tune"], "validation": validation_stats})
        results["models"][name] = {"candidate_count": len(model["policies"]),
                                   "input_hashes": model["hashes"],
                                   "direct_per_statement_equivalence_verified": direct_verified,
                                   "summary": {size: summarize(rows) for size, rows in by_size.items()},
                                   "repetitions": by_size}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
        out.write("\n")
    for name, result in results["models"].items():
        print(name, "candidates", result["candidate_count"], "direct_check",
              result["direct_per_statement_equivalence_verified"])
        print("N | mean_harm_percent | mean_net_gain_pp | baseline_percent | success_percent")
        for size in SIZES:
            row = result["summary"][str(size)]
            print("%d | %.3f | %.3f | %.1f | %.1f" % (
                size, row["mean_validation_harm_percent"], row["mean_validation_net_accuracy_gain_pp"],
                100 * row["selected_baseline_fraction"],
                100 * row["validation_harm_at_most_1pct_and_positive_gain_fraction"]))
    print("written", args.out)


if __name__ == "__main__":
    main()
