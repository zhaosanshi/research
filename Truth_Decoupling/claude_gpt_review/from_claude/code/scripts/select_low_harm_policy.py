#!/usr/bin/env python3
"""Fit a damage-limited choice on A, freeze it, then evaluate it on B.

Consumes existing per-item model outputs. The consensus candidate changes a
baseline label only when the two previously selected estimators both disagree
with it. It needs baseline + DAS + mean-difference inference at deployment.
No labels enter the prediction rule; A labels only select the rule/configuration.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np


MODELS = ("llama3.1-8b", "qwen3-8b")


def stats(labels, base, output):
    labels, base, output = np.asarray(labels, dtype=bool), np.asarray(base), np.asarray(output)
    if labels.ndim != 1 or not len(labels) or labels.shape != base.shape or labels.shape != output.shape:
        raise ValueError("need aligned nonempty prediction arrays")
    if not (np.isfinite(base).all() and np.isfinite(output).all()):
        raise ValueError("non-finite prediction probabilities")
    before, after = (base > .5) == labels, (output > .5) == labels
    corrected, harmed = int((~before & after).sum()), int((before & ~after).sum())
    return {"n": len(labels), "baseline_correct": int(before.sum()), "corrected": corrected,
            "harmed": harmed, "net": corrected - harmed,
            "harm_rate": harmed / int(before.sum()) if before.any() else None,
            "accuracy_gain_pp": 100 * (corrected - harmed) / len(labels),
            "accuracy_after": float(after.mean())}


def consensus(base, das, mean_diff):
    change = ((das > .5) == (mean_diff > .5)) & ((das > .5) != (base > .5))
    return np.where(change, das, base)


def load_model_data(old_root, low_root, model):
    data, metadata, hashes = {}, {}, {}
    for name, root, tag in (("old", old_root, "das_tf_spans"),
                            ("low", low_root, "low_strength"),
                            ("md", old_root, "meandiff_spans")):
        stem = root / "E23" / model / ("knob_sweep_" + tag)
        jp, zp = Path(str(stem) + ".json"), Path(str(stem) + ".npz")
        metadata[name] = json.loads(jp.read_text())
        if metadata[name]["model"] != model:
            raise ValueError("wrong model metadata")
        # The archives contain numeric/string arrays only; untrusted pickle is unnecessary.
        with np.load(zp, allow_pickle=False) as archive:
            data[name] = {key: archive[key] for key in archive.files}
        hashes[name] = {"json": hashlib.sha256(jp.read_bytes()).hexdigest(),
                        "npz": hashlib.sha256(zp.read_bytes()).hexdigest()}
    ref = data["old"]
    for name in ("low", "md"):
        for key in ("ids", "labels", "half_a"):
            if not np.array_equal(ref[key], data[name][key]):
                raise ValueError("different %s in %s/%s" % (key, model, name))
        if not np.array_equal(ref["baseline__ptrue"] > .5, data[name]["baseline__ptrue"] > .5):
            raise ValueError("baseline decisions differ in %s/%s" % (model, name))
        if not np.allclose(ref["baseline__ptrue"], data[name]["baseline__ptrue"], atol=1e-6, rtol=0):
            raise ValueError("baseline probabilities differ in %s/%s" % (model, name))
    return data, metadata, hashes


def candidates(data, metadata):
    yield {"kind": "baseline", "name": "baseline", "cost": [0, 0, 0.]}
    for source in ("old", "low"):
        for name in sorted(metadata[source]["conds"]):
            match = re.fullmatch(r"E_L([0-9-]+)_a([0-9.]+)", name)
            if match:
                yield {"kind": "steer", "source": source, "condition": name,
                       "name": source + ":" + name,
                       "cost": [1, len(match[1].split("-")), float(match[2])]}
    das = metadata["old"]["selection"]["E"]["chosen"]
    md = metadata["md"]["selection"]["E"]["chosen"]
    if das and md:
        yield {"kind": "consensus", "das_condition": das, "md_condition": md,
               "name": "consensus:" + das + ":" + md, "cost": [3, 0, 0.]}


def predictions(data, policy, mask):
    base = data["old"]["baseline__ptrue"][mask]
    if policy["kind"] == "baseline":
        return base.copy()
    if policy["kind"] == "steer":
        return data[policy["source"]][policy["condition"] + "__ptrue"][mask]
    if policy["kind"] == "consensus":
        return consensus(base, data["old"][policy["das_condition"] + "__ptrue"][mask],
                         data["md"][policy["md_condition"] + "__ptrue"][mask])
    raise ValueError("unknown policy")


def select(rows, cap):
    feasible = [r for r in rows if r["tune"]["harm_rate"] is not None and r["tune"]["harm_rate"] <= cap]
    if not feasible:
        raise ValueError("no feasible candidate, including baseline")
    return min(feasible, key=lambda r: (-r["tune"]["net"], r["tune"]["harmed"],
                                        *r["policy"]["cost"], r["policy"]["name"]))


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        out.write(content)


def fit(args):
    result = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "tune_harm_cap": args.harm_cap, "teacher_target": .01,
              "selection": "maximize A net corrections within cap; tie fewer harms, lower cost, name",
              "evaluation_status": "not evaluated on B by this fitting program", "models": {}}
    for model in MODELS:
        data, metadata, hashes = load_model_data(args.old_root, args.low_root, model)
        mask = data["old"]["half_a"].astype(bool)
        labels = data["old"]["labels"][mask].astype(bool)
        base = data["old"]["baseline__ptrue"][mask]
        rows = [{"policy": p, "tune": stats(labels, base, predictions(data, p, mask))}
                for p in candidates(data, metadata)]
        selected = select(rows, args.harm_cap)
        result["models"][model] = {"input_hashes": hashes, "chosen": selected,
                                    "original_condition": metadata["old"]["selection"]["E"]["chosen"],
                                    "candidates_on_A": rows}
        print(model, selected["policy"]["name"], selected["tune"])
    write_new(args.policy, json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def pair_intervals(labels, base, before, after, pair_ids, seed=42, repeats=2000):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(pair_ids == pid) for pid in sorted(set(pair_ids))]
    if any(len(g) != 2 for g in groups):
        raise ValueError("expected exactly two statements per question")
    original, selected, harm_reduction, gain_difference = [], [], [], []
    for _ in range(repeats):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        old = stats(labels[ix], base[ix], before[ix]); new = stats(labels[ix], base[ix], after[ix])
        original.append(old["accuracy_gain_pp"]); selected.append(new["accuracy_gain_pp"])
        if old["harm_rate"] is not None:
            harm_reduction.append(100 * (old["harm_rate"] - new["harm_rate"]))
        gain_difference.append(new["accuracy_gain_pp"] - old["accuracy_gain_pp"])
    return {name: np.quantile(values, [.025, .975]).tolist() for name, values in
            (("original_gain_pp", original), ("selected_gain_pp", selected),
             ("harm_reduction_pp", harm_reduction), ("selected_minus_original_gain_pp", gain_difference))}


def evaluate(args):
    frozen = json.loads(args.policy.read_text())
    results = {"policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
               "scope": "retrospective B evaluation after this round's A-only selection; B was used in earlier rounds",
               "models": {}}
    for model in MODELS:
        data, metadata, hashes = load_model_data(args.old_root, args.low_root, model)
        if hashes != frozen["models"][model]["input_hashes"]:
            raise ValueError("input changed since policy freeze")
        mask = ~data["old"]["half_a"].astype(bool)
        labels = data["old"]["labels"][mask].astype(bool)
        base = data["old"]["baseline__ptrue"][mask]
        old = data["old"][frozen["models"][model]["original_condition"] + "__ptrue"][mask]
        chosen = frozen["models"][model]["chosen"]
        new = predictions(data, chosen["policy"], mask)
        ids = data["old"]["ids"][mask].astype(str)
        pairs = np.array([x.rsplit(":", 1)[0] for x in ids])
        result = {"policy": chosen["policy"], "tune": chosen["tune"],
                  "original_test": stats(labels, base, old), "protected_test": stats(labels, base, new),
                  "paired_bootstrap_95": pair_intervals(labels, base, old, new, pairs)}
        results["models"][model] = result
        print(model, json.dumps(result, ensure_ascii=False))
    write_new(args.out, json.dumps(results, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--harm-cap", type=float, default=.005)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 0 <= args.harm_cap <= 1:
        parser.error("harm cap must be in [0,1]")
    if args.stage == "fit":
        fit(args)
    else:
        if args.out is None:
            parser.error("evaluate requires --out")
        evaluate(args)


if __name__ == "__main__":
    main()
