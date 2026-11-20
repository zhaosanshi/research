"""Read-only interim bounds for three intermediate E doses using existing Pro votes.

From the experiment's Truth_Decoupling root, in the scoring container:
    python3 -B PATH_TO_THIS_SCRIPT > bounds.csv

An optional --model qwen3.8-27b filters output, never the ten verified historical
vote sources. Scored rows are reused across models only for identical grading
contracts and identical inputs. The two C caps were fixed at E=2. These bounds
condition on the historical votes, and are not confidence intervals.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

MODELS = {
    "llama3.1-8b": "Llama-3.1-8B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.8-27b": "Qwen3.8-27B",
    "gemma3-12b": "Gemma-3-12B",
    "gemma3-27b": "Gemma-3-27B",
}
DOSES = ("1.25", "1.5", "1.75")
VALUE = {"A": 1, "B": -1, "H": 0, "C": 0}
HEADER = ("model", "e_strength", "questions", "answered", "resolved", "missing",
          "conflicting_or_unresolved", "net_low", "net_high", "prompt_net", "matched_e2_net")


def fingerprint(obj):
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def pro_contract(code):
    # Run in a clean interpreter so two model-specific code trees cannot share imports.
    script = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
              "import judge_two_axis_run as j; print(json.dumps(j.judge_contract('pro', j.load_cfg())))")
    path = Path(code, "scripts/review_checks").resolve()
    return fingerprint(json.loads(subprocess.check_output(
        [sys.executable, "-B", "-c", script, str(path)], text=True)))


def vote_key(cell):
    return (cell["pair_id"], cell["input_sha"], cell["question"], cell["prediction"],
            tuple(cell["gold"]), tuple(cell["gold_refs"]), tuple(cell["incorrect"]), cell["best_answer"])


def collect_history(config, mt, judge):
    pools = defaultdict(lambda: defaultdict(set))
    contracts, checked = {}, set()
    for model, display in MODELS.items():
        code = config[model]["code"]
        if code not in contracts:
            contracts[code] = pro_contract(code)
        signature = contracts[code]
        data = mt.Data(display, 1)
        cohort = set(data.sp.qs)
        for work in (data.m["ec"][1], data.m["d"][1]):
            directory = Path(work)
            meta, cells, votes = judge.load_verified_single(
                work, "pro", generation_path=directory / "judge_input.json", require_complete=False)
            if fingerprint(meta["single_verification"]["judge_contract"]) != signature:
                raise ValueError(f"Different Pro contract in {work}")
            if directory.resolve() in checked:
                raise ValueError(f"Repeated historical work directory: {work}")
            checked.add(directory.resolve())
            for cell_id, row in votes.items():
                cell = cells[cell_id]
                if cell["pair_id"] not in cohort:
                    continue
                if row["status"] == "scored":
                    label = row["label"]
                    if label not in VALUE:
                        raise ValueError(f"Unexpected Pro label in {work}: {label!r}")
                elif row.get("basis") == "empty_output_no_judge_vote" and not cell["prediction"].strip():
                    label = "C"
                else:
                    label = "?"
                pools[signature][vote_key(cell)].add(label)
            print(f"Verified {model} {work}: {len(votes)} Pro rows", file=sys.stderr)
    if len(checked) != 2 * len(MODELS):
        raise ValueError("Expected ten distinct verified work directories.")
    return {model: pools[contracts[config[model]["code"]]] for model in MODELS}


def net_bounds(selected, cells, pool):
    counts = Counter()
    known_net = 0
    for record in selected:
        cell = cells[record["p"], record["cond"]]
        if not cell["prediction"].strip():
            counts["resolved"] += 1       # project rubric assigns empty output zero
            continue
        labels = pool.get(vote_key(cell))
        if not labels:
            counts["missing"] += 1
        elif len(labels) != 1 or "?" in labels:
            counts["uncertain"] += 1
        else:
            known_net += VALUE[next(iter(labels))]
            counts["resolved"] += 1
    unknown = counts["missing"] + counts["uncertain"]
    return counts, known_net - unknown, known_net + unknown


def evaluate(model, cfg, pool, mt, scoring, judge):
    data = mt.Data(MODELS[model], 1)
    parent = Path(data.m["ec"][1])
    mid = parent.parent / "judge_test_dosemid_test"
    _, rows, raw = judge._workspace(str(mid))   # verifies export, cells, full row grid and frozen audit selection
    prior = json.loads((parent / "judge_input.json").read_text())
    for field in ("model", "identity_sha256", "question_ids", "split", "protocol", "export_protocol"):
        if not raw.get(field) or raw[field] != prior[field]:
            raise ValueError(f"Middle-dose and parent exports disagree on {field} for {model}")

    cells = {(c["pair_id"], c["condition"]): c for c in rows}
    conditions = ["baseline_x"]
    for dose in DOSES:
        token = dose.replace(".", "-")
        conditions.extend([data.sp.cE(token)] + [data.sp.cEC(token, k) for k in mt.fr.LAD])
    if len(cells) != len(rows) or set(raw["conditions"]) != set(conditions):
        raise ValueError(f"Unexpected middle-dose grid for {model}")
    ids = data.sp.qs
    expected = {(p, c) for p in ids for c in conditions}
    if not expected.issubset(cells):
        raise ValueError(f"Incomplete middle-dose search grid for {model}")
    blank = {k: c["rule_abstain"] or not c["prediction"].strip() for k, c in cells.items()}
    texts = mt.fr.Texts({k: c["prediction"] for k, c in cells.items()})
    # final_records uses labels only to annotate the result, never to choose the
    # branch. Give every row a placeholder, then score the final selected cell above.
    sp = scoring.NewSplit(cfg["el"], cfg["cl"], dict.fromkeys(cells, "unscored"), blank, conditions, qs=ids)
    if list(sp.qs) != list(ids):
        raise ValueError(f"Middle-dose cohort changed for {model}")
    caps = tuple(scoring.LADNAME[cfg["caps"][k]] for k in ("e_blank", "own_blank"))
    prompt_net = mt.net(mt.handed_in(data, ("DA",))[0])
    e2_condition = data.sp.dE("2")
    reference, eligible = mt.must(data, e2_condition)
    confidence = data.rows("conf_E2", data.m["d"][1], e2_condition)
    if any(not isinstance(confidence[p].get("mean_logp"), (float, int))
           or not math.isfinite(confidence[p]["mean_logp"]) for p, ok in zip(ids, eligible) if ok):
        raise ValueError(f"Missing finite confidence on an answered reference item for {model}")
    order = mt.ranked(data, eligible, confidence)
    for dose in DOSES:
        policy = ("N1", dose.replace(".", "-"), *caps)
        selected = mt.fr.records(sp, policy, texts)
        n = sum(mt.fr.answered(selected, sp))
        counts, low, high = net_bounds(selected, cells, pool)
        e2_net = mt.net(mt.keep(reference, order, n)[0]) if len(order) >= n else None
        yield (model, dose, len(selected), n, counts["resolved"], counts["missing"], counts["uncertain"],
               low, high, prompt_net, e2_net)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, action="append")
    args = parser.parse_args()
    sys.path[:0] = [str(Path("Iteration_01/diag").resolve()),
                    str(Path("Iteration_01/code/scripts/review_checks").resolve())]
    import dose_table
    import judge_two_axis_run
    import newsets_score
    config = newsets_score.models()
    pools = collect_history(config, dose_table.MT, judge_two_axis_run)
    results = []
    for model in dict.fromkeys(args.model or MODELS):
        results.extend(evaluate(model, config[model], pools[model], dose_table.MT, newsets_score, judge_two_axis_run))
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(HEADER)
    out.writerows(results)


if __name__ == "__main__":
    main()
