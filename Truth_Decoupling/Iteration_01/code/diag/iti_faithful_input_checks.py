# Read-only check of the two completed faithful ITI grids, run from Truth_Decoupling.
import hashlib
import json
from pathlib import Path

RUNS = {"llama3.1-8b": ("results_e44_factorial_das/llama3.1-8b_side_E2", "pospromptl16"),
        "gemma3-12b": ("Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b", "pospromptl24")}
KS = (16, 32, 48, 64)
ALPHAS = ("2-5", "5", "7-5", "10", "15", "20", "25", "30")


def read(path):
    return json.loads(path.read_bytes())


def check_rows(path, ids, rows, expected):
    if (not ids or len(set(ids)) != len(ids) or ids != expected
            or [row["pair_id"] for row in rows] != ids):
        raise SystemExit(f"{path}: missing, duplicate or out-of-order question IDs in list or rows")


def main():
    total = 0
    for model, (run, tag) in RUNS.items():
        work = Path(run) / "eval_extra"
        fit = Path("Iteration_01/baselines/iti") / model / "faithful_seed42"
        refs = {}
        for split in ("tune", "test"):
            path = work / f"judge_{split}_{tag}_{split}/judge_input.json"
            ref = read(path)
            refs[split] = ref["question_ids"]
            check_rows(path, refs[split], ref["evaluations"], refs[split])

        stems = {f"tune_k{k}_a{a}" for k in KS for a in ALPHAS}
        for name in ("choice.json", "choice_all.json"):
            stems.add("test_" + read(fit / name)["stem"])
        files = sorted((fit / "gen").glob("*.json"))
        if {p.stem for p in files} != stems:
            raise SystemExit(f"{model}: generation files differ from the declared grid and two test choices")
        native = {}
        for path in files:
            g = read(path)
            split = g["split"]
            check_rows(path, g["question_ids"], g["rows"], refs[split])
            key = (split, g["condition"])
            if (key in native or g["model"] != model or g["completed"] is not True
                    or not path.stem.startswith(split + "_")):
                raise SystemExit(f"{path}: unexpected generation identity")
            native[key] = (path, hashlib.sha256(path.read_bytes()).hexdigest(), g)

        covered = set()
        exports = sorted(work.glob("judge_*_iti_faithful_seed42*/judge_input.json"))
        for path in exports:
            ex = read(path)
            split, conditions = ex["split"], ex["conditions"]
            rows = ex["evaluations"]
            check_rows(path, ex["question_ids"], rows, refs[split])
            if (ex["model"] != model or ex["completed"] is not True or not conditions
                    or len(set(conditions)) != len(conditions) or set(conditions) != set(ex["source_files"])):
                raise SystemExit(f"{path}: unexpected export identity or condition set")
            for row in rows:
                if any(set(row[field]) != set(conditions) for field in ("outputs", "finished", "rule_abstain")):
                    raise SystemExit(f"{path}: missing or extra condition on {row['pair_id']}")
            for condition in conditions:
                key = (split, condition)
                gen_path, digest, g = native[key]
                if (ex["source_files"][condition] != {"file": gen_path.name, "sha256": digest}
                        or ex["identity_sha256"] != g["identity_sha256"]):
                    raise SystemExit(f"{path}: source identity differs for {condition}")
                for row, src in zip(rows, g["rows"]):
                    if (row["outputs"][condition], row["finished"][condition], row["rule_abstain"][condition]) != (
                            src["text"], src["finished"], src["rule_abstain"]):
                        raise SystemExit(f"{path}: exported answer differs on {condition}/{row['pair_id']}")
                covered.add(key)
        if covered != set(native):
            raise SystemExit(f"{model}: missing exported generations {sorted(set(native) - covered)}")
        total += len(files) + len(exports)
        print(f"{model}: {len(files)} generation files and {len(exports)} exports passed")
    print(f"{total} files passed plus {2 * len(RUNS)} reference inputs")


if __name__ == "__main__":
    main()
