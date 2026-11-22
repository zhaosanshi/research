# Read-only check of the faithful ITI files (queue_c task 1 step 4): in every gen file and judge_input.json,
# (1) question ids are unique, (2) the row / evaluation order equals the file's own question_ids,
# (3) question_ids equal the reference judge input of the same split. Run from the project root.
import glob, json, os, sys
RUNS = {"llama3.1-8b": ("results_e44_factorial_das/llama3.1-8b_side_E2", "pospromptl16"),
        "gemma3-12b": ("Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b", "pospromptl24")}
bad = 0
total = 0
def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)
def order_of(rows):
    seen, out = set(), []
    for r in rows:
        q = r["pair_id"]
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out
for model, (run, ref_tag) in RUNS.items():
    refs = {}
    for split in ("tune", "test"):
        rp = "%s/eval_extra/judge_%s_%s_%s/judge_input.json" % (run, split, ref_tag, split)
        r = load(rp)
        refs[split] = list(r["question_ids"])
        ok = len(set(refs[split])) == len(refs[split]) and order_of(r["evaluations"]) == refs[split]
        print("%s reference %s: %d ids, unique and ordered: %s" % (model, split, len(refs[split]), ok))
        bad += not ok
    files = sorted(glob.glob("Iteration_01/baselines/iti/%s/faithful_seed42/gen/*.json" % model))
    files += sorted(glob.glob("%s/eval_extra/judge_*_iti_faithful_seed42*/judge_input.json" % run))
    for p in files:
        d = load(p)
        split = d["split"]
        ids = list(d["question_ids"])
        rows = d["rows"] if "rows" in d else d["evaluations"]
        n_rows = len(rows)
        uniq = len(set(ids)) == len(ids)
        # judge inputs hold one evaluation per (question, condition); the question order must still be the file's own
        same_order = order_of(rows) == ids
        per_q = n_rows % len(ids) == 0 and all(rows[i]["pair_id"] == ids[i % len(ids)] for i in range(n_rows)) \
            if n_rows != len(ids) else True
        if n_rows != len(ids):
            # accept either layout: blocks of the full id list, or consecutive rows of one question
            k = n_rows // len(ids) if n_rows % len(ids) == 0 else 0
            per_q = per_q or (k > 0 and all(rows[i]["pair_id"] == ids[i // k] for i in range(n_rows)))
        as_ref = ids == refs[split]
        ok = uniq and same_order and per_q and as_ref
        total += 1
        bad += not ok
        print("%s %s: split %s, %d ids, %d rows | unique %s | rows in own order %s | layout %s | equals reference %s" % (
            "ok " if ok else "BAD", os.path.relpath(p), split, len(ids), n_rows, uniq, same_order, per_q, as_ref))
print("checked %d files (plus 4 reference inputs); not satisfied: %d" % (total, bad))
sys.exit(1 if bad else 0)
