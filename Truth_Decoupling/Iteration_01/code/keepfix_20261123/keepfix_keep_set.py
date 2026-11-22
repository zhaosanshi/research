# -*- coding: utf-8 -*-
"""Queue K (Claude Opus 5.5, 2026-11-23; discussion/work_queue/2026-11-23_queue_k.md, section 1 and section 2 item 3):
the keep questions = TriviaQA / PopQA training-split questions (newsets/splits/<task>_splits_20261119.json "train",
300 each, disjoint from tune and test) that the plain model answers right with the plain prompt x and no write.
Right = the alias match as diag/ns_prompt_table.py reads a pass: blank (rule abstention or empty text) -> X, otherwise
scripts/newsets/alias_match.verdict on the bank's alias list; R is kept.
Classes (counted, not used to choose; added from the GPT review of 2026-11-23, section 2 item 3): "both" = also right
with the prompt P = x_E2 (plain prompt + honest sentence) and no write, the prompt the keep loss is applied on;
"x_only" = right with x only (P wrong or blank).
Inputs: the x and x_E2 passes of each task on the train split (keepfix_ns_train.py; read with
newsets_gen.load_generation: complete, full pass, this model, this bank, bank order, no write), the banks and the split
files.
Outputs: --out-set (server only: question text, the answer's text and token ids) and --out-ids (ids and counts only; this
one may go into the repository). Neither is overwritten.
usage (CPU, from /workspace/Truth_Decoupling):
  python -I keepfix_keep_set.py --model MODEL --gen-root DIR --out-set FILE --out-ids FILE
        (DIR/<task>/<model>/train/x.json for task in triviaqa, popqa)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

NSD = Path("/workspace/Truth_Decoupling/Iteration_01/code/scripts/newsets")
if str(NSD) not in sys.path:
    sys.path.insert(0, str(NSD))
import newsets_gen as G   # noqa: E402
import alias_match as AM  # noqa: E402

NS = Path("/workspace/Truth_Decoupling/Iteration_01/newsets")
TASKS = ("triviaqa", "popqa")
PROTOCOL = "keepfix_keep_set_v1"
KEYS = ("text", "token_ids", "finished", "stop_id", "n_tokens")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-root", type=Path, required=True)
    ap.add_argument("--out-set", type=Path, required=True)
    ap.add_argument("--out-ids", type=Path, required=True)
    a = ap.parse_args()
    for p in (a.out_set, a.out_ids):
        if p.exists():
            raise SystemExit("%s exists" % p)
    items, ids_out, sources = [], {}, {}
    for task in TASKS:
        bank_p = NS / "bank" / ("%s_bank_20261119.jsonl" % task)
        split_p = NS / "splits" / ("%s_splits_20261119.json" % task)
        rows = [json.loads(l) for l in open(bank_p, encoding="utf-8")]
        bank = {r["id"]: r for r in rows if r["split"] == "train"}
        sp = json.load(open(split_p, encoding="utf-8"))
        if set(bank) != set(sp["train"]) or len(bank) != len(sp["train"]) or len(bank) != 300:
            raise SystemExit("%s: the bank's train rows are not the split file's 300 train ids" % task)
        others = set(sp["tune"]) | set(sp["test"])
        if others & set(bank):
            raise SystemExit("%s: train ids overlap tune / test" % task)
        passes = {}
        for arm in ("x", "x_E2"):
            gen_p = a.gen_root / task / a.model / "train" / (arm + ".json")
            g = G.load_generation(gen_p, SimpleNamespace(task=task, split="train", model=a.model, items=bank_p), list(bank))
            c = g["contract"]
            if c["arm"] != arm or c["condition"] != "baseline_" + arm or c["parts"] or c.get("positions") is not None:
                raise SystemExit("%s is not the no-write pass of %s" % (gen_p, arm))
            if c["max_new_tokens"] != G.ENGINE_MAX_NEW or c["batch_size"] != G.BATCH:
                raise SystemExit("%s: not the 48-token, batch-8 generation" % gen_p)
            verdict = {}
            for r in g["rows"]:
                blank = bool(r["rule_abstain"]) or not r["text"].strip()
                verdict[r["id"]] = AM.verdict(r["text"], bank[r["id"]]["gold"], not blank)
            passes[arm] = (g, c, gen_p, verdict)
        g, c, gen_p, vx = passes["x"]
        vp = passes["x_E2"][3]
        cnt = {"R": 0, "W": 0, "X": 0}
        keep, cls = [], {"both": [], "x_only": []}
        for r in g["rows"]:
            v = vx[r["id"]]
            cnt[v] += 1
            if v == "R":
                k = "both" if vp[r["id"]] == "R" else "x_only"
                keep.append(r["id"])
                cls[k].append(r["id"])
                items.append(dict({kk: r[kk] for kk in KEYS}, task=task, id=r["id"], question=bank[r["id"]]["question"],
                                  **{"class": k, "verdict_x_E2": vp[r["id"]]}))
        cp = {k: sum(1 for v in vp.values() if v == k) for k in ("R", "W", "X")}
        ids_out[task] = {"n_train": len(bank), "right": cnt["R"], "wrong": cnt["W"], "blank": cnt["X"],
                         "x_E2_right": cp["R"], "x_E2_wrong": cp["W"], "x_E2_blank": cp["X"],
                         "keep_both": len(cls["both"]), "keep_x_only": len(cls["x_only"]), "ids": keep,
                         "ids_both": cls["both"], "ids_x_only": cls["x_only"]}
        sources[task] = {"generation_file": str(gen_p), "generation_sha256": sha(gen_p),
                         "generation_x_E2_file": str(passes["x_E2"][2]), "generation_x_E2_sha256": sha(passes["x_E2"][2]),
                         "bank_file": str(bank_p), "bank_sha256": sha(bank_p), "splits_file": str(split_p),
                         "splits_sha256": sha(split_p), "identity_sha256": c["identity_sha256"], "engine_sha256": c["engine_sha256"]}
        print("%s %s train: %d questions, x: right %d, wrong %d, blank %d; x_E2: right %d, wrong %d, blank %d -> %d keep questions"
              " (right under both %d, under x only %d)" % (a.model, task, len(bank), cnt["R"], cnt["W"], cnt["X"], cp["R"], cp["W"],
                                                          cp["X"], len(keep), len(cls["both"]), len(cls["x_only"])), flush=True)
    counts = {t: {k: v for k, v in d.items() if not k.startswith("ids")} for t, d in ids_out.items()}
    full = {"protocol": PROTOCOL, "model": a.model, "rule": "alias match R of the plain prompt x, no write, train split",
            "counts": counts, "sources": sources, "items": items, "script_sha256": sha(__file__)}
    a.out_set.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out_set.with_suffix(a.out_set.suffix + ".new")
    tmp.write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
    tmp.rename(a.out_set)
    light = {"protocol": PROTOCOL + "_ids", "model": a.model, "rule": full["rule"], "tasks": ids_out,
             "n_keep": len(items), "sources": {t: {k: v for k, v in s.items() if k.endswith("sha256")} for t, s in sources.items()},
             "keep_set_sha256": sha(a.out_set), "script_sha256": sha(__file__)}
    tmp = a.out_ids.with_suffix(a.out_ids.suffix + ".new")
    tmp.write_text(json.dumps(light, indent=1), encoding="utf-8")
    tmp.rename(a.out_ids)
    print("keep set: %d questions (%s) -> %s; ids -> %s" % (len(items), ", ".join("%s %d" % (t, len(d["ids"])) for t, d in ids_out.items()),
                                                         a.out_set, a.out_ids), flush=True)


if __name__ == "__main__":
    main()
