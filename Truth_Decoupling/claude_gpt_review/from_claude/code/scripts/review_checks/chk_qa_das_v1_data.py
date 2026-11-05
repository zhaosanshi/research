#!/usr/bin/env python3
"""Round 16 data check (Claude): what the frozen QA-DAS v1 config yields from the real TruthfulQA splits.

usage: chk_qa_das_v1_data.py [CODE_DIR] [CONFIG_NAME]
   CODE_DIR     the experiment code root holding core/, configs/, scripts/ (default: the review snapshot in the container)
   CONFIG_NAME  experiment config name (default: E25_qa_das_v1)
env: TD_DATA_DIR must point at the processed data directory (as for every experiment script).

No model is loaded. The output answers four questions that the toy-network tests cannot:
  * how many estimation / tuning pairs survive the refusal cleaning (freeze() needs every tuning question to have a
    substantive gold answer: `n_substantive_gold == n_questions`);
  * how many questions of each split have refusal-only gold;
  * which questions / pairs the 3-question pilot sees;
  * whether any split lacks best_answer.
Round-16 result on the server (2026-11-05): train 96 -> 92 pairs (4 excluded: no substantive positive), tune 192 -> 192 pairs,
refusal-only gold: train 3, tune 0, test 6 (the 6 test questions are partitioned by E24's reporter, not dropped).
"""
import json
import sys

CODE = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "E25_qa_das_v1"
sys.path[:0] = [CODE + "/scripts", CODE]
import common  # noqa: F401,E402  (puts the code root on sys.path, like the scripts do)
import e25_qa_context as legacy  # noqa: E402
from core.config import load_json_config, resolve_config_path  # noqa: E402
from core.qa_scoring import reference_answers  # noqa: E402

cfg = load_json_config(resolve_config_path("experiments", CONFIG))
p = legacy.split_questions(cfg)
out = {"config": CONFIG, "split_counts": {k: len(v) for k, v in p["splits"].items()},
       "full_split_counts": p["full_split_counts"],
       "e_train_pairs": len(p["e_train"]), "e_tune_pairs": len(p["e_tune"]),
       "excluded_train": p["pair_audit"]["train"]["excluded"],
       "excluded_tune": p["pair_audit"]["tune"]["excluded"]}
for name in ("train", "tune", "test"):
    rows = p["splits"][name]
    ro = [(r["pair_id"], r["gold"]) for r in rows if reference_answers(r["gold"])[1]]
    out["refusal_only_gold_" + name] = {"n": len(ro), "examples": ro[:5]}
    out["no_best_answer_" + name] = sum(r["best_answer"] is None for r in rows)
out["first3_train"] = [(r["pair_id"], r["question"]) for r in p["splits"]["train"][:3]]
out["first3_tune"] = [(r["pair_id"], r["question"], r["gold"][:3], r["best_answer"]) for r in p["splits"]["tune"][:3]]
pilot = legacy.split_questions(cfg, 3)
out["pilot3"] = {"e_train": [(r["pair_id"], r["positive_answer"], r["negative_answer"]) for r in pilot["e_train"]],
                 "e_tune": [(r["pair_id"], r["positive_answer"], r["negative_answer"]) for r in pilot["e_tune"]],
                 "tune_questions": [(r["pair_id"], r["question"]) for r in pilot["splits"]["tune"]],
                 "test_questions": [(r["pair_id"], r["question"]) for r in pilot["splits"]["test"]]}
out["freeze_precondition_tune_all_substantive"] = out["refusal_only_gold_tune"]["n"] == 0
print(json.dumps(out, ensure_ascii=False, indent=1))
