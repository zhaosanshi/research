#!/usr/bin/env bash
# Both judges on a FIXED SUBSET of the test questions of an existing test export (Claude Fable 5.2, 2026-11-15).
# 赵磊 11-15 22:51: "如果使用deepseek Pro有用，就别做529那么多，按照老师做随机100个就ok了，这样就能看出端倪了。因为根据以前的经验，
# 529和100的结论差不多。" The subset is the Dr's 100 pre-fixed pairs (configs/splits/dr_100pairs_20261110.json: drawn once
# with a fixed seed from the 529 test pairs, the report set of the true/false comparison) unless SUBSET names another file.
# For each NAME, RUN/judge_test_<NAME> must hold a complete export (judge_input.json) of the test split. The script writes
# RUN/judge_test_<NAME><SUFFIX>/judge_input.json = the same export restricted to the subset's questions (same conditions,
# same answers, same order; nothing is generated), then runs the judging steps of ff_test_eval.sh judge_both on it:
# cells + AllenAI (GPU) + single, then DeepSeek Pro + single. A cell without a verdict stops the job, as there.
#   MODEL=<config> CODE_PATH=<code copy> RUN=<run dir> NAMES="testX_E_L24 testD_E_L24 ..." [SUFFIX=dr100]
#   [SUBSET=<json file with "pair_ids", container path>] [JUDGES="allenai pro"] [GPU=k]
#   [CACHE_FROM="<work dir> ..."  Pro reply caches to seed from: identical requests are not paid twice]  ff_test_subset_judge.sh
set -euo pipefail
: "${MODEL:?}" "${CODE_PATH:?}" "${RUN:?}" "${NAMES:?}"
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$CODE_PATH
SUFFIX=${SUFFIX:-dr100}
SUBSET=${SUBSET:-$CODE/configs/splits/dr_100pairs_20261110.json}
JUDGES=${JUDGES:-allenai pro}
MODELS=$ROOT/models
host() { case "$1" in "$ROOT"/*) echo "$HOST/${1#$ROOT/}" ;; *) echo "path $1 is outside $ROOT" >&2; exit 2 ;; esac; }
case "$SUFFIX" in ""|*[!A-Za-z0-9]*) echo "SUFFIX must be alphanumeric" >&2; exit 2 ;; esac
for n in $NAMES; do case "$n" in *[!A-Za-z0-9_]*) echo "NAME '$n': letters, digits and _ only" >&2; exit 2 ;; esac; done
for j in $JUDGES; do case "$j" in allenai|pro) ;; *) echo "JUDGES: unknown judge '$j' (allenai / pro)" >&2; exit 2 ;; esac; done
[ -n "$JUDGES" ] || { echo "JUDGES is empty" >&2; exit 2; }
has() { case " $JUDGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
[ -f "$(host "$RUN")/frozen_test.json" ] || { echo "$RUN has no frozen_test.json" >&2; exit 3; }
[ -f "$(host "$SUBSET")" ] || { echo "no subset file $SUBSET" >&2; exit 3; }
if has allenai; then [ -n "${GPU:-}" ] || { echo "AllenAI needs GPU" >&2; exit 2; }; fi
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$MODELS
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
pyj() { "${DX[@]}" python3 "$@"; }
step() { echo "=== $(date '+%F %T') [$MODEL subset $SUFFIX gpu${GPU:-none}] $* ==="; }
verdicts_complete() {  # as ff_test_eval.sh: an empty answer counts as an abstention; any other missing verdict stops,
  # except cells the judge itself declined (basis pro_declined; from_claude/45 section 3.5): they are listed and the job goes
  # on, because e44_round1_tables.py makes no table of this work dir until judge2/excluded_questions.json names each such
  # question with a written reason (the question then leaves the tables for both judges; nothing is imputed)
  local out bad empty declined
  out=$(pyj -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
empty = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'empty_output_no_judge_vote' and r.get('empty_output') is True]
decl = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'pro_declined']
print(sum(r['status'] != 'scored' for r in rows) - len(empty) - len(decl), len(empty), ','.join(r['cell_id'] for r in decl) or '-')" "$1/judge2/single_$2.jsonl") \
    || { echo "cannot read the $2 verdicts of $1" >&2; exit 3; }
  read -r bad empty declined <<< "$out"
  [ "$empty" = 0 ] || echo "$2: $empty empty answer(s) in $1 count as abstentions (never judged; two-axis standard of 11-08)"
  [ "$declined" = - ] || echo "$2 DECLINED these cells in $1 (no table until judge2/excluded_questions.json names their questions with a reason): $declined"
  [ "$bad" = 0 ] || { echo "$2 left $bad cells without a verdict in $1: re-judge them before any table" >&2; exit 3; }
}
for n in $NAMES; do
  src=$RUN/judge_test_$n; work=$RUN/judge_test_$n$SUFFIX
  [ -f "$(host "$src")/judge_input.json" ] || { echo "no export $src/judge_input.json" >&2; exit 3; }
  step "$n: the export restricted to the questions of ${SUBSET##*/}"
  mkdir -p "$(host "$work")/judge2/cache"
  python3 -B - "$(host "$src")/judge_input.json" "$(host "$SUBSET")" "$(host "$work")" "$src" "$SUBSET" <<'PY'
import hashlib, json, os, sys
src, subset, work, src_name, subset_name = sys.argv[1:6]
raw = open(src, "rb").read()
d = json.loads(raw)
if d.get("completed") is not True or d.get("split") != "test":
    sys.exit("%s is not a completed export of the test split" % src)
sraw = open(subset, "rb").read()
ids = json.loads(sraw)["pair_ids"]
if len(set(ids)) != len(ids):
    sys.exit("the subset repeats a pair id")
have = set(d["question_ids"])
missing = [i for i in ids if i not in have]
if missing:
    sys.exit("%d subset question(s) are not in the export, e.g. %s" % (len(missing), missing[:3]))
keep = set(ids)
out = dict(d)
out["question_ids"] = [q for q in d["question_ids"] if q in keep]
out["evaluations"] = [e for e in d["evaluations"] if e["pair_id"] in keep]
if [e["pair_id"] for e in out["evaluations"]] != out["question_ids"]:
    sys.exit("the export's evaluations are not in the order of its question ids")
text = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
path = os.path.join(work, "judge_input.json")
if os.path.exists(path):
    if open(path, encoding="utf-8").read() != text:
        sys.exit("%s exists and differs: not overwritten" % path)
else:
    open(path, "w", encoding="utf-8").write(text)
info = {"subset_of": src_name + "/judge_input.json", "source_sha256": hashlib.sha256(raw).hexdigest(),
        "subset_file": subset_name, "subset_sha256": hashlib.sha256(sraw).hexdigest(),
        "n_questions": len(out["question_ids"]), "n_source_questions": len(d["question_ids"]),
        "conditions": d["conditions"]}
open(os.path.join(work, "subset_info.json"), "w", encoding="utf-8").write(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
print("%d of %d questions kept, %d conditions" % (len(out["question_ids"]), len(d["question_ids"]), len(d["conditions"])))
PY
  if has allenai; then
    step "$n: cells + allenai + single"
    pyj scripts/review_checks/judge_two_axis_run.py cells "$work" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
    "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$work/judge2/cells.jsonl" --out "$work/judge2/allenai.jsonl" \
      --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
    pyj scripts/review_checks/judge_two_axis_run.py single "$work" --judge allenai > "$(host "$work")/single_allenai_stdout.md"
  else
    [ -f "$(host "$work")/judge2/single_allenai.jsonl" ] || { echo "no AllenAI pass in $work yet: run JUDGES=allenai first" >&2; exit 3; }
  fi
  verdicts_complete "$work" allenai
  if has pro; then
    for c in ${CACHE_FROM:-}; do                                    # identical requests are not paid twice
      [ -d "$(host "$c")/judge2/cache" ] || { echo "reply cache $c/judge2/cache does not exist (CACHE_FROM)" >&2; exit 3; }
      for f in "$(host "$c")"/judge2/cache/*; do
        [ -f "$f" ] || continue
        [ -e "$(host "$work")/judge2/cache/$(basename "$f")" ] || cp "$f" "$(host "$work")/judge2/cache/"
      done
    done
    step "$n: pro"
    pyj scripts/review_checks/judge_two_axis_run.py llm "$work" --judge pro
    pyj scripts/review_checks/judge_two_axis_run.py single "$work" --judge pro > "$(host "$work")/single_pro_stdout.md"
    verdicts_complete "$work" pro
  else
    step "$n: pro skipped (JUDGES=$JUDGES)"
  fi
done
step "done"
