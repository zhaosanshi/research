#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-17): step 0 of novelty/ours/2026-11-17_e_doubt_c_commit.md section 7 -- a DeepSeek Pro
# pass on tune work dirs that already hold their cells and a complete AllenAI pass (赵磊 11-17: "选调参集也用 Pro 判一遍").
# Nothing is generated, exported or re-judged by AllenAI; no GPU. Per work dir, the same calls as the Pro block of
# ff_explore_ec.sh / newmodel_e44.sh: [cache merge from SEED dirs] -> judge_two_axis_run.py llm --judge pro -> single -> the
# verdict check of ff_test_subset_judge.sh (a missing verdict stops the lane; Pro-declined cells are listed, never imputed).
# usage: ff_tune_pro_20261117.sh LANE_LOG_DIR WORK[@SEED_CACHE_DIR] ...   (container paths; one lane = one sequential list)
set -uo pipefail
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/Iteration_01/code
LOGD=$1; shift
host() { case "$1" in "$ROOT"/*) echo "$HOST/${1#$ROOT/}" ;; *) echo "path $1 is outside $ROOT" >&2; return 2 ;; esac; }
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
pyj() { "${DX[@]}" python3 "$@"; }
say() { echo "$(date '+%F %T') $*"; }
mkdir -p "$(host "$LOGD")" || exit 2
LH=$(host "$LOGD")
rm -f "$LH/LANE.DONE" "$LH/LANE.FAIL"
fail() { say "$*"; echo "$*" > "$LH/LANE.FAIL"; exit 1; }
check() {  # check WORK JUDGE: as verdicts_complete of ff_test_subset_judge.sh; prints "bad empty declined"
  pyj -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
empty = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'empty_output_no_judge_vote' and r.get('empty_output') is True]
decl = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'pro_declined']
print(len(rows), sum(r['status'] != 'scored' for r in rows) - len(empty) - len(decl), len(empty), ','.join(r['cell_id'] for r in decl) or '-')" "$1/judge2/single_$2.jsonl"
}
for item in "$@"; do
  W=${item%%@*}; SEED=""; [ "$item" = "$W" ] || SEED=${item#*@}
  WH=$(host "$W") || fail "bad work dir $W"
  say "== $W"
  [ -f "$WH/judge2/cells.jsonl" ] && [ -f "$WH/judge2/single_allenai.jsonl" ] || fail "$W: no cells or no AllenAI pass"
  [ -e "$WH/judge2/single_pro.jsonl" ] && fail "$W: a Pro pass exists already (refusing to redo it)"
  read -r n bad empty decl <<< "$(check "$W" allenai)" || fail "$W: cannot read the AllenAI verdicts"
  [ "$bad" = 0 ] || fail "$W: AllenAI left $bad cells without a verdict"
  say "AllenAI complete: $n cells ($empty empty answers)"
  mkdir -p "$WH/judge2/cache"
  if [ -n "$SEED" ]; then
    [ -f "$(host "$SEED")/judge2_cache_deepseek-v4-pro.jsonl" ] || fail "$W: no Pro reply cache in the seed $SEED"
    say "merge the Pro reply cache $SEED"
    pyj scripts/review_checks/judge2_cache_merge.py "$W/judge2/cache" "$SEED" || fail "$W: cache merge failed"
    pyj -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('merged caches give %d request keys different replies: stop' % n if n else 0)" "$W/judge2/cache/conflicts.json" || fail "$W: cache conflicts"
  fi
  say "pro"
  pyj scripts/review_checks/judge_two_axis_run.py llm "$W" --judge pro || fail "$W: the Pro pass failed"
  pyj scripts/review_checks/judge_two_axis_run.py single "$W" --judge pro > "$WH/single_pro_stdout.md" || fail "$W: single (pro) failed"
  read -r n bad empty decl <<< "$(check "$W" pro)" || fail "$W: cannot read the Pro verdicts"
  [ "$decl" = - ] || say "Pro DECLINED cells in $W: $decl"
  [ "$bad" = 0 ] || fail "$W: Pro left $bad cells without a verdict"
  say "Pro complete: $n cells ($empty empty answers, declined: $decl)"
done
say "lane done"; touch "$LH/LANE.DONE"
