#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-18): the stronger-caution rerun (results/ff_xu1) for Gemma-4-31B, Gemma-3-12B and
# Gemma-3-27B, done as for Gemma-4-12B on 11-16 (launchers/night_queue_12b.sh step 2). Nothing new is computed here:
#   source  make each new source directory with scripts/review_checks/make_e44_source.py --append-sentence U1 (the call
#           newmodel_e44.sh makes, plus the sentence; the 12B one of 11-16 was made so: same derived_from.dir, plain_prompt
#           record), then check it against the plain run's source directory: questions.json byte-identical; bank.json equal
#           in every field except config.qa_system (= plain + " " + U1) and the added derived_from.plain_prompt. Hard stop on
#           any other difference. U1 = results/ff_xu1/gemma4/12b/logs/plain_prompt_sentence_U1.txt, checked equal to the 12B
#           run's derived_from.plain_prompt.appended_sentence.
#   start   per model: the cards must have no compute process; the lines chain ff_ec_lines_chain.sh with the plain run's
#           MODEL / GPU-count / E_LAYERS / C_LAYERS (from its logs/chain/params.txt), then after 60 s the test chain
#           ff_ec_test_chain_v3.sh (it waits for the lines chain by itself). Gemma-3-27B: the lines chain runs with
#           CHAIN_RUNNER=launchers/newmodel_e44_expseg.sh (newmodel_e44.sh + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
#           the copy ff_gemma3_27b_c_retry2.sh used for the plain run's rerun, after c_run_1 ran out of memory at C layer 19
#           on one card), checked first to differ from newmodel_e44.sh in that one line only. The v3 chain has no runner hook
#           and ran on one card without the allocator setting for the plain run.
#   A record of every start (command, environment, time, PIDs) goes to <new run>/logs/xu1_launch.txt.
# usage (host): bash ff_xu1_stage1_20261118.sh source | start
set -euo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
LA=$R/Iteration_01/launchers
RC=$R/Iteration_01/code/scripts/review_checks
CODE=$W/Iteration_01/code
WRAP=$LA/newmodel_e44_expseg.sh
U1F=$R/Iteration_01/results/ff_xu1/gemma4/12b/logs/plain_prompt_sentence_U1.txt
REF12=$R/Iteration_01/results/ff_xu1/gemma4/12b/source/gemma4-12b/bank.json
LIKE=$W/results_qa_das_v1u/llama3.1-8b/full
# model | sub path | GPU | runner (default = the chain's own)
SPECS=("gemma4-31b-dm|gemma4/31b|2,3|"
       "gemma3-12b|gemma3/12b|0|"
       "gemma3-27b|gemma3/27b|1|$WRAP")
DX=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec python3)
die() { echo "STOP: $*" >&2; exit 1; }
U1=$(cat "$U1F")
python3 -c "import json,sys; a=json.load(open(sys.argv[1]))['derived_from']['plain_prompt']['appended_sentence']; sys.exit(a!=sys.argv[2])" \
  "$REF12" "$U1" || die "U1 file differs from the 12B run's appended sentence"

check_source() {  # check_source PLAIN_SRC_DIR XU_SRC_DIR (host paths)
  cmp -s "$1/questions.json" "$2/questions.json" || die "$2/questions.json differs from $1"
  python3 - "$1/bank.json" "$2/bank.json" "$U1" <<'EOF'
import json, sys
p, x, u1 = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
bad = []
def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            q = path + "/" + k
            if k not in a: bad.append("only new: " + q) if q != "/derived_from/plain_prompt" else None
            elif k not in b: bad.append("only plain: " + q)
            else: walk(a[k], b[k], q)
    elif a != b and path != "/config/qa_system":
        bad.append("differs: %s plain=%r new=%r" % (path, a, b))
walk(p, x, "")
if x["config"]["qa_system"] != p["config"]["qa_system"] + " " + u1: bad.append("qa_system is not plain + ' ' + U1")
if x["derived_from"].get("plain_prompt") != {"appended_sentence": u1, "qa_system_of_like": p["config"]["qa_system"]}:
    bad.append("derived_from.plain_prompt wrong")
if bad: print("\n".join(bad)); sys.exit(1)
print("ok: %s = plain source except config.qa_system (+ U1) and derived_from.plain_prompt" % sys.argv[2])
EOF
}

gpu_free() {  # gpu_free "2,3": none of the cards has a compute process
  local c u
  for c in ${1//,/ }; do
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$c")
    [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")" = 0 ] || return 1
  done
}

same_runner() {  # as ff_gemma3_27b_c_retry2.sh
  local d n
  d=$(diff "$RC/newmodel_e44.sh" "$WRAP") || true
  n=$(printf '%s\n' "$d" | grep -c '^[<>]') || true
  [ "$n" = 2 ] && [[ $d == *"> "*"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"* ]]
}

phase=${1:-}
case "$phase" in
source)
  for s in "${SPECS[@]}"; do
    IFS='|' read -r M SUB GPU RUNNER <<< "$s"
    PS=$R/Iteration_01/results/ff/$SUB/source/$M
    XS=$R/Iteration_01/results/ff_xu1/$SUB/source/$M
    [ -f "$PS/bank.json" ] || die "no plain source $PS"
    if [ ! -f "$XS/bank.json" ]; then
      mkdir -p "$R/Iteration_01/results/ff_xu1/$SUB/logs"
      "${DX[@]}" scripts/review_checks/make_e44_source.py --model "$M" --like "$LIKE" \
        --out "$W/Iteration_01/results/ff_xu1/$SUB/source/$M" --append-sentence "$U1"
    fi
    check_source "$PS" "$XS"
  done ;;
start)
  for s in "${SPECS[@]}"; do
    IFS='|' read -r M SUB GPU RUNNER <<< "$s"
    HP=$R/Iteration_01/results/ff/$SUB; HX=$R/Iteration_01/results/ff_xu1/$SUB; X=$W/Iteration_01/results/ff_xu1/$SUB
    check_source "$HP/source/$M" "$HX/source/$M"
    [ ! -e "$HX/logs/chain" ] && [ ! -e "$HX/logs/chain_ec_v3" ] || die "$HX already has chain logs"
    PAR=$(cat "$HP/logs/chain/params.txt")
    EL=$(sed -n 's/.* e_layers=\(.*\) c_layers=.*/\1/p' <<< "$PAR"); CL=$(sed -n 's/.* c_layers=\(.*\)$/\1/p' <<< "$PAR")
    [[ $PAR == "model=$M "* ]] && [ -n "$EL" ] && [ -n "$CL" ] || die "cannot read the plain run's params: $PAR"
    gpu_free "$GPU" || die "GPU $GPU has a compute process"
    EXTRA=()
    if [ -n "$RUNNER" ]; then same_runner || die "$WRAP differs from newmodel_e44.sh in more than the allocator line"
      EXTRA=(CHAIN_RUNNER="$RUNNER"); fi
    REC=$HX/logs/xu1_launch.txt
    {
      echo "$(date '+%F %T') launcher $0 (sha256 $(sha256sum "$0" | cut -c1-16))"
      echo "  plain run params: $PAR"
      echo "  lines: cd $LA; env MODEL=$M GPU=$GPU FF_ROOT=$X E_LAYERS=\"$EL\" C_LAYERS=\"$CL\" ${EXTRA[*]:-} setsid nohup bash ff_ec_lines_chain.sh > $HX/logs/chain_launch.out"
      [ -z "$RUNNER" ] || echo "  runner $RUNNER sha256 $(sha256sum "$RUNNER" | cut -c1-16) (= newmodel_e44.sh $(sha256sum "$RC/newmodel_e44.sh" | cut -c1-16) + allocator line)"
    } >> "$REC"
    cd "$LA"
    env MODEL="$M" GPU="$GPU" FF_ROOT="$X" E_LAYERS="$EL" C_LAYERS="$CL" "${EXTRA[@]}" \
      setsid nohup bash ff_ec_lines_chain.sh > "$HX/logs/chain_launch.out" 2>&1 < /dev/null & disown
    echo "  lines chain started, launcher-side pid $!" >> "$REC"
  done
  sleep 60
  for s in "${SPECS[@]}"; do
    IFS='|' read -r M SUB GPU RUNNER <<< "$s"
    HX=$R/Iteration_01/results/ff_xu1/$SUB; X=$W/Iteration_01/results/ff_xu1/$SUB; REC=$HX/logs/xu1_launch.txt
    grep -q "chain start" "$HX/logs/chain/chain_jobs.log" 2>/dev/null || { echo "$(date '+%F %T') lines chain did not start: v3 not started" >> "$REC"; continue; }
    echo "$(date '+%F %T') v3: cd $LA; MODEL=$M GPU=$GPU FF_ROOT=$X setsid nohup bash ff_ec_test_chain_v3.sh > $HX/logs/chain_ec_v3_launch.out" >> "$REC"
    cd "$LA"
    MODEL="$M" GPU="$GPU" FF_ROOT="$X" setsid nohup bash ff_ec_test_chain_v3.sh > "$HX/logs/chain_ec_v3_launch.out" 2>&1 < /dev/null & disown
    echo "  v3 chain started, launcher-side pid $!" >> "$REC"
  done ;;
*) die "usage: $0 source | start" ;;
esac
