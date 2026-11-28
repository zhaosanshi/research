#!/bin/bash
# Iteration_01 exploration (executor R-bing, Claude Opus 5.5, 2026-11-29 Beijing): work queue R item R6
# (discussion/work_queue/2026-11-28_queue_r.md section 4): the GPU and judge steps around diag/relayer_1128.py, which
# decides every step (what to run next, the tables, the choice). NOT REVIEWED BY GPT: only "smoke" has been run; "run"
# and "judge" wait for the review.
# Programs, all unchanged: scripts/capability/cap_mcq_sweep.py (MMLU, option probability), scripts/newsets/ns_positions.py
# (GSM8K, write rule prompt), scripts/review_checks/e44_positions.py / e44_positions2.py (TruthfulQA tune generation and
# export at the prompt rule; two-write selfcheck), judge_two_axis_run.py (cells / Pro). Model settings:
# code/configs/newsets/relayer_1128_models.json (run_dir = the new run folders of relayer_1128_train.sh setup).
# modes (host, from /data/sanshizhao/Truth_Decoupling):
#   smoke KEY        CARDS=<cards held by the caller>. Partial runs that must reproduce archived answers exactly:
#                    MMLU first batch (8 items): x and one archived arm, picks equal to the archived file;
#                    GSM8K first batch (8 items): neutral:E<el>.w.p<es>, rows equal to the archived full pass;
#                    TruthfulQA tune first batch (8 questions) in the NEW run: x:E<el>.w.p<es> and the two-write
#                    x:E<el>.w.p<es>+C<cl>.w.m1 (after a one-batch selfcheck), rows equal to the archived prompt-rule files;
#                    then the CPU modes inv and cap of relayer_1128.py. Outputs only under capability/relayer_1128/smoke
#                    and t8.* files of the new run. Marker RELAYER_SMOKE_<KEY>_DONE / _FAILED.
#   run KEY [up|down]  (formal; after the GPT review) cards through the locks, then: MMLU rounds until relayer_1128.py
#                    mmlu-next is empty; GSM8K rounds until gsm-next is empty; tune generation of tune-next gen; export
#                    of tune-next export into a new work dir judge_tune_relayer_1128_E<r>_tune with its cells; then
#                    judging (mode judge, off-peak); then choose; confirm-next (full MMLU selection half, full GSM8K);
#                    confirm; c-next (selfcheck of the pair on the whole split, four E + C arms), export
#                    judge_tune_relayer_1128_C_tune, judging, c-choose. Each step resumes from the files.
#   judge KEY WORK   DeepSeek Pro on <new run>/eval_extra/<WORK>, off-peak only (weekdays 09-12 and 14-18 Beijing
#                    excluded), reply cache seeded with ONE existing prompt-rule cache of the E run, conflicts stop it;
#                    spending cap CAP yuan for R6 (sum of the upper estimates of crow_pro_requests.py, relayer_1128_cost.txt).
# Cards: Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock -n, retried every 60 s), ours plus others at most 66560 MiB.
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/relayer_1128_all.log
REL="python3 -B Iteration_01/diag/relayer_1128.py"
CFG=Iteration_01/code/configs/newsets/relayer_1128_models.json
CAPR=Iteration_01/capability/relayer_1128
MMP=data_external/mmlu_20261119/all/test-00000-of-00001.parquet
GSP=data_external/gsm8k_20261119/main/test-00000-of-00001.parquet
MM_IDS=Iteration_01/capability/sets/mmlu_1000_20261119.json
HALF_IDS=$CAPR/sets/mmlu_selection_half_relayer_1128.json
G304=Iteration_01/capability/sets/gsm8k_first304_20261124.json
GFULL=Iteration_01/capability/sets/gsm8k_1319_20261119.json
MAXMIB=66560
CAP=${CAP:-20}
FLOOR=10
say() { echo "$(date '+%F %T') relayer: $*" >> $ALL; }

setm() {
  KEY=$1
  case $1 in
    llama) NG=llama3.1-8b; ORIG=results_e44_factorial_das/llama3.1-8b_side_E2; SRC=results_qa_das_v1u/llama3.1-8b/full
           CRUN=results_e44_factorial_das/llama3.1-8b; CEON=E1; NCARD=1; NEED=20000; EL=16; ES=3.5; CL=19; GCAP=512
           MMREF=Iteration_01/capability/depthscan/llama3.1-8b_L16.json; MMS=3.5; GMAIN=Iteration_01/capability/gen_main/gsm8k/llama3.1-8b/b8 ;;
    q9)    NG=qwen3.5-9b; ORIG=results_newmodels_ff_20261112/e44/qwen3.5-9b_E2; SRC=results_newmodels_ff_20261112/source/qwen3.5-9b
           CRUN=results_newmodels_ff_20261112/e44/qwen3.5-9b; CEON=E1; NCARD=1; NEED=22000; EL=15; ES=2; CL=16; GCAP=1024
           MMREF=Iteration_01/capability/depthscan/qwen3.5-9b_L15.json; MMS=2; GMAIN=Iteration_01/capability/gen_main1024/gsm8k/qwen3.5-9b/b8 ;;
    q27)   NG=qwen3.8-27b; ORIG=results_newmodels_ff_20261112/e44/qwen3.8-27b-dm; SRC=results_newmodels_ff_20261112/source/qwen3.8-27b-dm
           CRUN=$ORIG; CEON=E2; NCARD=2; NEED=34000; EL=28; ES=1.75; CL=33; GCAP=1024
           MMREF=Iteration_01/capability/sweep/qwen3.8-27b.json; MMS=1.5; GMAIN=Iteration_01/capability/gen_main1024/gsm8k/qwen3.8-27b/b8 ;;
    g12)   NG=gemma3-12b; ORIG=Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b; SRC=Iteration_01/results/ff/gemma3/12b/source/gemma3-12b
           CRUN=$ORIG; CEON=E2; NCARD=1; NEED=30000; EL=24; ES=2.5; CL=25; GCAP=512
           MMREF=Iteration_01/capability/layermap/gemma3-12b.json; MMS=2; GMAIN=Iteration_01/capability/gen_main/gsm8k/gemma3-12b/b8 ;;
    g27)   NG=gemma3-27b; ORIG=Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b; SRC=Iteration_01/results/ff/gemma3/27b/source/gemma3-27b
           CRUN=$ORIG; CEON=E2; NCARD=1; NEED=62000; EL=28; ES=1; CL=31; GCAP=512
           MMREF=Iteration_01/capability/layermap/gemma3-27b.json; MMS=1; GMAIN=Iteration_01/capability/gen_main/gsm8k/gemma3-27b/b8 ;;
    *) echo "unknown key $1" >&2; return 2 ;;
  esac
  RUN=Iteration_01/results/relayer_1128/$NG/run_E2
  CODE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["code"])' $CFG $NG) || return 2
  PYM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $CFG $NG) || return 2
  LOG=$L/relayer_1128_$KEY.log
  COMMON=(--source-run $W/$SRC --run-dir $W/$RUN --e-on E2 --c-run-dir $W/$CRUN --c-e-on $CEON --split tune
          --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  return 0
}
hy() { echo "${1//./-}"; }
dmodel() {  # dmodel ARGS...: a model program in the container, from the code copy's root, on CARDS
  docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models \
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES="$CARDS" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$W/$CODE -w $W/$CODE truth_dec "$PYM" "$@"
}
droot() {  # droot ARGS...: a model program from /workspace/Truth_Decoupling (cap_mcq_sweep, ns_positions), on CARDS
  docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_VISIBLE_DEVICES="$CARDS" -w $W truth_dec "$PYM" -I "$@"
}
dcpu() { docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec "$@"; }
st() { local name=$1 rc; shift; echo "=== $(date '+%F %T') $KEY $name: $*" >> $LOG; "$@" >> $LOG 2>&1; rc=$?; echo "$(date '+%F %T') $KEY $name exit=$rc" >> $LOG; return $rc; }
mmlu() {  # mmlu OUT IDS [--first-batches K] -- ARMS...
  local out=$1 ids=$2; shift 2
  local fb=(); [ "$1" = "--first-batches" ] && { fb=(--first-batches $2); shift 2; }
  [ "$1" = "--" ] && shift
  mkdir -p $(dirname $out)
  droot Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models $CFG --model $NG --items $MMP --ids $ids --out $out "${fb[@]}" --arms "$@"
}
gsm() {  # gsm OUT IDS [--first-batches K] -- ARMS...
  local out=$1 ids=$2; shift 2
  local fb=(); [ "$1" = "--first-batches" ] && { fb=(--first-batches $2); shift 2; }
  [ "$1" = "--" ] && shift
  mkdir -p $out
  droot Iteration_01/code/scripts/newsets/ns_positions.py prompt gen --models $CFG --model $NG --task gsm8k --items $GSP \
    --ids $ids --max-new-tokens $GCAP --batch-size 8 --out $out "${fb[@]}" --arms "$@"
}

smoke() {
  setm $1 || exit 2
  [ -n "${CARDS:-}" ] || { echo "smoke needs CARDS (held by the caller)" >&2; exit 2; }
  rm -f $L/RELAYER_SMOKE_${KEY}_FAILED
  local sd=$CAPR/smoke a1=x:E$EL.w.p$ES a2=x:E$EL.w.p$ES+C$CL.w.m1 ok=1
  local f1=x--E$EL-w-p$(hy $ES)-at-prompt_tune.json f2=x--E$EL-w-p$(hy $ES)-and-C$CL-w-m1-at-prompt_tune.json
  say "$KEY smoke on cards $CARDS"
  # 1. MMLU, first batch of the 1,000 list, against the archived file holding x:E<EL>.w.p<MMS>
  [ -s $sd/mmlu_$NG.json ] || st mmlu8 mmlu $sd/mmlu_$NG.json $MM_IDS --first-batches 1 -- x x:E$EL.w.p$MMS || ok=0
  [ $ok = 1 ] && { st mmlu8_cmp python3 - $sd/mmlu_$NG.json $MMREF x:E$EL.w.p$MMS <<'EOF'
import json, sys
a, b, arm = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
ok = a["identity_sha256"] == b["identity_sha256"] and a["ids_file_sha256"] == b["ids_file_sha256"] and a["n"] == 8
for name in ("x", arm):
    ra, rb = a["arms"][name]["rows"], b["arms"][name]["rows"][:8]
    same = [x["id"] == y["id"] and x["pick"] == y["pick"] for x, y in zip(ra, rb)]
    gap = max(abs(x["logp"][L] - y["logp"][L]) for x, y in zip(ra, rb) for L in "ABCD")
    ok &= all(same) and len(ra) == 8
    print("MMLU %s: %d of 8 picks equal to %s; largest log-prob difference %.4f" % (name, sum(same), sys.argv[2], gap))
    if name == arm:
        wa, wb = a["arms"][name]["writes"], b["arms"][name]["writes"]
        print("  write amount new %.6f archived %.6f" % (wa[0]["amount"], wb[0]["amount"]))
        ok &= abs(wa[0]["amount"] - wb[0]["amount"]) < 1e-6 * max(1.0, abs(wb[0]["amount"]))
sys.exit(0 if ok else 1)
EOF
  } || ok=0
  # 2. GSM8K, first batch, against the archived full pass of neutral:E<EL>.w.p<ES> (prompt rule)
  [ $ok = 1 ] && { st gsm8 gsm $sd/gsm/$NG/b8 $GFULL --first-batches 1 -- neutral:E$EL.w.p$ES || ok=0; }
  [ $ok = 1 ] && { st gsm8_cmp python3 - $sd/gsm/$NG/b8 $GMAIN/neutral--E$EL-w-p$(hy $ES)-at-prompt.json <<'EOF'
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/s*.neutral--E*-at-prompt.json")
if len(fs) != 1:
    sys.exit("expected one partial file, found %s" % fs)
a, b = json.load(open(fs[0])), json.load(open(sys.argv[2]))
ra, rb = a["rows"], b["rows"][:8]
same = [x["id"] == y["id"] and x["text"] == y["text"] and x["token_ids"] == y["token_ids"] for x, y in zip(ra, rb)]
pa, pb = a["contract"]["parts"][0], b["contract"]["parts"][0]
print("GSM8K: %d of 8 rows identical to %s; direction sha256 %s / %s; positions %s" % (
    sum(same), sys.argv[2], pa["direction_file_sha256"][:12], pb["direction_file_sha256"][:12], a["contract"]["positions"]))
sys.exit(0 if all(same) and len(ra) == 8 and pa["direction_file_sha256"] == pb["direction_file_sha256"] and a["contract"]["positions"] == "prompt" else 1)
EOF
  } || ok=0
  # 3. TruthfulQA tune, first batch, in the new run: single E and the two-write arm
  [ $ok = 1 ] && { st tqa8 dmodel scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --first-batches 1 --arms $a1 || ok=0; }
  [ $ok = 1 ] && { st selfcheck8 dmodel scripts/review_checks/e44_positions2.py prompt selfcheck "${COMMON[@]}" --first-batches 1 --arms $a2 || ok=0; }
  [ $ok = 1 ] && { st tqa8two dmodel scripts/review_checks/e44_positions2.py prompt generate "${COMMON[@]}" --first-batches 1 --arms $a2 || ok=0; }
  [ $ok = 1 ] && { st tqa8_cmp python3 - $RUN/eval_extra/gen $ORIG/eval_extra/gen $f1 $f2 <<'EOF'
import json, sys
new, old, f1, f2 = sys.argv[1:5]
ok = True
for f in (f1, f2):
    a, b = json.load(open("%s/t8.%s" % (new, f))), json.load(open("%s/%s" % (old, f)))
    ra, rb = a["rows"], b["rows"][:8]
    same = [x["pair_id"] == y["pair_id"] and x["text"] == y["text"] and x["token_ids"] == y["token_ids"] for x, y in zip(ra, rb)]
    pos = (a.get("positions"), (a.get("generation_contract") or {}).get("positions"))
    print("TruthfulQA %s: %d of 8 answers identical to the archived file; positions %s" % (f, sum(same), pos))
    ok &= all(same) and len(ra) == 8 and pos == ("prompt", "prompt")
sys.exit(0 if ok else 1)
EOF
  } || ok=0
  # 4. the CPU modes of the selection program on the new run
  [ $ok = 1 ] && { st inv dcpu $REL inv $KEY || ok=0; }
  [ $ok = 1 ] && { st cap dcpu $REL cap $KEY || ok=0; }
  [ $ok = 1 ] && { st next dcpu $REL mmlu-next $KEY || ok=0; }
  if [ $ok = 1 ]; then touch $L/RELAYER_SMOKE_${KEY}_DONE; say "$KEY smoke passed"; else touch $L/RELAYER_SMOKE_${KEY}_FAILED; say "$KEY smoke FAILED (see $LOG)"; exit 1; fi
}

# ----------------------------------------------------------------------------- formal run (after the GPT review)
CARDS=${CARDS:-}; FDS=()
getcards() {  # getcards N MIB
  local n=$1 need=$2 g fd used got
  while true; do
    got=(); FDS=()
    for g in 0 1 2 3; do
      [ ${#got[@]} -ge $n ] && break
      exec {fd}>"Iteration_01/newsets/selfjudge/.gpu$g.lock" || return 1
      if flock -n $fd; then
        used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1 == g {print $2}')
        if [[ $used =~ ^[0-9]+$ ]] && [ $((used + need)) -le $MAXMIB ]; then got+=($g); FDS+=($fd); continue; fi
      fi
      exec {fd}>&-
    done
    if [ ${#got[@]} -ge $n ]; then CARDS=$(IFS=,; echo "${got[*]}"); say "$KEY: cards $CARDS"; return 0; fi
    for fd in "${FDS[@]}"; do exec {fd}>&-; done
    FDS=(); sleep 60
  done
}
putcards() { local fd; for fd in "${FDS[@]}"; do exec {fd}>&-; done; say "$KEY: released cards $CARDS"; FDS=(); CARDS=""; }
fail() { say "$KEY FAILED: $*"; touch $L/RELAYER_RUN_${KEY}_FAILED; exit 1; }
export_cells() {  # export_cells NAME POSSCRIPT GENFILES...: export, work dir, cells
  local name=$1 ps=$2; shift 2
  local wk=$RUN/eval_extra/judge_tune_$name
  [ -f $wk/judge2/cells.jsonl ] && return 0
  st export_$name dmodel scripts/review_checks/$ps prompt export "${COMMON[@]}" --gen "$@" --name $name || return 1
  mkdir -p $wk/judge2/cache && cp $RUN/eval_extra/judge_tune/$name.json $wk/judge_input.json || return 1
  st cells_$name dmodel scripts/review_checks/judge_two_axis_run.py cells $W/$wk --generation judge_input.json
}
run_cards() {  # the card part of one model, resumable
  local order=$1 r arms gens n
  getcards $NCARD $NEED || fail "cards"
  r=0; arms=$(dcpu $REL mmlu-next $KEY) || fail "mmlu-next"
  while [ -n "$arms" ]; do
    r=$((r + 1)); n=$(ls $CAPR/mmlu1000/$NG 2>/dev/null | wc -l)
    st mmlu_round$r mmlu $CAPR/mmlu1000/$NG/round$((n + 1)).json $MM_IDS -- x $arms || fail "MMLU round $r"
    arms=$(dcpu $REL mmlu-next $KEY) || fail "mmlu-next"
  done
  r=0; arms=$(dcpu $REL gsm-next $KEY $order) || fail "gsm-next"
  while [ -n "$arms" ]; do
    r=$((r + 1))
    st gsm_round$r gsm $CAPR/gsm304/$NG/b8 $G304 -- $arms || fail "GSM8K round $r"
    arms=$(dcpu $REL gsm-next $KEY $order) || fail "gsm-next"
  done
  st cap dcpu $REL cap $KEY $order || fail "cap table"
  arms=$(dcpu $REL tune-next $KEY gen $order) || fail "tune-next gen"
  [ -n "$arms" ] && { st tune_gen dmodel scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --arms $arms || fail "tune generation"; }
  gens=$(dcpu $REL tune-next $KEY export $order) || fail "tune-next export"
  if [ -n "$gens" ]; then
    n=$(ls -d $RUN/eval_extra/judge_tune_relayer_1128_E*_tune 2>/dev/null | wc -l)
    export_cells relayer_1128_E$((n + 1))_tune e44_positions.py $gens || fail "tune export / cells"
  fi
  putcards
}
case "${1:-}" in
smoke) smoke ${2:?KEY} ;;
run)
  setm ${2:?KEY} || exit 2; ORDER=${3:-up}
  [ -e $L/RELAYER_SMOKE_${KEY}_DONE ] || fail "no smoke"
  [ -e $L/RELAYER_T_${KEY}_DONE ] || [ $KEY = llama ] || fail "training not done (RELAYER_T_${KEY}_DONE)"
  say "$KEY formal run starts (order $ORDER)"
  run_cards $ORDER
  say "$KEY: card part of the selection done; next: judge every judge_tune_relayer_1128_E*_tune, then choose, confirm, C" ;;
judge)
  echo "judging is written for the review in the plan text of this file's header; not enabled before GPT has reviewed it" >&2; exit 2 ;;
*) echo "usage: relayer_1128.sh smoke KEY | run KEY [up|down]" >&2; exit 2 ;;
esac
