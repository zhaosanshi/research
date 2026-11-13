#!/usr/bin/env bash
# Stage B of from_claude/45 section 2 (Claude, 2026-11-13 23:xx; 赵磊 20:4x "可以统一重验，……规则不是写死，是写下来"): the
# rules v1 on the 529 test questions, for a model whose tune-stage runs exist (Llama, 9B). Every model step runs the E44
# scripts unchanged through the evaluation-only entry scripts/review_checks/e44_split_test.py, which only widens the
# accepted split set by "test" (the run identity stays the one of the tune stage; every artifact is checked against it by
# the core). Judging (AllenAI, DeepSeek Pro, single) and the tables are the tune stage's scripts, unchanged.
#   MODEL=<config> CODE_PATH=<code copy of the runs> SOURCE_DIR=<source dir> E_RUN=<dir> C_RUN=<dir> E_ON=E2 C_E_ON=E1
#   EL=<E layer, R3> CLS="<C layers, R5>" ECLS="<C layers of E+C, R6 (+ secondary)>" [ES="1 2"] [KS="0.25 0.5 1 2"]
#   STAGE=<stage> [GPU=k] [LAYER=n] [TABLE_CODE=<copy with the table scripts; default CODE_PATH>] ff_test_eval.sh
# stages:
#   pilot    execution check on the first 3 test questions: freeze the 11 prompts as a pilot file (t.*: never read by a
#            full run); prints the 3 questions with their answers (GPU). No pilot generation (see the stage).
#   freeze   the 11 prompts on the 529 test questions in E_RUN (GPU); the file is then copied byte for byte into C_RUN
#            (the two runs share one identity; the core checks it whenever the file is read)
#   egen     E at EL: the must-answer arms of R2 / R4 (strengths of ES: w, md, rand both signs) and the single writes on x
#            that the E+C reproduction check compares against (w, rand both signs, each ES) (GPU)
#   cgen     C at every layer of CLS: the six arms of R5 at 0.25 (w, md, rand, both signs, base x); at every layer of ECLS
#            also x.w.m<k> for each KS (the E+C reproduction check) (GPU)
#   ejudge   export (prompts conf, conf_<E_ON>; the must-answer E arms) + AllenAI (GPU) + DeepSeek Pro (no cache seed: no
#            test answer was ever judged); work dir E_RUN/judge_test_testD_E_L<EL>
#   cjudge   LAYER=n of CLS: export (prompt x; the six arms) + AllenAI (GPU) + DeepSeek Pro; work dir C_RUN/judge_test_testS_C_L<n>
#            Both judge stages take JUDGES (default "allenai pro"): JUDGES=allenai = export + AllenAI only (GPU); JUDGES=pro =
#            DeepSeek Pro only (no GPU), on a work dir whose AllenAI pass is complete.
#   etables  e44_round1_tables.py cells (conf base, each ES, both judges) + ff_explore_dose_table.py (read-only)
#   ctables  LAYER=n: e44_round1_tables.py cells (x base, strength 0.25, both judges) (read-only)
# E+C on the test questions: ff_explore_ec.sh with SPLIT=test (same entry script).
set -euo pipefail
: "${MODEL:?}" "${STAGE:?}" "${CODE_PATH:?}" "${SOURCE_DIR:?}" "${E_RUN:?}" "${C_RUN:?}" "${EL:?}"
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$CODE_PATH
TCODE=${TABLE_CODE:-$CODE_PATH}
SRC=$SOURCE_DIR
E_ON=${E_ON:-E2}; C_E_ON=${C_E_ON:-E1}
ES=${ES:-1 2}; KS=${KS:-0.25 0.5 1 2}; CLS=${CLS:-}; ECLS=${ECLS:-}
MODELS=$ROOT/models
host() { case "$1" in "$ROOT"/*) echo "$HOST/${1#$ROOT/}" ;; *) echo "path $1 is outside $ROOT" >&2; exit 2 ;; esac; }
case "$EL" in ""|*[!0-9]*) echo "EL must be a layer number" >&2; exit 2 ;; esac
for L in $CLS $ECLS ${LAYER:-}; do case "$L" in *[!0-9]*) echo "layer '$L' is not a number" >&2; exit 2 ;; esac; done
for s in $ES $KS; do case "$s" in ""|*[!0-9.]*|.*|*.|*.*.*) echo "strength '$s' is not a number like 1 or 0.25" >&2; exit 2 ;; esac; done
for d in "$E_RUN" "$C_RUN"; do [ -f "$(host "$d")/frozen_train.json" ] || { echo "$d is not a finished run (no frozen_train.json)" >&2; exit 3; }; done
[ -f "$(host "$SRC")/bank.json" ] || { echo "no source directory $SRC" >&2; exit 3; }
for f in scripts/review_checks/e44_split_test.py scripts/review_checks/judge_two_axis_run.py scripts/allenai_tqa_judge.py; do
  [ -f "$(host "$CODE")/$f" ] || { echo "code copy $CODE has no $f" >&2; exit 3; }
done
for f in e44_round1_tables.py ff_explore_dose_table.py; do
  [ -f "$(host "$TCODE")/scripts/review_checks/$f" ] || { echo "table code copy $TCODE has no $f" >&2; exit 3; }
done
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$MODELS
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
cfg() { "${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))" "$CODE/configs/models/$MODEL.json" "$1" "$2"; }
PYM=$(cfg python python3)                 # model steps: the runtime the runs were made with
pym() { "${DX[@]}" "$PYM" scripts/review_checks/e44_split_test.py "$@"; }   # the E44 scripts, test split allowed
pyj() { "${DX[@]}" python3 "$@"; }        # judging steps: the old models' runtime, as in the tune stage
pyx() { "${DX[@]}" python3 scripts/review_checks/e44_split_test.py "$@"; }  # the judge export, in that runtime too (as the tune stage)
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$TCODE -w $TCODE truth_dec python3)
pyt() { "${TX[@]}" "$@"; }                 # read-only tables, from TABLE_CODE
need_gpu() { [ -n "${GPU:-}" ] || { echo "STAGE=$STAGE needs GPU" >&2; exit 2; }; }
step() { echo "=== $(date '+%F %T') [$MODEL test $STAGE gpu${GPU:-none} E$EL($E_ON) C[$CLS]($C_E_ON)] $* ==="; }
verdicts_complete() {  # a cell without a verdict stops the job (never imputed)
  local out bad empty
  # an EMPTY answer is never sent to a judge and counts as an abstention (two-axis standard of 2026-11-08: "空输出一律 C";
  # first seen on the 27B, 2026-11-14): it is counted and reported, not blocking; every other missing verdict still stops
  out=$(pyj -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
empty = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'empty_output_no_judge_vote' and r.get('empty_output') is True]
print(sum(r['status'] != 'scored' for r in rows) - len(empty), len(empty))" "$1/judge2/single_$2.jsonl") \
    || { echo "cannot read the $2 verdicts of $1" >&2; exit 3; }
  read -r bad empty <<< "$out"
  [ "$empty" = 0 ] || echo "$2: $empty empty answer(s) in $1 count as abstentions (never judged; two-axis standard of 11-08)"
  [ "$bad" = 0 ] || { echo "$2 left $bad cells without a verdict in $1: re-judge them before any table" >&2; exit 3; }
}
need_frozen() {
  [ -f "$(host "$E_RUN")/frozen_test.json" ] && [ -f "$(host "$C_RUN")/frozen_test.json" ] \
    && cmp -s "$(host "$E_RUN")/frozen_test.json" "$(host "$C_RUN")/frozen_test.json" \
    || { echo "run STAGE=freeze first (frozen_test.json in both runs, identical)" >&2; exit 3; }
}
JUDGES=${JUDGES:-allenai pro}
for j in $JUDGES; do case "$j" in allenai|pro) ;; *) echo "JUDGES: unknown judge '$j' (allenai / pro)" >&2; exit 2 ;; esac; done
[ -n "$JUDGES" ] || { echo "JUDGES is empty" >&2; exit 2; }
has() { case " $JUDGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
judge_both() {  # judge_both RUN NAME GEN_FILES PROMPTS: export + AllenAI, then DeepSeek Pro, on RUN/judge_test_<NAME>
  local run=$1 name=$2 gen=$3 prompts=$4 work f
  work=$run/judge_test_$name
  if ! has allenai; then                 # Pro only: the AllenAI pass (export, cells) must be complete already
    [ -f "$(host "$work")/judge2/single_allenai.jsonl" ] || { echo "no AllenAI pass in $work yet: run JUDGES=allenai first" >&2; exit 3; }
    verdicts_complete "$work" allenai
    judge_pro "$work"; return 0
  fi
  for f in $gen; do [ -f "$(host "$run")/gen/$f" ] || { echo "missing generation file $f in $run" >&2; exit 3; }; done
  step "export $name: prompts $prompts; $(echo $gen | wc -w) generation files"
  pyx export --source-run "$SRC" --run-dir "$run" --split test --gen $gen --prompts $prompts --name "$name" | tail -n 3
  mkdir -p "$(host "$work")/judge2/cache"
  [ -f "$(host "$work")/judge_input.json" ] || cp "$(host "$run")/judge_test/$name.json" "$(host "$work")/judge_input.json"
  cmp "$(host "$run")/judge_test/$name.json" "$(host "$work")/judge_input.json"
  step "cells + allenai + single"
  pyj scripts/review_checks/judge_two_axis_run.py cells "$work" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$work/judge2/cells.jsonl" --out "$work/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
  pyj scripts/review_checks/judge_two_axis_run.py single "$work" --judge allenai > "$(host "$work")/single_allenai_stdout.md"
  verdicts_complete "$work" allenai
  if has pro; then judge_pro "$work"; else step "pro skipped (JUDGES=$JUDGES)"; fi
}
judge_pro() {  # judge_pro WORK: DeepSeek Pro on the cells of the AllenAI pass (no GPU)
  step "pro"
  pyj scripts/review_checks/judge_two_axis_run.py llm "$1" --judge pro
  pyj scripts/review_checks/judge_two_axis_run.py single "$1" --judge pro > "$(host "$1")/single_pro_stdout.md"
  verdicts_complete "$1" pro
}
COMMON=(--source-run "$SRC" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
E_MUST=""; for s in $ES; do E_MUST="$E_MUST conf.w.p$s conf.md.p$s conf.rand.p$s conf.rand.m$s"; done
E_X="";    for s in $ES; do E_X="$E_X x.w.p$s x.rand.p$s x.rand.m$s"; done
C_R5="x.w.p0.25 x.w.m0.25 x.md.p0.25 x.md.m0.25 x.rand.p0.25 x.rand.m0.25"
C_EC="";   for k in $KS; do case " $C_R5 " in *" x.w.m$k "*) ;; *) C_EC="$C_EC x.w.m$k" ;; esac; done
tok() { local a=$1; echo "${a//./-}"; }

case "$STAGE" in
pilot)
  # only the freeze: the driver's pilot mode reads pilot directions (dirs/t.*), never the full run's, so a pilot
  # generation cannot use the trained directions (22:58 pilot); generation is checked after the full freeze instead
  # (the engine's selfcheck on the whole test split and the reproduction check of gen1)
  need_gpu
  step "pilot freeze: 11 prompts, first 3 test questions (t.frozen_test.json)"
  pym driver freeze "${COMMON[@]}" --run-dir "$E_RUN" --splits test --n-check 3
  "${DX[@]}" python3 -c "
import json, sys
fz = json.load(open(sys.argv[1] + '/t.frozen_test.json'))
print('pilot: split %s, %d questions, identity %s' % (fz['split'], len(fz['questions']), fz['identity_sha256'][:12]))
for q in fz['questions']:
    print('\n-', q['pair_id'], q['question'])
    for p in ('x', 'conf', 'conf_' + sys.argv[2], 'caut'):
        print('  %-8s %s' % (p, json.dumps(q['answers'][p]['text'], ensure_ascii=False)))
" "$E_RUN" "$E_ON" ;;
freeze)
  need_gpu
  step "freeze the 11 prompts on the test questions in $E_RUN"
  pym driver freeze "${COMMON[@]}" --run-dir "$E_RUN" --splits test
  if [ "$C_RUN" != "$E_RUN" ]; then
    [ -e "$(host "$C_RUN")/frozen_test.json" ] || cp -p "$(host "$E_RUN")/frozen_test.json" "$(host "$C_RUN")/frozen_test.json"
    cmp "$(host "$E_RUN")/frozen_test.json" "$(host "$C_RUN")/frozen_test.json"
    step "verify the copy in $C_RUN under the core's own check (no regeneration)"
    pym driver freeze "${COMMON[@]}" --run-dir "$C_RUN" --splits test
  fi ;;
egen)
  need_gpu; need_frozen
  step "E$EL test arms: $E_MUST $E_X"
  pym driver generate "${COMMON[@]}" --e-on "$E_ON" --run-dir "$E_RUN" --family E --layers "$EL" --split test --arms $E_MUST $E_X ;;
cgen)
  need_gpu; need_frozen; [ -n "$CLS" ] || { echo "cgen needs CLS" >&2; exit 2; }
  step "C test arms at [$CLS]: $C_R5"
  pym driver generate "${COMMON[@]}" --e-on "$C_E_ON" --run-dir "$C_RUN" --family C --layers $CLS --split test --arms $C_R5
  if [ -n "$ECLS" ] && [ -n "$C_EC" ]; then
    step "C single writes toward confident at [$ECLS] for the E+C reproduction check: $C_EC"
    pym driver generate "${COMMON[@]}" --e-on "$C_E_ON" --run-dir "$C_RUN" --family C --layers $ECLS --split test --arms $C_EC
  fi ;;
ejudge)
  if has allenai; then need_gpu; fi; need_frozen
  GEN=""; for a in $E_MUST; do GEN="$GEN E_L${EL}_$(tok "$a")_test.json"; done
  judge_both "$E_RUN" "testD_E_L$EL" "$GEN" "conf conf_$E_ON" ;;
cjudge)
  if has allenai; then need_gpu; fi; need_frozen; : "${LAYER:?cjudge needs LAYER}"
  case " $CLS " in *" $LAYER "*) ;; *) echo "LAYER $LAYER is not in CLS [$CLS]" >&2; exit 2 ;; esac
  GEN=""; for a in $C_R5; do GEN="$GEN C_L${LAYER}_$(tok "$a")_test.json"; done
  judge_both "$C_RUN" "testS_C_L$LAYER" "$GEN" "x" ;;
etables)
  W=$E_RUN/judge_test_testD_E_L$EL
  for s in $ES; do for J in allenai pro; do
    step "cells E$EL conf base strength $s, judge $J"
    pyt scripts/review_checks/e44_round1_tables.py cells "$E_RUN" E "$W" --judge "$J" --layers "$EL" --base conf --strength "$s"
  done; done
  step "dose table"
  pyt scripts/review_checks/ff_explore_dose_table.py --e-on "$E_ON" --work "$MODEL:$EL:$W" --strengths $ES ;;
ctables)
  : "${LAYER:?ctables needs LAYER}"
  for J in allenai pro; do
    step "cells C$LAYER x base strength 0.25, judge $J"
    pyt scripts/review_checks/e44_round1_tables.py cells "$C_RUN" C "$C_RUN/judge_test_testS_C_L$LAYER" --judge "$J" --layers "$LAYER" --strength 0.25
  done ;;
*) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step "$STAGE done"
