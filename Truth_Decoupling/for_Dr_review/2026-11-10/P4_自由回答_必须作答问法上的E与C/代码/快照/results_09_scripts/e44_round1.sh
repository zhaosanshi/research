#!/usr/bin/env bash
# E44 first round (plan 25 -> 26 section 6 -> 27 section 3). One branch (or one layer part of it) per process, one GPU per process.
#   BRANCH=E | C         main hypothesis (truth levels none / E1), main run dir
#   BRANCH=E2 | M | MD   declared side branches (E family only; each changes ONE thing), own run dir:
#                        E2 = other truth sentence, M = low level "popular belief" sentence, MD = mean-difference init at L3/10/16
#   STAGE=run            train -> generate (tune) -> reference readout (plain-prompt arms, tune) for LAYERS (default: all of the branch)
#   STAGE=judge          export every generation of the branch + AllenAI (needs all layers of the branch finished)
#   STAGE=pilot          three-question execution check with its own pilot freeze (t.* files, separate _pilot run dir, no judging)
# Code = GPT's 6bfca9f tree, unchanged. Nothing is selected here; Pro judging and layer selection are separate, later steps.
# The four cards produce byte-identical frozen answers (checked 2026-11-10 23:55, sha 107b117d...), so a branch may run on any card.
set -euo pipefail
: "${BRANCH:?}" "${GPU:?}" "${STAGE:?}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
ROOT=/workspace/Truth_Decoupling
MAIN=$ROOT/results_e44_factorial_das/llama3.1-8b
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
CODE=$ROOT/review_e44_claude/claude_gpt_review/from_claude/code
MODELS=$ROOT/models
ALL=$(seq 0 31 | tr '\n' ' ')
FACTOR=(); TRAIN_ONLY=()
case "$BRANCH" in
  E)  FAMILY=E; RUN=$MAIN;               DEF=$ALL;      LO=conf;   HI=conf_E1; CLO=caut;   CHI=caut_E1 ;;
  C)  FAMILY=C; RUN=$MAIN;               DEF=$ALL ;;
  E2) FAMILY=E; RUN=${MAIN}_side_E2;     DEF=$ALL;      LO=conf;   HI=conf_E2; CLO=caut;   CHI=caut_E2; FACTOR=(--e-on E2) ;;
  M)  FAMILY=E; RUN=${MAIN}_side_M;      DEF=$ALL;      LO=conf_M; HI=conf_E1; CLO=caut_M; CHI=caut_E1; FACTOR=(--e-off M) ;;
  MD) FAMILY=E; RUN=${MAIN}_side_initmd; DEF="3 10 16"; LO=conf;   HI=conf_E1; CLO=caut;   CHI=caut_E1; TRAIN_ONLY=(--init md) ;;
  *) echo "unknown BRANCH $BRANCH" >&2; exit 2 ;;
esac
if [ "$FAMILY" = E ]; then
  GEN_ARMS="$LO.wq.p1 $HI.wq.m1 $CLO.wq.p1 $CHI.wq.m1 $LO.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
  MC_ARMS="x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
else
  GEN_ARMS="conf.wq.p1 caut.wq.m1 conf_E1.wq.p1 caut_E1.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1"
  MC_ARMS="x.w.p1 x.w.m1"
fi
N_ARMS=$(echo $GEN_ARMS | wc -w)
step() { echo "=== $(date '+%F %T') [$BRANCH gpu$GPU $STAGE] $* ==="; }
cd "$CODE"

if [ "$STAGE" = judge ]; then
  NAME=round1_$BRANCH
  WORK=$RUN/judge_tune_$NAME
  GEN_FILES=$(cd "$RUN/gen" && ls ${FAMILY}_L*_tune.json)
  want=$(( $(echo $DEF | wc -w) * N_ARMS )); have=$(echo $GEN_FILES | wc -w)
  [ "$have" = "$want" ] || { echo "generation files: have $have, want $want" >&2; exit 3; }
  step "export $NAME ($have generation files)"
  python scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split tune --gen $GEN_FILES --name "$NAME" | tail -n 3
  mkdir -p "$WORK"
  [ -f "$WORK/judge_input.json" ] || cp "$RUN/judge_tune/$NAME.json" "$WORK/judge_input.json"
  cmp "$RUN/judge_tune/$NAME.json" "$WORK/judge_input.json"
  step "cells + allenai + single"
  python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  python scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 1
  python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WORK/single_allenai_stdout.md"
  step "judge $BRANCH done"; exit 0
fi

PIL=(); VAL=16; LAYERS=${LAYERS:-$DEF}
if [ "$STAGE" = pilot ]; then
  RUN=${RUN}_pilot_$BRANCH; PIL=(--n-check 3); VAL=1
  [ "$BRANCH" = MD ] || LAYERS="3 16"
elif [ "$STAGE" != run ]; then
  echo "unknown STAGE $STAGE" >&2; exit 2
fi
COMMON=(--source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
mkdir -p "$RUN"
if [ "$STAGE" = pilot ]; then
  step "pilot freeze (three questions per split; a three-question batch cannot reuse the full freeze)"
  python scripts/e44_factorial_das.py freeze "${COMMON[@]}" --splits train tune "${PIL[@]}"
elif [ "$RUN" != "$MAIN" ]; then                   # frozen answers are copied byte for byte; the code re-validates them
  for s in train tune; do
    if [ ! -f "$RUN/frozen_$s.json" ]; then cp "$MAIN/frozen_$s.json" "$RUN/.frozen_$s.$$" && mv -n "$RUN/.frozen_$s.$$" "$RUN/frozen_$s.json"; rm -f "$RUN/.frozen_$s.$$"; fi
    cmp "$MAIN/frozen_$s.json" "$RUN/frozen_$s.json"
  done
fi

step "train $FAMILY layers: $LAYERS"
python scripts/e44_factorial_das.py train "${COMMON[@]}" --family "$FAMILY" --layers $LAYERS --val-n "$VAL" "${FACTOR[@]}" "${TRAIN_ONLY[@]}" "${PIL[@]}"
step "generate tune arms: $GEN_ARMS"
python scripts/e44_factorial_das.py generate "${COMMON[@]}" --family "$FAMILY" --layers $LAYERS --split tune --arms $GEN_ARMS "${FACTOR[@]}" "${PIL[@]}"
step "reference readout (plain-prompt arms): $MC_ARMS"
python scripts/e44_factorial_das.py mc "${COMMON[@]}" --family "$FAMILY" --layers $LAYERS --split tune --arms $MC_ARMS "${FACTOR[@]}" "${PIL[@]}"
if [ "$STAGE" = pilot ]; then
  step "pilot summary"
  python scripts/e44_factorial_das.py summarize --source-run "$SRC" --run-dir "$RUN" --include-pilot | tail -n 70
fi
step "$STAGE $BRANCH done (layers: $LAYERS)"
