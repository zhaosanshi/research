#!/usr/bin/env bash
# Orchestrates the E44 first round on the four cards (one single-card process per card at a time; no model is split).
#   wave 1  main hypothesis: E layers 0-15 (card 0), E 16-31 (card 1), C 0-15 (card 2), C 16-31 (card 3)
#           then AllenAI judging of E (card 0) and C (card 1), main summary  -> marker MAIN_JUDGED
#   wave 2  declared side branches: M (cards 2 then 0), E2 (cards 3 then 1), MD after E2's first half (card 3)
#           then AllenAI judging of M, E2, MD                                  -> marker SIDE_JUDGED
# A failed step writes marker FAILED_<step> and the independent steps go on. Run inside the container.
set -uo pipefail
HERE=/workspace/Truth_Decoupling/review_e44_claude
LOGS=$HERE/logs_round1
MAIN=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b
SRC=/workspace/Truth_Decoupling/results_qa_das_v1u/llama3.1-8b/full
CODE=$HERE/claude_gpt_review/from_claude/code
mkdir -p "$LOGS"
LOW="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
HIGH="16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
say() { echo "=== $(date '+%F %T') $* ==="; }
run() {   # branch card layers tag
  say "start run $1 card $2 ($4)"
  if BRANCH=$1 GPU=$2 LAYERS="$3" STAGE=run bash "$HERE/e44_round1.sh" > "$LOGS/run_$1_$4.log" 2>&1; then say "ok run $1 ($4)"; else say "FAILED run $1 ($4)"; touch "$LOGS/FAILED_run_$1_$4"; return 1; fi
}
judge() { # branch card
  say "start judge $1 card $2"
  if BRANCH=$1 GPU=$2 STAGE=judge bash "$HERE/e44_round1.sh" > "$LOGS/judge_$1.log" 2>&1; then say "ok judge $1"; else say "FAILED judge $1"; touch "$LOGS/FAILED_judge_$1"; return 1; fi
}

say "wave 1: main hypothesis"
run E 0 "$LOW" a & p1=$!
run E 1 "$HIGH" b & p2=$!
run C 2 "$LOW" a & p3=$!
run C 3 "$HIGH" b & p4=$!
wait $p1; e1=$?; wait $p2; e2=$?; wait $p3; c1=$?; wait $p4; c2=$?
say "wave 1 finished: E $e1 $e2, C $c1 $c2"

# wave 2 starts on the cards that are free; judging of the main branches goes first on cards 0 and 1
( if [ $e1 = 0 ] && [ $e2 = 0 ]; then judge E 0; fi; run M 0 "$HIGH" b ) & q0=$!
( if [ $c1 = 0 ] && [ $c2 = 0 ]; then judge C 1; fi; run E2 1 "$HIGH" b ) & q1=$!
( run M 2 "$LOW" a ) & q2=$!
( run E2 3 "$LOW" a; run MD 3 "3 10 16" a ) & q3=$!

# marker for the main hypothesis as soon as both judgings are done (wave 2 keeps running)
( while :; do
    if grep -q "judge E done" "$LOGS/judge_E.log" 2>/dev/null && grep -q "judge C done" "$LOGS/judge_C.log" 2>/dev/null; then
      cd "$CODE" && python scripts/e44_factorial_das.py summarize --source-run "$SRC" --run-dir "$MAIN" > "$MAIN/summary_stdout.md" 2> "$LOGS/summarize_main.err" \
        && touch "$LOGS/MAIN_JUDGED" || touch "$LOGS/FAILED_summarize_main"
      break
    fi
    if compgen -G "$LOGS/FAILED_judge_[EC]" >/dev/null || compgen -G "$LOGS/FAILED_run_[EC]_*" >/dev/null; then touch "$LOGS/MAIN_FAILED"; break; fi
    sleep 20
  done ) & pm=$!

wait $q0; wait $q1; wait $q2; wait $q3; wait $pm
say "wave 2 runs finished"
judge M 2 & j1=$!
judge E2 3 & j2=$!
judge MD 0 & j3=$!
wait $j1; wait $j2; wait $j3
for b in E2 M initmd; do
  ( cd "$CODE" && python scripts/e44_factorial_das.py summarize --source-run "$SRC" --run-dir "${MAIN}_side_$b" > "${MAIN}_side_$b/summary_stdout.md" 2>> "$LOGS/summarize_side.err" ) || touch "$LOGS/FAILED_summarize_$b"
done
touch "$LOGS/SIDE_JUDGED"
say "all done; failure markers: $(ls "$LOGS" | grep -c '^FAILED' || true)"
