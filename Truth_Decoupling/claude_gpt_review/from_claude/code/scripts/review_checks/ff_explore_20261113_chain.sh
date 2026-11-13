#!/usr/bin/env bash
# Exploration chain of 2026-11-13 (Claude; from_claude/44). 赵磊 11-13: a design change looks for the setting common to
# several models and changes the least; the program is a core plus a periphery that passes each model's settings in as
# parameters; this is exploration (rules may be adjusted until general ones are found, then every model is validated the
# same way); compare Qwen3-8B with Qwen3.5-9B too, same family.
# Host-side orchestrator. Every job is one call of the periphery runner newmodel_e44.sh, in a clean environment, with
# parameters; logs and DONE / FAIL flags go to results_newmodels_ff_20261112/logs/explore/ (chain_jobs.log lists start /
# done / FAIL / skip). A job whose DONE flag exists is skipped, so the chain can be restarted (old FAIL flags are cleared
# at start; one chain at a time, lock directory .lock_chain). When a job fails, the later jobs of its group are marked
# FAIL ("not tried"), so nothing waits forever; independent groups go on. The core (core/*.py, e44_factorial_das.py) is
# the frozen one of 11-12, byte for byte.
#   GPU 0-2   9B, truth sentence E2 instead of E1 (VARIANT=E2; the only change against round 1): E trained on layers 8-18
#             (8-11 / 12-15 / 16-18) with round 1's arms; then round 1's six pre-Pro arms and the criterion's arms at
#             strengths 0.5 and 2; AllenAI on all eleven layers (one job per layer); round 1's judge + screen over the 11
#             layers (GPU 0). GPU 2 then runs the prompt-only diagnostics (e44_promptdiag_v1.json) for the 9B, Llama and
#             Qwen3-8B. Afterwards: Qwen3-8B round 1 (GPU 0: C 0-17, GPU 1: C 18-35, GPU 2: E 0-17).
#   GPU 3     9B, round 1's C directions unchanged (only the strength changes): strengths 0.5 and 0.25 with their random and
#             mean-difference controls on round 1's three Pro layers (AllenAI). Then Qwen3-8B: three-question execution
#             checks (E, C), step 0 (round 1's 11 prompts). Afterwards: Qwen3-8B round 1, E 18-35. GPU 3 also holds a
#             MiniMax-H3 service of another user (about 5.3 GB of 73), left alone, as on 11-12.
#   no GPU    DeepSeek Pro passes, each after its own layer's AllenAI pass (one request at a time per job); for the C trial
#             on layers 12 / 16 the reply cache is seeded from round 1's proB work dir of that layer (same answers judged
#             there), otherwise from step 0. Tables per layer as soon as that layer is judged.
# Qwen3-8B round 1 uses Llama's round-1 settings unchanged (E1, strength 1, all 36 layers): the same-family comparison with
# the 9B's round 1. Its Pro layers are chosen afterwards from the frozen screen, by the rules of plan 43 section 5.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
RUNNER=${CHAIN_RUNNER:-$R/code_ff_explore_20261113/scripts/review_checks/newmodel_e44.sh}   # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore}
POLL=${CHAIN_POLL:-30}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another explore chain holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL                                      # a restart retries every job without a DONE flag
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME VAR=value ...: one runner call in a clean environment; skipped when NAME.DONE exists
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if env "${CLEAN[@]}" CODE_PATH=$W/code_ff_explore_20261113 "$@" bash "$RUNNER" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}
give_up() {  # give_up NAME...: jobs that will not be tried get a FAIL flag, so nothing waits for them forever
  local n
  for n in "$@"; do
    [ -e "$LOG/$n.DONE" ] || [ -e "$LOG/$n.FAIL" ] || { echo "not tried: an earlier job of its group failed" > "$LOG/$n.FAIL"; note "skip  $n"; }
  done
}
wait_done() {  # wait_done NAME...: until each has DONE; returns 1 as soon as one has FAIL
  local f
  for f in "$@"; do
    until [ -e "$LOG/$f.DONE" ]; do [ -e "$LOG/$f.FAIL" ] && return 1; sleep "$POLL"; done
  done
}

Q9=qwen3.5-9b; Q8=qwen3-8b; LL=llama3.1-8b
V=(MODEL=$Q9 VARIANT=E2 E_ON=E2)
E2_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E2.wq.m1 conf.iso.p1 caut.wq.p1 caut_E2.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_BASES="x x_F conf conf_E2 caut caut_E2"
E_EXTRA_A="x.rand.m1 x_F.w.p1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E_S="x.w.p0.5 x.rand.p0.5 x.rand.m0.5 x_F.w.p0.5 x.w.p2 x.rand.p2 x.rand.m2 x_F.w.p2"
C_S="x.w.p0.5 x.w.m0.5 x.md.p0.5 x.md.m0.5 x.rand.p0.5 x.rand.m0.5 x.w.p0.25 x.w.m0.25 x.md.p0.25 x.md.m0.25 x.rand.p0.25 x.rand.m0.25"
E2L_0="8 9 10 11"; E2L_1="12 13 14 15"; E2L_2="16 17 18"; E2_PRO_LAYERS=" 9 11 12 13 14 15 18 "
Q8_A="$(seq -s ' ' 0 17)"; Q8_B="$(seq -s ' ' 18 35)"
DIAG=e44_promptdiag_v1.json

note "chain start (pid $$)"
{ job src_$Q8 MODEL=$Q8 STAGE=source && job src_$LL MODEL=$LL STAGE=source && job e2_seed "${V[@]}" STAGE=seed; } || exit 1

e2_names() { local L; echo "e2_run_g$1 e2_extra_g$1"; for L in $2; do echo "e2_allenai_L$L"; done; }
e2_group() {  # e2_group GPU LAYERS: train + run arms, extra arms, then AllenAI on the Pro arm set, one job per layer
  local g=$1 L
  { job e2_run_g$g "${V[@]}" STAGE=run BRANCH=E GPU=$g LAYERS="$2" &&
    job e2_extra_g$g "${V[@]}" STAGE=extra BRANCH=E GPU=$g LAYERS="$2" ARMS="$E_EXTRA_A $E_S"; } || { give_up $(e2_names $g "$2"); return 0; }
  for L in $2; do
    job e2_allenai_L$L "${V[@]}" STAGE=pro BRANCH=E LAYER=$L TAG=X JUDGES=allenai GPU=$g ARMS="$E2_PROA $E_S" BASES="$E2_BASES" || true
  done
}
gpu0() {
  e2_group 0 "$E2L_0"
  { wait_done e2_run_g0 e2_run_g1 e2_run_g2 &&
    job e2_judge "${V[@]}" STAGE=judge BRANCH=E GPU=0 LAYERS="$E2L_0 $E2L_1 $E2L_2" &&
    job e2_screen "${V[@]}" STAGE=screen BRANCH=E; } || give_up e2_judge e2_screen
  { wait_done q8_step0 && job q8_run_C_a MODEL=$Q8 STAGE=run BRANCH=C GPU=0 LAYERS="$Q8_A"; } || give_up q8_run_C_a
}
gpu1() {
  e2_group 1 "$E2L_1"
  { wait_done q8_step0 && job q8_run_C_b MODEL=$Q8 STAGE=run BRANCH=C GPU=1 LAYERS="$Q8_B"; } || give_up q8_run_C_b
}
gpu2() {
  e2_group 2 "$E2L_2"
  job diag_$Q9 MODEL=$Q9 STAGE=diag GPU=2 DIAG_PROMPTS=$DIAG || true
  job diag_$LL MODEL=$LL STAGE=diag GPU=2 DIAG_PROMPTS=$DIAG || true
  job diag_$Q8 MODEL=$Q8 STAGE=diag GPU=2 DIAG_PROMPTS=$DIAG || true
  { wait_done q8_step0 && job q8_run_E_a MODEL=$Q8 STAGE=run BRANCH=E GPU=2 LAYERS="$Q8_A"; } || give_up q8_run_E_a
}
gpu3() {
  local L
  if job c_s_extra MODEL=$Q9 STAGE=extra BRANCH=C GPU=3 LAYERS="12 16 17" ARMS="$C_S"; then
    for L in 12 16 17; do
      job c_s_allenai_L$L MODEL=$Q9 STAGE=pro BRANCH=C LAYER=$L TAG=S JUDGES=allenai GPU=3 ARMS="$C_S" BASES=x || true
    done
  else
    give_up c_s_allenai_L12 c_s_allenai_L16 c_s_allenai_L17
  fi
  { job q8_pilot_E MODEL=$Q8 STAGE=pilot BRANCH=E GPU=3 && job q8_pilot_C MODEL=$Q8 STAGE=pilot BRANCH=C GPU=3 &&
    job q8_step0 MODEL=$Q8 STAGE=step0 GPU=3; } || give_up q8_pilot_C q8_step0
  { wait_done q8_step0 && job q8_run_E_b MODEL=$Q8 STAGE=run BRANCH=E GPU=3 LAYERS="$Q8_B"; } || give_up q8_run_E_b
}
pro_e2() {  # pro_e2 LAYER...: DeepSeek Pro, each layer after its own AllenAI pass
  local L
  for L in "$@"; do
    { wait_done e2_allenai_L$L &&
      job e2_pro_L$L "${V[@]}" STAGE=pro BRANCH=E LAYER=$L TAG=X JUDGES=pro ARMS="$E2_PROA $E_S" BASES="$E2_BASES"; } || give_up e2_pro_L$L
  done
}
pro_c() {
  local L cf
  for L in 12 16 17; do
    cf=""; [ "$L" = 17 ] || cf=$W/results_newmodels_ff_20261112/e44/$Q9/judge_tune_proB_C_L$L/judge2/cache
    { wait_done c_s_allenai_L$L &&
      job c_s_pro_L$L MODEL=$Q9 STAGE=pro BRANCH=C LAYER=$L TAG=S JUDGES=pro ARMS="$C_S" BASES=x ${cf:+CACHE_FROM=$cf}; } || give_up c_s_pro_L$L
  done
}
pro_diag() {
  { wait_done q8_step0 && job q8_step0pro MODEL=$Q8 STAGE=step0pro; } || give_up q8_step0pro
  { wait_done diag_$Q9 && job diagpro_$Q9 MODEL=$Q9 STAGE=diagpro; } || give_up diagpro_$Q9
  { wait_done diag_$LL && job diagpro_$LL MODEL=$LL STAGE=diagpro CACHE_FROM=$W/results_e44_factorial_das/$LL/judge_tune_step0/judge2/cache; } || give_up diagpro_$LL
  { wait_done diag_$Q8 q8_step0pro && job diagpro_$Q8 MODEL=$Q8 STAGE=diagpro; } || give_up diagpro_$Q8
}
tables_e2() {  # tables_e2 LAYER: strengths 1 / 0.5 / 2 as soon as this layer is judged (both judges on Pro layers)
  local L=$1 S
  if [[ "$E2_PRO_LAYERS" == *" $L "* ]]; then
    wait_done e2_pro_L$L || return 0
    for S in 1 0.5 2; do job e2_tables_L${L}_s$S "${V[@]}" STAGE=tables BRANCH=E LAYERS=$L TAG=X STRENGTH=$S || true; done
  else
    wait_done e2_allenai_L$L || return 0
    for S in 1 0.5 2; do job e2_tables_L${L}_s$S "${V[@]}" STAGE=tables BRANCH=E LAYERS=$L TAG=X STRENGTH=$S JUDGES=allenai || true; done
  fi
}
tables_c() {  # tables_c LAYER: strengths 0.5 and 0.25, both judges
  local L=$1 S
  wait_done c_s_pro_L$L || return 0
  for S in 0.5 0.25; do job c_s_tables_L${L}_s$S MODEL=$Q9 STAGE=tables BRANCH=C LAYERS=$L TAG=S STRENGTH=$S || true; done
}
q8_after() {  # q8_after BRANCH GPU HALF_A HALF_B: Qwen3-8B round-1 judge + frozen screen once both halves are done
  { wait_done "$3" "$4" && job q8_judge_$1 MODEL=$Q8 STAGE=judge BRANCH=$1 GPU=$2 &&
    job q8_screen_$1 MODEL=$Q8 STAGE=screen BRANCH=$1; } || give_up q8_judge_$1 q8_screen_$1
}

gpu0 & gpu1 & gpu2 & gpu3 &
pro_e2 9 12 & pro_e2 11 13 & pro_e2 14 15 & pro_e2 18 & pro_c & pro_diag &
for L in 8 9 10 11 12 13 14 15 16 17 18; do tables_e2 $L & done
for L in 12 16 17; do tables_c $L & done
q8_after E 2 q8_run_E_a q8_run_E_b & q8_after C 0 q8_run_C_a q8_run_C_b &
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"
