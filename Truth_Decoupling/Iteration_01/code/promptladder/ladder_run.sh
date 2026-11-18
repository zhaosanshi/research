#!/usr/bin/env bash
# Prompt ladder launcher (Iteration_01, Claude Opus 5.5, 2026-11-18; DRAFT: not reviewed by GPT, never run on a GPU).
# Plan: compare two ways of re-asking the questions a model leaves blank after the E write -- a C write raised under the
# same prompt (exists), or the same E write under a ladder of firmer system prompts P1 -> P2 -> P3 -> conf (this file).
# For ONE finished E44 run (KEY) and ONE ladder file (LADDER: a flat JSON {"P1": text, "P2": text, "P3": text}, in ladder
# order) it makes greedy answers on tune (192) and test (529) with the run's own E direction written exactly as in its
# existing condition "must-answer prompt conf + E at strength s" (driver file gen/E_L<L>_conf-w-p<s>_<split>.json, s in
# ES, default 2): same direction file, layer, amount = s x gap, content positions incl. decode steps, chat template,
# greedy decoding, 48 new tokens, batches of 8. Only the system prompt differs.
# Engine: the run's own code copy of scripts/e44_eval_extra.py, run unchanged through promptladder/ladder_extra.py (prompt
# texts verbatim from LADDER, output dir OUT, and for test e44_split_test.py's split widening); the engine verifies the
# parent run (identity, frozen answers, direction file) at every call. Before the ladder arms of a split, the control arm
# conf:E<L>.w.p<s> (same engine, same call path) must reproduce the driver's file of that split token for token.
#   usage (host): KEY=llama|qwen9|qwen27|g12|g27 LADDER=<container path> STAGE=<stage> [GPU=k or k,k] [ES="2"]
#                 [OUT_ROOT=<container dir>] [RESUME=1] bash ladder_run.sh
#   STAGE=dryrun  CPU, writes nothing: the in-memory changes for tune and test; the run's identity fields that need no
#                 model; the tokenizer alone (must equal the run's identity) renders every question under conf and each
#                 ladder prompt (the written span must be token-identical to conf's); amount = s x gap = the driver's files
#   STAGE=smoke   GPU: the first batch of tune (8 questions; t8.* files, never exported): control and ladder arms; the
#                 control must equal the driver's first 8 rows; the answers are printed side by side
#   STAGE=gen     GPU: tune, then test; per split: control -> reproduction check -> ladder arms (the engine also makes each
#                 ladder prompt's no-write answers) -> export -> OUT/judge_<split>_ladder_<split>/judge_input.json (a copy,
#                 not judged; the source of the subset judging). A split whose full files exist already needs RESUME=1
#                 (the engine then verifies and skips them; it never overwrites)
#   STAGE=judge   SPLIT=tune|test SUBSET=<container path of {"pair_ids": [...]}> [SUFFIX=cand] [JUDGES="allenai pro"]
#                 [GPU=k, for AllenAI] [CACHE_FROM="<work dir> ..."]: the export restricted to SUBSET, judged in
#                 OUT/judge_<split>_ladder_<split><SUFFIX>. test: scripts/review_checks/ff_test_subset_judge.sh unchanged;
#                 tune: that script refuses tune exports, so ladder_extra.py subset + the same judging calls as there.
#                 DeepSeek Pro needs a complete AllenAI pass of the same work dir first (as there).
# Outputs: OUT = OUT_ROOT/<run dir name> (default OUT_ROOT /workspace/Truth_Decoupling/Iteration_01/promptladder/runs):
#   ladder_manifest.json (the inputs; every later stage must find the same), launch/<time>_<stage>.{json,log,end},
#   gen/<arm>_<split>.json, base_<P>_<split>.json, judge_<split>/ladder_<split>.json, judge_<split>_ladder_<split>*/,
#   frozen_test.json (a symlink to the run's; ff_test_subset_judge.sh looks for it). Nothing is written into the run,
#   the code copies or diag/. Order: dryrun -> smoke -> gen -> (candidate files) -> judge tune -> judge test. Any change
#   of the helper, the ladder texts or ES changes the manifest: use a fresh OUT_ROOT (or remove a smoke-only OUT).
set -euo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
die() { echo "ladder_run: $*" >&2; exit 2; }
host() { case "$1" in "$W"/*) echo "$R/${1#"$W"/}" ;; *) die "path $1 is outside $W" ;; esac; }
: "${KEY:?KEY=llama|qwen9|qwen27|g12|g27}" "${STAGE:?STAGE=dryrun|smoke|gen|judge}" "${LADDER:?LADDER=<container path of the ladder file>}"
case "$STAGE" in dryrun|smoke|gen|judge) ;; *) die "unknown STAGE $STAGE" ;; esac
# the five finished runs (runs, sources and code copies as their own test chains used them; E layer = R3 of each run)
case "$KEY" in
  llama)  MODEL=llama3.1-8b    CODE=$W/code_e45_20261112        SRC=$W/results_qa_das_v1u/llama3.1-8b/full
          RUN=$W/results_e44_factorial_das/llama3.1-8b_side_E2 EL=11 ;;
  qwen9)  MODEL=qwen3.5-9b     CODE=$W/code_ff_explore_20261113b SRC=$W/results_newmodels_ff_20261112/source/qwen3.5-9b
          RUN=$W/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 EL=15 ;;
  qwen27) MODEL=qwen3.8-27b-dm CODE=$W/code_ff_explore_20261113b SRC=$W/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
          RUN=$W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm EL=28 ;;
  g12)    MODEL=gemma3-12b     CODE=$W/Iteration_01/code         SRC=$W/Iteration_01/results/ff/gemma3/12b/source/gemma3-12b
          RUN=$W/Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b EL=22 ;;
  g27)    MODEL=gemma3-27b     CODE=$W/Iteration_01/code         SRC=$W/Iteration_01/results/ff/gemma3/27b/source/gemma3-27b
          RUN=$W/Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b EL=26 ;;
  *) die "unknown KEY $KEY (llama qwen9 qwen27 g12 g27)" ;;
esac
E_ON=E2                                           # every one of the five E directions was trained with the E2 sentence
ES=${ES:-2}
for s in $ES; do case "$s" in ""|*[!0-9.]*|.*|*.|*.*.*) die "strength '$s' is not a number like 2 or 0.5" ;; esac; done
[ -n "$ES" ] || die "ES is empty"
tok() { echo "${1//./-}"; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$HERE" in "$R"/*) ;; *) die "run this file from its place under $R (got $HERE)" ;; esac
SELFH=$HERE/$(basename "${BASH_SOURCE[0]}")
HELPERH=$HERE/ladder_extra.py
HELPER=$W/${HELPERH#"$R"/}
RUNH=$(host "$RUN"); SRCH=$(host "$SRC"); CODEH=$(host "$CODE"); LADH=$(host "$LADDER")
JCODE=$W/Iteration_01/code; JCODEH=$(host "$JCODE")      # judging scripts (byte-identical to the run's copy: checked below)

# ---- inputs: the reviewed engine files, the run's files, the ladder
pin() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || die "$1 is not the version this launcher was written for (sha256 $2)"; }
pin "$CODEH/scripts/e44_eval_extra.py" 4438a282d3e0c12c396e688e65daa2172297fc9a43fd407dd4b3e340869cab18
pin "$CODEH/scripts/e44_factorial_das.py" 9ef1759fe492337bdc8ac627fc846a251251842d77797faeae6deb6beaf61efc
pin "$CODEH/scripts/review_checks/e44_split_test.py" ac5ab93f17ae1f90cdb407598697c681eccb4edec6bd9c6cb10a8608b7859a1b
for f in scripts/review_checks/judge_two_axis_run.py scripts/allenai_tqa_judge.py configs/judges/two_axis_v5.json \
         configs/judges/two_axis_v6.json core/judge.py core/judge_two_axis.py core/judge_two_axis_contract.py \
         core/judge_cache_store.py core/qa_scoring.py core/readout.py; do
  cmp -s "$CODEH/$f" "$JCODEH/$f" || die "$f differs between $CODE and the judging copy $JCODE"
done
[ -f "$JCODEH/scripts/review_checks/ff_test_subset_judge.sh" ] || die "no ff_test_subset_judge.sh in $JCODE"
[ -f "$HELPERH" ] || die "no $HELPERH"
for f in frozen_train.json frozen_tune.json frozen_test.json dirs/E_L$EL.pt; do [ -f "$RUNH/$f" ] || die "$RUN has no $f"; done
for s in $ES; do for sp in tune test; do
  [ -f "$RUNH/gen/E_L${EL}_conf-w-p$(tok "$s")_$sp.json" ] || die "$RUN has no driver file gen/E_L${EL}_conf-w-p$(tok "$s")_$sp.json (strength $s)"
done; done
[ -f "$SRCH/bank.json" ] && [ -f "$SRCH/questions.json" ] || die "no source run $SRC"
[ -f "$LADH" ] || die "no ladder file $LADDER"
PN=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d, dict) and d; print(" ".join(d))' "$LADH") \
  || die "cannot read the ladder file $LADDER"
for p in $PN; do case "$p" in *[!A-Za-z0-9_]*) die "ladder prompt name '$p': letters, digits and _ only" ;; conf) die "conf is the last rung already" ;; esac; done
PYM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_config"].get("python") or "python3")' "$SRCH/bank.json")
OUT_ROOT=${OUT_ROOT:-$W/Iteration_01/promptladder/runs}
case "$OUT_ROOT" in "$W"/*) ;; *) die "OUT_ROOT must be an absolute path under $W" ;; esac
case "$OUT_ROOT/" in "$RUN"/*|"$CODE"/*|"$JCODE"/*|"$W"/Iteration_01/diag/*) die "OUT_ROOT $OUT_ROOT lies in the run, a code copy or diag/" ;; esac
case "$RUN/" in "$OUT_ROOT"/*) die "the run lies inside OUT_ROOT $OUT_ROOT" ;; esac
OUT=$OUT_ROOT/${RUN##*/}; OUTH=$(host "$OUT")

# ---- container calls
ENVS=(-e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models -e TD_DATA_DIR=$W/code/data/processed
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1)
DX=(docker exec -u 1000:1001 "${ENVS[@]}" -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e PYTHONPATH=$CODE -w $CODE truth_dec)
DC=(docker exec -u 1000:1001 "${ENVS[@]}" -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=$CODE -w $CODE truth_dec)       # CPU only
DJ=(docker exec -u 1000:1001 "${ENVS[@]}" -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e PYTHONPATH=$JCODE
    -e DEEPSEEK_API_KEY_FILE=$W/logs/.deepseek_key -w $JCODE truth_dec)
lx()  { "${DX[@]}" "$PYM" -B "$HELPER" extra --code "$CODE" --out "$OUT" "$@"; }   # the engine, model runtime of the run
lc()  { "${DC[@]}" "$PYM" -B "$HELPER" "$@"; }                                     # helper checks, CPU
pyh() { "${DC[@]}" python3 -B "$HELPER" "$@"; }                                     # JSON-only helper steps, CPU
pyj() { "${DJ[@]}" python3 "$@"; }                                                  # judging, as ff_test_subset_judge.sh
step() { echo "=== $(TZ=Asia/Shanghai date '+%F %T') [ladder $KEY $STAGE gpu${GPU:-none} E$EL s=$ES] $* ==="; }
COMMON=(--source-run "$SRC" --run-dir "$RUN" --e-on "$E_ON" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24
        --seed 42 --prompts-extra "$LADDER")
CTRL=""; LAD=""
for s in $ES; do CTRL="$CTRL conf:E$EL.w.p$s"; for p in $PN; do LAD="$LAD $p:E$EL.w.p$s"; done; done
gpu_free() {  # every card of GPU must hold no compute process (a busy card is reported, never worked around)
  local g uuid apps
  [ -n "${GPU:-}" ] || die "STAGE=$STAGE needs GPU"
  case "$GPU" in *[!0-9,]*|,*|*,|*,,*) die "GPU must be card numbers such as 1 or 2,3" ;; esac
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || die "nvidia-smi failed"
  for g in ${GPU//,/ }; do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g") || die "cannot read GPU $g"
    case "$apps" in *"$uuid"*) die "GPU $g is busy (a compute process holds it): ask before using it" ;; esac
  done
}
manifest_json() {  # the inputs that define OUT, as canonical JSON
  python3 - "$KEY" "$MODEL" "$RUN" "$SRC" "$CODE" "$EL" "$E_ON" "$ES" "$LADDER" "$LADH" "$HELPERH" "$CODEH" "$SRCH" <<'PY'
import hashlib, json, sys
key, model, run, src, code, el, e_on, es, lad, ladh, helper, codeh, srch = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
prompts = json.load(open(ladh, encoding="utf-8"))
m = {"protocol": "promptladder_v1", "key": key, "model": model, "run_dir": run, "source_run": src, "code_copy": code,
     "e_layer": int(el), "e_on": e_on, "strengths": es.split(), "control_prompt": "conf", "ladder_order": list(prompts),
     "ladder_file": lad, "ladder_sha256": sha(ladh), "ladder_prompts": prompts, "helper_sha256": sha(helper),
     "engine_sha256": {n: sha(codeh + "/" + n) for n in ("scripts/e44_eval_extra.py", "scripts/e44_factorial_das.py",
                                                       "scripts/review_checks/e44_split_test.py")},
     "source_sha256": {n: sha(srch + "/" + n) for n in ("bank.json", "questions.json")},
     "generation": {"batch_size": 8, "max_new_tokens": 48, "max_target_tokens": 24, "seed": 42, "output_projection": "float32"}}
print(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True))
PY
}
prepare_out() {  # create OUT with its manifest, or require that an existing OUT was made from the same inputs
  local m; m=$(manifest_json)
  if [ -e "$OUTH/ladder_manifest.json" ]; then
    [ "$(cat "$OUTH/ladder_manifest.json")" = "$m" ] || die "$OUT was made from other inputs (ladder_manifest.json differs): use another OUT_ROOT"
  else
    [ "$STAGE" != judge ] || die "no $OUT/ladder_manifest.json: run STAGE=gen first"
    if [ -e "$OUTH" ] && [ -n "$(ls -A "$OUTH")" ]; then die "$OUT exists without a manifest: not touched"; fi
    mkdir -p "$OUTH/launch" "$OUTH/gen"
    printf '%s\n' "$m" > "$OUTH/ladder_manifest.json"
  fi
}
record() {  # record LABEL PLAN: one launch at a time per OUT (lock dir); OUT/launch/<time>_<LABEL>.json (inputs, hashes,
  local ts                    # parameters), .log (this output), .end (exit status)
  mkdir "$OUTH/.lock" 2>/dev/null || die "another launch holds $OUT/.lock (remove it only if none runs)"
  trap 'rmdir "$OUTH/.lock" 2>/dev/null' EXIT
  ts=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)
  REC=$OUTH/launch/${ts}_$1
  [ ! -e "$REC.json" ] || die "launch record $REC.json exists"
  python3 - "$REC.json" "$OUTH/ladder_manifest.json" "$SELFH" "$STAGE" "${GPU:-}" "${SPLIT:-}" "${SUBSET:-}" \
      "$( [ -n "${SUBSET:-}" ] && sha256sum "$(host "$SUBSET")" | cut -d' ' -f1 )" "${SUFFIX:-}" "${JUDGES:-}" \
      "${CACHE_FROM:-}" "${RESUME:-0}" "$PYM" "$2" <<'PY'
import hashlib, json, socket, sys, time
out, man, me, stage, gpu, split, subset, subset_sha, suffix, judges, cache_from, resume, pym, plan = sys.argv[1:]
rec = {"time_beijing": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 3600)), "host": socket.gethostname(),
       "stage": stage, "gpu": gpu, "resume": resume, "model_python": pym, "plan": plan.split(),
       "launcher": me, "launcher_sha256": hashlib.sha256(open(me, "rb").read()).hexdigest(),
       "manifest": json.load(open(man, encoding="utf-8"))}
if stage == "judge":
    rec["judge"] = {"split": split, "subset": subset, "subset_sha256": subset_sha, "suffix": suffix, "judges": judges.split(),
                    "cache_from": cache_from.split()}
with open(out, "x", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
PY
  exec > >(tee -a "$REC.log") 2>&1
  trap 'rc=$?; echo "{\"exit\": $rc, \"end_beijing\": \"$(TZ=Asia/Shanghai date "+%F %T")\"}" > "$REC.end"; rmdir "$OUTH/.lock" 2>/dev/null' EXIT
  step "launch record $REC.json"
}
partial_name() { [[ "$1" =~ ^t[0-9]*\. ]]; }        # the engine's PARTIAL_RE
has_full() {  # has_full SPLIT: OUT/gen holds a full (not t<N>.*) file of SPLIT
  local f
  for f in "$OUTH"/gen/*_"$1".json; do [ -e "$f" ] || continue; partial_name "${f##*/}" || return 0; done
  return 1
}
counts() {  # counts EXPORT: answered / blank (rule or empty) per condition, judge-free
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
n = len(d["evaluations"])
for c in d["conditions"]:
    blank = sum(1 for e in d["evaluations"] if e["rule_abstain"][c] or not e["outputs"][c].strip())
    print("  %-22s answered %3d | blank (rule or empty) %3d | of %d" % (c, n - blank, blank, n))
PY
}
verdicts_complete() {  # as ff_test_subset_judge.sh: empty answers count as abstentions, Pro-declined cells are listed,
  local out bad empty declined       # any other missing verdict stops
  out=$(pyj -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
empty = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'empty_output_no_judge_vote' and r.get('empty_output') is True]
decl = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'pro_declined']
print(sum(r['status'] != 'scored' for r in rows) - len(empty) - len(decl), len(empty), ','.join(r['cell_id'] for r in decl) or '-')" "$1/judge2/single_$2.jsonl") \
    || die "cannot read the $2 verdicts of $1"
  read -r bad empty declined <<< "$out"
  [ "$empty" = 0 ] || echo "$2: $empty empty answer(s) in $1 count as abstentions (never judged; two-axis standard of 11-08)"
  [ "$declined" = - ] || echo "$2 DECLINED these cells in $1 (no table until judge2/excluded_questions.json names their questions with a reason): $declined"
  [ "$bad" = 0 ] || die "$2 left $bad cells without a verdict in $1: re-judge them before any table"
}
verdict_table() {  # verdict_table WORK JUDGE: right / wrong / hedged-or-uninformative / blank per condition
  python3 - "$(host "$1")/judge2/single_$2.jsonl" "$2" <<'PY'
import collections, json, sys
per = collections.defaultdict(collections.Counter)
for l in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(l)
    if r["status"] != "scored":
        s = "X" if (r.get("basis") == "empty_output_no_judge_vote" and r.get("empty_output") is True) else "?"
    elif sys.argv[2] == "allenai":
        s = "X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]
    else:
        s = {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
    per[r["condition"]][s] += 1
for c, k in per.items():
    print("  %-22s %s: right %3d wrong %3d hedge %3d blank %3d no verdict %d" % (c, sys.argv[2], k["R"], k["W"], k["H"], k["X"], k["?"]))
PY
}

echo "ladder: KEY=$KEY model=$MODEL run=$RUN source=$SRC code=$CODE E layer $EL (e_on $E_ON) strengths [$ES]"
echo "ladder: prompts [$PN] from $LADDER; control conf; out $OUT; model python $PYM"
case "$STAGE" in
dryrun)
  step "check: the in-memory changes for tune and for test (nothing runs)"
  lc check --code "$CODE" --out "$OUT" --split tune
  lc check --code "$CODE" --out "$OUT" --split test
  step "dryrun: identity fields, tokenizer, rendering, written span, write amount (CPU, tokenizer only, nothing written)"
  lc dryrun --code "$CODE" --source-run "$SRC" --run-dir "$RUN" --prompts-extra "$LADDER" --layer "$EL" --strengths $ES ;;
smoke)
  gpu_free; prepare_out; record smoke "$CTRL $LAD"
  step "smoke: the first batch of tune (8 questions, t8.* files): $CTRL $LAD"
  lx generate "${COMMON[@]}" --split tune --first-batches 1 --arms $CTRL $LAD
  F=""
  for s in $ES; do
    pyh reproduce "$OUT/gen/t8.conf--E$EL-w-p$(tok "$s")_tune.json" "$RUN/gen/E_L${EL}_conf-w-p$(tok "$s")_tune.json"
    F="$F $OUT/gen/t8.conf--E$EL-w-p$(tok "$s")_tune.json"
    for p in $PN; do F="$F $OUT/gen/t8.$p--E$EL-w-p$(tok "$s")_tune.json"; done
  done
  pyh show --n 8 $F ;;
gen)
  gpu_free; prepare_out; record gen "$CTRL |$LAD"
  for SPLIT in tune test; do
    if [ "${RESUME:-0}" != 1 ] && has_full "$SPLIT"; then
      die "$OUT/gen already holds full $SPLIT files: RESUME=1 verifies and continues them (nothing is overwritten)"
    fi
    step "$SPLIT: control$CTRL (the engine verifies the run and reproduces conf's frozen answers first)"
    lx generate "${COMMON[@]}" --split "$SPLIT" --arms $CTRL
    for s in $ES; do
      pyh reproduce "$OUT/gen/conf--E$EL-w-p$(tok "$s")_$SPLIT.json" "$RUN/gen/E_L${EL}_conf-w-p$(tok "$s")_$SPLIT.json"
    done
    step "$SPLIT: ladder arms$LAD"
    lx generate "${COMMON[@]}" --split "$SPLIT" --arms $LAD
    GEN=""
    for s in $ES; do GEN="$GEN conf--E$EL-w-p$(tok "$s")_$SPLIT.json"; for p in $PN; do GEN="$GEN $p--E$EL-w-p$(tok "$s")_$SPLIT.json"; done; done
    step "$SPLIT: export$GEN"
    lx export "${COMMON[@]}" --split "$SPLIT" --gen $GEN --name "ladder_$SPLIT"
    WF=$OUTH/judge_${SPLIT}_ladder_$SPLIT
    mkdir -p "$WF/judge2/cache"
    [ -f "$WF/judge_input.json" ] || cp "$OUTH/judge_$SPLIT/ladder_$SPLIT.json" "$WF/judge_input.json"
    cmp "$OUTH/judge_$SPLIT/ladder_$SPLIT.json" "$WF/judge_input.json"
    step "$SPLIT: answered / blank per condition (judge-free)"
    counts "$WF/judge_input.json"
  done
  if [ ! -e "$OUTH/frozen_test.json" ]; then
    ln -s "$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$RUNH/frozen_test.json" "$OUTH")" "$OUTH/frozen_test.json"
  fi
  cmp "$OUTH/frozen_test.json" "$RUNH/frozen_test.json" ;;
judge)
  : "${SPLIT:?SPLIT=tune|test}" "${SUBSET:?SUBSET=<container path of a JSON with pair_ids>}"
  case "$SPLIT" in tune|test) ;; *) die "SPLIT must be tune or test" ;; esac
  SUFFIX=${SUFFIX:-cand}; JUDGES=${JUDGES:-allenai pro}
  case "$SUFFIX" in ""|*[!A-Za-z0-9]*) die "SUFFIX must be alphanumeric" ;; esac
  [ -n "$JUDGES" ] || die "JUDGES is empty"
  for j in $JUDGES; do case "$j" in allenai|pro) ;; *) die "JUDGES: unknown judge '$j' (allenai / pro)" ;; esac; done
  has() { case " $JUDGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
  [ -f "$(host "$SUBSET")" ] || die "no subset file $SUBSET"
  for c in ${CACHE_FROM:-}; do [ -d "$(host "$c")/judge2/cache" ] || die "reply cache $c/judge2/cache does not exist (CACHE_FROM)"; done
  NAME=ladder_$SPLIT; SRCW=$OUT/judge_${SPLIT}_$NAME; WORK=$SRCW$SUFFIX
  [ -f "$(host "$SRCW")/judge_input.json" ] || die "no export $SRCW/judge_input.json: run STAGE=gen first"
  if has allenai; then gpu_free; fi
  prepare_out; record "judge_$SPLIT" "$WORK"
  if [ "$SPLIT" = test ]; then
    step "test: scripts/review_checks/ff_test_subset_judge.sh (unchanged) on $SRCW -> $WORK"
    env MODEL="$MODEL" CODE_PATH="$JCODE" RUN="$OUT" NAMES="$NAME" SUFFIX="$SUFFIX" SUBSET="$SUBSET" JUDGES="$JUDGES" \
      GPU="${GPU:-}" CACHE_FROM="${CACHE_FROM:-}" bash "$JCODEH/scripts/review_checks/ff_test_subset_judge.sh"
  else
    WH=$(host "$WORK")
    step "tune: the export restricted to the questions of ${SUBSET##*/}"
    mkdir -p "$WH/judge2/cache"
    pyh subset "$SRCW/judge_input.json" "$SUBSET" "$WORK" --split tune
    if has allenai; then
      step "tune: cells + allenai + single"
      pyj scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
      "${DJ[@]}" python3 scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
        --truth-model "$W/models/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$W/models/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
      pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WH/single_allenai_stdout.md"
    else
      [ -f "$WH/judge2/single_allenai.jsonl" ] || die "no AllenAI pass in $WORK yet: run JUDGES=allenai first"
    fi
    verdicts_complete "$WORK" allenai
    if has pro; then
      for c in ${CACHE_FROM:-}; do                                   # identical requests are not paid twice
        for f in "$(host "$c")"/judge2/cache/*; do
          [ -f "$f" ] || continue
          [ -e "$WH/judge2/cache/$(basename "$f")" ] || cp "$f" "$WH/judge2/cache/"
        done
      done
      step "tune: pro"
      pyj scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
      pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$WH/single_pro_stdout.md"
      verdicts_complete "$WORK" pro
    else
      step "tune: pro skipped (JUDGES=$JUDGES)"
    fi
  fi
  for j in $JUDGES; do step "verdicts in $WORK ($j)"; verdict_table "$WORK" "$j"; done ;;
esac
step "$STAGE done"
