#!/usr/bin/env bash
# Prompt ladder launcher (Iteration_01, Claude Opus 5.5, 2026-11-18; revised the same day after GPT's review of commit
# 02f4cc9, discussion/prompt_ladder/2026-11-18_gpt_review.md section 3: B1 B2 S1 S2 S3; still never run on a GPU).
# Plan: compare two ways of re-asking the questions a model leaves blank after the E write -- a C write raised under the
# same prompt (exists), or the same E write under a ladder of firmer system prompts P1 -> P2 -> P3 -> conf (this file).
# For ONE finished E44 run (KEY) and ONE ladder file (LADDER: a flat JSON {"P1": text, "P2": text, "P3": text}, in ladder
# order) it makes greedy answers on tune (192) and test (529) with the run's own E direction written exactly as in its
# existing condition "must-answer prompt conf + E at strength s" (driver file gen/E_L<L>_conf-w-p<s>_<split>.json, s in
# ES, default 2): same direction file, layer, amount = s x gap, content positions incl. decode steps, chat template,
# greedy decoding, 48 new tokens, batches of 8. Only the system prompt differs.
# Engine: the run's own code copy of scripts/e44_eval_extra.py, run unchanged through promptladder/ladder_extra.py (prompt
# texts verbatim from LADDER, output dir OUT, and for test e44_split_test.py's split widening); the engine verifies the
# parent run (identity, frozen answers, direction file) at every call. The control arm conf:E<L>.w.p<s> (same engine,
# same call path) must reproduce the driver's file token for token before the ladder arms of that split are made.
#   usage (host): KEY=llama|qwen9|qwen27|g12|g27 LADDER=<container path> STAGE=<stage> [GPU=k or k,k] [ES="2"]
#                 [CAND_S=<strength whose blanks are the candidates; default ES when ES is one value>]
#                 [OUT_ROOT=<container dir under Iteration_01/promptladder>] [RESUME=1] bash ladder_run.sh
#   STAGE=dryrun  CPU, writes nothing: the in-memory changes; the run's identity fields that need no model; the tokenizer
#                 alone renders every question under conf and each ladder prompt (written span = conf's); amount =
#                 s x gap = the driver's files; the candidate sets of both splits rebuilt from the run (B1)
#   STAGE=smoke   GPU: the first batch of tune (8 questions; t8.* files, never exported): the control alone first (it must
#                 equal the driver's first 8 rows), then the ladder arms, side by side. Creates OUT; its manifest holds the
#                 inputs AND the device mapping (GPU list); a pass is recorded in OUT/smoke_pass.json (S1)
#   STAGE=gen     GPU: refused unless OUT/smoke_pass.json passed for this very manifest (same model, prompts, helper and
#                 engine hashes, GPU list; S1). Tune, then test; per split: control -> reproduction check -> ladder arms
#                 (the engine also makes each ladder prompt's no-write answers) -> export -> OUT/judge_<split>_ladder_
#                 <split>/judge_input.json (a copy, not judged). A split whose full files exist already needs RESUME=1.
#                 A control that does not reproduce is moved to OUT/failed/ with a failure record, and OUT is closed:
#                 every later smoke / gen there is refused (a resume can never compare against or reuse it; S1)
#   STAGE=cand    SPLIT=tune|test, CPU: writes OUT/subsets/<split>_E<L>_x-w-p<s>_blank.json, the candidate set rebuilt
#                 from the run's own E+C grid export (B1); give that file as SUBSET
#   STAGE=judge   SPLIT=tune|test SUBSET=<container path> [SUFFIX=cand] [JUDGES=allenai (default) | pro | "allenai pro"]
#                 [GPU=k, for AllenAI] [CACHE_FROM=<ONE work dir, Pro reply cache seed>]. First, read-only: SUBSET must
#                 equal the candidate set rebuilt from the run (B1); with pro: DeepSeek off-peak window open with >= 2 h
#                 left (B2). AllenAI: test = scripts/review_checks/ff_test_subset_judge.sh unchanged (JUDGES=allenai), tune
#                 = ladder_extra.py subset + the same calls. Pro (both splits): cache seed merged by judge2_cache_merge.py,
#                 judge_two_axis_run.py llm under a timeout that ends 5 min before the next peak (stopped -> run the same
#                 command in the next window), then single. Every pass: the full expected cell set checked (S2)
# Outputs: OUT = OUT_ROOT/<run dir name>, OUT_ROOT canonical (realpath) and inside Iteration_01/promptladder/ (S3):
#   ladder_manifest.json, smoke_pass.json, failed/, subsets/, launch/<time>_<stage>.{json,log,end}, gen/<arm>_<split>.json,
#   base_<P>_<split>.json, judge_<split>/ladder_<split>.json, judge_<split>_ladder_<split>*/, frozen_test.json (a symlink to
#   the run's; ff_test_subset_judge.sh looks for it). Nothing is written into a run, a code copy or diag/.
#   Order: dryrun -> smoke -> gen -> cand -> judge (allenai, any time) -> judge (pro, off-peak). Any change of the helper,
#   the ladder texts, ES or the GPU list changes the manifest: use a fresh OUT_ROOT.
set -euo pipefail
export TZ=Asia/Shanghai                           # Beijing time: window() below asserts it
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
die() { echo "ladder_run: $*" >&2; exit 2; }
host() { case "$1" in "$W"/*) echo "$R/${1#"$W"/}" ;; *) die "path $1 is outside $W" ;; esac; }
cont() { case "$1" in "$R"/*) echo "$W/${1#"$R"/}" ;; *) die "path $1 is outside $R" ;; esac; }
: "${KEY:?KEY=llama|qwen9|qwen27|g12|g27}" "${STAGE:?STAGE=dryrun|smoke|gen|cand|judge}" "${LADDER:?LADDER=<container path of the ladder file>}"
case "$STAGE" in dryrun|smoke|gen|cand|judge) ;; *) die "unknown STAGE $STAGE" ;; esac
# the five finished runs (runs, sources and code copies as their own test chains used them; E layer = R3 of each run;
# GT / GX = the run's own E+C grid work dirs in eval_extra/ for tune / test, as Iteration_01/diag/policy_select2.py, except
# that test uses the full 529-question grid of the Gemma runs, not its Dr-100 part)
case "$KEY" in
  llama)  MODEL=llama3.1-8b    CODE=$W/code_e45_20261112        SRC=$W/results_qa_das_v1u/llama3.1-8b/full
          RUN=$W/results_e44_factorial_das/llama3.1-8b_side_E2 EL=11 GT=judge_tune_ecll19_tune GX=judge_test_tll19_test ;;
  qwen9)  MODEL=qwen3.5-9b     CODE=$W/code_ff_explore_20261113b SRC=$W/results_newmodels_ff_20261112/source/qwen3.5-9b
          RUN=$W/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 EL=15 GT=judge_tune_ec9b_tune GX=judge_test_t9b16_test ;;
  qwen27) MODEL=qwen3.8-27b-dm CODE=$W/code_ff_explore_20261113b SRC=$W/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
          RUN=$W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm EL=28 GT=judge_tune_ec2733_tune GX=judge_test_t2733_test ;;
  g12)    MODEL=gemma3-12b     CODE=$W/Iteration_01/code         SRC=$W/Iteration_01/results/ff/gemma3/12b/source/gemma3-12b
          RUN=$W/Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b EL=22 GT=judge_tune_ec25_tune GX=judge_test_t25_test ;;
  g27)    MODEL=gemma3-27b     CODE=$W/Iteration_01/code         SRC=$W/Iteration_01/results/ff/gemma3/27b/source/gemma3-27b
          RUN=$W/Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b EL=26 GT=judge_tune_ec31_tune GX=judge_test_t31_test ;;
  *) die "unknown KEY $KEY (llama qwen9 qwen27 g12 g27)" ;;
esac
ALL_RUNS=($W/results_e44_factorial_das/llama3.1-8b_side_E2 $W/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2
          $W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm $W/Iteration_01/results/ff/gemma3/12b $W/Iteration_01/results/ff/gemma3/27b)
ALL_CODES=($W/code_e45_20261112 $W/code_ff_explore_20261113b $W/Iteration_01/code $W/Iteration_01/diag)
E_ON=E2                                           # every one of the five E directions was trained with the E2 sentence
ES=${ES:-2}
num_ok() { case "$1" in ""|*[!0-9.]*|.*|*.|*.*.*) return 1 ;; esac; }
for s in $ES; do num_ok "$s" || die "strength '$s' is not a number like 2 or 0.5"; done
[ -n "$ES" ] || die "ES is empty"
set -f; set -- $ES; set +f
CAND_S=${CAND_S:-$([ $# = 1 ] && echo "$1" || true)}
case " $ES " in *" ${CAND_S:-none} "*) ;; *) [ "$STAGE" != cand ] && [ "$STAGE" != judge ] || die "CAND_S must be one of ES [$ES] (got '${CAND_S:-}')" ;; esac
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
for f in scripts/review_checks/judge_two_axis_run.py scripts/allenai_tqa_judge.py scripts/review_checks/judge2_cache_merge.py \
         configs/judges/two_axis_v5.json configs/judges/two_axis_v6.json core/judge.py core/judge_two_axis.py \
         core/judge_two_axis_contract.py core/judge_cache_store.py core/qa_scoring.py core/readout.py; do
  cmp -s "$CODEH/$f" "$JCODEH/$f" || die "$f differs between $CODE and the judging copy $JCODE"
done
[ -f "$JCODEH/scripts/review_checks/ff_test_subset_judge.sh" ] || die "no ff_test_subset_judge.sh in $JCODE"
[ -f "$HELPERH" ] || die "no $HELPERH"
for f in frozen_train.json frozen_tune.json frozen_test.json dirs/E_L$EL.pt; do [ -f "$RUNH/$f" ] || die "$RUN has no $f"; done
for s in $ES; do for sp in tune test; do
  [ -f "$RUNH/gen/E_L${EL}_conf-w-p$(tok "$s")_$sp.json" ] || die "$RUN has no driver file gen/E_L${EL}_conf-w-p$(tok "$s")_$sp.json (strength $s)"
done; done
for g in "$GT" "$GX"; do [ -f "$RUNH/eval_extra/$g/judge_input.json" ] || die "$RUN has no E+C grid eval_extra/$g"; done
[ -f "$SRCH/bank.json" ] && [ -f "$SRCH/questions.json" ] || die "no source run $SRC"
[ -f "$LADH" ] || die "no ladder file $LADDER"
PN=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d, dict) and d; print(" ".join(d))' "$LADH") \
  || die "cannot read the ladder file $LADDER"
for p in $PN; do case "$p" in *[!A-Za-z0-9_]*) die "ladder prompt name '$p': letters, digits and _ only" ;; conf) die "conf is the last rung already" ;; esac; done
PYM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_config"].get("python") or "python3")' "$SRCH/bank.json")
GPUN=${GPU:-}
if [ -n "$GPUN" ]; then case "$GPUN" in *[!0-9,]*|,*|*,|*,,*) die "GPU must be card numbers such as 1 or 2,3" ;; esac; fi

# ---- S3: OUT_ROOT canonical (symlinks and .. resolved) and checked before anything is written
OUT_ROOT=${OUT_ROOT:-$W/Iteration_01/promptladder/runs}
case "$OUT_ROOT" in "$W"/*) ;; *) die "OUT_ROOT must be an absolute path under $W" ;; esac
OUT_ROOT_H=$(realpath -m -- "$(host "$OUT_ROOT")") || die "cannot resolve OUT_ROOT $OUT_ROOT"
PLH=$(realpath -m -- "$R/Iteration_01/promptladder")
case "$OUT_ROOT_H/" in "$PLH"/?*) ;; *) die "OUT_ROOT $OUT_ROOT resolves to $OUT_ROOT_H, not inside $PLH" ;; esac
OUTH=$(realpath -m -- "$OUT_ROOT_H/${RUN##*/}")
case "$OUTH" in "$OUT_ROOT_H"/*) ;; *) die "the output dir resolves to $OUTH, outside OUT_ROOT" ;; esac
for p in "${ALL_RUNS[@]}" "${ALL_CODES[@]}" "$SRC" "$RUN"; do
  ph=$(realpath -m -- "$(host "$p")")
  case "$OUTH/" in "$ph"/*) die "the output dir $OUTH lies inside $ph" ;; esac
  case "$ph/" in "$OUTH"/*) die "$ph lies inside the output dir $OUTH" ;; esac
done
OUT=$(cont "$OUTH")

# ---- container calls
ENVS=(-e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models -e TD_DATA_DIR=$W/code/data/processed
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1)
DX=(docker exec -u 1000:1001 "${ENVS[@]}" -e CUDA_VISIBLE_DEVICES="$GPUN" -e PYTHONPATH=$CODE -w $CODE truth_dec)
DC=(docker exec -u 1000:1001 "${ENVS[@]}" -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=$CODE -w $CODE truth_dec)       # CPU only
DJ=(docker exec -u 1000:1001 "${ENVS[@]}" -e CUDA_VISIBLE_DEVICES="$GPUN" -e PYTHONPATH=$JCODE
    -e DEEPSEEK_API_KEY_FILE=$W/logs/.deepseek_key -w $JCODE truth_dec)
lx()  { "${DX[@]}" "$PYM" -B "$HELPER" extra --code "$CODE" --out "$OUT" "$@"; }   # the engine, model runtime of the run
lc()  { "${DC[@]}" "$PYM" -B "$HELPER" "$@"; }                                     # helper checks, CPU
pyh() { "${DC[@]}" python3 -B "$HELPER" "$@"; }                                     # JSON-only helper steps, CPU
pyj() { "${DJ[@]}" python3 "$@"; }                                                  # judging, as ff_test_subset_judge.sh
step() { echo "=== $(date '+%F %T') [ladder $KEY $STAGE gpu${GPUN:-none} E$EL s=$ES] $* ==="; }
COMMON=(--source-run "$SRC" --run-dir "$RUN" --e-on "$E_ON" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24
        --seed 42 --prompts-extra "$LADDER")
CTRL=""; LAD=""
for s in $ES; do CTRL="$CTRL conf:E$EL.w.p$s"; for p in $PN; do LAD="$LAD $p:E$EL.w.p$s"; done; done
gpu_free() {  # every card of GPU must hold no compute process (a busy card is reported, never worked around)
  local g uuid apps
  [ -n "$GPUN" ] || die "STAGE=$STAGE needs GPU"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || die "nvidia-smi failed"
  for g in ${GPUN//,/ }; do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g") || die "cannot read GPU $g"
    case "$apps" in *"$uuid"*) die "GPU $g is busy (a compute process holds it): ask before using it" ;; esac
  done
}
manifest_json() {  # manifest_json GEN_GPUS: the inputs that define OUT, incl. the generation device mapping, canonical JSON
  python3 - "$KEY" "$MODEL" "$RUN" "$SRC" "$CODE" "$EL" "$E_ON" "$ES" "$LADDER" "$LADH" "$HELPERH" "$CODEH" "$SRCH" "$1" <<'PY'
import hashlib, json, sys
key, model, run, src, code, el, e_on, es, lad, ladh, helper, codeh, srch, gpus = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
prompts = json.load(open(ladh, encoding="utf-8"))
bank = json.load(open(srch + "/bank.json", encoding="utf-8"))
m = {"protocol": "promptladder_v1", "key": key, "model": model, "run_dir": run, "source_run": src, "code_copy": code,
     "e_layer": int(el), "e_on": e_on, "strengths": es.split(), "control_prompt": "conf", "ladder_order": list(prompts),
     "ladder_file": lad, "ladder_sha256": sha(ladh), "ladder_prompts": prompts, "helper_sha256": sha(helper),
     "engine_sha256": {n: sha(codeh + "/" + n) for n in ("scripts/e44_eval_extra.py", "scripts/e44_factorial_das.py",
                                                       "scripts/review_checks/e44_split_test.py")},
     "source_sha256": {n: sha(srch + "/" + n) for n in ("bank.json", "questions.json")},
     "generation": {"batch_size": 8, "max_new_tokens": 48, "max_target_tokens": 24, "seed": 42, "output_projection": "float32"},
     "device_mapping": {"generation_gpus": gpus, "n_gpus": len(gpus.split(",")),
                        "model_device_map": bank["model_config"].get("device_map")}}
print(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True))
PY
}
manifest_gpus() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["device_mapping"]["generation_gpus"])' "$OUTH/ladder_manifest.json"; }
prepare_out() {  # prepare_out new|existing: create OUT with its manifest (smoke), or require an existing OUT made from the
  local m gpus                                   # same inputs and device mapping (gen, cand, judge)
  if [ "$1" = new ]; then gpus=$GPUN; else
    [ -e "$OUTH/ladder_manifest.json" ] || die "no $OUT/ladder_manifest.json: run STAGE=smoke first"
    gpus=$(manifest_gpus) || die "unreadable $OUT/ladder_manifest.json"
    [ "$STAGE" != gen ] || [ "$gpus" = "$GPUN" ] || die "$OUT was generated with GPU=$gpus, not GPU=$GPUN: a different device mapping needs a fresh OUT_ROOT"
  fi
  m=$(manifest_json "$gpus")
  if [ -e "$OUTH/ladder_manifest.json" ]; then
    [ "$(cat "$OUTH/ladder_manifest.json")" = "$m" ] || die "$OUT was made from other inputs or another device mapping (ladder_manifest.json differs): use another OUT_ROOT"
  else
    if [ -e "$OUTH" ] && [ -n "$(ls -A "$OUTH")" ]; then die "$OUT exists without a manifest: not touched"; fi
    mkdir -p "$OUTH/launch" "$OUTH/gen"
    printf '%s\n' "$m" > "$OUTH/ladder_manifest.json"
  fi
}
manifest_sha() { sha256sum "$OUTH/ladder_manifest.json" | cut -d' ' -f1; }
closed_guard() {  # S1: a control failure closes OUT for good (its record and the moved control file stay as evidence)
  local f
  for f in "$OUTH"/failed/*.json; do
    [ -e "$f" ] && die "a control did not reproduce in $OUT earlier (${f##*/}): this OUT is closed; use a fresh OUT_ROOT"
  done
  return 0
}
smoke_passed() {  # S1: gen only after a passing smoke for this very manifest
  [ -f "$OUTH/smoke_pass.json" ] || die "no passing smoke in $OUT: run STAGE=smoke with the same GPU=$GPUN first"
  python3 - "$OUTH/smoke_pass.json" "$(manifest_sha)" <<'PY' || die "$OUT/smoke_pass.json belongs to another manifest: run STAGE=smoke again in a fresh OUT_ROOT"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
sys.exit(0 if d.get("passed") is True and d.get("manifest_sha256") == sys.argv[2] else 1)
PY
}
record() {  # record LABEL PLAN [BINDING] [PRO_WINDOW]: one launch at a time per OUT (lock dir); OUT/launch/<time>_<LABEL>
  local ts                    # .json (inputs, hashes, parameters, candidate binding), .log (this output), .end (exit status)
  mkdir "$OUTH/.lock" 2>/dev/null || die "another launch holds $OUT/.lock (remove it only if none runs)"
  trap 'rmdir "$OUTH/.lock" 2>/dev/null' EXIT
  ts=$(date +%Y%m%dT%H%M%S)
  REC=$OUTH/launch/${ts}_$1
  [ ! -e "$REC.json" ] || die "launch record $REC.json exists"
  python3 - "$REC.json" "$OUTH/ladder_manifest.json" "$SELFH" "$STAGE" "$GPUN" "${SPLIT:-}" "${SUBSET:-}" \
      "$( [ -n "${SUBSET:-}" ] && sha256sum "$(host "$SUBSET")" | cut -d' ' -f1 )" "${SUFFIX:-}" "${JUDGES:-}" \
      "${CACHE_FROM:-}" "${RESUME:-0}" "$PYM" "$2" "${3:-}" "${4:-}" <<'PY'
import hashlib, json, socket, sys, time
out, man, me, stage, gpu, split, subset, subset_sha, suffix, judges, cache_from, resume, pym, plan, binding, prow = sys.argv[1:]
rec = {"time_beijing": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 3600)), "host": socket.gethostname(),
       "stage": stage, "gpu": gpu, "resume": resume, "model_python": pym, "plan": plan.split(),
       "launcher": me, "launcher_sha256": hashlib.sha256(open(me, "rb").read()).hexdigest(),
       "manifest_sha256": hashlib.sha256(open(man, "rb").read()).hexdigest(), "manifest": json.load(open(man, encoding="utf-8"))}
if stage == "judge":
    rec["judge"] = {"split": split, "subset": subset, "subset_sha256": subset_sha, "suffix": suffix, "judges": judges.split(),
                    "cache_from": cache_from.split(), "candidate_binding": json.loads(binding) if binding else None,
                    "pro_window": prow or None}
with open(out, "x", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
PY
  exec > >(tee -a "$REC.log") 2>&1
  trap 'rc=$?; echo "{\"exit\": $rc, \"end_beijing\": \"$(date "+%F %T")\"}" > "$REC.end"; rmdir "$OUTH/.lock" 2>/dev/null' EXIT
  step "launch record $REC.json"
}
partial_name() { [[ "$1" =~ ^t[0-9]*\. ]]; }        # the engine's PARTIAL_RE
has_full() {  # has_full SPLIT: OUT/gen holds a full (not t<N>.*) file of SPLIT
  local f
  for f in "$OUTH"/gen/*_"$1".json; do [ -e "$f" ] || continue; partial_name "${f##*/}" || return 0; done
  return 1
}
control_check() {  # control_check SPLIT PREFIX: each control must equal the driver's file token for token; one that does not
  local sp=$1 pre=$2 s f out ts                 # is moved to OUT/failed/ with a failure record, and the launch stops (S1)
  for s in $ES; do
    f=$OUTH/gen/${pre}conf--E$EL-w-p$(tok "$s")_$sp.json
    if out=$(pyh reproduce "$(cont "$f")" "$RUN/gen/E_L${EL}_conf-w-p$(tok "$s")_$sp.json" 2>&1); then
      echo "$out"; CTRL_OK="${CTRL_OK:-}$out"$'\n'; continue
    fi
    echo "$out"
    ts=$(date +%Y%m%dT%H%M%S); mkdir -p "$OUTH/failed"
    if [ -f "$f" ]; then mv -n "$f" "$OUTH/failed/${f##*/}.FAILED.$ts"; fi
    python3 - "$OUTH/failed/control_${sp}_s$(tok "$s")_$ts.json" "$f" "$OUTH/failed/${f##*/}.FAILED.$ts" "$out" "$(manifest_sha)" "${REC:-}" <<'PY'
import hashlib, json, os, sys
rec, f, moved, out, msha, launch = sys.argv[1:]
d = {"what": "control not reproduced: the output dir is closed", "control_file": f, "moved_to": moved if os.path.exists(moved) else None,
     "moved_sha256": hashlib.sha256(open(moved, "rb").read()).hexdigest() if os.path.exists(moved) else None,
     "reproduce_output": out, "manifest_sha256": msha, "launch_record": launch or None}
with open(rec, "x", encoding="utf-8") as fh:
    fh.write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
PY
    die "control $sp s=$s not reproduced: moved to failed/, record failed/control_${sp}_s$(tok "$s")_$ts.json; $OUT is closed (fresh OUT_ROOT)"
  done
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
# ---- B2: DeepSeek off-peak window. hm() and window() are verbatim copies of Iteration_01/launchers/
# ff_xu1_stage1_after_20261118.sh (sha256 aa4dc86d55cbd433; selftest_fns.sh compares them line by line). This launcher has
# no dry mode: DRY stays 0, so window() always computes the real Beijing peak windows.
DRY=0
hm() { date -d "@$1" '+%F %T %a'; }
window() {  # NOW -> "RUN <stop epoch>" | "WAIT <start epoch>" (as ff_g3_fresh_pro_offpeak_20261118.sh)
  python3 - "$1" "$DRY" "${XU_TEST_WINDOWS:-}" <<'PY'
import sys, time
now, dry, spec = int(sys.argv[1]), sys.argv[2] == "1", sys.argv[3]
if dry:
    for w in spec.split():
        s, e = map(int, w.split("-"))
        if now < s: print("WAIT", s); sys.exit()
        if now < e: print("RUN", e); sys.exit()
    print("NONE 0"); sys.exit()
def peak(t):
    lt = time.localtime(t)                      # TZ=Asia/Shanghai from the environment
    m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (540 <= m < 720 or 840 <= m < 1080)
def allowed(t):                                 # off-peak, >= 30 s after the peak ended, >= 5 min before the next one
    return not peak(t) and not peak(t - 30) and not peak(t + 300)
assert time.strftime("%z") == "+0800", "TZ is not Asia/Shanghai"
g = now - now % 30
if allowed(now):
    for k in range(1, 4 * 24 * 120):
        if not allowed(g + 30 * k): print("RUN", g + 30 * k); sys.exit()
else:
    for k in range(1, 4 * 24 * 120):
        if allowed(g + 30 * k): print("WAIT", g + 30 * k); sys.exit()
sys.exit("no window boundary within 4 days")
PY
}
PRO_MIN_LEFT=7200
pro_window() {  # sets PRO_STOP (5 min before the next peak) and PRO_LEFT, or refuses: peak hours, or < 2 h left
  local now out mode t
  now=$(date +%s)
  out=$(window "$now") || die "window computation failed"
  read -r mode t <<< "$out"
  [ "$mode" = RUN ] || die "DeepSeek peak hours (Beijing Mon-Fri 09:00-12:00 and 14:00-18:00, with a 30 s / 5 min margin): Pro is not started; the next off-peak window opens $(hm "$t")"
  PRO_STOP=$t; PRO_LEFT=$((t - now))
  [ "$PRO_LEFT" -ge "$PRO_MIN_LEFT" ] || die "only $((PRO_LEFT / 60)) min left in this off-peak window (it ends $(hm "$t")): Pro needs at least $((PRO_MIN_LEFT / 3600)) h; start it in the next window"
  echo "DeepSeek off-peak window open until $(hm "$PRO_STOP") ($((PRO_LEFT / 60)) min left)"
}
strict_verdicts() { pyh verdicts --work "$WORK" --judge "$1" --subset "$SUBSET"; }   # S2

echo "ladder: KEY=$KEY model=$MODEL run=$RUN source=$SRC code=$CODE E layer $EL (e_on $E_ON) strengths [$ES]"
echo "ladder: prompts [$PN] from $LADDER; control conf; out $OUT; model python $PYM"
case "$STAGE" in
dryrun)
  step "check: the in-memory changes for tune and for test (nothing runs)"
  lc check --code "$CODE" --out "$OUT" --split tune
  lc check --code "$CODE" --out "$OUT" --split test
  step "dryrun: identity fields, tokenizer, rendering, written span, write amount (CPU, tokenizer only, nothing written)"
  lc dryrun --code "$CODE" --source-run "$SRC" --run-dir "$RUN" --prompts-extra "$LADDER" --layer "$EL" --strengths $ES
  for s in $ES; do for sp in tune test; do
    if [ "$sp" = test ]; then g=$GX; else g=$GT; fi
    step "candidates of $sp at strength $s, rebuilt from the run's E+C grid eval_extra/$g (read-only)"
    out=$(lc bind --code "$CODE" --run-dir "$RUN" --source-run "$SRC" --grid "$RUN/eval_extra/$g" --layer "$EL" --strength "$s" --split "$sp")
    printf '%s\n' "$out" | grep -v '^BINDING ' || true
  done; done ;;
smoke)
  [ -n "$GPUN" ] || die "STAGE=smoke needs GPU (its device mapping goes into the manifest)"
  closed_guard; gpu_free; prepare_out new; record smoke "$CTRL |$LAD"
  step "smoke: the control alone on the first batch of tune (8 questions, t8.* files):$CTRL"
  lx generate "${COMMON[@]}" --split tune --first-batches 1 --arms $CTRL
  CTRL_OK=""; control_check tune t8.
  step "smoke: the ladder arms on the same 8 questions:$LAD"
  lx generate "${COMMON[@]}" --split tune --first-batches 1 --arms $LAD
  F=""
  for s in $ES; do F="$F $OUT/gen/t8.conf--E$EL-w-p$(tok "$s")_tune.json"; for p in $PN; do F="$F $OUT/gen/t8.$p--E$EL-w-p$(tok "$s")_tune.json"; done; done
  pyh show --n 8 $F
  python3 - "$OUTH/smoke_pass.json" "$(manifest_sha)" "$REC.json" "$CTRL_OK" <<'PY'
import json, os, sys, time
path, msha, rec, lines = sys.argv[1:]
d = {"passed": True, "manifest_sha256": msha, "launch_record": rec, "reproduced": [l for l in lines.splitlines() if l.strip()],
     "time_beijing": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 3600))}
if os.path.exists(path):
    old = json.load(open(path, encoding="utf-8"))
    sys.exit(0 if old.get("passed") is True and old.get("manifest_sha256") == msha else "smoke_pass.json exists for another manifest")
with open(path, "x", encoding="utf-8") as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
PY
  step "smoke passed: $OUT/smoke_pass.json (STAGE=gen may now run with GPU=$GPUN)" ;;
gen)
  [ -n "$GPUN" ] || die "STAGE=gen needs GPU (the device mapping of the smoke)"
  prepare_out existing; closed_guard; smoke_passed
  gpu_free; record gen "$CTRL |$LAD"
  for SPLIT in tune test; do
    if [ "${RESUME:-0}" != 1 ] && has_full "$SPLIT"; then
      die "$OUT/gen already holds full $SPLIT files: RESUME=1 verifies and continues them (nothing is overwritten)"
    fi
    step "$SPLIT: control$CTRL (the engine verifies the run and reproduces conf's frozen answers first)"
    lx generate "${COMMON[@]}" --split "$SPLIT" --arms $CTRL
    control_check "$SPLIT" ""
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
cand)
  : "${SPLIT:?SPLIT=tune|test}"
  case "$SPLIT" in tune) g=$GT ;; test) g=$GX ;; *) die "SPLIT must be tune or test" ;; esac
  prepare_out existing
  F=$OUTH/subsets/${SPLIT}_E${EL}_x-w-p$(tok "$CAND_S")_blank.json
  mkdir -p "$OUTH/subsets"
  step "candidates of $SPLIT, rebuilt from the run's E+C grid eval_extra/$g"
  out=$(lc bind --code "$CODE" --run-dir "$RUN" --source-run "$SRC" --grid "$RUN/eval_extra/$g" --layer "$EL" \
        --strength "$CAND_S" --split "$SPLIT" --write "$(cont "$F")")
  printf '%s\n' "$out" | grep -v '^BINDING ' || true
  echo "SUBSET=$(cont "$F")" ;;
judge)
  : "${SPLIT:?SPLIT=tune|test}" "${SUBSET:?SUBSET=<container path of the candidate file of STAGE=cand>}"
  case "$SPLIT" in tune) g=$GT ;; test) g=$GX ;; *) die "SPLIT must be tune or test" ;; esac
  SUFFIX=${SUFFIX:-cand}; JUDGES=${JUDGES:-allenai}                 # B2: no paid judge unless asked for
  case "$SUFFIX" in ""|*[!A-Za-z0-9]*) die "SUFFIX must be alphanumeric" ;; esac
  [ -n "$JUDGES" ] || die "JUDGES is empty"
  for j in $JUDGES; do case "$j" in allenai|pro) ;; *) die "JUDGES: unknown judge '$j' (allenai / pro)" ;; esac; done
  has() { case " $JUDGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
  set -f; set -- ${CACHE_FROM:-}; set +f
  [ $# -le 1 ] || die "CACHE_FROM takes exactly one work dir (got $#): merge several reply caches first (judge2_cache_merge.py)"
  CACHE1=${1:-}
  if [ -n "$CACHE1" ]; then
    has pro || die "CACHE_FROM seeds the Pro reply cache: it needs JUDGES with pro"
    [ -f "$(host "$CACHE1")/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl" ] || die "no Pro reply cache in $CACHE1/judge2/cache"
  fi
  [ -f "$(host "$SUBSET")" ] || die "no subset file $SUBSET"
  prepare_out existing
  NAME=ladder_$SPLIT; SRCW=$OUT/judge_${SPLIT}_$NAME; WORK=$SRCW$SUFFIX; WH=$(host "$WORK")
  [ -f "$(host "$SRCW")/judge_input.json" ] || die "no export $SRCW/judge_input.json: run STAGE=gen first"
  step "B1: SUBSET must equal the $SPLIT candidate set rebuilt from the run's E+C grid eval_extra/$g (read-only)"
  out=$(lc bind --code "$CODE" --run-dir "$RUN" --source-run "$SRC" --grid "$RUN/eval_extra/$g" --layer "$EL" \
        --strength "$CAND_S" --split "$SPLIT" --subset "$SUBSET") || { printf '%s\n' "${out:-}"; die "not judged: see the binding check above"; }
  printf '%s\n' "$out" | grep -v '^BINDING ' || true
  BINDING=$(printf '%s\n' "$out" | sed -n 's/^BINDING //p')
  NCAND=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["candidates"]["n"])' "$BINDING")
  PROW=""
  if has pro; then pro_window; PROW="stop $(hm "$PRO_STOP") ($PRO_STOP), $PRO_LEFT s left at the start"; fi
  if has allenai && [ "$NCAND" != 0 ]; then gpu_free; fi
  record "judge_$SPLIT" "$WORK" "$BINDING" "$PROW"
  if [ "$NCAND" = 0 ]; then step "no candidate on $SPLIT: nothing needs re-asking, nothing judged"; exit 0; fi
  if has allenai; then
    if [ -f "$WH/judge2/single_allenai.jsonl" ]; then
      step "AllenAI pass exists in $WORK: checked, not redone"
    elif [ "$SPLIT" = test ]; then
      step "test: scripts/review_checks/ff_test_subset_judge.sh (unchanged), JUDGES=allenai, on $SRCW -> $WORK"
      env -u CACHE_FROM MODEL="$MODEL" CODE_PATH="$JCODE" RUN="$OUT" NAMES="$NAME" SUFFIX="$SUFFIX" SUBSET="$SUBSET" \
        JUDGES=allenai GPU="$GPUN" bash "$JCODEH/scripts/review_checks/ff_test_subset_judge.sh"
    else
      step "tune: the export restricted to the questions of ${SUBSET##*/}"
      mkdir -p "$WH/judge2/cache"
      pyh subset "$SRCW/judge_input.json" "$SUBSET" "$WORK" --split tune
      step "tune: cells + allenai + single"
      pyj scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
      "${DJ[@]}" python3 scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
        --truth-model "$W/models/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$W/models/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
      pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WH/single_allenai_stdout.md"
    fi
    strict_verdicts allenai
  fi
  if has pro; then
    [ -f "$WH/judge2/single_allenai.jsonl" ] || die "no AllenAI pass in $WORK yet: run JUDGES=allenai first (it makes the cells)"
    has allenai || strict_verdicts allenai
    if [ -e "$WH/judge2/single_pro.jsonl" ]; then
      step "a Pro pass exists in $WORK: checked, not redone"
    else
      pro_window                                      # again: the AllenAI pass took time
      if [ -n "$CACHE1" ]; then                       # as launchers/ff_tune_pro_20261117.sh: merge by request key, stop on conflicts
        step "merge the Pro reply cache of $CACHE1"
        pyj scripts/review_checks/judge2_cache_merge.py "$WORK/judge2/cache" "$CACHE1/judge2/cache"
        pyj -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('merged caches give %d request keys different replies: stop' % n if n else 0)" "$WORK/judge2/cache/conflicts.json"
      fi
      step "pro, under a timeout of $PRO_LEFT s (until $(hm "$PRO_STOP"), 5 min before the next peak)"
      rc=0
      "${DJ[@]}" timeout --signal=TERM --kill-after=60 "$PRO_LEFT" python3 scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro || rc=$?
      if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
        python3 -c '
import json, os, sys
p = sys.argv[1]
if os.path.exists(p):
    for l in open(p, encoding="utf-8"):
        if l.strip(): json.loads(l)' "$WH/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl" || die "the Pro reply cache is unreadable after the stop (not repaired): $WH/judge2/cache"
        die "Pro stopped at the end of the off-peak window ($(hm "$PRO_STOP")): run the same command in the next window (answered requests are cached)"
      fi
      [ "$rc" = 0 ] || die "the Pro pass failed (exit $rc)"
      pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$WH/single_pro_stdout.md"
    fi
    strict_verdicts pro
  fi
  for j in $JUDGES; do step "verdicts in $WORK ($j)"; verdict_table "$WORK" "$j"; done ;;
esac
step "$STAGE done"
