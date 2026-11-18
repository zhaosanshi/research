#!/usr/bin/env bash
# CPU self-test of promptladder/ladder_extra.py and ladder_run.sh (no GPU, no judge call; writes only into
# Iteration_01/promptladder/selftest_20261118, which it removes at the end). Uses the engine's own existing files of the
# Gemma-3-12B run (eval_extra/gen/conf--E22-w-p2_*.json were made by e44_eval_extra.py in the E+C selection).
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling; W=/workspace/Truth_Decoupling
P=$R/Iteration_01/promptladder; PW=$W/Iteration_01/promptladder
T=$P/selftest_20261118; TW=$PW/selftest_20261118
RUNW=$W/Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b; RUNH=$R/${RUNW#$W/}
SRCW=$W/Iteration_01/results/ff/gemma3/12b/source/gemma3-12b
CODE=$W/Iteration_01/code
LAD=$PW/ladder_prompts_draft_20261118.json
DC=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=$CODE -w $CODE truth_dec)
h() { "${DC[@]}" python3 -B $PW/ladder_extra.py "$@"; }
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
expect_ok()   { local name=$1; shift; if out=$("$@" 2>&1); then ok "$name"; echo "$out" | tail -n 3 | sed 's/^/      /'; else bad "$name"; echo "$out" | tail -n 8 | sed 's/^/      /'; fi; }
expect_fail() { local name=$1 pat=$2; shift 2; if out=$("$@" 2>&1); then bad "$name (it ran)"; echo "$out" | tail -n 3; elif echo "$out" | grep -q -- "$pat"; then ok "$name"; echo "$out" | grep -- "$pat" | tail -n 1 | cut -c1-200 | sed 's/^/      /'; else bad "$name (other error)"; echo "$out" | tail -n 5; fi; }
[ ! -e "$T" ] || { echo "$T exists"; exit 2; }
mkdir -p "$T/out/gen" "$T/work_tune"
touch "$T/.ref"
cp "$RUNH/eval_extra/gen/conf--E22-w-p2_tune.json" "$RUNH/eval_extra/gen/conf--E22-w-p2_test.json" "$T/out/gen/"
before=$(cd "$RUNH" && find . -newer "$T/.ref" | wc -l)
COMMON=(--source-run $SRCW --run-dir $RUNW --e-on E2 --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $LAD)

echo "== 1. reproduce: an engine file of the run against the driver's file (as genconf did)"
expect_ok "reproduce tune" h reproduce $RUNW/eval_extra/gen/conf--E22-w-p2_tune.json $RUNW/gen/E_L22_conf-w-p2_tune.json
expect_ok "reproduce test" h reproduce $RUNW/eval_extra/gen/conf--E22-w-p2_test.json $RUNW/gen/E_L22_conf-w-p2_test.json
expect_fail "reproduce refuses another strength" "layer / strength" h reproduce $RUNW/eval_extra/gen/conf--E22-w-p1_tune.json $RUNW/gen/E_L22_conf-w-p2_tune.json

echo "== 2. export through the helper (output dir redirected; tune plain, test with the split widening)"
expect_ok "export tune" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split tune --gen conf--E22-w-p2_tune.json --name selftest_tune
expect_ok "export test" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split test --gen conf--E22-w-p2_test.json --name selftest_test
expect_ok "export again = identical (kept)" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split tune --gen conf--E22-w-p2_tune.json --name selftest_tune
python3 - "$T/out/judge_tune/selftest_tune.json" "$RUNH/judge_tune_proD_E_L22/judge_input.json" "$T/out/judge_test/selftest_test.json" "$RUNH/judge_test_testD_E_L22/judge_input.json" <<'PY' && ok "exported answers = the run's judged export of the same condition" || bad "exported answers differ"
import json, sys
for mine, ref in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    a, b = json.load(open(mine)), json.load(open(ref))
    assert a["conditions"] == ["baseline_conf", "E_L22_conf-w-p2"], a["conditions"]
    assert a["identity_sha256"] == b["identity_sha256"] and a["question_ids"] == b["question_ids"]
    rb = {e["pair_id"]: e for e in b["evaluations"]}
    for e in a["evaluations"]:
        for c in a["conditions"]:
            assert e["outputs"][c] == rb[e["pair_id"]]["outputs"][c] and e["rule_abstain"][c] == rb[e["pair_id"]]["rule_abstain"][c], (e["pair_id"], c)
    print("      %s: %d questions, conditions %s, identity %s, equal to %s" % (mine.rsplit("/", 1)[-1], len(a["evaluations"]), a["conditions"], a["identity_sha256"][:12], ref.rsplit("/", 2)[-2]))
PY

echo "== 3. refusals"
expect_fail "out dir inside the run" "must be separate directories" h extra --code $CODE --out $RUNW/eval_extra export "${COMMON[@]}" --split tune --gen conf--E22-w-p2_tune.json --name never
expect_fail "export without --gen" "never collects files" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split tune --name never
expect_fail "--overwrite" "not allowed" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split tune --gen conf--E22-w-p2_tune.json --name never --overwrite
expect_fail "arm of another kind" "only <ladder prompt" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --arms P1:E22.rand.p2
expect_fail "arm on the plain prompt x" "only <ladder prompt" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --arms x:E22.w.p2
expect_fail "two layers in one call" "one layer" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --arms P1:E22.w.p2 P2:E21.w.p2
printf '{"conf": "You are a test."}' > "$T/bad_conf.json"
printf '{"P1": "Same text.", "P2": "Same text."}' > "$T/bad_dup.json"
printf '{"P1": "One.", "P1": "Two."}' > "$T/bad_key.json"
printf '{"P1": " padded"}' > "$T/bad_pad.json"
expect_fail "ladder name of the run (conf)" "prompts of the run already" h extra --code $CODE --out $TW/out export --source-run $SRCW --run-dir $RUNW --e-on E2 --prompts-extra $TW/bad_conf.json --split tune --gen conf--E22-w-p2_tune.json --name never
expect_fail "two ladder prompts with one text" "same text" h extra --code $CODE --out $TW/out export --source-run $SRCW --run-dir $RUNW --e-on E2 --prompts-extra $TW/bad_dup.json --split tune --gen conf--E22-w-p2_tune.json --name never
expect_fail "duplicate key in the ladder file" "duplicate JSON key" h extra --code $CODE --out $TW/out export --source-run $SRCW --run-dir $RUNW --e-on E2 --prompts-extra $TW/bad_key.json --split tune --gen conf--E22-w-p2_tune.json --name never
expect_fail "text with outer spaces" "outer spaces" h extra --code $CODE --out $TW/out export --source-run $SRCW --run-dir $RUNW --e-on E2 --prompts-extra $TW/bad_pad.json --split tune --gen conf--E22-w-p2_tune.json --name never

echo "== 4. subset (tune) and candidates"
printf '{"pair_ids": ["tqa:685", "tqa:270", "tqa:480"]}' > "$T/sub3.json"
expect_ok "subset tune: 3 questions" h subset $TW/out/judge_tune/selftest_tune.json $TW/sub3.json $TW/work_tune --split tune
expect_ok "subset again = identical (kept)" h subset $TW/out/judge_tune/selftest_tune.json $TW/sub3.json $TW/work_tune --split tune
expect_fail "subset of the wrong split" "not a completed export of the test split" h subset $TW/out/judge_tune/selftest_tune.json $TW/sub3.json $TW/work_tune --split test
printf '{"pair_ids": ["tqa:685", "tqa:6"]}' > "$T/sub_other.json"
expect_fail "subset with a foreign question" "not in the export" h subset $TW/out/judge_tune/selftest_tune.json $TW/sub_other.json $TW/work_tune --split tune
python3 - "$T/work_tune/judge_input.json" <<'PY' && ok "subset judge_input holds exactly the 3 questions, export order" || bad "subset content"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["question_ids"] == [e["pair_id"] for e in d["evaluations"]] and set(d["question_ids"]) == {"tqa:685", "tqa:270", "tqa:480"}
assert d["split"] == "tune" and d["completed"] is True
PY
expect_ok "candidates (stdout): blanks of E alone at 2 on the plain prompt, tune, Pro file" h candidates --work $RUNW/eval_extra/judge_tune_ec25_tune --condition E_L22_x-w-p2 --judge pro --out -
expect_fail "candidates of a missing condition" "no rows of condition" h candidates --work $RUNW/eval_extra/judge_tune_ec25_tune --condition E_L22_P9-w-p2 --judge pro --out -

echo "== 5. show (existing engine files)"
expect_ok "show 2 questions" h show --n 2 $RUNW/eval_extra/gen/conf--E22-w-p2_tune.json $RUNW/eval_extra/gen/conf--E22-w-p1_tune.json

echo "== 6. launcher refusals (nothing created)"
L=$P/ladder_run.sh
expect_fail "unknown KEY" "unknown KEY" env KEY=nope LADDER=$LAD STAGE=gen bash $L
expect_fail "bad strength" "is not a number" env KEY=g12 LADDER=$LAD STAGE=gen ES=2x bash $L
expect_fail "strength without driver file" "no driver file" env KEY=g12 LADDER=$LAD STAGE=gen ES=3 bash $L
expect_fail "GPU stage without GPU" "needs GPU" env KEY=g12 LADDER=$LAD STAGE=gen OUT_ROOT=$TW/runs bash $L
G=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | head -n 1)
GI=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$G" | cut -d, -f1)
if [ -n "$GI" ]; then expect_fail "busy GPU $GI refused" "is busy" env KEY=g12 LADDER=$LAD STAGE=gen GPU=$GI OUT_ROOT=$TW/runs bash $L; fi
expect_fail "OUT_ROOT inside the run" "lies in the run" env KEY=g12 LADDER=$LAD STAGE=gen GPU=0 OUT_ROOT=$RUNW/x bash $L
expect_fail "judge before gen" "run STAGE=gen first" env KEY=g12 LADDER=$LAD STAGE=judge SPLIT=tune SUBSET=$TW/sub3.json JUDGES=pro OUT_ROOT=$TW/runs bash $L
[ ! -e "$T/runs" ] && ok "no output dir was created by the refused launches" || bad "an output dir was created"

after=$(cd "$RUNH" && find . -newer "$T/.ref" | wc -l)
[ "$before" = 0 ] && [ "$after" = 0 ] && ok "no file in the run directory is newer than the test start" || bad "files in the run changed ($before / $after)"
echo "== $pass passed, $fail failed"
rm -rf "$T"
[ "$fail" = 0 ]
