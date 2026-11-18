#!/usr/bin/env bash
# Unit test of ladder_run.sh's bookkeeping functions (manifest, launch record, lock, partial-file test), extracted verbatim
# from the launcher; nothing is generated or judged. Writes only into Iteration_01/promptladder/selftest_fns_20261118,
# which it removes at the end.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling; W=/workspace/Truth_Decoupling
L=$R/Iteration_01/promptladder/ladder_run.sh
T=$R/Iteration_01/promptladder/selftest_fns_20261118
[ ! -e "$T" ] || { echo "$T exists"; exit 2; }
mkdir -p "$T"
for f in die host tok step partial_name; do eval "$(grep -E "^$f\(\) " "$L")"; done
for f in manifest_json prepare_out record has_full; do eval "$(awk -v f="$f" '$0 ~ "^"f"\\(\\) " {p=1} p {print} p && /^}/ {p=0}' "$L")"; done
for f in die host tok step partial_name manifest_json prepare_out record has_full; do declare -F "$f" > /dev/null || { echo "function $f not extracted"; exit 2; }; done
KEY=g12 MODEL=gemma3-12b EL=22 E_ON=E2 ES=2 GPU=9 PYM=python3 STAGE=gen SELFH=$L
RUN=$W/Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b SRC=$W/Iteration_01/results/ff/gemma3/12b/source/gemma3-12b
CODE=$W/Iteration_01/code LADDER=$W/Iteration_01/promptladder/ladder_prompts_draft_20261118.json
LADH=$(host "$LADDER") HELPERH=$R/Iteration_01/promptladder/ladder_extra.py CODEH=$(host "$CODE") SRCH=$(host "$SRC")
OUT=$W/Iteration_01/promptladder/selftest_fns_20261118/runs/gemma3-12b; OUTH=$(host "$OUT")
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }

( prepare_out ) && [ -f "$OUTH/ladder_manifest.json" ] && [ -d "$OUTH/launch" ] && [ -d "$OUTH/gen" ] && ok "prepare_out creates OUT with its manifest" || bad "prepare_out (new)"
python3 - "$OUTH/ladder_manifest.json" <<'PY' && ok "manifest holds the inputs (ladder order, texts, hashes)" || bad "manifest content"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["key"] == "g12" and m["e_layer"] == 22 and m["strengths"] == ["2"] and m["ladder_order"] == ["P1", "P2", "P3"]
assert m["ladder_sha256"].startswith("6633a8d6") and set(m["engine_sha256"]) == {"scripts/e44_eval_extra.py", "scripts/e44_factorial_das.py", "scripts/review_checks/e44_split_test.py"}
print("      manifest: key %s, layer %s, strengths %s, ladder %s (%s), helper %s" % (m["key"], m["e_layer"], m["strengths"], m["ladder_order"], m["ladder_sha256"][:12], m["helper_sha256"][:12]))
PY
( prepare_out ) && ok "prepare_out again with the same inputs" || bad "prepare_out (same)"
out=$( ES="1 2"; prepare_out 2>&1 ) && bad "other strengths accepted" || { echo "$out" | grep -q "made from other inputs" && ok "other strengths refused: $out" || bad "other strengths: $out"; }
out=$( STAGE=judge; OUTH=$T/runs/none; prepare_out 2>&1 ) && bad "judge without manifest accepted" || { echo "$out" | grep -q "run STAGE=gen first" && ok "judge without manifest refused" || bad "judge without manifest: $out"; }
mkdir -p "$T/runs/stray"; touch "$T/runs/stray/x"
out=$( OUTH=$T/runs/stray; prepare_out 2>&1 ) && bad "stray OUT accepted" || { echo "$out" | grep -q "exists without a manifest" && ok "existing OUT without a manifest refused" || bad "stray OUT: $out"; }

( record gen "conf:E22.w.p2 | P1:E22.w.p2"; echo "inside the launch" ) > "$T/rec1.out" 2>&1
sleep 1
r1=$(ls "$OUTH"/launch/*_gen.json 2>/dev/null | head -n 1)
[ -n "$r1" ] && [ -f "${r1%.json}.log" ] && [ -f "${r1%.json}.end" ] && [ ! -e "$OUTH/.lock" ] && ok "record writes .json/.log/.end and releases the lock" || bad "record files"
grep -q '"exit": 0' "${r1%.json}.end" && grep -q "inside the launch" "${r1%.json}.log" && ok "the .end file has exit 0; the .log has the launch output" || bad "record end/log"
python3 - "$r1" <<'PY' && ok "the record holds plan, launcher sha256 and the manifest" || bad "record content"
import json, sys
r = json.load(open(sys.argv[1]))
assert r["stage"] == "gen" and r["plan"] == ["conf:E22.w.p2", "|", "P1:E22.w.p2"] and len(r["launcher_sha256"]) == 64
assert r["manifest"]["ladder_order"] == ["P1", "P2", "P3"] and r["gpu"] == "9"
print("      record %s: time %s, plan %s" % (sys.argv[1].rsplit("/", 1)[-1], r["time_beijing"], r["plan"]))
PY
sleep 1
( record gen x; exit 3 ) > /dev/null 2>&1
r2=$(ls -t "$OUTH"/launch/*_gen.end | head -n 1)
grep -q '"exit": 3' "$r2" && [ ! -e "$OUTH/.lock" ] && ok "a failing launch leaves exit 3 in its .end and releases the lock" || bad "failing launch"
mkdir "$OUTH/.lock"
out=$( record gen x 2>&1 ) && bad "second concurrent launch accepted" || { echo "$out" | grep -q "another launch holds" && ok "a second concurrent launch is refused" || bad "lock: $out"; }
[ -d "$OUTH/.lock" ] && ok "the refused launch left the other launch's lock alone" || bad "lock removed by the refused launch"
rmdir "$OUTH/.lock"

touch "$OUTH/gen/t8.conf--E22-w-p2_tune.json" "$OUTH/gen/t8.P1--E22-w-p2_tune.json"
has_full tune && bad "partial files counted as full" || ok "t8.* files are partial (has_full tune = no)"
touch "$OUTH/gen/t1--E22-w-p2_test.json"
has_full test && ok "a full file of a prompt named t1 is full (has_full test = yes)" || bad "t1 prompt file"
has_full tune && bad "test file counted for tune" || ok "has_full looks only at its split"
partial_name "t12.x_tune.json" && ! partial_name "t1--E22-w-p2_tune.json" && ! partial_name "conf--E22-w-p2_tune.json" && ok "partial_name = the engine's ^t[0-9]*\\." || bad "partial_name"
echo "== $pass passed, $fail failed"
rm -rf "$T"
[ "$fail" = 0 ]
