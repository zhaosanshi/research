#!/usr/bin/env bash
# Unit test of ladder_run.sh's bookkeeping and gating functions, extracted verbatim from the launcher (nothing is
# generated or judged; no GPU, no DeepSeek). Writes only into Iteration_01/promptladder/selftest_fns_20261118, which it
# removes at the end. Revised 2026-11-18 after GPT's review: device mapping in the manifest, smoke pass bound to the
# manifest, a failed control closes the output dir (S1); the off-peak window, copied verbatim, with a stubbed clock (B2).
set -uo pipefail
export TZ=Asia/Shanghai
R=/data/sanshizhao/Truth_Decoupling; W=/workspace/Truth_Decoupling
L=$R/Iteration_01/promptladder/ladder_run.sh
XU=$R/Iteration_01/launchers/ff_xu1_stage1_after_20261118.sh
T=$R/Iteration_01/promptladder/selftest_fns_20261118
[ ! -e "$T" ] || { echo "$T exists"; exit 2; }
mkdir -p "$T"
one() { grep -E "^$1\(\) " "$2"; }
multi() { awk -v f="$1" '$0 ~ "^"f"\\(\\) " {p=1} p {print} p && /^}/ {exit}' "$2"; }
for f in die host cont tok step partial_name manifest_gpus manifest_sha hm strict_verdicts; do eval "$(one "$f" "$L")"; done
for f in manifest_json prepare_out closed_guard smoke_passed record has_full control_check window pro_window; do eval "$(multi "$f" "$L")"; done
for f in die host cont tok step partial_name manifest_gpus manifest_sha hm manifest_json prepare_out closed_guard smoke_passed \
         record has_full control_check window pro_window; do declare -F "$f" > /dev/null || { echo "function $f not extracted"; exit 2; }; done
eval "$(grep -E '^PRO_MIN_LEFT=' "$L")"; DRY=0
KEY=g12 MODEL=gemma3-12b EL=22 E_ON=E2 ES=2 GPUN=9 PYM=python3 STAGE=smoke SELFH=$L
RUN=$W/Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b SRC=$W/Iteration_01/results/ff/gemma3/12b/source/gemma3-12b
CODE=$W/Iteration_01/code LADDER=$W/Iteration_01/promptladder/ladder_prompts_draft_20261118.json
LADH=$(host "$LADDER") HELPERH=$R/Iteration_01/promptladder/ladder_extra.py CODEH=$(host "$CODE") SRCH=$(host "$SRC")
OUTH=$T/runs/gemma3-12b; OUT=$(cont "$OUTH")
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
refused() { local name=$1 pat=$2; shift 2; local out; if out=$("$@" 2>&1); then bad "$name (accepted)"; elif echo "$out" | grep -q -- "$pat"; then ok "$name: $(echo "$out" | grep -- "$pat" | tail -n 1 | cut -c1-170)"; else bad "$name (other error: $out)"; fi; }

echo "== S1: the manifest holds the device mapping; a different GPU list needs a fresh OUT_ROOT"
( prepare_out new ) && python3 - "$OUTH/ladder_manifest.json" <<'PY' && ok "STAGE=smoke GPU=9 creates OUT; manifest device_mapping = {generation_gpus 9, n_gpus 1, model_device_map None}" || bad "manifest"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["device_mapping"] == {"generation_gpus": "9", "n_gpus": 1, "model_device_map": None}, m["device_mapping"]
assert m["ladder_order"] == ["P1", "P2", "P3"] and m["ladder_sha256"].startswith("6633a8d6") and m["e_layer"] == 22
PY
( STAGE=gen; GPUN=9; prepare_out existing ) && ok "STAGE=gen GPU=9 finds the same manifest" || bad "gen same GPU"
gen8() { STAGE=gen; GPUN=8; prepare_out existing; }
smoke8() { STAGE=smoke; GPUN=8; prepare_out new; }
gen_fresh() { STAGE=gen; GPUN=9; OUTH=$T/runs/none; prepare_out existing; }
refused "STAGE=gen with GPU=8 on an OUT made with GPU=9" "a different device mapping needs a fresh OUT_ROOT" gen8
refused "STAGE=smoke with GPU=8 on an OUT made with GPU=9" "another device mapping" smoke8
( STAGE=judge; GPUN=""; prepare_out existing ) && ok "STAGE=judge (no GPU) checks the manifest with its recorded device mapping" || bad "judge manifest"
refused "STAGE=gen on a fresh OUT (no manifest)" "run STAGE=smoke first" gen_fresh

echo "== S1: gen only after a passing smoke of this very manifest"
refused "gen without smoke" "no passing smoke" smoke_passed
printf '{"passed": true, "manifest_sha256": "%s"}\n' "0000" > "$OUTH/smoke_pass.json"
refused "gen with a smoke pass of another manifest" "belongs to another manifest" smoke_passed
printf '{"passed": false, "manifest_sha256": "%s"}\n' "$(manifest_sha)" > "$OUTH/smoke_pass.json"
refused "gen with a smoke record that did not pass" "belongs to another manifest" smoke_passed
printf '{"passed": true, "manifest_sha256": "%s"}\n' "$(manifest_sha)" > "$OUTH/smoke_pass.json"
( smoke_passed ) && ok "gen accepted after a passing smoke of this manifest" || bad "smoke pass"

echo "== S1: a control that does not reproduce is moved aside, recorded, and closes OUT (a resume cannot reuse it)"
mkdir -p "$OUTH/gen"; echo '{"control": "made by a stub"}' > "$OUTH/gen/conf--E22-w-p2_tune.json"
pyh() { echo "- conf--E22-w-p2_tune.json: 190 / 192 identical with the driver file (stub)"; return 1; }
out=$( REC=$T/rec; control_check tune "" 2>&1 ); rc=$?
mv_ok=$(ls "$OUTH"/failed/conf--E22-w-p2_tune.json.FAILED.* 2>/dev/null | wc -l)
recf=$(ls "$OUTH"/failed/control_tune_s2_*.json 2>/dev/null | head -n 1)
[ "$rc" != 0 ] && [ "$mv_ok" = 1 ] && [ ! -e "$OUTH/gen/conf--E22-w-p2_tune.json" ] && [ -n "$recf" ] && ok "failed control: launch stops (exit $rc), file moved to failed/*.FAILED.<time>, gen/ no longer holds it" || bad "failed control ($rc, $mv_ok, $recf)"
python3 - "$recf" <<'PY' && ok "the failure record names the moved file, its sha256 and the reproduce output" || bad "failure record"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["moved_to"].endswith(".FAILED." + d["moved_to"].rsplit(".", 1)[1]) and len(d["moved_sha256"]) == 64 and "190 / 192" in d["reproduce_output"]
PY
refused "resume after a failed control (closed OUT)" "this OUT is closed" closed_guard
pyh() { echo "- stub: 192 / 192 identical with the driver file"; return 0; }
echo '{"control": "new"}' > "$OUTH/gen/conf--E22-w-p2_tune.json"
( CTRL_OK=""; control_check tune "" > /dev/null ) && ok "a reproducing control passes control_check" || bad "control pass"
rm -rf "$OUTH/failed"
( closed_guard ) && ok "closed_guard passes when no failure record exists" || bad "closed_guard"

echo "== B2: window() and hm() are verbatim copies of the xu1 launcher's"
[ "$(multi window "$L")" = "$(multi window "$XU")" ] && [ -n "$(multi window "$XU")" ] && ok "window(): identical text ($(multi window "$XU" | wc -l) lines) to $(basename "$XU")" || bad "window() differs"
[ "$(one hm "$L")" = "$(one hm "$XU")" ] && ok "hm(): identical text" || bad "hm() differs"

echo "== B2: Pro only off-peak, >= 2 h left (stubbed clock: date +%s returns the test time)"
date() { local a; for a in "$@"; do case "$a" in -d*|--date*) command date "$@"; return ;; esac; done; command date -d "@$FAKE_NOW" "$@"; }
probe() { FAKE_NOW=$2; local out; if out=$( pro_window 2>&1 ); then echo "ALLOW|$out"; else echo "REFUSE|$out"; fi; }
while IFS='|' read -r label epoch want; do
  res=$(probe "$label" "$epoch")
  case "$want:$res" in
    peak:REFUSE*"peak hours"*|short:REFUSE*"at least 2 h"*|ok:ALLOW*) ok "$label -> ${res%%|*}: $(echo "${res#*|}" | cut -c1-120)" ;;
    *) bad "$label: expected $want, got $res" ;;
  esac
done <<'EOF'
Wed 2026-11-18 10:00:00 (morning peak)|1794967200|peak
Wed 12:00:10 (10 s after the peak: inside the 30 s margin)|1794974410|peak
Wed 12:00:40 (window open until 13:55: 1 h 54 min)|1794974440|short
Wed 13:56:00 (5 min before the afternoon peak)|1794981360|peak
Wed 18:00:31 (window open until Thu 08:55)|1794996031|ok
Fri 2026-11-20 18:30 (until Mon 08:55)|1795170600|ok
Sat 2026-11-21 10:00 (weekend)|1795226400|ok
Sun 2026-11-22 23:00 (until Mon 08:55: 9 h 55 min)|1795359600|ok
Mon 2026-11-23 06:00 (until 08:55: 2 h 55 min)|1795384800|ok
Mon 07:00 (until 08:55: 1 h 55 min)|1795388400|short
EOF
FAKE_NOW=1794999600; pro_window > /dev/null 2>&1
[ "${PRO_STOP:-}" = "$(TZ=Asia/Shanghai command date -d '2026-11-19 08:55:00' +%s)" ] && [ "${PRO_LEFT:-}" = $((PRO_STOP - 1794999600)) ] && ok "Wed 19:00: the Pro timeout ends Thu 08:55:00, 5 min before the peak (${PRO_LEFT} s)" || bad "stop time (${PRO_STOP:-} ${PRO_LEFT:-})"
unset -f date

echo "== launch record: binding and window saved; lock; failure exit status"
( STAGE=judge; SPLIT=tune; SUBSET=$W/Iteration_01/promptladder/ladder_prompts_draft_20261118.json; JUDGES=pro; record judge_tune "work" '{"candidates": {"n": 76, "ids_sha256": "abc"}, "population": {"n": 192}}' "stop x"; echo "inside the launch" ) > "$T/rec1.out" 2>&1
sleep 1
r1=$(ls "$OUTH"/launch/*_judge_tune.json 2>/dev/null | head -n 1)
python3 - "$r1" <<'PY' && ok "judge record holds the candidate binding, the Pro window, the manifest sha256 and the launcher sha256" || bad "record content"
import json, sys
r = json.load(open(sys.argv[1]))
j = r["judge"]
assert j["candidate_binding"]["candidates"]["n"] == 76 and j["pro_window"] == "stop x" and len(r["manifest_sha256"]) == 64
assert len(r["launcher_sha256"]) == 64 and r["manifest"]["device_mapping"]["generation_gpus"] == "9"
PY
[ -f "${r1%.json}.end" ] && grep -q '"exit": 0' "${r1%.json}.end" && [ ! -e "$OUTH/.lock" ] && ok "the .end file has exit 0; the lock is released" || bad "record end/lock"
mkdir "$OUTH/.lock"
refused "a second concurrent launch" "another launch holds" record gen x
[ -d "$OUTH/.lock" ] && ok "the refused launch left the other launch's lock alone" || bad "lock removed by the refused launch"
rmdir "$OUTH/.lock"
sleep 1
( record gen x; exit 3 ) > /dev/null 2>&1
r2=$(ls -t "$OUTH"/launch/*_gen.end | head -n 1)
grep -q '"exit": 3' "$r2" && [ ! -e "$OUTH/.lock" ] && ok "a failing launch leaves exit 3 in its .end and releases the lock" || bad "failing launch"

echo "== partial files"
touch "$OUTH/gen/t8.conf--E22-w-p2_tune.json" "$OUTH/gen/t8.P1--E22-w-p2_tune.json"; rm -f "$OUTH/gen/conf--E22-w-p2_tune.json"
has_full tune && bad "partial files counted as full" || ok "t8.* files are partial (has_full tune = no)"
touch "$OUTH/gen/t1--E22-w-p2_test.json"
has_full test && ok "a full file of a prompt named t1 is full (has_full test = yes)" || bad "t1 prompt file"
partial_name "t12.x_tune.json" && ! partial_name "t1--E22-w-p2_tune.json" && ! partial_name "conf--E22-w-p2_tune.json" && ok "partial_name = the engine's ^t[0-9]*\\." || bad "partial_name"
echo "== $pass passed, $fail failed"
rm -rf "$T"
[ "$fail" = 0 ]
