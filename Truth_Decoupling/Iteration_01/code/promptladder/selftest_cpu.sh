#!/usr/bin/env bash
# CPU self-test of promptladder/ladder_extra.py and ladder_run.sh (no GPU, no judge call, no DeepSeek; writes only into
# Iteration_01/promptladder/selftest_20261118, which it removes at the end). Revised 2026-11-18 after GPT's review of
# commit 02f4cc9 (B1 B2 S1 S2 S3): each counterexample of the review is reproduced and must now be refused, and the normal
# path must still pass. Real inputs: the Gemma-3-12B run (its engine files, E+C grids and judged work dirs, read-only);
# the Llama run for "another run's" inputs. Mutations are made on copies inside the test folder only.
set -uo pipefail
export TZ=Asia/Shanghai
R=/data/sanshizhao/Truth_Decoupling; W=/workspace/Truth_Decoupling
P=$R/Iteration_01/promptladder; PW=$W/Iteration_01/promptladder
T=$P/selftest_20261118; TW=$PW/selftest_20261118
RUNW=$W/Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b; RUNH=$R/${RUNW#$W/}
SRCW=$W/Iteration_01/results/ff/gemma3/12b/source/gemma3-12b
LLRUNW=$W/results_e44_factorial_das/llama3.1-8b_side_E2; LLSRCW=$W/results_qa_das_v1u/llama3.1-8b/full
CODE=$W/Iteration_01/code; LLCODE=$W/code_e45_20261112
LAD=$PW/ladder_prompts_draft_20261118.json
L=$P/ladder_run.sh
DC=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=$CODE -w $CODE truth_dec)
h() { "${DC[@]}" python3 -B $PW/ladder_extra.py "$@"; }
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
expect_ok()   { local name=$1; shift; local out; if out=$("$@" 2>&1); then ok "$name"; echo "$out" | grep -v '^BINDING ' | tail -n 2 | cut -c1-230 | sed 's/^/      /'; else bad "$name"; echo "$out" | tail -n 6 | sed 's/^/      /'; fi; }
expect_fail() { local name=$1 pat=$2; shift 2; local out; if out=$("$@" 2>&1); then bad "$name (it ran)"; echo "$out" | tail -n 3; elif echo "$out" | grep -q -- "$pat"; then ok "$name"; echo "$out" | grep -- "$pat" | tail -n 1 | cut -c1-230 | sed 's/^/      /'; else bad "$name (other error)"; echo "$out" | tail -n 5; fi; }
[ ! -e "$T" ] || { echo "$T exists"; exit 2; }
mkdir -p "$T/out/gen" "$T/work_tune" "$T/bin"
touch "$T/.ref"
cp "$RUNH/eval_extra/gen/conf--E22-w-p2_tune.json" "$RUNH/eval_extra/gen/conf--E22-w-p2_test.json" "$T/out/gen/"
COMMON=(--source-run $SRCW --run-dir $RUNW --e-on E2 --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $LAD)
BIND12=(bind --code $CODE --run-dir $RUNW --source-run $SRCW --layer 22 --strength 2)

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
PY

echo "== 3. refusals of the helper; S3: exact option names only (an abbreviation was taken by the engine's parser)"
expect_fail "out dir inside the run" "must be separate directories" h extra --code $CODE --out $RUNW/eval_extra export "${COMMON[@]}" --split tune --gen conf--E22-w-p2_tune.json --name never
expect_fail "export without --gen" "never collects files" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split tune --name never
expect_fail "--overwrite" "not accepted" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split tune --gen conf--E22-w-p2_tune.json --name never --overwrite
expect_fail "--overw (GPT's abbreviation of --overwrite)" "not accepted" h extra --code $CODE --out $TW/out export "${COMMON[@]}" --split tune --gen conf--E22-w-p2_tune.json --name never --overw
expect_fail "--first-batch (abbreviation of --first-batches)" "not accepted" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --first-batch 1 --arms P1:E22.w.p2
expect_fail "--c-run-dir (unknown to the launcher's calls)" "not accepted" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --c-run-dir $RUNW --arms P1:E22.w.p2
expect_fail "bind --wri (abbreviation of --write)" "unrecognized arguments" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_tune_ec25_tune --split tune --wri $TW/x.json
expect_fail "arm of another kind" "only <ladder prompt" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --arms P1:E22.rand.p2
expect_fail "arm on the plain prompt x" "only <ladder prompt" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --arms x:E22.w.p2
expect_fail "two layers in one call" "one layer" h extra --code $CODE --out $TW/out generate "${COMMON[@]}" --split tune --arms P1:E22.w.p2 P2:E21.w.p2
printf '{"conf": "You are a test."}' > "$T/bad_conf.json"
printf '{"P1": "Same text.", "P2": "Same text."}' > "$T/bad_dup.json"
printf '{"P1": "One.", "P1": "Two."}' > "$T/bad_key.json"
printf '{"P1": " padded"}' > "$T/bad_pad.json"
for b in "conf|prompts of the run already" "dup|same text" "key|duplicate JSON key" "pad|outer spaces"; do
  expect_fail "ladder file bad_${b%%|*}" "${b#*|}" h extra --code $CODE --out $TW/out export --source-run $SRCW --run-dir $RUNW --e-on E2 --prompts-extra $TW/bad_${b%%|*}.json --split tune --gen conf--E22-w-p2_tune.json --name never
done

echo "== 4. B1, real runs: the candidate set of THIS run and split, rebuilt from its own E+C grid; SUBSET must equal it"
expect_ok "bind Gemma-3-12B tune (normal path), written as a candidate file" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_tune_ec25_tune --split tune --write $TW/cand_g12_tune.json
expect_ok "bind Gemma-3-12B tune with that file as SUBSET" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_tune_ec25_tune --split tune --subset $TW/cand_g12_tune.json
expect_ok "bind Gemma-3-12B test (normal path, 529 generated, 6 special gold removed)" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_test_t25_test --split test --write $TW/cand_g12_test.json
expect_ok "bind Llama-3.1-8B tune (another run, its own code copy)" h bind --code $LLCODE --run-dir $LLRUNW --source-run $LLSRCW --layer 11 --strength 2 --grid $LLRUNW/eval_extra/judge_tune_ecll19_tune --split tune --write $TW/cand_llama_tune.json
python3 - "$T" <<'PY' && ok "built: a same-size foreign set (one candidate swapped for a non-candidate), a repeated id, an empty set" || bad "subset fixtures"
import json, sys
T = sys.argv[1]
d = json.load(open(T + "/cand_g12_tune.json"))
ids, pop = d["pair_ids"], json.load(open(T + "/cand_g12_tune.json"))["binding"]
q = [x["pair_id"] for x in json.load(open("/data/sanshizhao/Truth_Decoupling/Iteration_01/results/ff/gemma3/12b/source/gemma3-12b/questions.json"))["splits"]["tune"]]
non = [x for x in q if x not in set(ids)]
json.dump({"pair_ids": ids[:-1] + [non[0]]}, open(T + "/foreign_same_size.json", "w"))
json.dump({"pair_ids": ids[:-2] + [ids[0], ids[-1]]}, open(T + "/repeated.json", "w"))
json.dump({"pair_ids": []}, open(T + "/empty.json", "w"))
assert len(ids) == 76 and len(ids[:-1] + [non[0]]) == 76
PY
expect_fail "foreign SUBSET of the right size (76)" "is not this run's tune candidate set" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_tune_ec25_tune --split tune --subset $TW/foreign_same_size.json
expect_fail "another run's SUBSET (Llama's 77)" "is not this run's tune candidate set" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_tune_ec25_tune --split tune --subset $TW/cand_llama_tune.json
expect_fail "empty SUBSET" "is not this run's tune candidate set" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_tune_ec25_tune --split tune --subset $TW/empty.json
expect_fail "SUBSET repeating an id (76 entries)" "repeated" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_tune_ec25_tune --split tune --subset $TW/repeated.json
expect_fail "the tune SUBSET given for test" "is not this run's test candidate set" h "${BIND12[@]}" --grid $RUNW/eval_extra/judge_test_t25_test --split test --subset $TW/cand_g12_tune.json
expect_fail "another run's grid for this run" "is not a work dir in this run's eval_extra" h "${BIND12[@]}" --grid $LLRUNW/eval_extra/judge_tune_ecll19_tune --split tune

echo "== 5. B1, GPT's counterexamples on a copy of the real tune grid (fake run dir: the run's files linked, the grid copied)"
FR=$T/fake/gemma3-12b; FRW=$TW/fake/gemma3-12b; G=$FR/eval_extra/judge_tune_ec25_tune
mkdir -p "$FR/dirs" "$G/judge2"
rel() { python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$2"; }   # valid on host and container
ln -s "$(rel "$RUNH/frozen_tune.json" "$FR")" "$FR/frozen_tune.json"; ln -s "$(rel "$RUNH/dirs/E_L22.pt" "$FR/dirs")" "$FR/dirs/E_L22.pt"
cp "$RUNH/eval_extra/judge_tune_ec25_tune/judge2/cells.jsonl" "$RUNH/eval_extra/judge_tune_ec25_tune/judge2/single_pro.jsonl" \
   "$RUNH/eval_extra/judge_tune_ec25_tune/judge2/single_allenai.jsonl" "$G/judge2/"
python3 - "$RUNH/eval_extra/judge_tune_ec25_tune/judge_input.json" "$G/judge_input.json" "$RUNW/frozen_tune.json" "$FRW/frozen_tune.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["sources"]["sha256"][sys.argv[4]] = d["sources"]["sha256"].pop(sys.argv[3])     # the only edit: the frozen file's path
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY
for f in judge_input.json judge2/cells.jsonl judge2/single_pro.jsonl judge2/single_allenai.jsonl; do cp "$G/$f" "$G/$f.orig"; done
BF=(bind --code $CODE --run-dir $FRW --source-run $SRCW --layer 22 --strength 2 --grid $FRW/eval_extra/judge_tune_ec25_tune --split tune)
fx() { python3 - "$G" "$1" <<'PY'
import json, sys
G, what = sys.argv[1], sys.argv[2]
C = "E_L22_x-w-p2"
def rl(p): return [json.loads(l) for l in open(p) if l.strip()]
def wl(p, rows): open(p, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
for f in ("judge_input.json", "judge2/cells.jsonl", "judge2/single_pro.jsonl", "judge2/single_allenai.jsonl"):
    open(G + "/" + f, "w").write(open(G + "/" + f + ".orig").read())
d = json.load(open(G + "/judge_input.json"))
blank = [e["pair_id"] for e in d["evaluations"] if e["rule_abstain"][C] or not e["outputs"][C].strip()]
q1 = blank[0]; q2 = [e["pair_id"] for e in d["evaluations"] if e["pair_id"] != q1][0]
if what == "dup_missing_judged":      # GPT: q1 twice (one blank, one answered), q2 missing, the row count unchanged
    rows = rl(G + "/judge2/single_pro.jsonl")
    i2 = [i for i, r in enumerate(rows) if r["condition"] == C and r["pair_id"] == q2][0]
    r1 = [r for r in rows if r["condition"] == C and r["pair_id"] == q1][0]
    rows[i2] = dict(r1, rule_abstain=False, empty_output=False)
    wl(G + "/judge2/single_pro.jsonl", rows)
elif what == "dup_missing_export":
    ev = d["evaluations"]; i2 = [i for i, e in enumerate(ev) if e["pair_id"] == q2][0]
    ev[i2] = dict([e for e in ev if e["pair_id"] == q1][0]); json.dump(d, open(G + "/judge_input.json", "w"))
elif what == "flag":
    rows = rl(G + "/judge2/single_allenai.jsonl")
    for r in rows:
        if r["condition"] == C and r["pair_id"] == q1: r["rule_abstain"] = not r["rule_abstain"]
    wl(G + "/judge2/single_allenai.jsonl", rows)
elif what == "answer":
    rows = rl(G + "/judge2/cells.jsonl")
    for r in rows:
        if r["condition"] == C and r["pair_id"] == q2: r["prediction"] = r["prediction"] + " (edited)"
    wl(G + "/judge2/cells.jsonl", rows)
elif what == "question":
    for e in d["evaluations"]:
        if e["pair_id"] == q2: e["question"] = e["question"] + " (edited)"
    json.dump(d, open(G + "/judge_input.json", "w"))
elif what == "split":
    d["split"] = "test"; json.dump(d, open(G + "/judge_input.json", "w"))
elif what == "direction":
    [s for s in d["condition_specs"] if s["name"] == C][0]["parts"][0]["direction_file_sha256"] = "0" * 64
    json.dump(d, open(G + "/judge_input.json", "w"))
elif what == "nojudge":
    import os; os.remove(G + "/judge2/single_pro.jsonl"); os.remove(G + "/judge2/single_allenai.jsonl")
print(q1, q2)
PY
}
read -r Q1 Q2 <<< "$(fx none)"
out=$(h "${BF[@]}" 2>&1) && echo "$out" | grep -q "candidates 76 (sha256 $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["binding"]["candidates"]["ids_sha256"][:12])' "$T/cand_g12_tune.json"))" \
  && ok "unchanged copy: same 76 candidates (same set sha256) as the real run" || { bad "fixture"; echo "$out" | tail -n 3; }
fx dup_missing_judged > /dev/null; expect_fail "GPT: q1 twice (blank + answered), q2 missing, 3 rows -> 3 rows (judged file)" "questions twice \['$Q1'\], missing \['$Q2'\]" h "${BF[@]}"
fx dup_missing_export > /dev/null; expect_fail "the same in the export itself" "not every question of the split exactly once" h "${BF[@]}"
fx flag > /dev/null;     expect_fail "a judged row's rule flag disagrees with the raw answer" "flags differ from the raw answer" h "${BF[@]}"
fx answer > /dev/null;   expect_fail "an answer text changed in the cells" "its cell differs from the export" h "${BF[@]}"
fx question > /dev/null; expect_fail "a question text changed in the export" "question text differs from the source run" h "${BF[@]}"
fx split > /dev/null;    expect_fail "an export of another split" "not a completed export of the tune split" h "${BF[@]}"
fx direction > /dev/null; expect_fail "the condition written with another direction file" "with this run's direction file" h "${BF[@]}"
fx nojudge > /dev/null;  expect_fail "no judged rows at all" "has no judged rows" h "${BF[@]}"

echo "== 6. S2: the full expected cell set of a judged work dir (fixture: 3 questions of the run's judged tune export)"
VW=$T/vwork; VWW=$TW/vwork; D=$RUNH/judge_tune_proD_E_L22
mkdir -p "$VW/judge2"
python3 - "$D" "$T" <<'PY'
import json, sys
D, T = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(D + "/judge2/single_pro.jsonl")]
bad = {r["pair_id"] for r in rows if r["status"] != "scored"}
ids = [q for q in json.load(open(D + "/judge_input.json"))["question_ids"] if q not in bad][:3]
json.dump({"pair_ids": ids}, open(T + "/sub3.json", "w"))
PY
h subset "$RUNW/judge_tune_proD_E_L22/judge_input.json" "$TW/sub3.json" "$VWW" --split tune > /dev/null
vx() { python3 - "$D" "$VW" "$T/sub3.json" "$1" <<'PY'
import json, sys
D, V, S, what = sys.argv[1:]
ids = set(json.load(open(S))["pair_ids"])
cells = [json.loads(l) for l in open(D + "/judge2/cells.jsonl") if json.loads(l)["pair_id"] in ids]
rows = [json.loads(l) for l in open(D + "/judge2/single_pro.jsonl") if json.loads(l)["pair_id"] in ids]
other = [json.loads(l) for l in open(D + "/judge2/single_pro.jsonl") if json.loads(l)["pair_id"] not in ids][:1]
if what == "missing": rows = rows[1:]
elif what == "repeated": rows = rows + rows[:1]
elif what == "foreign": rows = rows + other
elif what == "input": rows[0] = dict(rows[0], input_sha="0" * 64)
elif what == "noverdict": rows[0] = dict(rows[0], status="no_verdict", label=None, basis="missing_vote")
elif what == "empty_declined":
    rows[0] = dict(rows[0], status="no_verdict", label=None, basis="empty_output_no_judge_vote", empty_output=True)
    rows[1] = dict(rows[1], status="no_verdict", label=None, basis="pro_declined")
open(V + "/judge2/cells.jsonl", "w").write("".join(json.dumps(c) + "\n" for c in cells))
open(V + "/judge2/single_pro.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
PY
}
vx none;      expect_ok "complete: 3 questions x 14 conditions, each cell once" h verdicts --work $VWW --judge pro --subset $TW/sub3.json
vx missing;   expect_fail "a missing row" "missing \['" h verdicts --work $VWW --judge pro --subset $TW/sub3.json
vx repeated;  expect_fail "a repeated row" "repeated \['" h verdicts --work $VWW --judge pro --subset $TW/sub3.json
vx foreign;   expect_fail "a row of another question" "foreign \['" h verdicts --work $VWW --judge pro --subset $TW/sub3.json
vx input;     expect_fail "a row judging another input" "other input \['" h verdicts --work $VWW --judge pro --subset $TW/sub3.json
vx noverdict; expect_fail "a non-empty answer without verdict" "without verdict \['" h verdicts --work $VWW --judge pro --subset $TW/sub3.json
vx empty_declined; expect_ok "an empty answer and a declined cell are listed, not fatal" h verdicts --work $VWW --judge pro --subset $TW/sub3.json
expect_fail "the work dir holds other questions than SUBSET" "holds other questions than the subset" h verdicts --work $VWW --judge pro --subset $TW/cand_g12_tune.json

echo "== 7. the launcher on a prepared output dir (manifest made by the launcher's own function; no GPU, no DeepSeek)"
OR=$TW/runs; OH=$T/runs/gemma3-12b; OW=$OR/gemma3-12b
eval "$(grep -E '^host\(\) ' "$L")"; eval "$(grep -E '^die\(\) ' "$L")"
eval "$(awk '/^manifest_json\(\) /{p=1} p{print} p && /^}/{exit}' "$L")"
mkdir -p "$OH/launch" "$OH/gen" "$OH/judge_tune_ladder_tune"
( KEY=g12 MODEL=gemma3-12b RUN=$RUNW SRC=$SRCW CODE=$CODE EL=22 E_ON=E2 ES=2 LADDER=$LAD LADH=$(host "$LAD") HELPERH=$P/ladder_extra.py \
  CODEH=$(host "$CODE") SRCH=$(host "$SRCW"); manifest_json 9 ) > "$OH/ladder_manifest.json"
echo '{}' > "$OH/judge_tune_ladder_tune/judge_input.json"
printf '#!/bin/bash\nfor a in "$@"; do case "$a" in -d*|--date*) exec /usr/bin/date "$@" ;; esac; done\nexec /usr/bin/date -d "@${LADDER_TEST_EPOCH:?}" "$@"\n' > "$T/bin/date"; chmod +x "$T/bin/date"
lr() { env PATH=$T/bin:$PATH LADDER_TEST_EPOCH=${EPOCH:-1794999600} KEY=g12 LADDER=$LAD OUT_ROOT=$OR "$@" bash "$L"; }
expect_ok "STAGE=cand SPLIT=tune writes the candidate file (normal path)" lr STAGE=cand SPLIT=tune
CAND=$OW/subsets/tune_E22_x-w-p2_blank.json
python3 -c 'import json,sys; a=json.load(open(sys.argv[1]))["pair_ids"]; b=json.load(open(sys.argv[2]))["pair_ids"]; sys.exit(a != b or len(a) != 76)' "$OH/subsets/tune_E22_x-w-p2_blank.json" "$T/cand_g12_tune.json" \
  && ok "the launcher's candidate file = the helper's (76 ids, same order)" || bad "cand file"
nrec() { ls "$OH"/launch/*.json 2>/dev/null | wc -l; }
EPOCH=1794967200 expect_fail "B2: JUDGES=pro at Wed 10:00 (peak; stubbed clock)" "peak hours" lr STAGE=judge SPLIT=tune SUBSET=$CAND JUDGES=pro
EPOCH=1794967200 expect_fail "B2: JUDGES=\"allenai pro\" at Wed 10:00: refused before AllenAI too" "peak hours" lr STAGE=judge SPLIT=tune SUBSET=$CAND JUDGES="allenai pro"
EPOCH=1794974440 expect_fail "B2: JUDGES=pro at Wed 12:00:40 (window until 13:55)" "at least 2 h" lr STAGE=judge SPLIT=tune SUBSET=$CAND JUDGES=pro
[ "$(nrec)" = 0 ] && ok "no launch record (nor lock) was made by the refused Pro starts" || bad "records after refusals: $(nrec)"
EPOCH=1794999600 expect_fail "B2: JUDGES=pro at Wed 19:00: window open; stops before any paid call (no AllenAI pass yet)" "no AllenAI pass in" lr STAGE=judge SPLIT=tune SUBSET=$CAND JUDGES=pro
r=$(ls "$OH"/launch/*_judge_tune.json 2>/dev/null | head -n 1)
python3 - "$r" <<'PY' && ok "that launch record saved the binding (source export, population and candidate sha256) and the window" || bad "judge record"
import json, sys
j = json.load(open(sys.argv[1]))["judge"]
b = j["candidate_binding"]
assert b["candidates"]["n"] == 76 and b["population"]["n"] == 192 and len(b["population"]["ids_sha256"]) == 64
assert len(b["source"]["export_sha256"]) == 64 and b["subset"]["sha256"] and "2026-11-19 08:55:00" in j["pro_window"]
print("      candidates %d (sha256 %s), population %d (sha256 %s), export sha256 %s, window: %s" % (
    b["candidates"]["n"], b["candidates"]["ids_sha256"][:12], b["population"]["n"], b["population"]["ids_sha256"][:12],
    b["source"]["export_sha256"][:12], j["pro_window"]))
PY
grep -q '"exit": 2' "${r%.json}.end" && [ ! -e "$OH/.lock" ] && ok "its .end has exit 2 and the lock is released" || bad "end/lock"
EPOCH=1795226400 expect_fail "B2: JUDGES=pro on Saturday 10:00 (weekend): window open" "no AllenAI pass in" lr STAGE=judge SPLIT=tune SUBSET=$CAND JUDGES=pro
expect_fail "B2: JUDGES unset = AllenAI only (asks for a GPU, never for Pro)" "STAGE=judge needs GPU" lr STAGE=judge SPLIT=tune SUBSET=$CAND
code_lines() { grep -v '^[[:space:]]*#' "$L"; }     # the launcher without its comment lines
code_lines | grep -c 'judge_two_axis_run.py llm' | grep -qx 1 && code_lines | grep 'judge_two_axis_run.py llm' | grep -q 'timeout --signal=TERM --kill-after=60 "$PRO_LEFT"' \
  && code_lines | grep 'ff_test_subset_judge.sh"$' | grep -q 'JUDGES=allenai' && ok "the one Pro call runs under timeout \$PRO_LEFT; ff_test_subset_judge.sh is only called with JUDGES=allenai" || bad "Pro call sites"
expect_fail "B1: a foreign SUBSET of the right size" "is not this run's tune candidate set" lr STAGE=judge SPLIT=tune SUBSET=$TW/foreign_same_size.json
expect_fail "B1: an empty SUBSET" "is not this run's tune candidate set" lr STAGE=judge SPLIT=tune SUBSET=$TW/empty.json
expect_fail "B1: another run's SUBSET" "is not this run's tune candidate set" lr STAGE=judge SPLIT=tune SUBSET=$TW/cand_llama_tune.json
expect_fail "S2: CACHE_FROM with two dirs" "exactly one work dir" lr STAGE=judge SPLIT=tune SUBSET=$CAND JUDGES=pro CACHE_FROM="$RUNW/judge_tune_proD_E_L22 $RUNW/eval_extra/judge_tune_ec25_tune"
expect_fail "S2: CACHE_FROM without pro" "it needs JUDGES with pro" lr STAGE=judge SPLIT=tune SUBSET=$CAND CACHE_FROM=$RUNW/judge_tune_proD_E_L22
expect_fail "S1: gen without a smoke pass" "no passing smoke" lr STAGE=gen GPU=9
printf '{"passed": true, "manifest_sha256": "%s"}\n' "$(sha256sum "$OH/ladder_manifest.json" | cut -d' ' -f1)" > "$OH/smoke_pass.json"
expect_fail "S1: gen with the smoke pass but another GPU list (8 vs 9)" "a different device mapping needs a fresh OUT_ROOT" lr STAGE=gen GPU=8
expect_fail "S1: gen with the smoke pass and GPU=9: checks pass, stops at the card query (no model touched)" "cannot read GPU 9" lr STAGE=gen GPU=9
mkdir -p "$OH/failed"; echo '{"what": "test"}' > "$OH/failed/control_tune_s2_test.json"
expect_fail "S1: resume after a failed control (RESUME=1)" "this OUT is closed" lr STAGE=gen GPU=9 RESUME=1
expect_fail "S1: smoke after a failed control" "this OUT is closed" lr STAGE=smoke GPU=9

echo "== 8. S3: OUT_ROOT is canonicalized before any check or write"
ln -s "$RUNH/eval_extra" "$T/link_into_run"
before=$(find "$RUNH" -newer "$T/.ref" | wc -l)
expect_fail "OUT_ROOT with .. into a run" "not inside" env KEY=g12 LADDER=$LAD STAGE=dryrun OUT_ROOT=$PW/../results/ff/gemma3/12b/e44/gemma3-12b bash $L
expect_fail "OUT_ROOT through a symlink into a run" "not inside" env KEY=g12 LADDER=$LAD STAGE=dryrun OUT_ROOT=$TW/link_into_run bash $L
expect_fail "OUT_ROOT outside promptladder/" "not inside" env KEY=g12 LADDER=$LAD STAGE=dryrun OUT_ROOT=$W/Iteration_01/results/promptladder bash $L
expect_fail "OUT_ROOT = promptladder/ itself" "not inside" env KEY=g12 LADDER=$LAD STAGE=dryrun OUT_ROOT=$PW bash $L
after=$(find "$RUNH" -newer "$T/.ref" | wc -l)
[ "$before" = 0 ] && [ "$after" = 0 ] && [ ! -e "$W/Iteration_01/results/promptladder" ] && ok "nothing was created in the run or in results/" || bad "files appeared ($before / $after)"

echo "== 9. launcher refusals kept"
expect_fail "unknown KEY" "unknown KEY" env KEY=nope LADDER=$LAD STAGE=gen bash $L
expect_fail "bad strength" "is not a number" env KEY=g12 LADDER=$LAD STAGE=gen ES=2x bash $L
expect_fail "strength without driver file" "no driver file" env KEY=g12 LADDER=$LAD STAGE=gen ES=3 bash $L
expect_fail "GPU stage without GPU" "needs GPU" env KEY=g12 LADDER=$LAD STAGE=smoke OUT_ROOT=$TW/runs2 bash $L
G1=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | head -n 1)
GI=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$G1" | cut -d, -f1)
if [ -n "$GI" ]; then expect_fail "busy GPU $GI refused" "is busy" env KEY=g12 LADDER=$LAD STAGE=smoke GPU=$GI OUT_ROOT=$TW/runs2 bash $L; fi
expect_fail "judge before smoke/gen (no manifest)" "run STAGE=smoke first" env KEY=g12 LADDER=$LAD STAGE=judge SPLIT=tune SUBSET=$CAND JUDGES=pro OUT_ROOT=$TW/runs2 bash $L
[ ! -e "$T/runs2" ] && ok "no output dir was created by the refused launches" || bad "an output dir was created"

n=0
for d in "$RUNH" "$R/results_e44_factorial_das/llama3.1-8b_side_E2" "$R/Iteration_01/code" "$R/code_e45_20261112" "$R/code_ff_explore_20261113b"; do
  n=$((n + $(find "$d" -newer "$T/.ref" 2>/dev/null | wc -l)))
done
[ "$n" = 0 ] && ok "no file in the two runs read here or in the three code copies is newer than the test start" || bad "$n entries changed"
echo "info: entries of Iteration_01/diag newer than the test start (other sessions write there; this test writes only into $T):"
find "$R/Iteration_01/diag" -newer "$T/.ref" -printf '      %TH:%TM:%TS %p\n' 2>/dev/null | head -n 10
echo "== $pass passed, $fail failed"
rm -rf "$T"
[ "$fail" = 0 ]
