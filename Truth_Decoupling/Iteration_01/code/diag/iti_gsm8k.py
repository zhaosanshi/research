# Iteration_01 analysis (Claude Fable 5.2, 2026-11-22), read-only, no model, no judge call. Exploration.
# GSM8K (1,319 items, rule scoring with diag/cap_score.py's own functions, token cap 512, batch 8, neutral system
# prompt) under the ITI baseline (scripts/newsets/iti_baseline.py gsm8k) against the model's usual no-write pass, with
# the wording of the capability design's evidence rule (lines at -1 point of accuracy and +3 points of no answer), as
# diag/gsm8k_main_table.py prints it for the project's own direction. For reference the project's single E at the
# main-table setting, written at "prompt" and at the training rule (every generated token too), is printed when the
# files are there.
# usage (in the scoring container, from the project root, for pyarrow):
#   docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/iti_gsm8k.py MODEL FIT STEM [STEM ...]
import glob
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import cap_score as CS

IDS = "Iteration_01/capability/sets/gsm8k_1319_20261119.json"
ROOT = "Iteration_01/capability/gen_main/gsm8k"
BASE = "Iteration_01/baselines/iti"


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def reasons(judged):
    c = {}
    for st, w in judged.values():
        if st == "X":
            c[w] = c.get(w, 0) + 1
    return ", ".join("%s %d" % kv for kv in sorted(c.items(), key=lambda kv: -kv[1])) or "none"


def line(name, judged, base, ids, rows):
    n = len(ids)
    da = CS.ci([(judged[i][0] == "R") - (base[i][0] == "R") for i in ids])
    dn = CS.ci([(judged[i][0] == "X") - (base[i][0] == "X") for i in ids])
    print("    %-34s accuracy %.1f%% (change %+.1f [%+.1f, %+.1f]) | no answer %.1f%% (change %+.1f [%+.1f, %+.1f]; %s) | %d answers reached the token cap | right->wrong %d, wrong->right %d | %s" % (
        name, 100.0 * sum(judged[i][0] == "R" for i in ids) / n, 100 * da[0], 100 * da[1], 100 * da[2],
        100.0 * sum(judged[i][0] == "X" for i in ids) / n, 100 * dn[0], 100 * dn[1], 100 * dn[2], reasons(judged),
        sum(not r["finished"] for r in rows),
        sum(base[i][0] == "R" and judged[i][0] == "W" for i in ids), sum(base[i][0] == "W" and judged[i][0] == "R" for i in ids),
        CS.evidence(da, dn)))


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: iti_gsm8k.py MODEL FIT STEM [STEM ...]")
    model, fit, stems = sys.argv[1], sys.argv[2], sys.argv[3:]
    ids = read(IDS)["ids"]
    gold = CS.gold_of("gsm8k")
    g0 = read("%s/%s/b8/neutral.json" % (ROOT, model))
    c0 = g0.get("contract") or {}
    if not (g0.get("completed") and c0.get("full_pass") and c0.get("model") == model and c0.get("max_new_tokens") == 512
            and c0.get("batch_size") == 8 and [r["id"] for r in g0["rows"]] == ids):
        raise SystemExit("the usual no-write pass is not the complete 512-token, batch-8 pass of this model")
    base = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"])) for r in g0["rows"]}
    n = len(ids)
    print("# GSM8K, %d items, rule scoring, token cap 512; %s: no write accuracy %.1f%%, no answer %.1f%% (%s); changes are against this pass" % (
        n, model, 100.0 * sum(base[i][0] == "R" for i in ids) / n, 100.0 * sum(base[i][0] == "X" for i in ids) / n, reasons(base)))
    for stem in stems:
        g = read("%s/%s/%s/gsm8k/%s.json" % (BASE, model, fit, stem))
        st = g["setting"]
        if not (g.get("protocol") == "iti_baseline_v1" and g.get("phase") == "gsm8k" and g.get("completed") is True
                and g.get("model") == model and g.get("max_new_tokens") == 512 and g.get("batch_size") == 8
                and g["ids"] == ids and [r["id"] for r in g["rows"]] == ids
                and g.get("ids_file_sha256") == CS.sha_file(IDS) and g.get("system_text") == c0.get("system_text")):
            raise SystemExit("%s is not a complete GSM8K pass of this model under the same prompt and cap" % stem)
        judged = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"])) for r in g["rows"]}
        line("ITI K %d alpha %g (%s sigma)" % (st["k"], st["alpha"], st["sigma"]), judged, base, ids, g["rows"])
    for p in sorted(glob.glob("%s/%s/b8/neutral--E*.json" % (ROOT, model))):
        g = read(p)
        c = g.get("contract") or {}
        if not (g.get("completed") and c.get("full_pass") and c.get("max_new_tokens") == 512 and [r["id"] for r in g["rows"]] == ids):
            continue
        judged = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"])) for r in g["rows"]}
        line("the project's %s at %s" % (c["arm"].split(":")[1], c.get("positions") or "content"), judged, base, ids, g["rows"])
    print("(file names: %s)" % ", ".join(os.path.basename(p) for p in sorted(glob.glob("%s/%s/b8/neutral--E*.json" % (ROOT, model)))))


if __name__ == "__main__":
    main()
