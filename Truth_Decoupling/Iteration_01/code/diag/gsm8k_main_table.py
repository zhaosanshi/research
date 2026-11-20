# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# GSM8K under the settings of the main table (launcher explore_20261120/gsm8k_main.sh): per model a full pass with
# nothing written and one with single E at the model's main-table layer and strength, on the 1,319 frozen test items,
# system prompt "neutral", 512 new tokens, batch 8 (the generator settings fixed in
# novelty/ours/2026-11-19_capability_preservation_design.md). Rule scoring with diag/cap_score.py's own functions
# (gold = the number after "####"; the parsed final number of the answer; no pick = no answer), so the state of an item
# is R correct, W answered but wrong, X no answer; all 1,319 items stay in the denominator.
# Printed: accuracy and no-answer rate of each pass, the paired change against no write with a bootstrap interval, and
# the wording of the design's evidence rule (kept / damage cannot be excluded / damage; lines at -1 point of accuracy
# and +3 points of no answer). N1's re-asks are not part of this run: they touch only the items blank under E, whose
# number is printed.
# usage (host, from /data/sanshizhao/Truth_Decoupling, inside the container for pyarrow):
#   docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/gsm8k_main_table.py [MODEL ...]
import glob
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import cap_score as CS

ROOT = "Iteration_01/capability/gen_main/gsm8k"
IDS = "Iteration_01/capability/sets/gsm8k_1319_20261119.json"


def load_pass(path, model, ids):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    c = g.get("contract") or {}
    if not (g.get("completed") and c.get("full_pass") and c.get("model") == model and c.get("task") == "gsm8k"
            and [r["id"] for r in g["rows"]] == ids):
        raise SystemExit("%s is not a complete full pass of %s over the frozen GSM8K list" % (path, model))
    return g


def main():
    with open(IDS, encoding="utf-8") as f:
        ids = json.load(f)["ids"]
    gold = CS.gold_of("gsm8k")
    models = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(d)) for d in glob.glob(ROOT + "/*/b8"))
    print("# GSM8K, %d items, rule scoring; changes are against the model's own no-write pass" % len(ids))
    for model in models:
        d = "%s/%s/b8" % (ROOT, model)
        files = [p for p in sorted(glob.glob(d + "/*.json")) if not os.path.basename(p).startswith("s")]
        passes = {}
        for p in files:
            try:
                g = load_pass(p, model, ids)
            except SystemExit as e:
                print("\n## %s: %s" % (model, e))
                continue
            passes[g["contract"]["arm"]] = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"]))[0] for r in g["rows"]}
        if "neutral" not in passes:
            print("\n## %s: no complete no-write pass yet (%d files)" % (model, len(files)))
            continue
        base = passes["neutral"]
        n = len(ids)
        print("\n## %s: no write: accuracy %.1f%%, no answer %.1f%%" % (
            model, 100.0 * sum(base[i] == "R" for i in ids) / n, 100.0 * sum(base[i] == "X" for i in ids) / n))
        for arm, st in passes.items():
            if arm == "neutral":
                continue
            da = CS.ci([(st[i] == "R") - (base[i] == "R") for i in ids])
            dn = CS.ci([(st[i] == "X") - (base[i] == "X") for i in ids])
            print("    %-22s accuracy %.1f%% (change %+.1f [%+.1f, %+.1f]) | no answer %.1f%% (change %+.1f [%+.1f, %+.1f]; %d items blank) | right->wrong %d, wrong->right %d | %s" % (
                arm, 100.0 * sum(st[i] == "R" for i in ids) / n, 100 * da[0], 100 * da[1], 100 * da[2],
                100.0 * sum(st[i] == "X" for i in ids) / n, 100 * dn[0], 100 * dn[1], 100 * dn[2], sum(st[i] == "X" for i in ids),
                sum(base[i] == "R" and st[i] == "W" for i in ids), sum(base[i] == "W" and st[i] == "R" for i in ids), CS.evidence(da, dn)))


if __name__ == "__main__":
    main()
