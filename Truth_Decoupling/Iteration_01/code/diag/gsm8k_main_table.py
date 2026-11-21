# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# GSM8K under the settings of the main table (launcher explore_20261120/gsm8k_main.sh): per model a full pass with
# nothing written and one with single E at the model's main-table layer and strength, on the 1,319 frozen test items,
# system prompt "neutral", 512 new tokens, batch 8 (the generator settings fixed in
# novelty/ours/2026-11-19_capability_preservation_design.md). Rule scoring with diag/cap_score.py's own functions
# (gold = the number after "####"; the parsed final number of the answer; no pick = no answer), so the state of an item
# is R correct, W answered but wrong, X no answer; all 1,319 items stay in the denominator.
# Printed: accuracy and no-answer rate of each pass, the paired change against no write with a bootstrap interval, and
# the wording of the design's evidence rule (kept / damage cannot be excluded / damage; lines at -1 point of accuracy
# and +3 points of no answer). N1's re-asks are not part of this run: they touch only the items BLANK under E (the rule
# detector's abstention or an empty answer; an answer cut off at the token cap is not blank), whose number is printed
# next to the number of all items without an answer. The reasons of "no answer" are printed per pass (refusal = the rule detector's abstention or an
# empty answer; truncated = the 512-token cap was reached without a final answer; the others are the parser's names),
# because a change in answer length moves the truncated count without any change in ability.
# usage (host, from /data/sanshizhao/Truth_Decoupling, inside the container for pyarrow):
#   docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/gsm8k_main_table.py [--cap N] [MODEL ...]
# --cap N reads the passes generated with another token cap (launcher variable NEWTOK; directory capability/gen_main<N>).
# A pass written at another position rule (launchers/explore_20261120/gsm8k_pos.sh) is shown as ARM@POS.
import glob
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import cap_score as CS

ROOT = "Iteration_01/capability/gen_main/gsm8k"
IDS = "Iteration_01/capability/sets/gsm8k_1319_20261119.json"


def load_pass(path, model, ids, cap):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    c = g.get("contract") or {}
    if not (g.get("completed") and c.get("full_pass") and c.get("model") == model and c.get("task") == "gsm8k"
            and c.get("max_new_tokens") == cap and c.get("batch_size") == 8 and [r["id"] for r in g["rows"]] == ids):
        raise SystemExit("%s is not a complete full pass of %s over the frozen GSM8K list at cap %d, batch 8" % (path, model, cap))
    return g


def reasons(judged):
    c = {}
    for st, w in judged.values():
        if st == "X":
            c[w] = c.get(w, 0) + 1
    return ", ".join("%s %d" % kv for kv in sorted(c.items(), key=lambda kv: -kv[1])) or "none"


def main():
    with open(IDS, encoding="utf-8") as f:
        ids = json.load(f)["ids"]
    gold = CS.gold_of("gsm8k")
    argv, root, cap = sys.argv[1:], ROOT, 512
    if "--cap" in argv:
        k = argv.index("--cap")
        if k + 1 >= len(argv) or not argv[k + 1].isdigit():
            raise SystemExit("--cap needs a number")
        cap, argv = int(argv[k + 1]), argv[:k] + argv[k + 2:]
        if cap != 512:
            root = "Iteration_01/capability/gen_main%d/gsm8k" % cap
    models = argv or sorted(os.path.basename(os.path.dirname(d)) for d in glob.glob(root + "/*/b8"))
    print("# GSM8K, %d items, rule scoring, token cap %d; changes are against the model's own no-write pass" % (len(ids), cap))
    for model in models:
        d = "%s/%s/b8" % (root, model)
        files = [p for p in sorted(glob.glob(d + "/*.json")) if not os.path.basename(p).startswith("s")]
        passes, why, ntok, blank = {}, {}, {}, {}
        for p in files:
            try:
                g = load_pass(p, model, ids, cap)
            except SystemExit as e:
                print("\n## %s: %s" % (model, e))
                continue
            judged = {r["id"]: CS.judge_row("gsm8k", r, gold(r["id"])) for r in g["rows"]}
            pos = g["contract"].get("positions")
            key = g["contract"]["arm"] + ("@" + pos if pos not in (None, "content") else "")   # another write rule (gsm8k_pos.sh)
            if key in passes:
                raise SystemExit("%s: two files for %s" % (model, key))
            passes[key] = {i: v[0] for i, v in judged.items()}
            why[key] = reasons(judged)
            blank[key] = sum(1 for st_, w_ in judged.values() if st_ == "X" and w_ == "refusal")
            ntok[key] = sum(not r["finished"] for r in g["rows"])
        if "neutral" not in passes:
            print("\n## %s: no complete no-write pass yet (%d files)" % (model, len(files)))
            continue
        base = passes["neutral"]
        n = len(ids)
        print("\n## %s: no write: accuracy %.1f%%, no answer %.1f%% (%s; %d answers reached the token cap)" % (
            model, 100.0 * sum(base[i] == "R" for i in ids) / n, 100.0 * sum(base[i] == "X" for i in ids) / n,
            why["neutral"], ntok["neutral"]))
        for arm, st in passes.items():
            if arm == "neutral":
                continue
            da = CS.ci([(st[i] == "R") - (base[i] == "R") for i in ids])
            dn = CS.ci([(st[i] == "X") - (base[i] == "X") for i in ids])
            print("    %-22s accuracy %.1f%% (change %+.1f [%+.1f, %+.1f]) | no answer %.1f%% (change %+.1f [%+.1f, %+.1f]; %d items, of which %d blank = what N1 would re-ask) | right->wrong %d, wrong->right %d | %s" % (
                arm, 100.0 * sum(st[i] == "R" for i in ids) / n, 100 * da[0], 100 * da[1], 100 * da[2],
                100.0 * sum(st[i] == "X" for i in ids) / n, 100 * dn[0], 100 * dn[1], 100 * dn[2], sum(st[i] == "X" for i in ids), blank[arm],
                sum(base[i] == "R" and st[i] == "W" for i in ids), sum(base[i] == "W" and st[i] == "R" for i in ids), CS.evidence(da, dn)))
            both = [i for i in ids if base[i] != "X" and st[i] != "X"]
            if both:
                db = CS.ci([(st[i] == "R") - (base[i] == "R") for i in both])
                print("    %-22s no answer by reason: %s; %d answers reached the token cap | on the %d items answered in both passes: accuracy change %+.1f [%+.1f, %+.1f]" % (
                    "", why[arm], ntok[arm], len(both), 100 * db[0], 100 * db[1], 100 * db[2]))


if __name__ == "__main__":
    main()
