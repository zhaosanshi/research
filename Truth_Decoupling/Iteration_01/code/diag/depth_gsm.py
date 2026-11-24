# Iteration_01 (main session, Claude Fable 5.2, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_n.md, progress
# entry of 19 h): is single E at the late layers of the depth scan inside the capability budget on a free-form task?
# GSM8K on the first 304 items of the frozen list (the first 38 batches of the full pass at batch 8), single E at the
# layer / strength pairs the depth scan chose on MMLU, written at the core rule "prompt", cap 512 new tokens.
# Read-only, no model, no judge call. The no-write reference is the archived full pass (gen_main, the same batches);
# the archived full pass of the main-table arm (E16 at 3.5, prompt rule) is shown on the same 304 items as a reference.
# Scoring: cap_score.judge_row (the frozen GSM8K parser); a refusal or an unparsed answer counts as not right.
# modes (from /workspace/Truth_Decoupling in the scoring container, or the host if pyarrow is there):
#   python3 -B Iteration_01/diag/depth_gsm.py ids          write the id list of the first 304 items (never overwrites)
#   python3 -B Iteration_01/diag/depth_gsm.py table MODEL  accuracy per arm and its change against no write
import glob
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")

N = 304
FULL_IDS = "Iteration_01/capability/sets/gsm8k_1319_20261119.json"
SUB_IDS = "Iteration_01/capability/sets/gsm8k_first304_20261124.json"
MAIN = "Iteration_01/capability/%s/gsm8k/%s/b8"
# (cap of new tokens, directory of the archived full passes): the cap each model was measured at in the main table
CAPS = {"llama3.1-8b": (512, "gen_main"), "qwen3.5-9b": (1024, "gen_main1024")}
OUT = "Iteration_01/capability/depthscan_gsm8k/%s/b8"
REF_ARM = {"llama3.1-8b": "neutral--E16-w-p3-5-at-prompt.json", "qwen3.5-9b": "neutral--E15-w-p2-at-prompt.json"}


def die(msg):
    raise SystemExit("depth_gsm: " + msg)


def write_ids():
    if os.path.exists(SUB_IDS):
        die("%s exists" % SUB_IDS)
    d = json.load(open(FULL_IDS, encoding="utf-8"))
    sub = dict(d, ids=d["ids"][:N], rows=N)
    with open(SUB_IDS, "w", encoding="utf-8") as f:
        json.dump(sub, f)
    print("wrote %s: %d ids, %s .. %s" % (SUB_IDS, len(sub["ids"]), sub["ids"][0], sub["ids"][-1]))
    return 0


def load(path, model, ids, positions, full, cap):
    g = json.load(open(path, encoding="utf-8"))
    c = g.get("contract") or {}
    bad = []
    if g.get("completed") is not True:
        bad.append("not completed")
    if c.get("model") != model or c.get("task") != "gsm8k" or c.get("max_new_tokens") != cap or c.get("batch_size") != 8:
        bad.append("contract model / task / cap / batch: %s %s %s %s" % (c.get("model"), c.get("task"), c.get("max_new_tokens"), c.get("batch_size")))
    if g.get("positions") != positions or c.get("positions") != positions:
        bad.append("positions %s / %s, wanted %s" % (g.get("positions"), c.get("positions"), positions))
    got = [r["id"] for r in g["rows"]]
    if (got[:len(ids)] if full else got) != ids:
        bad.append("rows are not the wanted items in order (%d rows)" % len(got))
    if bad:
        die("%s: %s" % (path, "; ".join(bad)))
    return {r["id"]: r for r in g["rows"]}, g.get("arm")


def table(model):
    import cap_score as CS
    ids = json.load(open(SUB_IDS, encoding="utf-8"))["ids"]
    if len(ids) != N or ids != json.load(open(FULL_IDS, encoding="utf-8"))["ids"][:N]:
        die("%s is not the first %d ids of the frozen list" % (SUB_IDS, N))
    gold = CS.gold_of("gsm8k")
    if model not in CAPS:
        die("no cap registered for %s" % model)
    cap, mdir = CAPS[model]
    main = MAIN % (mdir, model)
    base, _ = load(main + "/neutral.json", model, ids, None, True, cap)
    def states(rows):
        return [CS.judge_row("gsm8k", rows[i], gold(i)) for i in ids]
    sb = states(base)
    rb = [int(s == "R") for s, _ in sb]
    print("# Plan N, GSM8K on the first %d frozen items, %s: single E at the prompt rule, cap %d, batch 8 (depth_gsm.py)" % (N, model, cap))
    print("Reference: the archived no-write full pass on the same items (same batches). Change = arm minus no write, in points;")
    print("interval: bootstrap over items (cap_score.ci). 'not right' = wrong, refusal or unparsed.")
    print("| arm | source | right of %d | accuracy | change against no write [95%%] | refusals | unparsed / unfinished | mean new tokens |" % N)
    print("|---|---|---|---|---|---|---|---|")
    def row(name, source, rows):
        st = states(rows)
        r = [int(s == "R") for s, _ in st]
        m, lo, hi = CS.ci([a - b for a, b in zip(r, rb)])
        ref = sum(1 for s, w in st if s == "X" and w == "refusal")
        unp = sum(1 for s, w in st if s == "X" and w != "refusal")
        tok = sum(rows[i]["n_tokens"] for i in ids) / float(N)
        print("| %s | %s | %d | %.1f%% | %+.1f [%+.1f, %+.1f] | %d | %d | %.0f |" % (
            name, source, sum(r), 100.0 * sum(r) / N, 100 * m, 100 * lo, 100 * hi, ref, unp, tok))
    row("no write", "archived full pass", base)
    rf = main + "/" + REF_ARM.get(model, "")
    if REF_ARM.get(model) and os.path.exists(rf):
        rows, arm = load(rf, model, ids, "prompt", True, cap)
        row(arm, "archived full pass", rows)
    files = sorted(glob.glob((OUT % model) + "/neutral--E*-at-prompt.json"),
                   key=lambda p: [int(x) if x.isdigit() else x for x in os.path.basename(p).replace("-", " ").split()])
    if not files:
        die("no arm files in %s" % (OUT % model))
    for p in files:
        rows, arm = load(p, model, ids, "prompt", False, cap)
        row(arm, "this run", rows)
    return 0


def main():
    a = sys.argv[1:]
    if a == ["ids"]:
        return write_ids()
    if len(a) == 2 and a[0] == "table":
        return table(a[1])
    die("usage: depth_gsm.py ids | table MODEL")


if __name__ == "__main__":
    sys.exit(main())
