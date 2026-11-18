# Iteration_01 analysis (Claude Opus 5.5, 2026-11-19), read-only, no model, no judge call.
# Work queue F1, capability check (novelty/ours/2026-11-19_capability_preservation_design.md). Rule scoring of the
# generations of scripts/newsets/newsets_gen.py on MMLU (letter) and GSM8K (final number), with scripts/capability/
# cap_prompts.py. N1 uses the per-question records of diag/final_records.py UNCHANGED: condition names of the GSM8K base
# prompt "neutral" are read as "x" (the base prompt's name plays no part in the logic), so the ladder, the two caps and the
# veto of E's blanks are exactly the TruthfulQA rules. State per item: R correct, W answered but wrong, X no letter /
# number (abstained or unparseable); blank (what N1 re-asks) = the rule detector's abstention or an empty answer.
# Criteria (design section 7): accuracy more than 1 point below no write, or the no-answer rate more than 3 points above,
# is reported as damage.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/cap_score.py mmlu|gsm8k [--batch 8] [MODEL ...]
#                                                      python3 -B Iteration_01/diag/cap_score.py --selftest
import glob
import json
import os
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/capability")
import policy_select2 as ps   # noqa: E402
import final_records as fr    # noqa: E402
import cap_prompts as CP      # noqa: E402

CFG = "Iteration_01/code/configs/newsets/frozen_models_20261119.json"
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
LADNAME = {0.25: "0-25", 0.5: "0-5", 1: "1", 2: "2"}
DATA = {"mmlu": "data_external/mmlu_20261119/all/test-00000-of-00001.parquet",
        "gsm8k": "data_external/gsm8k_20261119/main/test-00000-of-00001.parquet"}
BASE = {"mmlu": "x", "gsm8k": "neutral"}
N_BOOT, SEED = 4000, 20261119


class CapSplit(ps.Split):
    def __init__(self, el, cl, st, blank, qs):
        self.m = dict(e=el, c=cl, order="EC")
        self.st, self.blank, self.dst, self.dblank = st, blank, None, None
        self.cks = ["2"]
        self.qs = list(qs)
        self.n_dropped = 0


def gold_of(task):
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA[task]).to_pylist()
    if task == "mmlu":
        per = {}
        for r in rows:
            per.setdefault(r["subject"], []).append(r)
        return lambda i: "ABCD"[per[i.split(":")[1]][int(i.split(":")[2])]["answer"]]
    return lambda i: CP.gsm8k_gold(rows[int(i.split(":")[1])]["answer"])


def state(task, text, gold):
    if task == "mmlu":
        pick = CP.mmlu_pick(text)
        return "X" if pick is None else ("R" if pick == gold else "W")
    pick = CP.gsm8k_pick(text)
    return "X" if pick is None else ("R" if abs(pick - gold) < 1e-6 else "W")


def load(task, model, batch):
    """(st, blank, texts, item order, re-asked items) with conditions renamed to the x base prompt."""
    d = "Iteration_01/capability/gen/%s/%s/b%d" % (task, model, batch)
    gold = gold_of(task)
    st, blank, tx, order, reask = {}, {}, {}, None, set()
    for p in sorted(glob.glob(d + "/*.json")):
        name = os.path.basename(p)
        if name == "n1_blank_ids.json":
            continue
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        if g.get("task") != task or g.get("batch_size") != batch:
            raise SystemExit("%s: other task or batch size" % name)
        cond = g["condition"].replace("_%s-" % BASE[task], "_x-").replace("X_%s--" % BASE[task], "X_x--")
        if cond == "baseline_" + BASE[task]:
            cond = "baseline_x"
        if g["full_pass"]:
            ids = [r["id"] for r in g["rows"]]
            if order is not None and ids != order:
                raise SystemExit("full passes cover different items")
            order = ids
        else:
            reask |= {r["id"] for r in g["rows"]}
        for r in g["rows"]:
            k = (r["id"], cond)
            st[k] = state(task, r["text"], gold(r["id"]))
            blank[k] = bool(r["rule_abstain"]) or not r["text"].strip()
            tx[k] = r["text"]
    if order is None:
        raise SystemExit("no full pass in %s" % d)
    return st, blank, tx, order, reask


def ci(v):
    rng = random.Random(SEED)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return sum(v) / n, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]


def report(task, names, batch):
    with open(CFG, encoding="utf-8") as f:
        M = json.load(f)["models"]
    print("capability check %s, batch %d: accuracy and no-answer rate per condition; differences vs no write (95%%, items resampled)" % (task, batch))
    for name in names:
        m = M[name]
        st, blank, tx, order, reask = load(task, name, batch)
        sp = CapSplit(m["el"], m["cl"], st, blank, order)
        t = fr.Texts(tx, tx)
        k1, k2 = LADNAME[m["caps"]["e_blank"]], LADNAME[m["caps"]["own_blank"]]
        cand = [p for p in order if blank[(p, sp.cE("2"))]]
        if set(cand) - reask:
            raise SystemExit("%s: %d items blank under E were not re-asked" % (name, len(set(cand) - reask)))
        cfgs = [("no write", ("base",)), ("E 2", ("E", "2")), ("N1 (caps own %s / E's %s)" % (k2, k1), ("N1", "2", k1, k2)),
                ("C alone at 2 (toward answering)", ("C", "2"))]
        base = fr.states(fr.records(sp, ("base",), t))
        print("\n== %s %s: %d items; blank under E: %d (re-asked with E + C)" % (task, name, len(order), len(cand)))
        for label, cfg in cfgs:
            s = fr.states(fr.records(sp, cfg, t))
            acc = sum(x == "R" for x in s) / len(s)
            na = sum(x == "X" for x in s) / len(s)
            if cfg[0] == "base":
                print("    %-34s accuracy %.3f | no answer %.3f" % (label, acc, na))
                continue
            da = ci([(x == "R") - (y == "R") for x, y in zip(s, base)])
            dn = ci([(x == "X") - (y == "X") for x, y in zip(s, base)])
            damage = da[0] < -0.01 or dn[0] > 0.03
            print("    %-34s accuracy %.3f (%+.1f [%+.1f, %+.1f] points) | no answer %.3f (%+.1f [%+.1f, %+.1f]) | %s" % (
                label, acc, 100 * da[0], 100 * da[1], 100 * da[2], na, 100 * dn[0], 100 * dn[1], 100 * dn[2],
                "DAMAGE by the criteria" if damage else "within the criteria"))


def selftest():
    # 3 items, E layer 11, C layer 19: i1 answered by E; i2 E's blank, ladder gives back the no-write answer (vetoed);
    # i3 own blank released at C 0.25 with a correct letter
    st, blank, tx = {}, {}, {}

    def put(p, c, text, s):
        st[(p, c)], blank[(p, c)], tx[(p, c)] = s, text == "I don't know.", text
    sp = CapSplit(11, 19, st, blank, ["i1", "i2", "i3"])
    for p, base, e in (("i1", ("B", "R"), ("B", "R")), ("i2", ("C", "W"), ("I don't know.", "X")),
                       ("i3", ("I don't know.", "X"), ("I don't know.", "X"))):
        put(p, "baseline_x", *base)
        put(p, sp.cE("2"), *e)
        put(p, sp.cC("2"), "A", "W")
    for k in fr.LAD:
        put("i2", sp.cEC("2", k), "C", "W")
        put("i3", sp.cEC("2", k), "D", "R")
    t = fr.Texts(tx, tx)
    assert fr.states(fr.records(sp, ("N1", "2", "0-25", "0-25"), t)) == ["R", "X", "R"]
    assert fr.states(fr.records(sp, ("E", "2"), t)) == ["R", "X", "X"]
    assert state("mmlu", "B", "B") == "R" and state("mmlu", "I don't know.", "B") == "X" and state("mmlu", "C", "B") == "W"
    assert state("gsm8k", "The answer is 18", 18.0) == "R" and state("gsm8k", "no idea", 18.0) == "X"
    m, lo, hi = ci([1, 0, 1, 1])
    assert abs(m - 0.75) < 1e-12 and lo <= m <= hi
    print("selftest PASS")


def main():
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
        return
    if not a or a[0] not in DATA:
        raise SystemExit("usage: cap_score.py mmlu|gsm8k [--batch 8] [MODEL ...]")
    task, rest = a[0], a[1:]
    batch = 8
    if rest[:1] == ["--batch"]:
        batch, rest = int(rest[1]), rest[2:]
    report(task, rest or list(ORDER), batch)


if __name__ == "__main__":
    main()
