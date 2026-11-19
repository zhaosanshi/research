# Iteration_01 analysis (Claude Opus 5.5, 2026-11-19), read-only, no model, no judge call.
# Work queue F1, capability check (novelty/ours/2026-11-19_capability_preservation_design.md). Rule scoring of the
# generations of scripts/newsets/newsets_gen.py on MMLU (letter) and GSM8K (final number), with scripts/capability/
# cap_prompts.py (mmlu_parse / gsm8k_parse). N1 uses the per-question records of diag/final_records.py UNCHANGED:
# condition names of the GSM8K base prompt "neutral" are read as "x" (the base prompt's name plays no part in the logic),
# so the ladder, the two caps and the veto of E's blanks are exactly the TruthfulQA rules.
# State per item: R correct, W answered but wrong, X no pick; every X keeps its reason (refusal = the rule detector's
# abstention, truncated, negated, several, none); all items stay in the accuracy denominator.
# Inputs are checked before anything is counted (GPT review 2026-11-19, section 2 last row): every full pass covers the
# frozen ID list in order, for this model, task, batch size and token cap, completed; the re-ask files cover exactly the
# items blank under E (rule detector or empty), from that same E file, for all four C strengths.
# Evidence (design section 7; GPT review section 4): "kept" only when the 95% interval of the accuracy change lies above
# -1 point AND that of the no-answer change below +3 points; a point estimate beyond a line is "damage"; otherwise
# "damage cannot be excluded".
# usage (host or container, from the project root): python3 -B Iteration_01/diag/cap_score.py mmlu|gsm8k [--batch 8] [MODEL ...]
#                                                   python3 -B Iteration_01/diag/cap_score.py --selftest
import glob
import hashlib
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
IDS = {"mmlu": "Iteration_01/capability/sets/mmlu_1000_20261119.json",
       "gsm8k": "Iteration_01/capability/sets/gsm8k_1319_20261119.json"}
BASE = {"mmlu": "x", "gsm8k": "neutral"}
NEW = {"mmlu": 16, "gsm8k": 512}
N_BOOT, SEED = 4000, 20261119


class CapSplit(ps.Split):
    def __init__(self, el, cl, st, blank, qs):
        self.m = dict(e=el, c=cl, order="EC")
        self.st, self.blank, self.dst, self.dblank = st, blank, None, None
        self.cks = ["2"]
        self.qs = list(qs)
        self.n_dropped = 0


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def gold_of(task):
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA[task]).to_pylist()
    if task == "mmlu":
        per = {}
        for r in rows:
            per.setdefault(r["subject"], []).append(r)
        return lambda i: "ABCD"[per[i.split(":")[1]][int(i.split(":")[2])]["answer"]]
    return lambda i: CP.gsm8k_gold(rows[int(i.split(":")[1])]["answer"])


def judge_row(task, r, gold):
    """(state, reason) of one generated row."""
    if r["rule_abstain"] or not r["text"].strip():
        return "X", "refusal"
    if task == "mmlu":
        pick, why = CP.mmlu_parse(r["text"])
        return ("X", why) if pick is None else ("R" if pick == gold else "W", why)
    pick, why = CP.gsm8k_parse(r["text"], bool(r["finished"]))
    return ("X", why) if pick is None else ("R" if abs(pick - gold) < 1e-6 else "W", why)


def rename(task, cond):
    b = BASE[task]
    if cond == "baseline_" + b:
        return "baseline_x"
    return cond.replace("_%s-" % b, "_x-").replace("X_%s--" % b, "X_x--")


def load(task, model, batch, m, gold=None):
    """(st, blank, texts, reasons, item order, re-asked items); stops on any input that does not match the plan."""
    d = "Iteration_01/capability/gen/%s/%s/b%d" % (task, model, batch)
    gold = gold or gold_of(task)
    with open(IDS[task], encoding="utf-8") as f:
        frozen = json.load(f)["ids"]
    e_cond = "E_L%d_x-w-p2" % m["el"]
    full_need = {"baseline_x", e_cond, "C_L%d_x-w-m2" % m["cl"]}
    lad_need = {"X_x--E%d-w-p2-and-C%d-w-m%s" % (m["el"], m["cl"], k) for k in fr.LAD}
    st, blank, tx, why, full, reask = {}, {}, {}, {}, {}, {}
    for p in sorted(glob.glob(d + "/*.json")):
        if os.path.basename(p) == "n1_blank_ids.json":
            continue
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        name = os.path.basename(p)
        if (g.get("task"), g.get("model"), g.get("batch_size"), g.get("max_new_tokens"), g.get("completed")) != (task, model, batch, NEW[task], True):
            raise SystemExit("%s: other task, model, batch size or token cap, or not completed" % name)
        cond = rename(task, g["condition"])
        ids = [r["id"] for r in g["rows"]]
        if len(set(ids)) != len(ids):
            raise SystemExit("%s: an item appears twice" % name)
        if g["full_pass"]:
            if ids != frozen:
                raise SystemExit("%s: does not cover the frozen ID list in order" % name)
            full[cond] = g
        else:
            reask[cond] = (g, ids)
        for r in g["rows"]:
            k = (r["id"], cond)
            st[k], why[k] = judge_row(task, r, gold(r["id"]))
            blank[k] = bool(r["rule_abstain"]) or not r["text"].strip()
            tx[k] = r["text"]
    if set(full) != full_need:
        raise SystemExit("%s: full passes %s, expected %s" % (d, sorted(full), sorted(full_need)))
    cand = [p for p in frozen if blank[(p, e_cond)]]
    bl_path = d + "/n1_blank_ids.json"
    if cand:
        if set(reask) != lad_need:
            raise SystemExit("%s: re-ask arms %s, expected %s" % (d, sorted(reask), sorted(lad_need)))
        with open(bl_path, encoding="utf-8") as f:
            listed = json.load(f)
        if listed != cand:
            raise SystemExit("%s: the blank list is not the E file's blank items" % d)
        for cond, (g, ids) in reask.items():
            if ids != cand or g.get("only_ids_sha256") != sha_file(bl_path):
                raise SystemExit("%s: %s does not re-ask exactly the blank items of this E file" % (d, cond))
    elif reask:
        raise SystemExit("%s: re-ask files without any blank item" % d)
    return st, blank, tx, why, frozen, set(cand)


def ci(v, seed=SEED):
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return sum(v) / n, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]


def evidence(da, dn):
    """da, dn = (point, low, high) of the accuracy and no-answer changes as fractions."""
    if da[0] < -0.01 or dn[0] > 0.03:
        return "damage (point estimate beyond a line)"
    if da[1] >= -0.01 and dn[2] <= 0.03:
        return "kept (both intervals exclude damage beyond the lines)"
    return "damage cannot be excluded (point estimates within, an interval crosses a line)"


def n1_steps(sp, t, k1, k2):
    """Per item: logical generations of N1 (1 for E, plus every C strength tried, including the vetoed one), and the
    counts of re-asked, released, vetoed items."""
    recs = fr.records(sp, ("N1", "2", k1, k2), t)
    gens, cnt = [], {"re-asked": 0, "released": 0, "vetoed": 0}
    for r in recs:
        n = 1
        if r["cand"] is not None:
            cap = k2 if r["cand"] == "own" else k1
            lad = [] if cap is None else fr.LAD[:fr.LAD.index(cap) + 1]
            if lad:
                cnt["re-asked"] += 1
                tried = lad if r["note"] in ("above the cap", "blank at every dose") else lad[:lad.index(r["rung"]) + 1]
                n += len(tried)
            cnt["released"] += r["note"] == "released"
            cnt["vetoed"] += r["note"].startswith("vetoed")
        gens.append(n)
    return recs, gens, cnt


def report(task, names, batch):
    with open(CFG, encoding="utf-8") as f:
        M = json.load(f)["models"]
    gold = gold_of(task)
    print("capability check %s, batch %d: accuracy and no-answer rate per condition; changes vs no write (95%%, items resampled)" % (task, batch))
    joint = {}
    for name in names:
        m = M[name]
        st, blank, tx, why, order, cand = load(task, name, batch, m, gold)
        sp = CapSplit(m["el"], m["cl"], st, blank, order)
        t = fr.Texts(tx, tx)
        k1, k2 = LADNAME[m["caps"]["e_blank"]], LADNAME[m["caps"]["own_blank"]]
        base_recs = fr.records(sp, ("base",), t)
        base = fr.states(base_recs)
        print("\n== %s %s: %d items; blank under E: %d" % (task, name, len(order), len(cand)))
        rows = [("no write", base_recs), ("E 2", fr.records(sp, ("E", "2"), t))]
        n1, gens, cnt = n1_steps(sp, t, k1, k2)
        rows.append(("N1 (caps own %s / E's %s)" % (k2, k1), n1))
        rows.append(("C = 2 on every item (stress control)", fr.records(sp, ("C", "2"), t)))
        for label, recs in rows:
            s = fr.states(recs)
            reasons = {}
            for r in recs:
                if r["state"] == "X":
                    w = why[(r["p"], r["cond"])]
                    reasons[w] = reasons.get(w, 0) + 1
            acc = sum(x == "R" for x in s) / len(s)
            na = sum(x == "X" for x in s) / len(s)
            head = "    %-38s accuracy %.3f | no answer %.3f %s" % (label, acc, na, json.dumps(reasons, sort_keys=True))
            if label == "no write":
                print(head)
                continue
            da = ci([(x == "R") - (y == "R") for x, y in zip(s, base)])
            dn = ci([(x == "X") - (y == "X") for x, y in zip(s, base)])
            print(head)
            print("      change: accuracy %+.1f [%+.1f, %+.1f] points, no answer %+.1f [%+.1f, %+.1f] -> %s" % (
                100 * da[0], 100 * da[1], 100 * da[2], 100 * dn[0], 100 * dn[1], 100 * dn[2], evidence(da, dn)))
            joint.setdefault(label.split(" (")[0], []).append([(x == "R") - (y == "R") for x, y in zip(s, base)])
        e_states = fr.states(rows[1][1])
        changed = sum(a != b for a, b in zip(fr.states(n1), e_states))
        print("      N1: re-asked %d, released %d, vetoed %d, final state differs from E 2 on %d; generations per item %.3f" % (
            cnt["re-asked"], cnt["released"], cnt["vetoed"], changed, sum(gens) / len(gens)))
    if len(names) > 1:
        print("\n== five-model (or listed-model) totals on the same items, items resampled jointly")
        for label, per in joint.items():
            n = len(per[0])
            q = [sum(v[i] for v in per) for i in range(n)]
            rng = random.Random(SEED)
            bs = sorted(sum(q[rng.randrange(n)] for _ in range(n)) / (n * len(per)) for _ in range(N_BOOT))
            print("    %-38s mean accuracy change %+.2f [%+.2f, %+.2f] points" % (
                label, 100 * sum(q) / (n * len(per)), 100 * bs[int(0.025 * N_BOOT)], 100 * bs[int(0.975 * N_BOOT) - 1]))


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
    _, gens, cnt = n1_steps(sp, t, "0-25", "0-25")
    assert gens == [1, 2, 2] and cnt == {"re-asked": 2, "released": 1, "vetoed": 1}, (gens, cnt)
    assert judge_row("mmlu", {"text": "B", "rule_abstain": False, "finished": True}, "B") == ("R", "leading")
    assert judge_row("mmlu", {"text": "The answer is not D.", "rule_abstain": False, "finished": True}, "D") == ("X", "negated")
    assert judge_row("mmlu", {"text": "I don't know.", "rule_abstain": True, "finished": True}, "B") == ("X", "refusal")
    assert judge_row("gsm8k", {"text": "We used 7 boxes and next we must", "rule_abstain": False, "finished": False}, 7.0) == ("X", "truncated")
    assert judge_row("gsm8k", {"text": "The answer is 1/2.", "rule_abstain": False, "finished": True}, 0.5) == ("R", "final")
    assert rename("gsm8k", "baseline_neutral") == "baseline_x" and rename("gsm8k", "X_neutral--E1-w-p2-and-C2-w-m1") == "X_x--E1-w-p2-and-C2-w-m1"
    assert evidence((0.0, -0.013, 0.013), (0.0, -0.01, 0.01)).startswith("damage cannot be excluded")
    assert evidence((0.0, -0.005, 0.005), (0.0, -0.01, 0.02)).startswith("kept")
    assert evidence((-0.02, -0.03, -0.01), (0.0, 0.0, 0.0)).startswith("damage (")
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
