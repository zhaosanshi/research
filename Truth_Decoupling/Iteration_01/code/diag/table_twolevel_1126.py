#!/usr/bin/env python3
# Iteration_01 analysis (Claude Opus 5.5, executor wei, 2026-11-26), read-only, no model, no judge call. Exploration,
# not reviewed by GPT. Plan discussion/work_queue/2026-11-26_queue_o.md, section 6 ("two-level E", readings written
# 15:54 before this program ran). TruthfulQA TUNE split only (192 questions), five models, main-table settings at the
# write rule "prompt"; built on diag/two_level_sim_1126.py (same helpers, same text comparison, same interval).
# Rule: hand in N1's answer; but on a question answered (verdict right or wrong) both with no write and under single E
# at the main strength, with a DIFFERENT text after norm() (lower case, every run of non [a-z0-9] to one space), hand
# in the answer of single E at the second-level strength s2 (blank / hedge there = 0).
# Inputs: diag/main_table_prompt.prompt_split (no write, single E, N1 ladder; tune) and the single-E answers at s2 with
# DeepSeek Pro verdicts from the work dirs listed in WORK (existing ones and the new judge_tune_twolevel_1126_l<l>_tune
# of launchers/explore_20261120/run_twolevel_1126.sh). MMLU side offline from capability/ecmmlu_1126/<model>.json.
# N1's tune nets are reproduced first (+27, +63, +80, +74, +53); a mismatch stops the table.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/table_twolevel_1126.py [--partial]
#   --partial: a second-level work dir without complete Pro verdicts is reported and skipped (default: stop).
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as A  # noqa: E402

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
fr = MP.fr
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
VAL = {"R": 1, "W": -1, "N": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
MAIN = {"llama3.1-8b": (16, "3.5"), "qwen3.5-9b": (15, "2"), "qwen3.8-27b": (28, "1.75"),
        "gemma3-12b": (24, "2.5"), "gemma3-27b": (28, "1")}
REF_N1 = {"llama3.1-8b": 27, "qwen3.5-9b": 63, "qwen3.8-27b": 80, "gemma3-12b": 74, "gemma3-27b": 53}
# second-level candidates and the work dir (judge_tune_<name>_tune) holding their Pro verdicts at the prompt rule
WORK = {"llama3.1-8b": (("5", "pospromptl16"), ("7", "pospromptl16")),
        "qwen3.5-9b": (("2.5", "depthE"), ("3", "twolevel_1126_l15")),
        "qwen3.8-27b": (("2.5", "twolevel_1126_l28"), ("3", "twolevel_1126_l28")),
        "gemma3-12b": (("3", "twolevel_1126_l24"), ("4", "pospromptl24")),
        "gemma3-27b": (("1.5", "twolevel_1126_l28"), ("2", "twolevel_1126_l28"))}
MMLU_S2 = {"llama3.1-8b": "5", "qwen3.5-9b": "3", "qwen3.8-27b": "3", "gemma3-12b": "4", "gemma3-27b": "2"}
MMLU = "Iteration_01/capability/ecmmlu_1126/%s.json"
PARTIAL = "--partial" in sys.argv[1:]


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def dash(x):
    return x.replace(".", "-")


def ci(d, seed=20261126, n=2000):
    rng = random.Random(seed); m = len(d); out = []
    for _ in range(n):
        out.append(sum(d[rng.randrange(m)] for _ in range(m)))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


def second_level(run, layer, s2, name, ids, base_st, base_tx):
    """{q: state}, {q: blank} of single E at s2 at the prompt rule, or (None, why)."""
    w = os.path.join(run, "eval_extra", "judge_tune_%s_tune" % name)
    if not MP.has(w):
        return None, None, "no Pro verdicts in %s" % os.path.basename(w)
    with open(os.path.join(w, "judge_input.json"), encoding="utf-8") as f:
        j = json.load(f)
    cond = "E_L%d_x-w-p%s" % (layer, dash(s2))
    gfile = "x--E%d-w-p%s-at-prompt_tune.json" % (layer, dash(s2))
    if j.get("split") != "tune" or cond not in j["conditions"] or gfile not in j["sources"]["gen"]:
        raise SystemExit("%s: not the tune export of %s from %s" % (w, cond, gfile))
    spec = [c for c in j["condition_specs"] if c["name"] == cond][0]
    parts = [(p["family"], p["layer"], p["kind"], p["beta"]) for p in spec["parts"]]
    if parts != [("E", layer, "w", float(s2))]:
        raise SystemExit("%s: %s has parts %s" % (w, cond, parts))
    st, bl = ps.load(w, "pro")
    tx = fr.raw_texts(w)
    miss = [q for q in ids if st.get((q, cond), "?") == "?"]
    if miss:
        return None, None, "%d of %d questions without a Pro verdict for %s in %s" % (len(miss), len(ids), cond, os.path.basename(w))
    for q in ids:
        k = (q, "baseline_x")
        if k in st and (st[k] != base_st[q] or tx.get(k) != base_tx[q]):
            raise SystemExit("%s: the no-write cell of %s differs from the main table's" % (w, q))
    return {q: st[(q, cond)] for q in ids}, {q: bool(bl[(q, cond)]) for q in ids}, os.path.basename(w)


def mmlu_block(model, layer, s_main):
    s2 = MMLU_S2[model]
    with open(MMLU % model, encoding="utf-8") as f:
        d = json.load(f)
    names = ("x", "x:E%d.w.p%s" % (layer, s_main), "x:E%d.w.p%s" % (layer, s2))
    rows = [d["arms"][a]["rows"] for a in names]
    ids = [r["id"] for r in rows[0]]
    if d.get("n") != 1000 or any([r["id"] for r in rr] != ids for rr in rows) or len(ids) != 1000:
        raise SystemExit("%s: the MMLU arms do not cover the same 1000 items" % model)
    gold = [r["gold"] for r in rows[0]]
    px, pm, p2 = ([r["pick"] for r in rr] for rr in rows)
    two = [p2[i] if pm[i] != px[i] else pm[i] for i in range(len(ids))]
    acc = lambda p: 100.0 * sum(a == g for a, g in zip(p, gold)) / len(gold)
    a0 = acc(px)
    print("    MMLU 1000 items (no write %.1f%%), change in points: single E %s %+.1f | single E %s %+.1f | two-level %+.1f (items switched to the %s pick: %d)" % (
        a0, s_main, acc(pm) - a0, s2, acc(p2) - a0, acc(two) - a0, s2, sum(1 for i in range(len(ids)) if pm[i] != px[i])))


def main():
    reading = {}
    for model in ORDER:
        display, layer, s = A.KEY[model]
        if (layer, s.replace("-", ".")) != MAIN[model]:
            raise SystemExit("%s: iti_table KEY %s / %s, task %s" % (model, layer, s, MAIN[model]))
        data, err = MP.prompt_split(display, layer, 0, s)
        if data is None:
            raise SystemExit("%s: main-table tune tables not complete: %s" % (model, err))
        sp, tx, _, ids = data
        pick = ML.n1_pick(sp, tx, s)
        cE = sp.cE(s)
        b = {q: G[sp.st[(q, "baseline_x")]] for q in ids}
        e = {q: G[sp.st[(q, cE)]] for q in ids}
        n1s = S.system(sp, tx, pick)
        n1 = dict(zip(ids, [G[v] for v in n1s[0]]))
        n1ans = dict(zip(ids, n1s[1]))
        net = lambda dd: sum(VAL[dd[q]] for q in ids)
        print("\n## %s (display %s) tune %d questions: E layer %d, main strength %s; N1 setting %s" % (
            model, display, len(ids), layer, s.replace("-", "."), (pick,)))
        if len(ids) != 192 or net(n1) != REF_N1[model]:
            raise SystemExit("%s: N1 not reproduced (net %+d against %+d, %d questions)" % (model, net(n1), REF_N1[model], len(ids)))
        t0 = {q: norm(tx.d[(q, "baseline_x")]) for q in ids}
        t1 = {q: norm(tx.d[(q, cE)]) for q in ids}
        react = [q for q in ids if b[q] != "N" and e[q] != "N" and t0[q] != t1[q]]
        print("    no write %+d | single E %+d | N1 %+d (reproduced; answers %d) | reacted (answered both, text changed): %d = %s" % (
            net(b), net(e), net(n1), sum(n1ans.values()), len(react),
            ", ".join("%s%s %d" % (x, y, sum(1 for q in react if b[q] == x and e[q] == y)) for x in "RW" for y in "RW")))
        _, run, _ = MP.usual(display, 0)
        base_st = {q: sp.st[(q, "baseline_x")] for q in ids}
        base_tx = {q: tx.d[(q, "baseline_x")] for q in ids}
        reading[model] = []
        for s2, name in WORK[model]:
            e2s, e2b, where = second_level(run, layer, s2, name, ids, base_st, base_tx)
            if e2s is None:
                if not PARTIAL:
                    raise SystemExit("%s: second level %s: %s" % (model, s2, where))
                print("    second level %-4s: %s (skipped, --partial)" % (s2, where))
                continue
            e2 = {q: G[e2s[q]] for q in ids}
            final = dict(n1)
            fans = dict(n1ans)
            for q in react:
                final[q] = e2[q]
                fans[q] = not e2b[q]
            d = [VAL[final[q]] - VAL[n1[q]] for q in ids]
            lo, hi = ci(d)
            c2 = collections.Counter(e2[q] for q in react)
            cn = collections.Counter(n1[q] for q in react)
            parts = []
            for x in "RW":
                for y in "RW":
                    qs = [q for q in react if b[q] == x and e[q] == y]
                    c3 = collections.Counter(e2[q] for q in qs)
                    parts.append("%s%s->R %d W %d N %d" % (x, y, c3["R"], c3["W"], c3["N"]))
            print("    second level %-4s (%s): reacted %d | on them second level R %d W %d N %d, N1 R %d W %d N %d | two-level net %+d (answers %d) | two-level minus N1 %+d [%+d, %+d] | single E %s on all 192: net %+d (answers %d)" % (
                s2, where, len(react), c2["R"], c2["W"], c2["N"], cn["R"], cn["W"], cn["N"], net(final), sum(fans.values()),
                sum(d), lo, hi, s2, net(e2), sum(1 for q in ids if not e2b[q])))
            print("        by (no write, main E) verdicts: %s" % "; ".join(parts))
            reading[model].append((s2, lo))
        mmlu_block(model, layer, s.replace("-", "."))
    print("\n## pre-registered reading (plan section 6): candidates whose two-level minus N1 interval is above 0")
    k = 0
    for model in ORDER:
        above = [s2 for s2, lo in reading[model] if lo > 0]
        k += bool(above)
        print("    %-12s candidates read %d, with the lower end above 0: %d %s" % (model, len(reading[model]), len(above), above))
    print("    models with at least one such candidate: %d of %d" % (k, len(ORDER)))


if __name__ == "__main__":
    main()
