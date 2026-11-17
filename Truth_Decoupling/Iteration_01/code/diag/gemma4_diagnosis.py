# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# 赵磊 2026-11-17: "gemma4和其他模型都不一致，必须要找到玄机和问题，光归结于gemma4不一样，太武断了。"
# Five read-only tabulations on the saved generations and verdicts, to see WHERE Gemma-4 differs from the other models:
#   A  of the questions a model does not answer with NO write (rule detector), how many each write releases, and what the
#      released answers are worth (Pro): E at 1 and 2, the random directions at the E layer, C toward answering alone
#   B  the E line over layers on the plain prompt at strength 1 (gen summaries of the main run): refusals released / newly
#      created by E and by the random direction of the same size, and how many answers change
#   C  verdict changes (Pro, tune) under the random directions at the E layer, next to E itself
#   D  generated answers that start with the stray word "thought" (the marker of Gemma-4's thinking channel), per write group
#   E  the prompt survey of step 0 (AllenAI, tune): refusals under the must-answer, plain and cautious prompts, with and
#      without the truth sentence
# The size of each write relative to the hidden state and to the prompt difference is printed with part C.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/gemma4_diagnosis.py
import collections
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps

MODELS = {n: m for n, m in ps.MODELS.items() if "(tune AllenAI)" not in n}
LEAK = re.compile(r"^\W*thought\b", re.I)


def run_of(m):
    return m["ec"][0].split("/eval_extra/")[0]


def part_a():
    print("# A. Questions not answered with no write (rule detector): what each write releases")
    for name, m in MODELS.items():
        for i, label in ((0, "tune"), (1, "test")):
            sp = ps.Split(m, i)
            conds = sorted({c for _, c in sp.st})
            arms = [("E 1", sp.cE("1")), ("E 2", sp.cE("2"))]
            arms += [("E-layer random " + c.split("rand-")[1], c) for c in conds if c.startswith("E_L%d_x-rand-" % m["e"])]
            arms += [("C toward answering " + ps.NAME[k], sp.cC(k)) for k in ps.LAD]
            b0 = [p for p in sp.qs if sp.blank[(p, "baseline_x")]]
            base = collections.Counter(sp.st[(p, "baseline_x")] for p in sp.qs)
            print("\n## %s, %s: %d questions; no write R %d W %d H %d X %d; not answered (rule) %d" % (
                name, label, len(sp.qs), base["R"], base["W"], base["H"], base["X"], len(b0)))
            for an, c in arms:
                rel = [p for p in b0 if not sp.blank[(p, c)]]
                st = collections.Counter(sp.st.get((p, c), "?") for p in rel)
                print("    %-26s releases %3d of %3d (%3.0f%%) -> R %d W %d H %d X %d%s" % (
                    an, len(rel), len(b0), 100.0 * len(rel) / max(1, len(b0)), st["R"], st["W"], st["H"], st["X"],
                    (" | without a verdict %d" % st["?"]) if st["?"] else ""))


def part_b():
    print("\n\n# B. The E line over layers, plain prompt, strength 1 (tune, 192): E against the random direction of the same size")
    for name, m in MODELS.items():
        run = run_of(m)
        tab = collections.defaultdict(dict)
        for f in sorted(glob.glob(run + "/gen/E_L*_x-*_tune.json")):
            mt = re.match(r"E_L(\d+)_x-(w|rand)-([pm])1_tune\.json", os.path.basename(f))
            if not mt:
                continue
            d = json.load(open(f, encoding="utf-8"))
            rows = d["rows"]
            own = [r for r in rows if r["baseline_refusal_rule"]]
            tab[int(mt.group(1))][mt.group(2) + "-" + mt.group(3)] = (
                sum(1 for r in own if not r["refusal_rule"]), len(own),
                sum(1 for r in rows if not r["baseline_refusal_rule"] and r["refusal_rule"]), d["summary"]["changed_normalized"])
        print("\n## %s (%d layers found)" % (name, len(tab)))
        tot = collections.defaultdict(lambda: [0, 0])
        for L in sorted(tab):
            line = "    L%-2d" % L
            for k in ("w-p", "rand-p", "rand-m"):
                if k in tab[L]:
                    rel, own, new, ch = tab[L][k]
                    line += " | %-6s released %2d/%2d, new refusals %2d, answers changed %3d" % (k, rel, own, new, ch)
                    g = "E" if k == "w-p" else "random"
                    tot[g][0] += rel
                    tot[g][1] += own
            print(line)
        for g in ("E", "random"):
            if tot[g][1]:
                print("    all layers, %-6s: released %d of %d (%.1f%%)" % (g, tot[g][0], tot[g][1], 100.0 * tot[g][0] / tot[g][1]))


def part_c():
    print("\n\n# C. Tune (192): score change and verdict changes (Pro) under E and under the random directions at the E layer; size of the E write")
    for name, m in MODELS.items():
        sp = ps.Split(m, 0)
        e = m["e"]
        base = [sp.st[(p, "baseline_x")] for p in sp.qs]
        sc = lambda v: sum(x == "R" for x in v) - sum(x == "W" for x in v)
        line = "%-30s no write %4d |" % (name, sc(base))
        for label, c in (("E 1", sp.cE("1")), ("E 2", sp.cE("2")), ("random p1", "E_L%d_x-rand-p1" % e), ("random m1", "E_L%d_x-rand-m1" % e),
                         ("random p2", "E_L%d_x-rand-p2" % e), ("random m2", "E_L%d_x-rand-m2" % e)):
            v = [sp.st.get((p, c), "?") for p in sp.qs]
            keep = [i for i, x in enumerate(v) if x != "?"]
            vv, bb = [v[i] for i in keep], [base[i] for i in keep]
            line += " %s %+d (verdicts changed %d%s) |" % (label, sc(vv) - sc(bb), sum(a != b for a, b in zip(vv, bb)),
                                                           ("; %d without a verdict" % (len(v) - len(vv))) if len(vv) < len(v) else "")
        print(line)
        f = "%s/eval_extra/gen/x--E%d-w-p1_tune.json" % (run_of(m), e)
        if os.path.exists(f):
            d = json.load(open(f, encoding="utf-8"))
            p = d["parts"][0]
            rel = [r["write"]["prefill_rel"] for r in d["rows"] if r.get("write")]
            print("%-30s E at 1: amount %.3f | write / |h| %.4f | norm of the prompt difference %.2f | cos(w, prompt difference) %.3f" % (
                "", p["amount"], statistics.mean(rel), p["md_norm"], p["cosine_w_md"]))


def part_d():
    print("\n\n# D. Generated answers that start with the stray word 'thought'")
    for name, m in MODELS.items():
        for i, lab in ((0, "tune"), (1, "test")):
            tot, leak, ex = collections.Counter(), collections.Counter(), {}
            for l in open(m["ec"][i] + "/judge2/cells.jsonl", encoding="utf-8"):
                r = json.loads(l)
                c = r["condition"]
                if c == "baseline_x":
                    g = "no write"
                elif c.startswith("X_"):
                    g = "E + C with a random part" if "rand" in c else "E + C"
                else:
                    g = ("E" if c.startswith("E_") else "C") + (" layer random " + c.split("-rand-")[1] if "-rand-" in c else " write")
                tot[g] += 1
                if LEAK.search(r["prediction"] or ""):
                    leak[g] += 1
                    ex.setdefault(g, r["prediction"][:60].replace("\n", " "))
            print("## %s, %s: %d answers, %d start with 'thought'" % (name, lab, sum(tot.values()), sum(leak.values())))
            for g in sorted(leak, key=lambda x: -leak[x]):
                print("     %-28s %3d of %4d   e.g. %r" % (g, leak[g], tot[g], ex[g]))


def part_e():
    print("\n\n# E. Step 0 prompt survey (AllenAI, tune 192, no write): refusals by prompt")
    seen = set()
    for name, m in MODELS.items():
        d = run_of(m).replace("_side_E2", "").replace("_E2", "") + "/judge_tune_step0"
        f = d + "/judge2/single_allenai.jsonl"
        if d in seen or not os.path.exists(f):
            print("## %s: no step-0 survey at %s" % (name, d))
            continue
        seen.add(d)
        c = collections.defaultdict(collections.Counter)
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            c[r["condition"]]["?" if r["status"] != "scored" else ("X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]])] += 1
        print("## %s" % name)
        print("     " + " | ".join("%s: refused %d" % (k.replace("baseline_", ""), c[k]["X"]) for k in sorted(c)))


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
