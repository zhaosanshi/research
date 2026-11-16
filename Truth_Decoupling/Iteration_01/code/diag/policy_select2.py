# Iteration_01 analysis (Claude Opus 5.5, 2026-11-16), read-only, no model, no judge call.
# 赵磊 11-16 plan, item 3: the formal evaluation of the method "E, then C only on the blanks", with two prompt baselines.
# Every setting is CHOSEN on the tune questions (192) and REPORTED on the test questions.
#   Plain prompt (answering optional):
#     single E   E alone at s in {1, 2}
#     method     E at s; on the questions E alone leaves blank (rule detector) C is raised 0.25 -> 0.5 -> 1 -> 2 until the
#                model answers, up to a cap K (K = none is E alone), as policy_select.py
#     prompt B   E at s; on the same blanks the question is asked again with the must-answer prompt (same E write at s)
#   Must-answer prompt (no blanks):
#     E write    E at s in {1, 2} on the must-answer prompt
#     prompt A   the truth sentence (E2) in the must-answer prompt, nothing written (baseline_conf_E2); no setting to choose
# Score = right - LAM * wrong, LAM in {0, 0.5, 1}; ties go to the weaker setting. Test differences: paired bootstrap
# (4000 resamples, seed 20261116). States as policy_select.py / policy_offline.py.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/policy_select2.py [MODEL ...]
import json
import random
import sys

# copied from policy_select.py (importing it would run its whole report)
LAD = ["0-25", "0-5", "1", "2"]
NAME = {"0-25": "0.25", "0-5": "0.5", "1": "1", "2": "2", None: "none"}


def load(work, judge):
    st, blank = {}, {}
    for l in open(work + "/judge2/single_%s.jsonl" % judge, encoding="utf-8"):
        r = json.loads(l)
        k = (r["pair_id"], r["condition"])
        if r["status"] != "scored":
            s = "X" if (r.get("basis") == "empty_output_no_judge_vote" and r.get("empty_output") is True) else "?"
        elif judge == "allenai":
            s = "X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]
        else:
            s = {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
        st[k] = s
        blank[k] = bool(r.get("rule_abstain")) or bool(r.get("empty_output"))
    return st, blank

G = "Iteration_01/results/ff/gemma4/12b/e44/gemma4-12b"
GX = "Iteration_01/results/ff_xu1/gemma4/12b/e44/gemma4-12b"
LR = "results_e44_factorial_das/llama3.1-8b_side_E2"
NR = "results_newmodels_ff_20261112/e44/qwen3.5-9b_E2"
QR = "results_newmodels_ff_20261112/e44/qwen3.8-27b-dm"
# per model: E+C work dirs (tune, test), must-answer work dirs (tune, test), the judge of each split, layers, order of the pair
MODELS = {
    "Llama-3.1-8B": dict(ec=(LR + "/eval_extra/judge_tune_ecll19_tune", LR + "/eval_extra/judge_test_tll19_test"),
                         d=(LR + "/judge_tune_proD_E_L11", LR + "/judge_test_testD_E_L11"), judge=("pro", "pro"), e=11, c=19, order="EC"),
    "Qwen3.5-9B": dict(ec=(NR + "/eval_extra/judge_tune_ec9b_tune", NR + "/eval_extra/judge_test_t9b16_test"),
                       d=(NR + "/judge_tune_proD_E_L15", NR + "/judge_test_testD_E_L15"), judge=("pro", "pro"), e=15, c=16, order="EC"),
    "Qwen3.8-27B": dict(ec=(QR + "/eval_extra/judge_tune_ec2733_tune", QR + "/eval_extra/judge_test_t2733_test"),
                        d=(QR + "/judge_tune_proD_E_L28", QR + "/judge_test_testD_E_L28"), judge=("pro", "pro"), e=28, c=33, order="EC"),
    # Gemma 4: AllenAI only on the tune questions (selection), DeepSeek Pro on the Dr's 100 test questions (report)
    "Gemma-4-12B": dict(ec=(G + "/eval_extra/judge_tune_ec21_tune", G + "/eval_extra/judge_test_t21_testdr100"),
                        d=(G + "/judge_tune_proD_E_L24", G + "/judge_test_testD_E_L24dr100"), judge=("allenai", "pro"), e=24, c=21, order="CE"),
    "Gemma-4-12B stronger caution": dict(ec=(GX + "/eval_extra/judge_tune_ec25_tune", GX + "/eval_extra/judge_test_t25_testdr100"),
                                         d=(GX + "/judge_tune_proD_E_L24", GX + "/judge_test_testD_E_L24dr100"), judge=("allenai", "pro"),
                                         e=24, c=25, order="EC"),
}


class Split:
    def __init__(self, m, i):
        self.m = m
        self.st, self.blank = load(m["ec"][i], m["judge"][i])
        self.dst, _ = load(m["d"][i], m["judge"][i])
        need = ["baseline_x"] + [self.cE(s) for s in "12"] + [self.cEC(s, k) for s in "12" for k in LAD]
        dneed = ["baseline_conf", "baseline_conf_E2"] + [self.dE(s) for s in "12"]
        for have, want, w in ((self.st, need, m["ec"][i]), (self.dst, dneed, m["d"][i])):
            miss = sorted(set(want) - {c for _, c in have})
            if miss:
                raise SystemExit("%s lacks %s" % (w, miss))
        qs = sorted(p for p, c in self.st if c == "baseline_x")
        self.qs = [p for p in qs if all(self.st.get((p, c), "?") != "?" for c in need) and all(self.dst.get((p, c), "?") != "?" for c in dneed)]
        self.n_dropped = len(qs) - len(self.qs)

    def cE(self, s):
        return "E_L%d_x-w-p%s" % (self.m["e"], s)

    def dE(self, s):
        return "E_L%d_conf-w-p%s" % (self.m["e"], s)

    def cEC(self, s, k):
        m = self.m
        return ("X_x--E%d-w-p%s-and-C%d-w-m%s" % (m["e"], s, m["c"], k)) if m["order"] == "EC" else ("X_x--C%d-w-m%s-and-E%d-w-p%s" % (m["c"], k, m["e"], s))

    def states(self, cfg):
        kind = cfg[0]
        if kind == "base":
            return [self.st[(p, "baseline_x")] for p in self.qs]
        if kind == "E":
            return [self.st[(p, self.cE(cfg[1]))] for p in self.qs]
        if kind == "EC":
            s, cap = cfg[1], cfg[2]
            out = []
            for p in self.qs:
                e = self.cE(s)
                state = self.st[(p, e)]
                if self.blank[(p, e)]:
                    for k in LAD[:LAD.index(cap) + 1]:
                        c = self.cEC(s, k)
                        if not self.blank[(p, c)]:
                            state = self.st[(p, c)]
                            break
                out.append(state)
            return out
        if kind == "B":                                         # E at s; its blanks asked again with the must-answer prompt
            s = cfg[1]
            e = self.cE(s)
            return [self.dst[(p, self.dE(s))] if self.blank[(p, e)] else self.st[(p, e)] for p in self.qs]
        if kind == "Dnone":
            return [self.dst[(p, "baseline_conf")] for p in self.qs]
        if kind == "DA":
            return [self.dst[(p, "baseline_conf_E2")] for p in self.qs]
        if kind == "DE":
            return [self.dst[(p, self.dE(cfg[1]))] for p in self.qs]
        raise ValueError(cfg)


def score(states, lam):
    return sum(s == "R" for s in states) - lam * sum(s == "W" for s in states)


def describe(cfg):
    k = cfg[0]
    return {"base": lambda: "no write", "E": lambda: "E %s" % cfg[1], "EC": lambda: "E %s, C up to %s on blanks" % (cfg[1], NAME[cfg[2]]),
            "B": lambda: "E %s, blanks re-asked must-answer" % cfg[1], "Dnone": lambda: "must-answer, no write",
            "DA": lambda: "must-answer + truth sentence", "DE": lambda: "must-answer + E %s" % cfg[1]}[k]()


def ci(v):
    rng = random.Random(20261116)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) for _ in range(4000))
    if any(x != int(x) for x in v):
        return "%+.1f [%+.1f, %+.1f]" % (sum(v), bs[100], bs[3899])
    return "%+d [%+d, %+d]" % (sum(v), bs[100], bs[3899])


FAM = {
    "single E": [("E", "1"), ("E", "2")],
    "method": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in LAD],
    "prompt B": [("B", "1"), ("B", "2")],
    "E write": [("DE", "1"), ("DE", "2")],
    "prompt A": [("DA",)],
    "must none": [("Dnone",)],
}
ROWS = (("plain prompt", ("single E", "method", "prompt B"), (("method", "prompt B"), ("method", "single E"))),
        ("must-answer prompt", ("must none", "prompt A", "E write"), (("E write", "prompt A"), ("prompt A", "must none"))))

for name in (sys.argv[1:] or MODELS):
    m = MODELS[name]
    tune, test = Split(m, 0), Split(m, 1)
    print("\n## %s (tune: %s, %d questions; test: %s, %d questions; dropped for a missing verdict: tune %d, test %d)" % (
        name, m["judge"][0], len(tune.qs), m["judge"][1], len(test.qs), tune.n_dropped, test.n_dropped))
    for lam in (0, 0.5, 1):
        print("- score = right - %s x wrong" % lam)
        pick = {f: max(c, key=lambda x: (score(tune.states(x), lam), -c.index(x))) for f, c in FAM.items()}
        for title, fams, diffs in ROWS:
            row = {}
            for f in fams:
                s = test.states(pick[f])
                row[f] = s
                print("    [%s] %-9s chosen on tune: %-34s | test: right %3d wrong %3d blank %3d | score %6.1f" % (
                    title, f, describe(pick[f]), sum(x == "R" for x in s), sum(x == "W" for x in s), sum(x == "X" for x in s), score(s, lam)))
            val = lambda s: [(1 if x == "R" else 0) - lam * (1 if x == "W" else 0) for x in s]
            for a, b in diffs:
                d = [x - y for x, y in zip(val(row[a]), val(row[b]))]
                dr = [(x == "R") - (y == "R") for x, y in zip(row[a], row[b])]
                print("    [%s] %s minus %s on test: score %s | right answers %s" % (title, a, b, ci(d), ci(dr)))
