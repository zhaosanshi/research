# Iteration_01 analysis (Claude Opus 5.5, 2026-11-17), read-only: no model, no judge call, no GPU. Standard library only.
# Purpose: can a judge-free "is the answer stable" signal keep the right answers the C ladder releases and leave the wrong
# ones blank? Verdicts: DeepSeek Pro on both splits. Score = right - wrong only (right +1, blank 0, wrong -1, vague 0).
#
# Definitions
#   candidates  questions the rule detector marks blank under E alone at s. (ii) blank with no write too; (i) answered with
#               no write (E silenced them).
#   release     the smallest k* on 0.25 -> 0.5 -> 1 -> 2 (<= cap K) at which the joint arm "E s + C k" is not blank; the
#               released answer is that arm's text, its state that arm's Pro state.
#   normalise   lower case; every non-alphanumeric character -> space; collapse spaces; strip. "same" = equal after
#               normalising. Loose version: word sets of the normalised texts with Jaccard >= 0.5.
#   signals     A-next   k* < 2 and the joint arm at the next ladder rung k+ (the full ladder, whatever the cap) is not
#                        blank and its text is the same as at k*
#               A-E      the joint arm at the other E strength s' and the same k* is not blank and its text is the same
#               A-must   the must-answer prompt with E at s (E_L<e>_conf-w-p<s>) gives the same text (no d dir: skipped)
#               A-Calone C alone at k* is not blank and its text is the same (diagnostic only)
#               A signal whose text is missing counts as not holding; such cases are counted in the output.
# Policies (settings chosen on tune by right - wrong, ties to the earlier = weaker setting, as c_first.py; reported on test):
#   single E (s in {1,2}); single C (as policy_select2 "single C"); M0 = c_first "method"; M1 = c_first "baseonly";
#   M2/M4/M6 = method with a gate (A-next / A-must / A-E): a released answer is used only if the gate holds, else the
#              question keeps its E-alone state; settings (s, K)
#   M3/M5/M7 = (ii) released as usual up to K2; (i) released up to K1 only if the gate holds; settings (s, K1, K2)
# Predictions written before the run (copied verbatim from the task, not to be changed):
# - G1：E 压住后放出的回答里，A-next 成立的比不成立的正确率高 15 个百分点以上（测试集，至少 3 个模型；只算两组都不少于 20 条的模型）。
# - G2：M3 在 Llama-3.1-8B 和两个 Qwen 上的测试集得分高于 M0。
# - G3：M3 比 M1 低不超过 3 分（各模型）。
# - G4：M3 减"只推 E"的区间在 0 以上的模型，五个有留空题的模型里至少四个。
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/release_gate.py [MODEL ...]
import collections
import json
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf

LAD = ps.LAD
NAME = ps.NAME
CAPS = [None] + LAD
ORDER = ["Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B", "Gemma-4-12B",
         "Gemma-4-12B stronger caution", "Gemma-4-31B"]
MODELS = cf.MODELS
SIGS = ("next", "E", "must", "Calone")


def norm(t):
    return " ".join("".join(ch if ch.isalnum() else " " for ch in t.lower()).split())


def same(a, b, loose):
    if a is None or b is None:
        return False
    a, b = norm(a), norm(b)
    if not loose:
        return a == b
    x, y = set(a.split()), set(b.split())
    if not x and not y:
        return True
    return len(x & y) / len(x | y) >= 0.5


def texts(work):
    out = {}
    for l in open(work + "/judge2/cells.jsonl", encoding="utf-8"):
        r = json.loads(l)
        k = (r["pair_id"], r["condition"])
        if k in out and out[k] != r["prediction"]:
            raise SystemExit("%s: conflicting duplicate cell %s" % (work, k))
        out[k] = r["prediction"]
    return out


class Sp:
    def __init__(self, m, i):
        self.sp = ps.Split(m, i)
        self.m = m
        self.tx = texts(m["ec"][i])
        self.dtx = texts(m["d"][i]) if m.get("d") else None
        sp = self.sp
        need = [sp.cE(s) for s in "12"] + [sp.cEC(s, k) for s in "12" for k in LAD]
        miss = [(p, c) for p in sp.qs for c in need if (p, c) not in self.tx]
        if miss:
            raise SystemExit("%s: joint/E cells without text: %s" % (m["ec"][i], miss[:5]))
        self.missing = collections.Counter()      # (signal, loose) -> signal cases with a missing text
        self.cache = {}

    def release(self, p, s, cap):
        """(class, k*) for a candidate, k* None if nothing within cap; None if p is not a candidate."""
        sp = self.sp
        if not sp.blank[(p, sp.cE(s))]:
            return None
        g = "ii" if sp.blank[(p, "baseline_x")] else "i"
        if cap is None:
            return g, None
        for k in LAD[:LAD.index(cap) + 1]:
            if not sp.blank[(p, sp.cEC(s, k))]:
                return g, k
        return g, None

    def sig(self, name, p, s, k, loose):
        key = (name, p, s, k, loose)
        if key in self.cache:
            return self.cache[key]
        sp = self.sp
        t = self.tx[(p, sp.cEC(s, k))]
        other = None
        ok_blank = True
        if name == "next":
            if k == "2":
                self.cache[key] = False
                return False
            c = sp.cEC(s, LAD[LAD.index(k) + 1])
            other = self.tx.get((p, c))
            ok_blank = (p, c) in sp.blank and not sp.blank[(p, c)]
        elif name == "E":
            c = sp.cEC("2" if s == "1" else "1", k)
            other = self.tx.get((p, c))
            ok_blank = (p, c) in sp.blank and not sp.blank[(p, c)]
        elif name == "Calone":
            c = sp.cC(k)
            other = self.tx.get((p, c))
            ok_blank = (p, c) in sp.blank and not sp.blank[(p, c)]
        elif name == "must":
            if self.dtx is None:
                raise ValueError("no must-answer texts")
            other = self.dtx.get((p, sp.dE(s)))
        if other is None:
            self.missing[(name, loose)] += 1
        r = ok_blank and same(t, other, loose)
        self.cache[key] = r
        return r

    def states(self, cfg, loose=False):
        sp = self.sp
        kind = cfg[0]
        if kind in ("base", "E", "C", "EC"):
            return sp.states(cfg)
        if kind == "BO":
            return cf.states(sp, cfg)
        out = []
        for p in sp.qs:
            if kind == "G":                         # ("G", gate, s, K)
                gate, s, cap = cfg[1], cfg[2], cfg[3]
                state = sp.st[(p, sp.cE(s))]
                r = self.release(p, s, cap)
                if r and r[1] is not None and self.sig(gate, p, s, r[1], loose):
                    state = sp.st[(p, sp.cEC(s, r[1]))]
            elif kind == "H":                       # ("H", gate, s, K1, K2)
                gate, s, k1, k2 = cfg[1], cfg[2], cfg[3], cfg[4]
                state = sp.st[(p, sp.cE(s))]
                r = self.release(p, s, k2)
                if r and r[0] == "ii" and r[1] is not None:
                    state = sp.st[(p, sp.cEC(s, r[1]))]
                r = self.release(p, s, k1)
                if r and r[0] == "i" and r[1] is not None and self.sig(gate, p, s, r[1], loose):
                    state = sp.st[(p, sp.cEC(s, r[1]))]
            else:
                raise ValueError(cfg)
            out.append(state)
        return out


def val(st):
    return [(x == "R") - (x == "W") for x in st]


def sc(st):
    return sum(val(st))


def ci_b(v):
    """same resampling as ps.ci; returns (sum, lo, hi)"""
    rng = random.Random(20261116)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) for _ in range(4000))
    return sum(v), bs[100], bs[3899]


def cis(a, b):
    t, lo, hi = ci_b([x - y for x, y in zip(val(a), val(b))])
    return "%+d [%+d, %+d]" % (t, lo, hi), (t, lo, hi)


def desc(cfg):
    k = cfg[0]
    if k in ("base", "E", "C", "EC", "BO"):
        return cf.describe(cfg)
    if k == "G":
        return "E %s, C <= %s, gate %s" % (cfg[2], NAME[cfg[3]], cfg[1])
    return "E %s, (i) C <= %s gated %s, (ii) C <= %s" % (cfg[2], NAME[cfg[3]], cfg[1], NAME[cfg[4]])


def cnt(c):
    return "R %d W %d H %d X %d" % (c["R"], c["W"], c["H"], c["X"])


def families(tune, test, has_d):
    E = [("E", s) for s in "12"]
    fam = collections.OrderedDict()
    fam["single E"] = list(E)
    fam["single C"] = [("C", k) for k in LAD if k in tune.sp.cks and k in test.sp.cks]
    fam["M0"] = E + [("EC", s, k) for s in "12" for k in LAD]
    fam["M1"] = E + [("BO", s, k) for s in "12" for k in LAD]
    for a, b, g in (("M2", "M3", "next"), ("M4", "M5", "must"), ("M6", "M7", "E")):
        if g == "must" and not has_d:
            continue
        fam[a] = E + [("G", g, s, k) for s in "12" for k in LAD]
        fam[b] = E + [("H", g, s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    if not fam["single C"]:
        del fam["single C"]
    return fam


def diagnose(name, sp, label, s, loose, res):
    """K = 2: released answers by class, signal holding or not, state; also by k*"""
    print("  [diag %s] %s, E %s, cap 2:" % (label, "loose" if loose else "strict", s))
    for g in ("i", "ii"):
        rel = [(p, r[1]) for p in sp.sp.qs for r in [sp.release(p, s, "2")] if r and r[0] == g]
        n_none = sum(k is None for _, k in rel)
        rel = [(p, k) for p, k in rel if k is not None]
        print("    class (%s): candidates %d, released %d, still blank at cap 2 %d" % (g, len(rel) + n_none, len(rel), n_none))
        for sg in SIGS:
            if sg == "must" and sp.dtx is None:
                continue
            tot = {True: collections.Counter(), False: collections.Counter()}
            byk = collections.defaultdict(collections.Counter)
            for p, k in rel:
                h = sp.sig(sg, p, s, k, loose)
                st = sp.sp.st[(p, sp.sp.cEC(s, k))]
                tot[h][st] += 1
                byk[(k, h)][st] += 1
            for h in (True, False):
                c = tot[h]
                n = sum(c.values())
                acc = c["R"] / n if n else float("nan")
                print("      A-%-6s %-5s n %3d  %s  right-wrong %+d  acc %.3f" % (sg, "holds" if h else "fails", n, cnt(c), c["R"] - c["W"], acc))
                res[(name, label, loose, g, sg, h)] = (n, c["R"], c["W"])
            print("        by k*: " + " | ".join("k*=%s %s: %s" % (NAME[k], "holds" if h else "fails", cnt(byk[(k, h)]))
                                                 for k in LAD for h in (True, False) if sum(byk[(k, h)].values())))


def report(name, res):
    m = MODELS[name]
    tune, test = Sp(m, 0), Sp(m, 1)
    has_d = tune.dtx is not None and test.dtx is not None
    fam = families(tune, test, has_d)
    print("\n## %s (tune %d questions, test %d; dropped for a missing verdict: tune %d, test %d)" % (
        name, len(tune.sp.qs), len(test.sp.qs), tune.sp.n_dropped, test.sp.n_dropped))
    for lab, sp in (("tune", tune), ("test", test)):
        for s in "12":
            c = collections.Counter(sp.release(p, s, "2")[0] for p in sp.sp.qs if sp.release(p, s, "2"))
            print("  %s, E %s: candidates (i) %d, (ii) %d" % (lab, s, c["i"], c["ii"]))
            res[(name, lab, "ncand", s)] = c["i"] + c["ii"]
    for loose in (False, True):
        lt = "loose" if loose else "strict"
        pick = {f: max(c, key=lambda x: (sc(tune.states(x, loose)), -c.index(x))) for f, c in fam.items()}
        s_m = pick["M0"][1]
        res[(name, "s_method", lt)] = s_m
        print("  -- %s signals; method's E strength chosen on tune: %s" % (lt, s_m))
        for lab, sp in (("test", test), ("tune", tune)):
            diagnose(name, sp, lab, s_m, loose, res)
        row = {}
        for f in fam:
            st = test.states(pick[f], loose)
            row[f] = st
            c = collections.Counter(st)
            print("  [%s] %-8s chosen on tune: %-44s | tune %4d | test: %s | score %4d" % (
                lt, f, desc(pick[f]), sc(tune.states(pick[f], loose)), cnt(c), sc(st)))
            res[(name, lt, f)] = (sc(tune.states(pick[f], loose)), desc(pick[f]), c, sc(st))
        for f in fam:
            for b in ("single E", "single C", "M0", "M1"):
                if f == b or b not in row:
                    continue
                s_, t = cis(row[f], row[b])
                res[(name, lt, f, b)] = t
                print("  [%s] %s minus %s on test: %s" % (lt, f, b, s_))
        mis = {k: v for k, v in test.missing.items()}
        mis2 = {k: v for k, v in tune.missing.items()}
        print("  [%s] signal checks with a missing text (counted as not holding; cumulative per split): test %s | tune %s" % (
            lt, mis or "none", mis2 or "none"))


def predictions(res):
    print("\n## predictions (strict signals)")
    g1 = []
    for n in ORDER:
        a = res.get((n, "test", False, "i", "next", True))
        b = res.get((n, "test", False, "i", "next", False))
        if a and b:
            ok = a[0] >= 20 and b[0] >= 20
            d = (a[1] / a[0] - b[1] / b[0]) * 100 if a[0] and b[0] else float("nan")
            print("  G1 %-30s holds n %3d acc %s | fails n %3d acc %s | diff %s pp | eligible %s" % (
                n, a[0], "%.3f" % (a[1] / a[0]) if a[0] else "-", b[0], "%.3f" % (b[1] / b[0]) if b[0] else "-",
                "%+.1f" % d if d == d else "-", ok))
            if ok and d > 15:
                g1.append(n)
    print("  G1: models with diff > 15 pp among eligible: %s -> %s" % (g1, "holds" if len(g1) >= 3 else "does not hold"))
    for n in ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B"):
        print("  G2 %-30s M3 %d vs M0 %d -> %s" % (n, res[(n, "strict", "M3")][3], res[(n, "strict", "M0")][3],
                                                  "higher" if res[(n, "strict", "M3")][3] > res[(n, "strict", "M0")][3] else "not higher"))
    for n in ORDER:
        a, b = res[(n, "strict", "M3")][3], res[(n, "strict", "M1")][3]
        print("  G3 %-30s M3 %d M1 %d M3-M1 %+d -> %s" % (n, a, b, a - b, "ok" if a - b >= -3 else "fails"))
    k = 0
    with_blank = []
    for n in ORDER:
        t = res[(n, "strict", "M3", "single E")]
        nc = res[(n, "test", "ncand", res[(n, "s_method", "strict")])]
        print("  G4 %-30s test candidates at method's s: %d | M3 - single E %+d [%+d, %+d]" % (n, nc, t[0], t[1], t[2]))
        if nc > 0:
            with_blank.append(n)
            k += t[1] > 0
    print("  G4: models with test candidates: %d; with interval above 0: %d" % (len(with_blank), k))


def main():
    res = {}
    names = sys.argv[1:] or ORDER
    for n in names:
        report(n, res)
    if not sys.argv[1:]:
        print("\n## task 3: M0 / M1 minus single C on test (paired bootstrap, as ps.ci)")
        for n in ORDER:
            a, b = res.get((n, "strict", "M0", "single C")), res.get((n, "strict", "M1", "single C"))
            print("  %-30s M0 - single C %s | M1 - single C %s" % (
                n, "%+d [%+d, %+d]" % a if a else "-", "%+d [%+d, %+d]" % b if b else "-"))
        predictions(res)


if __name__ == "__main__":
    main()
