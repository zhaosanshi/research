# Iteration_01 analysis (Claude Opus 5.5, 2026-11-17), read-only: no model, no GPU, no judge call.
# Task B of the 11-17 offline batch: the gain of E (score with E at s minus score with no write, score = right - wrong,
# Pro states as policy_select2.load) split EXACTLY by how each question's state moves (no write -> E at s):
#   changed answer  both states in R / W / H: W->R (+2), R->W (-2), and the moves that touch H (H->R +1, H->W -1, R->H -1,
#                   W->H +1), plus the unchanged R->R / W->W / H->H (0)
#   silenced        R / W / H -> X: W->X (+1), R->X (-1), H->X (0)
#   released        X -> R / W / H: X->R (+1), X->W (-1), X->H (0)
#   (X->X contributes 0.)  The three parts add up to the total difference; the script checks it and stops otherwise.
# Questions: policy_select2.Split.qs. usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/e_gain_split.py [MODEL ...]      (tests: python3 -B Iteration_01/diag/e_gain_split.py --test)
import collections
import random
import sys

VAL = {"R": 1, "W": -1, "H": 0, "X": 0}
ANS = "RWH"


def split_gain(pairs):
    """pairs: list of (no-write state, written state). Returns the transition counter and the three net parts and the total."""
    t = collections.Counter(pairs)
    for a, b in t:
        if a not in VAL or b not in VAL:
            raise ValueError("state outside R/W/H/X: %s -> %s" % (a, b))
    part = {"changed": 0, "silenced": 0, "released": 0, "blank both": 0}
    for (a, b), n in t.items():
        d = (VAL[b] - VAL[a]) * n
        if a in ANS and b in ANS:
            part["changed"] += d
        elif a in ANS and b == "X":
            part["silenced"] += d
        elif a == "X" and b in ANS:
            part["released"] += d
        else:
            part["blank both"] += d
    total = sum(VAL[b] - VAL[a] for a, b in pairs)
    return t, part, total


def selftest():
    rng = random.Random(1)
    for _ in range(3000):
        n = rng.randrange(0, 40)
        pairs = [(rng.choice("RWHX"), rng.choice("RWHX")) for _ in range(n)]
        t, part, total = split_gain(pairs)
        assert part["blank both"] == 0, part
        assert part["changed"] + part["silenced"] + part["released"] == total, (pairs, part, total)
        assert sum(t.values()) == n
    # hand-made case: W->R, R->W, W->X, X->R, H->W, X->X
    t, part, total = split_gain([("W", "R"), ("R", "W"), ("W", "X"), ("X", "R"), ("H", "W"), ("X", "X")])
    assert part == {"changed": 2 - 2 - 1, "silenced": 1, "released": 1, "blank both": 0}, part
    assert total == 1, total
    t, part, total = split_gain([])
    assert total == 0 and part["changed"] == part["silenced"] == part["released"] == 0
    try:
        split_gain([("R", "?")])
        raise AssertionError("a '?' state must be refused")
    except ValueError:
        pass
    print("selftest: ok (3000 random cases + 3 fixed)")


ORDER = ["Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B", "Gemma-4-12B",
         "Gemma-4-12B stronger caution", "Gemma-4-31B"]


def main(names):
    sys.path.insert(0, "Iteration_01/diag")
    import policy_select2 as ps
    import c_first
    rows = []
    for name in names:
        m = c_first.MODELS[name]
        for i, split in ((1, "test"), (0, "tune")):
            sp = ps.Split(m, i)
            if m["judge"][i] != "pro":
                raise SystemExit("%s %s: not Pro" % (name, split))
            for s in "12":
                pairs = [(sp.st[(q, "baseline_x")], sp.st[(q, sp.cE(s))]) for q in sp.qs]
                t, part, total = split_gain(pairs)
                if part["changed"] + part["silenced"] + part["released"] != total or part["blank both"] != 0:
                    raise SystemExit("%s %s E %s: the parts do not add up" % (name, split, s))
                chk = ps.score(sp.states(("E", s)), 1) - ps.score(sp.states(("base",)), 1)
                if chk != total:
                    raise SystemExit("%s %s E %s: total %d differs from policy_select2.score %d" % (name, split, s, total, chk))
                rows.append((split, name, s, len(sp.qs), t, part, total))
    for split in ("test", "tune"):
        print("\n## %s split (Pro; score = right - wrong; E at s minus no write)" % split)
        print("| model | E | questions | total | changed: questions / net | W->R | R->W | W->W | R->R | moves with H: questions / net | "
              "silenced: questions / net | W->X | R->X | H->X | released: questions / net | X->R | X->W | X->H | parts sum |")
        print("|" + "---|" * 20)
        for sp_, name, s, n, t, part, total in rows:
            if sp_ != split:
                continue
            g = lambda a, b: t.get((a, b), 0)
            nch = sum(v for (a, b), v in t.items() if a in ANS and b in ANS)
            hmov = [(a, b) for a in ANS for b in ANS if "H" in (a, b) and a != b]
            nh = sum(g(a, b) for a, b in hmov)
            neth = sum((VAL[b] - VAL[a]) * g(a, b) for a, b in hmov)
            nsi = sum(g(a, "X") for a in ANS)
            nre = sum(g("X", b) for b in ANS)
            print("| %s | %s | %d | %+d | %d / %+d | %d | %d | %d | %d | %d / %+d | %d / %+d | %d | %d | %d | %d / %+d | %d | %d | %d | %+d |" % (
                name, s, n, total, nch, part["changed"], g("W", "R"), g("R", "W"), g("W", "W"), g("R", "R"), nh, neth,
                nsi, part["silenced"], g("W", "X"), g("R", "X"), g("H", "X"), nre, part["released"], g("X", "R"), g("X", "W"), g("X", "H"),
                part["changed"] + part["silenced"] + part["released"]))
        print("(changed net = 2 x (W->R - R->W) + moves with H net; H->H, R->R, W->W are 0; X->X counts toward no part and is 0)")


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        selftest()
    else:
        selftest()
        main(sys.argv[1:] or ORDER)
