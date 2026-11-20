# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# The first MMLU capability run (the model writes its own answer and may decline; 1,000 items; diag/cap_score.py), scored
# the way the TruthfulQA answers are scored: right +1, no pick 0, wrong -1. cap_score.py reports accuracy with every item
# in the denominator, where an item the model declines counts like a wrong one; under the paper's own score a declined
# item costs nothing and a wrong one costs a point. Same inputs, same checks, same per-item states (cap_score.load and
# final_records), same systems: no write, E at strength 2, the frozen N1, C = 2 on every item.
# Also printed: where the items go that E takes away from "right" and from "wrong" (to no pick / to the other state).
# usage (host or container, from the project root): python3 -B Iteration_01/diag/cap_net_score.py [--batch 8] [MODEL ...]
import json
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
import cap_score as CS

fr = CS.fr


def ci(v, n_boot=4000, seed=20261119):
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return 100.0 * sum(v) / n, 100.0 * bs[int(0.025 * n_boot)], 100.0 * bs[int(0.975 * n_boot) - 1]


def val(x):
    return (x == "R") - (x == "W")


def main():
    args = sys.argv[1:]
    batch = 8
    if "--batch" in args:
        i = args.index("--batch")
        batch = int(args[i + 1])
        del args[i:i + 2]
    names = args or list(CS.ORDER)
    with open(CS.CFG, encoding="utf-8") as f:
        M = json.load(f)["models"]
    gold = CS.gold_of("mmlu")
    print("MMLU, the model writes its own answer and may decline (1,000 items): net score per 100 items = right - wrong; changes vs no write [95%, items resampled]")
    joint = {}
    for name in names:
        m = M[name]
        st, blank, tx, why, order, cand = CS.load("mmlu", name, batch, m, gold)
        sp = CS.CapSplit(m["el"], m["cl"], st, blank, order)
        t = fr.Texts(tx, tx)
        k1, k2 = CS.LADNAME[m["caps"]["e_blank"]], CS.LADNAME[m["caps"]["own_blank"]]
        base = fr.states(fr.records(sp, ("base",), t))
        n1, _, _ = CS.n1_steps(sp, t, k1, k2)
        rows = [("E 2", fr.states(fr.records(sp, ("E", "2"), t))), ("N1", fr.states(n1)), ("C 2 on every item", fr.states(fr.records(sp, ("C", "2"), t)))]
        n = len(base)
        print("\n== %s: %d items | no write: right %d, wrong %d, no pick %d, net %+.1f per 100" % (
            name, n, base.count("R"), base.count("W"), base.count("X"), 100.0 * sum(map(val, base)) / n))
        for label, s in rows:
            d = [val(x) - val(y) for x, y in zip(s, base)]
            a = [(x == "R") - (y == "R") for x, y in zip(s, base)]
            m_, lo, hi = ci(d)
            am, alo, ahi = ci(a)
            flow = {"R>X": 0, "R>W": 0, "W>X": 0, "W>R": 0, "X>R": 0, "X>W": 0}
            for x, y in zip(s, base):
                if x != y:
                    flow["%s>%s" % (y, x)] += 1
            print("    %-18s right %3d, wrong %3d, no pick %3d | net change %+.1f [%+.1f, %+.1f] | accuracy change %+.1f [%+.1f, %+.1f] | moved: right to no pick %d, right to wrong %d, wrong to no pick %d, wrong to right %d, no pick to right %d, no pick to wrong %d" % (
                label, s.count("R"), s.count("W"), s.count("X"), m_, lo, hi, am, alo, ahi,
                flow["R>X"], flow["R>W"], flow["W>X"], flow["W>R"], flow["X>R"], flow["X>W"]))
            joint.setdefault(label, []).append(d)
    if len(names) > 1:
        print("\n== the listed models together (same items, resampled jointly): mean net change per 100 items")
        for label, per in joint.items():
            n = len(per[0])
            q = [sum(v[i] for v in per) / float(len(per)) for i in range(n)]
            print("    %-18s %+.2f [%+.2f, %+.2f]" % ((label,) + ci(q)))


if __name__ == "__main__":
    main()
