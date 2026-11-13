# -*- coding: utf-8 -*-
"""Read-only summary of exploration chain g (from_claude/44 section 5.8): the learned E direction at strengths 1 / 2 / 3 on
the must-answer base (conf), next to the prompt with the truth sentence (baseline_conf_<E> against baseline_conf), the
random directions and the mean-difference direction of the same strength; per model and layer, one table per judge.
Every number uses the helpers of e44_round1_tables.py (same labels, paired bootstrap, G_same = right-answer change on
the questions answered both times whose answer grew by at most one word).
usage: ff_explore_dose_table.py --e-on E2 --work LABEL:LAYER:WORK_DIR [--work ...] [--strengths 1 2 3]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402


def cell(state, words, ids, b, c):
    v, n = t.per_question(state, words, ids, b, c)
    return {"dR": sum(v["dR"]), "ci": t.ci(v["dR"])[0], "dAns": sum(v["dAns"]), "same": sum(v["same"]), "n_same": n["same"],
            "words": sum(words[(i, c)] for i in ids) / len(ids), "v": v}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e-on", required=True)
    p.add_argument("--work", action="append", required=True, help="LABEL:LAYER:WORK_DIR")
    p.add_argument("--strengths", nargs="+", default=["1", "2", "3"])
    a = p.parse_args()
    works = []
    for w in a.work:
        label, layer, path = w.split(":", 2)
        works.append((label, int(layer), path))
    base, prompt = "baseline_conf", "baseline_conf_" + a.e_on
    for judge in ("pro", "allenai"):
        print("## judge %s: must-answer base (conf), change in right answers against conf without any write\n" % judge)
        print("Prompt = conf with the %s sentence, no write. Direction = conf, no sentence, the learned E direction at the "
              "given strength (strength 1 = the frozen size of the write). Each cell: dR [95%%] / dAns / G_same (n) / mean words. "
              "Random = the smaller margin of the direction over the two random directions (both signs) of the same "
              "strength, a point estimate (the per-layer cells tables carry the intervals); mean difference = dR of the "
              "driver's md arm of the same strength, which writes strength x |mean difference| (not size-matched to the "
              "learned write, strength x gap).\n" % a.e_on)
        head = "| model | layer | conf alone: words | prompt + %s |" % a.e_on + "".join(" direction %s |" % s for s in a.strengths) + \
               " margin over random (%s) | mean difference (%s) |" % (" / ".join(a.strengths), " / ".join(a.strengths))
        print(head)
        print("|" + "---|" * (head.count("|") - 1))
        for label, layer, path in works:
            gen, specs, ids, state, rule, words = t.load(path, judge)
            pc = cell(state, words, ids, base, prompt)
            row = "| %s | %d | %.1f | %s / %+d / %+d (%d) / %.1f |" % (
                label, layer, sum(words[(i, base)] for i in ids) / len(ids), pc["ci"], pc["dAns"], pc["same"], pc["n_same"], pc["words"])
            margins, mds = [], []
            for s in a.strengths:
                main_c = "E_L%d_conf-w-p%s" % (layer, s.replace(".", "-"))
                if main_c not in specs:
                    row += " — |"
                    margins.append("—"); mds.append("—")
                    continue
                m = cell(state, words, ids, base, main_c)
                row += " %s / %+d / %+d (%d) / %.1f |" % (m["ci"], m["dAns"], m["same"], m["n_same"], m["words"])
                rs = ["E_L%d_conf-rand-%s%s" % (layer, sg, s.replace(".", "-")) for sg in "pm"]
                if all(r in specs for r in rs):
                    margins.append("%+d" % min(m["dR"] - cell(state, words, ids, base, r)["dR"] for r in rs))
                else:
                    margins.append("—")
                md = "E_L%d_conf-md-p%s" % (layer, s.replace(".", "-"))
                mds.append("%+d" % cell(state, words, ids, base, md)["dR"] if md in specs else "—")
            row += " %s | %s |" % (" / ".join(margins), " / ".join(mds))
            print(row)
        print()


if __name__ == "__main__":
    main()
