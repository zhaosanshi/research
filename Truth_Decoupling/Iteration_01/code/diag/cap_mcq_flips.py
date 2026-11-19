# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# Why does writing E lower MMLU accuracy? Item-level look at the option-probability runs of
# scripts/capability/cap_mcq_prob.py (arms: no write, E 2, C = 2), per model:
#   1 picks per letter; 2 transitions of the pick (right/wrong before -> after); 3 flip rate by the no-write margin
#   (top1 - top2 log-probability, terciles); 4 mean change of each letter's share (letter bias); 5 entropy and top share
#   over the four letters; 6 change of the share of the no-write pick, for right and for wrong no-write picks;
#   7 options that contain a negation word (not / no / none / never / neither / false / cannot / wrong / incorrect / n't):
#   how often they are picked; 8 subjects with the largest summed change over the models.
# usage (needs pyarrow, so inside the container):
#   docker exec -i -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -I Iteration_01/diag/cap_mcq_flips.py \
#       data_external/mmlu_20261119/all/test-00000-of-00001.parquet Iteration_01/capability/mcq/MODEL.json ...
import collections
import json
import math
import re
import sys

import pyarrow.parquet as pq

L4 = "ABCD"
NEG = re.compile(r"(\bnot\b|\bno\b|\bnone\b|\bnever\b|\bneither\b|\bnor\b|\bfalse\b|\bcannot\b|\bwrong\b|\bincorrect\b|n't\b)", re.I)


def share(logp):
    m = max(logp.values())
    e = {k: math.exp(v - m) for k, v in logp.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def main():
    rows = pq.read_table(sys.argv[1]).to_pylist()
    per = collections.defaultdict(list)
    for r in rows:
        per[r["subject"]].append(r)
    subj = collections.defaultdict(lambda: [0, 0])     # subject -> [summed change of correct over models, n items x models]
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        arms = list(d["arms"])
        x = {r["id"]: r for r in d["arms"][arms[0]]["rows"]}
        ids = list(x)
        txt = {}
        for i in ids:
            _, s, k = i.split(":")
            txt[i] = per[s][int(k)]["choices"]
        print("\n== %s (%d items); arms %s" % (d["model"], len(ids), ", ".join(arms)))
        gold = collections.Counter(x[i]["gold"] for i in ids)
        print("    gold letters      " + " ".join("%s %3d" % (c, gold[c]) for c in L4))
        margin = {}
        for i in ids:
            v = sorted(x[i]["logp"].values(), reverse=True)
            margin[i] = v[0] - v[1]
        cut = sorted(margin.values())
        t1, t2 = cut[len(cut) // 3], cut[2 * len(cut) // 3]
        neg_all = sum(bool(NEG.search(o)) for i in ids for o in txt[i]) / (4.0 * len(ids))
        for arm in arms:
            a = {r["id"]: r for r in d["arms"][arm]["rows"]}
            picks = collections.Counter(a[i]["pick"] for i in ids)
            sh = {i: share(a[i]["logp"]) for i in ids}
            ent = sum(-sum(p * math.log(p) for p in sh[i].values() if p > 0) for i in ids) / len(ids)
            top = sum(max(sh[i].values()) for i in ids) / len(ids)
            negp = sum(bool(NEG.search(txt[i][L4.index(a[i]["pick"])])) for i in ids) / float(len(ids))
            print("    %-14s picks %s | accuracy %.3f | entropy %.3f, top share %.3f | picks a negation option %.3f (all options %.3f)" % (
                arm, " ".join("%s %3d" % (c, picks[c]) for c in L4), sum(a[i]["pick"] == a[i]["gold"] for i in ids) / float(len(ids)),
                ent, top, negp, neg_all))
            if arm == arms[0]:
                continue
            tr = collections.Counter()
            byt = collections.defaultdict(lambda: [0, 0])
            negflip = [0, 0]
            for i in ids:
                b_ok, a_ok = x[i]["pick"] == x[i]["gold"], a[i]["pick"] == a[i]["gold"]
                same = x[i]["pick"] == a[i]["pick"]
                key = ("R" if b_ok else "W") + ">" + ("R" if a_ok else "W") + ("" if (b_ok or a_ok or same) else "'")
                tr[key] += 1
                t = 0 if margin[i] < t1 else (1 if margin[i] < t2 else 2)
                byt[t][0] += (not same)
                byt[t][1] += 1
                if b_ok and not a_ok:
                    negflip[1] += 1
                    negflip[0] += bool(NEG.search(txt[i][L4.index(a[i]["pick"])])) and not NEG.search(txt[i][L4.index(x[i]["gold"])])
                if ":E" in arm:
                    s, _ = i.split(":")[1], 0
                    subj[s][0] += int(a_ok) - int(b_ok)
                    subj[s][1] += 1
            shx = {i: share(x[i]["logp"]) for i in ids}
            bias = {c: sum(sh[i][c] - shx[i][c] for i in ids) / len(ids) for c in L4}
            dtop_r = [sh[i][x[i]["pick"]] - shx[i][x[i]["pick"]] for i in ids if x[i]["pick"] == x[i]["gold"]]
            dtop_w = [sh[i][x[i]["pick"]] - shx[i][x[i]["pick"]] for i in ids if x[i]["pick"] != x[i]["gold"]]
            print("        pick transitions: %s (W>W' = wrong to another wrong)" % ", ".join("%s %d" % (k, tr[k]) for k in ("R>R", "R>W", "W>R", "W>W", "W>W'")))
            print("        pick changed, by no-write margin tercile (low, mid, high): %s" % ", ".join("%d/%d" % tuple(byt[t]) for t in (0, 1, 2)))
            print("        mean change of each letter's share: %s" % " ".join("%s %+.3f" % (c, bias[c]) for c in L4))
            print("        mean change of the share of the no-write pick: when it was right %+.3f (n=%d), when it was wrong %+.3f (n=%d)" % (
                sum(dtop_r) / max(1, len(dtop_r)), len(dtop_r), sum(dtop_w) / max(1, len(dtop_w)), len(dtop_w)))
            print("        right -> wrong: %d; of these the new pick is a negation option and the gold is not: %d" % (negflip[1], negflip[0]))
    order = sorted(subj.items(), key=lambda kv: kv[1][0])
    print("\n== subjects, summed change of correct picks under E over the models (items x models)")
    print("    worst: " + "; ".join("%s %+d (%d)" % (k, v[0], v[1]) for k, v in order[:12]))
    print("    best:  " + "; ".join("%s %+d (%d)" % (k, v[0], v[1]) for k, v in order[-6:]))
    print("    total %+d over %d" % (sum(v[0] for v in subj.values()), sum(v[1] for v in subj.values())))


if __name__ == "__main__":
    main()
