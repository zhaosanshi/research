#!/usr/bin/env python3
"""summarize_lens.py -- per-layer table of the logit lens: mean log10 P(user) and median rank(user) over filler seeds,
one column per condition, one block per model file.

  python summarize_lens.py results/lens_v1_*.jsonl
"""
import json
import math
import statistics
import sys
from collections import defaultdict


def main(paths):
    for p in paths:
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        by_cond = defaultdict(list)
        for r in recs:
            by_cond[r["condition"]].append(r)
        conds = sorted(by_cond)
        n_layers = recs[0]["n_states"]
        print(f"\n===== {p}  model={recs[0]['model'].split('/')[-1]}  tokenizer={recs[0]['tokenizer'].split('/')[-1]}  "
              f"seeds={sorted({r['filler_seed'] for r in recs})}  ctx~{int(statistics.mean(r['n_ctx_tokens'] for r in recs))}")
        print("  final layer: " + " | ".join(
            f"{c}: P(user)={statistics.mean(r['layers'][-1]['p_user'] for r in by_cond[c]):.2e} "
            f"P(eot)={statistics.mean(r['layers'][-1]['p_eot'] for r in by_cond[c]):.3f}" for c in conds))
        print("  layer | " + " | ".join(f"{c:>22s}" for c in conds) + "      (log10 P(user), median rank)")
        for i in range(n_layers):
            cells = []
            for c in conds:
                lp = [math.log10(max(r["layers"][i]["p_user"], 1e-30)) for r in by_cond[c]]
                rk = [r["layers"][i]["rank_user"] for r in by_cond[c]]
                cells.append(f"{statistics.mean(lp):7.2f}  r={int(statistics.median(rk)):>7d}")
            top1 = by_cond[conds[-1]][0]["layers"][i]["top3"][0][0]
            print(f"  {i:5d} | " + " | ".join(f"{x:>22s}" for x in cells) + f"   top1[{conds[-1]} s{by_cond[conds[-1]][0]['filler_seed']}]={top1!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
