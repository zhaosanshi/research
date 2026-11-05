#!/usr/bin/env python3
"""summarize_cross.py -- cell x seed table of P('user') at slot_start from turn_boundary_probe JSONL output.

  python summarize_cross.py results/cross_v1_32k_g*.jsonl results/cross_v1_8k_g3.jsonl
"""
import json
import statistics
import sys
from collections import defaultdict

FAKE_MARKERS = ("<USER_REQUEST>", "\n<", "<ADDITIONAL_METADATA>")


def main(paths):
    cells = defaultdict(dict)        # cond -> seed -> rec
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("measure") != "slot_start":
                continue
            cells[r["condition"]][r.get("filler_seed", 0)] = r

    def p_user(r):
        return dict(r.get("top_role") or []).get("user", 0.0) or 0.0

    conds = sorted(cells, key=lambda c: (c.split("_")[0], c))
    print(f"{'cond':10s} {'kind':6s} w c {'ctx':>6s} | " + " ".join(f"{'s'+str(s):>8s}" for s in range(1, 6)) +
          f" | {'mean':>8s} {'max':>8s} | {'p_eot':>7s} {'p_role':>7s}  forced-'user' looks like a fake turn?")
    for c in conds:
        recs = cells[c]
        seeds = sorted(recs)
        kw = recs[seeds[0]]["kwargs"]
        vals = {s: p_user(recs[s]) for s in seeds}
        ctx = int(statistics.mean(recs[s]["n_ctx_tokens"] for s in seeds))
        mean = statistics.mean(vals.values())
        mx = max(vals.values())
        pe = statistics.mean(recs[s]["p_eot"] or 0 for s in seeds)
        pr = statistics.mean(recs[s]["p_role"] or 0 for s in seeds)
        ff = [(recs[s].get("force_first") or {}).get("user") or "" for s in seeds]
        n_fake = sum(1 for t in ff if any(m in t for m in FAKE_MARKERS))
        cols = " ".join(f"{vals[s]:8.5f}" if s in vals else f"{'-':>8s}" for s in range(1, 6))
        print(f"{c:10s} {kw.get('kind','coding'):6s} {int(bool(kw.get('wrapper')))} {kw.get('contam_k',0)} {ctx:6d} | {cols} | "
              f"{mean:8.5f} {mx:8.5f} | {pe:7.5f} {pr:7.5f}  {n_fake}/{len(ff)}")
    print()
    for c in conds:
        recs = cells[c]
        s = sorted(recs)[0]
        r = recs[s]
        ff = (r.get("force_first") or {}).get("user") or ""
        print(f"--- {c} seed {s}: top3={r['top10'][:3]}")
        print(f"    unforced : {r.get('unforced', '')[:120]!r}")
        print(f"    force[user]: {ff[:200]!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
