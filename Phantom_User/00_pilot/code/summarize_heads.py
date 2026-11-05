#!/usr/bin/env python3
"""summarize_heads.py -- aggregate head_attrib.py output over filler seeds.
Per model file and condition: sanity checks, per-layer attention/MLP DLA(user), the heads with the largest mean
DLA(user) (promoters) and the most negative (suppressors) with their attention mass on the fake 'user' tokens, the
earlier real user headers and the current header, and the ablation table.

  python summarize_heads.py results/heads_v1_*.jsonl [--top 10]
"""
import argparse
import json
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--layers-from", type=int, default=None, help="print per-layer rows from this layer (default: half)")
    args = ap.parse_args()
    for p in args.paths:
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        by_cond = defaultdict(list)
        for r in recs:
            by_cond[r["condition"]].append(r)
        n_layers = len(recs[0]["per_layer"])
        start = args.layers_from if args.layers_from is not None else n_layers // 2
        print(f"\n######## {p}  model={recs[0]['model'].split('/')[-1]}  seeds={sorted({r['filler_seed'] for r in recs})}")
        worst_decomp = max(r["decomp_max_abs_err"] for r in recs)
        worst_attn = max(pl["attn_recomp_rel_err"] for r in recs for pl in r["per_layer"])
        logit_gap = max(abs(r["logit_user"] - r["logit_user_from_resid"]) for r in recs)
        print(f"  checks: residual decomposition max abs err {worst_decomp:.3g}; logit(user) true vs from-residual max gap {logit_gap:.3g}; "
              f"attention recomputation max rel err {worst_attn:.3g}")
        for cond in sorted(by_cond):
            rs = by_cond[cond]
            n = len(rs)
            pu = statistics.mean(r["p_user"] for r in rs)
            print(f"\n  == {cond}  ({rs[0]['desc']})  n={n}  mean P(user)={pu:.2e}  top1={[r['top1'] for r in rs]}  "
                  f"fake 'user' tokens in context={[r['n_spans']['fake_user_tok'] for r in rs]}")
            # per layer
            print(f"     layer   attn DLA(user)   mlp DLA(user)   attn DLA(diff)   mlp DLA(diff)")
            for i in range(start, n_layers):
                a = statistics.mean(r["per_layer"][i]["attn_dla_user"] for r in rs)
                m = statistics.mean(r["per_layer"][i]["mlp_dla_user"] for r in rs)
                ad = statistics.mean(r["per_layer"][i]["attn_dla_diff"] for r in rs)
                md = statistics.mean(r["per_layer"][i]["mlp_dla_diff"] for r in rs)
                print(f"     L{i:2d}      {a:+8.3f}        {m:+8.3f}        {ad:+8.3f}        {md:+8.3f}")
            emb = statistics.mean(r["embed_dla_user"] for r in rs)
            tot_attn = sum(statistics.mean(r["per_layer"][i]["attn_dla_user"] for r in rs) for i in range(n_layers))
            tot_mlp = sum(statistics.mean(r["per_layer"][i]["mlp_dla_user"] for r in rs) for i in range(n_layers))
            print(f"     totals: embed {emb:+.3f}  attn {tot_attn:+.3f}  mlp {tot_mlp:+.3f}  = logit(user) {emb + tot_attn + tot_mlp:+.3f}  "
                  f"(true mean {statistics.mean(r['logit_user'] for r in rs):+.3f})")
            # heads: mean over seeds
            agg = defaultdict(lambda: {"dla": [], "diff": [], "fake": [], "hdrprev": [], "curhdr": [], "bos": [], "tool": [], "top": []})
            for r in rs:
                for d in r["heads"]:
                    k = (d["layer"], d["head"])
                    agg[k]["dla"].append(d["dla_user"])
                    agg[k]["diff"].append(d["dla_diff"])
                    agg[k]["fake"].append(d["mass"]["fake_user_tok"])
                    agg[k]["hdrprev"].append(d["mass"]["hdr_user_prev"])
                    agg[k]["curhdr"].append(d["mass"]["cur_hdr"])
                    agg[k]["bos"].append(d["mass"]["bos"])
                    agg[k]["tool"].append(d["mass"]["tool_body"] + d["mass"]["tool_hdr_user"])
                    agg[k]["top"].append(d["top5"][0][1] if d["top5"][0][1] != "<|begin_of_text|>" else (d["top5"][1][1] if len(d["top5"]) > 1 else ""))
            rows = []
            for k, v in agg.items():
                rows.append((k, statistics.mean(v["dla"]), min(v["dla"]), max(v["dla"]), statistics.mean(v["diff"]),
                             statistics.mean(v["fake"]), statistics.mean(v["hdrprev"]), statistics.mean(v["curhdr"]),
                             statistics.mean(v["bos"]), statistics.mean(v["tool"]), v["top"]))
            rows.sort(key=lambda x: -x[1])
            hdr = "     L.H      mean DLA(user) [min,max]   DLA(diff) | attn mass: fakeUserTok hdrUserPrev curHdr bos tool | top non-BOS token per seed"
            print("     promoters (largest mean DLA(user)):")
            print(hdr)
            for k, mdla, lo, hi, mdiff, fk, hp, ch, bos, tool, tops in rows[:args.top]:
                print(f"     L{k[0]:2d}.H{k[1]:<2d}  {mdla:+8.3f} [{lo:+6.2f},{hi:+6.2f}]  {mdiff:+8.3f} | "
                      f"{fk:.3f} {hp:.3f} {ch:.3f} {bos:.3f} {tool:.3f} | {tops}")
            print("     suppressors (most negative mean DLA(user)):")
            for k, mdla, lo, hi, mdiff, fk, hp, ch, bos, tool, tops in rows[-5:][::-1]:
                print(f"     L{k[0]:2d}.H{k[1]:<2d}  {mdla:+8.3f} [{lo:+6.2f},{hi:+6.2f}]  {mdiff:+8.3f} | "
                      f"{fk:.3f} {hp:.3f} {ch:.3f} {bos:.3f} {tool:.3f} | {tops}")
            # ablations
            print("     ablations at the slot (geometric-mean P(user) over seeds; before -> after):")
            ab = defaultdict(list)
            for r in rs:
                for a in r["ablations"]:
                    ab[(a["kind"], a["k"])].append((r["p_user"], a["p_user"], r["p_top1"], a["p_top1"]))
            def gmean(xs):
                return 10 ** statistics.mean(__import__("math").log10(max(x, 1e-30)) for x in xs)
            for (kind, k), v in sorted(ab.items(), key=lambda x: (x[0][1], x[0][0])):
                before = gmean([x[0] for x in v]); after = gmean([x[1] for x in v])
                pt_b = statistics.mean(x[2] for x in v); pt_a = statistics.mean(x[3] for x in v)
                print(f"       {kind:13s} k={k:2d}   P(user) {before:.2e} -> {after:.2e}  (x{after / before:.3g})   P(top1) {pt_b:.3f} -> {pt_a:.3f}")
            # which heads were in the top-10 ablation sets, by frequency
            cnt = defaultdict(int)
            for r in rs:
                for a in r["ablations"]:
                    if a["kind"] == "top_dla_user" and a["k"] == 10:
                        for lh in a["heads"]:
                            cnt[tuple(lh)] += 1
            common = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
            print("       heads in the top-10 set, (layer,head):count over seeds:", ", ".join(f"L{l}.H{h}:{c}" for (l, h), c in common[:14]))


if __name__ == "__main__":
    main()
