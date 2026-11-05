#!/usr/bin/env python3
"""summarize_e2e.py -- tabulate e2e_sample.py output and print every spontaneous fake turn in full.

  python summarize_e2e.py results/e2e_*.jsonl [--full]
"""
import json
import sys


def main(argv):
    full = "--full" in argv
    paths = [a for a in argv if not a.startswith("--")]
    recs = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: (r["cell"], r["target_tokens"], r["filler_seed"]))
    print(f"{'cell':5s} {'len':>6s} {'seed':>4s} {'ctx':>6s} {'P(user)':>8s} {'p_role':>8s} | {'1st-draws hits':>22s} | "
          f"{'full fake':>9s} {'stopped':>7s} {')]}':>4s} | sec")
    tot_fake = tot_full = tot_hits = tot_draws = 0
    for r in recs:
        hits = r.get("first_hits", {})
        hs = ",".join(f"{k!r}:{v['n']}" for k, v in hits.items()) or "-"
        fulls = r.get("full_samples", [])
        n_paren = sum(1 for s in fulls if s.get("paren_tail"))
        print(f"{r['cell']:5s} {r['target_tokens']:6d} {r['filler_seed']:4d} {r['n_ctx_tokens']:6d} {r['p_user']:8.5f} {r['p_role']:8.5f} | "
              f"{r.get('first_hits_total',0):3d}/{r.get('n_first',0):<5d}{hs[:12]:>12s} | "
              f"{r.get('full_fake',0):4d}/{r.get('full_n',0):<4d} {r.get('full_stopped',0):4d}/{r.get('full_n',0):<2d} {n_paren:4d} | {r.get('seconds')}")
        tot_fake += r.get("full_fake", 0)
        tot_full += r.get("full_n", 0)
        tot_hits += r.get("first_hits_total", 0)
        tot_draws += r.get("n_first", 0)
    print(f"\nTOTAL full-turn samples: {tot_fake}/{tot_full} fake turns "
          f"({100.0 * tot_fake / max(1, tot_full):.2f}%) | first-token draws on watched tokens: {tot_hits}/{tot_draws} "
          f"({100.0 * tot_hits / max(1, tot_draws):.3f}%)")
    print("\n=== spontaneous fake turns (full samples, no forcing) ===")
    for r in recs:
        for s in r.get("full_samples", []):
            if s.get("fake_turn"):
                print(f"\n[{r['cell']} len={r['target_tokens']} seed={r['filler_seed']} sample={s['i']} first={s['first']!r} "
                      f"stopped={s['stopped']} self_answer={s['self_answer']} n_new={s['n_new']}]")
                print(s["text"] if full else s["text"][:700])
    print("\n=== continuations after a sampled role/special first token ===")
    for r in recs:
        for c in r.get("hit_continuations", [])[:3]:
            print(f"\n[{r['cell']} len={r['target_tokens']} seed={r['filler_seed']} first={c['first']!r} cat={c['cat']} "
                  f"fake={c['fake_turn']} self_answer={c['self_answer']} stopped={c['stopped']}]")
            print(c["text"] if full else c["text"][:500])


if __name__ == "__main__":
    main(sys.argv[1:])
