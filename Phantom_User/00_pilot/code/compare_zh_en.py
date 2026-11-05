#!/usr/bin/env python3
"""compare_zh_en.py -- side-by-side P('user') per condition: Chinese materials (pilot) vs English materials (paper).

Probe records (turn_boundary_probe.py output, one line per condition x filler seed):
  python compare_zh_en.py --zh results/xmodel_v1_Llama-3.1-8B-Instruct.jsonl --en results/en_cross_Llama-3.1-8B-Instruct.jsonl
  python compare_zh_en.py --zh results/trig_v1_g*.jsonl results/read_v1_g*.jsonl results/paren_v1_g0.jsonl \
                               results/place_v1_g*.jsonl results/ask_probe_v1.jsonl --en results/en_controls_Llama-3.1-8B-Instruct.jsonl
Sampling records (e2e_sample.py output, one line per cell x seed):
  python compare_zh_en.py --e2e --zh results/e2e_m10_32k.jsonl --en results/en_e2e_m10.jsonl results/en_e2e_m13.jsonl
"""
import argparse
import json
import math
import statistics
from collections import defaultdict

import re

# A forced-'user' continuation counts as a fake turn only if it carries the wrapper or has the role-line form
# ('user' directly followed by a line break and a message).  'user, I will move...' (vocative / sentence start) and
# an echoed '<tool_result>' block do not count.
FAKE_WRAP = re.compile(r"<USER_REQUEST>|<ADDITIONAL_METADATA>|The current local time is")
FAKE_ROLE_LINE = re.compile(r"^(?:user|User|USER|human|Human|用户)[ \t]*[:：]?[ \t]*\n\S")


def is_fake(text: str) -> bool:
    return bool(FAKE_WRAP.search(text) or FAKE_ROLE_LINE.search(text))


def load(paths):
    cells = defaultdict(dict)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "condition" in r:
                    cells[r["condition"]][r.get("filler_seed", 0)] = r
    return cells


def p_user(r):
    return dict(r.get("top_role") or []).get("user", 0.0) or 0.0


def gmean(vals, floor=1e-9):
    vals = [max(v, floor) for v in vals]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def fake_count(recs):
    ff = [(r.get("force_first") or {}).get("user") or "" for r in recs]
    return sum(1 for t in ff if is_fake(t)), len(ff)


def probe_table(zh, en):
    conds = sorted(set(zh) | set(en), key=lambda c: (c.split("_")[0], c))
    print(f"{'cond':15s} {'meas':10s} | {'ZH gmean':>9s} {'ZH med':>9s} {'ZH max':>9s} n | {'EN gmean':>9s} {'EN med':>9s} {'EN max':>9s} n "
          f"| {'EN/ZH':>7s} | forced-user fake ZH   EN   | p_eot ZH   EN")

    def stats(recs):
        if not recs:
            return None
        v = [p_user(r) for r in recs.values()]
        return gmean(v), statistics.median(v), max(v), len(v)

    def fmt(s):
        return f"{s[0]:9.2e} {s[1]:9.2e} {s[2]:9.2e} {s[3]}" if s else f"{'-':>9s} {'-':>9s} {'-':>9s} -"

    for c in conds:
        rz, re_ = zh.get(c, {}), en.get(c, {})
        sz, se = stats(rz), stats(re_)
        meas = (next(iter(rz.values())) if rz else next(iter(re_.values())))["measure"]
        fz = fake_count(list(rz.values())) if rz else (0, 0)
        fe = fake_count(list(re_.values())) if re_ else (0, 0)
        pez = statistics.mean((r["p_eot"] or 0) for r in rz.values()) if rz else float("nan")
        pee = statistics.mean((r["p_eot"] or 0) for r in re_.values()) if re_ else float("nan")
        ratio = f"{se[0] / sz[0]:7.2f}" if (sz and se and sz[0] > 0) else f"{'-':>7s}"
        print(f"{c:15s} {meas:10s} | {fmt(sz)} | {fmt(se)} | {ratio} | {fz[0]}/{fz[1]:<6} {fe[0]}/{fe[1]:<6} | {pez:6.3f} {pee:6.3f}")


def load_e2e(paths):
    cells = defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "cell" in r:
                    cells[r["cell"]].append(r)
    return cells


def e2e_table(zh, en):
    for c in sorted(set(zh) | set(en)):
        for name, recs in (("ZH", zh.get(c, [])), ("EN", en.get(c, []))):
            if not recs:
                continue
            n_first = sum(r["n_first"] for r in recs)
            user_hits = sum(v["n"] for r in recs for k, v in r["first_hits"].items() if k.strip().lower() == "user")
            any_hits = sum(r["first_hits_total"] for r in recs)
            conts = [h for r in recs for h in r["hit_continuations"]]
            fake_conts = sum(1 for h in conts if h["fake_turn"])
            full_n = sum(r["full_n"] for r in recs)
            full_fake = sum(r["full_fake"] for r in recs)
            pu = gmean([r["p_user"] for r in recs])
            ctx = int(statistics.mean(r["n_ctx_tokens"] for r in recs))
            print(f"{c:12s} {name} seeds={len(recs)} ctx~{ctx:6d} P(user) gmean={pu:.2e} | first-token draws: 'user' {user_hits}/{n_first}, "
                  f"any watched {any_hits}/{n_first} | hit continuations fake {fake_conts}/{len(conts)} | full samples fake {full_fake}/{full_n}")
        # show the first fake continuation of each language so the shape can be eyeballed
        for name, recs in (("ZH", zh.get(c, [])), ("EN", en.get(c, []))):
            for r in recs:
                hits = [h for h in r["hit_continuations"] if h["fake_turn"]] + [s for s in r["full_samples"] if s["fake_turn"]]
                if hits:
                    print(f"    {name} example (seed {r['filler_seed']}): {hits[0]['text'][:300]!r}")
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", nargs="+", default=[])
    ap.add_argument("--en", nargs="+", default=[])
    ap.add_argument("--e2e", action="store_true", help="inputs are e2e_sample.py records")
    args = ap.parse_args()
    if args.e2e:
        e2e_table(load_e2e(args.zh), load_e2e(args.en))
    else:
        probe_table(load(args.zh), load(args.en))


if __name__ == "__main__":
    main()
