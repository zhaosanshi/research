# -*- coding: utf-8 -*-
"""Rules R5 and R6 of from_claude/45 (Claude, 2026-11-14 01:xx; read-only, no model, no judge call).
R5: a C layer qualifies when, at strength 0.25 on the plain prompt x, the toward-caution arm x.w.p0.25 meets the frozen
    criterion v4 (column A, tune questions) under BOTH judges: the KEY line of e44_round1_tables.py cells (family C,
    strength 0.25) reads A=TWO-WAY or A=ONE-WAY(x.w.p0.25) for judge allenai and for judge pro.
R6: the C layer of E+C is the lowest R5 layer above the R3 E layer (none: no E+C under the rule).
Input: the logs of the runner's tables jobs (STAGE=tables BRANCH=C TAG=S STRENGTH=0.25), one per shortlisted layer, which
contain the KEY lines of both judges.
usage: ff_rule_c_pick.py --e-layer 20 --table LAYER:LOG [--table ...] --out JSON
prints the per-layer statuses, then "r5: <layers>" and "r6: <layer or none>"; writes JSON (an existing file that differs is
never overwritten)."""
import argparse
import json
import re
from pathlib import Path

KEY = re.compile(r"^KEY family=C judge=(allenai|pro) layer=(\d+) A=(\S+) B=(\S+)")
OK = ("TWO-WAY", "ONE-WAY(x.w.p0.25)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e-layer", type=int, required=True)
    p.add_argument("--table", action="append", required=True, help="LAYER:LOG of the layer's tables job at strength 0.25")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    status = {}
    for spec in a.table:
        layer, log = spec.split(":", 1)
        layer = int(layer)
        if layer in status:
            raise SystemExit("layer %d given twice" % layer)
        found = {}
        for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
            m = KEY.match(line)
            if m:
                if int(m.group(2)) != layer:
                    raise SystemExit("%s holds a KEY line of layer %s, expected %d" % (log, m.group(2), layer))
                if "EXPLORATORY strength=0.25" not in line:
                    raise SystemExit("%s: a KEY line is not at strength 0.25: %s" % (log, line))
                if m.group(1) in found:
                    raise SystemExit("%s: two KEY lines for judge %s" % (log, m.group(1)))
                found[m.group(1)] = {"A": m.group(3), "B": m.group(4)}
        if set(found) != {"allenai", "pro"}:
            raise SystemExit("%s: KEY lines for both judges are required, found %s" % (log, sorted(found)))
        for j, f in found.items():   # red team 01:51: an "incomplete" (a missing arm) must not silently read as "not in R5"
            if not (f["A"] in ("TWO-WAY", "not_met") or re.fullmatch(r"ONE-WAY\(x\.w\.[pm]0\.25\)", f["A"])):
                raise SystemExit("%s: judge %s has column A = %s, outside TWO-WAY / ONE-WAY(arm) / not_met" % (log, j, f["A"]))
        status[layer] = found
    r5 = sorted(L for L, f in status.items() if all(f[j]["A"] in OK for j in ("allenai", "pro")))
    above = [L for L in r5 if L > a.e_layer]
    r6 = min(above) if above else None
    print("| C layer | AllenAI A / B | Pro A / B | R5 (toward caution 0.25 met under both judges, column A) |")
    print("|---|---|---|---|")
    for L in sorted(status):
        f = status[L]
        print("| %d | %s / %s | %s / %s | %s |" % (L, f["allenai"]["A"], f["allenai"]["B"], f["pro"]["A"], f["pro"]["B"], "yes" if L in r5 else "no"))
    print("\nr5: %s" % (" ".join(str(L) for L in r5) or "none"))
    print("r6: %s" % ("none" if r6 is None else r6))
    out = {"rule": "R5 / R6 of from_claude/45: R5 = C layers whose x.w.p0.25 meets the frozen criterion v4, column A, under both "
                   "judges (KEY A in %s); R6 = the lowest R5 layer above the R3 E layer" % (list(OK),),
           "e_layer": a.e_layer, "status": {str(L): status[L] for L in sorted(status)}, "r5": r5, "r6": r6}
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    po = Path(a.out)
    if po.exists():
        if po.read_text(encoding="utf-8") != text:
            raise SystemExit("%s exists and differs: not overwritten" % po)
    else:
        po.parent.mkdir(parents=True, exist_ok=True)
        po.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
