# -*- coding: utf-8 -*-
"""Read-only: one row per judged condition of an export (any arms, e.g. strength sweeps), each against its own base prompt.
Coverage = answered (right or wrong) out of all questions; accuracy = right among answered. Uses e44_round1_tables.py.
usage: e44_arms_table.py WORK [--judge allenai|pro]"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t
work = sys.argv[1]; judge = sys.argv[sys.argv.index("--judge") + 1] if "--judge" in sys.argv else "allenai"
gen, specs, ids, state, rule, words = t.load(work, judge)
N = len(ids)
def beta(arm):
    m = re.search(r"\.([pm])(\d+(?:\.\d+)?)$", arm); return (1 if m.group(1) == "p" else -1) * float(m.group(2))
print("# %s, judge %s, %d questions (each arm against its own base prompt, no write)\n" % (Path(work).name, judge, N))
print("| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |")
print("|---|---|---|---|---|---|---|---|---|---|")
rows = []
for name, s in specs.items():
    if s.get("kind") == "prompt_control" or "arm" not in s:
        r, w, h, x = t.counts(state, ids, name)
        rows.append((s.get("name"), -99, 0.0, "| %s | — | %d / %d / %d / %d | %d | %.1f%% | — | — | — | — | %.1f |" % (
            name.replace("baseline_", "prompt "), r, w, h, x, r + w, 100.0 * r / (r + w) if r + w else float("nan"), sum(words[(i, name)] for i in ids) / N)))
        continue
    b = s["base_condition"]; v, n = t.per_question(state, words, ids, b, name)
    r, w, h, x = t.counts(state, ids, name)
    kr = (sum(v["keep"]) / n["keep"]) if n["keep"] else None
    rows.append((s["arm"].rsplit(".", 1)[0], s["layer"], beta(s["arm"]), "| L%d %s | %+g | %d / %d / %d / %d | %d | %.1f%% | %s | %+d | %d, %s, %d / %d | %d / %d / %d / %d | %.1f |" % (
        s["layer"], s["arm"], beta(s["arm"]), r, w, h, x, r + w, 100.0 * r / (r + w) if r + w else float("nan"), t.ci(v["dR"])[0], sum(v["dAns"]),
        n["keep"], "undefined" if kr is None else "%+.1f pp" % (100 * kr), n["up"], n["down"], n["lost_right"], n["lost_wrong"], n["gain_right"], n["gain_wrong"],
        sum(words[(i, name)] for i in ids) / N)))
for _, _, _, line in sorted(rows, key=lambda z: (z[1], z[0], z[2])):
    print(line)
