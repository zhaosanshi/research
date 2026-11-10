#!/usr/bin/env python3
"""Read-only tables for the E43 review (host python, no torch): generation files + one single-judge working directory.

  1. per E arm and layer: answers ending with a sentence mark, mean words, negation words, refusals, and the
     per-position write size against the endpoint gap
  2. judged flips of an arm split by how the answer text changed (same text / changed, not longer / changed, longer)
  3. judged change split by whether the gold best answer contains a negation word
usage: e43_extra_tables.py RUN_DIR WORK_DIR [--family E] [--arms main recover md_main] [--split tune]
"""
import argparse, collections, json, os, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("--family", default="E")
ap.add_argument("--arms", nargs="+", default=["main", "recover", "md_main"])
ap.add_argument("--split", default="tune")
a = ap.parse_args()

NEG = re.compile(r"\b(no|not|never|none|nothing|nobody|neither|nor|cannot|without|isn't|aren't|wasn't|weren't|don't|doesn't|"
                 r"didn't|won't|wouldn't|can't|couldn't|shouldn't|hasn't|haven't|hadn't)\b|n't\b", re.I)
norm = lambda t: " ".join((t or "").casefold().split()).rstrip(".!? ")
words = lambda t: len((t or "").split())
endp = lambda t: (t or "").rstrip().endswith((".", "!", "?"))

frozen = json.loads((a.run / ("frozen_%s.json" % a.split)).read_text())
print("## frozen prompts (%s, n=%d): sentence-end / mean words / negation / refusals" % (a.split, len(frozen["questions"])))
for name in ("x", "x_E", "x_conf", "x_caut"):
    tx = [q["answers"][name]["text"] for q in frozen["questions"]]
    print("- %s: %d / %.1f / %d / %d" % (name, sum(map(endp, tx)), sum(map(words, tx)) / len(tx), sum(1 for t in tx if NEG.search(t)),
                                       sum(q["answers"][name]["refusal_rule"] for q in frozen["questions"])))
ft = a.run / "frozen_train.json"
if ft.exists():
    gold = [q["best_answer"] for q in json.loads(ft.read_text())["questions"] if q.get("gold_target", {}).get("usable")]
    print("- gold best answers used as E training targets (n=%d): sentence-end %d, mean words %.1f, negation %d" % (
        len(gold), sum(map(endp, gold)), sum(map(words, gold)) / len(gold), sum(1 for g in gold if NEG.search(g))))

print("\n## 1. %s arms on %s: sentence-end / mean words / negation / refusals | target, endpoint gap, mean |write| at prompt content positions, at decode steps" % (a.family, a.split))
print("| L | " + " | ".join(a.arms) + " |")
print("|---|" + "---|" * len(a.arms))
for L in range(32):
    tl = a.run / "train_logs" / ("%s_L%d.json" % (a.family, L))
    if not tl.exists():
        continue
    t = json.loads(tl.read_text())
    gap = t["endpoints"]["plus"]["mean"] - t["endpoints"]["minus"]["mean"]
    cells = []
    for arm in a.arms:
        p = a.run / "gen" / ("%s_L%d_%s_%s.json" % (a.family, L, arm, a.split))
        if not p.exists():
            cells.append("—"); continue
        d = json.loads(p.read_text()); rows = d["rows"]; tx = [r["text"] for r in rows]
        dg = [r["write"]["decode_gap_abs_mean"] for r in rows if r["write"]["decode_gap_abs_mean"] is not None]
        cells.append("%d / %.1f / %d / %d | %+.2f, %+.2f, %.2f, %.2f" % (
            sum(map(endp, tx)), sum(map(words, tx)) / len(tx), sum(1 for x in tx if NEG.search(x)), d["summary"]["refusal_after"],
            d["target"], gap, d["summary"]["prefill_gap_abs_mean"], sum(dg) / max(1, len(dg))))
    if any(c != "—" for c in cells):
        print("| %d | %s |" % (L, " | ".join(cells)))

gen = json.loads((a.work / "judge_input.json").read_text())
specs = {s["name"]: s for s in gen["condition_specs"]}
ev = {e["pair_id"]: e for e in gen["evaluations"]}
ids = gen["question_ids"]
st = {}
for line in (a.work / "judge2" / "single_allenai.jsonl").open():
    r = json.loads(line)
    st[(r["pair_id"], r["condition"])] = "R" if r["rule_abstain"] else r["label"]


def split(cond, base):
    out = {}
    for kind in ("same", "short", "long"):
        n = wr = rw = 0
        for i in ids:
            b, x = ev[i]["outputs"][base], ev[i]["outputs"][cond]
            k = "same" if norm(x) == norm(b) else ("long" if words(x) >= words(b) + 2 else "short")
            if k != kind:
                continue
            n += 1
            wr += (st[(i, base)], st[(i, cond)]) == ("B", "A")
            rw += (st[(i, base)], st[(i, cond)]) == ("A", "B")
        out[kind] = (n, wr, rw)
    return out


print("\n## 2. judged flips by how the text changed (AllenAI, rule refusals excluded). cell = n | wrong→right | right→wrong")
print("| condition (vs its base) | same text, only surface | changed, not longer (< +2 words) | changed and longer (≥ +2 words) |")
print("|---|---|---|---|")
for name, s in specs.items():
    if s["kind"] == "prompt_control" and name != "baseline_x_E":
        continue
    if s.get("arm") == "none":
        continue
    base = "baseline_x" if name == "baseline_x_E" else s["base_condition"]
    sp = split(name, base)
    print("| %s (vs %s) | %s |" % (name, base, " | ".join("%d | %d | %d" % sp[k] for k in ("same", "short", "long"))))

neg = {i: bool(NEG.search(ev[i]["best_answer"] or "")) for i in ids}
print("\n## 3. by gold best answer with / without a negation word (n = %d / %d). cell = answered-correct before→after, refusals before→after, wrong→right, right→wrong" % (
    sum(neg.values()), len(ids) - sum(neg.values())))
print("| condition (vs its base) | gold has negation | gold has no negation |")
print("|---|---|---|")
for name, s in specs.items():
    if (s["kind"] == "prompt_control" and name != "baseline_x_E") or s.get("arm") == "none":
        continue
    base = "baseline_x" if name == "baseline_x_E" else s["base_condition"]
    cells = []
    for flag in (True, False):
        sub = [i for i in ids if neg[i] == flag]
        c = collections.Counter(st[(i, name)] for i in sub); b = collections.Counter(st[(i, base)] for i in sub)
        t = collections.Counter((st[(i, base)], st[(i, name)]) for i in sub)
        cells.append("%d→%d, %d→%d, %d, %d" % (b["A"], c["A"], b["R"], c["R"], t[("B", "A")], t[("A", "B")]))
    print("| %s (vs %s) | %s | %s |" % (name, base, cells[0], cells[1]))
