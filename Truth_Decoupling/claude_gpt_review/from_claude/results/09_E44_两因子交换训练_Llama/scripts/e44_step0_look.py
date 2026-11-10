# -*- coding: utf-8 -*-
"""Read-only look at the E44 step-0 frozen answers (no model, CPU). Prints per-prompt counts, pairwise agreement,
the 2x2 factor table of refusals, and a few raw rows."""
import json, re, sys
run = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else "tune"
d = json.load(open("%s/frozen_%s.json" % (run, split), encoding="utf-8"))
qs = d["questions"]
names = list(d["prompt_systems"])
norm = lambda t: re.sub(r"[^a-z0-9 ]+", "", (t or "").lower()).strip()
print("split %s, %d questions, %d prompts" % (split, len(qs), len(names)))
print("\n| prompt | refusals (rule) | unfinished | ends .!? | mean words | system text tail |")
print("|---|---|---|---|---|---|")
for n in names:
    recs = [q["answers"][n] for q in qs]
    print("| %s | %d | %d | %d | %.1f | ...%s |" % (
        n, sum(bool(r.get("refusal_rule")) for r in recs), sum(not r["finished"] for r in recs),
        sum((r["text"] or "").rstrip().endswith((".", "!", "?")) for r in recs),
        sum(len((r["text"] or "").split()) for r in recs) / len(recs), d["prompt_systems"][n][-70:].replace("|", "/")))
pairs = [("x", "conf"), ("x", "caut"), ("conf", "caut"), ("conf", "conf_E1"), ("caut", "caut_E1"), ("conf_E1", "caut_E1"),
         ("x", "x_E1"), ("conf", "conf_E2"), ("conf_E1", "conf_E2"), ("conf", "conf_M"), ("conf_M", "conf_E1"), ("x", "x_F")]
print("\n| pair | same answer (normalized) | refusal a only | refusal b only | both refuse |")
print("|---|---|---|---|---|")
for a, b in pairs:
    if a not in names or b not in names:
        continue
    same = ra = rb = rr = 0
    for q in qs:
        A, B = q["answers"][a], q["answers"][b]
        same += norm(A["text"]) == norm(B["text"])
        fa, fb = bool(A.get("refusal_rule")), bool(B.get("refusal_rule"))
        ra += fa and not fb; rb += fb and not fa; rr += fa and fb
    print("| %s vs %s | %d | %d | %d | %d |" % (a, b, same, ra, rb, rr))
print("\nraw rows (first 6 questions):")
for q in qs[:6]:
    print("\nQ:", q["question"])
    for n in ("x", "conf", "conf_E1", "conf_E2", "conf_M", "caut", "caut_E1"):
        if n in names:
            print("  %-8s %s" % (n, q["answers"][n]["text"][:150].replace("\n", " ")))
