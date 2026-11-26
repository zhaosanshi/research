"""Read question text and subset IDs only for held-out inventory; no unused labels/responses scored."""

import csv
import json
import re
import sys
import collections
from pathlib import Path

sys.path.insert(0, "Iteration_01/diag")
import gate_capture_1126 as cap
import gate_fit_1126 as fit

norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
raw = Path("Iteration_01/newsets/raw_falseqa_20261123")
bank = Path("Iteration_01/newsets/falseqa/falseqa_full_bank_20261123.jsonl")
used = [json.loads(ll) for ll in bank.read_text().splitlines() if ll.strip()]
dev = [r for r in used if r["set"] == "dev"]
held = [r for r in used if r["set"] == "heldout"]
topics = set(map(lambda r: norm(r["question"]), used))
pid = {r["pair_id"] for r in held}
model = fit.MODELS[0]
cfg = json.load(open(cap.FROZEN_MODELS))["models"][model]
sets = cap.items_of(model, cfg)
print("gate input groups", [(k, len(x["items"])) for k, x in sets.items()])
seen = {norm(v[1]) for x in sets.values() for v in x["items"]} | topics
for split in ("test", "valid"):
    rows = [
        norm(r["question"]) for r in csv.DictReader((raw / (split + ".csv")).open())
    ]
    n = len(rows) // 2
    old = {r["pair_id"] for r in (held if split == "test" else dev)}
    rem = {}
    same = set()
    for i, (x, y) in enumerate(zip(rows[:n], rows[n:])):
        k = "fq" + split[0] + str(i).zfill(4)
        if x == y:
            same.add(k)
            continue
        if k not in old:
            rem[k] = (x, y)
    overlap = [k for k, (a, b) in rem.items() if a in seen or b in seen]
    dup = collections.Counter(q for r in rem.values() for q in r)
    dups = [k for k, (a, b) in rem.items() if dup[a] > 1 or dup[b] > 1]
    print(
        split,
        "pairs",
        n,
        "same-text",
        len(same),
        "sampled",
        len(old),
        "remaining",
        len(rem),
        "overlap current gate inputs or used bank",
        len(overlap),
        "within remaining redundant-question pairs",
        len(dups),
        "eligible exclude all these",
        len(set(rem) - set(overlap) - set(dups)),
    )
