"""Read-only: Qwen3.5-9B, (QA)^2 100 tune questions, the four E+C arms: pilot run vs the 570-run's copy.
Counts question x dose cells whose answer text differs, and cells with the same text but a different score."""
import json
import sys
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import qaqa_pilot_table as Q
m = "qwen3.5-9b"
s = Q.SET[m][1]
pilot = Q.Raw(m, "qaqa", Q.TUNE_BANK, ["test_tune100"])
full = Q.Raw(m, "qaqafull", "Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl",
             ("test_full570", "test_full570_tuned", "test_full570_strong"))
ids = list(pilot.ids)
assert len(ids) == 100 and set(ids) <= set(full.ids)
td = sc = 0
for k in Q.LAD:
    a = pilot.ec(s, k); b = full.ec(s, k)
    pilot.need(a); full.need(b)
    t = sum(pilot.text(a, q).strip() != full.text(b, q).strip() for q in ids)
    c = sum(pilot.text(a, q).strip() == full.text(b, q).strip() and pilot.cell(a, q) != full.cell(b, q) for q in ids)
    print("dose", k, "arm", a, b, "text differs", t, "same text different score", c,
          "net pilot", sum(pilot.cell(a, q) for q in ids), "net 570-copy", sum(full.cell(b, q) for q in ids))
    td += t; sc += c
print("total text differs", td, "same text different score", sc)
