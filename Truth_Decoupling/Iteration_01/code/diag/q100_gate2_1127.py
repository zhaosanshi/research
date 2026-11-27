#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Opus 5.5, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5, segments A and B: the 100-question passes. Zhao Lei 2026-11-27 07:40 asked why every question was being judged
# when 100 questions show the rough answer (his standing rule of 2026-11-15 / 2026-11-24). CPU helper (container python3,
# from /workspace/Truth_Decoupling); it never loads a model and never calls a judge. Not reviewed by GPT.
# Subsets (fixed before any 100-question reading):
#   A, FalseQA held-out 600 (300 pairs): 50 pairs = random.Random(20261127).sample(sorted held-out pair ids, 50), sorted,
#     written once to Iteration_01/newsets/gate2_1127/q100_pairs_gate2_1127.json (exclusive create). The bank's "pro100"
#     flag is true on every row (a format field of falseqa_prep.py), so it is not a subset.
#   B, TruthfulQA test: the Dr's 100 pre-fixed pairs (configs/splits/dr_100pairs_20261110.json of the code copies, the
#     subset of scripts/review_checks/ff_test_subset_judge.sh) intersected with the model's main list.
# The 100-question judge input of a model = its full-run judge input (read only, unchanged) restricted to the subset's
# questions: same conditions, same answers, same withheld (emptied) cells, same order. It goes into a new work dir:
#   A newsets/judge/falseqa/<model>/test_gate2_1127_q100/judge_input_gate2_1127.json
#   B <run>/eval_extra/judge_test_gate2b_1127_q100_test/judge_input_gate2b_1127.json
# Commands:
#   draw-a                 write the FalseQA draw unless it exists; print it
#   judge-input-a MODEL    the A input (ON questions of the 50 pairs)
#   judge-input-b MODEL    the B input (the Dr's 100 in the main list)
#   work-a MODEL / work-b MODEL   print the 100-question work dir (container path)
import json
import os
import random
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]

DRAW_A = "Iteration_01/newsets/gate2_1127/q100_pairs_gate2_1127.json"
DR100 = "code_newmodels_20261112/configs/splits/dr_100pairs_20261110.json"
DR100_SHA = "a115131dcd0554d3"
SEED_A = 20261127
N_PAIRS_A = 50


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha16(p):
    import hashlib
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def held_pairs():
    import falseqa_gate2_1127 as AH
    rows = AH.bank_rows()
    return rows, sorted({r["pair_id"] for r in rows if r["set"] == "heldout"})


def draw_a():
    if os.path.exists(DRAW_A):
        d = read(DRAW_A)
        print("draw exists: %d pairs, seed %s, sha256 %s" % (len(d["pairs"]), d["seed"], sha16(DRAW_A)))
        return d
    rows, pairs = held_pairs()
    if len(pairs) != 300:
        raise SystemExit("held-out pairs %d, expected 300" % len(pairs))
    pick = sorted(random.Random(SEED_A).sample(pairs, N_PAIRS_A))
    d = {"name": "q100_pairs_gate2_1127", "seed": SEED_A, "rule": "random.Random(%d).sample(sorted held-out pair ids, %d), sorted"
         % (SEED_A, N_PAIRS_A), "n_heldout_pairs": len(pairs), "pairs": pick,
         "questions": sorted(r["id"] for r in rows if r["set"] == "heldout" and r["pair_id"] in set(pick))}
    os.makedirs(os.path.dirname(DRAW_A), exist_ok=True)
    with open(DRAW_A, "x", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    print("draw written: %d pairs, %d questions -> %s (sha256 %s)" % (len(pick), len(d["questions"]), DRAW_A, sha16(DRAW_A)))
    return d


def dr100():
    if sha16(DR100) != DR100_SHA:
        raise SystemExit("%s: sha256 %s, expected %s" % (DR100, sha16(DR100), DR100_SHA))
    d = read(DR100)
    if len(d["pair_ids"]) != 100 or len(set(d["pair_ids"])) != 100:
        raise SystemExit("the Dr's list does not hold 100 distinct ids")
    return list(d["pair_ids"])


def work_a(m, mode):
    import falseqa_gate2_1127 as AH
    if mode == "full":
        return str(AH.work(m, "full"))
    return os.path.join(AH.NSD, "judge", "falseqa", m, "test_%s_q100" % AH.TAG)


def work_b(B, mode):
    import tqa_gate2b_1127 as BH
    if mode == "full":
        return BH.work(B, "full")
    return os.path.join(B.run, "eval_extra", "judge_test_%s_q100_test" % BH.TAG)


def restrict(src, keep, tag, info):
    """the full-run judge input src restricted to the questions keep (pair_id strings), order kept"""
    d = read(src)
    ev = [e for e in d["evaluations"] if e["pair_id"] in keep]
    if len(ev) != len(keep):
        raise SystemExit("%s: %d of the %d subset questions found" % (src, len(ev), len(keep)))
    out = dict(d)
    out["evaluations"] = ev
    out["question_ids"] = [e["pair_id"] for e in ev]
    t = dict(d[tag])
    t["needed"] = {c: [q for q in l if (("falseqa:" + q) if tag == "gate2_1127" else q) in keep] for c, l in d[tag]["needed"].items()}
    t["mode"] = "q100 (restriction of the full-run input)"
    t.pop("withheld_nonempty_cells", None)
    out[tag] = t
    out["q100"] = dict(info, source=src, source_sha256=sha16(src), questions=len(ev))
    return json.loads(json.dumps(out))


def write_input(w, name, payload):
    out = os.path.join(w, name)
    if os.path.exists(out):
        if read(out) != payload:
            raise SystemExit("%s exists and differs: stop" % out)
        print("%s exists and equals the build" % out)
        return
    os.makedirs(os.path.join(w, "judge2", "cache"), exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("written %s" % out)


def judge_input_a(m):
    import falseqa_gate2_1127 as AH
    d = draw_a()
    rows, _ = held_pairs()
    P = set(d["pairs"])
    B = AH.base_state(m, rows)
    pair = {r["id"]: r["pair_id"] for r in rows}
    on = [q for q in B.on if pair[q] in P]
    keep = {"falseqa:" + q for q in on}
    src = os.path.join(work_a(m, "full"), AH.JIN)
    payload = restrict(src, keep, AH.TAG, {"segment": "A", "draw": DRAW_A, "draw_sha256": sha16(DRAW_A), "pairs": len(P),
                                           "subset_questions": sum(pair[q] in P for q in B.ids), "switch_on": len(on)})
    need = sum(len(v) for v in payload[AH.TAG]["needed"].values())
    print("%s A: %d pairs, %d held-out questions, switch ON %d; cells that can be handed in %d" % (
        m, len(P), payload["q100"]["subset_questions"], len(on), need))
    write_input(work_a(m, "q100"), AH.JIN, payload)


def judge_input_b(m):
    import tqa_gate2b_1127 as BH
    B = BH.base(m)
    ids = set(dr100())
    keep = {q for q in B.ids if q in ids}
    src = os.path.join(work_b(B, "full"), BH.JIN)
    payload = restrict(src, keep, BH.TAG, {"segment": "B", "subset": DR100, "subset_sha256": DR100_SHA, "in_main_list": len(keep),
                                          "main_list": len(B.ids)})
    need = sum(len(v) for v in payload[BH.TAG]["needed"].values())
    print("%s B: the Dr's 100, %d in the main list of %d; cells that can be handed in %d" % (m, len(keep), len(B.ids), need))
    write_input(work_b(B, "q100"), BH.JIN, payload)


def main():
    a = sys.argv[1:]
    if a == ["draw-a"]:
        draw_a()
    elif len(a) == 2 and a[0] == "judge-input-a":
        judge_input_a(a[1])
    elif len(a) == 2 and a[0] == "judge-input-b":
        judge_input_b(a[1])
    elif len(a) == 2 and a[0] == "work-a":
        print(work_a(a[1], "q100"))
    elif len(a) == 2 and a[0] == "work-b":
        import tqa_gate2b_1127 as BH
        print(work_b(BH.base(a[1]), "q100"))
    else:
        raise SystemExit("usage: q100_gate2_1127.py draw-a | judge-input-a MODEL | judge-input-b MODEL | work-a MODEL | work-b MODEL")


if __name__ == "__main__":
    main()
