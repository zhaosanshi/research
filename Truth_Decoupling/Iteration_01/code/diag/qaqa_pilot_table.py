# Iteration_01 analysis (Claude Fable 5.2, 2026-11-22), read-only, no model, no judge call. Exploration.
# The (QA)^2 pilot (launchers/explore_20261120/qaqa_pilot.sh): 100 evaluation questions, one model, every setting
# frozen from TruthfulQA at the write rule "prompt" (nothing is chosen on these questions). DeepSeek Pro, two-axis
# rubric, reference = the set's abstractive answer; value of an answer: A (true and informative) +1, B (false) -1,
# C / H (no information, hedged) 0, empty output 0.
# Systems: no write; prompt only (conf_E2); single C; single E; N1 = E on every question, a question blank under E
# (the rule detector's abstention or an empty answer) is re-asked with E + C at 0.25 / 0.5 / 1 / 2 up to the cap of its
# kind (the model's own blank / answered without a write and blank under E) and the first non-blank answer is taken,
# for E's blanks only if it differs from the no-write answer (release_origin.norm). A question that is not released
# keeps E's cell.
# Printed: net score, answers and right / wrong per system, overall and by question type (valid assumption, false
# assumption, unverifiable assumption); paired differences with intervals; what the re-ask released.
# Runs in the scoring container (the verdict files record container paths):
#   docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/qaqa_pilot_table.py llama3.1-8b
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import newsets_gen as G                              # noqa: E402
import newsets_score as S                            # noqa: E402
import release_origin as ro                          # noqa: E402
from judge_two_axis_run import load_verified_single  # noqa: E402

VALUE = {"A": 1, "B": -1, "C": 0, "H": 0}
LAD = ("0.25", "0.5", "1", "2")
# model key: (E layer, strength, C layer, cap for E's blanks, cap for own blanks, single C); the TruthfulQA choices at "prompt"
SET = {"llama3.1-8b": (16, "3.5", 19, "1", "2", "0.25")}
BANK = "Iteration_01/newsets/bank/qaqa_pilot_bank_20261122.jsonl"
TYPES = (("no_failure", "valid assumption"), ("failure", "false assumption"), ("unverifiable", "unverifiable assumption"))


def main():
    model = sys.argv[1]
    el, s, cl, cap_e, cap_own, kc = SET[model]
    with open(BANK, encoding="utf-8") as f:
        bank = [json.loads(l) for l in f]
    ids = [b["id"] for b in bank]
    qtype = {b["id"]: b["qtype"] for b in bank}
    gdir = Path(S.NS) / "gen_prompt" / "qaqa" / model / "test"
    args = SimpleNamespace(task="qaqa", split="test", model=model, items=BANK)
    e_arm = "x:E%d.w.p%s" % (el, s)
    arms = {"no write": "x", "prompt only": "conf_E2", "single C": "x:C%d.w.m%s" % (cl, kc), "single E": e_arm}
    arms.update({"EC " + k: "%s+C%d.w.m%s" % (e_arm, cl, k) for k in LAD})
    gen = {}
    for name, arm in arms.items():
        g = G.load_generation(gdir / (G.arm_token(arm) + ".json"), args, ids)
        k = g["contract"]
        if k["arm"] != arm or k.get("positions") != ("prompt" if ":" in arm else None):
            raise SystemExit("%s: not the pass %s at the expected write rule" % (name, arm))
        gen[name] = (k["condition"], {r["id"]: r for r in g["rows"]})
    work = Path(S.NS) / "judge" / "qaqa" / model / "test_pilot100"
    # require_complete=False: an empty output has no vote; every other cell must be scored (checked below)
    _, cells, votes = load_verified_single(str(work), "pro", generation_path=str(work / "judge_input.json"), require_complete=False)
    val = {}
    for key, c in cells.items():
        v = votes[key]
        q = c["pair_id"].split(":", 1)[1]
        if v["status"] == "scored":
            val[(q, c["condition"])] = VALUE[v["label"]]
        elif v.get("basis") == "empty_output_no_judge_vote":
            val[(q, c["condition"])] = 0
        else:
            raise SystemExit("unresolved vote for %s %s" % (q, c["condition"]))

    def blank(name, q):
        r = gen[name][1][q]
        return bool(r["rule_abstain"]) or not r["text"].strip()

    def cell(name, q):
        cond = gen[name][0]
        if (q, cond) not in val:
            if not gen[name][1][q]["text"].strip():
                return 0
            raise SystemExit("no verdict for %s %s" % (q, cond))
        return val[(q, cond)]

    systems = {name: ({q: cell(name, q) for q in ids}, {q: not blank(name, q) for q in ids}) for name in ("no write", "prompt only", "single C", "single E")}
    n1v, n1a = {}, {}
    rel = {"E's": {"released": [0, 0, 0], "vetoed": 0, "none": 0}, "own": {"released": [0, 0, 0], "none": 0}}
    for q in ids:
        if not blank("single E", q):
            n1v[q], n1a[q] = cell("single E", q), True
            continue
        own = blank("no write", q)
        lad = LAD[:LAD.index(cap_own if own else cap_e) + 1]
        k = next((k for k in lad if not blank("EC " + k, q)), None)
        kind = rel["own" if own else "E's"]
        if k is None:
            kind["none"] += 1
        elif not own and ro.norm(gen["EC " + k][1][q]["text"]) == ro.norm(gen["no write"][1][q]["text"]):
            kind["vetoed"] += 1
            k = None
        if k is None:
            n1v[q], n1a[q] = cell("single E", q), False
        else:
            v = cell("EC " + k, q)
            n1v[q], n1a[q] = v, True
            kind["released"][0 if v > 0 else 1 if v < 0 else 2] += 1
    systems["N1"] = (n1v, n1a)
    order = ("no write", "prompt only", "single C", "single E", "N1")
    print("# (QA)^2 pilot, %s, %d questions, write rule prompt, E%d at %s, caps E's %s / own %s, single C %s (all from TruthfulQA); DeepSeek Pro" % (
        model, len(ids), el, s, cap_e, cap_own, kc))
    groups = [("all", ids)] + [(label, [q for q in ids if qtype[q] == t]) for t, label in TYPES]
    for label, qs in groups:
        print("\n## %s (%d questions)" % (label, len(qs)))
        for name in order:
            v, a = systems[name]
            print("    %-12s net %+4d | answers %3d | right %3d | wrong %3d" % (
                name, sum(v[q] for q in qs), sum(a[q] for q in qs), sum(v[q] > 0 for q in qs), sum(v[q] < 0 for q in qs)))
        d = lambda x, y: S.ps.ci([systems[x][0][q] - systems[y][0][q] for q in qs])  # noqa: E731
        print("    N1 minus single E %s | N1 minus no write %s | N1 minus single C %s | N1 minus prompt only %s | single E minus no write %s" % (
            d("N1", "single E"), d("N1", "no write"), d("N1", "single C"), d("N1", "prompt only"), d("single E", "no write")))
    r = rel
    print("\n## the re-ask: answered without a write and blank under E: released %d (right %d, wrong %d, no information %d), same answer again and kept blank %d, not released %d | the model's own blanks: released %d (right %d, wrong %d, no information %d), not released %d" % (
        sum(r["E's"]["released"]), r["E's"]["released"][0], r["E's"]["released"][1], r["E's"]["released"][2], r["E's"]["vetoed"], r["E's"]["none"],
        sum(r["own"]["released"]), r["own"]["released"][0], r["own"]["released"][1], r["own"]["released"][2], r["own"]["none"]))
    chg = [q for q in ids if systems["no write"][1][q] and systems["single E"][1][q]
           and ro.norm(gen["no write"][1][q]["text"]) != ro.norm(gen["single E"][1][q]["text"])]
    for t, label in (("all", "all"),) + TYPES:
        qs = [q for q in chg if t == "all" or qtype[q] == t]
        b = sum(systems["single E"][0][q] > systems["no write"][0][q] for q in qs)
        w = sum(systems["single E"][0][q] < systems["no write"][0][q] for q in qs)
        print("    answered in both passes and E changed the answer, %s: %d, better %d, worse %d" % (label, len(qs), b, w))


if __name__ == "__main__":
    main()
