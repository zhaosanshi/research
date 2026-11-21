# Iteration_01 analysis (Claude Fable 5.2, 2026-11-22), read-only, no model, no judge call. Exploration.
# (QA)^2 with every setting frozen from TruthfulQA at the write rule "prompt" (nothing is chosen on these questions):
# the 100-question pilot of one model (launchers/explore_20261120/qaqa_pilot.sh; the defaults below) and the full
# 570-question run of the five models (qaqa_run.sh; --task qaqafull --bank ... --work test_full570).
# DeepSeek Pro, two-axis rubric, reference = the set's abstractive answer; value of an answer: A (true and
# informative) +1, B (false) -1, C / H (no information, hedged) 0, empty output 0.
# Systems: no write; prompt only (conf_E2); single C; single E; N1 = E on every question, a question blank under E
# (the rule detector's abstention or an empty answer) is re-asked with E + C at 0.25 / 0.5 / 1 / 2 up to the cap of its
# kind (the model's own blank / answered without a write and blank under E) and the first non-blank answer is taken,
# for E's blanks only if it differs from the no-write answer (release_origin.norm). A question that is not released
# keeps E's cell.
# Printed per model: net score, answers and right / wrong per system, overall and by question type (valid assumption,
# false assumption, unverifiable assumption); paired differences with intervals; what the re-ask released; how often
# an answer changed by E got better or worse. With several models: the models together (questions resampled jointly).
# --skip-ids FILE leaves out the questions listed in a pilot id file (for the model that ran the pilot).
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
# model key: (E layer, strength, C layer, cap for E's blanks, cap for own blanks, single C): the TruthfulQA choices at
# "prompt" (results/write_positions/main_table_prompt_budget1_20261121.txt)
SET = {
    "llama3.1-8b": (16, "3.5", 19, "1", "2", "0.25"),
    "qwen3.5-9b": (15, "2", 16, "0.25", "0.5", "0.25"),
    "qwen3.8-27b": (28, "1.75", 33, "0.5", "1", "0.5"),
    "gemma3-12b": (24, "2.5", 25, "2", "2", "0.5"),
    "gemma3-27b": (28, "1", 31, "1", "2", "0.25"),
}
TYPES = (("no_failure", "valid assumption"), ("failure", "false assumption"), ("unverifiable", "unverifiable assumption"))
ORDER = ("no write", "prompt only", "single C", "single E", "N1")
PAIRS = (("N1", "single E"), ("N1", "no write"), ("N1", "single C"), ("N1", "prompt only"), ("single E", "no write"))


def load(model, task, bank_path, work_name):
    el, s, cl, cap_e, cap_own, kc = SET[model]
    with open(bank_path, encoding="utf-8") as f:
        bank = [json.loads(l) for l in f]
    ids = [b["id"] for b in bank]
    qtype = {b["id"]: b["qtype"] for b in bank}
    gdir = Path(S.NS) / "gen_prompt" / task / model / "test"
    args = SimpleNamespace(task=task, split="test", model=model, items=bank_path)
    e_arm = "x:E%d.w.p%s" % (el, s)
    arms = {"no write": "x", "prompt only": "conf_E2", "single C": "x:C%d.w.m%s" % (cl, kc), "single E": e_arm}
    arms.update({"EC " + k: "%s+C%d.w.m%s" % (e_arm, cl, k) for k in LAD})
    gen = {}
    for name, arm in arms.items():
        g = G.load_generation(gdir / (G.arm_token(arm) + ".json"), args, ids)
        k = g["contract"]
        if k["arm"] != arm or k.get("positions") != ("prompt" if ":" in arm else None):
            raise SystemExit("%s %s: not the pass %s at the expected write rule" % (model, name, arm))
        gen[name] = (k["condition"], {r["id"]: r for r in g["rows"]})
    work = Path(S.NS) / "judge" / task / model / work_name
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
            raise SystemExit("%s: unresolved vote for %s %s" % (model, q, c["condition"]))

    def blank(name, q):
        r = gen[name][1][q]
        return bool(r["rule_abstain"]) or not r["text"].strip()

    def cell(name, q):
        cond = gen[name][0]
        if (q, cond) not in val:
            if not gen[name][1][q]["text"].strip():
                return 0
            raise SystemExit("%s: no verdict for %s %s" % (model, q, cond))
        return val[(q, cond)]

    systems = {name: ({q: cell(name, q) for q in ids}, {q: not blank(name, q) for q in ids}) for name in ORDER[:4]}
    n1v, n1a, events = {}, {}, {}
    for q in ids:
        if not blank("single E", q):
            n1v[q], n1a[q] = cell("single E", q), True
            continue
        own = blank("no write", q)
        lad = LAD[:LAD.index(cap_own if own else cap_e) + 1]
        k = next((k for k in lad if not blank("EC " + k, q)), None)
        kind = "own" if own else "E's"
        if k is None:
            events[q] = (kind, "none")
        elif not own and ro.norm(gen["EC " + k][1][q]["text"]) == ro.norm(gen["no write"][1][q]["text"]):
            events[q] = (kind, "vetoed")
            k = None
        if k is None:
            n1v[q], n1a[q] = cell("single E", q), False
        else:
            v = cell("EC " + k, q)
            n1v[q], n1a[q] = v, True
            events[q] = (kind, "right" if v > 0 else "wrong" if v < 0 else "no information")
    systems["N1"] = (n1v, n1a)
    changed = {q for q in ids if systems["no write"][1][q] and systems["single E"][1][q]
               and ro.norm(gen["no write"][1][q]["text"]) != ro.norm(gen["single E"][1][q]["text"])}
    return ids, qtype, systems, events, changed


def report(model, ids, qtype, systems, events, changed, title):
    el, s, cl, cap_e, cap_own, kc = SET[model]
    print("# (QA)^2 %s, %s, %d questions, write rule prompt, E%d at %s, caps E's %s / own %s, single C %s (all from TruthfulQA); DeepSeek Pro" % (
        title, model, len(ids), el, s, cap_e, cap_own, kc))
    groups = [("all", ids)] + [(label, [q for q in ids if qtype[q] == t]) for t, label in TYPES]
    for label, qs in groups:
        print("\n## %s (%d questions)" % (label, len(qs)))
        for name in ORDER:
            v, a = systems[name]
            print("    %-12s net %+4d | answers %3d | right %3d | wrong %3d" % (
                name, sum(v[q] for q in qs), sum(a[q] for q in qs), sum(v[q] > 0 for q in qs), sum(v[q] < 0 for q in qs)))
        print("    " + " | ".join("%s minus %s %s" % (x, y, S.ps.ci([systems[x][0][q] - systems[y][0][q] for q in qs])) for x, y in PAIRS))
    cnt = lambda kind, what: sum(1 for q in ids if events.get(q) == (kind, what))  # noqa: E731
    print("\n## the re-ask: answered without a write and blank under E: released right %d, wrong %d, no information %d; same answer again and kept blank %d; not released %d | the model's own blanks: released right %d, wrong %d, no information %d; not released %d" % (
        cnt("E's", "right"), cnt("E's", "wrong"), cnt("E's", "no information"), cnt("E's", "vetoed"), cnt("E's", "none"),
        cnt("own", "right"), cnt("own", "wrong"), cnt("own", "no information"), cnt("own", "none")))
    for t, label in (("all", "all"),) + TYPES:
        qs = [q for q in ids if q in changed and (t == "all" or qtype[q] == t)]
        b = sum(systems["single E"][0][q] > systems["no write"][0][q] for q in qs)
        w = sum(systems["single E"][0][q] < systems["no write"][0][q] for q in qs)
        print("    answered in both passes and E changed the answer, %s: %d, better %d, worse %d" % (label, len(qs), b, w))


def main():
    argv = sys.argv[1:]
    opt = {"--task": "qaqa", "--bank": "Iteration_01/newsets/bank/qaqa_pilot_bank_20261122.jsonl", "--work": "test_pilot100", "--skip-ids": None}
    while argv[:1] and argv[0] in opt:
        opt[argv[0]] = argv[1]
        argv = argv[2:]
    if not argv:
        raise SystemExit("usage: qaqa_pilot_table.py [--task T] [--bank FILE] [--work NAME] [--skip-ids FILE] MODEL [MODEL ...]")
    skip = set()
    if opt["--skip-ids"]:
        with open(opt["--skip-ids"], encoding="utf-8") as f:
            skip = set(json.load(f)["ids"])
    title = "pilot" if opt["--task"] == "qaqa" else "full run"
    done = {}
    for model in argv:
        try:
            ids, qtype, systems, events, changed = load(model, opt["--task"], opt["--bank"], opt["--work"])
        except (FileNotFoundError, ValueError) as err:
            print("\n# %s: not ready (%s)\n" % (model, str(err)[:160]))
            continue
        ids = [q for q in ids if q not in skip]
        report(model, ids, qtype, systems, events, changed, title + (", without the pilot questions" if skip else ""))
        print()
        done[model] = (ids, qtype, systems)
    if len(done) > 1:
        first = next(iter(done.values()))
        ids, qtype = first[0], first[1]
        if any(d[0] != ids for d in done.values()):
            raise SystemExit("the models do not have the same questions")
        print("# the %d models together (%s), questions resampled jointly" % (len(done), ", ".join(done)))
        for label, qs in [("all", ids)] + [(lab, [q for q in ids if qtype[q] == t]) for t, lab in TYPES]:
            print("\n## %s (%d questions per model)" % (label, len(qs)))
            print("    net: " + " | ".join("%s %+d" % (name, sum(d[2][name][0][q] for d in done.values() for q in qs)) for name in ORDER))
            for x, y in PAIRS:
                per = [[d[2][x][0][q] - d[2][y][0][q] for q in qs] for d in done.values()]
                t_, lo, hi = S.joint_ci(per)
                print("    %-28s %+d [%+d, %+d], %d of %d models positive" % ("%s minus %s" % (x, y), t_, lo, hi, sum(sum(v) > 0 for v in per), len(per)))


if __name__ == "__main__":
    main()
