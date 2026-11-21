"""Read-only diagnostics for the five published 1-point operating settings at 89aedc1.

Run inside the experiment's Truth_Decoupling root with python3 -B PATH.
No model, new judge, or test-set parameter selection.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SETTINGS = (("Llama-3.1-8B", 16, "3-5"), ("Qwen3.5-9B", 15, "2"),
            ("Qwen3.8-27B", 28, "1-75"), ("Gemma-3-12B", 24, "2-5"), ("Gemma-3-27B", 28, "1"))
VALUE = {"R": 1, "W": -1, "H": 0, "X": 0}


def orig(table, model, layer, split):
    alt = next((a for a in table.ML.AL.ALTS if a.model == model and a.layer == layer), None)
    source = alt.load(split) if alt else table.DM.merged_all(model, split)
    return source[1], source[2]


def items(table, sp, tx, cfg):
    states, answered, records = table.S.system(sp, tx, cfg)
    if len(records) != len(sp.qs) or [r["p"] for r in records] != list(sp.qs):
        raise ValueError("Misaligned question IDs")
    return {r["p"]: (VALUE[s], a, r["cand"], r["note"]) for s, a, r in zip(states, answered, records)}


def model_check(table, model, layer, strength):
    source = [orig(table, model, layer, i) for i in (0, 1)]
    prompt = []
    for i in (0, 1):
        z, reason = table.prompt_split(model, layer, i, strength)
        if z is None:
            raise ValueError(reason)
        original_ids = set(source[i][0].qs)
        shared = [q for q in z[3] if q in original_ids]
        if shared != z[3]:
            z, reason = table.prompt_split(model, layer, i, strength, qs=shared)
            if z is None or z[3] != shared:
                raise ValueError(reason or "Lost common questions")
        prompt.append(z)
    frozen = table.ML.n1_pick(*source[0], strength)
    chosen = table.ML.n1_pick(prompt[0][0], prompt[0][1], strength)
    print(json.dumps({"model": model, "training_rule_caps": frozen, "prompt_rule_caps": chosen}))
    e = ("E", strength)
    deltas = {}
    for i in (0, 1):
        sp, tx, _, qs = prompt[i]
        new_e = items(table, sp, tx, e)
        old_e = items(table, *source[i], e)
        new = items(table, sp, tx, chosen)
        old = items(table, *source[i], frozen)
        kept_caps = items(table, sp, tx, frozen)
        change = [kept_caps[q][0] - new[q][0] for q in qs]
        recovered = [kept_caps[q][0] - new_e[q][0] for q in qs]
        print(json.dumps({"split": ("tune", "test")[i], "model": model, "n": len(qs),
                          "reselected_net": sum(new[q][0] for q in qs),
                          "old_caps_net": sum(kept_caps[q][0] for q in qs),
                          "old_caps_actual_answers": sum(kept_caps[q][1] for q in qs),
                          "old_caps_minus_reselected": table.ps.ci(change),
                          "old_caps_minus_single_e": table.ps.ci(recovered)}))
        if i:
            deltas = dict(zip(qs, change))
            n = sum(kept_caps[q][1] for q in qs)
            _, run, _ = table.usual(model, 1)
            control, why = table.strong_prompt(run, layer, strength, qs, n)
            if control is None:
                raise ValueError(why)
            assert sum(control[1]) == n
            matched = [kept_caps[q][0] - VALUE[state] for q, state in zip(qs, control[0])]
            print(json.dumps({"model": model, "old_caps_strong_same_n": n,
                              "strong_net": sum(VALUE[state] for state in control[0]),
                              "old_caps_minus_strong": table.ps.ci(matched)}))
        if model == "Llama-3.1-8B":
            groups = defaultdict(Counter)
            overlap = defaultdict(Counter)
            for q in qs:
                for label, row, baseline in (("prompt", new[q], new_e[q]), ("content", old[q], old_e[q])):
                    if row[2] is not None:
                        groups[label, row[2]].update(n=1, released=int(row[3] == "released"),
                                                     net=row[0] - baseline[0])
                overlap[old[q][3] == "released", new[q][3] == "released"].update(
                    n=1, old_gain=old[q][0] - old_e[q][0], new_gain=new[q][0] - new_e[q][0])
            print(json.dumps({"split": ("tune", "test")[i],
                              "origin": [[*k, dict(x)] for k, x in groups.items()],
                              "overlap": [[*k, dict(x)] for k, x in overlap.items()]}))
        if model == "Qwen3.5-9B" and i:
            cfg = ("ALL", strength, "0-5")
            direct = items(table, sp, tx, cfg)
            n = sum(direct[q][1] for q in qs)
            _, run, _ = table.usual(model, 1)
            top, why = table.strong_prompt(run, layer, strength, qs, n)
            if top is None:
                raise ValueError(why)
            assert sum(top[1]) == n
            diff = [direct[q][0] - VALUE[b] for q, b in zip(qs, top[0])]
            print(json.dumps({"model": model, "fixed_c": 0.5, "n_answered": n,
                              "raw_joint": [sum(direct[q][0] == v for q in qs) for v in (1, -1)],
                              "matched_e": [top[0].count(v) for v in ("R", "W")], "diff": table.ps.ci(diff)}))
    return deltas


def main():
    sys.path.insert(0, str(Path("Iteration_01/diag").resolve()))
    import main_table_prompt as table
    cells = [model_check(table, *setting) for setting in SETTINGS]
    for k in (3, 5):
        common = sorted(set.intersection(*(set(c) for c in cells[:k])))
        print(json.dumps({"models": k, "common_questions": len(common),
                          "old_caps_minus_reselected": table.ps.ci([sum(c[q] for c in cells[:k]) for q in common])}))


if __name__ == "__main__":
    main()
