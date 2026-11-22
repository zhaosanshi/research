"""Read-only (QA)^2 replay for 548f5ef, from Truth_Decoupling in the scoring container.

Rebuild the pilot choices, reproduce the frozen and adapted primary comparisons,
then fix a mask or an answer source to diagnose their difference. All reported
intervals hold the chosen sets fixed, as in the published table. Group-ranked
results and answer-replacement diagnostics are exploratory.
"""
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("Iteration_01/diag").resolve()))
import qaqa_pilot_table as qt

BANK = Path("Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl")
SPLITS = Path("Iteration_01/newsets/splits")
WORKS = ["test_full570", "test_full570_tuned", "test_full570_strong"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_choices():
    pilot = [json.loads(line) for line in Path(qt.TUNE_BANK).read_text(encoding="utf-8").splitlines()]
    meta = read(SPLITS / "qaqa_pilot_20261122.json")
    if meta["ids"] != [r["id"] for r in pilot] or qt.G.sha_file(qt.TUNE_BANK) != meta["bank_sha256"]:
        raise ValueError("Pilot IDs differ from their bank")
    whole = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in whole}
    if len(by_id) != len(whole) or len(whole) != 570 or len(pilot) != 100 or any(by_id[r["id"]] != r for r in pilot):
        raise ValueError("Pilot and full bank disagree")
    choices = {}
    for name in qt.SET:
        fit = qt.Raw(name, "qaqa", qt.TUNE_BANK, ["test_tune100"])
        for fix, suffix in [(True, "_fixedE"), (False, "")]:
            archived = read(SPLITS / f"qaqa_tune_choice{suffix}_20261122.json")[name]
            with contextlib.redirect_stdout(io.StringIO()):
                actual = qt.tune(fit, fit.ids, fix)
            if any(actual[k] != v for k, v in archived.items()) or set(actual) - set(archived) - {"e_fixed"}:
                raise ValueError(f"Choice mismatch: {name}, fixed={fix}")
            if fix:
                choices[name] = actual
    return choices, set(meta["ids"])


def replay(raw, choice, ids):
    s = choice["n1_strength"]
    ordinary = raw.e(s)
    forced = "conf:" + ordinary.split(":", 1)[1]
    n1, issued, _ = raw.n1_system(s, choice["n1_cap_e"], choice["n1_cap_own"])
    required = [ordinary, forced, "conf", "conf_E2"]
    for arm in required:
        raw.need(arm)
    path = Path(qt.S.NS) / "conf" / "qaqafull" / raw.model / "test" / (qt.G.arm_token(forced) + "-at-prompt.json")
    document = read(path)
    if [r["id"] for r in document["rows"]] != raw.ids:
        raise ValueError("Missing, duplicated, or reordered confidence rows")
    conf = raw.confidence("qaqafull", forced)
    eligible = [q for q in ids if not raw.blank(forced, q)]
    if any(conf[q] is None or not math.isfinite(conf[q]) for q in eligible):
        raise ValueError("An actual strong answer has missing or non-finite confidence")
    ranked = sorted(eligible, key=lambda q: -conf[q])  # stable on frozen input order
    budget = sum(issued[q] for q in ids)
    if budget > len(ranked):
        raise ValueError(f"Unreachable N for {raw.model}: {budget} vs {len(ranked)}")
    top = set(ranked[:budget])
    eligible_set = set(eligible)
    both = {q for q in ids if issued[q] and q in eligible_set}
    top_overlap = set(ranked[:len(both)])
    cells = {}
    for q in ids:
        f = raw.cell(forced, q)
        e = raw.cell(ordinary, q)
        forced_base = raw.cell("conf", q)
        strong_value = f if q in top else 0
        select = (int(q in both) - int(q in top_overlap)) * f
        content = n1[q] - f if q in both else 0
        overall = n1[q] - strong_value
        # Same top-N list and exactly the same final answer count. When the
        # proposed writer is blank, keep the original must-answer response.
        replace_n1 = n1[q] - f if q in top and issued[q] else 0
        replace_e = e - f if q in top and not raw.blank(ordinary, q) else 0
        cells[q] = {
            "primary": overall, "selection": select, "content": content, "boundary": overall - select - content,
            "N1-E": n1[q] - e, "E-base": e - raw.cell("x", q),
            "forced_E-base": f - forced_base,
            "forced_E-honest": f - raw.cell("conf_E2", q),
            "forced_E-honest_both_answer": f - raw.cell("conf_E2", q)
               if q in eligible_set and not raw.blank("conf_E2", q) else 0,
            "forced_E-base_both_answer": f - forced_base if q in eligible_set and not raw.blank("conf", q) else 0,
            "swap_N1": replace_n1, "swap_E": replace_e, "swap_N1_minus_swap_E": replace_n1 - replace_e,
            "n1_issued": int(issued[q]), "strong_issued": int(q in top)
        }
    return cells, (budget, len(ranked), len(both))


def weights(n):
    # Same bootstrap index stream and order statistics as newsets_score.joint_ci.
    rng = random.Random(qt.S.SEED)
    return np.asarray([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n)
                       for _ in range(qt.S.N_BOOT)], dtype=np.int64)


def table(data, ids, names, keys):
    w = weights(len(ids))
    for key in keys:
        v = np.asarray([sum(data[m][q][key] for m in names) for q in ids], dtype=np.int64)
        draws = np.sort(w @ v)
        print(f"  {key}: {v.sum():+d} [{draws[int(.025*len(draws))]:+d}, {draws[int(.975*len(draws))-1]:+d}]")


def main():
    choices, skip = load_choices()
    all_models = list(qt.SET)
    raw = {m: qt.Raw(m, "qaqafull", str(BANK), WORKS) for m in all_models}
    first = raw[all_models[0]]
    if any(r.ids != first.ids or r.qtype != first.qtype for r in raw.values()):
        raise ValueError("Model questions or strata differ")
    held = [q for q in first.ids if q not in skip]
    assert len(held) == 470
    for setting, ids in [("fixed E, adapted caps", held), ("frozen", held), ("frozen", first.ids)]:
        outcome = {}
        for model, r in raw.items():
            _, strength, _, cap_e, cap_own, _ = qt.SET[model]
            c = choices[model] if setting.startswith("fixed") else dict(n1_strength=strength, n1_cap_e=cap_e, n1_cap_own=cap_own)
            outcome[model], counts = replay(r, c, ids)
            print(model, setting, len(ids), "N1 N / eligible strong / selectable overlap", counts)
            if setting.startswith("fixed"):
                n_values, n_ans, _ = r.n1_system(c["n1_strength"], c["n1_cap_e"], c["n1_cap_own"])
                single_values, single_ans = r.arm_system(r.c(c["single_C"]))
                worst = sorted(n_values[q] for q in ids if n_ans[q])
                best = sorted((single_values[q] for q in ids if single_ans[q]), reverse=True)
                k = min(len(worst), len(best))
                print("  N1-vs-C fixed-pool bound: common attainable K", k,
                      "N1 worst minus C best", sum(worst[:k]) - sum(best[:k]))
        for names in (all_models[:3], all_models):
            print(setting, len(ids), names)
            table(outcome, ids, names, ["primary", "selection", "content", "boundary", "N1-E", "E-base", "forced_E-base",
                                        "forced_E-base_both_answer", "forced_E-honest", "forced_E-honest_both_answer", "swap_N1", "swap_E", "swap_N1_minus_swap_E"])
        if setting.startswith("fixed"):
            for typ, _ in qt.TYPES:
                group = [q for q in ids if first.qtype[q] == typ]
                restricted = {}
                for m, r in raw.items():
                    restricted[m], counts = replay(r, choices[m], group)
                    assert counts[0] <= counts[1]
                print("strictly type-matched", typ, len(group))
                table(restricted, group, all_models, ["primary","N1-E","forced_E-base","forced_E-base_both_answer","forced_E-honest","forced_E-honest_both_answer"])


if __name__ == "__main__":
    main()
