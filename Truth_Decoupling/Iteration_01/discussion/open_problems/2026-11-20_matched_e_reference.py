"""Read-only supplement to dose_mid: compare each dose with frozen must-answer E=2 at the same actual N.

Run on the experiment host from Truth_Decoupling, with python3 -B PATH --dose 1 --dose 2
or, after both sets of Pro verdicts exist, --dose 1.25 --dose 1.5 --dose 1.75.
This is a fixed comparator across doses. It does not estimate the missing same-dose comparator.
"""
import argparse
import json
import math
import sys
from pathlib import Path

MODELS = ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B")
DOSES = ("1", "1.25", "1.5", "1.75", "2")


def keep_at_n(ids, source, confidence, n):
    if len(set(ids)) != len(ids) or not 0 <= n <= len(ids):
        raise ValueError("invalid cohort or target answer count")
    eligible = [p for p in ids if source[p][1]]
    for p in eligible:
        if p not in confidence or not math.isfinite(confidence[p]):
            raise ValueError("missing or non-finite confidence for an answered question: " + p)
    if len(eligible) < n:
        return None, len(eligible)
    # Stable sorting preserves the common cohort order on confidence ties.
    kept = set(sorted(eligible, key=lambda p: -confidence[p])[:n])
    return [source[p][0] if p in kept else "X" for p in ids], len(eligible)


def require_mid_votes(work, ids, conds):
    """Missing, duplicate or stale rows must not present themselves as an evaluable smaller cohort."""
    expected = {(p, c) for p in ids for c in conds}
    cd = Path(work) / "judge2"
    if not (cd / "single_pro.jsonl").is_file():
        raise SystemExit("waiting for completed Pro votes: " + str(cd))
    seen = {}
    with (cd / "cells.jsonl").open() as stream:
        for row in map(json.loads, stream):
            key = (row["pair_id"], row["condition"])
            if key in expected:
                if key in seen or not row.get("input_sha"):
                    raise ValueError("duplicate or unbound cell: " + str(key))
                seen[key] = row["input_sha"]
    if seen.keys() != expected:
        raise ValueError("source grid does not contain all expected cells")
    tickets = set()
    with (cd / "single_pro.jsonl").open() as stream:
        for row in map(json.loads, stream):
            key = (row["pair_id"], row["condition"])
            if key not in expected:
                continue
            if key in tickets or row.get("input_sha") != seen[key] or row.get("judge") != "pro":
                raise ValueError("duplicate, stale or wrong-judge vote: " + str(key))
            if row["status"] == "scored":
                if row.get("label") not in ("A", "B", "H", "C"):
                    raise ValueError("invalid scored label: " + str(key))
            elif not (row.get("basis") == "pro_declined"
                      or (row.get("basis") == "empty_output_no_judge_vote" and row.get("empty_output") is True)):
                raise ValueError("unresolved vote: " + str(key))
            tickets.add(key)
    if tickets != expected:
        raise ValueError("incomplete vote grid in " + str(cd))


def verified_single(work, judge):
    # The parent reader validates generation, cell/row/summary hashes and the
    # judge contract; run with the same mounted paths as the original scoring.
    sys.path.insert(0, str(Path("Iteration_01/code/scripts/review_checks").resolve()))
    from judge_two_axis_run import load_verified_single
    load_verified_single(str(work), judge, generation_path=str(Path(work) / "judge_input.json"),
                         require_complete=False)


def evaluate(name, doses):
    import dose_table as dt
    mt = dt.MT
    pairs = [mt.Data(name, i) for i in (0, 1)]
    mid = None
    if any(d not in ("1", "2") for d in doses):
        import dose_mid as dm
        for i, data in enumerate(pairs):
            work = dm.new_work(data, i)
            require_mid_votes(work, data.sp.qs, ["baseline_x"] + dm.needed(data.sp, dm.MID))
            verified_single(work, "pro")
            # The new and legacy rows are subsequently combined by dose_mid.
            verified_single(data.m["ec"][i], data.m["judge"][i])
        verified_single(pairs[1].m["d"][1], "pro")
        mid = [dm.merged(name, i) for i in (0, 1)]

    test = pairs[1]
    ref_states, ref_ans = mt.must(test, test.sp.dE("2"))
    source = dict(zip(test.sp.qs, zip(ref_states, ref_ans)))
    score_rows = test.rows("conf_E2", test.m["d"][1], test.sp.dE("2"))
    confidence = {p: r["mean_logp"] for p, r in score_rows.items() if r.get("mean_logp") is not None}
    for displayed in doses:
        dose = displayed.replace(".", "-")
        cands = [("E", dose)] + [("N1", dose, a, b) for a in dt.CAPS for b in dt.CAPS
                                  if a is not None or b is not None]
        if displayed in ("1", "2"):
            tune = pairs[0]
            pick = max(cands, key=lambda cfg: dt.ps.score(dt.ro.states(tune.sp, tune.tx, cfg), 1))
            states, ans = mt.plain(test, pick)
            ids = test.sp.qs
        else:
            _, dev_sp, dev_text, _, _ = mid[0]
            _, tst_sp, tst_text, _, _ = mid[1]
            pick = max(cands, key=lambda cfg: dt.ps.score(dm.S.system(dev_sp, dev_text, cfg)[0], 1))
            states, ans, _ = dm.S.system(tst_sp, tst_text, pick)
            ids = tst_sp.qs
        n = sum(ans)
        ref, available = keep_at_n(ids, source, confidence, n)
        header = dict(model=name, dose=displayed, selection=list(pick), questions=len(ids),
                      excluded_from_parent=sorted(set(test.sp.qs) - set(ids)), answered=n, control_available=available)
        if ref is None:
            yield displayed, dict(header, comparison="N unreachable"), None
            continue
        vals = mt.val
        dif = [a - b for a, b in zip(vals(states), vals(ref))]
        result = dict(header, control="must-answer E2, top N, mean own-token log probability",
                      result=[states.count("R"), states.count("W"), sum(vals(states))],
                      control_result=[ref.count("R"), ref.count("W"), sum(vals(ref))],
                      paired_net=dt.ps.ci(dif))
        yield displayed, result, dict(zip(ids, dif))


def report(names, doses):
    from collections import Counter
    import policy_select2 as ps
    by_dose = {dose: {} for dose in doses}
    for model in names:
        for dose, row, paired in evaluate(model, doses):
            print(json.dumps(row), flush=True)
            by_dose[dose][model] = paired
    for label, group in (("main_three", MODELS[:3]), ("all_five", MODELS)):
        if not set(group).issubset(names):
            continue
        for dose, values in by_dose.items():
            if any(values[m] is None for m in group):
                print(json.dumps(dict(group=label, dose=dose, comparison="N unreachable")))
                continue
            total = Counter()
            for model in group:
                total.update(values[model])
            # Use the published main table's union of question IDs. For an excluded
            # model/question cell its contribution is zero on every resample.
            print(json.dumps(dict(group=label, dose=dose, paired_net=ps.ci([total[q] for q in sorted(total)]),
                                  distinct_questions=len(total))), flush=True)


def selftest():
    ids = ["q2", "q1", "q3"]
    src = {"q1": ("W", True), "q2": ("R", True), "q3": ("X", False)}
    conf = {"q1": -0.1, "q2": -0.1}
    assert keep_at_n(ids, src, conf, 1) == (["R", "X", "X"], 2)
    assert keep_at_n(ids, src, conf, 3) == (None, 2)
    assert keep_at_n(ids, src, conf, 0) == (["X", "X", "X"], 2)
    try:
        keep_at_n(ids, src, {}, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unscored eligible record")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as path:
        d = Path(path) / "judge2"
        d.mkdir()
        key = dict(pair_id="q", condition="E", input_sha="same")
        (d / "cells.jsonl").write_text(json.dumps(key) + "\n")
        votes = d / "single_pro.jsonl"
        valid = dict(key, judge="pro", status="scored", label="A")
        votes.write_text(json.dumps(valid) + "\n")
        require_mid_votes(path, ["q"], ["E"])
        for bad in ([], [valid, valid], [dict(valid, input_sha="other")],
                    [dict(valid, status="no_verdict", basis="temporary_error")]):
            votes.write_text("".join(json.dumps(r) + "\n" for r in bad))
            try:
                require_mid_votes(path, ["q"], ["E"])
            except ValueError:
                continue
            raise AssertionError("accepted missing, duplicate, stale or unresolved votes")
    print("selftest PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--dose", action="append", choices=DOSES)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        sys.path.insert(0, str(Path("Iteration_01/diag").resolve()))
        report(tuple(dict.fromkeys(args.model or MODELS)), list(dict.fromkeys(args.dose or ("1", "2"))))
