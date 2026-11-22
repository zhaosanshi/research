"""CPU-only replay of the completed GSM8K N1 and the fixed-answer-pool comparison
against the same-layer, same-amount mean direction. Run from Truth_Decoupling
in the original scoring container with python3 -B PATH. No writes or judging.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SEED, REPS = 20261122, 4000
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/code/scripts/capability"]


def emit(**row):
    print(json.dumps(row), flush=True)


def check(ok, why):
    if not ok:
        raise ValueError(why)


def states(values):
    return np.array([int(x == "R") - int(x == "W") for x in values])


def boot_weights(n):
    # The same resampled question indices apply to every model in a group.
    rng = np.random.default_rng(SEED)
    return np.vstack([np.ones(n, dtype=int), rng.multinomial(n, np.full(n, 1/n), REPS)])


def interval(x, digits=0):
    low, high = np.quantile(x[1:], [.025, .975], method="nearest")
    return [round(float(v), digits) for v in (x[0], low, high)]


def compare_pools(weights, left, right):
    """Sharp lower bound on the score gap when both retain exactly K rule-answered rows.
    Keep the worst K in the left answer pool, the best K in the right one.
    Indicators distinguish a rule-answered zero-point hedge from a genuine rule blank.
    """
    la, ra = left[:, 1].astype(bool), right[:, 1].astype(bool)
    ln, rn = weights @ la, weights @ ra
    k = np.minimum(ln, rn)
    lr, lw = weights @ (la & (left[:, 0] == 1)), weights @ (la & (left[:, 0] == -1))
    rr, rw = weights @ (ra & (right[:, 0] == 1)), weights @ (ra & (right[:, 0] == -1))
    worst_left = np.maximum(0, k - ln + lr) - np.minimum(k, lw)
    best_right = np.minimum(k, rr) - np.maximum(0, k - rn + rw)
    return k, worst_left - best_right


def verified_grid(mp, run, split, tag, audit_ids):
    from judge_two_axis_run import load_verified_single
    work = Path(mp.work(run, split, tag))
    export_path = work / "judge_input.json"
    _, cells, votes = load_verified_single(str(work), "pro", generation_path=str(export_path), require_complete=False)
    export = json.loads(export_path.read_text())
    ids = export["question_ids"]
    check(export["completed"] and export["split"] == split and len(set(ids)) == len(ids), str(work))
    check(set(audit_ids) <= set(ids), f"Missing question in {work}")
    for name in export["sources"]["gen"]:
        src = (Path(run) / "eval_extra/gen" / name).resolve()
        check(name.endswith(f"-at-prompt_{split}.json"), str(src))
        raw = src.read_bytes()
        check(hashlib.sha256(raw).hexdigest() == export["sources"]["sha256"][str(src)], str(src))
        g = json.loads(raw)
        check(g["completed"] and g["positions"] == "prompt" and g["identity_sha256"] == export["identity_sha256"]
              and [r["pair_id"] for r in g["rows"]] == ids, str(src))
    return cells, votes


def mean_direction():
    import main_table_prompt as mp
    import posm_table as pm

    per_model = {}
    for name, layer, strength in pm.SETS:
        splits = []
        for i, split in enumerate(("tune", "test")):
            data, why = mp.prompt_split(name, layer, i, strength)
            check(data is not None, why)
            sp, texts, _, ids = data
            _, run, _ = mp.usual(name, i)
            for tag in (f"pospromptl{layer}", f"posprompt2l{layer}"):
                verified_grid(mp, run, split, tag, ids)
            splits.append((sp, texts))
        (tune, tune_text), (test, test_text) = splits
        pick = mp.ML.n1_pick(tune, tune_text, strength)
        n1, n_ans, _ = mp.S.system(test, test_text, pick)
        single, s_ans, _ = mp.S.system(test, test_text, ("E", strength))
        amount = pm.same_write(name, run, layer, strength)
        cells, votes = verified_grid(mp, run, "test", f"pospromptml{layer}", test.qs)
        condition = f"E_L{layer}_x-mdg-p{strength}"
        md = {}
        for cell_id, verdict in votes.items():
            cell = cells[cell_id]
            q = cell["pair_id"]
            if cell["condition"] != condition or q not in test.qs:
                continue
            check(q not in md, f"Duplicate MD vote {q}")
            if verdict["status"] == "scored":
                check(verdict["label"] in ("A", "B", "C", "H"), str(verdict["label"]))
                v = int(verdict["label"] == "A") - int(verdict["label"] == "B")
            else:
                check(verdict.get("basis") == "empty_output_no_judge_vote" and not cell["prediction"].strip(), q)
                v = 0
            md[q] = [v, int(not cell["rule_abstain"] and bool(cell["prediction"].strip()))]
        check(len(md) == len(test.qs) == len(set(test.qs)), f"Incomplete MD grid for {name}")
        rows = {q: [[int(sv), int(sa)], md[q], [int(nv), int(na)]]
                for q, sv, sa, nv, na in zip(test.qs, states(single), s_ans, states(n1), n_ans)}
        per_model[name] = rows
        x = np.asarray(list(rows.values()))
        result = {}
        for ix, method in ((0, "E"), (2, "N1")):
            k, b = compare_pools(np.ones((1, len(x)), dtype=int), x[:, ix], x[:, 1])
            result[method] = [int(k[0]), int(b[0]), int(sum(x[:, ix, 0] - x[:, 1, 0]))]
        emit(model=name, n=len(x), write_amount=amount, k_lower_bound_and_raw_net=result)
    for n_models in (3, 5):
        cohort = list(per_model)[:n_models]
        qs = sorted(set.intersection(*(set(per_model[m]) for m in cohort)))
        w = boot_weights(len(qs))
        for ix, method in ((0, "E"), (2, "N1")):
            raw_total = np.zeros(len(qs), dtype=int)
            bound = np.zeros(len(w), dtype=int)
            for m in cohort:
                x = np.asarray([per_model[m][q] for q in qs])
                raw_total += x[:, ix, 0] - x[:, 1, 0]
                _, b = compare_pools(w, x[:, ix], x[:, 1])
                bound += b
            emit(models=cohort, shared_n=len(qs), method=method,
                 natural_net=mp.ps.ci(raw_total.tolist()), equal_n_lower_bound=interval(bound))


def gsm():
    import cap_score as cs
    import gsm8k_n1_prompt as g
    import release_origin as ro

    ids = g.frozen_ids()
    gold = cs.gold_of("gsm8k")
    outcomes = []
    for model, (el, strength, cl, cap_after_e, cap_own, max_new) in g.SET.items():
        bc, base = g.load_arm(model, "neutral", None, ids)
        ec, e = g.load_arm(model, g.e_arm(model), "prompt", ids)
        check(all(bc[k] == ec[k] for k in ("frozen_file_sha256", "identity_sha256", "system_text", "user_template")), model)
        blanks = [q for q in ids if g.blank(e[q])]
        out = dict(e)
        calls = 0
        if blanks:
            source = Path(g.root(model)) / "n1_blank_ids_at-prompt.json"
            check(json.loads(source.read_text()) == blanks, str(source))
            retries = {}
            for q in blanks:
                own = g.blank(base[q])
                for k in g.LAD:
                    if float(k) > float(cap_own if own else cap_after_e):
                        break
                    if k not in retries:
                        arm = f"neutral:E{el}.w.p{strength}+C{cl}.w.m{k}"
                        meta, retries[k] = g.load_arm(model, arm, "prompt", blanks, source)
                        check(meta["parts"][0] == ec["parts"][0] and meta["identity_sha256"] == ec["identity_sha256"]
                              and meta["engine_sha256"] == ec["engine_sha256"], arm)
                    r = retries[k][q]
                    calls += 1
                    if g.blank(r):
                        continue
                    if own or ro.norm(r["text"]) != ro.norm(base[q]["text"]):
                        out[q] = r
                    break
        scored = np.array([[cs.judge_row("gsm8k", arm[q], gold(q))[0] for arm in (base, e, out)] for q in ids])
        correct = (scored == "R").astype(int)
        unanswered = (scored == "X").astype(int)
        outcomes.append((correct[:, 2] - correct[:, 0], unanswered[:, 2] - unanswered[:, 0]))
        emit(model=model, cap=max_new, blanks=len(blanks), candidate_calls=calls,
             changed_texts=sum(out[q]["text"] != e[q]["text"] for q in blanks),
             correct=correct.sum(0).tolist(), unanswered=unanswered.sum(0).tolist())
    w = boot_weights(len(ids))
    for count in (3, 5):
        for j, key in enumerate(("accuracy", "unanswered")):
            paired = np.mean([model[j] for model in outcomes[:count]], axis=0)
            emit(models=count, measure=key, n=len(ids), change_points=interval(100 * (w @ paired) / len(ids), 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("gsm", "md"))
    args = parser.parse_args()
    if args.only != "gsm":
        mean_direction()
    if args.only != "md":
        gsm()
