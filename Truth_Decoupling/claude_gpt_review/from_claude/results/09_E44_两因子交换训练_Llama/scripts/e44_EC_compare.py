# -*- coding: utf-8 -*-
"""Read-only comparison of from_claude/39: at the SAME number N of non-abstained answers, E+C against
"answering only the N answers the model itself is most confident in" (one direction, E+C at another C strength, or none).

Answered / abstained is decided without any correctness label: an answer is abstained when the rule detector flags it
(refusal_rule), and a selection system keeps the N non-abstained answers with the highest confidence (primary: mean
log-probability of its own answer tokens under its own prompt and writes, from e44_confidence.py; ties by question order).
Correctness comes from the judge labels already on disk (both judges, separately). The selection is fixed on the full
question set; the paired bootstrap resamples questions for the difference in right (and wrong) answers. Every row also
reports what random selection of N of its candidates would give on average, and the most it could give (oracle).

usage: e44_EC_compare.py RUN_DIR CONFIDENCE_JSON [--boot 10000] [--seed 20261112] [--json-out F]
       e44_EC_compare.py --selftest"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402

EC = {"0.25": "X_x--E10-w-p2-and-C31-w-m0-25", "0.5": "X_x--E10-w-p2-and-C31-w-m0-5", "1": "X_x--E10-w-p2-and-C31-w-m1"}
PRIMARY_K = "0.25"                       # chosen in from_claude/36 by the rule-abstention rule, before judging
# selection systems: name -> (confidence target, judge work dir under eval_extra/, condition name there)
SELECT = {
    "E+C -0.5, then selection (same prompt)": ("x_EC05", "judge_tune_ec_tune", EC["0.5"]),
    "E+C -1, then selection (same prompt)": ("x_EC1", "judge_tune_ec_tune", EC["1"]),
    "E + selection (must-answer prompt)": ("conf_E", "judge_tune_split_tune", "E_L10_conf-w-p2"),
    "E + selection (plain prompt)": ("x_E", "judge_tune_ec_tune", "E_L10_x-w-p2"),
    "C -0.25 + selection": ("x_C025", "judge_tune_ec_tune", "C_L31_x-w-m0-25"),
    "C -0.5 + selection": ("x_C05", "judge_tune_ec_tune", "C_L31_x-w-m0-5"),
    "C -1 + selection": ("x_C1", "judge_tune_ec_tune", "C_L31_x-w-m1"),
    "no write + selection (must-answer prompt)": ("conf", "judge_tune_split_tune", "baseline_conf"),
    "no write + selection (plain prompt)": ("x", "judge_tune_ec_tune", "baseline_x"),
}
DECISIVE = ("E+C -0.5, then selection (same prompt)", "E+C -1, then selection (same prompt)", "E + selection (must-answer prompt)")
EXPECT = {"conf_E": ("conf", ["E10.w.p2"]), "conf": ("conf", []), "x": ("x", []), "x_E": ("x", ["E10.w.p2"]),
          "x_C025": ("x", ["C31.w.m0.25"]), "x_C05": ("x", ["C31.w.m0.5"]), "x_C1": ("x", ["C31.w.m1"]),
          "x_EC025": ("x", ["E10.w.p2", "C31.w.m0.25"]), "x_EC05": ("x", ["E10.w.p2", "C31.w.m0.5"]),
          "x_EC1": ("x", ["E10.w.p2", "C31.w.m1"])}
MEASURES = ("mean_logp", "sum_logp", "first_logp")   # the first is primary (39 section 2); the others only for DECISIVE


def select_top(ids, cands, conf, n):
    """The n candidates with the highest confidence; ties by question order. None when fewer than n candidates."""
    if n > len(cands):
        return None
    order = {q: i for i, q in enumerate(ids)}
    return set(sorted(cands, key=lambda q: (-conf[q], order[q]))[:n])


def tally(ids, chosen, label):
    k = {"R": 0, "W": 0, "H": 0, "X": 0}
    for q in ids:
        if q in chosen:
            k[label[q]] += 1
    return k


def boot_diff(ids, a, b, label_a, label_b, n_boot, seed, what="R"):
    """Paired bootstrap over questions of (count of `what` among system a's answers) - (same for system b)."""
    import numpy as np
    da = np.array([int(q in a and label_a[q] == what) - int(q in b and label_b[q] == what) for q in ids], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sims = np.sort(da[rng.integers(0, len(ids), size=(n_boot, len(ids)))].sum(axis=1))
    return int(da.sum()), int(sims[int(0.025 * n_boot)]), int(sims[int(0.975 * n_boot) - 1])


def verdict(by_judge):
    """Pre-registered reading (39 section 2) of one comparator over both judges: by_judge = {judge: (dR, lo, hi)}."""
    v = list(by_judge.values())
    if all(d > 0 and lo > 0 for d, lo, _ in v):
        return "combination wins (both judges, intervals above 0)"
    if all(d < 0 and hi < 0 for d, _, hi in v):
        return "comparator wins (both judges, intervals below 0)"
    if all(d > 0 for d, _, _ in v):
        return "leans to the combination (both point estimates > 0; intervals include 0)"
    if all(d < 0 for d, _, _ in v):
        return "leans to the comparator (both point estimates < 0; intervals include 0)"
    return "cannot tell"


def text_sha(text):
    return hashlib.sha1((text or "").strip().encode("utf-8")).hexdigest()


def run(run_dir, conf_path, n_boot, seed):
    C = json.loads(Path(conf_path).read_text(encoding="utf-8"))
    if C.get("protocol") != "e44_confidence_v1" or C.get("partial"):
        raise SystemExit("confidence file is not a complete e44_confidence_v1 run")
    for name, (base, writes) in EXPECT.items():
        T = C["targets"].get(name)
        if T is None or T["base_prompt"] != base or T["writes"] != writes:
            raise SystemExit("confidence target %s is missing or was scored with another prompt / write" % name)
        g = T.get("greedy_reproduction", {})
        if g.get("rate", 0) < 0.98 or (writes and (T.get("first_difference_check") or {}).get("share_toward_target", 0) < 0.9):
            raise SystemExit("confidence target %s did not pass its checks" % name)
    ids = C["question_ids"]
    out = {"confidence_file_sha256": hashlib.sha256(Path(conf_path).read_bytes()).hexdigest(), "identity_sha256": C["identity_sha256"],
           "judges": {}, "verdicts": {}}
    for judge in ("pro", "allenai"):
        works = {}
        for w in ("judge_tune_ec_tune", "judge_tune_split_tune"):
            gen, specs, wids, state, rule, words = t.load(Path(run_dir) / "eval_extra" / w, judge)
            if wids != ids:
                raise SystemExit("%s covers other questions or another order than the confidence file" % w)
            texts = {(r["pair_id"], c): o for r in gen["evaluations"] for c, o in r["outputs"].items()}
            works[w] = (state, rule, texts, specs, words)

        def check_same(target, w, cond):
            st, ru, tx, specs, _ = works[w]
            if cond not in specs:
                raise SystemExit("%s is not judged in %s" % (cond, w))
            tr = {r["pair_id"]: r for r in C["targets"][target]["rows"]}
            for q in ids:   # the scored answers are the judged answers, and the abstention flag is the judge's rule flag
                if tr[q]["text_sha1"] != text_sha(tx[(q, cond)]) or tr[q]["refusal_rule"] != ru[(q, cond)]:
                    raise SystemExit("%s / %s: scored answer differs from the judged one for %s" % (target, cond, q))
            return tr
        st, ru, _, _, wd = works["judge_tune_ec_tune"]
        check_same("x_EC025", "judge_tune_ec_tune", EC["0.25"])
        res = {"operating_points": {}}
        for k, c in EC.items():
            ec_set = set(q for q in ids if not ru[(q, c)])
            ec_label = {q: st[(q, c)] for q in ids}
            n = len(ec_set)
            k_ec = tally(ids, ec_set, ec_label)
            rows = [{"system": "E+C -%s (C decides what to abstain on)" % k, "N": n, **k_ec,
                     "words": sum(wd[(q, c)] for q in ec_set) / n, "comparison": None}]
            for sname, (target, w, cond) in SELECT.items():
                if cond == c:
                    continue                          # never compare a system with itself
                tr = check_same(target, w, cond)
                st2, ru2, _, _, wd2 = works[w]
                label = {q: st2[(q, cond)] for q in ids}
                cands = [q for q in ids if not ru2[(q, cond)]]
                r_all = sum(1 for q in cands if label[q] == "R")
                for m in MEASURES:
                    if m != MEASURES[0] and sname not in DECISIVE:
                        continue
                    row = {"system": sname + ("" if m == MEASURES[0] else " [%s]" % m), "N": n, "candidates": len(cands),
                           "R_all_candidates": r_all}
                    chosen = select_top(ids, cands, {q: tr[q][m] for q in ids}, n)
                    if chosen is None:
                        row["reachable"] = False
                    else:
                        dR = boot_diff(ids, ec_set, chosen, ec_label, label, n_boot, seed, "R")
                        dW = boot_diff(ids, ec_set, chosen, ec_label, label, n_boot, seed, "W")
                        row.update(tally(ids, chosen, label))
                        row.update({"reachable": True, "words": sum(wd2[(q, cond)] for q in chosen) / n,
                                    "R_random": r_all * n / len(cands), "R_oracle": min(n, r_all),
                                    "comparison": {"dR": dR[0], "dR_lo": dR[1], "dR_hi": dR[2], "dW": dW[0], "dW_lo": dW[1], "dW_hi": dW[2]}})
                    rows.append(row)
            res["operating_points"][k] = rows
        out["judges"][judge] = res
    for k in EC:
        for sname in DECISIVE:
            per = {}
            for judge, res in out["judges"].items():
                r = next((r for r in res["operating_points"][k] if r["system"] == sname), None)
                if r and r.get("reachable"):
                    per[judge] = (r["comparison"]["dR"], r["comparison"]["dR_lo"], r["comparison"]["dR_hi"])
            if len(per) == 2:
                out["verdicts"]["%s | %s" % (k, sname)] = verdict(per)
    return out


def fmt(out):
    L = []
    for judge, res in out["judges"].items():
        L.append("\n## judge %s\n" % judge)
        for k, rows in res["operating_points"].items():
            tag = "PRIMARY, pre-registered" if k == PRIMARY_K else "secondary"
            L.append("### E+C with C -%s: N = %d non-abstained answers (%s)\n" % (k, rows[0]["N"], tag))
            L.append("| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                if r.get("reachable") is False:
                    L.append("| %s | %d (%d) | fewer candidates than N | — | — | — | — | — | — |" % (r["system"], r["candidates"], r["R_all_candidates"]))
                    continue
                acc = 100.0 * r["R"] / (r["R"] + r["W"]) if r["R"] + r["W"] else float("nan")
                if r["comparison"] is None:
                    L.append("| **%s** | — | **%d / %d / %d / %d** | %.1f%% | %.1f | — | — | — | — |" % (r["system"], r["R"], r["W"], r["H"], r["X"], acc, r["words"]))
                    continue
                c = r["comparison"]
                L.append("| %s | %d (%d) | %d / %d / %d / %d | %.1f%% | %.1f | %+d [%+d, %+d] | %+d [%+d, %+d] | %.1f | %d |" % (
                    r["system"], r["candidates"], r["R_all_candidates"], r["R"], r["W"], r["H"], r["X"], acc, r["words"],
                    c["dR"], c["dR_lo"], c["dR_hi"], c["dW"], c["dW_lo"], c["dW_hi"], r["R_random"], r["R_oracle"]))
            L.append("")
    L.append("## pre-registered readings (39 section 2), both judges together\n")
    for key, v in out["verdicts"].items():
        L.append("- C -%s: %s" % (key, v))
    return "\n".join(L)


def selftest():
    ids = ["a", "b", "c", "d", "e"]
    conf = {"a": -0.1, "b": -0.5, "c": -0.1, "d": -2.0, "e": -0.3}
    assert select_top(ids, ids, conf, 2) == {"a", "c"}                       # tie -0.1: both kept
    assert select_top(ids, ids, conf, 1) == {"a"}                            # tie broken by question order
    assert select_top(ids, ["b", "d"], conf, 3) is None                      # unreachable
    label = {"a": "R", "b": "W", "c": "R", "d": "R", "e": "H"}
    assert tally(ids, {"a", "b", "e"}, label) == {"R": 1, "W": 1, "H": 1, "X": 0}
    d, lo, hi = boot_diff(ids, {"a", "c", "d"}, {"a", "b"}, label, label, 2000, 1)
    assert d == 2 and lo <= d <= hi
    assert boot_diff(ids, {"a"}, {"a"}, label, label, 500, 1) == (0, 0, 0)
    assert boot_diff(ids, {"a", "b"}, {"a"}, label, label, 500, 1, "W")[0] == 1
    assert verdict({"pro": (5, 1, 9), "allenai": (3, 1, 6)}).startswith("combination wins")
    assert verdict({"pro": (-5, -9, -1), "allenai": (-3, -6, -1)}).startswith("comparator wins")
    assert verdict({"pro": (5, -1, 9), "allenai": (3, 1, 6)}).startswith("leans to the combination")
    assert verdict({"pro": (5, -1, 9), "allenai": (-3, -6, 1)}) == "cannot tell"
    print("selftest passed")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("confidence_json")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--json-out", default=None)
    a = p.parse_args()
    out = run(a.run_dir, a.confidence_json, a.boot, a.seed)
    print("# E+C against selection by the model's own confidence at the same N (from_claude/39)")
    print(fmt(out))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
