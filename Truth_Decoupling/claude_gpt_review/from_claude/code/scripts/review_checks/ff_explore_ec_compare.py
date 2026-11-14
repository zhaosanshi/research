# -*- coding: utf-8 -*-
"""Read-only comparison of from_claude/39 for any model and any pair of E / C layers (Claude, 2026-11-13;
from_claude/44): at the SAME number N of non-abstained answers, E+C against "answering only the N answers the model
itself is most confident in" (one direction, E+C at another C strength, or none). The Llama version is
results/09_E44_两因子交换训练_Llama/scripts/e44_EC_compare.py; this copy changes only what is model- or run-specific: the
condition names are built from the E / C layers and strengths, the two judged work dirs are arguments, and there is one
set of operating points per E strength (the primary C strength of each comes from the choice file written before any
judging). Selection, statistics and the pre-registered reading (39 section 2.1) are unchanged.

Answered / abstained is decided without any correctness label: an answer is abstained when the rule detector flags it
(refusal_rule), and a selection system keeps the N non-abstained answers with the highest confidence (primary: mean
log-probability of its own answer tokens under its own prompt and writes, from ff_explore_ec_confidence.py; ties by
question order). Correctness comes from the judge labels already on disk (both judges, separately). The selection is
fixed on the full question set; the paired bootstrap resamples questions for the difference in right (and wrong)
answers. Every row also reports what random selection of N of its candidates would give on average, and the most it
could give (oracle).

usage: ff_explore_ec_compare.py CONFIDENCE_JSON --ec-work DIR --conf-work DIR --choice CHOICE_JSON
                                [--boot 10000] [--seed 20261112] [--json-out F]
       ff_explore_ec_compare.py --selftest
2026-11-14 (stage B of from_claude/45, the 529 test questions): the questions whose only gold answer is an abstention are
left out exactly as e44_round1_tables.load leaves them out of every table (the test split has 6, the tune split none, so
tune results are unchanged); both work dirs must leave out the same ones, and both must be exports of the confidence
file's split."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402

MEASURES = ("mean_logp", "sum_logp", "first_logp")   # the first is primary (39 section 2); the others only for DECISIVE


def tok(s):
    return s.replace(".", "-")


def ordered(el, cl, e_part, c_part):
    return [e_part, c_part] if el < cl else [c_part, e_part]


def ec_condition(el, cl, s, k):
    return "X_x--" + "-and-".join(p.replace(".", "-") for p in ordered(el, cl, "E%d.w.p%s" % (el, s), "C%d.w.m%s" % (cl, k)))


def plan(el, cl, es, ks):
    """EXPECT (target -> (base, writes)), and per E strength: the E+C conditions by k and the selection systems."""
    expect = {"conf": ("conf", []), "x": ("x", [])}
    for s in es:
        e = "E%d.w.p%s" % (el, s)
        expect["conf_E%s" % s] = ("conf", [e])
        expect["x_E%s" % s] = ("x", [e])
        for k in ks:
            expect["x_E%sC%s" % (s, k)] = ("x", ordered(el, cl, e, "C%d.w.m%s" % (cl, k)))
    for k in ks:
        expect["x_C%s" % k] = ("x", ["C%d.w.m%s" % (cl, k)])
    per_s = {}
    for s in es:
        ec = {k: ec_condition(el, cl, s, k) for k in ks}
        select, decisive = {}, []
        for k in ks:
            name = "E+C -%s, then selection (same prompt)" % k
            select[name] = ("x_E%sC%s" % (s, k), "ec", ec[k])
            decisive.append(name)
        select["E + selection (must-answer prompt)"] = ("conf_E%s" % s, "conf", "E_L%d_conf-w-p%s" % (el, tok(s)))
        decisive.append("E + selection (must-answer prompt)")
        select["E + selection (plain prompt)"] = ("x_E%s" % s, "ec", "E_L%d_x-w-p%s" % (el, tok(s)))
        for k in ks:
            select["C -%s + selection" % k] = ("x_C%s" % k, "ec", "C_L%d_x-w-m%s" % (cl, tok(k)))
        select["no write + selection (must-answer prompt)"] = ("conf", "conf", "baseline_conf")
        select["no write + selection (plain prompt)"] = ("x", "ec", "baseline_x")
        per_s[s] = {"EC": ec, "SELECT": select, "DECISIVE": tuple(decisive)}
    return expect, per_s


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


def run(conf_path, ec_work, conf_work, choice_path, n_boot, seed):
    C = json.loads(Path(conf_path).read_text(encoding="utf-8"))
    if C.get("protocol") != "e44_confidence_v1" or C.get("partial"):
        raise SystemExit("confidence file is not a complete e44_confidence_v1 run")
    sp = C["spec"]
    el, cl, es, ks = sp["e_layer"], sp["c_layer"], sp["e_strengths"], sp["c_strengths"]
    choice = json.loads(Path(choice_path).read_text(encoding="utf-8"))
    if (choice["e_layer"], choice["c_layer"]) != (el, cl) or sorted(choice["choice"]) != sorted(es) or \
            any(k not in ks for k in choice["choice"].values()):
        raise SystemExit("the choice file is for other layers or strengths than the confidence file")
    if choice.get("split", "tune") != C.get("split") or choice.get("run_dir") != sp.get("run_dir") or \
            choice.get("c_sign", "m") != "m":   # red team 10:14 item 4; the selection names are built for C toward confident
        raise SystemExit("the choice file is for another split, run or C sign than the confidence file")
    EXPECT, PER_S = plan(el, cl, es, ks)
    for name, (base, writes) in EXPECT.items():
        T = C["targets"].get(name)
        if T is None or T["base_prompt"] != base or T["writes"] != writes:
            raise SystemExit("confidence target %s is missing or was scored with another prompt / write" % name)
        g = T.get("greedy_reproduction", {})
        if g.get("rate", 0) < 0.98 or (writes and (T.get("first_difference_check") or {}).get("share_toward_target", 0) < 0.9):
            raise SystemExit("confidence target %s did not pass its checks" % name)
    split = C.get("split")
    excluded, ids = None, None
    out = {"confidence_file_sha256": hashlib.sha256(Path(conf_path).read_bytes()).hexdigest(), "identity_sha256": C["identity_sha256"],
           "spec": sp, "split": split, "primary": choice["choice"], "judges": {}, "verdicts": {}}
    for judge in ("pro", "allenai"):
        works = {}
        for w, path in (("ec", ec_work), ("conf", conf_work)):
            gen, specs, wids, state, rule, words = t.load(Path(path), judge)
            if gen.get("split") != split:
                raise SystemExit("%s is an export of the %s split, the confidence file a %s run" % (w, gen.get("split"), split))
            if excluded is None:     # the abstention-gold questions that the table loader leaves out (none on tune)
                excluded = list(gen["_excluded_abstention_gold"])
                ids = [q for q in C["question_ids"] if q not in set(excluded)]
            elif list(gen["_excluded_abstention_gold"]) != excluded:
                raise SystemExit("%s leaves out other abstention-gold questions than the other work dir" % w)
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
        st, ru, _, _, wd = works["ec"]
        res = {"operating_points": {}}
        for s in es:
            EC, SELECT = PER_S[s]["EC"], PER_S[s]["SELECT"]
            DECISIVE = PER_S[s]["DECISIVE"]
            for k, c in EC.items():
                check_same("x_E%sC%s" % (s, k), "ec", c)   # the E+C answers scored are the E+C answers judged
                ec_set = set(q for q in ids if not ru[(q, c)])
                ec_label = {q: st[(q, c)] for q in ids}
                n = len(ec_set)
                k_ec = tally(ids, ec_set, ec_label)
                rows = [{"system": "E +%s, C -%s (C decides what to abstain on)" % (s, k), "N": n, **k_ec,
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
                res["operating_points"]["%s|%s" % (s, k)] = rows
        out["judges"][judge] = res
    for s in es:
        for k in PER_S[s]["EC"]:
            for sname in PER_S[s]["DECISIVE"]:
                per = {}
                for judge, res in out["judges"].items():
                    r = next((r for r in res["operating_points"]["%s|%s" % (s, k)] if r["system"] == sname), None)
                    if r and r.get("reachable"):
                        per[judge] = (r["comparison"]["dR"], r["comparison"]["dR_lo"], r["comparison"]["dR_hi"])
                if len(per) == 2:
                    out["verdicts"]["E +%s, C -%s%s | %s" % (s, k, " (primary)" if choice["choice"][s] == k else "", sname)] = verdict(per)
    out["n_questions"] = len(ids)
    out["excluded_abstention_gold"] = excluded
    return out


def fmt(out):
    L = ["\n%s split: %d questions (left out: %d whose only gold answer is an abstention%s)" % (
        out["split"], out["n_questions"], len(out["excluded_abstention_gold"]),
        ": " + ", ".join(out["excluded_abstention_gold"]) if out["excluded_abstention_gold"] else "")]
    for judge, res in out["judges"].items():
        L.append("\n## judge %s\n" % judge)
        for key, rows in res["operating_points"].items():
            s, k = key.split("|")
            tag = "PRIMARY, chosen by rule abstentions before judging" if out["primary"].get(s) == k else "secondary"
            L.append("### E +%s with C -%s: N = %d non-abstained answers (%s)\n" % (s, k, rows[0]["N"], tag))
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
    L.append("## pre-registered readings (39 section 2.1), both judges together\n")
    for key, v in out["verdicts"].items():
        L.append("- %s: %s" % (key, v))
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
    p.add_argument("confidence_json")
    p.add_argument("--ec-work", required=True, help="judged work dir of the E+C export (x-base conditions)")
    p.add_argument("--conf-work", required=True, help="judged work dir holding baseline_conf and E on conf")
    p.add_argument("--choice", required=True, help="the choice file of ff_explore_ec_check.py (primary k per E strength)")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--json-out", default=None)
    a = p.parse_args()
    out = run(a.confidence_json, a.ec_work, a.conf_work, a.choice, a.boot, a.seed)
    print("# E+C against selection by the model's own confidence at the same N (from_claude/39 procedure; from_claude/44)")
    print(fmt(out))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
