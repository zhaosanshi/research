# -*- coding: utf-8 -*-
"""Read-only summary for P2 of the doctor's package (Claude, 2026-11-14 19:xx): every number of the free-form table on the
529 test questions, recomputed from the judged work dirs with the table code itself (e44_round1_tables.load /
per_question / boot -- the same functions that made every table already reported), both judges, plus the provenance the
doctor asked for (work dir paths, sha256 of the judged inputs and of both judges' verdict files, model configs).
Nothing is written except the JSON named by --out; no model, no judge call.
usage (inside the container, PYTHONPATH = the 13b code copy): ff_p2_summary.py --out FILE.json"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402

W = Path("/workspace/Truth_Decoupling")
LLE = W / "results_e44_factorial_das/llama3.1-8b_side_E2"
LLM = W / "results_e44_factorial_das/llama3.1-8b"
Q9E = W / "results_newmodels_ff_20261112/e44/qwen3.5-9b_E2"
Q9 = W / "results_newmodels_ff_20261112/e44/qwen3.5-9b"
Q27 = W / "results_newmodels_ff_20261112/e44/qwen3.8-27b-dm"
LOGS = W / "results_newmodels_ff_20261112/logs"
CFG = {"llama": (W / "code_e45_20261112/configs/models/llama3.1-8b.json"),
       "9b": (W / "code_ff_explore_20261113b/configs/models/qwen3.5-9b.json"),
       "27b": (W / "code_ff_explore_20261113b/configs/models/qwen3.8-27b-dm.json")}
MODELS = {
    "llama": {"name": "Llama-3.1-8B", "el": 11, "e_work": LLE / "judge_test_testD_E_L11",
              "c": {19: (LLM / "judge_test_testS_C_L19", LOGS / "test_b/ll_ct_L19.log"),
                    31: (LLM / "judge_test_testS_C_L31", LOGS / "test_b/ll_ct_L31.log")},
              "ec": [("C19, rules v1 (R6 layer)", LLE / "eval_extra/judge_test_tll19_test", LLE / "eval_extra/ec_choice_tll19.json", 19),
                     ("C31, rules v1 (second C layer)", LLE / "eval_extra/judge_test_tll31_test", LLE / "eval_extra/ec_choice_tll31.json", 31),
                     ("C19, rules v1.1, E strength 2", LLE / "eval_extra/judge_test_tll19q2m_test", LLE / "eval_extra/ec_choice_tll19q2m.json", 19)],
              "compare": [("C19, rules v1", LLE / "eval_extra/judge_test_tll19_test/compare_same_N.json"),
                          ("C31, rules v1", LLE / "eval_extra/judge_test_tll31_test/compare_same_N.json")]},
    "9b": {"name": "Qwen3.5-9B", "el": 15, "e_work": Q9E / "judge_test_testD_E_L15",
           "c": {L: (Q9 / ("judge_test_testS_C_L%d" % L), LOGS / ("test_b/q9_ct_L%d.log" % L)) for L in (12, 16, 17)},
           "ec": [("C16, rules v1 (R6 layer; within tolerance, so also v1.1)", Q9E / "eval_extra/judge_test_t9b16_test", Q9E / "eval_extra/ec_choice_t9b16.json", 16)],
           "compare": [("C16, rules v1", Q9E / "eval_extra/judge_test_t9b16_test/compare_same_N.json")]},
    "27b": {"name": "Qwen3.8-27B", "el": 28, "e_work": Q27 / "judge_test_testD_E_L28",
            "c": {L: (Q27 / ("judge_test_testS_C_L%d" % L), LOGS / ("explore_p/t27_ct_L%d.log" % L)) for L in (33, 35, 37)},
            "ec": [("C33, rules v1 (C 0.25 overshoots the answers)", Q27 / "eval_extra/judge_test_t2733_test", Q27 / "eval_extra/ec_choice_t2733.json", 33),
                   ("C33, rules v1.1, E strength 1", Q27 / "eval_extra/judge_test_t2733q1m_test", Q27 / "eval_extra/ec_choice_t2733q1m.json", 33)],
            "compare": [("C33, rules v1", Q27 / "eval_extra/judge_test_t2733_test/compare_same_N.json")]},
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prov(work):
    work = Path(work)
    out = {"work_dir": str(work), "judge_input_sha256": sha(work / "judge_input.json")}
    for j in ("pro", "allenai"):
        out["single_%s_sha256" % j] = sha(work / "judge2" / ("single_%s.jsonl" % j))
    x = work / "judge2" / "excluded_questions.json"
    if x.exists():
        out["excluded_questions"] = json.loads(x.read_text(encoding="utf-8"))
    return out


def stat(L, base, cond):
    """dR [CI], dAns [CI], answered / right before and after, mean words, for one judge's load() result L."""
    gen, specs, ids, state, rule, words = L
    if base not in specs and not base.startswith("baseline_"):
        raise SystemExit("unknown base %s" % base)
    if cond not in specs:
        raise SystemExit("condition %s is not judged in this work dir" % cond)
    v, n = t.per_question(state, words, ids, base, cond)
    lo, hi = t.boot(v["dR"])
    alo, ahi = t.boot(v["dAns"])
    return {"dR": sum(v["dR"]), "dR_lo": lo, "dR_hi": hi, "dAns": sum(v["dAns"]), "dAns_lo": alo, "dAns_hi": ahi,
            "answered_base": n["ans_b"], "answered": n["ans_c"], "right_base": n["right_b"], "right": n["right_c"],
            "words_base": round(sum(words[(i, base)] for i in ids) / len(ids), 2),
            "words": round(sum(words[(i, cond)] for i in ids) / len(ids), 2), "n_questions": len(ids)}


def both(work, base, conds):
    out = {}
    for j in ("pro", "allenai"):
        L = t.load(Path(work), j)
        out[j] = {c: stat(L, base, c) for c in conds}
        out["n_questions"] = len(L[2])
        out["excluded_abstention_gold"] = L[0]["_excluded_abstention_gold"]
        out["excluded_by_hand"] = L[0].get("_excluded_by_hand", {})
    return out


def tok(s):
    return str(s).replace(".", "-")


def two(el, cl, e_part, c_part):
    return "X_x--" + "-and-".join(tok(p) for p in ((e_part, c_part) if el < cl else (c_part, e_part)))


def key_lines(log):
    out = {}
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^KEY family=C judge=(\w+) layer=(\d+) A=(\S+) B=(\S+)", line)
        if m:
            out[m.group(1)] = {"A": m.group(3), "B": m.group(4)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {"script_sha256": sha(__file__), "table_code_sha256": sha(t.__file__), "models": {}}
    for key, M in MODELS.items():
        el = M["el"]
        cfg = json.loads(Path(CFG[key]).read_text(encoding="utf-8"))
        R = {"name": M["name"], "config": {k: cfg.get(k) for k in ("name", "path", "modelscope_repo", "hf_repo", "dtype", "num_layers", "hidden_size")}}
        # E on the must-answer prompt (stage-B E work dir)
        ec = ["baseline_conf_E2"] + ["E_L%d_conf-%s-%s" % (el, kind, s) for s in ("p1", "p2") for kind in ("w", "md")] + \
             ["E_L%d_conf-rand-%s" % (el, s) for s in ("p1", "m1", "p2", "m2")]
        R["E_must_answer"] = {"provenance": prov(M["e_work"]), **both(M["e_work"], "baseline_conf", ec)}
        # C toward caution / toward confident at 0.25 on the plain prompt, with the frozen criterion v4 (B column) per judge
        R["C"] = {}
        for L, (work, log) in M["c"].items():
            conds = ["C_L%d_x-%s-%s" % (L, kind, s) for kind in ("w", "rand") for s in ("p0-25", "m0-25")]
            R["C"][str(L)] = {"provenance": prov(work), "criterion_v4": key_lines(log), **both(work, "baseline_x", conds)}
        # E+C on the plain prompt
        R["EC"] = []
        for label, work, choice_path, cl in M["ec"]:
            ch = json.loads(Path(choice_path).read_text(encoding="utf-8"))
            sign = ch.get("c_sign", "m")
            other = "p" if sign == "m" else "m"
            conds = []
            for s, k in ch["choice"].items():
                conds += ["E_L%d_x-w-p%s" % (el, tok(s)), "C_L%d_x-w-%s%s" % (cl, sign, tok(k)),
                          two(el, cl, "E%d.w.p%s" % (el, s), "C%d.w.%s%s" % (cl, sign, k)),
                          two(el, cl, "E%d.rand.p%s" % (el, s), "C%d.w.%s%s" % (cl, sign, k)),
                          two(el, cl, "E%d.rand.m%s" % (el, s), "C%d.w.%s%s" % (cl, sign, k)),
                          two(el, cl, "E%d.w.p%s" % (el, s), "C%d.rand.%s%s" % (cl, sign, k)),
                          two(el, cl, "E%d.w.p%s" % (el, s), "C%d.rand.%s%s" % (cl, other, k))]
            for s in ("1", "2"):    # the random-direction E alone of the E+C export, when it is there
                for sg in ("p", "m"):
                    conds.append("E_L%d_x-rand-%s%s" % (el, sg, s))
            gen = json.loads((Path(work) / "judge_input.json").read_text(encoding="utf-8"))
            conds = [c for c in dict.fromkeys(conds) if c in {s["name"] for s in gen["condition_specs"]}]
            R["EC"].append({"label": label, "c_layer": cl, "choice": ch["choice"], "c_sign": sign,
                            "baseline_rule_abstentions": ch["baseline_rule_abstentions"],
                            "candidates": ch.get("candidates"), "provenance": prov(work),
                            **both(work, "baseline_x", conds)})
        # equal-N comparison (already computed by ff_explore_ec_compare.py; its JSON is copied with its sha256)
        R["equal_N"] = []
        for label, path in M["compare"]:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            rows = {}
            for judge in ("pro", "allenai"):
                for opk, oprows in d["judges"][judge]["operating_points"].items():
                    s, k = opk.split("|")
                    if d["primary"].get(s) != k:
                        continue
                    for r in oprows[1:]:
                        if "[" in r["system"]:
                            continue
                        rows.setdefault("%s|%s" % (s, k), {}).setdefault(judge, {})[r["system"]] = (
                            None if not r.get("reachable") else r["comparison"])
                    rows["%s|%s" % (s, k)].setdefault("N", oprows[0]["N"])
            R["equal_N"].append({"label": label, "file": str(path), "sha256": sha(path), "verdicts": d["verdicts"],
                                 "n_questions": d.get("n_questions"), "excluded_by_hand": d.get("excluded_by_hand", {}),
                                 "primary_rows": rows})
        res["models"][key] = R
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
