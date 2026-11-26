#!/usr/bin/env python3
# Iteration_01 exploration (executor xu, Claude Opus 5.5, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5, segment B (progress entry 2026-11-27 05:12): TruthfulQA test split (main list per model), N1 at the larger E
# strength s2 with and without the frozen switch (stage 1, pool B, chosen layer, ON = p > 0.5). Not reviewed by GPT.
# CPU helper (container python3, from /workspace/Truth_Decoupling); it never loads a model and never calls a judge.
# Frozen: E layer / s / C layer / s2 in SET (checked against gate_sim_1126.STRENGTH / EC, iti_table.KEY and the run's C
# layer); the N1 caps = main_table_layers.n1_pick on the tune split at s (expected values in CAPS_EXPECTED, a difference
# stops); TruthfulQA test is read exactly as diag/gate_sim_1126.py reads it (iti_table: MP.prompt_split, main list cx).
# New arms per model (write rule "prompt", the engine e44_eval_extra.py through scripts/review_checks/e44_positions.py
# (single write) and e44_positions2.py (two writes), as launchers/explore_20261120/pospall.sh made the main table's arms):
#   E2 = x:E<el>.w.p<s2>;  EC2_<k> = x:E<el>.w.p<s2>+C<cl>.w.m<k>, k = 0.25 0.5 1 2.
# Cells that can be handed in by the ungated N1 at s2 (the only ones judged), on every question of the main list:
#   E2's answer when its text is non-empty; where E2 is blank by the rule (refusal rule flag or empty text, the rule of
#   policy_select2.load's blank flag): the EC2 rungs up to the cap of the question's class (own blank = the no-write
#   answer is blank by the rule -> cap for own blanks, else the cap for E's blanks), non-empty texts only.
# Commands:
#   entry-compare MODEL ARM             the partial file t16.<arm>-at-prompt_test.json equals the archived full file of
#                                       the same arm on its first 16 rows (whole rows) and question ids
#   check-gen MODEL                     the five new files: complete, test, prompt rule, arm, cohort, baseline texts,
#                                       engine version, parts (E part = the E(s) file's direction with beta s2 and the
#                                       amount scaled s2 / s; C part = the C part of the E(s) + C(k) file)
#   smoke-ids MODEL                     newsets/gate2b_1127/smoke_ids_gate2b_1127.json (3 main-list ids: ON and E2
#                                       answered; ON and E2 blank and released by a rung; OFF) unless it exists
#   judge-input MODEL full|smoke        the work dir's judge_input_gate2b_1127.json from the unchanged engine export
#                                       judge_test/gate2b_1127_tqa_test.json (rows checked against the generation files;
#                                       main-list questions only, smoke: the 3 ids; cells that cannot be handed in, the
#                                       no-write cells included, get an empty output = no judge call)
import json
import os
import sys
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/code/scripts/capability", "Iteration_01/gpt_tools_20261126/gate_audit"]
import iti_table as A             # noqa: E402
import gate_sim_1126 as GS        # noqa: E402
import release_origin as RO       # noqa: E402
import two_caps as TC             # noqa: E402

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
TAG = "gate2b_1127"
SET = {"llama3.1-8b": (16, "3.5", 19, "5"), "qwen3.5-9b": (15, "2", 16, "3"), "qwen3.8-27b": (28, "1.75", 33, "3"),
       "gemma3-12b": (24, "2.5", 25, "4"), "gemma3-27b": (28, "1", 31, "2")}
CAPS_EXPECTED = {"llama3.1-8b": ("1", "2"), "qwen3.5-9b": ("0-25", "0-5"), "qwen3.8-27b": ("0-5", "1"),
                 "gemma3-12b": ("2", "2"), "gemma3-27b": ("1", "2")}          # (cap of E's blanks, cap of own blanks)
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
LAD = ps.LAD                                   # "0-25" "0-5" "1" "2"
DOT = {"0-25": "0.25", "0-5": "0.5", "1": "1", "2": "2"}
EXPORT_NAME = "gate2b_1127_tqa_test"
JIN = "judge_input_%s.json" % TAG
SMOKE_IDS = "Iteration_01/newsets/gate2b_1127/smoke_ids_gate2b_1127.json"
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
VAL = {"R": 1, "W": -1, "N": 0}


def dash(s):
    return s.replace(".", "-")


def tok(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def arms(m):
    el, s, cl, s2 = SET[m]
    e2 = "x:E%d.w.p%s" % (el, s2)
    out = {"E2": e2}
    for k in LAD:
        out["EC2_" + k] = "%s+C%d.w.m%s" % (e2, cl, DOT[k])
    return out


def conds(m):
    """export condition names (engine condition_of with the arm_token change of e44_positions2.py)"""
    el, s, cl, s2 = SET[m]
    out = {"E2": "E_L%d_x-w-p%s" % (el, dash(s2))}
    for k in LAD:
        out["EC2_" + k] = "X_" + tok(arms(m)["EC2_" + k]) + "-at-prompt"
    return out


def gen_name(arm, partial=""):
    return "%s%s-at-prompt_test.json" % (partial, tok(arm))


def base(m):
    """TruthfulQA test exactly as gate_sim_1126 reads it, plus the run dir and the frozen setting checks."""
    el, s, cl, s2 = SET[m]
    display, layer, sd = A.KEY[m]
    if GS.STRENGTH[m] != (s, s2) or GS.EC[m] != (el, cl) or layer != el or sd != dash(s):
        raise SystemExit("%s: frozen setting %s differs from gate_sim_1126 %s / %s or iti_table.KEY %s" % (
            m, SET[m], GS.STRENGTH[m], GS.EC[m], A.KEY[m]))
    D, run, _ = MP.usual(display, 1)
    if D.m["c"] != cl:
        raise SystemExit("%s: C layer of the run %s, frozen %d" % (m, D.m["c"], cl))
    test, err = MP.prompt_split(display, layer, 1, sd)
    if test is None:
        raise SystemExit("%s: %s" % (m, err))
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, sd, qs=cx)
    tune, err2 = MP.prompt_split(display, layer, 0, sd)
    pick = ML.n1_pick(tune[0], tune[1], sd)
    if pick[0] != "N1" or pick[1] != sd or (pick[2], pick[3]) != CAPS_EXPECTED[m]:
        raise SystemExit("%s: N1 pick on tune %s, expected caps %s" % (m, pick, CAPS_EXPECTED[m]))
    sp, tx, _, ids = test
    return SimpleNamespace(m=m, display=display, el=el, s=s, sd=sd, cl=cl, s2=s2, run=run, sp=sp, tx=tx, ids=list(ids),
                           pick=pick, cap_e=pick[2], cap_own=pick[3])


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_new(B):
    """{name: {pair_id: row}} of the five new generation files (complete test files at the prompt rule)."""
    out = {}
    for n, arm in arms(B.m).items():
        p = os.path.join(B.run, "eval_extra", "gen", gen_name(arm))
        d = read(p)
        if d.get("completed") is not True or d.get("partial") is not False or d.get("split") != "test" or \
                d.get("positions") != "prompt" or d.get("arm") != arm or d.get("max_new_tokens") != 48:
            raise SystemExit("%s: %s is not the complete test file of %s at the prompt rule" % (B.m, p, arm))
        out[n] = {r["pair_id"]: r for r in d["rows"]}
    return out


def blank_row(r):
    """blank by the rule: policy_select2.load's flag = rule_abstain (export: rule_abstains(text) = the generation
    row's refusal_rule) or empty_output (judge: not prediction.strip())"""
    return bool(r["refusal_rule"]) or not r["text"].strip()


def needed(B, new, qs):
    """{name: [ids]} of the cells the ungated N1 at s2 can hand in, on the questions qs."""
    out = {n: [] for n in arms(B.m)}
    for q in qs:
        e2 = new["E2"][q]
        if e2["text"].strip():
            out["E2"].append(q)
        if not blank_row(e2):
            continue
        cap = B.cap_own if B.sp.blank[(q, "baseline_x")] else B.cap_e
        for k in TC.lad(cap):
            if new["EC2_" + k][q]["text"].strip():
                out["EC2_" + k].append(q)
    return out


def n1_rule(qs, base_blank, base_text, e_cell, rung_cell, cap_e, cap_own):
    """N1 by the method's text rule (final_records.ladder_record with veto): e_cell[q] = (state, blank, text);
    rung_cell[k][q] = (state, blank, text) for k in LAD. Returns {q: dict(state, answered, note, cond, cand, rung)}."""
    out = {}
    for q in qs:
        st, bl, _ = e_cell[q]
        if not bl:
            out[q] = dict(state=st, answered=True, note="E answered", cond="E", cand=None, rung=None)
            continue
        cand = "own" if base_blank[q] else "E's"
        cap = cap_own if cand == "own" else cap_e
        if cap is None:
            out[q] = dict(state=st, answered=False, note="not re-asked", cond="E", cand=cand, rung=None)
            continue
        k = next((k for k in TC.lad(cap) if not rung_cell[k][q][1]), None)
        if k is None:
            later = next((k for k in LAD if not rung_cell[k][q][1]), None)
            out[q] = dict(state=st, answered=False, note="above the cap" if later is not None else "blank at every dose",
                          cond="E", cand=cand, rung=later)
            continue
        if cand == "E's" and RO.norm(base_text[q]) == RO.norm(rung_cell[k][q][2]):
            out[q] = dict(state=st, answered=False, note="vetoed", cond="E", cand=cand, rung=k)
            continue
        rs, rb, _ = rung_cell[k][q]
        out[q] = dict(state=rs, answered=not rb, note="released", cond=k, cand=cand, rung=k)
    return out


def entry_compare(m, arm):
    B = base(m)
    full = read(os.path.join(B.run, "eval_extra", "gen", gen_name(arm)))
    part = read(os.path.join(B.run, "eval_extra", "gen", gen_name(arm, "t16.")))
    if part.get("partial") is not True or part.get("arm") != arm or full.get("partial") is not False:
        raise SystemExit("%s: %s: not a partial / full pair" % (m, arm))
    n = len(part["rows"])
    if n != 16 or part["question_ids"] != full["question_ids"][:16]:
        raise SystemExit("%s: %s: partial file covers %d rows / other questions" % (m, arm, n))
    bad = [i for i in range(n) if part["rows"][i] != full["rows"][i]]
    keys = ("parts", "positions", "own_sha256", "identity_sha256", "frozen_file_sha256", "max_new_tokens", "base_prompt")
    diff = [k for k in keys if part.get(k) != full.get(k)]
    if bad or diff:
        raise SystemExit("%s: %s: %d of %d rows differ (%s); fields differing %s" % (m, arm, len(bad), n, bad[:4], diff))
    print("%s entry check %s: %d / %d rows identical to the archived file (whole rows), fields %s equal" % (m, arm, n, n, "/".join(keys)))


def check_gen(m):
    B = base(m)
    el, s, cl, s2 = SET[m]
    g = os.path.join(B.run, "eval_extra", "gen")
    e1 = read(os.path.join(g, gen_name("x:E%d.w.p%s" % (el, s))))
    for n, arm in arms(m).items():
        d = read(os.path.join(g, gen_name(arm)))
        if d.get("completed") is not True or d.get("partial") is not False or d.get("split") != "test" or \
                d.get("positions") != "prompt" or d.get("arm") != arm or d.get("max_new_tokens") != 48:
            raise SystemExit("%s: %s is not complete / test / prompt / 48 tokens" % (m, arm))
        for k in ("question_ids", "cohort_sha256", "identity_sha256", "frozen_file_sha256", "own_sha256", "base_prompt"):
            if d.get(k) != e1.get(k):
                raise SystemExit("%s: %s: %s differs from the E(s) file" % (m, arm, k))
        if d["generation_contract"].get("seed") != e1["generation_contract"].get("seed"):
            raise SystemExit("%s: %s: seed" % (m, arm))
        if [r["baseline_text"] for r in d["rows"]] != [r["baseline_text"] for r in e1["rows"]]:
            raise SystemExit("%s: %s: baseline texts differ from the E(s) file" % (m, arm))
        pe, p1 = d["parts"][0], e1["parts"][0]
        keys = ("family", "layer", "kind", "direction_run", "direction_file", "direction_file_sha256",
                "direction_record_sha256", "training_frozen_file_sha256", "e_on", "e_off", "gap")
        if any(pe.get(k) != p1.get(k) for k in keys) or pe["beta"] != float(s2) or \
                abs(pe["amount"] - p1["amount"] * float(s2) / float(s)) > 1e-6 * abs(p1["amount"] * float(s2) / float(s)):
            raise SystemExit("%s: %s: E part %s against E(s) %s" % (m, arm, pe, p1))
        line = "E part = E(s) direction, beta %s, amount %.6g = %.6g x %s / %s" % (pe["beta"], pe["amount"], p1["amount"], s2, s)
        if n.startswith("EC2_"):
            k = n[4:]
            ec1 = read(os.path.join(g, gen_name("x:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, DOT[k]))))
            if len(d["parts"]) != 2 or d["parts"][1] != ec1["parts"][1] or ec1["parts"][0] != p1:
                raise SystemExit("%s: %s: C part differs from the existing E(s) + C(%s) file" % (m, arm, k))
            line += "; C part = the C part of the existing E(s) + C(%s) file (beta %s)" % (DOT[k], d["parts"][1]["beta"])
        elif len(d["parts"]) != 1:
            raise SystemExit("%s: %s: parts" % (m, arm))
        nb = sum(blank_row(r) for r in d["rows"])
        print("%s %s: %d rows, blank by the rule %d; %s" % (m, arm, len(d["rows"]), nb, line))


def smoke_ids(m):
    import falseqa_matched as FM
    if os.path.exists(SMOKE_IDS):
        print("smoke ids exist: %s" % read(SMOKE_IDS))
        return
    B = base(m)
    new = load_new(B)
    on = set(FM.verify_gate(m, "tq_test", B.ids))
    rc = {k: {q: ("?", blank_row(new["EC2_" + k][q]), new["EC2_" + k][q]["text"]) for q in B.ids} for k in LAD}
    ec = {q: ("?", blank_row(new["E2"][q]), new["E2"][q]["text"]) for q in B.ids}
    bb = {q: B.sp.blank[(q, "baseline_x")] for q in B.ids}
    bt = {q: B.tx.ec[(q, "baseline_x")] for q in B.ids}
    r = n1_rule(B.ids, bb, bt, ec, rc, B.cap_e, B.cap_own)
    a = next(q for q in B.ids if q in on and r[q]["note"] == "E answered")
    b = next(q for q in B.ids if q in on and r[q]["note"] == "released")
    c = next(q for q in B.ids if q not in on)
    os.makedirs(os.path.dirname(SMOKE_IDS), exist_ok=True)
    with open(SMOKE_IDS, "x", encoding="utf-8") as f:
        json.dump({"model": m, "ids": [a, b, c], "kinds": ["ON, E2 answered", "ON, E2 blank, released", "OFF"]}, f, indent=1)
    print("smoke ids written: %s %s %s" % (a, b, c))


def work(B, mode):
    return os.path.join(B.run, "eval_extra", "judge_test_%s_%s_test" % (TAG, "smoke" if mode == "smoke" else "tqa"))


def judge_input(m, mode):
    B = base(m)
    new = load_new(B)
    cn = conds(m)
    ex = read(os.path.join(B.run, "eval_extra", "judge_test", EXPORT_NAME + ".json"))
    if ex.get("completed") is not True or ex.get("split") != "test" or sorted(ex["conditions"]) != sorted(["baseline_x"] + list(cn.values())):
        raise SystemExit("%s: the export is not the five new arms (conditions %s)" % (m, ex.get("conditions")))
    rows = {r["pair_id"]: r for r in ex["evaluations"]}
    bad = 0
    for q, r in rows.items():
        for n, c in cn.items():
            g = new[n][q]
            if r["outputs"][c] != g["text"] or r["rule_abstain"][c] != bool(g["refusal_rule"]):
                bad += 1
        if r["outputs"]["baseline_x"] != B.tx.ec.get((q, "baseline_x"), r["outputs"]["baseline_x"]):
            bad += 1
    if bad:
        raise SystemExit("%s: %d export cells differ from the generation files / the usual no-write texts" % (m, bad))
    qs = list(B.ids) if mode == "full" else [q for q in B.ids if q in set(read(SMOKE_IDS)["ids"])]
    if mode == "smoke" and len(qs) != 3:
        raise SystemExit("smoke ids")
    need = needed(B, new, qs)
    keep = {(q, cn[n]) for n, l in need.items() for q in l}
    out_rows, withheld = [], 0
    for q in qs:
        r = json.loads(json.dumps(rows[q]))
        for c in list(r["outputs"]):
            if (q, c) not in keep and r["outputs"][c].strip():
                r["outputs"][c] = ""
                withheld += 1
        out_rows.append(r)
    payload = dict(ex)
    payload["evaluations"] = out_rows
    payload["question_ids"] = qs
    payload[TAG] = {"mode": mode, "main_list": len(B.ids), "questions": len(qs), "caps": [B.cap_e, B.cap_own],
                    "needed": {cn[n]: l for n, l in need.items()}, "withheld_nonempty_cells": withheld,
                    "rule": "main-list questions; E2 answer (non-empty text); where E2 is blank by the rule: E2+C rungs "
                            "up to the cap of the class (non-empty); every other cell, the no-write cells included, has an "
                            "empty output (no judge call)"}
    payload = json.loads(json.dumps(payload))
    w = work(B, mode)
    out = os.path.join(w, JIN)
    if os.path.exists(out):
        if read(out) != payload:
            raise SystemExit("%s exists and differs: stop" % out)
        print("%s exists and equals the build" % out)
        return
    os.makedirs(os.path.join(w, "judge2", "cache"), exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("%s %s: %d questions; export rows equal the generation files; cells that can be handed in %s; withheld non-empty %d -> %s" % (
        m, mode, len(qs), " ".join("%s %d" % (n, len(l)) for n, l in need.items()), withheld, out))


def main():
    a = sys.argv[1:]
    if a[:1] == ["entry-compare"] and len(a) == 3:
        entry_compare(a[1], a[2])
    elif a[:1] == ["check-gen"] and len(a) == 2:
        check_gen(a[1])
    elif a[:1] == ["smoke-ids"] and len(a) == 2:
        smoke_ids(a[1])
    elif a[:1] == ["judge-input"] and len(a) == 3 and a[2] in ("full", "smoke"):
        judge_input(a[1], a[2])
    else:
        raise SystemExit("usage: tqa_gate2b_1127.py entry-compare MODEL ARM | check-gen MODEL | smoke-ids MODEL | judge-input MODEL full|smoke")


if __name__ == "__main__":
    main()
