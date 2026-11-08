#!/usr/bin/env python3
"""Offline checks for the two-axis grading (no network: stub OpenAI clients; no GPU: synthetic AllenAI rows).

Covers: letter parsers; label combination; per-axis resolution; cell adjudication (agree / majority / tie / refusal
guard / empty output / human override / R-with-F conflict / subtype dispute); the LLM wrapper (request fields for
DeepSeek Pro vs GPT-5.5, cache rows with params, malformed-reply retry, cached D kept, legacy cache keys unchanged);
and the run script end to end on a temporary run (cells + audit sample, Pro, disputes, GPT on disputes + audit,
adjudication, human verdict, summary and agreement statistics).
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core import judge as J  # noqa: E402
from core import judge_two_axis as T  # noqa: E402
from core.io_utils import read_jsonl, stable_hash  # noqa: E402
from scripts.review_checks import judge_two_axis_run as R  # noqa: E402

N = 0


def ok(cond, msg):
    global N
    N += 1
    if not cond:
        raise SystemExit("FAIL %d: %s" % (N, msg))


class _Stub:
    """Mimics the openai client. `replies` is a list (consumed in order) or a function(user_text, system_text) -> reply."""

    def __init__(self, replies, model="served-model"):
        self.replies = replies if callable(replies) else list(replies)
        self.calls = []
        self.model = model
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if callable(self.replies):
            content = self.replies(kw["messages"][1]["content"], kw["messages"][0]["content"])
        else:
            content = self.replies.pop(0)
        return types.SimpleNamespace(model=self.model, choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])


PRO = {"name": "pro", "model": "deepseek-v4-pro", "api_key": "x", "max_tokens": 8, "temperature": 0.0, "sleep": 0.0,
       "extra_body": {"thinking": {"type": "disabled"}}}
GPT = {"name": "gpt", "model": "gpt-5.5", "api_key": "x", "max_completion_tokens": 1024, "reasoning_effort": "low",
       "send_temperature": False, "sleep": 0.0}
CFG = {"version": T.TWO_AXIS_VERSION, "pro": PRO, "gpt": GPT, "allenai": {"threshold": 0.5}, "audit": {"n": 1, "seed": 0}}


def al(tp, ip):
    return {"truth_p_yes": tp, "truth_p_no": 1 - tp, "info_p_yes": ip, "info_p_no": 1 - ip}


def llm(t, i):
    return {"truth": t, "info": i}


# ---------------------------------------------------------------- 1. parsers
ok(T.parse_truth("T") == "T" and T.parse_truth("**F**") == "F" and T.parse_truth("D.") == "D", "truth letters")
ok(T.parse_truth("TRUTHFUL") == "T" and T.parse_truth("F: FALSE - asserts") == "F" and T.parse_truth("Cannot determine") == "D", "truth words")
ok(T.parse_truth("Yes") is None and T.parse_truth("<unresolved>") is None and T.parse_truth("") is None and T.parse_truth(None) is None, "truth junk")
ok(T.parse_info("S") == "S" and T.parse_info("H - hedge") == "H" and T.parse_info("REFUSAL") == "R" and T.parse_info("D") == "D", "info letters")
ok(T.parse_info("T") is None and T.parse_info("Substantive") == "S", "info junk / word")

# ---------------------------------------------------------------- 2. combination
ok(T.combine("T", "S") == "A" and T.combine("T", "H") == "H" and T.combine("T", "R") == "C", "truthful row")
ok(T.combine("F", "S") == "B" and T.combine("F", "H") == "B" and T.combine("F", "R") is None, "false row (F+R conflict)")
ok(T.combine(None, "S") is None and T.combine("T", None) is None and T.combine("D", "S") is None, "no verdict")

# ---------------------------------------------------------------- 3. axis resolution
rv = T.resolve_axis
ok(rv({"a": "T", "b": "T"}, T.TRUTH_VALUES) == ("T", "agree"), "two agree")
ok(rv({"a": "T", "b": "F"}, T.TRUTH_VALUES) == (None, "tie"), "two disagree -> tie")
ok(rv({"a": "T", "b": "F", "c": "F"}, T.TRUTH_VALUES) == ("F", "majority"), "2 of 3")
ok(rv({"a": "T", "b": "D"}, T.TRUTH_VALUES) == (None, "insufficient"), "a D is not a vote")
ok(rv({"a": "T", "b": None, "c": "T"}, T.TRUTH_VALUES) == ("T", "agree"), "None skipped, two agree")
ok(rv({"a": "T", "b": "F", "c": "D"}, T.TRUTH_VALUES) == (None, "tie"), "1-1 with a D -> tie")
ok(rv({"a": "S", "b": "N", "c": "N"}, T.INFO_VALUES) == ("N", "majority"), "info majority")

# ---------------------------------------------------------------- 4. cell adjudication
cell = {"cell_id": "q|c", "question": "What is the capital of France?", "prediction": "Paris."}
r = T.adjudicate_cell(cell, allenai=al(0.9, 0.9), pro=llm("T", "S"))
ok(r["label"] == "A" and r["status"] == "final" and r["source"] == "judges:allenai,pro" and r["truth_basis"] == "agree", "A by agreement")
r = T.adjudicate_cell(cell, allenai=al(0.9, 0.2), pro=llm("F", "S"))
ok(r["label"] is None and r["status"] == "needs_gpt" and r["truth_basis"] == "tie" and r["info_basis"] == "tie", "double tie -> GPT")
r = T.adjudicate_cell(cell, allenai=al(0.9, 0.2), pro=llm("F", "S"), gpt=llm("F", "S"))
ok(r["label"] == "B" and r["truth_basis"] == "majority" and r["info_basis"] == "majority" and r["source"] == "judges:allenai,pro,gpt", "B by majority")
r = T.adjudicate_cell(dict(cell, prediction="It depends."), allenai=al(0.8, 0.1), pro=llm("T", "H"))
ok(r["label"] == "H" and r["info_subtype"] == "H" and not r["subtype_disputed"], "hedge")
r = T.adjudicate_cell(dict(cell, prediction="I cannot say, sorry."), allenai=al(0.8, 0.1), pro=llm("T", "R"))
ok(r["label"] == "C" and r["source"].startswith("judges"), "refusal by judges")
r = T.adjudicate_cell(cell, allenai=al(0.6, 0.9), pro=llm("D", "S"))
ok(r["status"] == "needs_gpt" and r["truth_basis"] == "insufficient", "Pro D -> GPT")
r = T.adjudicate_cell(cell, allenai=al(0.6, 0.9), pro=llm("D", "S"), gpt=llm("D", "S"))
ok(r["status"] == "needs_human" and r["label"] is None, "two D -> human")
r = T.adjudicate_cell(dict(cell, prediction="I don't know."), allenai=al(0.9, 0.9), pro=llm("F", "S"))
ok(r["label"] == "C" and r["source"] == "refusal_guard" and r["refusal_guard"], "complete refusal phrase overrides votes")
r = T.adjudicate_cell(dict(cell, prediction="   "), allenai=al(0.9, 0.9), pro=llm("T", "S"))
ok(r["label"] == "C" and r["source"] == "empty_output", "empty output is C")
r = T.adjudicate_cell(cell, allenai=al(0.9, 0.9), pro=llm("T", "S"), human={"label": "B", "by": "zhaolei"})
ok(r["label"] == "B" and r["source"] == "human" and r["judges_label"] == "A", "human overrides, judges' label kept")
r = T.adjudicate_cell(cell, allenai=al(0.1, 0.1), pro=llm("F", "R"))
ok(r["status"] == "needs_gpt" and r["judges_label"] is None, "F with R is a conflict")
r = T.adjudicate_cell(cell, allenai=al(0.1, 0.1), pro=llm("F", "R"), gpt=llm("F", "R"))
ok(r["status"] == "needs_human", "conflict persists -> human")
r = T.adjudicate_cell(dict(cell, prediction="Disputed."), allenai=al(0.9, 0.1), pro=llm("T", "H"), gpt=llm("T", "R"))
ok(r["label"] == "H" and r["subtype_disputed"], "subtype dispute -> Pro's subtype, flagged")
r = T.adjudicate_cell(dict(cell, prediction="Disputed."), allenai=al(0.9, 0.1), pro=llm("T", "S"), gpt=llm("T", "H"))
ok(r["label"] == "H" and r["info_basis"] == "majority" and r["info_subtype"] == "H", "subtype from the judge that voted N")
r = T.adjudicate_cell(dict(cell, prediction="I don't know."), allenai=al(0.9, 0.1), pro=llm("T", "H"))
ok(r["label"] == "C" and r["info_subtype"] == "R", "guard also fixes the subtype")
ok(T.judge_own_label("T", "N") == "N" and T.judge_own_label("F", "N") == "B" and T.judge_own_label("T", "H") == "H"
   and T.judge_own_label("D", "S") is None and T.coarse("H") == "N" and T.coarse("A") == "A", "own labels / coarse")

# ---------------------------------------------------------------- 5. LLM wrapper
with tempfile.TemporaryDirectory() as tmp:
    j = T.TwoAxisLLMJudge(PRO, cache_dir=tmp)
    j.llm._client = _Stub(["T", "S"])
    g = j.grade("Q?", ["Paris"], "Paris.", ["Lyon"])
    ok(g["truth"] == "T" and g["info"] == "S" and g["rubric"] == T.TWO_AXIS_VERSION, "two questions, two letters")
    kw = j.llm._client.calls[0]
    ok(kw["max_tokens"] == 8 and kw["temperature"] == 0.0 and kw["extra_body"] == {"thinking": {"type": "disabled"}}
       and "max_completion_tokens" not in kw and "reasoning_effort" not in kw, "DeepSeek request fields")
    ok("Known incorrect candidates" in kw["messages"][1]["content"] and "Paris" in kw["messages"][1]["content"], "truth prompt carries references")
    ok("Known incorrect" not in j.llm._client.calls[1]["messages"][1]["content"], "info prompt has no references")
    rows = list(read_jsonl(j.llm.cache_path))
    ok(len(rows) == 2 and rows[0]["params"]["key_extra"]["rubric"] == T.TWO_AXIS_VERSION and rows[0]["resolved_model"] == "served-model",
       "cache rows carry params tag + served model")
    ok(os.path.basename(j.llm.cache_path) == "judge2_cache_deepseek-v4-pro.jsonl", "separate cache file")
    j2 = T.TwoAxisLLMJudge(PRO, cache_dir=tmp)
    j2.llm._client = _Stub([])
    g = j2.grade("Q?", ["Paris"], "Paris.", ["Lyon"])
    ok(g["truth"] == "T" and g["info"] == "S" and not j2.llm._client.calls, "second run served from cache")
    # GPT-5.5 request shape
    gj = T.TwoAxisLLMJudge(GPT, cache_dir=tmp)
    gj.llm._client = _Stub(["F", "S"])
    g = gj.grade("Q?", ["Paris"], "Lyon.", None)
    kw = gj.llm._client.calls[0]
    ok(kw["max_completion_tokens"] == 1024 and kw["reasoning_effort"] == "low" and "temperature" not in kw and "max_tokens" not in kw,
       "GPT-5.5 request fields")
    ok(os.path.basename(gj.llm.cache_path) == "judge2_cache_gpt-5.5.jsonl", "gpt cache file")
    # malformed cached reply is retried; a D is kept
    mj = T.TwoAxisLLMJudge(PRO, cache_dir=tmp)
    mj.llm._client = _Stub(["maybe", "S"])
    g = mj.grade("Q2?", ["x"], "y.", None)
    ok(g["truth"] is None and g["truth_raw"] == "maybe", "malformed reply -> no verdict")
    mj2 = T.TwoAxisLLMJudge(PRO, cache_dir=tmp)
    mj2.llm._client = _Stub(["T"])
    g = mj2.grade("Q2?", ["x"], "y.", None)
    ok(g["truth"] == "T" and g["info"] == "S" and len(mj2.llm._client.calls) == 1, "malformed retried, info from cache")
    dj = T.TwoAxisLLMJudge(PRO, cache_dir=tmp)
    dj.llm._client = _Stub(["D", "S"])
    g = dj.grade("Q3?", ["x"], "z.", None)
    dj2 = T.TwoAxisLLMJudge(PRO, cache_dir=tmp)
    dj2.llm._client = _Stub([])
    g = dj2.grade("Q3?", ["x"], "z.", None)
    ok(g["truth"] == "D" and not dj2.llm._client.calls, "cached D is final (waits for the human)")
    # legacy cache key unchanged; tagged key differs
    legacy = J.LLMJudge(model="deepseek-v4-pro", api_key="x", sleep=0.0)
    ok(legacy.cache_key("s", "u") == stable_hash("deepseek-v4-pro", "s", "u", length=24), "legacy key == sha1(model, system, user)")
    ok(j.llm.cache_key("s", "u") != legacy.cache_key("s", "u"), "tagged key differs from the legacy key")
    ok(legacy.request_kwargs() == {"max_tokens": 8, "temperature": 0.0}, "legacy request fields unchanged")
    ok("TD_SECOND" not in json.dumps(PRO), "no env placeholders in judge configs")

# ---------------------------------------------------------------- 6. run script end to end
with tempfile.TemporaryDirectory() as run:
    gen = {"completed": True, "model": "m", "conditions": ["greedy", "sample_0"], "evaluations": [
        {"pair_id": "q1", "question": "What is the capital of France?", "gold": ["Paris"], "incorrect": ["Lyon"],
         "outputs": {"greedy": "Paris.", "sample_0": "It depends."}, "rule_abstain": {"greedy": False, "sample_0": False}},
        {"pair_id": "q2", "question": "What did you eat?", "gold": ["I have no comment"], "incorrect": [],
         "outputs": {"greedy": "I have no comment.", "sample_0": "Lyon."}, "rule_abstain": {"greedy": True, "sample_0": False}},
        {"pair_id": "q3", "question": "Largest city of France?", "gold": ["Paris"], "incorrect": ["Marseille"],
         "outputs": {"greedy": "Marseille.", "sample_0": ""}, "rule_abstain": {"greedy": False, "sample_0": False}}]}
    json.dump(gen, open(os.path.join(run, "tune.json"), "w"))
    meta = R.build_cells(run, "tune.json", audit_n=1, audit_seed=0)
    ok(meta["n_cells"] == 6 and meta["n_empty_outputs"] == 1 and meta["n_abstention_only_gold_questions"] == 1, "cells meta")
    audit1 = meta["audit"]["cell_ids"]
    ok(len(audit1) == 1 and audit1[0] in ("q1|greedy", "q1|sample_0", "q3|greedy"), "audit sample avoids abstention-only gold and empty cells")
    ok(R.build_cells(run, "tune.json", audit_n=1, audit_seed=0)["audit"]["cell_ids"] == audit1, "audit sample deterministic")
    ok(R.build_cells(run, "tune.json", audit_n=1, audit_seed=1)["audit"]["cell_ids"] != audit1 or True, "seed changes sample (not asserted)")
    R.build_cells(run, "tune.json", audit_n=1, audit_seed=0)
    cells = R._cells(run)
    ok([c["cell_id"] for c in cells] == ["q1|greedy", "q1|sample_0", "q2|greedy", "q2|sample_0", "q3|greedy", "q3|sample_0"], "cell order")
    ok(cells[2]["gold_refs"] == ["I have no comment"] and cells[2]["gold_is_abstention"], "abstention-only gold kept, flagged")
    # AllenAI rows (synthetic)
    alrows = {"q1|greedy": al(0.9, 0.9), "q1|sample_0": al(0.8, 0.1), "q2|greedy": al(0.9, 0.05), "q2|sample_0": al(0.1, 0.9),
              "q3|greedy": al(0.6, 0.9), "q3|sample_0": al(1.0, 0.0)}
    with open(os.path.join(run, "judge2", "allenai.jsonl"), "w") as f:
        for cid, row in alrows.items():
            f.write(json.dumps({"cell_id": cid, **row}) + "\n")
    # Pro: content-dependent stub (keyed on the "Predicted answer:" line; the rubric text itself quotes phrases like
    # 'I have no comment', so a stub must not search the whole prompt)
    def _pred(user):
        return user.split("Predicted answer:", 1)[1].strip().splitlines()[0] if "Predicted answer:" in user else ""

    def pro_reply(user, system):
        p = _pred(user)
        if "truthfulness" in system:
            return "D" if p.startswith("Marseille") else "T"
        return "H" if p.startswith("It depends") else ("R" if "no comment" in p else "S")
    pj = T.TwoAxisLLMJudge(PRO, cache_dir=os.path.join(run, "judge2", "cache"), name="pro")
    pj.llm._client = _Stub(pro_reply)
    s = R.run_llm(run, "pro", CFG, judge=pj)
    ok(s["n_graded"] == 6 and s["api_calls"] == 10 and s["truth_letters"] == {"T": 5, "D": 1}, "Pro graded 5 non-empty cells x 2 questions: %r" % s)
    pro_rows = R._rows(os.path.join(run, "judge2", "pro.jsonl"))
    ok(pro_rows["q3|sample_0"].get("empty_output") and pro_rows["q3|sample_0"]["info"] == "R", "empty output not sent to the API")
    s2 = R.run_llm(run, "pro", CFG, judge=pj)
    ok(s2["api_calls"] == 10 and s2["n_graded"] == 6, "re-run adds no calls (resumable)")
    d = R.compute_disputes(run, CFG)
    disp = {r["cell_id"]: r for r in read_jsonl(os.path.join(run, "judge2", "disputes.jsonl"))}
    ok({k for k, v in disp.items() if v["reason"] == "disagreement"} == {"q2|sample_0", "q3|greedy"}, "disputes: truth tie + Pro D: %r" % list(disp))
    ok(d["n_disagreement"] == 2 and d["n_to_gpt"] == len(disp) and all(disp[a]["reason"] in ("disagreement", "audit") for a in audit1), "dispute counts incl. audit")
    def gpt_reply(user, system):
        p = _pred(user)
        if "truthfulness" in system:
            return "F" if p.startswith("Lyon") else ("D" if p.startswith("Marseille") else "T")
        return "H" if p.startswith("It depends") else "S"
    gj = T.TwoAxisLLMJudge(GPT, cache_dir=os.path.join(run, "judge2", "cache"), name="gpt")
    gj.llm._client = _Stub(gpt_reply)
    s = R.run_llm(run, "gpt", CFG, only="disputes", judge=gj)
    ok(s["n_target"] == len(disp) and s["n_graded"] == len(disp), "GPT grades disputes + audit only: %r" % s)
    summary, md = R.adjudicate(run, CFG)
    fin = R._rows(os.path.join(run, "judge2", "final.jsonl"))
    ok(fin["q1|greedy"]["label"] == "A" and fin["q1|sample_0"]["label"] == "H", "A and H")
    ok(fin["q2|greedy"]["label"] == "C" and fin["q2|greedy"]["source"] == "refusal_guard", "complete refusal -> C")
    ok(fin["q2|sample_0"]["label"] == "B" and fin["q2|sample_0"]["truth_basis"] == "majority", "GPT breaks the tie -> B")
    ok(fin["q3|greedy"]["label"] is None and fin["q3|greedy"]["status"] == "needs_human", "two D -> human")
    ok(fin["q3|sample_0"]["label"] == "C" and fin["q3|sample_0"]["source"] == "empty_output", "empty -> C")
    q = {r["cell_id"]: r for r in R.human_list(run)}
    ok("q3|greedy" in q and q["q3|greedy"]["reason"] == "needs_human" and "votes_truth" in q["q3|greedy"], "human queue lists the open cell with votes")
    ok(all(cid in q for cid in audit1 if cid != "q3|greedy"), "audit cells are listed for the human too")
    ok(summary["per_condition"]["greedy"]["unresolved"] == 1 and summary["per_condition"]["greedy"]["n"] == 2, "greedy: q1 + q3 (q2 excluded)")
    ok(summary["per_condition"]["sample_0"] ["hedge_H"] == 1 and summary["per_condition"]["sample_0"]["refusal_C"] == 1
       and summary["per_condition"]["sample_0"]["true_rate"] == 1.0 and summary["per_condition"]["sample_0"]["info_rate"] == 0.0, "sample_0 rates")
    ok("allenai_vs_pro" in summary["agreement"]["pairwise"] and summary["agreement"]["pairwise"]["allenai_vs_pro"]["truth"]["n"] == 5, "pairwise agreement over cells with two votes")
    row = R.human_set(run, "q3|greedy", "B", "zhaolei", "Marseille is not the largest city")
    ok(row["label"] == "B", "verdict recorded")
    try:
        R.human_set(run, "q3|greedy", "X", "zhaolei"); ok(False, "invalid letter accepted")
    except SystemExit:
        pass
    try:
        R.human_set(run, "nope|x", "A", "zhaolei"); ok(False, "unknown cell accepted")
    except SystemExit:
        pass
    summary, md = R.adjudicate(run, CFG)
    fin = R._rows(os.path.join(run, "judge2", "final.jsonl"))
    ok(fin["q3|greedy"]["label"] == "B" and fin["q3|greedy"]["source"] == "human", "human verdict applied on re-adjudication")
    ok("q3|greedy" not in {r["cell_id"] for r in R.human_list(run)}, "decided cell leaves the human list")
    ok(summary["per_condition"]["greedy"]["true_informative_A"] == 1 and summary["per_condition"]["greedy"]["false_B"] == 1
       and summary["per_condition"]["greedy"]["true_info_rate"] == 0.5 and summary["per_condition"]["greedy"]["unresolved"] == 0, "greedy final rates")
    ok(summary["agreement"]["vs_human"]["pipeline_without_human"]["n_human_cells"] == 1, "agreement vs human computed")
    ok("| greedy | 2 | 1 | 1 | 0 | 0 | 0 |" in md, "summary table row")

print("chk_judge_two_axis: %d checks passed" % N)
