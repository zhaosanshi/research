#!/usr/bin/env python3
"""Offline checks for the judge's abstention -> human review path (no network, stub client)."""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core import judge as J  # noqa: E402
from core.io_utils import read_jsonl  # noqa: E402
from scripts.review_checks.judge_human_review import pending  # noqa: E402

N = 0


def ok(cond, msg):
    global N
    N += 1
    if not cond:
        raise SystemExit("FAIL %d: %s" % (N, msg))


class _Stub:
    """Mimics openai's client: replies are scripted per call; `.model` is the resolved model."""
    def __init__(self, replies):
        self.replies = list(replies); self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        content = self.replies.pop(0)
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(model="deepseek-v4-pro-0813", choices=[types.SimpleNamespace(message=msg)])


def make_judge(tmp, replies, guard=False):
    j = J.LLMJudge(model="deepseek-v4-pro", api_key="x", cache_path=os.path.join(tmp, J.cache_filename("deepseek-v4-pro")),
                   sleep=0.0, full_personal_refusal_guard=guard)
    j._client = _Stub(replies)
    return j


with tempfile.TemporaryDirectory() as tmp:
    q, gold, inc = "What is the capital of France?", ["Paris"], ["Lyon"]
    # 1. an ordinary A: label, cache row carries resolved model + timestamp
    j = make_judge(tmp, ["A"])
    g = j.qa_grade(q, gold, "It is Paris.", inc)
    ok(g["label"] == "correct" and "judge_abstained" not in g, "A -> correct")
    rows = list(read_jsonl(j.cache_path))
    ok(rows[-1]["resolved_model"] == "deepseek-v4-pro-0813" and rows[-1]["ts"], "cache row records resolved model and ts")
    ok(not os.path.exists(j.review_queue_path), "an A is not queued")
    # 2. a D: unresolved, flagged, queued exactly once; a re-grade hits the cache and does not re-queue
    j = make_judge(tmp, ["D"])
    g = j.qa_grade(q, gold, "The capital moved in 1999.", inc)
    ok(g["label"] == "unresolved" and g.get("judge_abstained") is True, "D -> unresolved + judge_abstained")
    ok(j.n_judge_abstained == 1, "abstention counter")
    queue = list(read_jsonl(j.review_queue_path))
    ok(len(queue) == 1 and queue[0]["judge_raw"] == "D" and queue[0]["gold"] == ["Paris"], "queued once with the inputs")
    g2 = j.qa_grade(q, gold, "The capital moved in 1999.", inc)
    ok(g2["label"] == "unresolved" and j._client.calls == 1, "cached D is final: no second API call")
    ok(len(list(read_jsonl(j.review_queue_path))) == 1, "not queued twice")
    ok(len(pending(tmp)) == 1, "pending() lists it")
    # 3. a malformed non-empty reply is cached and queued; a transient <unresolved> is neither
    j = make_judge(tmp, ["I think it is fine."])
    g = j.qa_grade(q, gold, "Marseille.", inc)
    ok(g["label"] == "unresolved" and "judge_abstained" not in g, "malformed -> unresolved, not an abstention")
    ok(len(pending(tmp)) == 2, "malformed reply is queued")
    j = make_judge(tmp, ["", "", "", "", ""])
    g = j.qa_grade(q, gold, "Nice.", inc)
    ok(g["raw"] == "<unresolved>" and len(pending(tmp)) == 2, "transient failure is not queued")
    # 4. 赵磊's verdict supersedes the D on reload; pending() drops it; reader regex sees a plain letter
    key = queue[0]["key"]
    row = J.record_human_verdict(j.cache_path, key, "deepseek-v4-pro", "B", "zhaolei", "wrong year")
    ok(row["answer"] == "B (human: zhaolei)" and row["source"] == "human", "human row shape")
    j = make_judge(tmp, [])
    g = j.qa_grade(q, gold, "The capital moved in 1999.", inc)
    ok(g["label"] == "wrong" and g.get("human_review") is True and j._client.calls == 0, "human verdict read from cache, no API call")
    ok(len(pending(tmp)) == 1, "pending() drops the decided item")
    import re
    ok(re.match(r"\s*\**\s*([ABC])\b", row["answer"]).group(1) == "B", "control-cache regex reads the human row as B")
    # 5. guard path untouched: a pure refusal never reaches the judge, so it is never queued
    j = make_judge(tmp, ["A"], guard=True)
    g = j.qa_grade(q, gold, "I don't know.", inc)
    ok(g["label"] == "abstain" and g["decision_source"] == "full_personal_refusal_guard" and j._client.calls == 0, "guard still decides pure refusals")
    ok(len(pending(tmp)) == 1, "guard path does not queue")
    # 6. letters A-D parse, anything else does not
    for raw, lab in (("A", "correct"), ("**B**", "wrong"), (" C.", "abstain"), ("D", "unresolved")):
        m = J._LETTER.match(raw); ok(m and J._LABELS[m.group(1)] == lab, "parse %r" % raw)
    ok(J._LETTER.match("E") is None and J._LETTER.match("Answer: A") is None, "non-letters rejected")
    # 7. the rubric offers D and names the human reviewer; the version changed with it
    ok("D: CANNOT_DETERMINE" in J.QA_GRADER_TEMPLATE and "A, B, C, or D" in J.QA_GRADER_TEMPLATE, "rubric offers D")
    ok(J.QA_GRADER_VERSION != "qa_semantic_v3", "rubric version bumped")
    # 8. a verdict other than A/B/C, or without an author, is refused
    for bad in (("D", "zhaolei"), ("A", "")):
        try:
            J.record_human_verdict(j.cache_path, key, "deepseek-v4-pro", bad[0], bad[1]); ok(False, "accepted %r" % (bad,))
        except ValueError:
            ok(True, "refused %r" % (bad,))

print("OK: %d checks passed (judge abstention -> human review)" % N)
