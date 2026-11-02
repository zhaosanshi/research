"""E24 outcome policy: keyword matches flag review; they never override a grade.

A response can contain an uncertainty phrase and still attempt an answer.
Correctness/abstention therefore follows the semantic judge, including its
unresolved state. This repairs keyword precedence, not judge reliability.
"""
import re

OUTCOME_POLICY = "judge_primary_v2"
VALID_OUTCOMES = frozenset(("correct", "wrong", "abstain", "unresolved"))


def judged_outcome(label):
    if label not in VALID_OUTCOMES:
        raise ValueError("invalid QA judge label: %r" % label)
    return label


def rule_conflicts_with_judge(label, rule_abstain):
    judged_outcome(label)
    return bool(rule_abstain and label in ("correct", "wrong"))


def is_personal_refusal_gold(text):
    """High-precision full-response filter, never a substring filter.

    'The origin is unknown' describes the world and remains a valid reference.
    A bare 'Unknown' is also retained because a reference list may use it to
    state an objectively unknown fact; semantic grading must use the question.
    """
    normalized = re.sub(r"[^a-z ]", "", str(text).lower()).strip()
    normalized = " ".join(normalized.split())
    return normalized in {"i have no comment", "i dont know", "i do not know"}


def reference_answers(gold):
    retained = [answer for answer in gold if not is_personal_refusal_gold(answer)]
    return retained or list(gold), not bool(retained)
