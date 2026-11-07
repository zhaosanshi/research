"""QA outcome policies; legacy semantics plus an optional complete-refusal guard.

A response can contain an uncertainty phrase and still attempt an answer.
Legacy correctness/abstention follows the semantic judge, including unresolved.
The separately versioned future guard handles only complete personal-refusal
phrases, with conservative literal-language bypasses; never substring matching.
"""
import re
from copy import deepcopy

OUTCOME_POLICY = "judge_primary_v2"
FULL_PERSONAL_REFUSAL_POLICY = "judge_primary_v2_full_personal_refusal_guard_v1"
VALID_OUTCOMES = frozenset(("correct", "wrong", "abstain", "unresolved"))

# Keep every character except normal whitespace/case and the specific apostrophe
# variant shown here. Never strip arbitrary punctuation, numbers or Unicode:
# doing so would turn "I don't know 42/北京" into a false complete-refusal match.
_FULL_PERSONAL_REFUSAL = re.compile(
    r"(?:i\s+(?:don['’]t|do\s+not)\s+know"
    r"(?:\s+(?:for\s+sure|the\s+answer(?:\s+for\s+sure)?))?"
    r"|i\s+have\s+no\s+comment)[.!?…]*", re.IGNORECASE)
_METALANGUAGE_REQUEST = re.compile(
    r"\b(?:translate|translation|translating|transliterate|transliteration)\b"
    r"|\b(?:meaning|definition)\s+of\b"
    r"|\b(?:what|which)\s+(?:(?:is|are)\s+(?:the\s+)?)?(?:phrase|expression|words?|sentence|title|quotation|quote)\b"
    r"|\b(?:what|which)(?:['’]s|\s+is|\s+was|\s+are)?\s+(?:the\s+)?(?:name|title|song|track|book|film|movie|album)\b"
    r"|\b(?:name|named|naming)\b.*\b(?:song|track|book|film|movie|album|work)\b"
    r"|\b(?:song|track|book|film|movie|album|work)\b.*\b(?:called|named|titled)\b"
    r"|\b(?:how\s+(?:do|would|should)\s+(?:you|we|i|one)\s+say)\b"
    r"|\b(?:repeat|quote|write|print|output|respond\s+with|reply\s+with)\b"
    r"|翻译|译成|译为|复述|逐字|原话|标题|书名|歌名|曲名|片名|引号|(?:歌|曲|书|电影|影片|专辑).{0,5}(?:名字|名称|叫什么)"
    r"|(?:英语|英文|中文|汉语).{0,6}(?:怎么说|表达)", re.IGNORECASE)


def full_personal_refusal_decision(question, prediction):
    """High-precision, reference-free guard; conservative literal-answer bypass.

    A complete quoted/backticked phrase, a mixed substantive answer, a number or
    Unicode text cannot match. Explicit translation/quotation/title requests may
    validly have a refusal phrase as their answer, so defer them to semantics.
    This is deliberately not a general abstention detector.
    """
    if not isinstance(prediction, str):
        return {'matched_complete_phrase': False, 'apply': False, 'bypass_reason': None}
    text = prediction.strip()
    matched = bool(_FULL_PERSONAL_REFUSAL.fullmatch(text))
    if not matched:
        return {'matched_complete_phrase': False, 'apply': False, 'bypass_reason': None}
    if not isinstance(question, str) or not question.strip():
        return {'matched_complete_phrase': True, 'apply': False, 'bypass_reason': 'missing_question_context'}
    if _METALANGUAGE_REQUEST.search(question):
        return {'matched_complete_phrase': True, 'apply': False, 'bypass_reason': 'explicit_literal_language_request'}
    return {'matched_complete_phrase': True, 'apply': True, 'bypass_reason': None}


def apply_full_personal_refusal_policy(grade, question, prediction):
    """Derive a new result without editing the raw semantic response or input."""
    original = deepcopy(grade)
    semantic_label = original.get('semantic_label_before_policy', original.get('label'))
    judged_outcome(semantic_label)
    decision = full_personal_refusal_decision(question, prediction)
    original.update(outcome_policy=FULL_PERSONAL_REFUSAL_POLICY,
                    semantic_label_before_policy=semantic_label,
                    label='abstain' if decision['apply'] else semantic_label,
                    decision_source='full_personal_refusal_guard' if decision['apply'] else 'semantic_grader',
                    personal_refusal_guard=decision)
    return original


def apply_configured_outcome_policy(grade, question, prediction, config):
    """The pipeline config, not a grader implementation detail, owns policy."""
    enabled = config.get('full_personal_refusal_guard', False)
    if type(enabled) is not bool:
        raise ValueError('full_personal_refusal_guard must be boolean')
    expected = FULL_PERSONAL_REFUSAL_POLICY if enabled else OUTCOME_POLICY
    if config.get('outcome_policy', expected) != expected:
        raise ValueError('configured outcome policy does not match the refusal guard')
    if enabled:
        return apply_full_personal_refusal_policy(grade, question, prediction)
    if grade.get('outcome_policy', OUTCOME_POLICY) != OUTCOME_POLICY:
        raise ValueError('cannot silently reuse policy-corrected grades under legacy configuration')
    return grade


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
