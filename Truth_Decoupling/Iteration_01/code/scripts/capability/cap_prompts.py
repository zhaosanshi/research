"""Prompts and rule scoring for the capability check (work queue F0 / F1): MMLU and GSM8K.

User-turn texts only; the system prompt is set by the caller (MMLU: the plain prompt x; GSM8K: "neutral").
  mmlu_user(subject, question, choices) -> str
  gsm8k_user(question) -> str
  mmlu_parse(response) -> (letter or None, reason)
  gsm8k_parse(response, finished) -> (number or None, reason)
  gsm8k_gold(answer_field) -> float   (the number after "####")
Accepted: one single, unambiguous final option / number. Reasons for None are kept apart (GPT review 2026-11-19,
discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md section 2, last row):
  MMLU  "negated" (e.g. "The answer is not D."), "several" (e.g. "B or C"), "none" (no option letter)
  GSM8K "truncated" (the generation hit the token cap and gives no explicit final answer), "none" (no number)
A None pick stays in the accuracy denominator; the report counts the reasons, refusals and truncation separately.

Usage: python3 -I cap_prompts.py --selftest
"""
import re
import sys
from fractions import Fraction

LETTERS = "ABCD"


def mmlu_user(subject, question, choices):
    if len(choices) != 4:
        raise ValueError("MMLU item must have 4 choices")
    lines = ["The following is a multiple choice question about %s." % subject.replace("_", " "), "", question.strip()]
    lines += ["%s. %s" % (LETTERS[i], str(c).strip()) for i, c in enumerate(choices)]
    lines += ["", "Answer with the letter of the correct option only."]
    return "\n".join(lines)


def gsm8k_user(question):
    return question.strip() + "\n\nSolve the problem step by step. At the end, write \"The answer is\" followed by the final number."


# a letter standing alone (not part of a word); the article "A" before a lower-case word is not an option
_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
_NEG = re.compile(r"\b(?:not|isn't|isnt|never|cannot|can't|nor)\s+(?:be\s+)?(?:option\s+|choice\s+|answer\s+)?\(?\*{0,2}[ABCD]\b", re.I)
_LEAD = re.compile(r"^\s*(?:\*\*)?\(?([ABCD])\)?(?:\*\*)?(?=\s*$|[.:)\]]|\s+[^a-z\s])")
_SAYS = re.compile(r"\banswer\s*(?:is|:)\s*(?:option\s*)?(?:\*\*)?\(?([ABCD])\)?(?:\*\*)?(?![A-Za-z])", re.I)


def _option_letters(text):
    out = []
    for m in _LETTER.finditer(text):
        # "A" opening a sentence before a lower-case word is the article ("A common view ..."); mid-sentence it is an option
        opens = re.search(r"(?:^|[.!?:]\s+|\n\s*)$", text[:m.start()]) is not None
        if m.group(1) == "A" and opens and re.match(r"\s+[a-z]", text[m.end():]):
            continue
        out.append(m.group(1))
    return out


def mmlu_parse(response):
    """One single, unambiguous final option. Order: a stated final answer ("the answer is X"), then a leading option
    label ("B.", "(C)", "**D**"), then the only option letter mentioned. Refused: a stated answer followed by another
    option ("B or C"), a negated option without a stated final answer ("The answer is not D."), several letters."""
    t = response.strip()
    negated = {m.group(0)[-1].upper() for m in _NEG.finditer(t)}
    says = list(_SAYS.finditer(t))
    if says:
        last = says[-1]
        x = last.group(1).upper()
        if any(c != x for c in _option_letters(t[last.end():])):
            return None, "several"
        return (None, "negated") if x in negated else (x, "stated")
    lead = _LEAD.search(t)
    if lead:
        return (None, "negated") if negated else (lead.group(1), "leading")
    if negated:
        return None, "negated"
    letters = set(_option_letters(t))
    if len(letters) == 1:
        return letters.pop(), "single"
    return (None, "several") if letters else (None, "none")


def mmlu_pick(response):
    return mmlu_parse(response)[0]


_NUM = r"-?\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*/\s*[0-9]+)?"
_FINAL = re.compile(r"the\s+(?:final\s+)?answer\s+is\s*[:=]?\s*(?:\*\*)?\s*\$?\s*(" + _NUM + ")", re.I)
_ANYNUM = re.compile(_NUM)


def _to_number(s):
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        if "/" in s:
            a, b = s.split("/")
            if float(b) == 0:
                return None
            return float(Fraction(a) / Fraction(b))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_parse(response, finished=True):
    """Number after the last 'The answer is'; else, for a finished answer only, the last number; a truncated answer
    without an explicit final answer is 'truncated' (no pick)."""
    hits = _FINAL.findall(response)
    if hits:
        v = _to_number(hits[-1])
        return (v, "final") if v is not None else (None, "none")
    if not finished:
        return None, "truncated"
    nums = _ANYNUM.findall(response)
    if not nums:
        return None, "none"
    v = _to_number(nums[-1])
    return (v, "last_number") if v is not None else (None, "none")


def gsm8k_pick(response, finished=True):
    return gsm8k_parse(response, finished)[0]


def gsm8k_gold(answer_field):
    if "####" not in answer_field:
        raise ValueError("GSM8K answer without ####")
    v = _to_number(answer_field.rsplit("####", 1)[1].strip())
    if v is None:
        raise ValueError("unparseable GSM8K gold")
    return v


def selftest():
    u = mmlu_user("high_school_physics", "Q?", ["a", "b", "c", "d"])
    assert u.startswith("The following is a multiple choice question about high school physics.") and "\nD. d\n" in u
    cases = {"B": "B", "B.": "B", "(C) because": "C", "**D**": "D", "The answer is C.": "C", "Answer: (a)": "A",
             "A common view is that the answer is B.": "B", "I think it's D": "D", "Either A or B": None,
             "I don't know.": None, "": None, "Option B is right": "B",
             # GPT review counter-examples
             "The answer is not D.": None, "The answer is B or C.": None, "It is not B, so the answer is C.": "C",
             "B. Paris": "B", "C) 42": "C", "A": "A", "A.": "A"}
    for r, want in cases.items():
        got = mmlu_pick(r)
        assert got == want, (r, got, want)
    assert mmlu_parse("The answer is not D.")[1] == "negated" and mmlu_parse("The answer is B or C.")[1] == "several"
    assert gsm8k_gold("... #### 1,234") == 1234.0 and gsm8k_gold("x #### -3") == -3.0
    g = {("So 5 + 7 = 12. The answer is 12.", True): 12.0, ("The answer is $1,250.", True): 1250.0,
         ("The final answer is: **18**", True): 18.0, ("First 3, then 4, so 7 apples.", True): 7.0,
         ("The answer is 5. Wait, the answer is 6.", True): 6.0, ("I am not sure.", True): None, ("The answer is -2", True): -2.0,
         # GPT review counter-examples
         ("We used 7 boxes and next we must", False): None, ("The answer is 1/2.", True): 0.5,
         ("The answer is 7.", False): 7.0}
    for (r, fin), want in g.items():
        got = gsm8k_pick(r, fin)
        assert got == want, (r, fin, got, want)
    assert gsm8k_parse("We used 7 boxes and next we must", False)[1] == "truncated"
    print("selftest PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
