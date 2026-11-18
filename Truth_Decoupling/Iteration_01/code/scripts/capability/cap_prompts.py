"""Prompts and rule scoring for the capability check (work queue F0 / F1): MMLU and GSM8K.

User-turn texts only; the system prompt is the same as the free-form pipeline's plain prompt (set by the caller).
  mmlu_user(subject, question, choices) -> str
  gsm8k_user(question) -> str
  mmlu_pick(response) -> "A".."D" or None
  gsm8k_pick(response) -> float or None
  gsm8k_gold(answer_field) -> float   (the number after "####")
A None pick counts as "no answer" (abstained or unparseable); the report keeps it as its own column.

Usage: python3 -I cap_prompts.py --selftest
"""
import re
import sys

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


_START = re.compile(r"^\s*(?:\*\*)?\(?([ABCD])\)?(?:\*\*)?(?=[\s.:)\]]|$)")
_SAYS = re.compile(r"\banswer\s*(?:is|:)?\s*(?:option\s*)?(?:\*\*)?\(?([ABCD])\)?(?:\*\*)?(?=[\s.:)\]]|$)", re.I)
_ANY = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def mmlu_pick(response):
    """First letter at the very start; else 'answer is X'; else the single distinct standalone letter; else None."""
    m = _START.search(response)
    if m and not (m.group(1) == "A" and re.match(r"\s+[a-z]", response[m.end():])):  # "A common view ..." is an article
        return m.group(1)
    m = _SAYS.search(response)
    if m:
        return m.group(1).upper()
    found = set(_ANY.findall(response))
    return found.pop() if len(found) == 1 else None


_NUM = r"-?\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?"
_FINAL = re.compile(r"the\s+(?:final\s+)?answer\s+is\s*[:=]?\s*(?:\*\*)?\s*(" + _NUM + ")", re.I)
_ANYNUM = re.compile(_NUM)


def _to_float(s):
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_pick(response):
    """Number after the last 'The answer is'; else the last number in the response; else None."""
    hits = _FINAL.findall(response)
    if hits:
        return _to_float(hits[-1])
    nums = _ANYNUM.findall(response)
    return _to_float(nums[-1]) if nums else None


def gsm8k_gold(answer_field):
    if "####" not in answer_field:
        raise ValueError("GSM8K answer without ####")
    v = _to_float(answer_field.rsplit("####", 1)[1].strip())
    if v is None:
        raise ValueError("unparseable GSM8K gold")
    return v


def selftest():
    u = mmlu_user("high_school_physics", "Q?", ["a", "b", "c", "d"])
    assert u.startswith("The following is a multiple choice question about high school physics.") and "\nD. d\n" in u
    cases = {"B": "B", "B.": "B", "(C) because": "C", "**D**": "D", "The answer is C.": "C", "Answer: (a)": "A",
             "A common view is that the answer is B.": "B", "I think it's D": "D", "Either A or B": None,
             "I don't know.": None, "": None, "Option B is right": "B"}
    for r, want in cases.items():
        got = mmlu_pick(r)
        assert got == want, (r, got, want)
    assert gsm8k_gold("... #### 1,234") == 1234.0 and gsm8k_gold("x #### -3") == -3.0
    g = {"So 5 + 7 = 12. The answer is 12.": 12.0, "The answer is $1,250.": 1250.0, "The final answer is: **18**": 18.0,
         "First 3, then 4, so 7 apples.": 7.0, "The answer is 5. Wait, the answer is 6.": 6.0, "I am not sure.": None,
         "The answer is -2": -2.0}
    for r, want in g.items():
        got = gsm8k_pick(r)
        assert got == want, (r, got, want)
    print("selftest PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
