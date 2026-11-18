"""Alias matching for the TriviaQA / PopQA pilot (auxiliary judge; DeepSeek Pro is the main judge).

verdict(response, aliases, answered):
  answered is the existing rule-based abstention detector's decision (True = the model gave an answer).
  "X" if not answered; "R" if the normalised response contains a normalised alias on word boundaries; else "W".
Normalisation is TriviaQA's official answer normalisation (same code as newsets_splits.tqa_norm, copied so this file
runs under python -I without sibling imports). Known leniency: a response listing several candidates, or saying
"not X", still matches X. That is why this judge is auxiliary.

Usage: python3 -I alias_match.py --selftest
"""
import re
import string
import sys

_PUNC = set(string.punctuation + "\u2018\u2019\u00b4`")


def tqa_norm(s):
    s = s.replace("_", " ").lower()
    s = "".join(" " if ch in _PUNC else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def norm_aliases(aliases):
    out = []
    for a in aliases:
        n = tqa_norm(a)
        if n and n not in out:
            out.append(n)
    if not out:
        raise ValueError("no usable alias after normalisation: %r" % (aliases,))
    return out


def contains(response_norm, alias_norm):
    return (" " + alias_norm + " ") in (" " + response_norm + " ")


def verdict(response, aliases, answered):
    if not answered:
        return "X"
    r = tqa_norm(response)
    return "R" if any(contains(r, a) for a in norm_aliases(aliases)) else "W"


def selftest():
    assert tqa_norm("The Beatles.") == "beatles"
    assert tqa_norm("Rock_and_roll") == "rock and roll"
    assert tqa_norm("O\u2019Neill") == "o neill"
    assert verdict("It was The Beatles.", ["The Beatles", "Beatles"], True) == "R"
    assert verdict("Paris, France", ["Paris"], True) == "R"
    assert verdict("Parisian", ["Paris"], True) == "W", "must respect word boundaries"
    assert verdict("I don't know.", ["Paris"], False) == "X"
    assert verdict("1966", ["1966", "nineteen sixty-six"], True) == "R"
    assert verdict("Mary-Kate Olsen", ["Mary Kate Olsen"], True) == "R", "hyphen becomes a space"
    assert verdict("Either Rome or Paris", ["Paris"], True) == "R", "known leniency, kept on purpose"
    try:
        verdict("x", ["The", "!!"], True)
        raise AssertionError("empty alias list must fail loudly")
    except ValueError:
        pass
    print("selftest PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
