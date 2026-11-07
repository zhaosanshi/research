"""Shared helpers for the round-19 independent recomputation (stdlib only)."""
import json
import math
from fractions import Fraction
from pathlib import Path

SP = Path('/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/'
          '1202de30-5944-41d9-af49-9348a318988a/scratchpad/r19_recompute')
REPO = Path('/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review')
GPT = REPO / 'from_gpt'
R19 = GPT / 'audit_20261106_round19'
RELEASE = REPO / 'from_claude/results/03_E26同作答率_E27生成适配/E27_release'
V1U = REPO / 'from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u'

# extracted (untrusted) archives, each in its own directory
X_RANDOM = SP / 'random_controls'
X_DOSE = SP / 'gendose_full192'
X_DOSE_PILOT = SP / 'gendose_pilot'
X_PURE = SP / 'pure_das'
X_R18 = SP / 'r18_generation'

FFFD = '�'
LABELS = ('correct', 'wrong', 'abstain')


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def by_pid(doc):
    out = {}
    for e in doc['evaluations']:
        assert e['pair_id'] not in out, ('duplicate pair id', e['pair_id'])
        out[e['pair_id']] = e
    return out


def counts(evals, cond):
    c = {k: 0 for k in LABELS}
    for e in evals:
        c[e['outcome'][cond]] += 1
    return c


def exact_sign_p(plus, minus):
    """Two-sided exact binomial (sign) test on discordant pairs, p0=1/2."""
    n = plus + minus
    if n == 0:
        return 1.0
    k = min(plus, minus)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def fixed_broken(evals_a, evals_b, cond_a, cond_b):
    """a = reference (before), b = after; items matched by pair id."""
    a, b = by_pid_list(evals_a), by_pid_list(evals_b)
    assert set(a) == set(b)
    fixed = sorted(p for p in a if a[p]['outcome'][cond_a] != 'correct' and b[p]['outcome'][cond_b] == 'correct')
    broken = sorted(p for p in a if a[p]['outcome'][cond_a] == 'correct' and b[p]['outcome'][cond_b] != 'correct')
    return fixed, broken


def by_pid_list(evals):
    if isinstance(evals, dict):
        return evals
    return {e['pair_id']: e for e in evals}


def transitions(evals, cond_a, cond_b, evals_b=None):
    a = by_pid_list(evals)
    b = by_pid_list(evals_b if evals_b is not None else evals)
    t = {}
    for p in a:
        key = a[p]['outcome'][cond_a] + '->' + b[p]['outcome'][cond_b]
        t[key] = t.get(key, 0) + 1
    return dict(sorted(t.items()))


def pct(x):
    return round(100.0 * x, 4)


def _bytes_to_unicode_inverse():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_INV = _bytes_to_unicode_inverse()


def tok_bytes(s):
    """GPT-2 ByteLevel token string -> raw bytes (None for special tokens)."""
    if s is None or (s.startswith('<|') and s.endswith('|>')):
        return None
    return bytes(_INV[ch] for ch in s)


def utf8_prefix_ok(data: bytes) -> bool:
    """True if data is a valid UTF-8 stream possibly ending in an incomplete (pending) sequence."""
    import codecs
    try:
        codecs.getincrementaldecoder('utf-8')('strict').decode(data, final=False)
        return True
    except UnicodeDecodeError:
        return False
