"""Read-only. From the Truth_Decoupling root, export --tqa on the scoring host, --qa2 in the
scoring container; then --compare TQA.json QA2.json. Each export returns only item IDs and
whether the forced E answer repeats the no-write answer after the N1 pipeline's normalization.
Pass --base TQA-cells.json QA2-cells.json (exports of 2026-11-22_blank_credit_replay_c.py).
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path("Iteration_01/diag").resolve()))
MODELS = ('llama3.1-8b', 'qwen3.5-9b', 'qwen3.8-27b', 'gemma3-12b', 'gemma3-27b')


def read_tqa():
    import tq_same_n_decomp as td
    data = {}
    for model in MODELS:
        display, layer, strength = td.A.KEY[model]
        split, err = td.MP.prompt_split(display, layer, 1, strength)
        if split is None:
            raise ValueError(err)
        sp, tx, _, ids = split
        _, run, _ = td.MP.usual(display, 1)
        raw = td.MP.fr.raw_texts(td.MP.work(run, 'test', f'pospromptdl{layer}'))
        cond = f'E_L{layer}_conf-w-p{strength}'
        data[model] = {q: [bool(sp.blank[q, 'baseline_x']),
                           td.MP.ro.norm(tx.ec[q, 'baseline_x']) == td.MP.ro.norm(raw[q, cond])]
                       for q in ids}
    return data


def read_qa2():
    import qaqa_pilot_table as Q
    root = Path(Q.S.NS)
    skip = set(json.loads((root/'splits/qaqa_pilot_20261122.json').read_text())['ids'])
    data = {}
    for model in MODELS:
        raw = Q.Raw(model, 'qaqafull', str(root/'bank/qaqa_full_bank_20261122.jsonl'),
                    ('test_full570', 'test_full570_tuned', 'test_full570_strong'))
        f = 'conf:' + raw.e(Q.SET[model][1]).split(':')[1]
        raw.need('x')
        raw.need(f)
        data[model] = {q: [raw.blank('x', q), Q.ro.norm(raw.text('x', q)) == Q.ro.norm(raw.text(f, q))]
                       for q in raw.ids if q not in skip}
    return data


def evaluate(rows, same, ids, stratify, common):
    # columns of base cells: E net / answered, N1 net / answered, forced E net / answered / confidence, origin, type.
    blanks = [q for q in ids if not rows[q][1]]
    if any(bool(rows[q][7]) != bool(same[q][0]) for q in ids):
        raise ValueError('raw text and base cells disagree on original abstention')
    released = {q for q in blanks if rows[q][3]}
    # Same veto as N1: own blanks pass; on E-induced blanks, discard replies repeating the original answer.
    candidates = [q for q in blanks if rows[q][5] and rows[q][6] is not None and
                  (rows[q][7] or not same[q][1])]
    ranked = sorted(candidates, key=lambda q: -rows[q][6])
    if common:
        released.intersection_update(ranked)
    total_released = len(released)
    take = set()
    deficits = []
    for origin in ([True, False] if stratify else [None]):
        eligible = [q for q in ranked if origin is None or rows[q][7] == origin]
        quota = sum(origin is None or rows[q][7] == origin for q in released)
        if len(eligible) < quota:
            deficits.append(('own' if origin is True else 'induced' if origin is False else 'any', quota, len(eligible)))
        take.update(eligible[:quota])
    if deficits:
        return deficits, total_released, None
    if len(take) != total_released:
        raise ValueError('different actual answer counts')
    delta = {q: (rows[q][2] if q in released else rows[q][0]) -
                 (rows[q][4] if q in take else rows[q][0]) for q in ids}
    return [], total_released, delta


def minimal_trimming(bank, matches, ids, models, stratify, weights):
    """Only when the comparator is short, omit exactly the deficit from N1 in that stratum.
    Bound the score over *every* possible deletion; gold scores never choose the reported system."""
    import numpy as np
    n = len(ids)
    fixed = np.zeros(n, int)
    deficits = []
    for m in models:
        rows, same = bank[m], matches[m]
        blanks = [i for i, q in enumerate(ids) if not rows[q][1]]
        for own in ([True, False] if stratify else [None]):
            sub = [i for i in blanks if own is None or rows[ids[i]][7] == own]
            releases = [i for i in sub if rows[ids[i]][3]]
            eligible = [i for i in sub if rows[ids[i]][5] and rows[ids[i]][6] is not None
                        and (rows[ids[i]][7] or not same[ids[i]][1])]
            eligible.sort(key=lambda i: -rows[ids[i]][6])
            for i in releases:
                fixed[i] += rows[ids[i]][2] - rows[ids[i]][0]
            for i in eligible[:len(releases)]:
                fixed[i] -= rows[ids[i]][4] - rows[ids[i]][0]
            missing = len(releases) - len(eligible)
            if missing > 0:
                gains = np.array([rows[ids[i]][2] - rows[ids[i]][0] for i in releases])
                deficits.append((missing, releases, gains))
    # One row for the observed bank, then all bootstrap draws. Inside each draw take the worst
    # and best legal deletions. The interval encompasses the possible fixed-trim percentile intervals.
    w = np.vstack((np.ones(n, dtype=int), weights))
    low = w @ fixed
    high = low.copy()
    for missing, releases, gains in deficits:
        changes = np.sort(w[:, releases] * gains, axis=1)
        low -= changes[:, -missing:].sum(axis=1)
        high -= changes[:, :missing].sum(axis=1)
    return int(low[0]), int(high[0]), int(np.sort(low[1:])[100]), int(np.sort(high[1:])[3899])


def compare(base, matches):
    import numpy as np
    weights = {}
    for label, bank, same in zip(('TQA', 'QA2'), base, matches):
        for count in (3, 5):
            models = MODELS[:count]
            roster = set.intersection(*(set(bank[m]) for m in models))
            ids = sorted(roster) if label == 'TQA' else [q for q in bank[models[0]] if q in roster]
            n = len(ids)
            if n not in weights:
                rng = random.Random(20261116)
                weights[n] = np.array([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n)
                                       for _ in range(4000)])
            print(label, count, n, 'questions')
            for common in (False, True):
                for stratify in (False, True):
                    effects = [evaluate(bank[m], same[m], ids, stratify, common) for m in models]
                    bad = {m: result[0] for m, result in zip(models, effects) if result[0]}
                    print(' same origin', stratify, ' restrict N1 to common eligible questions', common,
                          ' N1 releases each', [result[1] for result in effects], end=' ')
                    if bad:
                        print('unreachable target', bad, 'other available model deltas',
                              [None if d is None else sum(d.values()) for _, _, d in effects])
                        low, high, left, right = minimal_trimming(bank, same, ids, models, stratify, weights[n])
                        print('  minimally reduce N1 in short strata only: all possible observed net differences',
                              [low, high], 'outer 95% bootstrap envelope', [left, right])
                        continue
                    qsum = np.array([sum(d[q] for _, _, d in effects) for q in ids])
                    draws = np.sort(weights[n] @ qsum)
                    print('net advantage', int(qsum.sum()), [int(draws[100]), int(draws[3899])],
                          'per model', [sum(d.values()) for _, _, d in effects])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--tqa', action='store_true')
    group.add_argument('--qa2', action='store_true')
    group.add_argument('--compare', metavar=('TQA_MATCH', 'QA2_MATCH'), nargs=2)
    p.add_argument('--base', metavar=('TQA_CELLS', 'QA2_CELLS'), nargs=2)
    args = p.parse_args()
    if args.compare:
        if not args.base:
            p.error('--compare needs --base')
        read = lambda ps: [json.loads(Path(name).read_text()) for name in ps]
        compare(read(args.base), read(args.compare))
    else:
        print(json.dumps(read_tqa() if args.tqa else read_qa2(), separators=(',', ':')))
