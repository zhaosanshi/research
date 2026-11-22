"""Read-only audit of N1 against two-pass E on the displayed common cohort.
Run --tqa on scoring host and --qa2 in scoring container, at Truth_Decoupling root;
redirect each output to a private JSON file. Run --compare TQA.json QA2.json on CPU
with numpy. No model or judge invocation. Only the displayed summaries are public.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import sys

MODELS = ('llama3.1-8b', 'qwen3.5-9b', 'qwen3.8-27b', 'gemma3-12b', 'gemma3-27b')
VALUE = {'R': 1, 'W': -1, 'X': 0, 'H': 0}
sys.path.insert(0, str(Path('Iteration_01/diag').resolve()))


def read_tqa():
    import tq_same_n_decomp as td
    out = {}
    for model in MODELS:
        display, layer, s = td.A.KEY[model]
        train, test = [td.MP.prompt_split(display, layer, i, s)[0] for i in (0, 1)]
        src, = [r for r in td.ML.sources(display)[2] if r[0] == layer]
        splits = []
        for i, (split, old) in enumerate(((train, src[1]), (test, src[3]))):
            keep = set(old.qs)
            ids = [q for q in split[3] if q in keep]
            if ids != split[3]:
                split = td.MP.prompt_split(display, layer, i, s, qs=ids)[0]
            if split is None or split[3] != ids or len(set(ids)) != len(ids):
                raise ValueError((model, i, 'roster'))
            splits.append(split)
        train, test = splits
        sp, tx, _, ids = test
        cfg = td.ML.n1_pick(train[0], train[1], s)
        es, ea = td.S.system(sp, tx, ('E', s))[:2]
        ns, na = td.S.system(sp, tx, cfg)[:2]
        _, run, _ = td.MP.usual(display, 1)
        meta = json.loads((Path(run)/'eval_extra'/f'confidence_pospromptdl{layer}_test.json').read_text())
        source = meta['targets']['conf_E' + td.MP.DM.show(s)]
        raw = Path(run)/'eval_extra'/'gen'/source['source']['file']
        if hashlib.sha256(raw.read_bytes()).hexdigest() != source['source']['file_sha256']:
            raise ValueError((model, 'forced confidence generation sha256'))
        raw_ids = [r['pair_id'] for r in source['rows']]
        actual = json.loads(raw.read_text())
        if (raw_ids != [r['pair_id'] for r in actual['rows']] or len(raw_ids) != len(set(raw_ids))
                or actual['positions'] != 'prompt' or not actual['completed'] or actual['split'] != 'test'):
            raise ValueError((model, 'confidence or generation identity'))
        forced = td.forced(run, layer, s, ids)
        out[model] = {q: [VALUE[es[i]], int(ea[i]), VALUE[ns[i]], int(na[i]), *forced[q],
                          bool(sp.blank[q, 'baseline_x']), None] for i, q in enumerate(ids)}
    return out


def read_qa2():
    import qaqa_pilot_table as Q
    root = Path(Q.S.NS)
    skip = set(json.loads((root/'splits/qaqa_pilot_20261122.json').read_text())['ids'])
    choices = json.loads((root/'splits/qaqa_tune_choice_fixedE_20261122.json').read_text())
    out = {}
    for model in MODELS:
        r = Q.Raw(model, 'qaqafull', str(root/'bank/qaqa_full_bank_20261122.jsonl'),
                  ('test_full570', 'test_full570_tuned', 'test_full570_strong'))
        s = Q.SET[model][1]
        ch = choices[model]
        if ch['n1_strength'] != s or len(skip) != 100 or not skip.issubset(r.ids) or len(set(r.ids)) != 570:
            raise ValueError((model, 'unexpected choice or evaluation roster'))
        e = r.e(s)
        f = 'conf:' + e.split(':')[1]
        for arm in ['x', e, f, *[r.ec(s, k) for k in Q.LAD]]:
            r.need(arm)
        es, ea = r.arm_system(e)
        ns, na, _ = r.n1_system(s, ch['n1_cap_e'], ch['n1_cap_own'])
        fs, fa = r.arm_system(f)
        conf = r.confidence('qaqafull', f)
        out[model] = {q: [es[q], int(ea[q]), ns[q], int(na[q]), fs[q], int(fa[q]), conf[q],
                          r.blank('x', q), r.qtype[q]] for q in r.ids if q not in skip}
    return out


COLUMNS = ('primary', 'N1_vs_Efill', 'which_blanks', 'same_blank_text', 'boundary',
           'N1_release_count', 'Efill_release_count', 'common_support_vs_Efill', 'common_support_origin_quota')


def per_model(rows, ids, gold=False):
    # For qa2 the input order is bank order. For joint TQA it is sorted common ID order.
    for q in ids:
        e, ea, n, na, f, fa, score, _, _ = rows[q]
        if (not {e, n, f} <= {-1, 0, 1} or (ea and (not na or e != n))
                or (fa and score is None) or (score is not None and (not math.isfinite(score) or score > 0))):
            raise ValueError((q, 'bad score or E answer changed'))
    blanks = {q for q in ids if not rows[q][1]}
    allowed = {q for q in blanks if not gold or rows[q][8] != 'no_failure'}
    released = {q for q in allowed if rows[q][3]}
    N = sum(rows[q][1] for q in ids) + len(released)
    rank = sorted((q for q in ids if rows[q][5] and rows[q][6] is not None), key=lambda q: -rows[q][6])
    eb = [q for q in rank if q in allowed]
    if N > len(rank) or len(released) > len(eb):
        raise ValueError('unreachable answer count')
    global_top = set(rank[:N])
    ef = set(eb[:len(released)])
    available = released.intersection(eb)
    ef_common = set(eb[:len(available)])
    ef_origin = set()
    for own in (True, False):
        quota = sum(rows[q][7] == own for q in available)
        group = [q for q in eb if rows[q][7] == own]
        ef_origin.update(group[:quota])
    if len(ef_origin) != len(available):
        raise ValueError('origin controls have unequal coverage')
    cells = {}
    for q in ids:
        e, _, n, _, f, *_ = rows[q]
        gated_n = n if q in released else e
        ef_value = f if q in ef else e
        delta = gated_n - ef_value
        # Blank rows keep their original judge score when no candidate replaces them.
        which = (int(q in available) - int(q in ef_common)) * (f - e)
        text = n - f if q in available else 0
        common_n = n if q in available else e
        cells[q] = (gated_n - (f if q in global_top else 0), delta, which, text, delta - which - text,
                    int(q in released), int(q in ef), common_n - (f if q in ef_common else e),
                    common_n - (f if q in ef_origin else e))
    return cells


def compare(tqa, qa2):
    import numpy as np
    boot_rosters = {}
    for count in (3, 5):
        models = MODELS[:count]
        draws = np.random.default_rng(20261122)
        tasks = []
        for label, bank, gold in [('tqa', tqa, False), ('qa2', qa2, False), ('qa2-gold', qa2, True)]:
            common = set.intersection(*(set(bank[m]) for m in models))
            ids = sorted(common) if label == 'tqa' else [q for q in bank[models[0]] if q in common]
            expected_count = (521 if count == 3 else 520) if label == 'tqa' else 470
            if len(ids) != expected_count:
                raise ValueError('wrong common roster')
            if label == 'tqa':
                pre = [per_model(bank[m], list(bank[m])) for m in models]
                print(label, count, 'crop after selection, N1/Efill releases',
                      [sum(d[q][i] for d in pre for q in ids) for i in (5, 6)])
            values = [per_model(bank[m], ids, gold) for m in models]
            table = np.array([[sum(d[q][j] for d in values) for j in range(len(COLUMNS))] for q in ids])
            # One draw per question, all models together. Fixed selected systems; no refitting within a draw.
            total = len(ids)
            if total not in boot_rosters:
                rng = random.Random(20261116)  # same draws, ranks and question order as the released tables
                boot_rosters[total] = np.array([np.bincount([rng.randrange(total) for _ in range(total)],
                                                            minlength=total) for _ in range(4000)])
            ci = np.sort(boot_rosters[total] @ table, axis=0)[[100, 3899]]
            print(label, count, 'models', len(ids), 'questions')
            for col, value, left, right in zip(COLUMNS, table.sum(axis=0), *ci):
                if col.endswith('_count'):
                    print(' ', col, value)
                else:
                    print(' ', col, value, [int(left), int(right)])
            ordinary_count = sum(bank[m][q][1] for m in models for q in ids)
            print('  original E answers / total N1 answers', ordinary_count, ordinary_count + table[:,5].sum())
            print('  individual N1 vs Efill', [sum(d[q][1] for q in ids) for d in values])
            if not gold:
                # Independent bank-level samples for the descriptive cross-dataset difference.
                weights = draws.multinomial(total, np.full(total, 1 / total), size=10000)
                sims = weights @ table[:, 1]
                tasks.append((table[:, 1].sum() * 100 / (count * total), sims * 100 / (count * total)))
        effect = tasks[0][1] - tasks[1][1]
        print('TQA less QA2 per 100 model/question, point and interval',
              np.round([tasks[0][0] - tasks[1][0], *np.quantile(effect, [.025, .975])], 2).tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--tqa', action='store_true')
    g.add_argument('--qa2', action='store_true')
    g.add_argument('--compare', metavar=('TQA', 'QA2'), nargs=2)
    args = parser.parse_args()
    if args.compare:
        compare(*(json.loads(Path(p).read_text()) for p in args.compare))
    else:
        print(json.dumps(read_tqa() if args.tqa else read_qa2(), separators=(',', ':'), allow_nan=False))
