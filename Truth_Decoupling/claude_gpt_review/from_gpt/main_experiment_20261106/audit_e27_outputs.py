#!/usr/bin/env python3
"""Independent E27 artifact audit; no production metric imports/model/API calls.

Input is the exported {das,md,random: {source,sha256,summary,data}} bundle.
The declared sha256 describes the remote source bytes; only this input bundle's
hash can be verified locally from this export. No labels are changed or dropped.
Refusal-only reference questions remain a separately counted, fixed cohort.

Example:
python3 audit_e27_outputs.py --input qwen_alpha3_tune_audit.json \
    --out-dir . --stem qwen_alpha3_tune --notes qwen_alpha3_tune_review_notes.json
The same script accepts a completed test-split bundle without relabeling it tune.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

LABELS = ('correct', 'wrong', 'abstain')
INITS = ('das', 'md', 'random')
ARMS = ('baseline', 'original_das', 'das_generation_adapter', 'md_generation_adapter', 'random_generation_adapter')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_paired(a, b):
    pos, neg = int((a & ~b).sum()), int((~a & b).sum())
    n = pos + neg
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2**n) if n else 1.
    return {'a_only': pos, 'b_only': neg, 'two_sided_exact_p': p}


def personal_refusal(text):
    text = ' '.join(re.sub(r'[^a-z ]', '', text.lower()).split())
    return text in {'i have no comment', 'i dont know', 'i do not know'}


def transition(a, b):
    return {x + '->' + y: int(((a == x) & (b == y)).sum()) for x in LABELS for y in LABELS}


def surface(text):
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in text.casefold()).split())


def quality(row, name):
    text = row['outputs'][name]
    lines = [s.strip() for s in text.splitlines() if len(s.strip()) >= 20]
    return {'unfinished': not row['finished'][name], 'replacement_character': '\ufffd' in text,
            'symbol_decoration': any(unicodedata.category(c) == 'So' for c in text),
            'htags_literal': 'htags' in text, 'repeated_long_line': len(lines) != len(set(lines)),
            'chars': len(text), 'words': len(text.split())}


def evaluate(input_path, n_boot=2000, seed=0, source_questions=None):
    bundle = read(input_path)
    assert set(bundle) == set(INITS), 'need all three initialization controls'
    first = bundle['das']['data']
    ids = [r['pair_id'] for r in first['evaluations']]
    assert len(ids) == len(set(ids)) and ids
    indexed = {init: {r['pair_id']: r for r in b['data']['evaluations']} for init, b in bundle.items()}
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'pass': bool(ok)})
        if not ok:
            raise ValueError('artifact audit failed: ' + name)
    identities = ['model', 'split', 'source', 'prompt', 'questions_sha256', 'judge_rubric_id',
                  'outcome_policy', 'gold_policy']
    row_fields = ['pair_id', 'question', 'gold', 'incorrect', 'judge_reference_answers', 'gold_is_abstention']
    for init, b in bundle.items():
        data, summary = b['data'], b['summary']
        check(init + '/completed', data['completed'] and data['grading_completed'] and summary['completed'])
        check(init + '/cohort_order', [r['pair_id'] for r in data['evaluations']] == ids)
        for key in identities:
            check(init + '/' + key, data[key] == first[key])
        check(init + '/no_C_intervention', data['C'] == {'enabled': False})
        adapted_spec = next(c for c in data['condition_specs'] if c['name'] == init + '_generation_adapter')
        original_spec = next(c for c in data['condition_specs'] if c['name'] == 'original_das')
        check(init + '/same_deployment_operator_settings', all(adapted_spec[k] == original_spec[k]
              for k in ('layers', 'alpha', 'mode', 'uses_reference_answer')) and not adapted_spec['uses_reference_answer'])
        for pid in ids:
            row, common = indexed[init][pid], indexed['das'][pid]
            for key in row_fields:
                check(init + '/' + pid + '/' + key, row[key] == common[key])
            retained = [g for g in row['gold'] if not personal_refusal(g)]
            check(init + '/' + pid + '/reference_filter', row['judge_reference_answers'] == (retained or row['gold'])
                  and row['gold_is_abstention'] is (not bool(retained)))
            for name in ['baseline', 'original_das', init + '_generation_adapter']:
                check(init + '/' + pid + '/' + name + '/resolved', row['outcome'][name] in LABELS)
                check(init + '/' + pid + '/' + name + '/judge', row['outcome'][name] == row['judge'][name]['label'])
                parsed = re.match(r'\s*\**\s*([ABC])\b', row['judge'][name]['raw'])
                check(init + '/' + pid + '/' + name + '/raw_judge_letter', parsed is not None and
                      {'A': 'correct', 'B': 'wrong', 'C': 'abstain'}[parsed.group(1)] == row['outcome'][name])
                if name in ('baseline', 'original_das'):
                    for key in ('outputs', 'outcome', 'finished', 'judge'):
                        check(init + '/' + pid + '/' + name + '/' + key, row[key][name] == common[key][name])
        for name in ('baseline', 'original_das'):
            a = next(c for c in data['condition_specs'] if c['name'] == name)
            bspec = next(c for c in first['condition_specs'] if c['name'] == name)
            check(init + '/' + name + '/condition', a == bspec)
    arms = {name: indexed['das' if name in ARMS[:3] else name.split('_')[0]] for name in ARMS}
    keep = np.array([not indexed['das'][i]['gold_is_abstention'] for i in ids])
    main_ids = [i for i, yes in zip(ids, keep) if yes]
    outcomes = {name: np.array([rs[i]['outcome'][name] for i in main_ids]) for name, rs in arms.items()}
    n = len(main_ids)
    cohort_provenance = None
    if source_questions is not None:
        questions = read(source_questions)
        check('source_questions_sha256', sha(source_questions) == first['source']['questions_sha256'])
        expected_ids = [q['pair_id'] for q in questions['splits'][first['split']]]
        train_ids = {q['pair_id'] for q in questions['e_train']}
        check('source_split_identity', expected_ids == ids)
        check('source_estimation_disjoint', not (train_ids & set(ids)))
        other_splits = {k: {q['pair_id'] for q in v} for k, v in questions['splits'].items() if k != first['split']}
        check('source_other_splits_disjoint', all(not (set(ids) & v) for v in other_splits.values()))
        cohort_provenance = {'questions_sha256': sha(source_questions), 'estimation_pairs': len(train_ids),
                             'splits': {k: len(v) for k, v in questions['splits'].items()},
                             'exact_split_ids_matched': True, 'estimation_disjoint': True,
                             'other_split_overlap': {k: len(set(ids) & v) for k, v in other_splits.items()}}
    for init, b in bundle.items():
        check(init + '/saved_denominators', b['summary']['n_questions'] == len(ids)
              and b['summary']['n_substantive_gold'] == n and b['summary']['n_resolved_all_conditions'] == n)
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, n, (n_boot, n))
    summary_rows, metrics, draws = {}, {}, {}
    for name, y in outcomes.items():
        correct, wrong, answered = y == 'correct', y == 'wrong', y != 'abstain'
        counts = {s: int((y == s).sum()) for s in LABELS}
        m = {'correct_rate': float(correct.mean()), 'wrong_rate': float(wrong.mean()),
             'answer_rate': float(answered.mean()),
             'precision_answered': float(correct.sum() / answered.sum()) if answered.any() else None,
             'correct_minus_wrong_rate': float((correct.astype(int) - wrong.astype(int)).mean())}
        denominator = answered[boot].sum(1)
        ds = {'correct_rate': correct[boot].mean(1), 'wrong_rate': wrong[boot].mean(1),
              'answer_rate': answered[boot].mean(1),
              'precision_answered': correct[boot].sum(1) / denominator if denominator.min() > 0 else None,
              'correct_minus_wrong_rate': (correct[boot].astype(int) - wrong[boot].astype(int)).mean(1)}
        metrics[name], draws[name] = m, ds
        qs = {i: quality(arms[name][i], name) for i in ids}
        summary_rows[name] = {'n': n, 'counts': counts, 'answered': int(answered.sum()), 'metrics': m,
                             'ci95': {k: np.percentile(v, [2.5, 97.5]).tolist() if v is not None else None for k, v in ds.items()},
                             'quality': {k: {'count': sum(v[k] for v in qs.values()),
                                             'ids': [i for i, v in qs.items() if v[k]]}
                                         for k in ('unfinished', 'replacement_character', 'symbol_decoration', 'htags_literal', 'repeated_long_line')},
                             'quality_denominator': len(ids),
                             'quality_main_cohort': {k: sum(qs[i][k] for i in main_ids)
                                 for k in ('unfinished', 'replacement_character', 'symbol_decoration', 'htags_literal', 'repeated_long_line')},
                             'mean_chars': float(np.mean([v['chars'] for v in qs.values()])),
                             'mean_words': float(np.mean([v['words'] for v in qs.values()]))}
        owner = 'das' if name in ARMS[:3] else name.split('_')[0]
        declared = bundle[owner]['summary']['rates'][name]
        check(name + '/zero_unresolved', declared['n_unresolved'] == 0)
        for status in LABELS:
            check(name + '/saved_rate/' + status, abs(declared['rates'][status] - counts[status] / n) < 1e-12)
        check(name + '/saved_answer_rate', abs(declared['rates']['answer_rate'] - m['answer_rate']) < 1e-12)
        check(name + '/saved_truncation', abs(declared['truncated_rate'] - summary_rows[name]['quality']['unfinished']['count'] / len(ids)) < 1e-12)
    comparisons = {}
    transitions = {}
    for a, b in [('das_generation_adapter', x) for x in ARMS if x != 'das_generation_adapter']:
        key = a + '_minus_' + b
        comparisons[key] = {'a': a, 'b': b, 'n': n,
                           'metrics': {k: {'delta': metrics[a][k] - metrics[b][k] if metrics[a][k] is not None and metrics[b][k] is not None else None,
                                           'ci95': np.percentile(draws[a][k] - draws[b][k], [2.5, 97.5]).tolist()
                                           if draws[a][k] is not None and draws[b][k] is not None else None} for k in metrics[a]},
                           'correctness': exact_paired(outcomes[a] == 'correct', outcomes[b] == 'correct'),
                           'answered': exact_paired(outcomes[a] != 'abstain', outcomes[b] != 'abstain'),
                           'wrong': exact_paired(outcomes[a] == 'wrong', outcomes[b] == 'wrong')}
        transitions[b + '_to_' + a] = transition(outcomes[b], outcomes[a])
    for init in INITS:
        name = init + '_generation_adapter'
        for before in ('baseline', 'original_das'):
            transitions[before + '_to_' + name] = transition(outcomes[before], outcomes[name])
        declared = bundle[init]['summary']['transitions_vs_baseline']
        for name in ('original_das', init + '_generation_adapter'):
            check(init + '/saved_transition/' + name, transition(outcomes['baseline'], outcomes[name]) == declared[name])
    changed, surface_conflicts, exact_conflicts = [], [], []
    for pid in ids:
        row = indexed['das'][pid]
        by_text, by_surface = defaultdict(list), defaultdict(list)
        comparison = {}
        for name in ARMS:
            r = arms[name][pid]
            comparison[name] = {'output': r['outputs'][name], 'outcome': r['outcome'][name],
                                'judge_raw': r['judge'][name]['raw'], 'quality': quality(r, name)}
            by_text[r['outputs'][name]].append((name, r['outcome'][name]))
            by_surface[surface(r['outputs'][name])].append((name, r['outcome'][name]))
        for bucket, dest in ((by_text, exact_conflicts), (by_surface, surface_conflicts)):
            for text, members in bucket.items():
                if len({label for _, label in members}) > 1:
                    dest.append({'pair_id': pid, 'surface': text, 'members': members})
        if len({v['outcome'] for v in comparison.values()}) > 1:
            changed.append({'pair_id': pid, 'question': row['question'], 'gold': row['gold'],
                            'incorrect': row['incorrect'], 'gold_is_abstention': row['gold_is_abstention'], 'arms': comparison})
    result = {'input_sha256': sha(input_path), 'model': first['model'], 'split': first['split'],
              'n_total': len(ids), 'n_main': n, 'excluded_refusal_only_gold_ids': [i for i, yes in zip(ids, keep) if not yes],
              'n_boot': n_boot, 'seed': seed, 'metric_unit': 'one original QA question',
              'common_baseline_and_original_DAS_exactly_identical': True,
              'conditions': first['condition_specs'], 'input_sources': {i: {k: bundle[i][k] for k in ('source', 'sha256')} for i in INITS},
              'cohort_provenance': cohort_provenance,
              'source_hash_limit': 'remote source-file hashes are declared in the export; verified locally: this combined input file only',
              'arms': summary_rows, 'comparisons': comparisons, 'transitions': transitions,
              'exact_text_label_conflicts': exact_conflicts, 'surface_only_label_conflicts': surface_conflicts,
              'label_change_questions': len(changed), 'checks': {'n': len(checks), 'failures': [c for c in checks if not c['pass']]},
              'limits': ['自然作答率不同；这不是E26的同作答率比较。',
                         '这份是调参集；经过配置搜索后的统计只作探索记录，不能写成留出测试结论。' if first['split'] == 'tune' else
                         '这是所提供冻结配置的留出划分，仍需对照记录的配置选择历史。',
                         '原裁判标签完整保留；数字核对一致不等于逐条语义判分都正确。']}
    return result, changed


def markdown(result, notes=None):
    r = result
    lines = ['# E27 Qwen alpha3 三初始化独立复核：' + r['split'], '',
             '**划分：%s；主表%d题，原始%d题。不是同作答率比较。**' % (r['split'], r['n_main'], r['n_total']), '',
             '比较的是以DAS、均值差或随机方向初始化后，继续优化生成目标的适配器；不能把它简称为原DAS目标已经在自由生成上成功。', '',
             '| 条件 | 对 | 错 | 弃答 | 回答数 | 已回答准确率 | 未正常结束 | 含替换字符 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, a in r['arms'].items():
        c = a['counts']; prec = a['metrics']['precision_answered']
        lines.append('| %s | %d | %d | %d | %d | %s | %d | %d |' %
                     (name, c['correct'], c['wrong'], c['abstain'], a['answered'],
                      '—' if prec is None else '%.4f' % prec, a['quality']['unfinished']['count'], a['quality']['replacement_character']['count']))
    lines += ['', '质量列以全部%d条原始输出为分母；正确／错误／弃答等主指标以%d题为分母。' % (r['n_total'], r['n_main']),
              '', '## DAS适配器直接配对差', '',
              '| 相对条件 | 正确数净差 | 正确率差95%区间 | 改对／改错 | 双侧精确p |',
              '|---|---:|---|---|---:|']
    for c in r['comparisons'].values():
        m = c['metrics']['correct_rate']; t = c['correctness']
        lines.append('| %s | %+d | [%+.4f, %+.4f] | %d / %d | %.6g |' %
                     (c['b'], t['a_only'] - t['b_only'], *m['ci95'], t['a_only'], t['b_only'], t['two_sided_exact_p']))
    lines += ['', '自然作答数量不同，下面两项不能冒充同作答率胜出：', '',
              '| 相对条件 | 已回答准确率差及区间 | （对−错）/总题数差及区间 |',
              '|---|---|---|']
    for c in r['comparisons'].values():
        m, u = c['metrics']['precision_answered'], c['metrics']['correct_minus_wrong_rate']
        ms = '—' if m['delta'] is None or m['ci95'] is None else '%+.4f [%+.4f, %+.4f]' % (m['delta'], *m['ci95'])
        us = '—' if u['delta'] is None or u['ci95'] is None else '%+.4f [%+.4f, %+.4f]' % (u['delta'], *u['ci95'])
        lines.append('| %s | %s | %s |' % (c['b'], ms, us))
    t = r['transitions']['baseline_to_das_generation_adapter']
    lines += ['', '## 变化来自哪里', '',
              '原弃答→正确 %d；原错误→正确 %d；原正确→错误 %d；原正确→弃答 %d。' %
              (t['abstain->correct'], t['wrong->correct'], t['correct->wrong'], t['correct->abstain']),
              '原弃答→错误 %d；原错误→弃答 %d。因此错误总数不变时也不能说没有伤害或没有新错题。' %
              (t['abstain->wrong'], t['wrong->abstain']), '',
              '共同基线及原DAS逐题原文、完成状态和判分逐项完全相同；%d项内部核对无失败。' % r['checks']['n'],
              '所有九格转移、自然作答率与条件准确率配对区间见重算JSON。', '', '## 解释边界', '']
    lines.extend('- ' + x for x in r['limits'])
    if notes:
        lines += ['', '## 人工原文复核', '', notes['conclusion'], '',
                  '| 题号 | 复核意见 |', '|---|---|']
        for pid, item in notes['cases'].items():
            if item['category'] == 'reviewed_no_major_flag':
                continue
            lines.append('| %s | %s |' % (pid, item['note'].replace('|', '/')))
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--stem', required=True)
    p.add_argument('--notes', type=Path)
    p.add_argument('--source-questions', type=Path, help='optional frozen source questions.json for exact split/disjointness checks')
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    result, changed = evaluate(args.input, args.n_boot, args.seed, args.source_questions)
    notes = read(args.notes) if args.notes else None
    if notes:
        assert notes['input_sha256'] == result['input_sha256']
        assert set(notes['cases']) == {r['pair_id'] for r in changed}, 'manual review must cover every changed question'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / (args.stem + '_recomputed.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    (args.out_dir / (args.stem + '_changed_cases.json')).write_text(json.dumps(changed, ensure_ascii=False, indent=2, allow_nan=False))
    (args.out_dir / (args.stem + '_review.md')).write_text(markdown(result, notes))
    print(json.dumps({'split': result['split'], 'n': result['n_main'], 'checks': result['checks'],
                      'changed_questions': len(changed)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
