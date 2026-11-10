"""Independent read-only E43 arithmetic. No Claude table script is imported or called.

Only pre-captured train/tune raw generations, probabilities, literal Pro replies
and direction records are read. No models, test questions or API calls.
"""
import argparse
import collections
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import torch


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row['cell_id']
        if key in out:
            raise ValueError('duplicate cell ID: ' + key)
        out[key] = row
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    # Explicitly reproduce the reported surface convention; also report exact bytes separately.
    return ' '.join(text.casefold().split()).rstrip('.!? ')


def word_count(text):
    return len(text.split())


def pure_idk(text):
    return bool(re.fullmatch(r"\s*i\s+(?:don't|do not)\s+know[.!?]?\s*", text, re.I))


def label_allenai(row):
    t, i = row['truth_p_yes'], row['info_p_yes']
    if not (math.isfinite(t) and math.isfinite(i) and 0 <= t <= 1 and 0 <= i <= 1):
        raise ValueError('invalid probability')
    return 'B' if t < .5 else ('A' if i >= .5 else 'N')


def label_pro(row):
    # Every captured reply is a literal single letter; do not reuse project parsing logic.
    truth, info = row['truth_raw'].strip(), row['info_raw'].strip()
    if truth not in ('T', 'F') or info not in ('S', 'H', 'R'):
        raise ValueError('this audit expected literal complete Pro replies')
    assert row['truth'] == truth and row['info'] == info
    return 'B' if truth == 'F' else {'S': 'A', 'H': 'H', 'R': 'C'}[info]


def load_judged(root, run):
    directory = root / run
    data = read(directory / 'judge_input.json')
    cells = jsonl(directory / 'judge2/cells.jsonl')
    raw_allenai = jsonl(directory / 'judge2/allenai.jsonl')
    raw_pro = jsonl(directory / 'judge2/pro.jsonl') if (directory / 'judge2/pro.jsonl').exists() else {}
    result = {}
    for item in data['evaluations']:
        for condition in data['conditions']:
            cid = item['pair_id'] + '|' + condition
            cell = cells[cid]
            assert cell['question'] == item['question'] and cell['prediction'] == item['outputs'][condition]
            assert raw_allenai[cid]['input_sha'] == cell['input_sha']
            if raw_pro:
                assert raw_pro[cid]['input_sha'] == cell['input_sha']
            a = label_allenai(raw_allenai[cid])
            record = {'pair_id': item['pair_id'], 'question': item['question'],
                      'text': item['outputs'][condition], 'refusal_rule': item['rule_abstain'][condition],
                      'raw_allenai': a, 'allenai': 'R' if item['rule_abstain'][condition] else a,
                      'pro': label_pro(raw_pro[cid]) if raw_pro else None}
            result.setdefault(condition, {})[item['pair_id']] = record
    assert sum(map(len, result.values())) == len(cells) == len(raw_allenai)
    return result


def comparison(rows, base, judge):
    assert set(rows) == set(base)
    counts = collections.Counter()
    bucket_counts = {n: collections.Counter() for n in ['normalized_same', 'changed_not_longer', 'longer_by_at_least_two']}
    flips = []
    for pid, after in rows.items():
        before = base[pid]
        a, b = before[judge], after[judge]
        counts[a + '->' + b] += 1
        if normalized(before['text']) == normalized(after['text']):
            bucket = 'normalized_same'
        elif word_count(after['text']) >= word_count(before['text']) + 2:
            bucket = 'longer_by_at_least_two'
        else:
            bucket = 'changed_not_longer'
        bucket_counts[bucket]['n'] += 1
        if a == 'B' and b == 'A':
            kind = 'wrong_to_right'
        elif a == 'A' and b == 'B':
            kind = 'right_to_wrong'
        else:
            kind = None
        if kind:
            bucket_counts[bucket][kind] += 1
            flips.append({'pair_id': pid, 'bucket': bucket, 'kind': kind,
                          'before': before['text'], 'after': after['text'], 'question': before['question']})
    a_before = sum(r[judge] == 'A' for r in base.values())
    a_after = sum(r[judge] == 'A' for r in rows.values())
    bw = counts['B->A'] - counts['A->B']
    return {'n': len(rows), 'A_before': a_before, 'A_after': a_after, 'total_A_delta': a_after - a_before,
            'wrong_to_right': counts['B->A'], 'right_to_wrong': counts['A->B'], 'B_A_net': bw,
            'other_state_contribution_to_A': a_after - a_before - bw,
            'transition_matrix': dict(sorted(counts.items())),
            'buckets': {name: {k: values.get(k, 0) for k in ['n', 'wrong_to_right', 'right_to_wrong']}
                        for name, values in bucket_counts.items()}, 'flips': flips,
            'exact_text_equal': sum(rows[k]['text'] == base[k]['text'] for k in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = read(args.manifest)
    assert not manifest['changed_during_capture']
    for name, v in manifest['files'].items():
        assert sha(args.inputs / name) == v['sha256'], name
    root = args.inputs
    frozen = read(root / 'frozen_tune.json')
    frozen_train = read(root / 'frozen_train.json')
    trial = load_judged(root, 'judge_tune_protrial')
    late = load_judged(root, 'judge_tune_Elate')
    output = {'scope': 'Independent arithmetic on captured E43 train/tune artifacts; no Claude table scripts used',
              'capture_time_unix': manifest['captured_unix'], 'source_manifest_sha256': sha(args.manifest),
              'definitions': {'allenai': 'F -> B; T/S -> A; T/N -> N; recorded rule refusal separately R',
                              'pro': 'raw literal T/F plus S/H/R; F is B; T gives A/H/C',
                              'surface': 'casefold, collapse whitespace, strip trailing .!?',
                              'longer': 'whitespace word count increases by >= 2',
                              'scope_warning': 'B-to-A net excludes transitions between A and refusal/uninformative. Layer totals repeat the same questions.'},
              'conditions': {}, 'prompt_format': {}, 'comparisons': {}, 'layers': {}, 'raw_refusal_details': [],
              'new_calls': 0, 'test_set_accessed': False}
    for condition, records in trial.items():
        output['conditions'][condition] = {name: dict(collections.Counter(r[name] for r in records.values()))
                                          for name in ['raw_allenai', 'allenai', 'pro']}
        for r in records.values():
            if r['refusal_rule']:
                output['raw_refusal_details'].append({'condition': condition, **r, 'pure_idk': pure_idk(r['text'])})
    refs = output['raw_refusal_details']
    output['refusal_totals'] = {'rule_refusal_cells': len(refs), 'pure_idk_cells': sum(x['pure_idk'] for x in refs),
        'pure_idk_allenai_A': sum(x['pure_idk'] and x['raw_allenai'] == 'A' for x in refs),
        'allenai_raw_labels': dict(collections.Counter(x['raw_allenai'] for x in refs)),
        'pro_labels': dict(collections.Counter(x['pro'] for x in refs)),
        'non_pure_idk_texts': [x for x in refs if not x['pure_idk']],
        'unique_question_answer_pairs_in_trial': len({(r['question'], r['text']) for v in trial.values() for r in v.values()})}
    for name in ['x', 'x_E', 'x_conf', 'x_caut']:
        rows = [q['answers'][name] for q in frozen['questions']]
        output['prompt_format'][name] = {'n': len(rows), 'ends_with_period': sum(r['text'].strip().endswith('.') for r in rows),
            'mean_whitespace_words': statistics.mean(word_count(r['text']) for r in rows),
            'rule_refusal_count': sum(r['refusal_rule'] for r in rows),
            'pure_idk_count': sum(pure_idk(r['text']) for r in rows)}
    output['x_caut_exact_equal_x'] = sum(q['answers']['x']['text'] == q['answers']['x_caut']['text'] for q in frozen['questions'])
    output['x_caut_normalized_equal_x'] = sum(normalized(q['answers']['x']['text']) == normalized(q['answers']['x_caut']['text']) for q in frozen['questions'])
    gold = [q['best_answer'] for q in frozen_train['questions'] if q.get('gold_target', {}).get('usable')]
    output['gold_target_format'] = {'n': len(gold), 'ends_with_period': sum(x.strip().endswith('.') for x in gold),
                                    'mean_whitespace_words': statistics.mean(word_count(x) for x in gold)}
    for judge in ['allenai', 'pro']:
        for condition in ['baseline_x_E', 'E_L18_main']:
            output['comparisons'][judge + '/' + condition] = comparison(trial[condition], trial['baseline_x'], judge)
    aggregate = {b: collections.Counter() for b in ['normalized_same', 'changed_not_longer', 'longer_by_at_least_two']}
    all_flips = []
    net_all_A = 0
    for layer in range(16, 22):
        c = 'E_L%d_main' % layer
        result = comparison(late[c], late['baseline_x'], 'allenai')
        output['comparisons']['allenai/' + c + '/late'] = result
        for name, values in result['buckets'].items():
            aggregate[name].update(values)
        all_flips += [dict(x, layer=layer) for x in result['flips']]
        net_all_A += result['total_A_delta']
    output['six_layer_total'] = {'buckets': {k: dict(v) for k, v in aggregate.items()},
        'wrong_to_right': sum(x['kind'] == 'wrong_to_right' for x in all_flips),
        'right_to_wrong': sum(x['kind'] == 'right_to_wrong' for x in all_flips),
        'sum_total_A_delta': net_all_A,
        'surface_flip_unique_question_ids': sorted({x['pair_id'] for x in all_flips if x['bucket'] == 'normalized_same'}),
        'surface_flips': [x for x in all_flips if x['bucket'] == 'normalized_same'],
        'unique_questions_with_any_B_A_flip': len({x['pair_id'] for x in all_flips})}
    for layer in list(range(3)) + list(range(16, 28)):
        record = torch.load(root / ('dirs/E_L%d.pt' % layer), map_location='cpu', weights_only=False)
        plus, minus = record['endpoints']['plus'], record['endpoints']['minus']
        pairs = [a - b for a, b in zip(plus['values'], minus['values'])]
        g = read(root / ('gen/E_L%d_main_tune.json' % layer))
        rec = read(root / ('gen/E_L%d_recover_tune.json' % layer))
        gap = plus['mean'] - minus['mean']
        write_intended = statistics.mean(r['write']['prefill_gap_abs'] for r in g['rows'])
        write_actual = statistics.mean(r['write']['prefill_stored_norm'] for r in g['rows'])
        output['layers'][str(layer)] = {'plus_mean': plus['mean'], 'minus_mean': minus['mean'],
            'endpoint_gap': gap, 'exactly_equal_means': plus['mean'] == minus['mean'],
            'paired_max_abs_gap': max(map(abs, pairs)), 'paired_mean_gap': statistics.mean(pairs),
            'nonzero_pair_gap_count': sum(x != 0 for x in pairs),
            'intended_mean_prefill_gap': write_intended, 'actual_mean_prefill_norm': write_actual,
            'actual_write_over_abs_endpoint_gap': write_actual / abs(gap) if gap else None,
            'main_periods': sum(r['text'].strip().endswith('.') for r in g['rows']),
            'main_sentence_end_punctuation': sum(r['text'].strip().endswith(('.', '!', '?')) for r in g['rows']),
            'recover_periods': sum(r['text'].strip().endswith('.') for r in rec['rows']),
            'main_mean_words': statistics.mean(word_count(r['text']) for r in g['rows']),
            'online_epoch1_ce_plus': record['training']['history'][0]['ce_plus'],
            'post_training_ce_plus': statistics.mean(r['ce_plus'] for r in record['per_pair_ce_after'])}
        if layer == 16:
            md = read(root / 'gen/E_L16_md_main_tune.json')
            md_gap = record['endpoints_md']['plus']['mean'] - record['endpoints_md']['minus']['mean']
            output['layers'][str(layer)]['mean_difference_control'] = {
                'endpoint_gap': md_gap,
                'actual_mean_prefill_norm': statistics.mean(r['write']['prefill_stored_norm'] for r in md['rows']),
                'intended_mean_prefill_gap': statistics.mean(r['write']['prefill_gap_abs'] for r in md['rows'])}
    switch = {q['pair_id'] for q in frozen_train['questions']
              if not q['answers']['x_conf']['refusal_rule'] and q['answers']['x_caut']['refusal_rule']}
    c5 = torch.load(root / 'dirs/C_L5.pt', map_location='cpu', weights_only=False)
    ce = sorted(r['ce_plus'] for r in c5['per_pair_ce_after'] if r['pair_id'] in switch)
    output['C_L5_switch'] = {'n_frozen_switch_pairs': len(switch), 'n_eligible_trained_switch_pairs': len(ce),
                            'ce_plus_values': ce, 'ce_plus_le_0_08': sum(x <= .08 for x in ce),
                            'ce_plus_round2_le_0_08': sum(round(x, 2) <= .08 for x in ce),
                            'endpoint_gap': c5['endpoints']['plus']['mean'] - c5['endpoints']['minus']['mean']}
    for name, v in manifest['files'].items():
        assert sha(args.inputs / name) == v['sha256'], name
    output['inputs_unchanged'] = True
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['conditions', 'refusal_totals', 'gold_target_format', 'six_layer_total']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
