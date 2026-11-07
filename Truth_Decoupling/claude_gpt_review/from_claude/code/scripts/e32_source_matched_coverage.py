#!/usr/bin/env python3
"""Equal actual-coverage diagnostics for the live-source generation experiments.

Every condition uses the same unmodified-model, mean answer-token log likelihood.
Only question and generated text enter this score. Actual substantive-answer
eligibility is measured offline by the reference-aware semantic grader; A/B
correctness never enters ordering. These are tuning diagnostics, not a newly
unseen test set. Original generation and grading files remain untouched.
"""
import argparse
import math
from pathlib import Path

import torch

import common
import e25_qa_das as qa
import e27_qa_generation as source_loader
from core.experiment_provenance import implementation_identity, require_unchanged
from core.judge import QA_GRADER_VERSION
from core.model_io import forward_logits
from core.qa_context import encode_answer, pack_answers, answer_logprobs
from core.qa_precision import fp32_output_projection
from core.qa_scoring import reference_answers
from core.selective_eval import evaluate_coverage

PROTOCOL = 'source_generation_equal_simple_coverage_v1'
COVERAGES = (.25, .5, .75)
SOURCE_PROTOCOLS = ('same_prefix_source_interchange_v1', 'supervised_same_prefix_source_interchange_v1',
                    'supervised_source_pair_preference_v1',
                    'source_natural_completion_diagnostic_v1',
                    'supervised_source_pair_preference_question_mask_v1',
                    'source_actual_norm_random_orientation_control_v1',
                    'source_question_mask_uniform256_diagnostic_v1')
CONTROL_NAMES = ['baseline', 'source_only', 'equal_logit_mix', 'source_das_swap', 'source_random_swap']
PREFERENCE_NAMES = CONTROL_NAMES + ['trained_source_swap_das_init', 'pair_preference_source_swap']
MASKED_NAME = 'pair_preference_question_masked_source'
MASK_MODE = 'question_attention_keys_masked_positions_preserved_v1'


def sha256_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def extended_source_names(raw, meta, question_ids, full_ids, pilot):
    """Validate the registered controls without changing their generation method identity."""
    identity, parent = raw['identity'], raw.get('control_identity') or {}
    protocol = raw['protocol']
    expected_config = {k: meta['config'][k] for k in ('qa_system', 'qa_user', 'max_new_tokens', 'batch_size')}
    parent_protocol = SOURCE_PROTOCOLS[3] if protocol == SOURCE_PROTOCOLS[6] else SOURCE_PROTOCOLS[2]
    if (identity.get('model') != meta['model'] or identity.get('config') != expected_config
            or identity.get('question_ids') != question_ids
            or parent.get('protocol') != parent_protocol
            or any(parent.get(k) != identity.get(k) for k in ('model', 'source', 'layer', 'config', 'source_system'))
            or not parent_cohort_matches(parent.get('question_ids'), question_ids, full_ids, pilot)
            or any(obj.get('alpha') != 1. or obj.get('same_prefix_tokens') is not True
                   or obj.get('gold_used_at_inference') is not False for obj in (identity, parent))
            or not sha256_value(identity.get('control_generation_sha256'))):
        raise ValueError('extended source control has no matching frozen parent')
    if protocol in (SOURCE_PROTOCOLS[3], SOURCE_PROTOCOLS[6]):
        budget = identity.get('budget_override') or {}
        loop_key = ('only_python_generation_loop_limit_changes' if protocol == SOURCE_PROTOCOLS[3]
                    else 'only_generation_loop_limit_changes')
        hashes = (('parent_generation_sha256', 'parent_scored_sha256', 'adapter_sha256')
                  if protocol == SOURCE_PROTOCOLS[3] else
                  ('masked_parent_generation_sha256', 'masked_parent_grade_sha256', 'control_grade_sha256', 'adapter_sha256'))
        if (budget.get('parent_max_new_tokens') != expected_config['max_new_tokens']
                or budget.get('max_new_tokens') != 256 or budget.get(loop_key) is not True
                or any(identity.get(k) is not True for k in ('no_forced_eos', 'no_text_trimming', 'all_questions_retained'))
                or any(not sha256_value(identity.get(k)) for k in hashes)):
            raise ValueError('unverified natural-completion budget diagnostic')
        if protocol == SOURCE_PROTOCOLS[3] and identity['parent_generation_sha256'] != identity['control_generation_sha256']:
            raise ValueError('completion generation parent bindings disagree')
        if protocol == SOURCE_PROTOCOLS[6]:
            masked_parent = raw.get('masked_parent_identity') or {}
            if (masked_parent.get('protocol') != SOURCE_PROTOCOLS[4]
                    or any(masked_parent.get(k) != identity.get(k) for k in
                           ('model','source','layer','config','condition','source_mode','source_system',
                            'adapter_sha256','source_mask_manifest_sha256'))
                    or not parent_cohort_matches(masked_parent.get('question_ids'), question_ids, full_ids, pilot)):
                raise ValueError('extended masked answer has no matching 48-token parent')
    if protocol in (SOURCE_PROTOCOLS[4], SOURCE_PROTOCOLS[6]):
        if (identity.get('condition') != MASKED_NAME or identity.get('source_mode') != MASK_MODE
                or not sha256_value(identity.get('source_mask_manifest_sha256'))
                or not sha256_value(identity.get('adapter_sha256'))):
            raise ValueError('question-masked source control identity differs')
        return PREFERENCE_NAMES + [MASKED_NAME]
    if protocol == SOURCE_PROTOCOLS[5]:
        if (identity.get('training_called') is not False or identity.get('seeds') != [42, 43, 44]
                or not sha256_value(identity.get('target_adapter_sha256'))
                or identity.get('target_adapter_sha256') != parent.get('adapter_sha256')
                or identity.get('parent_generation_sha256') != identity['control_generation_sha256']
                or not sha256_value(identity.get('parent_scored_sha256'))
                or not identity.get('learned_information_borrowed')):
            raise ValueError('actual-norm random control provenance differs')
        return PREFERENCE_NAMES + ['size_matched_random_s%d' % s for s in (42, 43, 44)]
    return PREFERENCE_NAMES


def parent_cohort_matches(parent_ids, child_ids, full_ids, pilot):
    """A three-question wiring check may reuse the first three of a full parent."""
    if child_ids != (full_ids[:3] if pilot else full_ids):
        return False
    return parent_ids == full_ids or (pilot and parent_ids == child_ids)


def inputs(args):
    ns, meta, bank, prepared = source_loader.source_run(args.source_run)
    path = args.run_dir / (args.stem + '.json')
    raw = qa.read(path)
    questions = prepared['splits']['tune'][:3] if args.stem == 'check' else prepared['splits']['tune']
    names = raw.get('conditions')
    if (raw.get('completed') is not True or raw.get('split') != 'tune' or raw.get('model') != meta['model']
            or not isinstance(names, list) or not names or len(names) != len(set(names))
            or {s['name'] for s in raw['condition_specs']} != set(names)
            or [r['pair_id'] for r in raw['evaluations']] != [r['pair_id'] for r in questions]):
        raise ValueError('need complete same-model source-generation tuning cohort')
    if args.stem == 'tune' and (raw.get('pilot') or raw.get('wiring_pilot')):
        raise ValueError('a wiring pilot cannot be presented as a full tuning comparison')
    identity = raw['identity']
    if raw.get('protocol') in SOURCE_PROTOCOLS[3:]:
        expected_names = extended_source_names(raw, meta, [q['pair_id'] for q in questions],
                                              [q['pair_id'] for q in prepared['splits']['tune']], args.stem == 'check')
    elif raw.get('protocol') == SOURCE_PROTOCOLS[2]:
        expected_names = CONTROL_NAMES + ['trained_source_swap_das_init', 'pair_preference_source_swap']
        parent = raw.get('control_identity') or {}
        parent_hash = identity.get('control_generation_sha256', '')
        expected_config = {k: meta['config'][k] for k in ('qa_system', 'qa_user', 'max_new_tokens', 'batch_size')}
        if (identity.get('condition') != 'pair_preference_source_swap'
                or parent.get('protocol') != SOURCE_PROTOCOLS[1]
                or parent.get('condition') != 'trained_source_swap_das_init'
                or identity.get('model') != meta['model'] or identity.get('config') != expected_config
                or identity.get('question_ids') != [q['pair_id'] for q in questions]
                or any(parent.get(k) != identity.get(k) for k in ('model', 'source', 'layer', 'config'))
                or not parent_cohort_matches(parent.get('question_ids'), identity['question_ids'],
                                             [q['pair_id'] for q in prepared['splits']['tune']], args.stem == 'check')
                or identity.get('alpha') != 1.0 or identity.get('same_prefix_tokens') is not True
                or parent.get('alpha') != 1.0 or parent.get('same_prefix_tokens') is not True
                or not isinstance(parent_hash, str) or len(parent_hash) != 64
                or any(c not in '0123456789abcdef' for c in parent_hash)):
            raise ValueError('preference source generation has no matching frozen six-arm parent')
    else:
        expected_names = (CONTROL_NAMES if raw.get('protocol') == SOURCE_PROTOCOLS[0] else
                          CONTROL_NAMES + [identity.get('condition')])
    if (raw.get('protocol') not in SOURCE_PROTOCOLS or identity.get('protocol') != raw['protocol']
            or names != expected_names or [c['name'] for c in raw['condition_specs']] != names
            or any(c.get('uses_reference_answer') is not False for c in raw['condition_specs'])
            or identity.get('gold_used_at_inference') is not False
            or raw.get('protocol_fingerprint') != qa.fingerprint(identity)
            or raw.get('bank_protocol_fingerprint') != qa.fingerprint(identity)):
        raise ValueError('unsupported or reference-informed source generation identity')
    if raw['protocol'] == SOURCE_PROTOCOLS[1] and identity['condition'] not in (
            'trained_source_swap_das_init', 'trained_source_swap_random_init'):
        raise ValueError('unsupported trained source condition')
    if identity['source'] != source_loader.source_identity(ns, meta, [identity['layer']]):
        raise ValueError('generation source differs from the unmodified scoring model/source')
    qa.verify_question_rows([{**r, 'idx': i} for i, r in enumerate(raw['evaluations'])], questions)
    for row in raw['evaluations']:
        if set(row['outputs']) != set(names) or set(row['finished']) != set(names):
            raise ValueError('generation condition missing from a question')
        if any(not isinstance(row['outputs'][name], str) or type(row['finished'][name]) is not bool for name in names):
            raise ValueError('invalid generation text or completion flag')
    return ns, meta, prepared, raw, path


def score_identity(meta, raw, raw_path, stem):
    return {'protocol': PROTOCOL, 'source_generation_sha256': qa.digest(raw_path),
        'model': meta['model'], 'source': raw['identity']['source'], 'conditions': raw['conditions'],
        'question_ids': [r['pair_id'] for r in raw['evaluations']],
        'score': 'unmodified-model mean log probability of re-encoded generated answer tokens, no EOS',
        'input_fields': ['question', 'generated_answer'], 'uses_gold_in_score': False,
        'batch_size': 1, 'precision': 'float32 output projection, unchanged native backbone',
        'pilot': stem == 'check', 'exploratory_tuning_only': True}


def validate_score_rows(rows, raw, partial=False):
    expected = raw['evaluations']
    if (len(rows)>len(expected) or [r['pair_id'] for r in rows] !=
            [r['pair_id'] for r in (expected[:len(rows)] if partial else expected)]):
        raise ValueError('score fixed cohort/order changed')
    names = raw['conditions']
    for row, original in zip(rows, expected):
        if set(row['scores']) != set(names) or any(isinstance(v,bool) or not isinstance(v,(int,float))
                                                  or not math.isfinite(v) for v in row['scores'].values()):
            raise ValueError('score conditions or finite values changed')
        same_text = {}
        for name in names:
            text, value = original['outputs'][name], row['scores'][name]
            if text in same_text and same_text[text] != value:
                raise ValueError('identical generated text has different scores')
            same_text[text] = value
        if row.get('unique_generated_strings_scored') != len(same_text):
            raise ValueError('unique generated string count changed')


def score(args, bundle=None):
    ns, meta, prepared, raw, raw_path = inputs(args)
    code = implementation_identity(__file__)
    identity = {**score_identity(meta,raw,raw_path,args.stem), 'implementation':code}
    path = args.run_dir / (args.stem + '_simple_scores.json')
    out = qa.read(path) if path.exists() else {'identity': identity, 'rows': [], 'completed': False}
    if out['identity'] != identity:
        raise ValueError('score source/config/implementation changed; keep the old artifact and use a new directory')
    validate_score_rows(out['rows'],raw,partial=True)
    if out['completed']:
        if len(out['rows']) != len(raw['evaluations']):
            raise ValueError('completed score artifact is missing questions')
        return out
    if bundle is None:
        bundle = qa.open_bundle(meta, prepared)
    qa.verify_bundle(bundle, meta, prepared)
    versions = [(p, p._version) for p in bundle.model.parameters()]
    with fp32_output_projection(bundle):
        for record in raw['evaluations'][len(out['rows']):]:
            scores, unique = {}, {}
            for name in raw['conditions']:
                prediction = record['outputs'][name]
                if prediction not in unique:
                    value = -1e6
                    if prediction.strip():
                        item = encode_answer(bundle, meta['config']['qa_system'], record['question'], prediction,
                                             meta['config']['qa_user'])
                        if record.get('prompt_token_ids', {}).get('base') != item['prefix_ids']:
                            raise ValueError('scoring prompt differs from the original unmodified-model prompt')
                        packed = pack_answers([item], bundle.tokenizer.pad_token_id, bundle.device)
                        value = float(answer_logprobs(forward_logits(bundle, packed['inputs']), packed)[0]) / item['answer_length']
                        if not bool(torch.isfinite(torch.tensor(value))):
                            raise RuntimeError('nonfinite answer log likelihood')
                    unique[prediction] = value
                scores[name] = unique[prediction]
            out['rows'].append({'pair_id': record['pair_id'], 'scores': scores,
                                'unique_generated_strings_scored': len(unique)})
            if len(out['rows']) % 16 == 0:
                qa.save(path, out)
    if any(p._version != version for p, version in versions):
        raise RuntimeError('unmodified-model scorer changed model parameters')
    require_unchanged(code, __file__)
    validate_score_rows(out['rows'],raw)
    out['completed'] = True
    qa.save(path, out)
    return out


def report(args):
    _, meta, _, raw, raw_path = inputs(args)
    score_path = args.run_dir / (args.stem + '_simple_scores.json')
    scored_path = args.run_dir / (args.stem + '_scored_items.json')
    scores, graded = qa.read(score_path), qa.read(scored_path)
    ids = [r['pair_id'] for r in raw['evaluations']]
    names = raw['conditions']
    expected_identity = score_identity(meta,raw,raw_path,args.stem)
    if (scores.get('completed') is not True or any(scores['identity'].get(k)!=v for k,v in expected_identity.items())
            or not isinstance(scores['identity'].get('implementation'),dict)
            or scores['identity']['source_generation_sha256'] != qa.digest(raw_path)
            or scores['identity']['conditions'] != names or scores['identity']['question_ids'] != ids
            or [r['pair_id'] for r in scores['rows']] != ids):
        raise ValueError('score artifact is incomplete or foreign')
    validate_score_rows(scores['rows'],raw)
    if (graded.get('grading_completed') is not True or graded.get('completed') is not True
            or graded['grading_source_generation_sha256'] != qa.digest(raw_path)
            or [r['pair_id'] for r in graded['evaluations']] != ids or graded['conditions'] != names
            or graded.get('model') != raw['model'] or graded.get('split') != raw['split']
            or graded.get('identity') != raw['identity'] or graded.get('condition_specs') != raw['condition_specs']):
        raise ValueError('semantic grading is incomplete or foreign')
    qa.verify_question_rows([{**r,'idx':i} for i,r in enumerate(graded['evaluations'])],raw['evaluations'])
    arms = {name: [] for name in names}
    quality = {name: {'replacement_character': 0, 'unfinished': 0} for name in names}
    for original, row, score_row in zip(raw['evaluations'], graded['evaluations'], scores['rows']):
        if (row['outputs'] != original['outputs'] or row['finished'] != original['finished']
                or set(score_row['scores']) != set(names)):
            raise ValueError('graded answers or score condition set differ from raw generation')
        references, only_refusal = reference_answers(original['gold'])
        if row.get('gold_is_abstention') is not bool(only_refusal):
            raise ValueError('reference-only-refusal partition changed')
        if row.get('judge_reference_answers') != references or set(row.get('outcome',{})) != set(names) or set(row.get('judge',{})) != set(names):
            raise ValueError('grading references or condition coverage changed')
        for name in names:
            label = row['outcome'][name]
            if (label not in ('correct', 'wrong', 'abstain') or label != row['judge'][name]['label']
                    or row['judge'][name].get('rubric_id') != QA_GRADER_VERSION):
                raise ValueError('unresolved or inconsistent semantic grade')
            if only_refusal:
                continue
            arms[name].append({'id': str(row['pair_id']), 'pair_id': str(row['pair_id']),
                               'outcome': label, 'score': score_row['scores'][name]})
            quality[name]['replacement_character'] += '\ufffd' in row['outputs'][name]
            quality[name]['unfinished'] += not row['finished'][name]
        for a in names:
            for b in names:
                if original['outputs'][a] == original['outputs'][b] and (score_row['scores'][a] != score_row['scores'][b]
                                                                      or row['outcome'][a] != row['outcome'][b]):
                    raise ValueError('identical question/answer received different score or grade across conditions')
    comparison = evaluate_coverage(arms, COVERAGES, n_boot=args.n_boot, seed=42)
    result = {'protocol': PROTOCOL, 'model': meta['model'], 'pilot': args.stem == 'check',
        'exploratory_tuning_only': True, 'primary_quality': 'original semantic grades; malformed/unfinished counts separately shown',
        'source_files': {str(p.name): qa.digest(p) for p in (raw_path, score_path, scored_path)},
        'score_uses_gold': False, 'actual_answer_eligibility': 'reference-aware offline semantic refusal annotation; A/B both eligible',
        'natural': {name: {'n': len(rs), 'correct': sum(r['outcome'] == 'correct' for r in rs),
                          'answered': sum(r['outcome'] != 'abstain' for r in rs)} for name, rs in arms.items()},
        'quality': quality, 'matched_coverage': comparison}
    qa.save(args.run_dir / (args.stem + '_matched_coverage.json'), result)
    lines = ['# 真实来源交换：同实际作答率探索性比较', '', '全部来自调参集，不是新的确认性测试。',
             '所有组使用同一个未干预模型给生成答案打平均词分；实质作答由离线语义标注统计。', '']
    for point in comparison['coverages']:
        lines += ['## 目标 %.0f%%' % (point['target_coverage'] * 100), '',
                  '| 方法 | 实际回答 | 答对 | 已回答准确率 | 状态 |', '|---|---:|---:|---:|---|']
        for name, row in point['arms'].items():
            lines.append('| %s | %s | %s | %s | %s |' % (name, row.get('answered'), row.get('correct'),
                row.get('accuracy_answered'), '可行' if row['feasible'] else '实际回答不足'))
        lines.append('')
    (args.run_dir / (args.stem + '_matched_coverage.md')).write_text('\n'.join(lines))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('score', 'report'))
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--stem', choices=('check', 'tune'), default='tune')
    p.add_argument('--n-boot', type=int, default=2000)
    args = p.parse_args()
    if args.n_boot < 1:
        p.error('n-boot must be positive')
    if args.phase == 'score':
        score(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
