"""Copy a verified control cache into an independent run without changing labels."""
import hashlib
import json
import os
from pathlib import Path
import re

from .judge import qa_cache_key
from .qa_scoring import (OUTCOME_POLICY, FULL_PERSONAL_REFUSAL_POLICY,
                         apply_full_personal_refusal_policy)
from .qa_generation_adapter import digest
from .io_utils import atomic_json_dump


def seed_control_cache(*, source_dir, run_dir, judge_config, scored_rows, conditions,
                       scored_sha256, stem):
    if source_dir is None:
        raise ValueError('supply --controls-judge-cache for an independent copy of the original control judgments')
    filename = 'judge_cache_%s.jsonl' % re.sub(r'[^A-Za-z0-9_.-]', '_', judge_config['model'])
    source_cache = Path(source_dir) / filename
    target_cache = Path(run_dir) / 'judge_cache' / filename
    if source_cache.resolve() == target_cache.resolve():
        raise ValueError('evaluation must use an independent cache copy')
    guard = judge_config.get('full_personal_refusal_guard', False)
    if type(guard) is not bool:
        raise ValueError('invalid configured full-refusal guard')
    expected_policy = FULL_PERSONAL_REFUSAL_POLICY if guard else OUTCOME_POLICY
    if judge_config.get('outcome_policy', expected_policy) != expected_policy:
        raise ValueError('control cache outcome policy configuration differs')
    # A guarded all-refusal pilot can legitimately have made no semantic calls.
    # Missing entries are allowed below only for verifiable guard-only grades.
    source_bytes = source_cache.read_bytes() if source_cache.exists() or not guard else b''
    def entries(data):
        result = {}
        for line in data.decode().splitlines():
            if line.strip():
                row = json.loads(line); result[row['key']] = row
        return result
    source_entries = entries(source_bytes)
    for row in scored_rows:
        for condition in conditions:
            key = qa_cache_key(judge_config['model'], row['question'], row['judge_reference_answers'],
                               row['outputs'][condition], row.get('incorrect') or None,config=judge_config)
            cached = source_entries.get(key, {})
            match = re.match(r'\s*\**\s*([ABC])\b', cached.get('answer', ''))
            label = {'A': 'correct', 'B': 'wrong', 'C': 'abstain'}.get(match.group(1)) if match else None
            recorded = row['judge'][condition]
            if guard:
                derived = apply_full_personal_refusal_policy(
                    {'label': label or 'unresolved', 'raw': cached.get('answer',''),
                     'rubric_id': recorded.get('rubric_id')}, row['question'], row['outputs'][condition])
                policy_keys = ('label','outcome_policy','semantic_label_before_policy','decision_source','personal_refusal_guard')
                if (any(recorded.get(k) != derived[k] for k in policy_keys)
                        or recorded.get('raw') != cached.get('answer','')
                        or (label is None and (not derived['personal_refusal_guard']['apply']
                                              or recorded.get('semantic_call_skipped_by_guard') is not True))):
                    raise ValueError('control cache/raw semantic grade does not derive the recorded outcome policy: ' + row['pair_id'])
            elif (recorded.get('outcome_policy', OUTCOME_POLICY) != OUTCOME_POLICY
                  or label != recorded['label']):
                raise ValueError('control cache is missing or differs from its recorded grade: ' + row['pair_id'])
    target_entries = entries(target_cache.read_bytes()) if target_cache.exists() else {}
    added = 0
    for key, row in source_entries.items():
        if key in target_entries and target_entries[key]['answer'] != row['answer']:
            raise ValueError('cache collision would change an existing grade; do not overwrite it')
        if key not in target_entries: target_entries[key] = row; added += 1
    target_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target_cache) + '.tmp')
    temporary.write_text(''.join(json.dumps(target_entries[k], ensure_ascii=False) + '\n' for k in sorted(target_entries)))
    os.replace(temporary, target_cache)
    atomic_json_dump({'source_cache': str(source_cache), 'source_cache_sha256': hashlib.sha256(source_bytes).hexdigest(),
                     'target_cache': str(target_cache), 'target_cache_sha256_after_merge': digest(target_cache),
                     'source_entries': len(source_entries), 'added_entries': added,
                     'control_scored_items_sha256': scored_sha256},
                    str(Path(run_dir) / (stem + '_judge_cache_seed.json')))
    return {'judge_cache_dir': str(target_cache.parent), 'frozen_control_source_sha256': scored_sha256,
            'frozen_control_grades': {r['pair_id']: {c: r['judge'][c] for c in conditions} for r in scored_rows}}
