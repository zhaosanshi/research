"""Summarize saved real-model corruption traces without model or judge calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(path):
    path = Path(path)
    d = json.loads(path.read_text())
    assert d['completed'] and d['reference_reproduced']
    summary = {'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'reference_reproduced': True, 'cases': {}}
    for pid, case in d['cases'].items():
        assert case['reference_output_identical'] and case['bytes']['byte_mapping_verified']
        events = case['bytes']['invalid_utf8_events'] + case['bytes']['literal_replacement_events']
        entry = {'text': case['bytes']['default_decoded'], 'events': events, 'steps': {}}
        batch = next(b for b in d['batches'] if b['start'] == case['batch_start'])
        for step in sorted({e[k] for e in events for k in ('origin_token_index', 'deciding_token_index')}):
            edits = [e for e in case['edits'] if e['step'] == step]
            replay = next(x for x in batch['common_prefix_replays'] if x['step'] == step)
            row = {'edits': edits, 'variants': {}}
            for name, by_row in replay['variants'].items():
                info = by_row[str(case['batch_row'])]
                top = next(t for t in info['tokens'] if t['id'] == info['argmax_id'])
                actual_id = case['effective_ids'][step]
                actual = next(t for t in info['tokens'] if t['id'] == actual_id)
                stops = d['generation_config']['eos_token_id']
                stops = stops if isinstance(stops, list) else [stops]
                eos = [t for t in info['tokens'] if t['id'] in stops]
                best_eos = max(eos, key=lambda t: t['logit'])
                row['variants'][name] = {'argmax_id': top['id'], 'token': top['token_string'],
                    'top_probability': math.exp(top['log_probability']),
                    'actual_generated_token_id': actual_id,
                    'actual_minus_best_eos_logit_margin': actual['logit'] - best_eos['logit'],
                    'stop_tokens': {str(t['id']): {'logit': t['logit'], 'probability': math.exp(t['log_probability'])} for t in eos},
                    'cached_top1_match': info.get('cached_top1_match'),
                    'cached_max_diff': info.get('cached_processed_score_max_abs_difference')}
            entry['steps'][str(step)] = row
        summary['cases'][pid] = entry
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = summarize(a.input)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('summarized', len(result['cases']), 'exactly reproduced cases')
