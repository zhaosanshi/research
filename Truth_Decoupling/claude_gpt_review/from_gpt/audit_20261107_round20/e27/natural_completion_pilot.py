"""Three unfinished E34 answers at a larger natural budget; no token repair."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--ce-run', type=Path, required=True)
    p.add_argument('--parent-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.code_root / 'scripts'), str(args.code_root)]
    import e25_qa_das as qa
    import e27_qa_generation as ga
    import e29_qa_source_interchange as se
    from core import qa_source_pair_preference as pair
    from core.qa_precision import fp32_output_projection
    from core.qa_source_interchange import greedy_source_generate
    ns, meta, bank, prepared = ga.source_run(args.source_run)
    adapter = pair.load(args.parent_run)
    raw_path = args.parent_run / 'tune.json'; raw = qa.read(raw_path)
    assert adapter['source'] == ga.source_identity(ns, meta, [adapter['layer']])
    assert raw['identity']['adapter_sha256'] == qa.digest(args.parent_run / 'preference_adapter.pt')
    assert raw['identity']['control_generation_sha256'] == qa.digest(args.ce_run / 'tune.json')
    for name in ('core/qa_source_interchange.py', 'core/model_io.py', 'core/qa_precision.py'):
        assert raw['identity']['implementation']['files'][name] == qa.digest(args.code_root / name)
    method = pair.METHOD
    rows = [row for row in raw['evaluations'] if not row['finished'][method]][:3]
    assert len(rows) == 3
    bundle = qa.open_bundle(meta, prepared)
    result = {'protocol': 'natural_completion_three_item_probe_v1', 'model': meta['model'],
              'parent_raw_sha256': qa.digest(raw_path), 'parent_budget': 48, 'diagnostic_budget': 256,
              'adapter_config_unchanged': adapter['config'], 'script_sha256': qa.digest(__file__),
              'no_forced_eos': True, 'no_text_trimming': True, 'rows': []}
    with fp32_output_projection(bundle):
        for row in rows:
            base, source = se.prompt_tokens(bundle, adapter['config'], row['question'])
            assert row['prompt_token_ids'] == {'base': base, 'source': source}
            old_ids = row['generation_token_ids'][method]
            assert len(old_ids) == 48
            print('OLD_48', row['pair_id'], repr(row['outputs'][method]), flush=True)
            new = greedy_source_generate(bundle, base, source, mode='source_interchange', layer=adapter['layer'],
                                          direction=adapter['W'], max_new_tokens=256)
            assert new['token_ids'][:48] == old_ids, 'original 48-token prefix changed: ' + row['pair_id']
            ngrams = Counter(tuple(new['token_ids'][i:i + 8]) for i in range(max(0, len(new['token_ids']) - 7)))
            maximum = max(ngrams.values(), default=0)
            result['rows'].append({'pair_id': row['pair_id'], 'question': row['question'], 'gold': row['gold'],
                                   'original_output': row['outputs'][method], 'original_token_ids': old_ids,
                                   'first_48_token_ids_identical': True, 'new': new,
                                   'maximum_repeated_8_token_ngram': maximum,
                                   'repetition_review_flag': maximum >= 3})
            args.output.parent.mkdir(parents=True, exist_ok=True); qa.save(args.output, result)
            print('NATURAL_256', row['pair_id'], 'finished', new['finished'], 'tokens', len(new['token_ids']),
                  'repeat8', maximum, repr(new['output']), flush=True)
    result['completed'] = True; qa.save(args.output, result)


if __name__ == '__main__': main()
