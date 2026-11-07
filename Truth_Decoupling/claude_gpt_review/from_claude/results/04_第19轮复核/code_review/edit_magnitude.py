"""Per-arm actual edit magnitude in the frozen-control pilot (trace_displacement=True there).

Only active steps (<= first EOS of that row) are kept, as in compare_real_traces.py.
"""
import json
import statistics
import sys
import tarfile

EOS = {'llama3.1-8b': {128001, 128008, 128009}, 'qwen3-8b': {151645, 151643}}
arch = sys.argv[1]
with tarfile.open(arch) as t:
    for model in ('llama3.1-8b', 'qwen3-8b'):
        raw = json.load(t.extractfile(model + '/pilot3/tune.json'))
        rows_by_id = {r['pair_id']: r for r in raw['evaluations']}
        by_arm = {}
        for a in raw['audit']:
            name = a['condition']
            for rec in a['displacement'] or []:
                for pid in a['stored_ids']:
                    i = a['source_batch_ids'].index(pid)
                    ids = rows_by_id[pid]['generation_token_ids'][name]
                    stop = next((j for j, x in enumerate(ids) if x in EOS[model]), len(ids) - 1)
                    if rec['step'] > stop:
                        continue
                    key = 'prompt' if rec['step'] == 0 else 'decode'
                    by_arm.setdefault(name, {'prompt': [], 'decode': []})[key].append(rec['kept_relative_norm'][i])
        print(model)
        for name, v in by_arm.items():
            p, d = v['prompt'], v['decode']
            print('  %-24s prompt-step rel edit %s | decode median %.4f max %.4f (n=%d)' % (
                name, ['%.4f' % x for x in p], statistics.median(d), max(d), len(d)))
