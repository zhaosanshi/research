"""Recompute selected f234a98 early-layer counts from captured raw data; no calls."""
import argparse
import collections
import gzip
import json
import re
from pathlib import Path

from recompute_e43_numbers import label_allenai, comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, default=Path(__file__).with_name('e43_early_selected_raw.json.gz'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(gzip.decompress(args.inputs.read_bytes()))
    cells = {x['cell_id']: x for x in data['selected_rows']['cells.jsonl']}
    raw = {x['cell_id']: x for x in data['selected_rows']['allenai.jsonl']}
    singles = {x['cell_id']: x for x in data['selected_rows']['single_allenai.jsonl']}
    assert len(cells) == len(raw) == len(singles) == 960
    by_condition = {}
    for cid, cell in cells.items():
        label = label_allenai(raw[cid])
        assert raw[cid]['input_sha'] == cell['input_sha'] == singles[cid]['input_sha']
        assert singles[cid]['label'] == label
        by_condition.setdefault(cell['condition'], {})[cell['pair_id']] = {
            'pair_id': cell['pair_id'], 'question': cell['question'], 'text': cell['prediction'],
            'allenai': 'R' if cell['rule_abstain'] else label}
    output = {'scope': 'f234a98 addendum only, existing raw probabilities, no model or judge calls',
              'original_files': data['files'], 'conditions': {}, 'comparisons': {}}
    for condition, rows in by_condition.items():
        output['conditions'][condition] = dict(collections.Counter(row['allenai'] for row in rows.values()))
        if condition != 'baseline_x':
            generation = data['generation']['gen/' + condition + '_tune.json']
            outputs = {row['pair_id']: row for row in generation['rows']}
            assert set(outputs) == set(rows)
            assert all(outputs[pid]['text'] == row['text'] for pid, row in rows.items())
            output['comparisons'][condition] = comparison(rows, by_condition['baseline_x'], 'allenai')
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output['conditions']))


if __name__ == '__main__':
    main()
