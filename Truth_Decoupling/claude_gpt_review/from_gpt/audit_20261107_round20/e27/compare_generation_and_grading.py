"""Separate generation reproducibility from judge-label drift using actual artifacts."""
import io
import json
from pathlib import Path
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'from_gpt/main_experiment_20261106/artifacts/generation_results.tar.gz'
NEW = ROOT / 'from_claude/results/03_E26同作答率_E27生成适配/E27_release'


def main():
    result = []
    with tarfile.open(OLD) as tar:
        names = {m.name for m in tar.getmembers()}
        for model in ('llama3.1-8b', 'qwen3-8b'):
            for init in ('das', 'md', 'random'):
                prefix = model + '/ce_alpha3_ep3/' + init + '/'
                old = torch.load(io.BytesIO(tar.extractfile(prefix + 'adapter.pt').read()), map_location='cpu', weights_only=True)
                new = torch.load(NEW / model / (init + '_s42') / 'adapter.pt', map_location='cpu', weights_only=True)
                row = {'model': model, 'init': init, 'initial_vectors_equal': all(torch.equal(old['initial_W'][l], new['initial_W'][l]) for l in old['initial_W']),
                       'trained_vectors_equal': all(torch.equal(old['W'][l], new['W'][l]) for l in old['W']),
                       'maximum_vector_difference': max(float((old['W'][l] - new['W'][l]).abs().max()) for l in old['W']),
                       'losses_equal': old['losses'] == new['losses'], 'splits': {}}
                for split in ('tune', 'test'):
                    name = prefix + split + '_scored_items.json'
                    if name not in names:
                        continue
                    a = json.load(tar.extractfile(name)); b = json.load(open(NEW / model / (init + '_s42') / (split + '_scored_items.json')))
                    arows = {r['pair_id']: r for r in a['evaluations']}; brows = {r['pair_id']: r for r in b['evaluations']}
                    assert set(arows) == set(brows)
                    comparisons = {}
                    for cond in a['conditions']:
                        differences = []
                        for pid, x in arows.items():
                            y = brows[pid]
                            if x['outcome'][cond] != y['outcome'][cond]:
                                differences.append({'pair_id': pid, 'question': x['question'], 'same_output': x['outputs'][cond] == y['outputs'][cond],
                                                    'output': x['outputs'][cond], 'old_label': x['outcome'][cond], 'new_label': y['outcome'][cond],
                                                    'gold_is_abstention': x['gold_is_abstention']})
                        comparisons[cond] = {'raw_n': len(arows), 'substantive_n': sum(not r['gold_is_abstention'] for r in arows.values()),
                                             'output_mismatches': sum(x['outputs'][cond] != brows[pid]['outputs'][cond] for pid, x in arows.items()),
                                             'finished_mismatches': sum(x['finished'][cond] != brows[pid]['finished'][cond] for pid, x in arows.items()),
                                             'old_fffd_raw': sum('\ufffd' in x['outputs'][cond] for x in arows.values()),
                                             'new_fffd_raw': sum('\ufffd' in x['outputs'][cond] for x in brows.values()),
                                             'old_fffd_substantive': sum('\ufffd' in x['outputs'][cond] for x in arows.values() if not x['gold_is_abstention']),
                                             'new_fffd_substantive': sum('\ufffd' in x['outputs'][cond] for x in brows.values() if not x['gold_is_abstention']),
                                             'grade_changes': differences}
                    row['splits'][split] = comparisons
                result.append(row)
    target = Path(__file__).with_name('generation_vs_grading.json')
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for row in result:
        print(row['model'], row['init'], 'W_equal', row['trained_vectors_equal'], 'loss_equal', row['losses_equal'],
              {s: {c: (v['output_mismatches'], len(v['grade_changes'])) for c, v in d.items()} for s, d in row['splits'].items()})


if __name__ == '__main__':
    main()
