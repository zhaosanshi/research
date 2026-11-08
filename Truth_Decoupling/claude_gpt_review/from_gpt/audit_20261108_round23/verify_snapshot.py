"""Read-only audit of Claude's captured v6 reruns; makes no model/API calls."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--old-inputs', type=Path, required=True, help='R22 extracted runs/ directory')
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    code = Path(__file__).resolve().parents[2] / 'from_claude/code'
    sys.path[:0] = [str(code), str(code / 'scripts/review_checks')]
    import judge_two_axis_run as runner
    cfg = json.loads((code / 'configs/judges/two_axis_v6.json').read_text())
    manifest = json.loads(args.manifest.read_text())
    for name, record in manifest['files'].items():
        assert sha(args.inputs / name) == record['sha256'], name
    runtime = args.inputs / 'runtime/review_r23_claude/claude_gpt_review/from_claude/code'
    checked = []
    for source in sorted(runtime.rglob('*')):
        if source.is_file():
            rel = source.relative_to(runtime)
            assert source.read_bytes() == (code / rel).read_bytes(), rel
            checked.append(str(rel))
    result = {'scope': 'Frozen in-progress v6 snapshot, not completed/final grading',
              'captured_unix': manifest['captured_unix'],
              'runtime_files_matching_git': checked, 'runs': {}, 'new_calls': 0}
    ballots, cells_by_hash = {}, {}
    for name, old_name in [('oracle-tune-j6', 'oracle-tune'), ('select-tune-v2', 'select-tune')]:
        run = args.inputs / name
        meta, cells, _ = runner._workspace(str(run))
        pro, stats = runner._verify_rows(runner._rows(run / 'judge2/pro.jsonl'), cells, 'pro', cfg)
        assert not stats['stale'] and stats['legacy_backfilled'] == 0
        old = json.loads((args.old_inputs / old_name / 'tune.json').read_text())
        new = json.loads((run / 'tune.json').read_text())
        old_rows = {row['pair_id']: row for row in old['evaluations']}
        assert list(old_rows) == [row['pair_id'] for row in new['evaluations']]
        changes = {}
        for field in ['outputs', 'generation_token_ids']:
            changes[field] = {cond: sum(row[field][cond] != old_rows[row['pair_id']][field][cond]
                                        for row in new['evaluations']) for cond in new['conditions']}
        result['runs'][name] = {'n_cells': meta['n_cells'], 'n_questions': meta['n_questions'],
            'bound_pro_rows': len(pro), 'pro_binding_check': stats,
            'allenai_rows': len(runner._rows(run / 'judge2/allenai.jsonl')),
            'gpt_file_exists': (run / 'judge2/gpt.jsonl').exists(),
            'final_file_exists': (run / 'judge2/final.jsonl').exists(),
            'generation_changes_from_old': changes, 'generation_sha256': sha(run / 'tune.json')}
        ballots[name] = {row['input_sha']: row for row in pro.values()}
        cells_by_hash[name] = {cell['input_sha']: cell for cell in cells}
        if name == 'select-tune-v2':
            orientation = new['identity']['direction_orientation']['qadas']
            assert set(orientation) == {'28', '29', '30'}
            assert all(v['flipped'] and v['cosine_after'] > 0 for v in orientation.values())
            train = set(new['identity']['orientation_source']['estimation_pair_ids'])
            assert not train.intersection(old_rows)
            signed_errors, pick_checks = [], 0
            for row in new['evaluations']:
                before = old_rows[row['pair_id']]
                pool = [n for n in new['conditions'] if not row['rule_abstain'][n] and row['outputs'][n].strip()] or new['conditions']
                for cond in new['conditions']:
                    for layer in orientation:
                        key = 'coord_qadas_L' + layer
                        signed_errors.append(abs(row['readout'][cond][key] + before['readout'][cond][key]))
                for family, layers in new['identity']['available_read_layers_by_family'].items():
                    normalized = []
                    for layer in layers:
                        values = [row['readout'][n]['coord_%s_L%s' % (family, layer)] for n in pool]
                        mean = sum(values) / len(values)
                        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
                        normalized.append([(v - mean) / std if std > 1e-12 else 0.0 for v in values])
                    scores = [sum(v[i] for v in normalized) / len(normalized) for i in range(len(pool))]
                    pick = pool[max(range(len(pool)), key=lambda i: scores[i])]
                    assert row['picks']['coord_' + family + '_mean'] == pick, row['pair_id']
                    pick_checks += 1
            result['readout'] = {'qadas_orientation': orientation,
                'train_evaluation_overlap': 0, 'signed_coordinate_comparisons': len(signed_errors),
                'max_abs_new_plus_old_qadas_coordinate': max(signed_errors),
                'independently_recomputed_standardized_picks': pick_checks}
    names = list(ballots)
    common = ballots[names[0]].keys() & ballots[names[1]].keys()
    conflicts = []
    for identity in sorted(common):
        a, b = (ballots[n][identity] for n in names)
        if (a['truth'], a['info']) != (b['truth'], b['info']):
            cell = cells_by_hash[names[0]][identity]
            conflicts.append({'input_sha': identity, 'pair_id': cell['pair_id'],
                              'question': cell['question'], 'prediction': cell['prediction'],
                              'votes': {n: {'truth': ballots[n][identity]['truth'],
                                            'info': ballots[n][identity]['info'],
                                            'judge_contract_sha': ballots[n][identity]['judge_contract_sha']}
                                        for n in names}})
    result['partial_pro_cross_run'] = {'shared_complete_inputs': len(common), 'conflicts': conflicts,
        'interpretation': 'Pro ballots only; no GPT/final existed at capture. Do not call this a final-label conflict or infer which ballot is correct.'}
    for name, record in manifest['files'].items():
        assert sha(args.inputs / name) == record['sha256'], name
    result['captured_inputs_unchanged'] = True
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'runtime_files': len(checked), 'runs': result['runs'],
                      'partial_shared_pro_inputs': len(common), 'partial_pro_conflicts': len(conflicts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
