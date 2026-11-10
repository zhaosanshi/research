#!/usr/bin/env python3
"""Export verified E43 artifacts as one immutable, self-contained judge input.

The export contains the exact frozen cohort and every selected generation arm.
It validates source, model/prompt identity, frozen answers and direction files;
references never enter generation here. Default selection excludes pilot files,
while explicitly selecting a pilot or incomplete generation is an error. An
existing output is reusable only when its full content is identical.
"""
import argparse
import json
from pathlib import Path

import common  # noqa: F401
import e43_qa_prompt_das as E
from core.readout import rule_abstains
from core.io_utils import atomic_json_dump

EXPORT_PROTOCOL = 'e43_judge_generation_export_v2'


def read(path):
    def unique(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key in ' + str(path))
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def _record(rec, description):
    if not isinstance(rec.get('text'), str) or type(rec.get('finished')) is not bool:
        raise ValueError('invalid answer text/completion flag: ' + description)
    return rec['text'], rec['finished'], bool(rule_abstains(rec['text']))


def export(source_run, run_dir, split='tune', gen_names=None, name='judge_input'):
    if split not in ('train', 'tune'):
        raise ValueError('only train/tune splits may be exported')
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('output name must be a simple file stem')
    source_run, run_dir = Path(source_run).resolve(), Path(run_dir).resolve()
    gen_dir = (run_dir / 'gen').resolve()
    if gen_names is None:
        files = [path for path in sorted(gen_dir.glob('*_%s.json' % split)) if '.pilot.' not in path.name]
    else:
        files = []
        for value in gen_names:
            if not isinstance(value, str) or Path(value).name != value:
                raise ValueError('generation names must be files inside this run/gen directory')
            files.append(gen_dir / value)
    if len(set(files)) != len(files):
        raise ValueError('generation file selected more than once')
    for path in files:
        path.resolve().relative_to(gen_dir)
        if not path.is_file():
            raise ValueError('selected generation is not a file: ' + str(path))
    frozen_path = run_dir / ('frozen_%s.json' % split)
    source_paths = [source_run / 'bank.json', source_run / 'questions.json', frozen_path, *files]
    before = {str(path): E.file_sha(path) for path in source_paths}
    prepared = read(source_run / 'questions.json')
    questions = prepared['splits'][split]
    frozen = read(frozen_path)
    E.validate_frozen(frozen, questions, require_full=True)
    if frozen.get('completed') is not True or frozen.get('pilot') is not False or frozen['split'] != split:
        raise ValueError('only an explicitly complete non-pilot freeze can be exported')
    expected_sources = {key: before[str(source_run / key)] for key in ('bank.json', 'questions.json')}
    if frozen['identity'].get('source_files_sha256') != expected_sources:
        raise ValueError('frozen answers use different source questions or bank')
    if frozen.get('model') != frozen['identity'].get('model') or frozen.get('model') != read(source_run / 'bank.json').get('model'):
        raise ValueError('frozen model identity differs from the source run')
    generations = []
    for path in files:
        data = read(path)
        if data.get('completed') is not True or data.get('pilot') is not False:
            raise ValueError('pilot/incomplete generations cannot enter a full judge export: ' + path.name)
        direction = data.get('direction_file')
        if not isinstance(direction, str) or Path(direction).name != direction:
            raise ValueError('invalid generation direction path')
        dp = run_dir / 'dirs' / direction
        before[str(dp)] = E.file_sha(dp)
        E.validate_generation(data, frozen, questions, run_dir=run_dir)
        generations.append(data)
    conditions = ['baseline_' + prompt for prompt in E.PROMPT_NAMES]
    specs = [{'name': 'baseline_' + prompt, 'base_condition': 'baseline_' + prompt,
              'kind': 'prompt_control', 'uses_reference_answer': False} for prompt in E.PROMPT_NAMES]
    by_generation = []
    for data in generations:
        condition = '%s_L%d_%s' % (data['family'], data['layer'], data['arm'])
        conditions.append(condition)
        specs.append({'name': condition, 'base_condition': 'baseline_' + data['base_prompt'],
                      'family': data['family'], 'layer': data['layer'], 'arm': data['arm'],
                      'kind': 'none' if data['arm'] == 'none' else 'intervention', 'uses_reference_answer': False})
        by_generation.append({row['pair_id']: row for row in data['rows']})
    if len(set(conditions)) != len(conditions):
        raise ValueError('duplicate generation conditions')
    rows = []
    for question, fq in zip(questions, frozen['questions']):
        pid = question['pair_id']
        outputs, finished, rule = {}, {}, {}
        for prompt in E.PROMPT_NAMES:
            condition = 'baseline_' + prompt
            outputs[condition], finished[condition], rule[condition] = _record(fq['answers'][prompt], condition)
        for data, lookup in zip(generations, by_generation):
            condition = '%s_L%d_%s' % (data['family'], data['layer'], data['arm'])
            row = lookup[pid]
            if row.get('baseline_text') != fq['answers'][data['base_prompt']]['text']:
                raise ValueError('generation comparison baseline differs from frozen answer: ' + condition)
            outputs[condition], finished[condition], rule[condition] = _record(row, condition)
        rows.append({'pair_id': pid, 'question': question['question'], 'gold': list(question['gold']),
                     'incorrect': list(question.get('incorrect') or []), 'best_answer': question.get('best_answer'),
                     'outputs': outputs, 'finished': finished, 'rule_abstain': rule})
    payload = {'completed': True, 'model': frozen['model'], 'protocol': frozen['protocol'], 'export_protocol': EXPORT_PROTOCOL,
               'split': split, 'identity': frozen['identity'], 'identity_sha256': frozen['identity_sha256'],
               'conditions': conditions, 'condition_specs': specs, 'question_ids': [q['pair_id'] for q in questions],
               'sources': {'frozen': frozen_path.name, 'gen': [path.name for path in files], 'sha256': before},
               'evaluations': rows}
    # Validate the exact downstream schema before committing an export.
    from scripts.review_checks.judge_two_axis_run import _derive_cells
    _derive_cells(payload)
    for path, digest in before.items():
        if E.file_sha(path) != digest:
            raise RuntimeError('source artifact changed during judge export: ' + path)
    path = run_dir / ('judge_%s' % split) / (name + '.json')
    if path.exists():
        if read(path) != payload:
            raise ValueError('existing judge input differs; use a new --name or run directory, never overwrite judged inputs')
    else:
        atomic_json_dump(payload, str(path))
    return {'path': str(path), 'n_questions': len(rows), 'conditions': conditions, 'export_protocol': EXPORT_PROTOCOL}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--split', default='tune', choices=('train', 'tune'))
    p.add_argument('--gen', nargs='*', default=None)
    p.add_argument('--name', default='judge_input')
    args = p.parse_args()
    print(json.dumps(export(args.source_run, args.run_dir, args.split, args.gen, args.name), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
