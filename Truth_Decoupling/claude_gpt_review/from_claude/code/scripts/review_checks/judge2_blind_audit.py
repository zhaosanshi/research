#!/usr/bin/env python3
"""Content-bound independent blind-review packets (v2).

sample RUN_DIR ... --out DIR --tag TAG [--n-per-run 25] [--seed 1] [--exclude FILE ...]
  Requires verified machine adjudication, possibly with unresolved cells. Excludes known prior human decisions and
  explicitly exposed content. The reviewer receives only TAG_sheet.md, never the key/manifest with model ballots.
apply DIR --tag TAG --verdicts '1A 2B(H) 3C' --by WHO
  First validates the entire packet and all requested verdicts. Records an immutable blind-verdict ledger, then
  propagates those real verdicts through human_set. An interrupted propagation can be retried with the same verdicts.
report DIR --tag TAG
  Uses only this packet's recorded blind verdicts, compared with frozen, verified pre-human model decisions. Ordinary
  arbitration records are never treated as independent blind labels. Machine/input changes require a new packet.

Old packets without a v2 manifest are not silently upgraded; inspect their originals or export a new unexposed sample.
No software can prove the reviewer never saw an answer elsewhere: exclusion coverage and that limitation are recorded.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from core.io_utils import read_jsonl, write_jsonl, atomic_json_dump
from core.judge_two_axis import TRUTH_VALUES, allenai_votes, judge_own_label, coarse, normalize_human_verdict
from core.judge_two_axis_contract import cell_sha, canonical_sha, file_sha, CELL_FIELDS
from scripts.review_checks import judge_two_axis_run as R

VERSION = 'qa_blind_audit_v2'
VERDICT = re.compile(r"^(\d+)([ABCH])(?:\(([SHR])\))?$")
SOURCE_NAMES = ('meta.json', 'cells.jsonl', 'allenai.jsonl', 'allenai_meta.json',
                'pro.jsonl', 'pro_meta.json', 'gpt.jsonl', 'gpt_meta.json')


def _content_key(question, prediction):
    return question.strip(), prediction.strip()


def _paths(out_dir, tag):
    if not tag or Path(tag).name != tag:
        raise ValueError('tag must be a simple filename component')
    return {name: Path(out_dir) / (tag + suffix) for name, suffix in
            [('key', '_key.jsonl'), ('sheet', '_sheet.md'), ('manifest', '_manifest.json'), ('verdicts', '_verdicts.json')]}


def _sources(run_dir):
    return {name: file_sha(Path(run_dir) / 'judge2' / name) if (Path(run_dir) / 'judge2' / name).exists() else None
            for name in SOURCE_NAMES}


def _load_exclusions(paths):
    keys = set()
    for path in paths or []:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict) and 'cells' in data and 'cell_ids' in data:
            by_id = {c['cell_id']: c for c in read_jsonl(data['cells'])}
            if any(cid not in by_id for cid in data['cell_ids']):
                raise ValueError('exclusion references an unknown cell')
            items = [by_id[cid] for cid in data['cell_ids']]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError('invalid exclusion file: ' + str(path))
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('question'), str) or not isinstance(item.get('prediction'), str):
                raise ValueError('every exclusion must bind a question and complete answer')
            keys.add(_content_key(item['question'], item['prediction']))
    return keys


def sample(run_dirs, out_dir, tag, n_per_run, seed, exclude_paths, config=None):
    if type(n_per_run) is not int or n_per_run <= 0:
        raise ValueError('n_per_run must be a positive integer')
    paths = _paths(out_dir, tag)
    if any(p.exists() for p in paths.values()):
        raise ValueError('packet already exists; do not overwrite a sheet that may have been reviewed')
    run_dirs = [str(Path(p).resolve()) for p in run_dirs]
    if len(run_dirs) != len(set(run_dirs)):
        raise ValueError('duplicate run directory')
    cfg = R.load_cfg(config)
    excluded = _load_exclusions(exclude_paths)
    explicit_count = len(excluded)
    workspaces, snapshots = {}, {}
    code_before = R._aggregation_code()
    # All run inputs/ballots are validated before sampling; known arbitration is excluded from every run's pool.
    for run_dir in run_dirs:
        meta, cells, _ = R._workspace(run_dir)
        generation = Path(run_dir) / meta['generation']
        _, _, finals = R.load_verified_final(run_dir, generation, require_complete=False)
        judges, _ = R._load_judges(run_dir, cells, cfg, required=())
        humans = R._human_verdicts(run_dir, cells)
        by_id = {c['cell_id']: c for c in cells}
        for cid in humans:
            c = by_id[cid]
            excluded.add(_content_key(c['question'], c['prediction']))
        workspaces[run_dir] = cells, judges, finals
        snapshots[run_dir] = {'sources': _sources(run_dir), 'generation': meta['generation'],
                              'generation_sha256': file_sha(generation),
                              'pre_human_final_sha256': file_sha(Path(run_dir) / 'judge2/final.jsonl')}
    rng = random.Random(seed)
    chosen, seen = [], set()
    for run_dir in run_dirs:
        cells, judges, finals = workspaces[run_dir]
        pool = {}
        for cell in sorted(cells, key=lambda c: c['cell_id']):
            key = _content_key(cell['question'], cell['prediction'])
            if not cell['gold_is_abstention'] and cell['prediction'].strip() and key not in excluded and key not in seen:
                pool.setdefault(key, cell)
        for cell in rng.sample(list(pool.values()), min(n_per_run, len(pool))):
            seen.add(_content_key(cell['question'], cell['prediction']))
            cid = cell['cell_id']
            chosen.append({'run_dir': run_dir, 'cell_id': cid, 'input_sha': cell_sha(cell),
                           'reviewed_input': {key: cell[key] for key in CELL_FIELDS},
                           'machine_rows': {name: rows.get(cid) for name, rows in judges.items()},
                           'pipeline_before_human': finals[cid]})
    rng.shuffle(chosen)
    for number, item in enumerate(chosen, 1):
        item['number'] = number
    lines = ['# 盲评抽检 %s（%d条）' % (tag, len(chosen)), '',
             '只看题目、参考和回答；A真且有信息，B含假信息（需B(S)有信息、B(H)含糊或B(R)拒答），C真实拒答，H真实但含糊。', '',
             '| # | 题 | 参考答案 | 算错的答案 | 回答 |', '|---|---|---|---|---|']
    def esc(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    for item in chosen:
        c = item['reviewed_input']
        lines.append('| %d | %s | %s | %s | %s |' % (item['number'], esc(c['question']),
            esc(' / '.join(c['gold'])), esc(' / '.join(c['incorrect'])), esc(c['prediction'])))
    for run_dir, frozen in snapshots.items():
        if _sources(run_dir) != frozen['sources'] or file_sha(Path(run_dir) / frozen['generation']) != frozen['generation_sha256']:
            raise RuntimeError('grading inputs changed while exporting blind sample')
    if R._aggregation_code() != code_before:
        raise RuntimeError('grading code changed while exporting blind sample')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_jsonl(chosen, str(paths['key']))
    paths['sheet'].write_text('\n'.join(lines) + '\n')
    manifest = {'version': VERSION, 'config': cfg, 'aggregation_code': code_before, 'runs': snapshots,
                'key_sha256': file_sha(paths['key']), 'sheet_sha256': file_sha(paths['sheet']),
                'n': len(chosen), 'seed': seed, 'n_per_run_requested': n_per_run,
                'explicit_excluded_contents': explicit_count, 'total_known_excluded_contents': len(excluded),
                'limitation': 'Only known prior human records and explicit exclusions are screened; external reviewer exposure is not mechanically provable.'}
    atomic_json_dump(manifest, str(paths['manifest']))
    return {'n': len(chosen), 'per_run': dict(Counter(i['run_dir'] for i in chosen)),
            'excluded_contents': len(excluded), 'key': str(paths['key']), 'sheet': str(paths['sheet'])}


def parse_verdicts(text):
    out = {}
    for token in text.replace(',', ' ').split():
        match = VERDICT.fullmatch(token)
        if not match:
            raise ValueError('invalid verdict %r; expected 12A or 7B(H)' % token)
        number, letter, subtype = int(match[1]), match[2], match[3]
        if (letter == 'B' and subtype is None) or (letter != 'B' and subtype is not None):
            raise ValueError('only B requires an explicit S/H/R subtype')
        if number in out:
            raise ValueError('duplicate verdict number')
        out[number] = letter, subtype
    if not out:
        raise ValueError('no verdicts provided')
    return out


def _load_packet(out_dir, tag):
    paths = _paths(out_dir, tag)
    if not paths['manifest'].exists():
        raise ValueError('legacy/unbound blind packet: re-export verified, unexposed inputs; never infer its identity')
    manifest = R._read_json(paths['manifest'])
    if manifest.get('version') != VERSION or file_sha(paths['key']) != manifest.get('key_sha256') or file_sha(paths['sheet']) != manifest.get('sheet_sha256'):
        raise ValueError('blind sheet/key differs from its immutable manifest')
    if manifest['aggregation_code'] != R._aggregation_code():
        raise ValueError('grading implementation changed since sampling; use its frozen runtime or export a new packet')
    current = {}
    for run_dir, frozen in manifest['runs'].items():
        meta, cells, _ = R._workspace(run_dir)
        if _sources(run_dir) != frozen['sources'] or file_sha(Path(run_dir) / frozen['generation']) != frozen['generation_sha256']:
            raise ValueError('blind input or frozen machine ballots changed since sampling')
        R._load_judges(run_dir, cells, manifest['config'], required=())
        current[run_dir] = {cell['cell_id']: cell for cell in cells}
    key = {}
    for item in read_jsonl(str(paths['key'])):
        number = item.get('number')
        if type(number) is not int or number <= 0 or number in key:
            raise ValueError('invalid/duplicate blind item number')
        cell = current.get(item['run_dir'], {}).get(item['cell_id'])
        if cell is None or cell_sha(item['reviewed_input']) != item['input_sha'] or cell_sha(cell) != item['input_sha']:
            raise ValueError('blind verdict would refer to a different answer or references')
        key[number] = item
    if len(key) != manifest['n']:
        raise ValueError('blind sample count changed')
    return paths, manifest, key


def _ledger(paths, key):
    rows = R._read_json(paths['verdicts']) if paths['verdicts'].exists() else []
    if not isinstance(rows, list):
        raise ValueError('invalid blind-verdict ledger')
    by_number = {}
    package_sha = file_sha(paths['manifest'])
    for record in rows:
        number = record.get('number')
        item = key.get(number)
        if (item is None or number in by_number or record.get('version') != VERSION or record.get('package_sha256') != package_sha
                or record.get('input_sha') != item['input_sha'] or record.get('reviewed_input') != item['reviewed_input']
                or not isinstance(record.get('by'), str) or not record['by'].strip()):
            raise ValueError('blind verdict lacks this packet/content/reviewer identity')
        normalized = normalize_human_verdict(record['decision'])
        if normalized != record['decision']:
            raise ValueError('incomplete blind verdict axes')
        by_number[number] = record
    return by_number


def apply(out_dir, tag, verdicts_text, by, note=None):
    if not isinstance(by, str) or not by.strip():
        raise ValueError('record the actual reviewer')
    paths, manifest, key = _load_packet(out_dir, tag)
    verdicts = parse_verdicts(verdicts_text)
    if set(verdicts) - set(key):
        raise ValueError('verdict includes an item not on this sheet')
    existing = _ledger(paths, key)
    prepared = dict(existing)
    for number, (letter, subtype) in verdicts.items():
        item = key[number]
        decision = normalize_human_verdict({'label': letter, 'info_subtype': subtype, 'by': by})
        record = {'version': VERSION, 'package_sha256': file_sha(paths['manifest']), 'number': number,
                  'input_sha': item['input_sha'], 'reviewed_input': item['reviewed_input'], 'by': by,
                  'decision': decision, 'note': note or ('blind audit %s #%d' % (tag, number))}
        if number in existing and existing[number] != record:
            raise ValueError('blind verdict already recorded; do not silently overwrite independent ratings')
        if number not in existing:
            # A known assisted decision added after sample export also invalidates independence.
            humans = R._human_verdicts(item['run_dir'])
            if item['cell_id'] in humans:
                raise ValueError('item acquired a prior human decision; it is no longer an unexposed blind sample')
        prepared[number] = record
    # No writes until every number, content, source, axis and prior decision has passed.
    _load_packet(out_dir, tag)
    atomic_json_dump([prepared[n] for n in sorted(prepared)], str(paths['verdicts']))
    for number in sorted(verdicts):
        item, record = key[number], prepared[number]
        d = record['decision']
        R.human_set(item['run_dir'], item['cell_id'], d['label'], by, record['note'],
                    truth=d['truth'], info_subtype=d['info_subtype'], expected_input_sha=item['input_sha'])
    return {'recorded': len(verdicts), 'missing_items': sorted(set(key) - set(prepared)),
            'ledger': str(paths['verdicts'])}


def report(out_dir, tag):
    paths, manifest, key = _load_packet(out_dir, tag)
    ratings = _ledger(paths, key)
    pairs = {name: [] for name in ('allenai', 'pro', 'gpt', 'pipeline_before_human')}
    for number, rating in ratings.items():
        item, human = key[number], rating['decision']['label']
        for name, row in item['machine_rows'].items():
            if not row:
                continue
            if name == 'allenai':
                truth, info = allenai_votes(row, manifest['config']['allenai']['threshold'])
                own = judge_own_label(truth, info)
                if own is not None:
                    pairs[name].append((coarse(own), coarse(human)))
            else:
                own = judge_own_label(row.get('truth') if row.get('truth') in TRUTH_VALUES else None,
                                      row.get('info') if row.get('info') in ('S', 'H', 'R') else None)
                if own is not None:
                    pairs[name].append((own, human))
        rec = item['pipeline_before_human']
        if rec.get('status') == 'final' and rec.get('label') is not None:
            pairs['pipeline_before_human'].append((rec['label'], human))
    out = {'version': VERSION, 'n_items': len(key), 'n_with_blind_verdict': len(ratings),
           'missing_items': sorted(set(key) - set(ratings)), 'machine_decisions': 'verified and frozen before this blind review',
           'limitation': manifest['limitation']}
    for name, values in pairs.items():
        out[name] = dict(R._rate(values), coarse_labels=name == 'allenai',
                         coverage_of_blind_ratings=len(values) / len(ratings) if ratings else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('sample'); p.add_argument('run_dirs', nargs='+'); p.add_argument('--out', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--n-per-run', type=int, default=25); p.add_argument('--seed', type=int, default=1); p.add_argument('--exclude', nargs='*', default=[]); p.add_argument('--config')
    p = sub.add_parser('apply'); p.add_argument('out'); p.add_argument('--tag', required=True); p.add_argument('--verdicts', required=True); p.add_argument('--by', required=True); p.add_argument('--note')
    p = sub.add_parser('report'); p.add_argument('out'); p.add_argument('--tag', required=True)
    a = ap.parse_args()
    if a.cmd == 'sample':
        result = sample(a.run_dirs, a.out, a.tag, a.n_per_run, a.seed, a.exclude, a.config)
    elif a.cmd == 'apply':
        result = apply(a.out, a.tag, a.verdicts, a.by, a.note)
    else:
        result = report(a.out, a.tag)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
