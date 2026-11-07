"""Archive complete source experiments without copying credentials or mutable caches."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def include(path):
    if not path.is_file() or '__pycache__' in path.parts or 'judge_cache' in path.parts:
        return False
    if path.name.endswith('_grading_input.json'):
        return False  # Complete graded rows are already in *_scored_items.json.
    return path.suffix in ('.json', '.pt', '.py', '.log', '.exit') or path.name in (
        'exit_code', 'judge_exit', 'judge_v3_exit', 'judge_v4_exit', 'train_exit', 'check_exit', 'generate_exit')


def complete(run, stem):
    raw = json.loads((run / (stem + '.json')).read_text())
    scored = json.loads((run / (stem + '_scored_items.json')).read_text())
    summary = json.loads((run / (stem + '_scored.json')).read_text())
    assert raw['completed'] and scored['completed'] and scored['grading_completed'] and summary['completed']
    assert scored['grading_source_generation_sha256'] == summary['source_generation_sha256'] == sha(run / (stem + '.json'))


def pack(root, output, name, directories, extras=()):
    files = sorted({p for directory in directories for p in directory.rglob('*') if include(p)} | set(extras))
    assert files
    manifest = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    archive = output / (name + '.tar.xz')
    with tarfile.open(archive, 'w:xz', preset=8) as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
    with tarfile.open(archive) as tar:
        got = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers() if m.isfile()}
    assert got == {r['path']: r['sha256'] for r in manifest}
    result = {'archive': archive.name, 'archive_sha256': sha(archive), 'bytes': archive.stat().st_size,
              'files': manifest, 'verified_after_archive': True,
              'excluded': ['mutable judge caches', 'duplicate derived grading_input JSON', 'Python bytecode']}
    (output / (name + '_manifest.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(name, len(files), archive.stat().st_size, 'bytes; every archived file verified', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('original', 'layers', 'preference', 'completion'), required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    for model in ('llama3.1-8b', 'qwen3-8b'):
        if args.stage == 'original':
            five = args.root / 'results_r20_source_codex_20261107' / model
            six = args.root / 'results_r20_source_train_codex_20261107' / model
            complete(five / 'pilot3_v2', 'check'); complete(five / 'tune192', 'tune')
            complete(six / 'full92_ep3', 'tune')
            # The three-pair training pilot was inspected, then kept as raw
            # execution evidence; it was not used as a performance experiment.
            assert json.loads((six / 'pilot3/check.json').read_text())['completed']
            extras = [args.root / 'results_qa_das_v1u' / model / 'full/tune.json',
                      args.root / 'results_qa_das_v1u' / model / 'full/tune_scored_items.json']
            pack(args.root, args.output, 'source_original_layer_' + model,
                 [five / 'pilot3_v2', five / 'tune192', six / 'pilot3', six / 'full92_ep3'], extras)
        elif args.stage == 'layers':
            layers = [15, 23] if model.startswith('llama') else [17, 26]
            for layer in layers:
                base = args.root / 'results_r20_source_layers_codex_20261107' / model / ('L' + str(layer))
                for directory, stem in [('controls_pilot3', 'check'), ('train_pilot3', 'check'),
                                        ('controls_full', 'tune'), ('train_full92_ep3', 'tune')]:
                    complete(base / directory, stem)
                pack(args.root, args.output, 'source_layer_' + model + '_L' + str(layer), [base])
        elif args.stage == 'preference':
            base = args.root / 'results_e34_codex_20261107' / model
            complete(base / 'pilot3', 'check')
            complete(base / 'full92_ep3', 'check')
            complete(base / 'full92_ep3', 'tune')
            # Final E32 artifacts are part of the scientific delivery too.
            assert (base / 'full92_ep3/tune_simple_scores.json').exists()
            assert (base / 'full92_ep3/tune_matched_coverage.json').exists()
            pack(args.root, args.output, 'source_pair_preference_' + model,
                 [base / 'pilot3', base / 'full92_ep3'])
        else:
            base = args.root / 'results_e34_completion_codex_20261107' / model
            assert json.loads((base / 'pilot_probe/pilot.json').read_text())['completed']
            assert json.loads((base / 'uniform256/pilot.json').read_text())['completed']
            if model == 'qwen3-8b': complete(base / 'uniform256', 'tune')
            else: assert not (base / 'uniform256/tune.json').exists(), 'Do not present failed Llama pilot as full completion'
            pack(args.root, args.output, 'natural_completion_256_' + model,
                 [base / 'pilot_probe', base / 'uniform256'])


if __name__ == '__main__': main()
