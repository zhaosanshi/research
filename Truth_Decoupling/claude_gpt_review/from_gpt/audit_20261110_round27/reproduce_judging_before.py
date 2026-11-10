"""Reproduce the reviewed 44b57ac single/export bugs without API/model calls.

Run from a checkout with commit 44b57ac available. Only the two audited old files
are loaded into temporary modules; current shared judge fixtures create valid
synthetic inputs. The source file bytes are recorded in the output.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent.parent
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'scripts'), str(CODE / 'scripts/review_checks'),
               str(REVIEW / 'from_gpt/audit_20261108_round22/pipeline')]
from test_two_axis_pipeline_r22 import setup, generation, FakeJudge


def old_module(repo, relative, destination, module_name):
    content = subprocess.check_output(['git', 'show', '44b57ac:Truth_Decoupling/claude_gpt_review/from_claude/code/' + relative], cwd=repo)
    destination.write_bytes(content)
    spec = importlib.util.spec_from_file_location(module_name, destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'CODE_ROOT'):
        module.CODE_ROOT = str(CODE)
    return module, hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=REVIEW.parents[1])
    parser.add_argument('--output', type=Path, default=HERE / 'judging_before.json')
    args = parser.parse_args()
    report = {'reviewed_commit': '44b57ac', 'scope': 'Offline synthetic reproductions, not evidence that real experiments were polluted',
              'new_calls': 0, 'source_sha256': {}}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        R, digest = old_module(args.repo, 'scripts/review_checks/judge_two_axis_run.py', root / 'old_judge.py', 'r27_old_judge')
        report['source_sha256']['judge_two_axis_run.py'] = digest
        cfg = R.load_cfg()
        run = setup(root / 'single')
        R.run_llm(str(run), 'pro', cfg, judge=FakeJudge('pro', cfg))
        try:
            R.single_judge(str(run), 'pro', cfg)
        except NameError as exc:
            report['valid_pro_export_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        else:
            raise AssertionError('reviewed Pro path unexpectedly succeeded')
        # Repair only the missing name to expose the next independent defect.
        R.INFO_SUBTYPES = ('S', 'H', 'R')
        raw = generation()
        raw['evaluations'][0]['outputs'] = {condition: '' for condition in raw['conditions']}
        run = setup(root / 'empty', raw)
        R.run_llm(str(run), 'pro', cfg, judge=FakeJudge('pro', cfg))
        R.single_judge(str(run), 'pro', cfg)
        report['empty_placeholder_claimed_as_judge_vote'] = R._rows(run / 'judge2/single_pro.jsonl')['q0|baseline']
        report['single_has_verifier'] = hasattr(R, 'load_verified_single')
        X, digest = old_module(args.repo, 'scripts/e43_export_judge.py', root / 'old_export.py', 'r27_old_export')
        report['source_sha256']['e43_export_judge.py'] = digest
        source, run = root / 'source', root / 'export'
        source.mkdir()
        (run / 'gen').mkdir(parents=True)
        (source / 'questions.json').write_text(json.dumps({'splits': {'tune': [
            {'pair_id': 'q1', 'question': 'NEW QUESTION', 'gold': ['new gold'], 'incorrect': []}]}}))
        frozen = {'completed': True, 'pilot': False, 'model': 'm', 'protocol': 'p', 'identity_sha256': 'same',
                  'questions': [{'pair_id': 'q1', 'question': 'OLD QUESTION',
                                 'answers': {'x': {'text': 'old answer', 'finished': True}}}]}
        (run / 'frozen_tune.json').write_text(json.dumps(frozen))
        generated = {'completed': True, 'pilot': False, 'split': 'tune', 'identity_sha256': 'same',
                     'family': 'E', 'layer': 0, 'arm': 'main', 'rows': [
                         {'pair_id': 'q1', 'text': 'old steered answer', 'finished': True},
                         {'pair_id': 'foreign', 'text': 'ignored', 'finished': True}]}
        (run / 'gen/E_L0_main_tune.json').write_text(json.dumps(generated))
        argv = ['e43_export_judge.py', '--source-run', str(source), '--run-dir', str(run)]
        with patch.object(sys, 'argv', argv):
            X.main()
        output = run / 'judge_tune/judge_input.json'
        data = json.loads(output.read_text())
        report['export_mismatched_question_and_extra_row'] = {
            'original_question': 'OLD QUESTION', 'exported_question': data['evaluations'][0]['question'],
            'exported_outputs': data['evaluations'][0]['outputs'],
            'generation_rows': len(generated['rows']), 'exported_rows': len(data['evaluations'])}
        output.write_text('{}')
        with patch.object(sys, 'argv', argv):
            X.main()
        report['export_overwrites_existing_input'] = json.loads(output.read_text())['completed'] is True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
