"""Reproduce 77ad5b1 parser/retry behavior against the repaired implementation.

Both versions use synthetic responses only. No credentials, model, or network.
The historic sources are loaded in isolated modules; current production files
and all experiment artifacts remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / 'from_claude/code'
sys.path[:0] = [str(CODE), str(HERE)]
from core import judge_two_axis as current
from test_semantics_regressions import Responses


def historical(repo, commit, rel, name):
    source = subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=repo)
    module = types.ModuleType(name)
    module.__package__ = 'core'
    module.__file__ = name + '.py'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def grade_three_starts(module):
    out = []
    with tempfile.TemporaryDirectory() as directory:
        for _ in range(3):
            cfg = {'name': 'pro', 'model': 'offline-model', 'api_key': 'offline-fixture',
                   'base_url': 'https://offline.example.invalid/v1', 'max_tokens': 16,
                   'temperature': 0., 'sleep': 0., 'max_retries': 1}
            judge = module.TwoAxisLLMJudge(cfg, directory, name='pro')
            client = Responses()
            judge.llm._client = client
            row = judge.grade('q', ['gold'], 'prediction')
            out.append({'client_calls': len(client.calls), 'truth': row['truth'],
                        'truth_malformed_persistent': row['truth_malformed_persistent']})
    return out


def summarize(module):
    truth_examples = ['Truthfulness: F', 'Answer: **F**', 'T or FALSE', 'T - Final verdict: FALSE',
                      'T - (**F**)', 'T: TRUTHFUL - asserts nothing false',
                      'F\n\nThe predicted answer asserts a false claim.']
    info_examples = ['S or HEDGE', 'S - Final verdict: REFUSAL', 'H - HEDGE']
    cell = {'cell_id': 'q|c', 'question': 'Capital of France?', 'prediction': 'Paris.'}
    routes = {}
    for pred in ('', "I don't know.", 'Paris.'):
        r = module.adjudicate_cell({**cell, 'prediction': pred},
                                  pro={'truth': 'D', 'info': 'R'})
        routes[pred] = {k: r[k] for k in ('label', 'status', 'source')}
    contradictory = module.adjudicate_cell(cell, allenai={'truth_p_yes': .9, 'info_p_yes': .9},
        pro={'truth': module.parse_truth('T - Final verdict: FALSE'), 'info': 'S'})
    mixed = module.adjudicate_cell(cell, allenai={'truth_p_yes': .9, 'info_p_yes': .9},
        pro={'truth': 'F', 'info': 'H'}, gpt={'truth': None, 'info': 'D'})
    return {'truth_parse': {s: module.parse_truth(s) for s in truth_examples},
            'info_parse': {s: module.parse_info(s) for s in info_examples},
            'Pro_D_routes': routes, 'contradictory_reply_final': {'label': contradictory['label'], 'status': contradictory['status']},
            'mixed_GPT_D_and_missing_status': mixed['status'],
            'malformed_three_fresh_clients': grade_three_starts(module)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=HERE.parents[3])
    ap.add_argument('--output', type=Path, default=HERE / 'semantics_reproduction.json')
    args = ap.parse_args()
    prefix = 'Truth_Decoupling/claude_gpt_review/from_claude/code/core/'
    old_llm, sha_llm = historical(args.repo, '77ad5b1', prefix + 'judge.py', 'core._r24_old_llm')
    old_two, sha_two = historical(args.repo, '77ad5b1', prefix + 'judge_two_axis.py', 'core._r24_old_two')
    old_two.LLMJudge = old_llm.LLMJudge
    result = {'baseline_commit': '77ad5b1', 'baseline_source_sha256': {'judge.py': sha_llm, 'judge_two_axis.py': sha_two},
              'before': summarize(old_two), 'after': summarize(current),
              'new_real_API_requests': 0, 'new_human_verdicts': 0,
              'scope': 'Controlled offline execution; malformed examples are adversarial fixtures, not a count of observed production contamination.'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
