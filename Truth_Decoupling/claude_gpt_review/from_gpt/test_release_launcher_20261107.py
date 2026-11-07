"""Execute the release launcher against a recording Docker substitute."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent.parent / 'from_claude/code/scripts/review_checks/e27_release_rerun.sh'


class ReleaseLauncherTests(unittest.TestCase):
    def run_chain(self, *, code=True, fail=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); commands = root / 'commands.jsonl'; bindir = root / 'bin'; bindir.mkdir()
            docker = bindir / 'docker'
            docker.write_text('#!/usr/bin/env python3\nimport os,sys,json\na=sys.argv[1:]\n'
                'phase=a[a.index("scripts/e27_qa_generation.py")+1]\n'
                'with open(os.environ["RECORDED_CALLS"],"a") as f:f.write(json.dumps(a)+"\\n")\n'
                'raise SystemExit(9 if phase==os.environ.get("FAIL_PHASE") else 0)\n')
            docker.chmod(0o755)
            env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                       RECORDED_CALLS=str(commands), TD_PROJECT_ROOT='/project with spaces',
                       TD_HOST_PROJECT_ROOT=str(root / 'host with spaces'), TD_E27_RESULT_TAG='new_validation')
            env.pop('TD_REVIEW_CODE', None)
            if code: env['TD_REVIEW_CODE'] = '/reviewed code with spaces'
            if fail: env['FAIL_PHASE'] = fail
            process = subprocess.run(['bash', str(SCRIPT), 'llama3.1-8b', '0'], env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []
            return process, calls

    def test_explicit_code_space_paths_and_freeze_before_every_test(self):
        result, calls = self.run_chain()
        self.assertEqual(result.returncode, 0, result.stderr)
        phases = [a[a.index('scripts/e27_qa_generation.py') + 1] for a in calls]
        self.assertEqual(phases, ['train', 'check', 'generate-tune', 'judge-tune', 'freeze', 'generate-test', 'judge-test'] * 5)
        self.assertIn('/reviewed code with spaces', calls[0])
        self.assertIn('HF_HOME=/project with spaces/hf_cache', calls[0])
        self.assertFalse(any(a.startswith('HOME=') for a in calls[0]))

    def test_missing_code_and_freeze_failure_do_not_reach_test_generation(self):
        result, calls = self.run_chain(code=False)
        self.assertNotEqual(result.returncode, 0); self.assertFalse(calls)
        self.assertIn('TD_REVIEW_CODE', result.stderr)
        result, calls = self.run_chain(fail='freeze')
        self.assertNotEqual(result.returncode, 0)
        phases = [a[a.index('scripts/e27_qa_generation.py') + 1] for a in calls]
        self.assertEqual(phases, ['train', 'check', 'generate-tune', 'judge-tune', 'freeze'])


if __name__ == '__main__':
    unittest.main()
