#!/usr/bin/env python3
"""Run all distinct root/round19/round20/round21 regression cases and record source bytes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import unittest

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent
CODE = TEST_ROOT.parent / 'from_claude/code'
sys.path[:0] = [str(TEST_ROOT), str(CODE / 'scripts'), str(CODE)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    tests = list(TEST_ROOT.glob('test*.py'))
    for directory in ('audit_20261106_round19', 'audit_20261107_round20', 'audit_20261107_round21'):
        tests.extend((TEST_ROOT / directory).rglob('test*.py'))
    return sorted(p for p in tests if '__pycache__' not in p.parts and '/code_snapshot/' not in str(p))


def flatten(suite):
    for case in suite:
        if isinstance(case, unittest.TestSuite):
            yield from flatten(case)
        else:
            yield case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tests = files()
    def source_snapshot():
        source_files = {p for p in CODE.rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.json')}
        source_files.update(files())
        source_files.add(Path(__file__).resolve())
        return {str(p.relative_to(TEST_ROOT.parent.resolve())): sha(p) for p in sorted(p.resolve() for p in source_files)}
    before = source_snapshot()
    seen_modules, cases = {}, {}
    for path in tests:
        name = path.stem
        if name in seen_modules and seen_modules[name] != path:
            raise RuntimeError('duplicate test module name: ' + name)
        seen_modules[name] = path
        sys.path.insert(0, str(path.parent))
        try:
            if name in sys.modules:
                module = sys.modules[name]
                if Path(module.__file__).resolve() != path.resolve():
                    raise RuntimeError('test module loaded from another checkout: ' + name)
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            for case in flatten(unittest.defaultTestLoader.loadTestsFromModule(module)):
                if case.__class__.__module__ == module.__name__:
                    cases.setdefault(case.id(), case)
        finally:
            # Test modules may insert their own dependency directories. Remove
            # our entry rather than popping whichever path they added last.
            sys.path.remove(str(path.parent))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases[k] for k in sorted(cases)))
    after = source_snapshot()
    stable = before == after
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'tests_run': result.testsRun, 'distinct_test_ids': sorted(cases),
              'failures': [case.id() for case, _ in result.failures],
              'errors': [case.id() for case, _ in result.errors],
              'skipped': [(case.id(), reason) for case, reason in result.skipped],
              'source_unchanged_during_test': stable, 'source_sha256': after,
              'success': result.wasSuccessful() and stable}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
