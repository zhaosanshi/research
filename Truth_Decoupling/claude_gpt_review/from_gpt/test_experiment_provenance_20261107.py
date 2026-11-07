"""Mutating a dependency must invalidate an unchanged entry script."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'from_claude/code'))
from core.experiment_provenance import DEPENDENCY_SCRIPTS, implementation_identity, require_unchanged


class DependencyIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'core').mkdir(); (self.root / 'scripts').mkdir()
        (self.root / 'core/intervene.py').write_text('version = 1\n')
        for name in DEPENDENCY_SCRIPTS:
            (self.root / 'scripts' / name).write_text('version = 1\n')
        self.entry = self.root / 'scripts/run.py'; self.entry.write_text('version = 1\n')

    def test_hidden_dependency_change_rejected_with_entry_unchanged(self):
        old = implementation_identity(self.entry, self.root)
        (self.root / 'core/intervene.py').write_text('version = 2\n')
        new = implementation_identity(self.entry, self.root)
        self.assertEqual(old['files']['scripts/run.py'], new['files']['scripts/run.py'])
        self.assertNotEqual(old['files_fingerprint'], new['files_fingerprint'])
        with self.assertRaisesRegex(RuntimeError, 'implementation changed'):
            require_unchanged(old, self.entry, self.root)

    def test_e26_entry_dependency_and_new_core_file_are_included(self):
        old = implementation_identity(self.entry, self.root)
        (self.root / 'scripts/e26_matched_coverage.py').write_text('version = 2\n')
        with self.assertRaises(RuntimeError): require_unchanged(old, self.entry, self.root)
        new = implementation_identity(self.entry, self.root)
        (self.root / 'core/new_operator.py').write_text('version = 1\n')
        with self.assertRaises(RuntimeError): require_unchanged(new, self.entry, self.root)

    def test_complete_snapshot_matches_and_missing_dependency_fails(self):
        old = implementation_identity(self.entry, self.root)
        require_unchanged(old, self.entry, self.root)
        (self.root / 'scripts/e26_matched_coverage.py').unlink()
        with self.assertRaises(FileNotFoundError): implementation_identity(self.entry, self.root)


if __name__ == '__main__':
    unittest.main()
