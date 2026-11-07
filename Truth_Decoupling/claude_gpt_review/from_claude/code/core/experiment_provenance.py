"""Bind a diagnostic to its actual Python dependencies, not only its entry file."""
import hashlib
import json
from pathlib import Path


DEPENDENCY_SCRIPTS = ('common.py', 'e25_qa_das.py', 'e25_qa_context.py',
                      'e26_matched_coverage.py', 'e27_qa_generation.py')


def implementation_identity(entry_file, code_root=None):
    entry = Path(entry_file).resolve()
    root = Path(code_root).resolve() if code_root is not None else entry.parents[1]
    entry.relative_to(root)  # Do not permit files outside the declared snapshot.
    paths = set((root / 'core').glob('*.py'))
    paths.update(root / 'scripts' / name for name in DEPENDENCY_SCRIPTS)
    paths.add(entry)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'protocol': 'diagnostic_dependency_source_v1', 'entry': str(entry.relative_to(root)),
            'files': files, 'files_fingerprint': hashlib.sha256(encoded).hexdigest()}


def require_unchanged(expected, entry_file, code_root=None):
    if implementation_identity(entry_file, code_root=code_root) != expected:
        raise RuntimeError('diagnostic implementation changed during execution; do not mark this run complete')
