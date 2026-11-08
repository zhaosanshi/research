"""Local shared reply-cache transactions. Locks work across processes on macOS/Linux.

The per-request lock spans the API call; the cache-file lock only protects I/O.
Merging uses the same file lock, so it cannot replace a concurrent append.
All differing raw replies remain evidence. A conflict index is advisory: the
adjudicator also checks the actual cache, rather than trusting the index alone.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def cache_lock(path):
    return file_lock(str(path) + '.lock')


def request_lock(path, key):
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return file_lock(str(path) + '.requests/' + digest + '.lock')


def read_rows_unlocked(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get('key'), str) or not isinstance(row.get('answer'), str):
            raise ValueError('reply cache row requires string key and answer')
        rows.append(row)
    return rows


def read_cache_rows(path):
    with cache_lock(path):
        return read_rows_unlocked(path)


def replace_rows_unlocked(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def record_conflicts_unlocked(path, rows, sources=None):
    """Caller holds this cache's file lock. Update only this file's index entry."""
    path = Path(path)
    answers = {}
    for row in rows:
        values = answers.setdefault(row['key'], [])
        if row['answer'] not in values:
            values.append(row['answer'])
    conflicts = {key: values for key, values in answers.items() if len(values) > 1}
    index = path.parent / 'conflicts.json'
    with file_lock(str(index) + '.lock'):
        report = json.loads(index.read_text()) if index.exists() else {'keys': {}, 'detail': {}, 'files': {}}
        for field in ('keys', 'detail', 'files'):
            if not isinstance(report.get(field), dict):
                raise ValueError('invalid shared reply conflict index: ' + field)
        report['keys'][path.name] = sorted(conflicts)
        report['detail'][path.name] = conflicts
        old = report['files'].get(path.name, {})
        report['files'][path.name] = {'sources': list(sources) if sources is not None else old.get('sources', [str(path)]),
            'rows': len(rows), 'unique_keys': len(answers), 'conflicting_keys': len(conflicts),
            'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
        fd, temporary = tempfile.mkstemp(prefix='conflicts.', suffix='.tmp', dir=str(path.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            os.replace(temporary, index)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return report
