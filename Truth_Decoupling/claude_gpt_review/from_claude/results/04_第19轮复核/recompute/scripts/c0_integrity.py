"""Verify every extracted archive member against its manifest (size + sha256), and archive sha256s."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(xdir, manifest_path, archive_path):
    rows = load(manifest_path)
    if isinstance(rows, dict):
        rows = rows['files']
    listed = {r['path'] for r in rows}
    on_disk = set()
    for root, _, files in os.walk(xdir):
        for fn in files:
            on_disk.add(os.path.relpath(os.path.join(root, fn), xdir))
    bad = []
    for r in rows:
        p = xdir / r['path']
        if not p.exists():
            bad.append(('missing', r['path']))
            continue
        if p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            bad.append(('mismatch', r['path']))
    extra = sorted(on_disk - listed)
    print(f"{archive_path.name}: manifest rows={len(rows)} files_on_disk={len(on_disk)} "
          f"bad={bad} extra_unlisted={extra} archive_sha256={sha(archive_path)}")


check(X_RANDOM, R19 / 'random/evidence/random_controls_manifest.json', R19 / 'random/evidence/random_controls.tar.gz')
check(X_DOSE, R19 / 'unicode/evidence/generation_dose_full192_manifest.json', R19 / 'unicode/evidence/generation_dose_full192.tar.gz')
check(X_DOSE_PILOT, R19 / 'unicode/evidence/generation_dose_pilot_manifest.json', R19 / 'unicode/evidence/generation_dose_pilot.tar.gz')
check(X_PURE, R19 / 'models/evidence/pure_das_layer_manifest.json', R19 / 'models/evidence/pure_das_layer_results.tar.gz')
check(X_R18, GPT / 'main_experiment_20261106/artifacts/generation_manifest.json', GPT / 'main_experiment_20261106/artifacts/generation_results.tar.gz')
