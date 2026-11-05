#!/usr/bin/env python3
"""Regression check for the direction cache of verify_own_answers.py (GPT round 17 item F; Claude, 2026-11-06).

usage: chk_verify_cache.py [PATH_TO_verify_own_answers.py]
Executes the script's own cache-building loop, dir_key() and specs_for() with local fakes (no model, no data):
an old and a new DAS source both choose layer 11 in one call, the fake loader returns [1, 0] for the old source and
[0, 1] for the new one. Passes only if each condition gets its own direction, each source is loaded exactly once,
a second config of an already-loaded source does not reload it, and the mean-difference path is unchanged.
"""
import ast
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "verify_own_answers.py"
source = SRC.read_text()
tree = ast.parse(source)
loop = next(n for n in tree.body if isinstance(n, ast.For) and "load_das_direction" in ast.get_source_segment(source, n))
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("dir_key", "specs_for")]
if {f.name for f in funcs} != {"dir_key", "specs_for"}:
    raise SystemExit("FAIL: dir_key / specs_for not found in %s" % SRC)

old = {"estimator": "das", "layers": [11], "alpha": 3.0, "das_dir": "old", "das_variant": "tf"}
new = {"estimator": "das", "layers": [11], "alpha": 3.0, "das_dir": "new", "das_variant": "tf-ce-bidir"}
new_again = dict(new, alpha=2.0)
md = {"estimator": "md", "layers": [11], "alpha": 3.0, "das_dir": None, "das_variant": ""}
calls = []


def fake_load(d, comp, layer, variant=None, align_to=None):
    calls.append((d, variant, layer))
    return ([1.0, 0.0] if d == "old" else [0.0, 1.0]), {}


ns = {"configs": {"das_tf_full": old, "das_tfce_full": new, "das_tfce_again": new_again, "meandiff_full": md},
      "dirs": {11: {"md": [0.5, 0.5], "cB": [0.0, 0.0]}},
      "load_das_direction": fake_load,
      "SteerSpec": lambda w, l, **kw: {"W": w, "layer": l, **kw}}
exec(compile(ast.Module(body=funcs + [loop], type_ignores=[]), str(SRC), "exec"), ns)
checks = {
    "old condition uses the old direction": ns["specs_for"](old)[0]["W"] == [1.0, 0.0],
    "new condition uses the new direction": ns["specs_for"](new)[0]["W"] == [0.0, 1.0],
    "second config of the same source reuses it": ns["specs_for"](new_again)[0]["W"] == [0.0, 1.0],
    "each source loaded exactly once": sorted(calls) == [("new", "tf-ce-bidir", 11), ("old", "tf", 11)],
    "mean-difference path unchanged": ns["specs_for"](md)[0]["W"] == [0.5, 0.5],
    "baseline (None) gives no specs": ns["specs_for"](None) is None,
}
for k, ok in checks.items():
    print("%-46s %s" % (k, "PASS" if ok else "FAIL"))
if not all(checks.values()):
    raise SystemExit("FAIL: %d of %d checks" % (sum(not v for v in checks.values()), len(checks)))
print("all %d checks passed" % len(checks))
