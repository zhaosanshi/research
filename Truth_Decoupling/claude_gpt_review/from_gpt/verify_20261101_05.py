"""Verify AUC ties against direct positive/negative comparisons; no inference."""
import importlib.util
from pathlib import Path
import numpy as np


scripts = Path(__file__).resolve().parent.parent / "from_claude/code/scripts"
labels = np.array([True, False, True, False])
cases = [[.5, .5, .5, .5], [.8, .8, .2, .1], [.9, .1, .8, .2], [.1, .9, .2, .8]]
for filename in ("e23_compare_estimators.py", "e23_merge.py"):
    spec = importlib.util.spec_from_file_location(filename[:-3], scripts / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for scores in cases:
        scores = np.asarray(scores)
        expected = np.mean([float(a > b) + .5 * float(a == b) for a in scores[labels] for b in scores[~labels]])
        actual = module.auroc(scores, labels)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        order = [3, 1, 0, 2]
        np.testing.assert_allclose(module.auroc(scores[order], labels[order]), expected, rtol=0, atol=1e-12)
    assert np.isnan(module.auroc([.1, .2], [True, True]))
    print(filename, "PASS: ties, partial ties, perfect/reverse order, permutation, missing class")
