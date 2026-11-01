"""CPU tests of damage-limited selection; no real model/data needed."""
import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


script = Path(__file__).resolve().parent.parent / "from_claude/code/scripts/select_low_harm_policy.py"
spec = importlib.util.spec_from_file_location("policy", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PolicyTests(unittest.TestCase):
    def test_consensus_never_adds_damage_outside_intersection(self):
        for label, base, das, md in itertools.product((False, True), repeat=4):
            p = lambda x: np.array([.9 if x else .1])
            out = bool(module.consensus(p(base), p(das), p(md))[0] > .5)
            self.assertEqual(base == label and out != label,
                             base == label and das != label and md != label)

    def test_selection_prefers_less_damage_on_equal_gain(self):
        rows = [dict(policy=dict(name="baseline", cost=[0, 0, 0]), tune=dict(harm_rate=0, harmed=0, net=0)),
                dict(policy=dict(name="harmful", cost=[1, 1, 2]), tune=dict(harm_rate=.08, harmed=20, net=30)),
                dict(policy=dict(name="gentle", cost=[1, 1, 1]), tune=dict(harm_rate=.004, harmed=1, net=8)),
                dict(policy=dict(name="consensus", cost=[3, 0, 0]), tune=dict(harm_rate=0, harmed=0, net=8))]
        self.assertEqual(module.select(rows, .005)["policy"]["name"], "consensus")
        self.assertEqual(module.select(rows[:2], .005)["policy"]["name"], "baseline")

    def test_fit_ignores_B_and_evaluation_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); old = root / "old"; low = root / "low"
            labels = np.array([True, False, True, False] * 2)
            baseline = np.array([.9, .9, .1, .1] * 2)
            mask = np.array([True] * 4 + [False] * 4)
            ids = np.array(["q%d:%s" % (i // 2, "T" if i % 2 == 0 else "F") for i in range(8)])

            def save(change_test=False):
                for model in module.MODELS:
                    for target, tag, name, values in ((old, "das_tf_spans", "E_L1_a2", [.9, .1, .9, .9] * 2),
                                                       (old, "meandiff_spans", "E_L1_a3", [.9, .1, .1, .1] * 2),
                                                       (low, "low_strength", "E_L1_a0.5", [.9, .1, .1, .1] * 2)):
                        d = target / "E23" / model; d.mkdir(parents=True, exist_ok=True)
                        y, b, v = labels.copy(), baseline.copy(), np.array(values)
                        if change_test:
                            y[~mask] = ~y[~mask]; b[~mask] = .4; v[~mask] = .6
                        meta = {"model": model, "conds": {"baseline": {}, name: {}},
                                "selection": {"E": {"chosen": name}}}
                        (d / ("knob_sweep_" + tag + ".json")).write_text(json.dumps(meta))
                        np.savez(d / ("knob_sweep_" + tag + ".npz"), ids=ids, labels=y, half_a=mask,
                                 baseline__ptrue=b, **{name + "__ptrue": v})

            save()
            args = SimpleNamespace(old_root=old, low_root=low, harm_cap=.005, fallback_band=True, policy=root / "policy1.json", out=root / "test.json")
            with contextlib.redirect_stdout(io.StringIO()):
                module.fit(args); module.evaluate(args)
            first = json.loads(args.policy.read_text())
            save(change_test=True)
            with self.assertRaisesRegex(ValueError, "input changed"):
                module.evaluate(args)
            args.policy = root / "policy2.json"
            with contextlib.redirect_stdout(io.StringIO()):
                module.fit(args)
            second = json.loads(args.policy.read_text())
            for model in module.MODELS:
                self.assertEqual(first["models"][model]["chosen"], second["models"][model]["chosen"])
                self.assertEqual(first["models"][model]["chosen"]["policy"]["name"], "low:E_L1_a0.5")

    def test_confidence_band_uses_only_baseline_scores(self):
        base = np.array([.5, .61, .31, .7, .9])
        changed = 1 - base
        data = {"old": {"baseline__ptrue": base, "E_L1_a2__ptrue": changed}}
        p = {"kind": "confidence_band", "condition": "E_L1_a2", "lo": .6, "hi": .7}
        actual = module.predictions(data, p, np.ones(5, dtype=bool))
        np.testing.assert_array_equal(actual, [base[0], changed[1], changed[2], base[3], base[4]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
