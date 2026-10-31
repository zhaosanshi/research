"""CPU-only checks for full-layer routing and separate read/write selection.

Runs the real CLI parsers/layer-selection expressions and the reporting script.
Does not import unavailable experiment dependencies or start model training.
"""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


CODE = Path(__file__).resolve().parent.parent / "from_claude" / "code"
SCRIPTS = CODE / "scripts"
spec = importlib.util.spec_from_file_location("read_write_report", SCRIPTS / "report_read_write_layers.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def parsed_main(filename, argv):
    """Execute the production argparse block, ending just after parse_args()."""
    tree = ast.parse((SCRIPTS / filename).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    statements = []
    for stmt in main.body:
        statements.append(stmt)
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in stmt.targets):
            break
    else:
        raise AssertionError("parse_args assignment not found")
    namespace = {"argparse": argparse}
    with patch.object(sys, "argv", [filename] + argv):
        exec(compile(ast.Module(body=statements, type_ignores=[]), filename, "exec"), namespace)
    return namespace["args"], main


def run_assignment(main, name, namespace):
    node = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<production assignment>", "exec"), namespace)
    return namespace[name]


def fixture():
    summary = {"model": "synthetic", "num_layers": 3, "n_E": [4, 4], "direction_prompt_allow_idk": False,
               "das": {"variant": "tf", "E_trained_layers": [0, 1, 2], "n_pairs": {"E": 4}, "seeds": [42]}}
    profile = {"E": {"rows": {"0": {"sep": .99}, "1": {"sep": .7}, "2": {"sep": .6}}}}
    base = {"J": .4, "AUROC": .7, "B": .5, "ans": .5}
    sweep = {"model": "synthetic", "num_layers": 3, "layers": [0, 1, 2], "n_est": 8,
             "direction_prompt_allow_idk": False, "teacher_split": False, "tune_pairs": 2,
             "e_estimator": "das", "e_alphas": [2, 3],
             "das": {"variant": "tf", "per_layer": {str(l): {"seeds": [42]} for l in range(3)}},
             "conds": {"baseline": {"A": dict(base), "B": dict(base)}}}
    for layer in range(3):
        for alpha in (2, 3):
            a, b = dict(base), dict(base)
            a["J"] += .01 * (layer + 1) + .001 * alpha
            b["J"] += .20 if layer == 0 else .01
            sweep["conds"][f"E_L{layer}_a{alpha}"] = {"A": a, "B": b}
    return summary, profile, sweep


class LayerReviewTests(unittest.TestCase):
    def test_all_layers_use_loaded_model_and_include_endpoints(self):
        for model in ("llama3.1-8b", "qwen3-8b", "gemma3-12b"):
            n = json.loads((CODE / "configs/models" / (model + ".json")).read_text())["num_layers"]
            args, main = parsed_main("e1_directions.py", ["--model", model, "--das-all-layers"])
            ns = {"args": args, "bundle": types.SimpleNamespace(num_layers=n)}
            run_assignment(main, "layers", ns)
            trained = run_assignment(main, "train_layers", ns)
            self.assertEqual(trained, list(range(n)))
            args, main = parsed_main("e23_knob_sweep.py", ["--model", model, "--all-layers"])
            evaluated = run_assignment(main, "L", {"args": args, "bundle": types.SimpleNamespace(num_layers=n)})
            self.assertEqual(trained, evaluated)

    def test_explicit_single_layer_still_supported(self):
        args, main = parsed_main("e1_directions.py", ["--das-layers", "12"])
        self.assertEqual(run_assignment(main, "train_layers", {"args": args, "layers": list(range(32))}), [12])
        args, main = parsed_main("e23_knob_sweep.py", ["--model", "llama3.1-8b", "--layers", "12"])
        self.assertEqual(run_assignment(main, "L", {"args": args}), [12])

    def test_read_and_write_layers_can_differ(self):
        result = report.summarize(*fixture())
        self.assertEqual(result["read_layers"], [0])
        self.assertEqual(result["write_candidate"]["layer"], 2)
        self.assertEqual(result["read_write_distances"], {"0": 2})

    def test_test_scores_do_not_select_write_layer(self):
        args = fixture()
        before = report.summarize(*args)["write_candidate"]["condition"]
        for name, m in args[2]["conds"].items():
            if name != "baseline":
                m["B"]["J"] = .99 if "L1" in name else -.99
        self.assertEqual(report.summarize(*args)["write_candidate"]["condition"], before)

    def test_read_scores_do_not_select_write_layer(self):
        args = fixture()
        args[1]["E"]["rows"]["1"]["sep"] = 1.0
        result = report.summarize(*args)
        self.assertEqual(result["read_layers"], [1])
        self.assertEqual(result["write_candidate"]["layer"], 2)

    def test_read_ties_retained(self):
        args = fixture()
        args[1]["E"]["rows"]["1"]["sep"] = .99
        self.assertEqual(report.summarize(*args)["read_layers"], [0, 1])

    def test_missing_layer_or_strength_fails(self):
        args = fixture()
        del args[2]["conds"]["E_L2_a3"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            report.summarize(*args)
        args = fixture()
        args[2]["layers"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "every decoder"):
            report.summarize(*args)

    def test_no_feasible_candidate_is_reported_as_none(self):
        args = fixture()
        for name, m in args[2]["conds"].items():
            if name != "baseline":
                m["A"]["ans"] = .9
        result = report.summarize(*args)
        self.assertIsNone(result["write_candidate"])
        self.assertIn("无满足", report.markdown(result))

    def test_wrong_prompt_or_seed_fails(self):
        args = fixture()
        args[0]["direction_prompt_allow_idk"] = True
        with self.assertRaisesRegex(ValueError, "matching"):
            report.summarize(*args)
        args = fixture()
        args[2]["das"]["per_layer"]["0"]["seeds"] = [42, 43]
        with self.assertRaisesRegex(ValueError, "seeds differ"):
            report.summarize(*args)

    def test_report_cli_writes_result_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in zip(("summary.json", "layer_scores.json", "sweep.json"), fixture()):
                (root / name).write_text(json.dumps(data))
            cmd = [sys.executable, str(SCRIPTS / "report_read_write_layers.py"), "--e1-dir", str(root),
                   "--sweep", str(root / "sweep.json"), "--out", str(root / "report")]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads((root / "report.json").read_text())["write_candidate"]["layer"], 2)
            self.assertTrue((root / "report.md").is_file())
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
