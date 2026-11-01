"""Reproduce round-6 review evidence from uploaded results; no model inference."""
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent.parent / "from_claude"
RESULTS = ROOT / "results/01_E估计器/第5轮_窗口_C行_Gemma定位"
spec = importlib.util.spec_from_file_location("report_2x2", ROOT / "code/scripts/report_2x2.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# Check the actual C choices and saturation using production selection code.
selection = {}
for model in ("llama3.1-8b", "qwen3-8b"):
    conds = read(RESULTS / model / "knob_sweep_das_tf_C.json")["conds"]
    base = conds["baseline"]
    rows = {}
    for kind in ("Cp", "Cs"):
        for direction in ("abst", "ans"):
            name, _, _ = report.pick_c(conds, kind, direction)
            rows[name] = {"tune_delta": {k: conds[name]["A"][k] - base["A"][k] for k in ("ans", "J", "B")},
                          "test_self_report": conds[name]["B"]["ans"]}
            if kind == "Cs":
                expected = 0.0 if direction == "abst" else 1.0
                assert conds[name]["A"]["ans"] == expected
                assert conds[name]["B"]["ans"] == expected
    selection[model] = rows

# A direction can have worse standalone AUC and still improve this exact
# amplification operator. The downstream readout depends on both coordinates.
y = np.array([True, True, False, False])
old_score = np.array([3., -1., 1., -3.])
coordinate = np.array([-2., 2., -2., 2.])
h = np.column_stack((old_score - coordinate, coordinate))
w = np.array([0., 1.])
center = (h[y].mean(axis=0) + h[~y].mean(axis=0)) / 2
amplified = h + np.outer((h - center) @ w, w)  # alpha=1
head = np.array([1., 1.])
np.testing.assert_array_equal(center, [0., 0.])
aucs = [report.auroc(coordinate, y), report.auroc(h @ head, y), report.auroc(amplified @ head, y)]
np.testing.assert_array_equal(aucs, [.5, .75, 1.])

# Recover unconditional and same-first-token swap rates from saved aggregates.
trace = read(RESULTS / "gemma3-12b/trace_direction_tf_bidir_allpos_L18_L20_L22_L23_L24.json")
swaps = {}
for layer, values in trace["per_layer"].items():
    stats = values["stats"]["tune"]
    sw = stats["swap"]["das"]
    total, informative = stats["n_statements"], stats["n_informative_statements"]
    successes = round(sw["first_token_match_all"] * total)
    same_success = successes - sw["n_success_informative"]
    assert total == 384 and informative == 282
    swaps[layer] = {"different_first_token": [sw["n_success_informative"], informative],
                    "all": [successes, total], "same_first_token": [same_success, total - informative]}

# Execute the actual old diagnostic expression on a reversed model judgment.
tree = ast.parse((ROOT / "code/scripts/diag_trace_direction.py").read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "toward" for t in n.targets))
scope = {"np": np, "lab": np.array([True]), "pb": np.array([.1]), "ps": np.array([.9])}
exec(compile(ast.Module(body=[node], type_ignores=[]), "<diagnostic expression>", "exec"), scope)
old_toward = float(scope["toward"][0])
source_distance_reduction = abs(.1 - .9) - abs(.9 - .9)
assert old_toward < 0 < source_distance_reduction

print(json.dumps({"C_selection": selection,
                  "amplification_counterexample": {"h": h.tolist(), "w": w.tolist(), "center": center.tolist(),
                                                   "coordinate_baseline_amplified_AUC": aucs},
                  "Gemma_swap_counts": swaps,
                  "diagnostic_towards_source_counterexample": {"current_field": old_toward,
                                                               "true_distance_reduction": source_distance_reduction}},
                 ensure_ascii=False, indent=2))
print("PASS: uploaded-result checks and mathematical counterexamples; no model run")
