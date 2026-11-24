# True/false comparison on the new models: original model, mean difference, DAS

Rule on the tuning pairs (E: max dJ on the tuning pairs s.t. |d answer rate| <= 0.03; ties -> first in sweep order); reported on the Dr's 100 pre-fixed pairs (`dr_100pairs_20261110.json`) and on all 529 test pairs (the 100 are part of them). Pair-bootstrap 95% CIs, 2000 resamples, seed 0, the same resamples for every arm of a model. AUROC on p_true as in all earlier reports; AUROC on the log-odds margin alongside. Yes rate = the separate self-report pass ("Do you know the answer?"), not an actual answer rate. B = share of True answers (0.5 = no bias).

## 100 pairs

| model | arm | chosen on tuning pairs | AUROC | AUROC (margin) | J | Yes rate | B |
|---|---|---|---|---|---|---|---|
| gemma3-12b | original | - | 0.858 | 0.858 | +0.550 | 0.580 | 0.605 |
| gemma3-12b | meandiff | E_L25_a2 | 0.860 | 0.861 | +0.580 | 0.520 | 0.580 |
| gemma3-12b | DAS | E_L20_a3 | 0.879 | 0.879 | +0.560 | 0.550 | 0.590 |

| model | difference | AUROC [95% CI] | AUROC (margin) [CI] | J [CI] | Yes rate [CI] | B [CI] |
|---|---|---|---|---|---|---|
| gemma3-12b | DAS - original | +0.021 [+0.005, +0.040] | +0.021 [+0.005, +0.040] | +0.010 [-0.030, +0.050] | -0.030 [-0.080, +0.020] | -0.015 [-0.035, +0.005] |
| gemma3-12b | meandiff - original | +0.002 [-0.005, +0.010] | +0.003 [-0.004, +0.011] | +0.030 [-0.020, +0.080] | -0.060 [-0.120, -0.010] | -0.025 [-0.055, +0.000] |
| gemma3-12b | DAS - meandiff | +0.019 [+0.001, +0.040] | +0.018 [+0.001, +0.039] | -0.020 [-0.070, +0.020] | +0.030 [-0.010, +0.080] | +0.010 [-0.010, +0.035] |

## 529 pairs

| model | arm | chosen on tuning pairs | AUROC | AUROC (margin) | J | Yes rate | B |
|---|---|---|---|---|---|---|---|
| gemma3-12b | original | - | 0.870 | 0.870 | +0.578 | 0.665 | 0.548 |
| gemma3-12b | meandiff | E_L25_a2 | 0.874 | 0.874 | +0.578 | 0.622 | 0.527 |
| gemma3-12b | DAS | E_L20_a3 | 0.879 | 0.879 | +0.588 | 0.662 | 0.540 |

| model | difference | AUROC [95% CI] | AUROC (margin) [CI] | J [CI] | Yes rate [CI] | B [CI] |
|---|---|---|---|---|---|---|
| gemma3-12b | DAS - original | +0.009 [+0.002, +0.016] | +0.009 [+0.002, +0.016] | +0.009 [-0.008, +0.026] | -0.004 [-0.021, +0.013] | -0.009 [-0.017, +0.000] |
| gemma3-12b | meandiff - original | +0.004 [+0.000, +0.007] | +0.004 [+0.000, +0.007] | +0.000 [-0.019, +0.021] | -0.043 [-0.062, -0.026] | -0.021 [-0.031, -0.011] |
| gemma3-12b | DAS - meandiff | +0.005 [-0.002, +0.012] | +0.005 [-0.001, +0.012] | +0.009 [-0.009, +0.028] | +0.040 [+0.021, +0.059] | +0.012 [+0.004, +0.022] |

## Provenance

- gemma3-12b: DAS `/workspace/Truth_Decoupling/Iteration_01/results/tf/gemma3/12b/E23/gemma3-12b/knob_sweep_das_tfce_full.npz` (sha256 acd49bae6632…), mean difference `/workspace/Truth_Decoupling/Iteration_01/results/tf/gemma3/12b/E23/gemma3-12b/knob_sweep_meandiff_full.npz` (sha256 c788cf82ae0d…); tuning-set deltas DAS {"J": 0.05208333333333337, "ans": -0.02083333333333326, "AUROC": 0.005126953125, "B": 0.0}, mean difference {"J": 0.03645833333333337, "ans": -0.02083333333333326, "AUROC": 0.0008951822916666297, "B": -0.023437499999999944}
  - model: {"name": "gemma3-12b", "path": "/workspace/Truth_Decoupling/models/gemma-3-12b-it", "modelscope_repo": "google/gemma-3-12b-it", "hidden_size": 3840, "num_layers": 48, "dtype": "bfloat16", "python": null}; runtime of the DAS sweep: {"python": "/usr/bin/python3", "transformers": "4.57.1", "torch": "2.13.0a0+8145d630e8.nv26.06", "model_path": "/workspace/Truth_Decoupling/models/gemma-3-12b-it", "model_config_name": "gemma3-12b"}
