# True/false comparison on the new models: original model, mean difference, DAS

Rule on the tuning pairs (E: max dJ on the tuning pairs s.t. |d answer rate| <= 0.03; ties -> first in sweep order); reported on the Dr's 100 pre-fixed pairs (`dr_100pairs_20261110.json`) and on all 529 test pairs (the 100 are part of them). Pair-bootstrap 95% CIs, 2000 resamples, seed 0, the same resamples for every arm of a model. AUROC on p_true as in all earlier reports; AUROC on the log-odds margin alongside. Yes rate = the separate self-report pass ("Do you know the answer?"), not an actual answer rate. B = share of True answers (0.5 = no bias).

## 100 pairs

| model | arm | chosen on tuning pairs | AUROC | AUROC (margin) | J | Yes rate | B |
|---|---|---|---|---|---|---|---|
| gemma3-27b | original | - | 0.842 | 0.843 | +0.540 | 0.500 | 0.530 |
| gemma3-27b | meandiff | E_L42_a3 | 0.848 | 0.847 | +0.600 | 0.490 | 0.530 |
| gemma3-27b | DAS | E_L33_a3 | 0.817 | 0.816 | +0.590 | 0.510 | 0.545 |

| model | difference | AUROC [95% CI] | AUROC (margin) [CI] | J [CI] | Yes rate [CI] | B [CI] |
|---|---|---|---|---|---|---|
| gemma3-27b | DAS - original | -0.026 [-0.060, +0.007] | -0.026 [-0.061, +0.006] | +0.050 [+0.000, +0.110] | +0.010 [+0.000, +0.030] | +0.015 [-0.005, +0.035] |
| gemma3-27b | meandiff - original | +0.005 [-0.012, +0.022] | +0.005 [-0.012, +0.022] | +0.060 [+0.010, +0.120] | -0.010 [-0.040, +0.020] | +0.000 [-0.030, +0.030] |
| gemma3-27b | DAS - meandiff | -0.031 [-0.063, -0.003] | -0.031 [-0.063, -0.003] | -0.010 [-0.060, +0.030] | +0.020 [-0.020, +0.060] | +0.015 [-0.005, +0.040] |

## 529 pairs

| model | arm | chosen on tuning pairs | AUROC | AUROC (margin) | J | Yes rate | B |
|---|---|---|---|---|---|---|---|
| gemma3-27b | original | - | 0.865 | 0.865 | +0.571 | 0.558 | 0.499 |
| gemma3-27b | meandiff | E_L42_a3 | 0.860 | 0.860 | +0.586 | 0.577 | 0.497 |
| gemma3-27b | DAS | E_L33_a3 | 0.838 | 0.837 | +0.622 | 0.577 | 0.517 |

| model | difference | AUROC [95% CI] | AUROC (margin) [CI] | J [CI] | Yes rate [CI] | B [CI] |
|---|---|---|---|---|---|---|
| gemma3-27b | DAS - original | -0.028 [-0.042, -0.013] | -0.028 [-0.043, -0.013] | +0.051 [+0.026, +0.078] | +0.019 [+0.006, +0.034] | +0.018 [+0.007, +0.029] |
| gemma3-27b | meandiff - original | -0.005 [-0.011, +0.002] | -0.005 [-0.011, +0.002] | +0.015 [-0.008, +0.038] | +0.019 [+0.004, +0.036] | -0.002 [-0.015, +0.011] |
| gemma3-27b | DAS - meandiff | -0.023 [-0.036, -0.010] | -0.023 [-0.037, -0.010] | +0.036 [+0.015, +0.059] | +0.000 [-0.015, +0.017] | +0.020 [+0.008, +0.032] |

## Provenance

- gemma3-27b: DAS `/workspace/Truth_Decoupling/Iteration_01/results/tf/gemma3/27b/E23/gemma3-27b/knob_sweep_das_tfce_full.npz` (sha256 e990ddee96f3…), mean difference `/workspace/Truth_Decoupling/Iteration_01/results/tf/gemma3/27b/E23/gemma3-27b/knob_sweep_meandiff_full.npz` (sha256 babb66ea3b99…); tuning-set deltas DAS {"J": 0.046875, "ans": 0.005208333333333259, "AUROC": -0.01383463541666663, "B": 0.028645833333333315}, mean difference {"J": 0.03645833333333326, "ans": -0.01041666666666674, "AUROC": -0.00467936197916663, "B": -0.013020833333333315}
  - model: {"name": "gemma3-27b", "path": "/workspace/Truth_Decoupling/models/gemma-3-27b-it", "modelscope_repo": "google/gemma-3-27b-it", "hidden_size": 5376, "num_layers": 62, "dtype": "bfloat16", "python": null}; runtime of the DAS sweep: {"python": "/usr/bin/python3", "transformers": "4.57.1", "torch": "2.13.0a0+8145d630e8.nv26.06", "model_path": "/workspace/Truth_Decoupling/models/gemma-3-27b-it", "model_config_name": "gemma3-27b"}
