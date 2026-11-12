# True/false comparison on the new models: original model, mean difference, DAS

Rule on the tuning pairs (E: max dJ on the tuning pairs s.t. |d answer rate| <= 0.03; ties -> first in sweep order); reported on the Dr's 100 pre-fixed pairs (`dr_100pairs_20261110.json`) and on all 529 test pairs (the 100 are part of them). Pair-bootstrap 95% CIs, 2000 resamples, seed 0, the same resamples for every arm of a model. AUROC on p_true as in all earlier reports; AUROC on the log-odds margin alongside. Yes rate = the separate self-report pass ("Do you know the answer?"), not an actual answer rate. B = share of True answers (0.5 = no bias).

## 100 pairs

| model | arm | chosen on tuning pairs | AUROC | AUROC (margin) | J | Yes rate | B |
|---|---|---|---|---|---|---|---|
| qwen3.5-9b | original | - | 0.782 | 0.782 | +0.420 | 0.660 | 0.570 |
| qwen3.5-9b | meandiff | E_L17_a3 | 0.793 | 0.793 | +0.450 | 0.640 | 0.565 |
| qwen3.5-9b | DAS | E_L31_a3 | 0.840 | 0.840 | +0.540 | 0.645 | 0.570 |

| model | difference | AUROC [95% CI] | AUROC (margin) [CI] | J [CI] | Yes rate [CI] | B [CI] |
|---|---|---|---|---|---|---|
| qwen3.5-9b | DAS - original | +0.058 [+0.039, +0.078] | +0.058 [+0.039, +0.078] | +0.120 [+0.050, +0.190] | -0.015 [-0.040, +0.000] | +0.000 [-0.040, +0.040] |
| qwen3.5-9b | meandiff - original | +0.011 [+0.006, +0.017] | +0.011 [+0.006, +0.017] | +0.030 [-0.010, +0.080] | -0.020 [-0.050, +0.000] | -0.005 [-0.025, +0.015] |
| qwen3.5-9b | DAS - meandiff | +0.047 [+0.029, +0.067] | +0.047 [+0.029, +0.067] | +0.090 [+0.030, +0.160] | +0.005 [-0.015, +0.030] | +0.005 [-0.030, +0.045] |

## 529 pairs

| model | arm | chosen on tuning pairs | AUROC | AUROC (margin) | J | Yes rate | B |
|---|---|---|---|---|---|---|---|
| qwen3.5-9b | original | - | 0.825 | 0.825 | +0.488 | 0.698 | 0.527 |
| qwen3.5-9b | meandiff | E_L17_a3 | 0.834 | 0.834 | +0.501 | 0.694 | 0.521 |
| qwen3.5-9b | DAS | E_L31_a3 | 0.876 | 0.876 | +0.575 | 0.694 | 0.531 |

| model | difference | AUROC [95% CI] | AUROC (margin) [CI] | J [CI] | Yes rate [CI] | B [CI] |
|---|---|---|---|---|---|---|
| qwen3.5-9b | DAS - original | +0.051 [+0.043, +0.059] | +0.051 [+0.043, +0.059] | +0.087 [+0.055, +0.121] | -0.005 [-0.010, -0.001] | +0.004 [-0.014, +0.021] |
| qwen3.5-9b | meandiff - original | +0.010 [+0.008, +0.012] | +0.010 [+0.008, +0.012] | +0.013 [+0.000, +0.028] | -0.005 [-0.017, +0.008] | -0.007 [-0.014, +0.000] |
| qwen3.5-9b | DAS - meandiff | +0.041 [+0.035, +0.049] | +0.041 [+0.035, +0.049] | +0.074 [+0.045, +0.104] | +0.000 [-0.011, +0.011] | +0.010 [-0.008, +0.027] |

## Provenance

- qwen3.5-9b: DAS `/workspace/Truth_Decoupling/results_newmodels_20261112/E23/qwen3.5-9b/knob_sweep_das_tfce_full.npz` (sha256 4cbc00c02913…), mean difference `/workspace/Truth_Decoupling/results_newmodels_20261112/E23/qwen3.5-9b/knob_sweep_meandiff_full.npz` (sha256 8e7dc6e7de0b…); tuning-set deltas DAS {"J": 0.109375, "ans": 0.0026041666666667407, "AUROC": 0.05680338541666674, "B": 0.0078125}, mean difference {"J": 0.02604166666666674, "ans": -0.005208333333333259, "AUROC": 0.00981987847222221, "B": 0.0026041666666666297}
  - model: {"name": "qwen3.5-9b", "path": "/workspace/Truth_Decoupling/models/Qwen3.5-9B", "modelscope_repo": "Qwen/Qwen3.5-9B", "hidden_size": 4096, "num_layers": 32, "dtype": "bfloat16", "python": "/workspace/Truth_Decoupling/envs/tf519/bin/python"}; runtime of the DAS sweep: {"python": "/workspace/Truth_Decoupling/envs/tf519/bin/python", "transformers": "5.19.0", "torch": "2.13.0a0+8145d630e8.nv26.06", "model_path": "/workspace/Truth_Decoupling/models/Qwen3.5-9B", "model_config_name": "qwen3.5-9b"}
