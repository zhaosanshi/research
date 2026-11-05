# qwen3-8b: DAS-CE windows vs das_tf_spans (paired, same 529 test pairs)

Sweeps `das_tfce_spans` / `das_tf_spans` under `/workspace/Truth_Decoupling/results_das_tfce/E23/qwen3-8b` / `/workspace/Truth_Decoupling/results_das_tf/E23/qwen3-8b`. Rule on the tuning half (192 pairs): E = max dJ s.t. |d ans| <= 0.03. Test half reported with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True.

Baseline on the estimation pairs (the DAS training pairs; balanced classes): J=0.479, B=0.438 -> TPR=0.677, FPR=0.198 (the model answers False to 32.3% of true statements and True to 19.8% of false ones).

## Rule-chosen condition of each estimator (test half)

| estimator | chosen on tune | tune dJ | test dJ [CI] | test dAUROC [CI] | test d ans [CI] | test dB [CI] | oracle on test (dJ) |
|---|---|---|---|---|---|---|---|
| DAS-CE windows | E_L15-16-17_a2 | +0.135 | +0.034 [-0.023, +0.091] | +0.017 [-0.004, +0.038] | -0.015 [-0.060, +0.030] | -0.034 [-0.063, -0.002] | E_L18_a3 (+0.096) |
| das_tf_spans | E_L18_a3 | +0.078 | +0.051 [+0.026, +0.076] | +0.013 [+0.007, +0.019] | +0.008 [-0.009, +0.025] | +0.010 [-0.004, +0.025] | E_L16-17-18_a2 (+0.070) |

## Paired difference DAS-CE windows(E_L15-16-17_a2) - das_tf_spans(E_L18_a3) on the same resamples

| dJ | dAUROC | d ans | dB | P(dJ_DAS-CE windows > dJ_das_tf_spans) | P(dAUROC_DAS-CE windows > dAUROC_das_tf_spans) |
|---|---|---|---|---|---|
| -0.017 [-0.070, +0.036] | +0.005 [-0.014, +0.024] | -0.023 [-0.068, +0.021] | -0.044 [-0.070, -0.018] | 0.281 | 0.690 |

## Per layer, single-layer conditions (alpha picked by the rule on tune; test-half deltas; 'none' = every alpha leaks > 0.03 on tune, tune d ans shown)

| L | DAS-CE windows cfg | dJ | dAUROC | d ans | dB | das_tf_spans cfg | dJ | dAUROC | d ans | dB | DAS loss / cos(DAS, meandiff) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | E_L15_a3 | +0.0567 | +0.0238 | +0.0057 | +0.0095 | E_L15_a3 | +0.0170 | +0.0019 | +0.0038 | +0.0142 | 9.946 / +0.687; 4.869 / +0.673 |
| 16 | E_L16_a3 | +0.0699 | +0.0307 | +0.0095 | +0.0066 | E_L16_a3 | +0.0246 | +0.0059 | +0.0000 | +0.0123 | 9.762 / +0.686; 4.805 / +0.656 |
| 17 | E_L17_a2 | +0.0510 | +0.0262 | +0.0151 | +0.0009 | E_L17_a3 | +0.0302 | +0.0088 | -0.0038 | +0.0113 | 9.538 / +0.684; 4.702 / +0.659 |
| 18 | E_L18_a2 | +0.0699 | +0.0295 | +0.0170 | +0.0161 | E_L18_a3 | +0.0510 | +0.0128 | +0.0076 | +0.0104 | 9.240 / +0.689; 4.552 / +0.646 |
| 19 | none (tune d ans +0.042, +0.057) |  |  |  |  | none (tune d ans +0.031, +0.047) |  |  |  |  | 7.975 / +0.725; 3.820 / +0.650 |
| 20 | none (tune d ans +0.047, +0.057) |  |  |  |  | none (tune d ans +0.052, +0.047) |  |  |  |  | 5.803 / +0.745; 2.467 / +0.688 |
| 21 | none (tune d ans +0.068, +0.068) |  |  |  |  | none (tune d ans +0.052, +0.073) |  |  |  |  | 3.044 / +0.761; 0.844 / +0.715 |
| 22 | none (tune d ans +0.052, +0.068) |  |  |  |  | none (tune d ans +0.078, +0.094) |  |  |  |  | 1.684 / +0.733; 0.078 / +0.698 |
| 23 | none (tune d ans +0.031, +0.052) |  |  |  |  | none (tune d ans +0.047, +0.062) |  |  |  |  | 1.594 / +0.614; 0.015 / +0.701 |

## Multi-layer E conditions, DAS-CE windows (tune / test deltas; feasible = |tune d ans| <= 0.03)

| condition | feasible | tune dJ | tune d ans | test dJ | test dAUROC | test d ans | test dB |
|---|---|---|---|---|---|---|---|
| E_L15-16-17_a2 | yes | +0.135 | +0.010 | +0.034 | +0.017 | -0.015 | -0.034 |
| E_L15-16-17_a3 | no | +0.078 | -0.036 | -0.019 | -0.026 | -0.017 | -0.028 |
| E_L16-17-18_a2 | yes | +0.115 | +0.021 | +0.009 | +0.009 | -0.006 | -0.029 |
| E_L16-17-18_a3 | no | +0.068 | +0.036 | -0.034 | -0.031 | -0.009 | -0.034 |
| E_L17-18-19_a2 | no | +0.104 | +0.078 | +0.011 | +0.015 | +0.028 | -0.021 |
| E_L17-18-19_a3 | no | +0.083 | +0.057 | -0.023 | -0.066 | +0.015 | -0.019 |
| E_L18-19-20_a2 | no | +0.125 | +0.099 | +0.064 | +0.030 | +0.068 | +0.011 |
| E_L18-19-20_a3 | no | +0.146 | +0.099 | +0.030 | -0.023 | +0.066 | +0.011 |
| E_L19-20-21_a2 | no | +0.146 | +0.083 | +0.112 | +0.022 | +0.091 | +0.037 |
| E_L19-20-21_a3 | no | +0.156 | +0.083 | +0.112 | -0.031 | +0.089 | +0.039 |
| E_L20-21-22_a2 | no | +0.141 | +0.089 | +0.093 | -0.006 | +0.098 | +0.054 |
| E_L20-21-22_a3 | no | +0.156 | +0.094 | +0.089 | -0.088 | +0.098 | +0.058 |
| E_L21-22-23_a2 | no | +0.151 | +0.099 | +0.098 | +0.010 | +0.093 | +0.043 |
| E_L21-22-23_a3 | no | +0.146 | +0.104 | +0.095 | -0.017 | +0.095 | +0.045 |

## Multi-layer E conditions, das_tf_spans (tune / test deltas; feasible = |tune d ans| <= 0.03)

| condition | feasible | tune dJ | tune d ans | test dJ | test dAUROC | test d ans | test dB |
|---|---|---|---|---|---|---|---|
| E_L15-16-17_a2 | no | +0.104 | +0.031 | +0.089 | -0.012 | +0.032 | +0.048 |
| E_L15-16-17_a3 | no | +0.141 | +0.073 | +0.042 | -0.034 | +0.043 | +0.064 |
| E_L16-17-18_a2 | no | +0.135 | +0.031 | +0.070 | -0.002 | +0.019 | +0.029 |
| E_L16-17-18_a3 | no | +0.068 | +0.036 | +0.006 | -0.027 | +0.004 | -0.001 |
| E_L17-18-19_a2 | no | +0.109 | +0.083 | +0.053 | +0.005 | +0.015 | +0.019 |
| E_L17-18-19_a3 | no | +0.089 | +0.068 | -0.006 | -0.043 | +0.002 | -0.010 |
| E_L18-19-20_a2 | no | +0.146 | +0.104 | +0.078 | +0.014 | +0.083 | +0.054 |
| E_L18-19-20_a3 | no | +0.125 | +0.109 | +0.057 | -0.009 | +0.064 | +0.049 |
| E_L19-20-21_a2 | no | +0.109 | +0.099 | +0.049 | -0.008 | +0.093 | +0.051 |
| E_L19-20-21_a3 | no | +0.125 | +0.109 | +0.051 | -0.046 | +0.096 | +0.056 |
| E_L20-21-22_a2 | no | +0.083 | +0.094 | +0.025 | -0.052 | +0.106 | +0.056 |
| E_L20-21-22_a3 | no | +0.089 | +0.099 | +0.026 | -0.107 | +0.112 | +0.059 |
| E_L21-22-23_a2 | no | +0.057 | +0.109 | +0.021 | -0.012 | +0.106 | +0.061 |
| E_L21-22-23_a3 | no | +0.052 | +0.104 | +0.023 | -0.053 | +0.102 | +0.068 |
