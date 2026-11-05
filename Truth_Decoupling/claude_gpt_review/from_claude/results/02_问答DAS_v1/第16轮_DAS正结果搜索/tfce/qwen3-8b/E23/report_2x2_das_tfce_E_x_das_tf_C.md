# qwen3-8b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tf_C`

E sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/qwen3-8b`, C sweep under `/workspace/Truth_Decoupling/results_das_tf/E23/qwen3-8b` (E estimator of the C sweep: das variant tf). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.482, J +0.440, AUROC 0.812, B 0.445.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L35_a3 | 53 feasible of 72 | +0.006 [-0.004, +0.015] | +0.070 [+0.038, +0.104] | +0.031 [+0.022, +0.040] | +0.003 [-0.013, +0.020] |
| C prompt-contrast, toward abstention | Cp_L22_abst_a2 | 49 feasible of 72 | -0.233 [-0.268, -0.197] | -0.006 [-0.025, +0.013] | +0.002 [-0.001, +0.004] | -0.007 [-0.017, +0.004] |
| C prompt-contrast, toward answering | Cp_L30_ans_a2 | 40 feasible of 72 | +0.119 [+0.093, +0.147] | +0.009 [-0.008, +0.025] | +0.001 [+0.000, +0.002] | +0.018 [+0.010, +0.026] |
| C self-report, toward abstention | Cs_L30_abst_a2 | 16 feasible of 72 | -0.482 [-0.526, -0.439] | -0.004 [-0.017, +0.009] | +0.001 [-0.000, +0.003] | +0.004 [-0.003, +0.010] |
| C self-report, toward answering | Cs_L27_ans_a2 | 35 feasible of 72 | +0.518 [+0.474, +0.561] | +0.011 [-0.013, +0.034] | +0.006 [+0.004, +0.008] | +0.042 [+0.030, +0.054] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.004 | +0.000 | +0.002 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | -0.004 | -0.006 | +0.001 |
| 1 | a2 | -0.004 | +0.000 | +0.000 | a1 | +0.002 | -0.002 | +0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | -0.002 | -0.004 | -0.002 |
| 2 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.000 | -0.002 | -0.001 |
| 3 | a2 | -0.006 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | a2 | +0.002 | +0.000 | +0.000 |
| 4 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | -0.002 | +0.000 | +0.006 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 5 | a1 | -0.004 | -0.004 | -0.002 | a2 | +0.004 | +0.008 | +0.011 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 6 | a2 | -0.017 | -0.008 | -0.006 | a1 | +0.009 | +0.000 | +0.006 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 7 | a1 | -0.013 | -0.006 | -0.003 | a1 | +0.011 | -0.002 | +0.007 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 8 | a1 | -0.008 | -0.004 | +0.000 | a1 | +0.008 | -0.006 | -0.001 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  |
| 9 | a1 | -0.004 | -0.008 | +0.002 | a2 | +0.013 | -0.006 | -0.003 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a2: d ans +0.000, dJ -0.005, dB +0.008 on tune) |  |  |  |
| 10 | a2 | -0.013 | +0.013 | +0.018 | a2 | +0.023 | -0.009 | -0.005 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | -0.002 | +0.000 | +0.002 |
| 11 | a2 | -0.045 | +0.006 | +0.012 | a2 | +0.021 | -0.009 | -0.010 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | -0.002 | +0.000 | +0.004 |
| 12 | a1 | -0.009 | +0.008 | +0.011 | a2 | +0.019 | -0.011 | -0.021 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.000 | -0.006 | +0.003 |
| 13 | a1 | -0.015 | +0.006 | +0.016 | leak (a2: d ans +0.010, dJ +0.026, dB -0.039 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.000 | -0.002 | +0.001 |
| 14 | a2 | -0.060 | +0.013 | +0.029 | a2 | +0.060 | -0.011 | -0.002 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.002 | -0.004 | +0.002 |
| 15 | a2 | -0.042 | -0.004 | +0.011 | a1 | +0.026 | -0.008 | +0.000 | a2 | -0.004 | -0.002 | -0.003 | a1 | +0.004 | -0.002 | +0.003 |
| 16 | a2 | -0.070 | -0.004 | +0.002 | a1 | +0.021 | -0.009 | -0.007 | a2 | +0.002 | -0.004 | -0.004 | a1 | +0.004 | -0.004 | +0.004 |
| 17 | a1 | -0.051 | -0.004 | +0.008 | a1 | +0.025 | -0.009 | -0.007 | a2 | -0.006 | -0.008 | -0.002 | a1 | +0.004 | -0.002 | +0.003 |
| 18 | a2 | -0.025 | +0.004 | +0.026 | a1 | +0.015 | -0.002 | -0.012 | a1 | -0.006 | -0.004 | +0.000 | a2 | +0.021 | -0.006 | +0.005 |
| 19 | leak (a2: d ans -0.005, dJ +0.042, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.016, dJ +0.026, dB -0.039 on tune) |  |  |  | leak (a1: d ans +0.026, dJ +0.031, dB +0.052 on tune) |  |  |  | leak (a1: d ans -0.016, dJ +0.016, dB -0.034 on tune) |  |  |  |
| 20 | leak (a2: d ans -0.005, dJ +0.042, dB +0.026 on tune) |  |  |  | leak (a1: d ans -0.021, dJ +0.016, dB -0.055 on tune) |  |  |  | leak (a1: d ans +0.151, dJ +0.068, dB +0.102 on tune) |  |  |  | leak (a1: d ans -0.042, dJ +0.016, dB -0.076 on tune) |  |  |  |
| 21 | leak (a2: d ans -0.052, dJ +0.052, dB +0.089 on tune) |  |  |  | leak (a1: d ans -0.021, dJ +0.010, dB -0.078 on tune) |  |  |  | leak (a1: d ans +0.172, dJ +0.062, dB +0.156 on tune) |  |  |  | leak (a1: d ans -0.078, dJ +0.036, dB -0.107 on tune) |  |  |  |
| 22 | a2 | -0.233 | -0.006 | -0.007 | a1 | +0.049 | -0.008 | -0.013 | leak (a1: d ans +0.156, dJ +0.083, dB +0.161 on tune) |  |  |  | leak (a1: d ans -0.052, dJ +0.016, dB -0.128 on tune) |  |  |  |
| 23 | a2 | -0.140 | -0.011 | +0.011 | a2 | +0.076 | -0.009 | +0.001 | leak (a2: d ans -0.094, dJ -0.068, dB +0.352 on tune) |  |  |  | leak (a1: d ans +0.021, dJ -0.000, dB -0.115 on tune) |  |  |  |
| 24 | a2 | -0.164 | -0.015 | +0.000 | a2 | +0.091 | +0.011 | +0.030 | a1 | -0.306 | +0.015 | -0.008 | a1 | +0.236 | -0.036 | -0.022 |
| 25 | a2 | -0.161 | -0.006 | -0.005 | a2 | +0.068 | +0.002 | +0.022 | a2 | -0.456 | +0.021 | +0.020 | a1 | +0.210 | -0.026 | -0.017 |
| 26 | a2 | -0.117 | -0.004 | -0.036 | a1 | +0.053 | +0.008 | +0.032 | a1 | -0.336 | +0.006 | -0.003 | a1 | +0.251 | +0.013 | +0.031 |
| 27 | a2 | -0.110 | -0.006 | -0.033 | a1 | +0.049 | +0.013 | +0.031 | leak (a2: d ans -0.448, dJ +0.047, dB -0.107 on tune) |  |  |  | a2 | +0.518 | +0.011 | +0.042 |
| 28 | leak (a2: d ans -0.057, dJ +0.021, dB -0.036 on tune) |  |  |  | a1 | +0.051 | +0.006 | +0.027 | leak (a2: d ans -0.448, dJ +0.062, dB -0.109 on tune) |  |  |  | a2 | +0.518 | +0.011 | +0.042 |
| 29 | a2 | -0.138 | -0.004 | -0.021 | a1 | +0.066 | +0.006 | +0.022 | a1 | -0.465 | -0.008 | -0.013 | a1 | +0.448 | +0.004 | +0.028 |
| 30 | a1 | -0.072 | -0.009 | -0.010 | a2 | +0.119 | +0.009 | +0.018 | a2 | -0.482 | -0.004 | +0.004 | a1 | +0.486 | +0.013 | +0.022 |
| 31 | a2 | -0.159 | -0.008 | -0.030 | a2 | +0.115 | +0.011 | +0.030 | a1 | -0.471 | -0.019 | -0.038 | a2 | +0.518 | +0.002 | +0.009 |
| 32 | a1 | -0.079 | -0.006 | -0.016 | a1 | +0.079 | +0.013 | +0.027 | a1 | -0.476 | -0.009 | -0.035 | leak (a2: d ans +0.552, dJ +0.073, dB +0.099 on tune) |  |  |  |
| 33 | a2 | -0.168 | -0.009 | -0.033 | a1 | +0.087 | +0.009 | +0.026 | a1 | -0.478 | -0.025 | -0.054 | leak (a1: d ans +0.552, dJ +0.083, dB +0.089 on tune) |  |  |  |
| 34 | a1 | -0.098 | -0.004 | -0.019 | a1 | +0.089 | +0.011 | +0.028 | leak (a1: d ans -0.448, dJ +0.031, dB -0.094 on tune) |  |  |  | a2 | +0.518 | +0.000 | +0.004 |
| 35 | a1 | -0.115 | -0.002 | -0.018 | a1 | +0.108 | +0.011 | +0.036 | a1 | -0.469 | -0.004 | -0.023 | a1 | +0.457 | +0.009 | +0.039 |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf).

