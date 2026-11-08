# qwen3-8b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tfce_C`

E sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/qwen3-8b`, C sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/qwen3-8b` (E estimator of the C sweep: das variant tf-ce-bidir). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.482, J +0.440, AUROC 0.812, B 0.445.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L35_a3 | 53 feasible of 72 | +0.006 [-0.004, +0.015] | +0.070 [+0.038, +0.104] | +0.031 [+0.022, +0.040] | +0.003 [-0.013, +0.020] |
| C prompt-contrast, toward abstention | Cp_L35_abst_a2 | 53 feasible of 72 | -0.229 [-0.265, -0.195] | -0.013 [-0.032, +0.008] | +0.002 [+0.001, +0.003] | -0.031 [-0.043, -0.021] |
| C prompt-contrast, toward answering | Cp_L35_ans_a1 | 46 feasible of 72 | +0.106 [+0.081, +0.134] | +0.011 [-0.008, +0.030] | -0.000 [-0.001, +0.000] | +0.026 [+0.018, +0.036] |
| C self-report, toward abstention | Cs_L33_abst_a2 | 13 feasible of 72 | -0.482 [-0.526, -0.439] | -0.013 [-0.036, +0.009] | -0.001 [-0.003, +0.001] | -0.041 [-0.054, -0.028] |
| C self-report, toward answering | Cs_L24_ans_a2 | 21 feasible of 72 | +0.444 [+0.401, +0.486] | -0.017 [-0.043, +0.011] | -0.018 [-0.021, -0.014] | +0.044 [+0.030, +0.059] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | a1 | +0.000 | -0.006 | -0.003 | a2 | +0.004 | -0.002 | +0.003 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 1 | a1 | -0.006 | -0.008 | -0.004 | a2 | +0.008 | +0.006 | +0.010 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | a2 | +0.000 | -0.002 | +0.003 |
| 2 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.004 | -0.002 | +0.003 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 3 | a1 | -0.004 | -0.006 | -0.003 | a1 | +0.004 | +0.000 | +0.006 | leak (a2: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 4 | a2 | -0.008 | -0.004 | -0.004 | a2 | +0.006 | +0.002 | +0.012 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  |
| 5 | a2 | -0.008 | -0.008 | -0.006 | a1 | +0.008 | +0.000 | +0.008 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | -0.004 | -0.004 | +0.000 |
| 6 | a2 | -0.025 | -0.008 | -0.011 | a2 | +0.026 | +0.009 | +0.024 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.005, dB +0.003 on tune) |  |  |  |
| 7 | a2 | -0.023 | -0.009 | -0.010 | a1 | +0.013 | +0.006 | +0.012 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 8 | a1 | -0.008 | -0.004 | -0.002 | a2 | +0.015 | +0.004 | +0.000 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.002 | -0.004 | +0.000 |
| 9 | a2 | -0.019 | +0.000 | +0.006 | a2 | +0.017 | -0.004 | +0.000 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.006 | -0.002 | +0.007 |
| 10 | a2 | -0.009 | +0.015 | +0.023 | a2 | +0.023 | -0.008 | -0.008 | a1 | -0.002 | -0.004 | -0.002 | a1 | +0.004 | -0.006 | +0.005 |
| 11 | a2 | -0.026 | +0.017 | +0.026 | a2 | +0.021 | -0.009 | -0.010 | leak (a2: d ans +0.000, dJ +0.010, dB -0.005 on tune) |  |  |  | a1 | +0.002 | -0.004 | +0.004 |
| 12 | a1 | -0.006 | +0.015 | +0.023 | a2 | +0.017 | -0.015 | -0.026 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.004 | +0.000 | +0.011 |
| 13 | a1 | -0.008 | +0.011 | +0.021 | a2 | +0.011 | -0.006 | -0.035 | a1 | -0.006 | -0.006 | -0.003 | a1 | +0.002 | -0.002 | +0.005 |
| 14 | a2 | -0.066 | +0.006 | +0.024 | a2 | +0.068 | -0.009 | +0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.011 | -0.002 | +0.007 |
| 15 | a2 | -0.087 | -0.008 | -0.019 | a2 | +0.087 | +0.015 | +0.038 | a2 | -0.009 | -0.004 | -0.006 | a1 | +0.011 | -0.006 | +0.003 |
| 16 | a1 | -0.066 | -0.017 | -0.012 | a2 | +0.070 | +0.004 | +0.015 | a2 | -0.008 | -0.006 | -0.007 | a1 | +0.006 | +0.002 | +0.009 |
| 17 | a2 | -0.229 | -0.006 | -0.029 | a2 | +0.091 | +0.009 | +0.016 | a2 | +0.002 | -0.004 | -0.002 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 18 | a2 | -0.102 | -0.025 | -0.024 | a2 | +0.053 | +0.008 | +0.000 | a1 | +0.008 | +0.000 | +0.006 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 19 | a2 | -0.055 | -0.009 | -0.010 | leak (a1: d ans -0.010, dJ +0.016, dB -0.018 on tune) |  |  |  | leak (a1: d ans +0.031, dJ +0.026, dB +0.049 on tune) |  |  |  | leak (a1: d ans -0.021, dJ +0.010, dB -0.031 on tune) |  |  |  |
| 20 | leak (a2: d ans -0.005, dJ +0.047, dB +0.023 on tune) |  |  |  | leak (a1: d ans -0.026, dJ +0.016, dB -0.055 on tune) |  |  |  | leak (a1: d ans +0.125, dJ +0.073, dB +0.089 on tune) |  |  |  | leak (a1: d ans -0.042, dJ +0.016, dB -0.070 on tune) |  |  |  |
| 21 | a1 | -0.034 | +0.008 | +0.049 | leak (a1: d ans +0.010, dJ +0.010, dB -0.057 on tune) |  |  |  | leak (a1: d ans +0.135, dJ +0.057, dB +0.107 on tune) |  |  |  | leak (a1: d ans -0.057, dJ +0.010, dB -0.083 on tune) |  |  |  |
| 22 | leak (a2: d ans -0.167, dJ +0.021, dB +0.000 on tune) |  |  |  | a1 | +0.045 | -0.013 | -0.026 | leak (a1: d ans +0.010, dJ +0.052, dB +0.068 on tune) |  |  |  | a2 | +0.157 | -0.019 | +0.013 |
| 23 | a2 | -0.112 | +0.006 | +0.033 | a2 | +0.051 | -0.009 | -0.012 | leak (a2: d ans -0.328, dJ +0.078, dB +0.148 on tune) |  |  |  | a1 | +0.202 | -0.009 | +0.044 |
| 24 | a2 | -0.136 | +0.002 | +0.024 | a2 | +0.053 | -0.006 | +0.010 | leak (a2: d ans -0.438, dJ +0.104, dB +0.104 on tune) |  |  |  | a2 | +0.444 | -0.017 | +0.044 |
| 25 | a2 | -0.164 | -0.011 | -0.009 | a2 | +0.068 | +0.006 | +0.026 | leak (a2: d ans -0.427, dJ +0.047, dB -0.107 on tune) |  |  |  | leak (a2: d ans +0.432, dJ +0.021, dB +0.057 on tune) |  |  |  |
| 26 | a2 | -0.115 | -0.006 | -0.035 | a1 | +0.049 | +0.009 | +0.029 | leak (a2: d ans -0.448, dJ +0.016, dB -0.112 on tune) |  |  |  | leak (a2: d ans +0.542, dJ +0.083, dB +0.172 on tune) |  |  |  |
| 27 | a1 | -0.057 | +0.002 | -0.020 | a1 | +0.053 | +0.009 | +0.037 | leak (a2: d ans -0.448, dJ -0.042, dB -0.167 on tune) |  |  |  | leak (a2: d ans +0.552, dJ +0.062, dB +0.240 on tune) |  |  |  |
| 28 | a1 | -0.057 | -0.004 | -0.021 | a1 | +0.055 | +0.011 | +0.038 | leak (a2: d ans -0.448, dJ +0.016, dB -0.133 on tune) |  |  |  | leak (a2: d ans +0.552, dJ +0.068, dB +0.216 on tune) |  |  |  |
| 29 | a2 | -0.153 | -0.013 | -0.048 | a1 | +0.076 | +0.011 | +0.040 | a1 | -0.471 | -0.011 | -0.032 | leak (a2: d ans +0.552, dJ +0.078, dB +0.169 on tune) |  |  |  |
| 30 | a1 | -0.083 | -0.004 | -0.021 | a1 | +0.081 | +0.013 | +0.035 | leak (a2: d ans -0.448, dJ +0.052, dB -0.104 on tune) |  |  |  | leak (a2: d ans +0.552, dJ +0.062, dB +0.141 on tune) |  |  |  |
| 31 | a1 | -0.083 | -0.002 | -0.020 | a1 | +0.078 | +0.013 | +0.037 | leak (a2: d ans -0.448, dJ +0.047, dB -0.102 on tune) |  |  |  | leak (a2: d ans +0.552, dJ +0.026, dB +0.049 on tune) |  |  |  |
| 32 | a1 | -0.085 | -0.009 | -0.033 | leak (a2: d ans +0.161, dJ +0.073, dB +0.089 on tune) |  |  |  | leak (a2: d ans -0.448, dJ +0.016, dB -0.055 on tune) |  |  |  | leak (a2: d ans +0.552, dJ +0.068, dB +0.143 on tune) |  |  |  |
| 33 | a2 | -0.174 | -0.017 | -0.052 | a1 | +0.089 | +0.011 | +0.034 | a2 | -0.482 | -0.013 | -0.041 | leak (a1: d ans +0.552, dJ +0.068, dB +0.081 on tune) |  |  |  |
| 34 | leak (a2: d ans -0.188, dJ +0.021, dB -0.078 on tune) |  |  |  | a1 | +0.091 | +0.009 | +0.037 | leak (a1: d ans -0.448, dJ +0.010, dB -0.068 on tune) |  |  |  | leak (a2: d ans +0.552, dJ +0.036, dB +0.044 on tune) |  |  |  |
| 35 | a2 | -0.229 | -0.013 | -0.031 | a1 | +0.106 | +0.011 | +0.026 | leak (a2: d ans -0.448, dJ +0.026, dB -0.091 on tune) |  |  |  | leak (a2: d ans +0.552, dJ +0.057, dB +0.107 on tune) |  |  |  |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf-ce-bidir).

