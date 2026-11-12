# qwen3.8-27b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tfce_C`

E sweep under `/workspace/Truth_Decoupling/results_newmodels_20261112/E23/qwen3.8-27b`, C sweep under `/workspace/Truth_Decoupling/results_newmodels_20261112/E23/qwen3.8-27b` (E estimator of the C sweep: das variant tf-ce-bidir). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.682, J +0.503, AUROC 0.840, B 0.484.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L63_a3 | 124 feasible of 128 | +0.013 [+0.005, +0.024] | +0.064 [+0.026, +0.098] | +0.041 [+0.035, +0.047] | +0.078 [+0.060, +0.095] |
| C prompt-contrast, toward abstention | Cp_L62_abst_a2 | 55 feasible of 128 | -0.633 [-0.673, -0.590] | +0.008 [-0.006, +0.021] | +0.000 [-0.001, +0.001] | -0.011 [-0.019, -0.006] |
| C prompt-contrast, toward answering | Cp_L63_ans_a1 | 41 feasible of 128 | +0.220 [+0.186, +0.256] | +0.006 [-0.013, +0.025] | -0.000 [-0.001, +0.001] | +0.024 [+0.015, +0.033] |
| C self-report, toward abstention | Cs_L57_abst_a2 | 39 feasible of 128 | -0.677 [-0.713, -0.635] | +0.015 [-0.011, +0.042] | +0.001 [-0.001, +0.002] | +0.043 [+0.031, +0.056] |
| C self-report, toward answering | Cs_L60_ans_a2 | 34 feasible of 128 | +0.314 [+0.278, +0.355] | -0.006 [-0.025, +0.013] | -0.001 [-0.002, +0.000] | -0.027 [-0.039, -0.018] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | a1 | +0.004 | -0.008 | +0.004 | leak (a1: d ans +0.003, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  |
| 1 | a2 | +0.002 | -0.006 | -0.001 | a1 | +0.004 | -0.006 | -0.001 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.016, dB -0.003 on tune) |  |  |  |
| 2 | a1 | +0.004 | -0.009 | -0.001 | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.001 | -0.004 | +0.000 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 3 | a2 | +0.005 | +0.000 | +0.008 | a2 | +0.000 | +0.002 | -0.007 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.002 | -0.008 | +0.000 |
| 4 | a2 | +0.005 | -0.004 | +0.008 | a2 | +0.001 | -0.002 | -0.003 | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.016, dB +0.003 on tune) |  |  |  |
| 5 | a1 | +0.004 | -0.004 | +0.004 | a1 | -0.002 | -0.002 | -0.003 | a2 | +0.005 | -0.006 | -0.003 | a1 | +0.004 | -0.002 | +0.001 |
| 6 | leak (a1: d ans +0.000, dJ +0.005, dB +0.008 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.010, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.008 on tune) |  |  |  |
| 7 | a2 | +0.002 | +0.000 | +0.011 | a2 | +0.001 | +0.002 | -0.007 | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 8 | a2 | +0.008 | -0.002 | +0.018 | a2 | +0.001 | +0.002 | -0.007 | leak (a1: d ans +0.003, dJ -0.010, dB +0.005 on tune) |  |  |  | a1 | +0.003 | -0.006 | +0.001 |
| 9 | a2 | +0.007 | -0.000 | +0.019 | leak (a1: d ans -0.003, dJ +0.000, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | a2 | +0.001 | -0.002 | +0.003 |
| 10 | a1 | +0.009 | -0.004 | +0.011 | leak (a1: d ans +0.000, dJ -0.005, dB -0.013 on tune) |  |  |  | a1 | +0.005 | -0.002 | +0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 11 | a1 | +0.004 | +0.002 | +0.005 | a1 | -0.002 | +0.004 | -0.006 | leak (a1: d ans +0.000, dJ +0.005, dB +0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 12 | a1 | +0.006 | -0.002 | +0.010 | leak (a2: d ans -0.003, dJ +0.010, dB -0.010 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 13 | a2 | +0.001 | +0.004 | +0.017 | leak (a2: d ans +0.000, dJ +0.000, dB -0.016 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | -0.002 | +0.004 | +0.002 |
| 14 | a1 | +0.007 | -0.000 | +0.006 | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  | a1 | +0.002 | -0.006 | -0.001 | a2 | +0.002 | -0.002 | -0.001 |
| 15 | leak (a2: d ans +0.000, dJ -0.005, dB +0.018 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.016, dB -0.008 on tune) |  |  |  | a2 | -0.001 | -0.006 | -0.001 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 16 | a2 | +0.003 | -0.008 | +0.009 | leak (a2: d ans +0.000, dJ -0.010, dB -0.005 on tune) |  |  |  | a1 | +0.004 | -0.006 | +0.001 | leak (a1: d ans -0.003, dJ +0.005, dB -0.008 on tune) |  |  |  |
| 17 | a1 | +0.005 | -0.011 | +0.004 | a2 | -0.002 | -0.000 | -0.006 | leak (a1: d ans +0.000, dJ +0.010, dB +0.010 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  |
| 18 | a1 | +0.007 | -0.009 | +0.005 | leak (a2: d ans -0.003, dJ -0.005, dB -0.008 on tune) |  |  |  | a2 | +0.005 | -0.000 | +0.004 | a1 | +0.002 | -0.002 | +0.001 |
| 19 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.000 | -0.002 | +0.001 | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  | a1 | +0.007 | +0.000 | -0.002 |
| 20 | leak (a1: d ans +0.000, dJ +0.010, dB +0.010 on tune) |  |  |  | leak (a1: d ans -0.003, dJ +0.000, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.003, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 21 | leak (a2: d ans +0.000, dJ -0.010, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.010, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.005 | -0.004 | +0.006 | leak (a1: d ans -0.005, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 22 | leak (a2: d ans +0.003, dJ -0.010, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  | a1 | +0.001 | -0.006 | +0.003 | a2 | +0.000 | +0.000 | -0.002 |
| 23 | a2 | +0.005 | -0.002 | +0.005 | leak (a2: d ans -0.003, dJ -0.005, dB -0.008 on tune) |  |  |  | a2 | +0.005 | -0.004 | +0.004 | leak (a2: d ans -0.003, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 24 | a1 | +0.000 | -0.004 | +0.008 | leak (a1: d ans +0.000, dJ -0.010, dB -0.005 on tune) |  |  |  | leak (a2: d ans -0.010, dJ +0.021, dB +0.010 on tune) |  |  |  | a2 | +0.003 | -0.002 | -0.003 |
| 25 | a2 | +0.002 | -0.004 | +0.008 | leak (a1: d ans -0.005, dJ +0.000, dB -0.005 on tune) |  |  |  | a1 | +0.000 | -0.008 | +0.004 | leak (a2: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 26 | a1 | +0.002 | -0.000 | +0.006 | leak (a1: d ans -0.003, dJ -0.016, dB -0.003 on tune) |  |  |  | a1 | -0.001 | -0.004 | +0.006 | leak (a1: d ans -0.003, dJ +0.000, dB -0.005 on tune) |  |  |  |
| 27 | a2 | -0.008 | -0.009 | +0.001 | leak (a1: d ans -0.005, dJ +0.005, dB +0.003 on tune) |  |  |  | a2 | -0.002 | +0.004 | +0.008 | a2 | +0.005 | +0.009 | -0.005 |
| 28 | a2 | -0.002 | -0.000 | +0.004 | a2 | +0.009 | +0.000 | +0.000 | a2 | -0.002 | -0.002 | +0.009 | a2 | +0.007 | +0.002 | -0.001 |
| 29 | a2 | -0.008 | -0.011 | -0.009 | a2 | +0.009 | +0.000 | +0.008 | a1 | +0.002 | -0.006 | +0.005 | a1 | +0.005 | -0.002 | -0.003 |
| 30 | a1 | -0.007 | +0.002 | -0.007 | a1 | +0.004 | -0.002 | +0.005 | a1 | +0.004 | -0.002 | +0.007 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 31 | leak (a2: d ans +0.000, dJ +0.016, dB +0.039 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB -0.005 on tune) |  |  |  | a2 | +0.007 | -0.004 | +0.017 | leak (a1: d ans -0.003, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 32 | leak (a2: d ans -0.005, dJ +0.010, dB +0.073 on tune) |  |  |  | leak (a1: d ans -0.010, dJ +0.000, dB -0.021 on tune) |  |  |  | a2 | +0.004 | +0.002 | +0.014 | leak (a1: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  |
| 33 | leak (a2: d ans -0.010, dJ +0.026, dB +0.044 on tune) |  |  |  | leak (a1: d ans -0.010, dJ +0.010, dB -0.010 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.016, dB +0.034 on tune) |  |  |  | a1 | -0.004 | +0.004 | -0.013 |
| 34 | a2 | -0.017 | +0.004 | +0.004 | leak (a1: d ans -0.010, dJ +0.000, dB -0.010 on tune) |  |  |  | leak (a1: d ans +0.010, dJ +0.016, dB +0.049 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.021, dB -0.021 on tune) |  |  |  |
| 35 | a2 | -0.026 | +0.002 | +0.001 | leak (a1: d ans -0.005, dJ +0.000, dB -0.010 on tune) |  |  |  | leak (a1: d ans +0.005, dJ -0.005, dB +0.070 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.016, dB -0.034 on tune) |  |  |  |
| 36 | a2 | -0.018 | +0.002 | -0.012 | leak (a2: d ans +0.003, dJ -0.031, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.008, dJ -0.016, dB +0.076 on tune) |  |  |  | leak (a1: d ans -0.016, dJ -0.016, dB -0.034 on tune) |  |  |  |
| 37 | a2 | -0.016 | +0.000 | -0.015 | a1 | +0.017 | +0.002 | +0.005 | leak (a1: d ans +0.008, dJ +0.005, dB +0.070 on tune) |  |  |  | leak (a1: d ans -0.013, dJ -0.010, dB -0.042 on tune) |  |  |  |
| 38 | a2 | -0.019 | -0.004 | -0.028 | a2 | +0.028 | +0.004 | +0.021 | leak (a1: d ans +0.003, dJ -0.010, dB +0.073 on tune) |  |  |  | leak (a1: d ans -0.016, dJ -0.016, dB -0.039 on tune) |  |  |  |
| 39 | leak (a1: d ans +0.000, dJ +0.000, dB +0.010 on tune) |  |  |  | a1 | +0.015 | -0.002 | +0.010 | leak (a1: d ans +0.003, dJ -0.005, dB +0.086 on tune) |  |  |  | leak (a2: d ans -0.005, dJ +0.005, dB -0.070 on tune) |  |  |  |
| 40 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a2: d ans +0.000, dJ -0.010, dB +0.026 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.070 on tune) |  |  |  | leak (a1: d ans -0.008, dJ -0.005, dB -0.044 on tune) |  |  |  |
| 41 | a1 | -0.005 | -0.008 | +0.009 | leak (a2: d ans +0.008, dJ -0.021, dB +0.010 on tune) |  |  |  | leak (a2: d ans -0.003, dJ -0.042, dB +0.177 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.016, dB -0.039 on tune) |  |  |  |
| 42 | a1 | -0.007 | +0.004 | +0.021 | leak (a1: d ans -0.005, dJ -0.005, dB -0.013 on tune) |  |  |  | leak (a1: d ans -0.008, dJ +0.016, dB +0.081 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.010, dB -0.042 on tune) |  |  |  |
| 43 | leak (a1: d ans -0.005, dJ +0.026, dB +0.044 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.023 on tune) |  |  |  | leak (a1: d ans +0.021, dJ -0.026, dB +0.117 on tune) |  |  |  | leak (a2: d ans -0.008, dJ -0.026, dB -0.107 on tune) |  |  |  |
| 44 | leak (a2: d ans +0.000, dJ +0.026, dB +0.055 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.016, dB -0.013 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.026, dB +0.107 on tune) |  |  |  | leak (a2: d ans +0.018, dJ -0.010, dB -0.099 on tune) |  |  |  |
| 45 | leak (a1: d ans +0.005, dJ +0.021, dB +0.042 on tune) |  |  |  | leak (a2: d ans +0.000, dJ -0.031, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.010, dJ -0.026, dB +0.107 on tune) |  |  |  | leak (a2: d ans +0.036, dJ -0.016, dB -0.112 on tune) |  |  |  |
| 46 | leak (a1: d ans +0.010, dJ +0.031, dB +0.047 on tune) |  |  |  | a1 | +0.017 | -0.008 | -0.006 | leak (a1: d ans -0.008, dJ -0.010, dB +0.104 on tune) |  |  |  | leak (a2: d ans +0.036, dJ +0.000, dB -0.089 on tune) |  |  |  |
| 47 | leak (a1: d ans +0.003, dJ +0.010, dB +0.036 on tune) |  |  |  | leak (a2: d ans +0.008, dJ -0.021, dB +0.016 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.010, dB +0.104 on tune) |  |  |  | a2 | +0.072 | -0.036 | -0.037 |
| 48 | a2 | +0.002 | +0.008 | +0.025 | a2 | +0.018 | -0.008 | +0.028 | leak (a2: d ans -0.099, dJ -0.021, dB +0.120 on tune) |  |  |  | leak (a2: d ans +0.060, dJ -0.026, dB -0.023 on tune) |  |  |  |
| 49 | a1 | -0.002 | -0.004 | +0.011 | leak (a1: d ans +0.010, dJ -0.026, dB +0.013 on tune) |  |  |  | a1 | -0.051 | -0.000 | +0.019 | a1 | +0.057 | -0.006 | -0.001 |
| 50 | a1 | -0.009 | +0.002 | +0.024 | a1 | +0.023 | -0.002 | -0.009 | leak (a2: d ans -0.284, dJ +0.021, dB +0.026 on tune) |  |  |  | a1 | +0.068 | -0.004 | -0.004 |
| 51 | a1 | -0.012 | +0.002 | +0.031 | a1 | +0.032 | +0.008 | -0.034 | a2 | -0.321 | +0.004 | +0.030 | a2 | +0.181 | -0.009 | -0.031 |
| 52 | leak (a2: d ans -0.068, dJ -0.026, dB +0.107 on tune) |  |  |  | a1 | +0.040 | +0.000 | -0.038 | a1 | -0.143 | -0.004 | +0.017 | a1 | +0.106 | -0.004 | -0.008 |
| 53 | a1 | -0.022 | +0.002 | +0.026 | a1 | +0.038 | +0.009 | -0.026 | leak (a2: d ans -0.505, dJ +0.005, dB +0.070 on tune) |  |  |  | a1 | +0.110 | +0.004 | -0.017 |
| 54 | a1 | -0.036 | +0.000 | +0.036 | a2 | +0.113 | +0.015 | -0.047 | a1 | -0.190 | +0.017 | +0.039 | a1 | +0.139 | +0.006 | -0.046 |
| 55 | a1 | -0.043 | +0.002 | +0.031 | a1 | +0.054 | +0.011 | -0.026 | a1 | -0.233 | +0.002 | +0.033 | a1 | +0.162 | +0.009 | -0.033 |
| 56 | a1 | -0.045 | +0.002 | +0.029 | a2 | +0.166 | +0.013 | -0.035 | a1 | -0.250 | -0.000 | +0.026 | a1 | +0.183 | -0.002 | -0.026 |
| 57 | a1 | -0.053 | +0.002 | +0.026 | a1 | +0.065 | +0.015 | -0.028 | a2 | -0.677 | +0.015 | +0.043 | a1 | +0.181 | +0.006 | -0.026 |
| 58 | a1 | -0.053 | +0.002 | +0.031 | a1 | +0.075 | +0.013 | -0.024 | leak (a2: d ans -0.688, dJ +0.010, dB +0.062 on tune) |  |  |  | a1 | +0.182 | +0.009 | -0.033 |
| 59 | a1 | -0.070 | +0.006 | +0.026 | a2 | +0.172 | +0.004 | -0.034 | a1 | -0.311 | +0.006 | +0.024 | a1 | +0.200 | +0.006 | -0.031 |
| 60 | a1 | -0.125 | -0.002 | +0.026 | a1 | +0.102 | +0.008 | -0.025 | a2 | -0.682 | -0.002 | +0.005 | a2 | +0.314 | -0.006 | -0.027 |
| 61 | a2 | -0.359 | -0.002 | +0.022 | a1 | +0.121 | +0.009 | -0.027 | a2 | -0.682 | +0.008 | -0.026 | a1 | +0.234 | +0.004 | -0.004 |
| 62 | a2 | -0.633 | +0.008 | -0.011 | a1 | +0.189 | -0.000 | +0.019 | a2 | -0.682 | -0.002 | +0.010 | a2 | +0.318 | +0.000 | +0.021 |
| 63 | a1 | -0.289 | +0.002 | -0.027 | a1 | +0.220 | +0.006 | +0.024 | leak (a2: d ans -0.688, dJ +0.000, dB -0.130 on tune) |  |  |  | leak (a2: d ans +0.312, dJ -0.062, dB +0.161 on tune) |  |  |  |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf-ce-bidir).

