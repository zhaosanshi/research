# qwen3.5-9b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tfce_C`

E sweep under `/workspace/Truth_Decoupling/results_newmodels_20261112/E23/qwen3.5-9b`, C sweep under `/workspace/Truth_Decoupling/results_newmodels_20261112/E23/qwen3.5-9b` (E estimator of the C sweep: das variant tf-ce-bidir). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.698, J +0.488, AUROC 0.825, B 0.527.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L31_a3 | 64 feasible of 64 | -0.005 [-0.010, -0.001] | +0.087 [+0.055, +0.121] | +0.051 [+0.043, +0.059] | +0.004 [-0.014, +0.021] |
| C prompt-contrast, toward abstention | Cp_L16_abst_a1 | 14 feasible of 64 | -0.046 [-0.064, -0.029] | +0.015 [+0.002, +0.030] | -0.001 [-0.002, +0.001] | -0.013 [-0.021, -0.008] |
| C prompt-contrast, toward answering | Cp_L17_ans_a2 | 48 feasible of 64 | +0.087 [+0.062, +0.111] | +0.011 [-0.002, +0.025] | -0.000 [-0.001, +0.001] | +0.013 [+0.006, +0.021] |
| C self-report, toward abstention | Cs_L28_abst_a2 | 12 feasible of 64 | -0.698 [-0.735, -0.658] | +0.006 [-0.008, +0.019] | +0.004 [+0.003, +0.006] | +0.007 [+0.000, +0.013] |
| C self-report, toward answering | Cs_L31_ans_a1 | 23 feasible of 64 | +0.284 [+0.248, +0.323] | +0.011 [-0.004, +0.026] | -0.000 [-0.001, +0.001] | +0.015 [+0.009, +0.023] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | leak (a1: d ans +0.003, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.004 | +0.002 | -0.001 | a2 | -0.001 | -0.000 | -0.002 |
| 1 | leak (a1: d ans +0.003, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.001 | +0.006 | -0.005 | leak (a2: d ans +0.005, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | -0.001 | -0.002 | -0.003 |
| 2 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.001 | +0.004 | -0.002 | leak (a1: d ans +0.003, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.002 | +0.002 | -0.003 |
| 3 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.003, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.000 | -0.000 | -0.002 |
| 4 | a2 | -0.005 | +0.008 | +0.000 | a1 | +0.001 | +0.004 | +0.000 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.002 | +0.008 | -0.002 |
| 5 | leak (a2: d ans +0.003, dJ -0.010, dB +0.000 on tune) |  |  |  | a1 | +0.005 | +0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.001 | +0.006 | -0.005 |
| 6 | a2 | +0.005 | +0.004 | +0.000 | a2 | +0.007 | +0.002 | -0.003 | leak (a2: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 7 | leak (a1: d ans +0.003, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.012 | +0.004 | +0.000 | a2 | -0.005 | +0.004 | +0.000 | a2 | +0.002 | -0.004 | -0.004 |
| 8 | a1 | -0.001 | +0.004 | +0.002 | a2 | +0.017 | -0.000 | -0.009 | leak (a2: d ans +0.003, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.000 | +0.009 | -0.007 |
| 9 | leak (a1: d ans +0.010, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.017 | +0.009 | -0.005 | leak (a1: d ans +0.005, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.003 | +0.002 | -0.001 |
| 10 | a1 | -0.007 | +0.004 | +0.002 | a2 | +0.022 | +0.006 | -0.009 | leak (a2: d ans +0.005, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.000 | +0.006 | -0.001 |
| 11 | leak (a1: d ans +0.026, dJ +0.005, dB +0.013 on tune) |  |  |  | a2 | +0.013 | +0.009 | -0.010 | a1 | +0.005 | +0.006 | -0.005 | a2 | +0.005 | +0.006 | +0.001 |
| 12 | leak (a1: d ans +0.008, dJ -0.005, dB +0.008 on tune) |  |  |  | a1 | +0.008 | +0.013 | -0.014 | leak (a2: d ans +0.003, dJ -0.000, dB -0.005 on tune) |  |  |  | a1 | +0.001 | -0.000 | -0.002 |
| 13 | a2 | -0.032 | -0.002 | +0.007 | a1 | +0.017 | +0.009 | -0.010 | leak (a2: d ans +0.003, dJ -0.000, dB +0.010 on tune) |  |  |  | a1 | -0.003 | +0.009 | -0.010 |
| 14 | leak (a2: d ans -0.029, dJ -0.031, dB -0.042 on tune) |  |  |  | a1 | +0.017 | +0.006 | -0.005 | leak (a1: d ans +0.018, dJ -0.010, dB +0.010 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  |
| 15 | a1 | -0.014 | +0.004 | -0.002 | a1 | +0.009 | +0.006 | -0.009 | leak (a1: d ans +0.031, dJ +0.000, dB +0.021 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.016, dB -0.018 on tune) |  |  |  |
| 16 | a1 | -0.046 | +0.015 | -0.013 | a2 | +0.075 | +0.006 | +0.001 | leak (a1: d ans +0.060, dJ +0.016, dB +0.044 on tune) |  |  |  | leak (a1: d ans -0.018, dJ -0.026, dB -0.044 on tune) |  |  |  |
| 17 | leak (a2: d ans -0.086, dJ -0.031, dB -0.078 on tune) |  |  |  | a2 | +0.087 | +0.011 | +0.013 | leak (a2: d ans -0.099, dJ -0.135, dB +0.271 on tune) |  |  |  | leak (a2: d ans +0.065, dJ -0.057, dB -0.112 on tune) |  |  |  |
| 18 | leak (a2: d ans -0.081, dJ -0.047, dB -0.076 on tune) |  |  |  | a2 | +0.085 | +0.013 | +0.018 | leak (a2: d ans -0.122, dJ -0.057, dB +0.227 on tune) |  |  |  | leak (a2: d ans +0.089, dJ -0.047, dB -0.091 on tune) |  |  |  |
| 19 | leak (a2: d ans -0.047, dJ -0.062, dB -0.068 on tune) |  |  |  | a2 | +0.061 | +0.008 | +0.009 | leak (a2: d ans -0.227, dJ -0.083, dB +0.266 on tune) |  |  |  | leak (a2: d ans +0.221, dJ -0.036, dB -0.070 on tune) |  |  |  |
| 20 | a1 | -0.006 | +0.011 | -0.015 | a2 | +0.057 | +0.002 | -0.001 | a1 | -0.158 | +0.008 | -0.021 | leak (a2: d ans +0.263, dJ -0.062, dB -0.094 on tune) |  |  |  |
| 21 | leak (a2: d ans -0.039, dJ -0.052, dB -0.094 on tune) |  |  |  | a2 | +0.053 | +0.009 | +0.020 | leak (a2: d ans -0.427, dJ +0.010, dB +0.172 on tune) |  |  |  | leak (a2: d ans +0.263, dJ -0.047, dB -0.039 on tune) |  |  |  |
| 22 | leak (a2: d ans -0.039, dJ -0.016, dB -0.055 on tune) |  |  |  | a2 | +0.048 | +0.006 | +0.020 | leak (a2: d ans -0.576, dJ +0.005, dB +0.174 on tune) |  |  |  | leak (a2: d ans +0.216, dJ -0.099, dB -0.143 on tune) |  |  |  |
| 23 | a1 | -0.009 | +0.017 | -0.014 | a2 | +0.030 | +0.004 | +0.017 | leak (a2: d ans -0.659, dJ -0.062, dB +0.234 on tune) |  |  |  | leak (a2: d ans +0.310, dJ -0.083, dB -0.130 on tune) |  |  |  |
| 24 | leak (a2: d ans -0.034, dJ -0.042, dB -0.089 on tune) |  |  |  | a2 | +0.043 | -0.000 | +0.030 | leak (a2: d ans -0.659, dJ +0.005, dB +0.159 on tune) |  |  |  | leak (a2: d ans +0.310, dJ -0.073, dB -0.120 on tune) |  |  |  |
| 25 | leak (a2: d ans -0.039, dJ -0.026, dB -0.076 on tune) |  |  |  | a2 | +0.038 | +0.013 | +0.024 | leak (a2: d ans -0.659, dJ -0.010, dB +0.141 on tune) |  |  |  | leak (a2: d ans +0.310, dJ -0.089, dB -0.133 on tune) |  |  |  |
| 26 | a1 | -0.010 | +0.008 | -0.015 | a2 | +0.026 | +0.011 | +0.021 | leak (a2: d ans -0.659, dJ +0.005, dB +0.107 on tune) |  |  |  | leak (a2: d ans +0.310, dJ -0.089, dB -0.133 on tune) |  |  |  |
| 27 | a1 | -0.014 | +0.013 | -0.016 | a2 | +0.053 | +0.015 | +0.025 | leak (a2: d ans -0.659, dJ +0.021, dB +0.047 on tune) |  |  |  | leak (a2: d ans +0.336, dJ -0.135, dB -0.188 on tune) |  |  |  |
| 28 | a1 | -0.016 | +0.004 | -0.019 | a2 | +0.051 | +0.015 | +0.028 | a2 | -0.698 | +0.006 | +0.007 | leak (a2: d ans +0.341, dJ -0.125, dB -0.167 on tune) |  |  |  |
| 29 | leak (a2: d ans -0.044, dJ -0.047, dB -0.070 on tune) |  |  |  | a2 | +0.053 | +0.015 | +0.028 | a1 | -0.623 | +0.011 | +0.032 | leak (a2: d ans +0.341, dJ -0.120, dB -0.253 on tune) |  |  |  |
| 30 | leak (a2: d ans -0.049, dJ -0.047, dB -0.086 on tune) |  |  |  | leak (a2: d ans +0.070, dJ +0.010, dB +0.068 on tune) |  |  |  | a1 | -0.698 | +0.004 | +0.002 | a1 | +0.266 | -0.000 | -0.004 |
| 31 | a1 | -0.021 | +0.013 | +0.022 | leak (a2: d ans +0.081, dJ -0.047, dB -0.086 on tune) |  |  |  | a1 | -0.693 | +0.011 | -0.013 | a1 | +0.284 | +0.011 | +0.015 |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf-ce-bidir).

