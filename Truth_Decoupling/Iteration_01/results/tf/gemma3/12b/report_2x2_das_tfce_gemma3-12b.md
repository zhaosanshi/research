# gemma3-12b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tfce_C`

E sweep under `/workspace/Truth_Decoupling/Iteration_01/results/tf/gemma3/12b/E23/gemma3-12b`, C sweep under `/workspace/Truth_Decoupling/Iteration_01/results/tf/gemma3/12b/E23/gemma3-12b` (E estimator of the C sweep: das variant tf-ce-bidir). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.665, J +0.578, AUROC 0.870, B 0.548.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L20_a3 | 86 feasible of 96 | -0.004 [-0.021, +0.013] | +0.009 [-0.008, +0.026] | +0.009 [+0.002, +0.016] | -0.009 [-0.017, +0.000] |
| C prompt-contrast, toward abstention | Cp_L44_abst_a1 | 31 feasible of 96 | -0.437 [-0.478, -0.393] | -0.002 [-0.023, +0.021] | -0.001 [-0.004, +0.001] | -0.033 [-0.044, -0.023] |
| C prompt-contrast, toward answering | Cp_L46_ans_a1 | 17 feasible of 96 | +0.140 [+0.112, +0.170] | +0.002 [-0.019, +0.023] | +0.001 [-0.001, +0.003] | +0.027 [+0.018, +0.037] |
| C self-report, toward abstention | Cs_L29_abst_a2 | 33 feasible of 96 | -0.665 [-0.703, -0.624] | +0.013 [-0.009, +0.038] | +0.003 [-0.003, +0.011] | +0.009 [-0.003, +0.020] |
| C self-report, toward answering | Cs_L43_ans_a2 | 10 feasible of 96 | +0.335 [+0.297, +0.376] | -0.006 [-0.017, +0.006] | -0.002 [-0.006, +0.001] | +0.003 [-0.003, +0.009] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.000 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 1 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 2 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | a2 | +0.002 | +0.000 | +0.000 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 3 | a1 | +0.000 | -0.004 | +0.000 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | a1 | +0.000 | +0.000 | +0.000 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 4 | a2 | +0.000 | -0.004 | -0.002 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 5 | a2 | +0.000 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | a1 | +0.000 | +0.000 | +0.000 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 6 | a1 | +0.000 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 7 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 8 | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.000 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 9 | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 10 | leak (a1: d ans +0.000, dJ +0.000, dB +0.010 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 11 | leak (a1: d ans +0.000, dJ +0.010, dB +0.036 on tune) |  |  |  | leak (a2: d ans -0.016, dJ -0.005, dB -0.060 on tune) |  |  |  | a1 | +0.000 | -0.004 | -0.002 | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  |
| 12 | a2 | +0.009 | +0.009 | +0.014 | leak (a1: d ans -0.005, dJ -0.016, dB -0.018 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 13 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 14 | leak (a1: d ans +0.000, dJ -0.010, dB +0.010 on tune) |  |  |  | leak (a2: d ans +0.000, dJ -0.005, dB -0.018 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  |
| 15 | a1 | -0.004 | +0.000 | +0.000 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | a1 | -0.004 | -0.009 | -0.005 | leak (a1: d ans +0.000, dJ -0.010, dB +0.016 on tune) |  |  |  |
| 16 | leak (a2: d ans -0.052, dJ -0.089, dB -0.138 on tune) |  |  |  | a1 | +0.019 | +0.006 | +0.016 | leak (a1: d ans +0.000, dJ -0.016, dB +0.023 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.005, dB -0.018 on tune) |  |  |  |
| 17 | leak (a1: d ans +0.005, dJ +0.000, dB +0.036 on tune) |  |  |  | leak (a1: d ans -0.021, dJ -0.010, dB -0.036 on tune) |  |  |  | leak (a1: d ans +0.005, dJ +0.005, dB +0.039 on tune) |  |  |  | leak (a1: d ans -0.016, dJ -0.010, dB -0.036 on tune) |  |  |  |
| 18 | a2 | -0.011 | -0.008 | -0.025 | a2 | +0.023 | +0.004 | +0.017 | leak (a1: d ans +0.010, dJ -0.010, dB +0.031 on tune) |  |  |  | leak (a1: d ans -0.016, dJ -0.005, dB -0.034 on tune) |  |  |  |
| 19 | a2 | -0.011 | +0.000 | -0.021 | leak (a1: d ans +0.000, dJ -0.005, dB +0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.010, dB +0.010 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 20 | a2 | -0.034 | +0.002 | -0.010 | leak (a2: d ans +0.036, dJ -0.021, dB +0.042 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.034 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.010, dB -0.026 on tune) |  |  |  |
| 21 | leak (a1: d ans +0.005, dJ -0.005, dB +0.039 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.021, dB -0.036 on tune) |  |  |  | a1 | +0.000 | +0.000 | +0.008 | leak (a1: d ans +0.000, dJ +0.000, dB -0.010 on tune) |  |  |  |
| 22 | leak (a2: d ans +0.005, dJ +0.005, dB +0.044 on tune) |  |  |  | a2 | +0.013 | -0.019 | -0.013 | leak (a1: d ans +0.000, dJ +0.005, dB +0.044 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.016, dB -0.039 on tune) |  |  |  |
| 23 | a1 | -0.036 | +0.009 | -0.029 | a1 | +0.026 | -0.002 | +0.024 | leak (a1: d ans +0.026, dJ +0.000, dB +0.068 on tune) |  |  |  | leak (a1: d ans -0.036, dJ -0.021, dB -0.078 on tune) |  |  |  |
| 24 | a2 | -0.049 | +0.015 | -0.009 | leak (a2: d ans +0.021, dJ -0.026, dB +0.023 on tune) |  |  |  | leak (a1: d ans +0.057, dJ -0.052, dB +0.120 on tune) |  |  |  | leak (a1: d ans -0.036, dJ -0.068, dB -0.138 on tune) |  |  |  |
| 25 | a2 | -0.062 | +0.021 | +0.005 | leak (a1: d ans -0.021, dJ +0.000, dB -0.031 on tune) |  |  |  | leak (a1: d ans +0.005, dJ -0.005, dB +0.091 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.073, dB -0.125 on tune) |  |  |  |
| 26 | a2 | -0.191 | +0.011 | -0.021 | leak (a2: d ans +0.083, dJ -0.031, dB +0.016 on tune) |  |  |  | a1 | -0.112 | +0.017 | +0.009 | leak (a2: d ans +0.125, dJ -0.156, dB -0.224 on tune) |  |  |  |
| 27 | leak (a2: d ans -0.411, dJ -0.042, dB -0.120 on tune) |  |  |  | a1 | +0.079 | -0.004 | +0.025 | leak (a2: d ans -0.635, dJ -0.443, dB +0.362 on tune) |  |  |  | a2 | +0.318 | -0.023 | +0.013 |
| 28 | leak (a2: d ans -0.490, dJ -0.042, dB -0.104 on tune) |  |  |  | leak (a2: d ans +0.135, dJ -0.047, dB +0.086 on tune) |  |  |  | leak (a2: d ans -0.635, dJ -0.073, dB +0.130 on tune) |  |  |  | a1 | +0.181 | -0.013 | +0.031 |
| 29 | a1 | -0.229 | +0.004 | -0.028 | a1 | +0.100 | -0.004 | +0.025 | a2 | -0.665 | +0.013 | +0.009 | a1 | +0.174 | -0.009 | +0.018 |
| 30 | a1 | -0.197 | +0.006 | -0.033 | leak (a2: d ans +0.240, dJ -0.026, dB +0.096 on tune) |  |  |  | a2 | -0.665 | +0.002 | -0.027 | a1 | +0.219 | -0.006 | -0.009 |
| 31 | a1 | -0.206 | -0.002 | -0.016 | a1 | +0.104 | -0.006 | +0.020 | a1 | -0.645 | +0.004 | -0.026 | leak (a2: d ans +0.365, dJ -0.026, dB -0.039 on tune) |  |  |  |
| 32 | leak (a2: d ans -0.635, dJ -0.062, dB -0.115 on tune) |  |  |  | leak (a2: d ans +0.344, dJ -0.052, dB +0.120 on tune) |  |  |  | a1 | -0.662 | +0.000 | -0.025 | leak (a2: d ans +0.365, dJ -0.026, dB +0.013 on tune) |  |  |  |
| 33 | leak (a2: d ans -0.635, dJ -0.078, dB -0.180 on tune) |  |  |  | leak (a2: d ans +0.344, dJ -0.094, dB +0.141 on tune) |  |  |  | leak (a2: d ans -0.635, dJ -0.052, dB -0.062 on tune) |  |  |  | leak (a2: d ans +0.365, dJ -0.068, dB -0.102 on tune) |  |  |  |
| 34 | leak (a2: d ans -0.635, dJ -0.094, dB -0.161 on tune) |  |  |  | leak (a2: d ans +0.349, dJ -0.047, dB +0.117 on tune) |  |  |  | a1 | -0.665 | +0.000 | -0.026 | leak (a2: d ans +0.365, dJ +0.021, dB -0.016 on tune) |  |  |  |
| 35 | leak (a2: d ans -0.635, dJ -0.089, dB -0.180 on tune) |  |  |  | leak (a2: d ans +0.365, dJ -0.042, dB +0.099 on tune) |  |  |  | a1 | -0.665 | -0.006 | -0.024 | leak (a2: d ans +0.365, dJ -0.068, dB -0.133 on tune) |  |  |  |
| 36 | leak (a2: d ans -0.635, dJ -0.068, dB -0.133 on tune) |  |  |  | a1 | +0.130 | +0.002 | +0.022 | leak (a1: d ans -0.635, dJ -0.021, dB -0.042 on tune) |  |  |  | leak (a2: d ans +0.365, dJ -0.057, dB -0.122 on tune) |  |  |  |
| 37 | a1 | -0.346 | -0.002 | -0.029 | a1 | +0.127 | +0.002 | +0.018 | a1 | -0.665 | -0.004 | -0.026 | a1 | +0.287 | -0.006 | -0.020 |
| 38 | a1 | -0.338 | -0.006 | -0.022 | a1 | +0.115 | +0.002 | +0.016 | a1 | -0.665 | -0.002 | -0.003 | leak (a2: d ans +0.365, dJ -0.047, dB +0.065 on tune) |  |  |  |
| 39 | a1 | -0.344 | -0.004 | -0.025 | a1 | +0.115 | +0.004 | +0.019 | a1 | -0.665 | -0.004 | -0.006 | leak (a2: d ans +0.365, dJ -0.036, dB -0.096 on tune) |  |  |  |
| 40 | a1 | -0.353 | +0.000 | -0.030 | a1 | +0.115 | +0.004 | +0.021 | a1 | -0.665 | -0.004 | -0.008 | leak (a2: d ans +0.365, dJ -0.036, dB -0.081 on tune) |  |  |  |
| 41 | a1 | -0.301 | -0.002 | -0.035 | a1 | +0.119 | +0.002 | +0.022 | a1 | -0.665 | -0.006 | -0.007 | leak (a2: d ans +0.365, dJ -0.031, dB -0.083 on tune) |  |  |  |
| 42 | leak (a2: d ans -0.635, dJ -0.062, dB -0.125 on tune) |  |  |  | a1 | +0.125 | +0.006 | +0.020 | a1 | -0.665 | -0.008 | -0.006 | leak (a2: d ans +0.365, dJ -0.120, dB -0.164 on tune) |  |  |  |
| 43 | leak (a2: d ans -0.635, dJ -0.109, dB -0.195 on tune) |  |  |  | leak (a2: d ans +0.365, dJ -0.057, dB +0.122 on tune) |  |  |  | leak (a1: d ans -0.635, dJ -0.021, dB -0.042 on tune) |  |  |  | a2 | +0.335 | -0.006 | +0.003 |
| 44 | a1 | -0.437 | -0.002 | -0.033 | a1 | +0.130 | +0.009 | +0.027 | a1 | -0.665 | -0.006 | -0.016 | a1 | +0.333 | +0.000 | -0.004 |
| 45 | leak (a2: d ans -0.635, dJ -0.115, dB -0.177 on tune) |  |  |  | leak (a2: d ans +0.365, dJ -0.062, dB +0.130 on tune) |  |  |  | a1 | -0.665 | -0.009 | -0.012 | a1 | +0.333 | -0.006 | -0.027 |
| 46 | leak (a2: d ans -0.635, dJ -0.109, dB -0.190 on tune) |  |  |  | a1 | +0.140 | +0.002 | +0.027 | leak (a1: d ans -0.635, dJ -0.031, dB -0.073 on tune) |  |  |  | leak (a1: d ans +0.365, dJ -0.031, dB +0.104 on tune) |  |  |  |
| 47 | a1 | -0.435 | -0.002 | -0.022 | a1 | +0.138 | +0.002 | +0.024 | leak (a1: d ans -0.635, dJ -0.062, dB -0.125 on tune) |  |  |  | leak (a1: d ans +0.365, dJ -0.057, dB +0.122 on tune) |  |  |  |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf-ce-bidir).

