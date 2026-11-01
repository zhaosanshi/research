# gemma3-4b: the 2x2 from E sweep `meandiff_full` and C sweep `meandiff_C`

E sweep under `results_das_tf/E23/gemma3-4b`, C sweep under `results_das_tf/E23/gemma3-4b` (E estimator of the C sweep: meandiff). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.747, J +0.439, AUROC 0.792, B 0.601.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L16_a2 | 53 feasible of 68 | -0.011 [-0.028, +0.004] | +0.055 [+0.025, +0.085] | +0.029 [+0.022, +0.036] | -0.035 [-0.052, -0.018] |
| C prompt-contrast, toward abstention | Cp_L23_abst_a2 | 13 feasible of 68 | -0.694 [-0.730, -0.654] | +0.004 [-0.015, +0.023] | -0.005 [-0.013, +0.002] | -0.026 [-0.037, -0.016] |
| C prompt-contrast, toward answering | Cp_L33_ans_a1 | 11 feasible of 68 | +0.253 [+0.219, +0.289] | +0.002 [-0.026, +0.028] | -0.003 [-0.006, +0.001] | +0.056 [+0.043, +0.070] |
| C self-report, toward abstention | Cs_L23_abst_a2 | 12 feasible of 68 | -0.747 [-0.781, -0.711] | +0.028 [+0.004, +0.053] | +0.000 [-0.006, +0.006] | +0.020 [+0.008, +0.032] |
| C self-report, toward answering | Cs_L23_ans_a1 | 16 feasible of 68 | +0.251 [+0.217, +0.289] | +0.006 [-0.004, +0.017] | -0.005 [-0.009, -0.000] | +0.001 [-0.005, +0.007] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 1 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 2 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 3 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  | a1 | +0.000 | +0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  |
| 4 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 5 | a2 | -0.002 | +0.008 | -0.004 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 6 | a2 | -0.004 | +0.013 | -0.007 | a2 | +0.011 | -0.004 | -0.002 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 7 | leak (a1: d ans +0.016, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 8 | leak (a1: d ans +0.000, dJ +0.010, dB -0.016 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 9 | leak (a1: d ans +0.005, dJ +0.010, dB -0.005 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.002 | +0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 10 | leak (a1: d ans +0.016, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.005, dB -0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 11 | a2 | -0.023 | +0.000 | +0.028 | leak (a1: d ans +0.000, dJ -0.005, dB -0.039 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 12 | leak (a2: d ans -0.057, dJ +0.026, dB -0.008 on tune) |  |  |  | leak (a2: d ans +0.016, dJ -0.036, dB -0.008 on tune) |  |  |  | a2 | +0.002 | +0.009 | -0.005 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  |
| 13 | a2 | -0.123 | -0.002 | +0.001 | leak (a2: d ans +0.078, dJ -0.130, dB +0.138 on tune) |  |  |  | a1 | +0.000 | +0.008 | -0.004 | a1 | -0.002 | +0.002 | +0.005 |
| 14 | a2 | -0.051 | +0.013 | +0.024 | a2 | +0.009 | -0.008 | -0.008 | a1 | -0.008 | +0.013 | -0.010 | a2 | +0.021 | +0.013 | +0.027 |
| 15 | leak (a1: d ans -0.010, dJ +0.021, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.010, dJ -0.036, dB -0.013 on tune) |  |  |  | leak (a2: d ans -0.073, dJ +0.057, dB -0.055 on tune) |  |  |  | a1 | +0.034 | -0.009 | +0.033 |
| 16 | leak (a2: d ans -0.443, dJ -0.073, dB -0.286 on tune) |  |  |  | leak (a2: d ans +0.229, dJ -0.312, dB +0.318 on tune) |  |  |  | leak (a2: d ans -0.464, dJ -0.021, dB -0.240 on tune) |  |  |  | leak (a2: d ans +0.198, dJ -0.130, dB +0.201 on tune) |  |  |  |
| 17 | leak (a2: d ans -0.052, dJ +0.026, dB -0.029 on tune) |  |  |  | leak (a2: d ans +0.130, dJ -0.089, dB +0.169 on tune) |  |  |  | leak (a2: d ans -0.260, dJ +0.026, dB -0.102 on tune) |  |  |  | leak (a2: d ans +0.172, dJ -0.094, dB +0.141 on tune) |  |  |  |
| 18 | leak (a2: d ans -0.161, dJ -0.047, dB -0.201 on tune) |  |  |  | a1 | +0.049 | +0.004 | +0.028 | leak (a2: d ans -0.385, dJ -0.036, dB -0.154 on tune) |  |  |  | leak (a2: d ans +0.182, dJ -0.125, dB +0.151 on tune) |  |  |  |
| 19 | leak (a1: d ans +0.010, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a2: d ans +0.042, dJ -0.094, dB -0.161 on tune) |  |  |  | leak (a2: d ans -0.328, dJ -0.005, dB -0.096 on tune) |  |  |  | leak (a2: d ans +0.219, dJ -0.042, dB +0.115 on tune) |  |  |  |
| 20 | a1 | +0.011 | +0.028 | +0.024 | leak (a2: d ans +0.094, dJ -0.068, dB -0.096 on tune) |  |  |  | a2 | -0.242 | +0.009 | -0.010 | a1 | +0.110 | -0.002 | +0.031 |
| 21 | a2 | -0.091 | +0.028 | +0.005 | a1 | +0.083 | +0.002 | +0.043 | leak (a2: d ans -0.484, dJ -0.010, dB -0.094 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.047, dB +0.122 on tune) |  |  |  |
| 22 | leak (a2: d ans -0.125, dJ -0.031, dB -0.156 on tune) |  |  |  | leak (a2: d ans +0.240, dJ -0.120, dB +0.206 on tune) |  |  |  | leak (a2: d ans -0.552, dJ +0.000, dB -0.083 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.089, dB +0.164 on tune) |  |  |  |
| 23 | a2 | -0.694 | +0.004 | -0.026 | a1 | +0.153 | +0.011 | +0.030 | a2 | -0.747 | +0.028 | +0.020 | a1 | +0.251 | +0.006 | +0.001 |
| 24 | a1 | -0.316 | +0.011 | -0.032 | a1 | +0.217 | +0.015 | +0.038 | leak (a1: d ans -0.729, dJ +0.021, dB -0.021 on tune) |  |  |  | a1 | +0.253 | +0.011 | +0.013 |
| 25 | leak (a2: d ans -0.729, dJ -0.068, dB -0.190 on tune) |  |  |  | leak (a1: d ans +0.271, dJ -0.031, dB +0.083 on tune) |  |  |  | leak (a1: d ans -0.729, dJ -0.005, dB -0.060 on tune) |  |  |  | a1 | +0.253 | +0.006 | +0.033 |
| 26 | leak (a2: d ans -0.729, dJ -0.036, dB -0.138 on tune) |  |  |  | a1 | +0.250 | +0.006 | +0.046 | a1 | -0.747 | +0.002 | -0.027 | a1 | +0.253 | +0.011 | +0.026 |
| 27 | leak (a2: d ans -0.729, dJ -0.073, dB -0.292 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.125, dB +0.188 on tune) |  |  |  | leak (a1: d ans -0.729, dJ -0.005, dB -0.065 on tune) |  |  |  | a1 | +0.253 | -0.002 | +0.048 |
| 28 | leak (a2: d ans -0.729, dJ -0.062, dB -0.219 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.089, dB +0.143 on tune) |  |  |  | leak (a1: d ans -0.729, dJ -0.005, dB -0.070 on tune) |  |  |  | a1 | +0.253 | +0.008 | +0.045 |
| 29 | leak (a2: d ans -0.729, dJ -0.062, dB -0.172 on tune) |  |  |  | a1 | +0.246 | +0.006 | +0.050 | a1 | -0.747 | +0.011 | -0.030 | a1 | +0.253 | +0.008 | +0.040 |
| 30 | leak (a2: d ans -0.729, dJ -0.068, dB -0.237 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.177, dB +0.250 on tune) |  |  |  | leak (a1: d ans -0.729, dJ -0.005, dB -0.055 on tune) |  |  |  | a1 | +0.253 | +0.004 | +0.047 |
| 31 | leak (a2: d ans -0.729, dJ -0.078, dB -0.258 on tune) |  |  |  | leak (a1: d ans +0.271, dJ -0.031, dB +0.083 on tune) |  |  |  | leak (a1: d ans -0.729, dJ -0.010, dB -0.052 on tune) |  |  |  | a1 | +0.253 | +0.004 | +0.049 |
| 32 | leak (a2: d ans -0.729, dJ -0.083, dB -0.234 on tune) |  |  |  | leak (a1: d ans +0.271, dJ -0.016, dB +0.076 on tune) |  |  |  | leak (a1: d ans -0.729, dJ +0.005, dB -0.060 on tune) |  |  |  | leak (a1: d ans +0.271, dJ +0.000, dB +0.052 on tune) |  |  |  |
| 33 | leak (a2: d ans -0.729, dJ -0.031, dB -0.151 on tune) |  |  |  | a1 | +0.253 | +0.002 | +0.056 | leak (a1: d ans -0.729, dJ +0.000, dB -0.089 on tune) |  |  |  | leak (a1: d ans +0.271, dJ -0.036, dB +0.086 on tune) |  |  |  |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator meandiff).

