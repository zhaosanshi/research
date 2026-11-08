# gemma3-4b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tfce_C`

E sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/gemma3-4b`, C sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/gemma3-4b` (E estimator of the C sweep: das variant tf-ce-bidir). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.747, J +0.439, AUROC 0.792, B 0.601.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L12_a3 | 44 feasible of 68 | -0.008 [-0.023, +0.008] | +0.034 [+0.006, +0.062] | +0.032 [+0.025, +0.039] | -0.006 [-0.022, +0.009] |
| C prompt-contrast, toward abstention | Cp_L25_abst_a2 | 21 feasible of 68 | -0.747 [-0.781, -0.711] | +0.004 [-0.019, +0.026] | +0.004 [-0.003, +0.010] | -0.040 [-0.053, -0.027] |
| C prompt-contrast, toward answering | Cp_L23_ans_a2 | 22 feasible of 68 | +0.253 [+0.219, +0.289] | +0.004 [-0.017, +0.023] | -0.016 [-0.023, -0.010] | +0.026 [+0.017, +0.037] |
| C self-report, toward abstention | Cs_L26_abst_a2 | 25 feasible of 68 | -0.747 [-0.781, -0.711] | +0.013 [-0.004, +0.030] | +0.007 [+0.002, +0.012] | +0.009 [+0.000, +0.017] |
| C self-report, toward answering | Cs_L28_ans_a1 | 19 feasible of 68 | +0.253 [+0.219, +0.289] | +0.009 [-0.002, +0.023] | -0.002 [-0.006, +0.001] | -0.005 [-0.011, +0.001] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | a2 | +0.000 | +0.004 | -0.002 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 1 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.002 | -0.002 | +0.003 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  |
| 2 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 3 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 4 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.000 | +0.004 | -0.004 | a2 | +0.008 | +0.002 | +0.005 |
| 5 | a2 | -0.002 | +0.009 | -0.005 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a2: d ans -0.021, dJ +0.026, dB -0.029 on tune) |  |  |  | a2 | +0.026 | -0.002 | +0.031 |
| 6 | leak (a1: d ans +0.010, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.016, dJ +0.021, dB -0.016 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a2 | +0.000 | -0.002 | +0.005 |
| 7 | a1 | -0.009 | +0.009 | -0.016 | a1 | +0.015 | +0.013 | +0.020 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.010, dB +0.000 on tune) |  |  |  |
| 8 | leak (a2: d ans +0.062, dJ -0.057, dB +0.039 on tune) |  |  |  | leak (a1: d ans -0.052, dJ +0.000, dB -0.078 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  |
| 9 | leak (a1: d ans +0.000, dJ +0.010, dB -0.005 on tune) |  |  |  | leak (a2: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | a1 | +0.000 | +0.006 | -0.005 | a2 | +0.006 | +0.006 | +0.010 |
| 10 | leak (a1: d ans +0.010, dJ +0.010, dB -0.005 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB -0.005 on tune) |  |  |  | a1 | -0.006 | +0.011 | -0.009 | a2 | +0.013 | +0.011 | +0.025 |
| 11 | a2 | -0.013 | +0.002 | +0.009 | a1 | +0.006 | -0.002 | -0.005 | leak (a1: d ans +0.000, dJ +0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  |
| 12 | leak (a2: d ans -0.052, dJ +0.031, dB -0.021 on tune) |  |  |  | a1 | +0.019 | -0.004 | +0.008 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 13 | leak (a1: d ans +0.120, dJ -0.099, dB +0.180 on tune) |  |  |  | leak (a1: d ans -0.135, dJ -0.073, dB -0.198 on tune) |  |  |  | a2 | -0.013 | +0.008 | -0.017 | a1 | +0.009 | +0.009 | +0.014 |
| 14 | a2 | -0.089 | +0.006 | -0.024 | a1 | +0.013 | -0.002 | +0.005 | leak (a1: d ans +0.005, dJ +0.010, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.010, dJ +0.005, dB -0.013 on tune) |  |  |  |
| 15 | leak (a1: d ans +0.026, dJ +0.010, dB +0.073 on tune) |  |  |  | leak (a1: d ans -0.036, dJ -0.047, dB -0.086 on tune) |  |  |  | leak (a1: d ans +0.052, dJ -0.026, dB +0.112 on tune) |  |  |  | leak (a1: d ans -0.047, dJ -0.047, dB -0.112 on tune) |  |  |  |
| 16 | a1 | -0.026 | +0.030 | -0.011 | a1 | +0.032 | -0.015 | +0.032 | leak (a1: d ans +0.021, dJ -0.010, dB +0.089 on tune) |  |  |  | leak (a1: d ans -0.016, dJ -0.042, dB -0.094 on tune) |  |  |  |
| 17 | a2 | -0.095 | +0.053 | -0.002 | a1 | +0.059 | -0.013 | +0.061 | leak (a1: d ans +0.010, dJ -0.005, dB +0.081 on tune) |  |  |  | leak (a1: d ans -0.021, dJ -0.031, dB -0.109 on tune) |  |  |  |
| 18 | a1 | -0.062 | +0.009 | -0.046 | a2 | +0.055 | -0.008 | -0.038 | a1 | -0.043 | +0.021 | +0.044 | leak (a1: d ans +0.010, dJ -0.026, dB -0.102 on tune) |  |  |  |
| 19 | a1 | -0.085 | +0.002 | -0.043 | a1 | +0.053 | -0.019 | -0.028 | a2 | -0.465 | +0.021 | -0.029 | leak (a1: d ans +0.057, dJ -0.042, dB -0.099 on tune) |  |  |  |
| 20 | leak (a2: d ans -0.651, dJ -0.026, dB -0.049 on tune) |  |  |  | leak (a2: d ans +0.250, dJ -0.094, dB +0.135 on tune) |  |  |  | a1 | -0.420 | +0.017 | -0.035 | leak (a2: d ans +0.271, dJ -0.292, dB -0.469 on tune) |  |  |  |
| 21 | leak (a2: d ans -0.646, dJ -0.026, dB +0.070 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.177, dB +0.245 on tune) |  |  |  | leak (a2: d ans -0.729, dJ -0.245, dB +0.279 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.417, dB -0.615 on tune) |  |  |  |
| 22 | leak (a2: d ans -0.672, dJ -0.078, dB -0.299 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.411, dB +0.383 on tune) |  |  |  | leak (a2: d ans -0.724, dJ -0.130, dB +0.195 on tune) |  |  |  | leak (a2: d ans +0.271, dJ -0.396, dB -0.594 on tune) |  |  |  |
| 23 | a1 | -0.233 | +0.009 | -0.001 | a2 | +0.253 | +0.004 | +0.026 | leak (a2: d ans -0.729, dJ -0.135, dB +0.208 on tune) |  |  |  | leak (a1: d ans +0.271, dJ -0.047, dB -0.112 on tune) |  |  |  |
| 24 | a1 | -0.395 | +0.011 | -0.011 | a1 | +0.223 | +0.013 | +0.024 | a1 | -0.730 | +0.008 | +0.028 | leak (a1: d ans +0.271, dJ +0.005, dB -0.055 on tune) |  |  |  |
| 25 | a2 | -0.747 | +0.004 | -0.040 | leak (a2: d ans +0.271, dJ -0.031, dB +0.073 on tune) |  |  |  | a1 | -0.728 | +0.011 | +0.013 | leak (a1: d ans +0.271, dJ -0.010, dB -0.078 on tune) |  |  |  |
| 26 | a2 | -0.747 | +0.002 | -0.046 | a1 | +0.248 | +0.017 | +0.027 | a2 | -0.747 | +0.013 | +0.009 | leak (a1: d ans +0.271, dJ +0.010, dB -0.062 on tune) |  |  |  |
| 27 | leak (a2: d ans -0.729, dJ -0.052, dB -0.177 on tune) |  |  |  | a1 | +0.250 | +0.006 | +0.046 | leak (a1: d ans -0.729, dJ +0.031, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.271, dJ +0.010, dB -0.052 on tune) |  |  |  |
| 28 | a1 | -0.684 | +0.009 | -0.037 | a1 | +0.248 | +0.008 | +0.042 | a1 | -0.747 | +0.011 | -0.004 | a1 | +0.253 | +0.009 | -0.005 |
| 29 | leak (a2: d ans -0.729, dJ -0.047, dB -0.159 on tune) |  |  |  | a1 | +0.240 | +0.009 | +0.044 | a1 | -0.747 | +0.009 | -0.009 | a1 | +0.253 | +0.009 | -0.016 |
| 30 | leak (a2: d ans -0.729, dJ -0.026, dB -0.122 on tune) |  |  |  | a1 | +0.248 | +0.008 | +0.045 | a1 | -0.747 | +0.015 | +0.008 | a1 | +0.253 | +0.008 | +0.006 |
| 31 | leak (a2: d ans -0.729, dJ -0.083, dB -0.229 on tune) |  |  |  | leak (a1: d ans +0.271, dJ +0.010, dB +0.057 on tune) |  |  |  | a1 | -0.747 | +0.015 | +0.008 | a1 | +0.253 | +0.009 | -0.001 |
| 32 | a2 | -0.747 | +0.009 | -0.031 | a2 | +0.253 | +0.008 | +0.043 | a1 | -0.747 | +0.002 | +0.050 | a1 | +0.253 | -0.002 | -0.024 |
| 33 | leak (a2: d ans -0.729, dJ -0.010, dB -0.094 on tune) |  |  |  | a1 | +0.253 | +0.009 | +0.044 | a1 | -0.747 | +0.019 | +0.023 | a1 | +0.253 | +0.002 | -0.022 |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf-ce-bidir).

