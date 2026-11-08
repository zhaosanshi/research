# mistral-7b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tfce_C`

E sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/mistral-7b`, C sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/mistral-7b` (E estimator of the C sweep: das variant tf-ce-bidir). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.641, J +0.448, AUROC 0.760, B 0.602.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L13_a3 | 55 feasible of 64 | -0.002 [-0.017, +0.013] | +0.055 [+0.026, +0.085] | +0.018 [+0.009, +0.027] | -0.020 [-0.035, -0.005] |
| C prompt-contrast, toward abstention | Cp_L17_abst_a2 | 30 feasible of 64 | -0.042 [-0.059, -0.026] | -0.009 [-0.032, +0.013] | +0.009 [+0.002, +0.015] | -0.033 [-0.044, -0.023] |
| C prompt-contrast, toward answering | Cp_L21_ans_a2 | 39 feasible of 64 | +0.059 [+0.040, +0.079] | +0.000 [-0.013, +0.013] | -0.005 [-0.008, -0.002] | +0.013 [+0.007, +0.021] |
| C self-report, toward abstention | Cs_L18_abst_a2 | 9 feasible of 64 | -0.631 [-0.671, -0.592] | +0.019 [-0.006, +0.045] | +0.012 [+0.006, +0.018] | +0.042 [+0.027, +0.056] |
| C self-report, toward answering | Cs_L20_ans_a2 | 1 feasible of 64 | +0.359 [+0.319, +0.399] | -0.000 [-0.023, +0.025] | -0.005 [-0.011, +0.001] | +0.042 [+0.029, +0.055] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | a2 | +0.000 | +0.002 | +0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 1 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | a1 | +0.000 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 2 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | a1 | +0.000 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.010, dB -0.005 on tune) |  |  |  |
| 3 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.010, dB -0.005 on tune) |  |  |  |
| 4 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 5 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 6 | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 7 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | a2 | +0.002 | +0.002 | +0.005 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.010, dB +0.000 on tune) |  |  |  |
| 8 | a2 | -0.008 | -0.002 | -0.010 | a2 | +0.006 | -0.004 | +0.017 | leak (a1: d ans +0.000, dJ -0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 9 | leak (a1: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  | a2 | +0.002 | -0.000 | +0.017 | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB -0.005 on tune) |  |  |  |
| 10 | a1 | -0.006 | +0.002 | -0.009 | a1 | +0.006 | +0.002 | +0.016 | leak (a1: d ans +0.000, dJ +0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  |
| 11 | a1 | -0.008 | +0.008 | -0.013 | a1 | +0.006 | -0.011 | +0.025 | leak (a1: d ans +0.000, dJ +0.005, dB +0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  |
| 12 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | +0.000 | -0.008 | +0.000 | leak (a1: d ans +0.016, dJ -0.010, dB +0.016 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.005, dB -0.013 on tune) |  |  |  |
| 13 | leak (a2: d ans +0.010, dJ -0.016, dB +0.039 on tune) |  |  |  | a1 | -0.004 | -0.006 | -0.014 | leak (a1: d ans +0.062, dJ -0.010, dB +0.062 on tune) |  |  |  | leak (a1: d ans -0.031, dJ +0.000, dB -0.052 on tune) |  |  |  |
| 14 | leak (a2: d ans -0.005, dJ -0.010, dB +0.068 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.010, dB -0.021 on tune) |  |  |  | leak (a1: d ans +0.109, dJ -0.068, dB +0.122 on tune) |  |  |  | leak (a1: d ans -0.042, dJ -0.026, dB -0.112 on tune) |  |  |  |
| 15 | leak (a2: d ans -0.010, dJ +0.026, dB +0.013 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.021, dB -0.010 on tune) |  |  |  | leak (a1: d ans +0.036, dJ +0.010, dB +0.078 on tune) |  |  |  | leak (a2: d ans +0.021, dJ -0.057, dB -0.128 on tune) |  |  |  |
| 16 | a2 | -0.025 | -0.004 | -0.028 | a1 | +0.004 | -0.008 | -0.002 | a1 | -0.119 | +0.026 | +0.004 | leak (a2: d ans +0.276, dJ -0.078, dB +0.029 on tune) |  |  |  |
| 17 | a2 | -0.042 | -0.009 | -0.033 | a1 | +0.021 | -0.002 | +0.018 | leak (a2: d ans -0.458, dJ -0.146, dB +0.255 on tune) |  |  |  | leak (a2: d ans +0.375, dJ -0.094, dB +0.089 on tune) |  |  |  |
| 18 | a1 | -0.008 | -0.013 | -0.018 | a1 | +0.019 | -0.002 | +0.024 | a2 | -0.631 | +0.019 | +0.042 | leak (a2: d ans +0.375, dJ -0.057, dB +0.096 on tune) |  |  |  |
| 19 | a2 | -0.013 | -0.009 | -0.027 | a1 | +0.023 | +0.000 | +0.021 | leak (a2: d ans -0.625, dJ -0.010, dB -0.089 on tune) |  |  |  | leak (a1: d ans +0.375, dJ -0.031, dB +0.068 on tune) |  |  |  |
| 20 | a2 | -0.017 | -0.009 | -0.016 | a1 | +0.021 | +0.006 | +0.012 | leak (a2: d ans -0.625, dJ -0.016, dB -0.081 on tune) |  |  |  | a2 | +0.359 | -0.000 | +0.042 |
| 21 | a2 | -0.011 | -0.002 | -0.009 | a2 | +0.059 | +0.000 | +0.013 | leak (a2: d ans -0.625, dJ -0.016, dB -0.081 on tune) |  |  |  | leak (a1: d ans +0.375, dJ -0.021, dB +0.068 on tune) |  |  |  |
| 22 | a2 | -0.023 | -0.008 | -0.015 | a1 | +0.021 | +0.004 | +0.009 | leak (a2: d ans -0.625, dJ +0.005, dB -0.096 on tune) |  |  |  | leak (a1: d ans +0.375, dJ -0.026, dB +0.060 on tune) |  |  |  |
| 23 | a2 | -0.015 | -0.008 | -0.015 | a1 | +0.019 | +0.002 | +0.009 | leak (a2: d ans -0.625, dJ +0.010, dB -0.104 on tune) |  |  |  | leak (a1: d ans +0.375, dJ -0.036, dB +0.055 on tune) |  |  |  |
| 24 | a2 | -0.019 | -0.008 | -0.013 | a1 | +0.019 | -0.002 | +0.007 | leak (a2: d ans -0.625, dJ +0.010, dB -0.130 on tune) |  |  |  | leak (a1: d ans +0.375, dJ -0.042, dB +0.052 on tune) |  |  |  |
| 25 | a2 | -0.026 | -0.006 | -0.005 | a2 | +0.034 | +0.000 | +0.021 | leak (a2: d ans -0.625, dJ +0.010, dB -0.146 on tune) |  |  |  | leak (a1: d ans +0.375, dJ -0.042, dB +0.052 on tune) |  |  |  |
| 26 | a2 | -0.021 | +0.000 | +0.000 | a2 | +0.023 | -0.000 | +0.008 | a1 | -0.641 | -0.019 | -0.030 | leak (a1: d ans +0.375, dJ -0.036, dB +0.044 on tune) |  |  |  |
| 27 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.021 | -0.004 | -0.002 | a1 | -0.641 | -0.017 | -0.031 | leak (a1: d ans +0.375, dJ -0.042, dB +0.042 on tune) |  |  |  |
| 28 | a2 | -0.015 | +0.002 | +0.005 | a2 | +0.009 | -0.006 | -0.003 | a1 | -0.641 | -0.015 | -0.026 | leak (a1: d ans +0.375, dJ -0.026, dB +0.034 on tune) |  |  |  |
| 29 | a2 | -0.021 | -0.006 | -0.005 | a2 | +0.011 | -0.006 | +0.005 | a1 | -0.641 | -0.019 | -0.030 | leak (a1: d ans +0.375, dJ -0.031, dB +0.047 on tune) |  |  |  |
| 30 | a2 | -0.019 | -0.000 | +0.006 | a2 | +0.023 | +0.002 | +0.003 | a1 | -0.641 | -0.019 | -0.030 | leak (a1: d ans +0.375, dJ -0.031, dB +0.047 on tune) |  |  |  |
| 31 | a2 | -0.013 | +0.002 | +0.001 | a2 | +0.013 | +0.000 | +0.000 | a1 | -0.641 | -0.015 | -0.036 | leak (a1: d ans +0.375, dJ -0.031, dB +0.047 on tune) |  |  |  |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf-ce-bidir).

