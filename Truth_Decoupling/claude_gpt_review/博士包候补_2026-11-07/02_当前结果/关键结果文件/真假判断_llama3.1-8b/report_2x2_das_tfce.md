# llama3.1-8b: the 2x2 from E sweep `das_tfce_full` and C sweep `das_tfce_C`

E sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/llama3.1-8b`, C sweep under `/workspace/Truth_Decoupling/results_das_tfce/E23/llama3.1-8b` (E estimator of the C sweep: das variant tf-ce-bidir). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.654, J +0.371, AUROC 0.773, B 0.422.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L11_a3 | 60 feasible of 64 | +0.006 [-0.019, +0.030] | +0.091 [+0.060, +0.123] | +0.048 [+0.039, +0.058] | +0.008 [-0.009, +0.025] |
| C prompt-contrast, toward abstention | Cp_L25_abst_a2 | 30 feasible of 64 | -0.597 [-0.639, -0.556] | -0.017 [-0.030, -0.004] | -0.001 [-0.002, +0.000] | -0.012 [-0.019, -0.007] |
| C prompt-contrast, toward answering | Cp_L26_ans_a2 | 27 feasible of 64 | +0.248 [+0.210, +0.285] | +0.013 [-0.011, +0.036] | -0.001 [-0.003, +0.000] | +0.039 [+0.028, +0.051] |
| C self-report, toward abstention | Cs_L23_abst_a2 | 19 feasible of 64 | -0.654 [-0.694, -0.614] | +0.009 [-0.019, +0.036] | -0.002 [-0.004, -0.001] | +0.048 [+0.036, +0.061] |
| C self-report, toward answering | Cs_L25_ans_a2 | 12 feasible of 64 | +0.346 [+0.306, +0.386] | -0.017 [-0.032, -0.002] | -0.002 [-0.004, -0.001] | -0.012 [-0.020, -0.006] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 1 | a1 | +0.000 | -0.006 | +0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 2 | a2 | -0.002 | -0.002 | -0.005 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 3 | a2 | +0.000 | -0.002 | +0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 4 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 5 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a2: d ans +0.000, dJ -0.005, dB -0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 6 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | a2 | +0.000 | -0.013 | -0.010 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 7 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 8 | a2 | +0.000 | -0.008 | -0.008 | a2 | +0.002 | -0.004 | +0.002 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 9 | leak (a1: d ans +0.005, dJ -0.016, dB +0.013 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.005, dB -0.008 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a2: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 10 | leak (a1: d ans +0.016, dJ -0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans -0.010, dJ -0.005, dB -0.008 on tune) |  |  |  | a2 | -0.002 | +0.006 | +0.014 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 11 | leak (a1: d ans +0.047, dJ -0.005, dB +0.018 on tune) |  |  |  | leak (a1: d ans -0.036, dJ -0.021, dB -0.031 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.000, dB +0.005 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 12 | leak (a1: d ans +0.031, dJ -0.010, dB +0.010 on tune) |  |  |  | leak (a1: d ans -0.005, dJ -0.010, dB -0.005 on tune) |  |  |  | leak (a1: d ans +0.031, dJ +0.010, dB +0.068 on tune) |  |  |  | leak (a1: d ans -0.026, dJ -0.036, dB -0.070 on tune) |  |  |  |
| 13 | a2 | -0.059 | +0.009 | +0.022 | a1 | +0.017 | -0.009 | -0.001 | leak (a1: d ans +0.073, dJ +0.016, dB +0.148 on tune) |  |  |  | leak (a1: d ans -0.073, dJ -0.104, dB -0.120 on tune) |  |  |  |
| 14 | a2 | -0.045 | +0.002 | +0.003 | a1 | -0.002 | +0.004 | +0.009 | leak (a1: d ans +0.042, dJ +0.010, dB +0.135 on tune) |  |  |  | leak (a2: d ans -0.005, dJ -0.130, dB -0.154 on tune) |  |  |  |
| 15 | a2 | -0.057 | -0.017 | +0.003 | a2 | +0.028 | +0.008 | +0.011 | leak (a2: d ans -0.141, dJ +0.036, dB +0.122 on tune) |  |  |  | leak (a2: d ans +0.125, dJ -0.021, dB -0.021 on tune) |  |  |  |
| 16 | a1 | -0.032 | -0.013 | -0.009 | a2 | +0.030 | -0.004 | +0.011 | leak (a2: d ans -0.354, dJ +0.031, dB +0.104 on tune) |  |  |  | leak (a2: d ans +0.146, dJ -0.036, dB -0.039 on tune) |  |  |  |
| 17 | a1 | -0.100 | -0.023 | -0.017 | a1 | +0.070 | +0.008 | +0.040 | a1 | -0.276 | +0.026 | +0.047 | a1 | +0.130 | -0.019 | -0.004 |
| 18 | a2 | -0.352 | +0.009 | +0.026 | a2 | +0.123 | +0.000 | +0.008 | leak (a2: d ans -0.615, dJ -0.068, dB +0.294 on tune) |  |  |  | leak (a2: d ans +0.323, dJ -0.120, dB -0.143 on tune) |  |  |  |
| 19 | a2 | -0.350 | +0.004 | +0.009 | a2 | +0.136 | -0.008 | -0.011 | leak (a2: d ans -0.615, dJ -0.052, dB +0.193 on tune) |  |  |  | leak (a2: d ans +0.380, dJ -0.172, dB -0.174 on tune) |  |  |  |
| 20 | a2 | -0.378 | -0.002 | +0.009 | a2 | +0.153 | -0.006 | +0.003 | leak (a2: d ans -0.615, dJ -0.000, dB +0.130 on tune) |  |  |  | leak (a2: d ans +0.380, dJ -0.177, dB -0.188 on tune) |  |  |  |
| 21 | a2 | -0.459 | -0.011 | +0.000 | a2 | +0.193 | +0.006 | +0.018 | a1 | -0.592 | +0.013 | +0.056 | a1 | +0.223 | -0.028 | -0.031 |
| 22 | a2 | -0.529 | -0.017 | -0.009 | a2 | +0.217 | +0.008 | +0.032 | a1 | -0.595 | +0.009 | +0.035 | a1 | +0.229 | -0.025 | -0.033 |
| 23 | a2 | -0.529 | -0.011 | -0.013 | a2 | +0.223 | +0.004 | +0.043 | a2 | -0.654 | +0.009 | +0.048 | a1 | +0.233 | -0.019 | -0.023 |
| 24 | a2 | -0.595 | -0.009 | -0.003 | a2 | +0.234 | +0.011 | +0.026 | a1 | -0.639 | +0.011 | +0.028 | a1 | +0.272 | -0.019 | -0.026 |
| 25 | a2 | -0.597 | -0.017 | -0.012 | a2 | +0.238 | +0.006 | +0.027 | a2 | -0.654 | +0.013 | +0.058 | a2 | +0.346 | -0.017 | -0.012 |
| 26 | a1 | -0.233 | -0.011 | -0.009 | a2 | +0.248 | +0.013 | +0.039 | a2 | -0.654 | -0.009 | +0.005 | a2 | +0.346 | -0.008 | +0.008 |
| 27 | leak (a2: d ans -0.562, dJ -0.021, dB -0.042 on tune) |  |  |  | a2 | +0.255 | +0.013 | +0.050 | a2 | -0.654 | -0.008 | -0.008 | a1 | +0.285 | -0.002 | +0.018 |
| 28 | leak (a2: d ans -0.604, dJ -0.026, dB -0.034 on tune) |  |  |  | leak (a2: d ans +0.281, dJ +0.016, dB +0.065 on tune) |  |  |  | a2 | -0.654 | +0.009 | +0.014 | a1 | +0.291 | +0.004 | +0.017 |
| 29 | leak (a2: d ans -0.609, dJ -0.021, dB -0.042 on tune) |  |  |  | leak (a2: d ans +0.281, dJ +0.010, dB +0.057 on tune) |  |  |  | a2 | -0.654 | +0.004 | +0.009 | a1 | +0.289 | +0.002 | +0.022 |
| 30 | a1 | -0.437 | -0.019 | -0.017 | leak (a2: d ans +0.266, dJ +0.016, dB +0.055 on tune) |  |  |  | a2 | -0.654 | +0.002 | +0.005 | leak (a2: d ans +0.385, dJ +0.021, dB +0.083 on tune) |  |  |  |
| 31 | leak (a2: d ans -0.609, dJ +0.005, dB +0.143 on tune) |  |  |  | leak (a2: d ans +0.276, dJ -0.125, dB -0.172 on tune) |  |  |  | leak (a2: d ans -0.615, dJ -0.135, dB -0.188 on tune) |  |  |  | leak (a2: d ans +0.385, dJ +0.016, dB +0.148 on tune) |  |  |  |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf-ce-bidir).

