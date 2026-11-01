# llama3.1-8b: the 2x2 from E sweep `das_tf_spans` and C sweep `das_tf_C`

E sweep under `results_das_tf/E23/llama3.1-8b`, C sweep under `results_das_tf/E23/llama3.1-8b` (E estimator of the C sweep: das variant tf). Rules re-applied on the tuning half (192 pairs); test half (529 pairs) with pair-bootstrap 95% CIs (2000 resamples, seed 0). Baseline arrays identical across the two sweeps: True. Test baseline: answer rate 0.654, J +0.371, AUROC 0.773, B 0.422.

## Rule-chosen knobs (test half)

| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |
|---|---|---|---|---|---|---|
| E (amplify) | E_L10-11-12_a2 | 32 feasible of 40 | -0.011 [-0.051, +0.026] | +0.149 [+0.100, +0.197] | +0.036 [+0.020, +0.053] | +0.052 [+0.026, +0.078] |
| C prompt-contrast, toward abstention | Cp_L28_abst_a2 | 40 feasible of 64 | -0.637 [-0.677, -0.597] | -0.008 [-0.019, +0.002] | -0.000 [-0.001, +0.001] | -0.002 [-0.008, +0.003] |
| C prompt-contrast, toward answering | Cp_L26_ans_a2 | 35 feasible of 64 | +0.242 [+0.206, +0.278] | +0.008 [-0.004, +0.019] | -0.002 [-0.003, -0.000] | +0.004 [-0.002, +0.010] |
| C self-report, toward abstention | Cs_L19_abst_a2 | 18 feasible of 64 | -0.654 [-0.694, -0.614] | +0.006 [-0.009, +0.023] | +0.000 [-0.002, +0.002] | +0.014 [+0.007, +0.023] |
| C self-report, toward answering | Cs_L21_ans_a2 | 14 feasible of 64 | +0.346 [+0.306, +0.386] | -0.026 [-0.049, -0.004] | -0.005 [-0.007, -0.003] | -0.038 [-0.050, -0.026] |

Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). Both C estimators are shown; the protocol picks the estimator per model on the tuning set.

## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)

| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | a2 | +0.000 | +0.000 | +0.000 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 1 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.000 | -0.006 | -0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 2 | a1 | +0.002 | -0.002 | -0.001 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 3 | a1 | +0.000 | +0.000 | +0.000 | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 4 | a2 | +0.000 | -0.004 | +0.000 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  |
| 5 | a1 | -0.002 | -0.006 | -0.003 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 6 | a2 | +0.000 | -0.006 | -0.005 | a2 | +0.004 | +0.004 | +0.004 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 7 | a2 | +0.000 | -0.008 | -0.004 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 8 | a2 | -0.002 | -0.008 | -0.008 | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 9 | leak (a1: d ans +0.005, dJ +0.000, dB +0.000 on tune) |  |  |  | a2 | +0.002 | +0.004 | +0.004 | leak (a1: d ans +0.005, dJ +0.005, dB +0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 10 | leak (a1: d ans +0.010, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans -0.010, dJ +0.000, dB +0.000 on tune) |  |  |  | a1 | -0.002 | +0.002 | +0.003 | leak (a1: d ans +0.000, dJ -0.005, dB -0.003 on tune) |  |  |  |
| 11 | leak (a1: d ans +0.031, dJ -0.010, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.016, dJ -0.005, dB -0.003 on tune) |  |  |  | leak (a1: d ans +0.000, dJ +0.000, dB +0.000 on tune) |  |  |  | leak (a1: d ans -0.005, dJ +0.000, dB +0.000 on tune) |  |  |  |
| 12 | leak (a1: d ans +0.010, dJ +0.010, dB -0.010 on tune) |  |  |  | a1 | +0.004 | +0.002 | +0.010 | leak (a1: d ans +0.036, dJ +0.016, dB +0.044 on tune) |  |  |  | leak (a1: d ans -0.021, dJ -0.016, dB -0.023 on tune) |  |  |  |
| 13 | a2 | -0.066 | -0.006 | +0.007 | a1 | +0.019 | -0.002 | +0.005 | leak (a1: d ans +0.104, dJ +0.005, dB +0.227 on tune) |  |  |  | leak (a1: d ans -0.083, dJ -0.104, dB -0.135 on tune) |  |  |  |
| 14 | a2 | -0.051 | -0.008 | -0.006 | a1 | -0.002 | -0.002 | +0.010 | leak (a1: d ans +0.068, dJ +0.010, dB +0.161 on tune) |  |  |  | leak (a2: d ans -0.036, dJ -0.172, dB -0.185 on tune) |  |  |  |
| 15 | leak (a2: d ans -0.042, dJ -0.026, dB +0.018 on tune) |  |  |  | a2 | +0.009 | -0.017 | -0.020 | leak (a2: d ans -0.120, dJ -0.005, dB +0.133 on tune) |  |  |  | a1 | +0.028 | -0.025 | -0.035 |
| 16 | a1 | -0.008 | -0.002 | +0.012 | a1 | +0.004 | -0.009 | -0.007 | leak (a2: d ans -0.370, dJ +0.016, dB +0.112 on tune) |  |  |  | a2 | +0.149 | -0.023 | -0.030 |
| 17 | a2 | -0.244 | -0.006 | +0.005 | a2 | +0.091 | +0.009 | +0.048 | a2 | -0.650 | -0.006 | -0.007 | a2 | +0.244 | +0.002 | +0.044 |
| 18 | a2 | -0.357 | +0.002 | +0.022 | a2 | +0.125 | +0.002 | +0.005 | a1 | -0.573 | -0.006 | +0.001 | leak (a2: d ans +0.349, dJ -0.063, dB -0.089 on tune) |  |  |  |
| 19 | a2 | -0.329 | +0.013 | +0.027 | a2 | +0.125 | -0.026 | -0.028 | a2 | -0.654 | +0.006 | +0.014 | leak (a2: d ans +0.380, dJ -0.161, dB -0.159 on tune) |  |  |  |
| 20 | a2 | -0.380 | -0.004 | +0.011 | a2 | +0.151 | -0.008 | -0.006 | a2 | -0.654 | -0.006 | -0.007 | leak (a2: d ans +0.385, dJ -0.130, dB -0.143 on tune) |  |  |  |
| 21 | a2 | -0.435 | +0.011 | +0.030 | a2 | +0.180 | -0.011 | -0.019 | a2 | -0.654 | -0.009 | -0.009 | a2 | +0.346 | -0.026 | -0.038 |
| 22 | a2 | -0.488 | +0.008 | +0.032 | a2 | +0.197 | -0.011 | -0.023 | a1 | -0.605 | -0.008 | -0.006 | a1 | +0.231 | -0.021 | -0.024 |
| 23 | a2 | -0.486 | +0.000 | +0.030 | a2 | +0.206 | -0.013 | -0.016 | a2 | -0.654 | -0.015 | -0.009 | a1 | +0.233 | -0.026 | -0.025 |
| 24 | a2 | -0.575 | +0.023 | +0.049 | a2 | +0.221 | -0.021 | -0.029 | a2 | -0.654 | -0.006 | -0.003 | a1 | +0.267 | -0.019 | -0.025 |
| 25 | a2 | -0.573 | +0.015 | +0.043 | a2 | +0.223 | -0.021 | -0.035 | a2 | -0.654 | -0.013 | -0.016 | a1 | +0.270 | -0.009 | -0.007 |
| 26 | a2 | -0.611 | -0.009 | -0.003 | a2 | +0.242 | +0.008 | +0.004 | leak (a2: d ans -0.615, dJ -0.099, dB -0.138 on tune) |  |  |  | a2 | +0.346 | +0.011 | +0.036 |
| 27 | a2 | -0.607 | -0.013 | -0.012 | a2 | +0.244 | +0.008 | +0.021 | leak (a2: d ans -0.615, dJ -0.099, dB -0.159 on tune) |  |  |  | a1 | +0.280 | +0.017 | +0.046 |
| 28 | a2 | -0.637 | -0.008 | -0.002 | a1 | +0.136 | +0.002 | +0.014 | leak (a2: d ans -0.615, dJ -0.078, dB -0.112 on tune) |  |  |  | leak (a2: d ans +0.385, dJ +0.016, dB +0.096 on tune) |  |  |  |
| 29 | leak (a2: d ans -0.599, dJ -0.021, dB -0.031 on tune) |  |  |  | a2 | +0.261 | +0.011 | +0.038 | leak (a2: d ans -0.615, dJ -0.089, dB -0.148 on tune) |  |  |  | leak (a2: d ans +0.385, dJ +0.026, dB +0.128 on tune) |  |  |  |
| 30 | a1 | -0.427 | -0.013 | -0.012 | a1 | +0.159 | -0.002 | +0.020 | leak (a2: d ans -0.615, dJ -0.135, dB -0.188 on tune) |  |  |  | leak (a2: d ans +0.385, dJ -0.026, dB +0.180 on tune) |  |  |  |
| 31 | leak (a2: d ans -0.609, dJ -0.026, dB -0.070 on tune) |  |  |  | a1 | +0.166 | +0.013 | +0.035 | leak (a2: d ans -0.615, dJ -0.328, dB -0.372 on tune) |  |  |  | leak (a2: d ans +0.385, dJ -0.099, dB +0.372 on tune) |  |  |  |

C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator das, DAS variant tf).

