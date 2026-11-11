# judge_tune_split_tune, judge allenai, 192 questions (DESCRIPTION only; each condition against its own base prompt)

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| E_L10_conf-mdg-p1 `E_L10_conf-mdg-p1` | conf | 2 | 88 / 102 / 0 / 2 | 190 | 46.3% | +0 [-5, +5] | +1 | 189, +0.0 pp, 4 / 4 | 0 / 0 / 0 / 1 | 3.4 |
| E_L10_conf-mdg-p2 `E_L10_conf-mdg-p2` | conf | 2 | 84 / 105 / 1 / 2 | 189 | 44.4% | -4 [-11, +3] | +0 | 188, -1.6 pp, 4 / 7 | 1 / 0 / 0 / 1 | 3.3 |
| E_L10_conf-w-p1 `E_L10_conf-w-p1` | conf | 1 | 102 / 88 / 1 / 1 | 190 | 53.7% | +14 [+5, +23] | +1 | 188, +7.4 pp, 18 / 4 | 1 / 0 / 1 / 1 | 4.1 |
| E_L10_conf-w-p2 `E_L10_conf-w-p2` | conf | 0 | 127 / 65 / 0 / 0 | 192 | 66.1% | +39 [+25, +52] | +3 | 189, +19.0 pp, 44 / 8 | 0 / 0 / 3 / 0 | 5.3 |
| E_L10_conf-wpar-p1 `E_L10_conf-wpar-p1` | conf | 2 | 86 / 103 / 1 / 2 | 189 | 45.5% | -2 [-8, +3] | +0 | 188, -0.5 pp, 3 / 4 | 1 / 0 / 0 / 1 | 3.4 |
| E_L10_conf-wpar-p2 `E_L10_conf-wpar-p2` | conf | 2 | 89 / 101 / 0 / 2 | 190 | 46.8% | +1 [-5, +7] | +1 | 189, +0.5 pp, 5 / 4 | 0 / 0 / 0 / 1 | 3.4 |
| E_L10_conf-wperp-p1 `E_L10_conf-wperp-p1` | conf | 1 | 102 / 88 / 1 / 1 | 190 | 53.7% | +14 [+4, +24] | +1 | 188, +7.4 pp, 19 / 5 | 1 / 0 / 1 / 1 | 4.2 |
| E_L10_conf-wperp-p2 `E_L10_conf-wperp-p2` | conf | 0 | 124 / 68 / 0 / 0 | 192 | 64.6% | +36 [+23, +49] | +3 | 189, +17.5 pp, 39 / 6 | 0 / 0 / 3 / 0 | 5.5 |
| baseline_conf `baseline_conf` | — | 2 | 88 / 101 / 1 / 2 | 189 | 46.6% | — | — | — | — | 3.4 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.
