# judge_tune_split_tune, judge pro, 192 questions (DESCRIPTION only; each condition against its own base prompt)

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| E_L10_conf-mdg-p1 `E_L10_conf-mdg-p1` | conf | 2 | 70 / 111 / 9 / 2 | 181 | 38.7% | +0 [-5, +5] | -1 | 181, +0.0 pp, 3 / 3 | 0 / 1 / 0 / 0 | 3.4 |
| E_L10_conf-mdg-p2 `E_L10_conf-mdg-p2` | conf | 2 | 70 / 110 / 9 / 3 | 180 | 38.9% | +0 [-6, +6] | -2 | 180, +0.6 pp, 5 / 4 | 1 / 1 / 0 / 0 | 3.3 |
| E_L10_conf-w-p1 `E_L10_conf-w-p1` | conf | 1 | 83 / 97 / 8 / 4 | 180 | 46.1% | +13 [+3, +23] | -2 | 177, +7.3 pp, 17 / 4 | 3 / 2 / 3 / 0 | 4.1 |
| E_L10_conf-w-p2 `E_L10_conf-w-p2` | conf | 0 | 106 / 76 / 6 / 4 | 182 | 58.2% | +36 [+23, +49] | +0 | 177, +18.6 pp, 40 / 7 | 2 / 3 / 5 / 0 | 5.3 |
| E_L10_conf-wpar-p1 `E_L10_conf-wpar-p1` | conf | 2 | 68 / 112 / 10 / 2 | 180 | 37.8% | -2 [-7, +3] | -2 | 180, -0.6 pp, 2 / 3 | 1 / 1 / 0 / 0 | 3.4 |
| E_L10_conf-wpar-p2 `E_L10_conf-wpar-p2` | conf | 2 | 71 / 110 / 9 / 2 | 181 | 39.2% | +1 [-4, +6] | -1 | 181, +0.6 pp, 4 / 3 | 0 / 1 / 0 / 0 | 3.4 |
| E_L10_conf-wperp-p1 `E_L10_conf-wperp-p1` | conf | 1 | 84 / 95 / 10 / 3 | 179 | 46.9% | +14 [+4, +24] | -3 | 177, +8.5 pp, 19 / 4 | 3 / 2 / 2 / 0 | 4.2 |
| E_L10_conf-wperp-p2 `E_L10_conf-wperp-p2` | conf | 0 | 103 / 78 / 8 / 3 | 181 | 56.9% | +33 [+20, +46] | -1 | 177, +17.5 pp, 39 / 8 | 2 / 3 / 4 / 0 | 5.5 |
| baseline_conf `baseline_conf` | — | 2 | 70 / 112 / 9 / 1 | 182 | 38.5% | — | — | — | — | 3.4 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.
