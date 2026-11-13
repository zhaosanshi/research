TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 30

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 61 / 56 / 2 / 73 | 4.8 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |
| x.w.m1 | 61 / 56 / 2 / 73 | 4.8 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |
| x.md.p1 | 62 / 55 / 2 / 73 | 4.9 | +1 [-4, +6] | +0 [-4, +4] | +0 [-4, +4] | 115, +1 [-3, +5], 3 / 2 | 2 / 2 / 0 | 1 / 1 / 1 / 1 | +1 (105) |
| x.rand.p1 | 61 / 56 / 2 / 73 | 4.8 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |
| x.rand.m1 | 61 / 56 / 2 / 73 | 4.8 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 71 / 64 / 1 / 56; dR +0 [+0, +0]; answered-rate change +0.0 pp [-1.5, +1.5]; out / in 0 / 0; mean words 7.9 -> 7.8
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +0 [+0, +0]; vs x.rand.m1 +0 [+0, +0] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-1.5, +1.5]; moved out / in 0 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +0 on 117 questions; mean words 4.8 -> 4.8
KEY family=E judge=allenai layer=30 A=not_met B=not_met dR=+0 dR_F=+0 dAns=+0 dAns_F=+0 rand_margin=+0

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 30

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 48 / 2 / 98 | +0 [+0, +0] | -1 [-3, +0] | +1 [+0, +3] | 92, +0, 0 / 0 | 1 / 0 / 0 | +0 (90) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 45 / 38 / 0 / 109 | +0 [-3, +3] | -1 [-3, +0] | +1 [+0, +3] | 83, +1, 1 / 0 | 1 / 0 / 0 | +1 (81) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 4 / 0 | +0 [-3, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, +0, 1 / 1 | 2 / 0 / 2 | +0 (185) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 4 / 0 | +0 [-7, +7] | -2 [-5, +0] | +0 [+0, +0] | 188, +0, 6 / 6 | 2 / 0 / 2 | -1 (177) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 0 / 1 | 0 / 0 / 0 | -1 (190) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 1 / 2 | 0 / 0 / 0 | -1 (190) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 0 / 1 | 0 / 0 / 0 | -1 (189) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 4 / 0 | -1 [-5, +2] | -2 [-5, +0] | +0 [+0, +0] | 188, -1, 1 / 2 | 2 / 0 / 2 | -1 (186) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 112 / 74 / 5 / 1 | -1 [-6, +3] | +0 [+0, +0] | +0 [+0, +0] | 186, -1, 2 / 3 | 0 / 0 / 0 | -2 (183) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 71 / 64 / 1 / 56 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 135, +0, 0 / 0 | 0 / 0 / 0 | +0 (135) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 30

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 99 / 91 / 2 / 0 | 5.0 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -1 [-3, +0], 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (189) |
| conf.md.p1 | 100 / 88 / 4 / 0 | 4.6 | +0 [-7, +7] | -2 [-5, +0] | +0 [+0, +0] | 188, +0 [-7, +7], 6 / 6 | 2 / 0 / 2 | 0 / 2 / 0 / 0 | -1 (177) |
| conf.rand.p1 | 99 / 91 / 2 / 0 | 4.8 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -1 [-5, +2], 1 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (190) |
| conf.rand.m1 | 99 / 91 / 2 / 0 | 4.9 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -1 [-3, +0], 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (190) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +0 [-3, +3]; vs conf.rand.m1 +0 [+0, +0] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.5, +1.5]; moved out / in 0 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -1 on 189 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=30 A=not_met B=not_met dR=-1 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+0 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:55:10 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===
