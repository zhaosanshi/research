# 9B: E direction trained with E1 (round 1, 11-12) against E2 (11-13), paired per layer

Tune 192 questions. Each cell: change in right answers / change in answered questions against the base prompt with no write; rand = margin over the better random direction of the same strength. Source: the KEY lines of `../../../12_新模型自由回答首轮_Qwen3.5-9B/tables/pro_frozen_E_both_judges.md` (E1) and of `e2_tables_L<layer>_s<strength>.md` here (E2). Round 1 made full E1 tables only for the layers it sent to Pro (9, 11-15, 18), so E1 shows — on layers 8, 10, 16 and 17 (its AllenAI screen of every layer is `../../../12_新模型自由回答首轮_Qwen3.5-9B/tables/round1_screen_E_allenai.md`). E2 was sent to Pro on the same 7 layers; layers 8, 10, 16, 17 have AllenAI only. Status A on the conf base is "incomplete" by construction (no F prompt was frozen for conf); with the F condition left out, every conf-base row of E2 meets column A under both judges (`tested_A_without_F=met` in the KEY lines).

## Base: must-answer prompt (conf), strength 1 (exploratory base; no F prompt was frozen for it)

| layer | E1 Pro | E2 Pro | E1 AllenAI | E2 AllenAI | E2 status A (Pro / AllenAI) |
|---|---|---|---|---|---|
| 8 | — | — | — | +17 / +1 (rand +13) | — / incomplete |
| 9 | +7 / -2 (rand +5) | +16 / +0 (rand +14) | +19 / +3 (rand +15) | +15 / +0 (rand +12) | incomplete / incomplete |
| 10 | — | — | — | +16 / -1 (rand +14) | — / incomplete |
| 11 | +14 / -3 (rand +12) | +17 / +0 (rand +15) | +22 / +3 (rand +19) | +17 / -1 (rand +14) | incomplete / incomplete |
| 12 | +12 / -3 (rand +11) | +14 / -1 (rand +12) | +24 / +3 (rand +25) | +15 / -1 (rand +13) | incomplete / incomplete |
| 13 | +9 / -3 (rand +7) | +14 / +0 (rand +11) | +20 / +3 (rand +17) | +15 / -1 (rand +15) | incomplete / incomplete |
| 14 | +7 / -3 (rand +0) | +18 / +0 (rand +14) | +22 / +3 (rand +19) | +17 / +0 (rand +16) | incomplete / incomplete |
| 15 | +7 / -3 (rand +5) | +18 / +0 (rand +15) | +22 / +2 (rand +19) | +18 / +0 (rand +13) | incomplete / incomplete |
| 16 | — | — | — | +15 / +0 (rand +13) | — / incomplete |
| 17 | — | — | — | +18 / +0 (rand +15) | — / incomplete |
| 18 | +6 / -3 (rand +5) | +12 / -1 (rand +11) | +19 / +2 (rand +15) | +10 / -1 (rand +7) | incomplete / incomplete |

## Base: plain prompt (x), strength 1 (the frozen criterion's base)

| layer | E1 Pro | E2 Pro | E1 AllenAI | E2 AllenAI | E2 status A (Pro / AllenAI) |
|---|---|---|---|---|---|
| 8 | — | — | — | +8 / -7 (rand +8) | — / not_met |
| 9 | +5 / +0 (rand +2) | +3 / -5 (rand +3) | +17 / +3 (rand +14) | +8 / -5 (rand +7) | not_met / not_met |
| 10 | — | — | — | +7 / -10 (rand +5) | — / not_met |
| 11 | +6 / +3 (rand +8) | +2 / -11 (rand +2) | +21 / +4 (rand +20) | +6 / -11 (rand +5) | not_met / not_met |
| 12 | +10 / +2 (rand +11) | +0 / -11 (rand +2) | +21 / +3 (rand +16) | +5 / -11 (rand +3) | not_met / not_met |
| 13 | +6 / +1 (rand +8) | +0 / -12 (rand +2) | +20 / +2 (rand +20) | +4 / -13 (rand +5) | not_met / not_met |
| 14 | +8 / +0 (rand +6) | +2 / -13 (rand +1) | +20 / +1 (rand +18) | +7 / -13 (rand +4) | not_met / not_met |
| 15 | +10 / +2 (rand +12) | +3 / -10 (rand +2) | +20 / +2 (rand +23) | +8 / -11 (rand +9) | not_met / not_met |
| 16 | — | — | — | +6 / -9 (rand +4) | — / not_met |
| 17 | — | — | — | +8 / -6 (rand +6) | — / not_met |
| 18 | +8 / +2 (rand +10) | +8 / -2 (rand +11) | +22 / +3 (rand +22) | +8 / -2 (rand +8) | not_met / not_met |

## E2 on the plain prompt (x) at strengths 0.5 and 2 (dR / dAns on x; F = the same write on the complete-sentence prompt x_F)

| layer | Pro 0.5 | Pro 2 | Pro 2: F dR / dAns | AllenAI 0.5 | AllenAI 2 | AllenAI 2: F dR / dAns |
|---|---|---|---|---|---|---|
| 8 | — | — | — | +6 / -2 | +8 / -14 | +4 / -10 |
| 9 | +1 / -4 | +4 / -16 | +2 / -9 | +6 / -2 | +10 / -15 | +7 / -7 |
| 10 | — | — | — | +7 / -5 | +9 / -21 | +10 / -14 |
| 11 | +3 / -7 | +5 / -23 | +9 / -13 | +8 / -5 | +8 / -24 | +13 / -12 |
| 12 | -1 / -7 | +0 / -31 | +6 / -14 | +6 / -5 | +4 / -30 | +10 / -11 |
| 13 | +0 / -8 | -1 / -32 | +6 / -15 | +3 / -6 | +1 / -32 | +11 / -11 |
| 14 | +0 / -8 | +0 / -27 | +9 / -14 | +4 / -6 | +3 / -27 | +6 / -14 |
| 15 | -1 / -9 | -1 / -24 | +7 / -11 | +4 / -7 | +6 / -24 | +2 / -13 |
| 16 | — | — | — | +3 / -6 | +6 / -20 | +0 / -12 |
| 17 | — | — | — | +3 / -6 | +11 / -14 | +1 / -13 |
| 18 | +4 / -3 | +14 / -1 | -4 / -11 | +5 / -4 | +17 / -1 | +1 / -7 |
