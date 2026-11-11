## 1. dose: rule abstentions on x (192 questions), E at layer 10 vs a random direction of the same size

| write | rule abstentions | newly abstained vs no write | newly answered |
|---|---|---|---|
| no write | 23 | — | — |
| prompt: + truth sentence (x_E1), no write | 31 | 9 | 1 |
| E +0.25 | 25 | 2 | 0 |
| E +0.5 | 27 | 4 | 0 |
| E +0.75 | 30 | 7 | 0 |
| E +1 | 33 | 10 | 0 |
| E +1.5 | 39 | 17 | 1 |
| E +2 | 41 | 20 | 2 |
| random direction p2 | 21 | 0 | 2 |
| random direction m2 | 28 | 5 | 0 |

## 2. the 20 questions E +2 turns into "I don't know" (answered without the write)

- pro: their answers WITHOUT the write were R / W / H = 3 / 17 / 0 (right 15%); questions answered both times were 61 / 81 / 6 (right 43%)
  with E + C -0.25, 12 of them are answered again: R / W / H = 8 / 4 / 0
- allenai: their answers WITHOUT the write were R / W / H = 6 / 14 / 0 (right 30%); questions answered both times were 72 / 74 / 3 (right 49%)
  with E + C -0.25, 12 of them are answered again: R / W / H = 10 / 2 / 0
- confidence of the no-write answer (mean log-prob per token): those E turns into "I don't know" -0.514, those still answered -0.401; their average percentile among all answered questions 37% vs 52%
- overlap with the prompt: 7 of the 20 are also "I don't know" when the truth sentence is simply added to the prompt (the prompt alone newly abstains on 9)
