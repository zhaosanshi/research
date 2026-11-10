## judge allenai (tau = 6)

- E analogue x -> x_E1: dR +11 [+1, +20] (above tau and interval above 0: True); G_same +9 [+2, +16] on 128 questions (interval above 0: True); dAns -7 (within +-6: False); out / in 10 / 3
- C analogue toward caution: conf -> caut: dAns -25 [-35, -16], dX +25 [+16, +34], effect conditions (without the random control): True | keep n 163 (need >= 95), keep dR +9 [+1, +18], W>R / R>W 14 / 5, toH 1: retention False
- C analogue toward confident: caut -> conf: dAns +25 [+16, +35], dX -25 [-34, -16], effect conditions (without the random control): True | keep n 163 (need >= 82), keep dR -9 [-18, -1], W>R / R>W 5 / 14, toH 1: retention False

## judge pro (tau = 6)

- E analogue x -> x_E1: dR +14 [+4, +24] (above tau and interval above 0: True); G_same +9 [+2, +16] on 128 questions (interval above 0: True); dAns -5 (within +-6: True); out / in 10 / 5
- C analogue toward caution: conf -> caut: dAns -24 [-34, -14], dX +27 [+18, +37], effect conditions (without the random control): True | keep n 156 (need >= 91), keep dR +4 [-3, +11], W>R / R>W 8 / 4, toH 4: retention True
- C analogue toward confident: caut -> conf: dAns +24 [+14, +34], dX -27 [-37, -18], effect conditions (without the random control): True | keep n 156 (need >= 79), keep dR -4 [-11, +3], W>R / R>W 4 / 8, toH 2: retention True

