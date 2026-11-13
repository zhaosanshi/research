llama3.1-8b conditions: step0 baseline_x baseline_x_E1 baseline_x_F baseline_conf baseline_caut baseline_conf_E1 baseline_caut_E1 baseline_conf_E2 baseline_caut_E2 baseline_conf_M baseline_caut_M | diag baseline_x baseline_x_E1 baseline_x_E2 baseline_x_F baseline_x_F_E1 baseline_x_F_E2
qwen3-8b conditions: step0 baseline_x baseline_x_E1 baseline_x_F baseline_conf baseline_caut baseline_conf_E1 baseline_caut_E1 baseline_conf_E2 baseline_caut_E2 baseline_conf_M baseline_caut_M | diag baseline_x baseline_x_E1 baseline_x_E2 baseline_x_F baseline_x_F_E1 baseline_x_F_E2
qwen3.5-9b conditions: step0 baseline_x baseline_x_E1 baseline_x_F baseline_conf baseline_caut baseline_conf_E1 baseline_caut_E1 baseline_conf_E2 baseline_caut_E2 baseline_conf_M baseline_caut_M | diag baseline_x baseline_x_E1 baseline_x_E2 baseline_x_F baseline_x_F_E1 baseline_x_F_E2

## Same prompt in both dirs (generated twice): questions whose text / state differs
- llama3.1-8b x: allenai: text 0, state 0; pro: text 0, state 0
- llama3.1-8b x_E1: allenai: text 0, state 0; pro: text 0, state 0
- llama3.1-8b x_F: allenai: text 0, state 0; pro: text 0, state 0
- qwen3-8b x: allenai: text 0, state 0; pro: text 0, state 0
- qwen3-8b x_E1: allenai: text 0, state 0; pro: text 0, state 0
- qwen3-8b x_F: allenai: text 0, state 0; pro: text 0, state 0
- qwen3.5-9b x: allenai: text 0, state 0; pro: text 0, state 0
- qwen3.5-9b x_E1: allenai: text 0, state 0; pro: text 0, state 0
- qwen3.5-9b x_F: allenai: text 0, state 0; pro: text 0, state 0

### x -> x_E1 (diag dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 78 / 88 / 3 / 23 | 89 / 70 / 2 / 31 | +11 | -7 | +8 | 156: +12 (16 / 4), +7.7 pp | 10 / 3 | 4.0 -> 4.4 |
| llama3.1-8b | pro | 64 / 98 / 6 / 24 | 78 / 79 / 4 / 31 | +14 | -5 | +7 | 152: +14 (17 / 3), +9.2 pp | 10 / 5 | 4.0 -> 4.4 |
| qwen3-8b | allenai | 61 / 56 / 2 / 73 | 71 / 56 / 1 / 64 | +10 | +10 | -9 | 112: +3 (7 / 4), +2.7 pp | 5 / 15 | 4.8 -> 5.4 |
| qwen3-8b | pro | 54 / 61 / 3 / 74 | 66 / 60 / 1 / 65 | +12 | +11 | -9 | 110: +3 (8 / 5), +2.7 pp | 5 / 16 | 4.8 -> 5.4 |
| qwen3.5-9b | allenai | 103 / 63 / 1 / 25 | 131 / 44 / 0 / 17 | +28 | +9 | -8 | 161: +17 (21 / 4), +10.6 pp | 5 / 14 | 3.2 -> 6.8 |
| qwen3.5-9b | pro | 99 / 64 / 4 / 25 | 117 / 56 / 1 / 18 | +18 | +10 | -7 | 157: +10 (15 / 5), +6.4 pp | 6 / 16 | 3.2 -> 6.8 |

### x -> x_E2 (diag dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 78 / 88 / 3 / 23 | 101 / 57 / 0 / 34 | +23 | -8 | +11 | 153: +21 (31 / 10), +13.7 pp | 13 / 5 | 4.0 -> 4.6 |
| llama3.1-8b | pro | 64 / 98 / 6 / 24 | 84 / 66 / 5 / 37 | +20 | -12 | +13 | 145: +20 (27 / 7), +13.8 pp | 17 / 5 | 4.0 -> 4.6 |
| qwen3-8b | allenai | 61 / 56 / 2 / 73 | 68 / 52 / 0 / 72 | +7 | +3 | -1 | 106: +2 (8 / 6), +1.9 pp | 11 / 14 | 4.8 -> 4.9 |
| qwen3-8b | pro | 54 / 61 / 3 / 74 | 61 / 58 / 0 / 73 | +7 | +4 | -1 | 104: +2 (10 / 8), +1.9 pp | 11 / 15 | 4.8 -> 4.9 |
| qwen3.5-9b | allenai | 103 / 63 / 1 / 25 | 116 / 46 / 0 / 30 | +13 | -4 | +5 | 160: +13 (16 / 3), +8.1 pp | 6 / 2 | 3.2 -> 3.7 |
| qwen3.5-9b | pro | 99 / 64 / 4 / 25 | 111 / 47 / 4 / 30 | +12 | -5 | +5 | 154: +11 (13 / 2), +7.1 pp | 9 / 4 | 3.2 -> 3.7 |

### x -> x_F (diag dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 78 / 88 / 3 / 23 | 102 / 59 / 1 / 30 | +24 | -5 | +7 | 156: +24 (31 / 7), +15.4 pp | 10 / 5 | 4.0 -> 9.3 |
| llama3.1-8b | pro | 64 / 98 / 6 / 24 | 80 / 80 / 2 / 30 | +16 | -2 | +6 | 154: +12 (18 / 6), +7.8 pp | 8 / 6 | 4.0 -> 9.3 |
| qwen3-8b | allenai | 61 / 56 / 2 / 73 | 71 / 64 / 1 / 56 | +10 | +18 | -17 | 110: +0 (8 / 8), +0.0 pp | 7 / 25 | 4.8 -> 7.9 |
| qwen3-8b | pro | 54 / 61 / 3 / 74 | 61 / 70 / 5 / 56 | +7 | +16 | -18 | 107: -1 (7 / 8), -0.9 pp | 8 / 24 | 4.8 -> 7.9 |
| qwen3.5-9b | allenai | 103 / 63 / 1 / 25 | 134 / 41 / 2 / 15 | +31 | +9 | -10 | 162: +22 (26 / 4), +13.6 pp | 4 / 13 | 3.2 -> 10.1 |
| qwen3.5-9b | pro | 99 / 64 / 4 / 25 | 111 / 58 / 8 / 15 | +12 | +6 | -10 | 157: +5 (13 / 8), +3.2 pp | 6 / 12 | 3.2 -> 10.1 |

### x_F -> x_F_E1 (diag dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 102 / 59 / 1 / 30 | 118 / 41 / 1 / 32 | +16 | -2 | +2 | 152: +14 (21 / 7), +9.2 pp | 9 / 7 | 9.3 -> 12.2 |
| llama3.1-8b | pro | 80 / 80 / 2 / 30 | 89 / 67 / 3 / 33 | +9 | -4 | +3 | 150: +8 (18 / 10), +5.3 pp | 10 / 6 | 9.3 -> 12.2 |
| qwen3-8b | allenai | 71 / 64 / 1 / 56 | 87 / 57 / 0 / 48 | +16 | +9 | -8 | 132: +11 (14 / 3), +8.3 pp | 3 / 12 | 7.9 -> 9.5 |
| qwen3-8b | pro | 61 / 70 / 5 / 56 | 76 / 66 / 2 / 48 | +15 | +11 | -8 | 128: +9 (11 / 2), +7.0 pp | 3 / 14 | 7.9 -> 9.5 |
| qwen3.5-9b | allenai | 134 / 41 / 2 / 15 | 148 / 35 / 2 / 7 | +14 | +8 | -8 | 174: +8 (13 / 5), +4.6 pp | 1 / 9 | 10.1 -> 13.4 |
| qwen3.5-9b | pro | 111 / 58 / 8 / 15 | 128 / 52 / 5 / 7 | +17 | +11 | -8 | 168: +11 (14 / 3), +6.5 pp | 1 / 12 | 10.1 -> 13.4 |

### x_F -> x_F_E2 (diag dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 102 / 59 / 1 / 30 | 131 / 38 / 2 / 21 | +29 | +8 | -9 | 152: +16 (28 / 12), +10.5 pp | 9 / 17 | 9.3 -> 14.3 |
| llama3.1-8b | pro | 80 / 80 / 2 / 30 | 106 / 60 / 4 / 22 | +26 | +6 | -8 | 151: +17 (32 / 15), +11.3 pp | 9 / 15 | 9.3 -> 14.3 |
| qwen3-8b | allenai | 71 / 64 / 1 / 56 | 92 / 59 / 0 / 41 | +21 | +16 | -15 | 129: +9 (17 / 8), +7.0 pp | 6 / 22 | 7.9 -> 9.6 |
| qwen3-8b | pro | 61 / 70 / 5 / 56 | 78 / 71 / 3 / 40 | +17 | +18 | -16 | 125: +6 (13 / 7), +4.8 pp | 6 / 24 | 7.9 -> 9.6 |
| qwen3.5-9b | allenai | 134 / 41 / 2 / 15 | 148 / 35 / 0 / 9 | +14 | +8 | -6 | 175: +8 (13 / 5), +4.6 pp | 0 / 8 | 10.1 -> 11.5 |
| qwen3.5-9b | pro | 111 / 58 / 8 / 15 | 124 / 54 / 5 / 9 | +13 | +9 | -6 | 168: +6 (13 / 7), +3.6 pp | 1 / 10 | 10.1 -> 11.5 |

### x -> conf (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 78 / 88 / 3 / 23 | 88 / 101 / 1 / 2 | +10 | +23 | -21 | 165: -4 (5 / 9), -2.4 pp | 1 / 24 | 4.0 -> 3.4 |
| llama3.1-8b | pro | 64 / 98 / 6 / 24 | 70 / 112 / 9 / 1 | +6 | +20 | -23 | 160: -3 (5 / 8), -1.9 pp | 2 / 22 | 4.0 -> 3.4 |
| qwen3-8b | allenai | 61 / 56 / 2 / 73 | 100 / 90 / 2 / 0 | +39 | +73 | -73 | 116: -4 (4 / 8), -3.4 pp | 1 / 74 | 4.8 -> 4.9 |
| qwen3-8b | pro | 54 / 61 / 3 / 74 | 87 / 97 / 5 / 3 | +33 | +69 | -71 | 114: -2 (4 / 6), -1.8 pp | 1 / 70 | 4.8 -> 4.9 |
| qwen3.5-9b | allenai | 103 / 63 / 1 / 25 | 111 / 76 / 3 / 2 | +8 | +21 | -23 | 166: -3 (4 / 7), -1.8 pp | 0 / 21 | 3.2 -> 3.1 |
| qwen3.5-9b | pro | 99 / 64 / 4 / 25 | 100 / 87 / 1 / 4 | +1 | +24 | -21 | 163: -7 (2 / 9), -4.3 pp | 0 / 24 | 3.2 -> 3.1 |

### x -> caut (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 78 / 88 / 3 / 23 | 82 / 82 / 1 / 27 | +4 | -2 | +4 | 162: +7 (9 / 2), +4.3 pp | 4 / 2 | 4.0 -> 4.0 |
| llama3.1-8b | pro | 64 / 98 / 6 / 24 | 64 / 94 / 6 / 28 | +0 | -4 | +4 | 158: +1 (3 / 2), +0.6 pp | 4 / 0 | 4.0 -> 4.0 |
| qwen3-8b | allenai | 61 / 56 / 2 / 73 | 44 / 49 / 2 / 97 | -17 | -24 | +24 | 93: -3 (2 / 5), -3.2 pp | 24 / 0 | 4.8 -> 4.1 |
| qwen3-8b | pro | 54 / 61 / 3 / 74 | 42 / 52 / 1 / 97 | -12 | -21 | +23 | 93: -1 (1 / 2), -1.1 pp | 22 / 1 | 4.8 -> 4.1 |
| qwen3.5-9b | allenai | 103 / 63 / 1 / 25 | 101 / 59 / 2 / 30 | -2 | -6 | +5 | 158: +4 (7 / 3), +2.5 pp | 8 / 2 | 3.2 -> 2.6 |
| qwen3.5-9b | pro | 99 / 64 / 4 / 25 | 92 / 65 / 5 / 30 | -7 | -6 | +5 | 155: -2 (3 / 5), -1.3 pp | 8 / 2 | 3.2 -> 2.6 |

### caut -> conf (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 82 / 82 / 1 / 27 | 88 / 101 / 1 / 2 | +6 | +25 | -25 | 163: -9 (5 / 14), -5.5 pp | 1 / 26 | 4.0 -> 3.4 |
| llama3.1-8b | pro | 64 / 94 / 6 / 28 | 70 / 112 / 9 / 1 | +6 | +24 | -27 | 156: -4 (4 / 8), -2.6 pp | 2 / 26 | 4.0 -> 3.4 |
| qwen3-8b | allenai | 44 / 49 / 2 / 97 | 100 / 90 / 2 / 0 | +56 | +97 | -97 | 93: +3 (4 / 1), +3.2 pp | 0 / 97 | 4.1 -> 4.9 |
| qwen3-8b | pro | 42 / 52 / 1 / 97 | 87 / 97 / 5 / 3 | +45 | +90 | -94 | 93: +2 (4 / 2), +2.2 pp | 1 / 91 | 4.1 -> 4.9 |
| qwen3.5-9b | allenai | 101 / 59 / 2 / 30 | 111 / 76 / 3 / 2 | +10 | +27 | -28 | 160: -5 (1 / 6), -3.1 pp | 0 / 27 | 2.6 -> 3.1 |
| qwen3.5-9b | pro | 92 / 65 / 5 / 30 | 100 / 87 / 1 / 4 | +8 | +30 | -26 | 157: -3 (2 / 5), -1.9 pp | 0 / 30 | 2.6 -> 3.1 |

### conf -> conf_E1 (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 88 / 101 / 1 / 2 | 108 / 83 / 1 / 0 | +20 | +2 | -2 | 189: +19 (22 / 3), +10.1 pp | 0 / 2 | 3.4 -> 4.4 |
| llama3.1-8b | pro | 70 / 112 / 9 / 1 | 91 / 92 / 7 / 2 | +21 | +1 | +1 | 179: +19 (25 / 6), +10.6 pp | 3 / 4 | 3.4 -> 4.4 |
| qwen3-8b | allenai | 100 / 90 / 2 / 0 | 105 / 83 / 3 / 1 | +5 | -2 | +1 | 187: +6 (11 / 5), +3.2 pp | 3 / 1 | 4.9 -> 5.1 |
| qwen3-8b | pro | 87 / 97 / 5 / 3 | 95 / 89 / 4 / 4 | +8 | +0 | +1 | 182: +7 (13 / 6), +3.8 pp | 2 / 2 | 4.9 -> 5.1 |
| qwen3.5-9b | allenai | 111 / 76 / 3 / 2 | 136 / 54 / 0 / 2 | +25 | +3 | +0 | 187: +24 (27 / 3), +12.8 pp | 0 / 3 | 3.1 -> 6.4 |
| qwen3.5-9b | pro | 100 / 87 / 1 / 4 | 117 / 68 / 2 / 5 | +17 | -2 | +1 | 185: +19 (25 / 6), +10.3 pp | 2 / 0 | 3.1 -> 6.4 |

### conf -> conf_E2 (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 88 / 101 / 1 / 2 | 117 / 74 / 1 / 0 | +29 | +2 | -2 | 188: +27 (38 / 11), +14.4 pp | 1 / 3 | 3.4 -> 4.0 |
| llama3.1-8b | pro | 70 / 112 / 9 / 1 | 98 / 86 / 4 / 4 | +28 | +2 | +3 | 178: +24 (37 / 13), +13.5 pp | 4 / 6 | 3.4 -> 4.0 |
| qwen3-8b | allenai | 100 / 90 / 2 / 0 | 113 / 73 / 5 / 1 | +13 | -4 | +1 | 185: +16 (24 / 8), +8.6 pp | 5 / 1 | 4.9 -> 4.9 |
| qwen3-8b | pro | 87 / 97 / 5 / 3 | 103 / 80 / 6 / 3 | +16 | -1 | +0 | 181: +14 (22 / 8), +7.7 pp | 3 / 2 | 4.9 -> 4.9 |
| qwen3.5-9b | allenai | 111 / 76 / 3 / 2 | 135 / 54 / 1 / 2 | +24 | +2 | +0 | 187: +22 (24 / 2), +11.8 pp | 0 / 2 | 3.1 -> 2.8 |
| qwen3.5-9b | pro | 100 / 87 / 1 / 4 | 122 / 64 / 2 / 4 | +22 | -1 | +0 | 185: +23 (27 / 4), +12.4 pp | 2 / 1 | 3.1 -> 2.8 |

### caut -> caut_E1 (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 82 / 82 / 1 / 27 | 90 / 68 / 2 / 32 | +8 | -6 | +5 | 155: +8 (15 / 7), +5.2 pp | 9 / 3 | 4.0 -> 4.5 |
| llama3.1-8b | pro | 64 / 94 / 6 / 28 | 76 / 80 / 4 / 32 | +12 | -2 | +4 | 149: +10 (15 / 5), +6.7 pp | 9 / 7 | 4.0 -> 4.5 |
| qwen3-8b | allenai | 44 / 49 / 2 / 97 | 54 / 47 / 0 / 91 | +10 | +8 | -6 | 89: +4 (4 / 0), +4.5 pp | 4 / 12 | 4.1 -> 4.9 |
| qwen3-8b | pro | 42 / 52 / 1 / 97 | 49 / 52 / 0 / 91 | +7 | +7 | -6 | 90: +2 (4 / 2), +2.2 pp | 4 / 11 | 4.1 -> 4.9 |
| qwen3.5-9b | allenai | 101 / 59 / 2 / 30 | 125 / 51 / 0 / 16 | +24 | +16 | -14 | 159: +11 (16 / 5), +6.9 pp | 1 / 17 | 2.6 -> 4.8 |
| qwen3.5-9b | pro | 92 / 65 / 5 / 30 | 114 / 57 / 4 / 17 | +22 | +14 | -13 | 153: +14 (16 / 2), +9.2 pp | 4 / 18 | 2.6 -> 4.8 |

### caut -> caut_E2 (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 82 / 82 / 1 / 27 | 102 / 53 / 0 / 37 | +20 | -9 | +10 | 152: +18 (26 / 8), +11.8 pp | 12 / 3 | 4.0 -> 5.1 |
| llama3.1-8b | pro | 64 / 94 / 6 / 28 | 81 / 69 / 2 / 40 | +17 | -8 | +12 | 144: +15 (25 / 10), +10.4 pp | 14 / 6 | 4.0 -> 5.1 |
| qwen3-8b | allenai | 44 / 49 / 2 / 97 | 45 / 39 / 0 / 108 | +1 | -9 | +11 | 75: +3 (6 / 3), +4.0 pp | 18 / 9 | 4.1 -> 4.2 |
| qwen3-8b | pro | 42 / 52 / 1 / 97 | 45 / 40 / 0 / 107 | +3 | -9 | +10 | 76: +5 (8 / 3), +6.6 pp | 18 / 9 | 4.1 -> 4.2 |
| qwen3.5-9b | allenai | 101 / 59 / 2 / 30 | 117 / 35 / 0 / 40 | +16 | -8 | +10 | 148: +13 (16 / 3), +8.8 pp | 12 / 4 | 2.6 -> 2.9 |
| qwen3.5-9b | pro | 92 / 65 / 5 / 30 | 109 / 39 / 3 / 41 | +17 | -9 | +11 | 143: +15 (18 / 3), +10.5 pp | 14 / 5 | 2.6 -> 2.9 |

### conf -> conf_M (step0 dir)
| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |
|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | allenai | 88 / 101 / 1 / 2 | 64 / 126 / 2 / 0 | -24 | +1 | -2 | 187: -22 (3 / 25), -11.8 pp | 2 / 3 | 3.4 -> 3.1 |
| llama3.1-8b | pro | 70 / 112 / 9 / 1 | 58 / 131 / 2 / 1 | -12 | +7 | +0 | 180: -13 (6 / 19), -7.2 pp | 2 / 9 | 3.4 -> 3.1 |
| qwen3-8b | allenai | 100 / 90 / 2 / 0 | 83 / 104 / 5 / 0 | -17 | -3 | +0 | 186: -14 (5 / 19), -7.5 pp | 4 / 1 | 4.9 -> 4.5 |
| qwen3-8b | pro | 87 / 97 / 5 / 3 | 78 / 108 / 4 / 2 | -9 | +2 | -1 | 181: -10 (8 / 18), -5.5 pp | 3 / 5 | 4.9 -> 4.5 |
| qwen3.5-9b | allenai | 111 / 76 / 3 / 2 | 91 / 97 / 2 / 2 | -20 | +1 | +0 | 187: -20 (8 / 28), -10.7 pp | 0 / 1 | 3.1 -> 2.6 |
| qwen3.5-9b | pro | 100 / 87 / 1 / 4 | 81 / 108 / 1 / 2 | -19 | +2 | -2 | 187: -19 (8 / 27), -10.2 pp | 0 / 2 | 3.1 -> 2.6 |
