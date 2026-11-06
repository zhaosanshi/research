# E26 同实际作答率方法比较

精确配额表是批量筛题评估；排序不看正确性。冻结门槛表另报真实测试作答率。
QA分母为固定实质参考题；不够配额标不可行，未决判分不出表。

完整配置运行。

## tf / 目标 25%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 264/1058 | 210 | 0.7955 | [0.7462, 0.8409] | ok |
| baseline_simple | 264/1058 | 241 | 0.9129 | [0.8750, 0.9470] | ok |
| baseline_random | 264/1058 | 187 | 0.7083 | [0.6552, 0.7652] | ok |
| das_E | 264/1058 | 210 | 0.7955 | [0.7500, 0.8447] | ok |
| das_C | 264/1058 | 210 | 0.7955 | [0.7462, 0.8409] | ok |
| das_EC | 264/1058 | 209 | 0.7917 | [0.7538, 0.8485] | ok |
| md_E | 264/1058 | 215 | 0.8144 | [0.7614, 0.8523] | ok |
| md_C | 264/1058 | 210 | 0.7955 | [0.7462, 0.8409] | ok |
| md_EC | 264/1058 | 215 | 0.8144 | [0.7652, 0.8561] | ok |
| das_E_simple | 264/1058 | 250 | 0.9470 | [0.9167, 0.9735] | ok |
| das_C_simple | 264/1058 | 239 | 0.9053 | [0.8674, 0.9432] | ok |
| md_E_simple | 264/1058 | 244 | 0.9242 | [0.8902, 0.9545] | ok |
| md_C_simple | 264/1058 | 239 | 0.9053 | [0.8674, 0.9432] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.1212 | [-0.1591, -0.0720] | 0.0289 | ok |
| das_EC | das_E | -0.0038 | [-0.0076, 0.0152] | 1.0000 | ok |
| das_EC | das_C | -0.0038 | [-0.0227, 0.0303] | 1.0000 | ok |
| das_EC | das_E_simple | -0.1553 | [-0.1894, -0.1061] | 0.0043 | ok |
| das_EC | das_C_simple | -0.1136 | [-0.1477, -0.0682] | 0.0332 | ok |
| md_EC | baseline_simple | -0.0985 | [-0.1439, -0.0606] | 0.0754 | ok |
| md_EC | md_E | -0.0000 | [-0.0114, 0.0152] | 1.0000 | ok |
| md_EC | md_C | 0.0189 | [-0.0076, 0.0417] | 0.4244 | ok |
| md_EC | md_E_simple | -0.1098 | [-0.1553, -0.0720] | 0.0425 | ok |
| md_EC | md_C_simple | -0.0909 | [-0.1364, -0.0568] | 0.0879 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.2703 | 286 | 224 | 0.7832 |
| baseline_simple | 0.3043 | 322 | 290 | 0.9006 |
| baseline_random | 0.2136 | 226 | 159 | 0.7035 |
| das_E | 0.2807 | 297 | 239 | 0.8047 |
| das_C | 0.2779 | 294 | 232 | 0.7891 |
| das_EC | 0.2703 | 286 | 230 | 0.8042 |
| md_E | 0.2656 | 281 | 226 | 0.8043 |
| md_C | 0.2713 | 287 | 227 | 0.7909 |
| md_EC | 0.2533 | 268 | 219 | 0.8172 |
| das_E_simple | 0.2694 | 285 | 269 | 0.9439 |
| das_C_simple | 0.3072 | 325 | 291 | 0.8954 |
| md_E_simple | 0.3110 | 329 | 302 | 0.9179 |
| md_C_simple | 0.3100 | 328 | 296 | 0.9024 |

## tf / 目标 50%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 529/1058 | 376 | 0.7108 | [0.6767, 0.7448] | ok |
| baseline_simple | 529/1058 | 453 | 0.8563 | [0.8204, 0.8922] | ok |
| baseline_random | 529/1058 | 380 | 0.7183 | [0.6786, 0.7580] | ok |
| das_E | 529/1058 | 402 | 0.7599 | [0.7278, 0.7921] | ok |
| das_C | 529/1058 | 368 | 0.6957 | [0.6560, 0.7297] | ok |
| das_EC | 529/1058 | 399 | 0.7543 | [0.7240, 0.7883] | ok |
| md_E | 529/1058 | 380 | 0.7183 | [0.6843, 0.7505] | ok |
| md_C | 529/1058 | 370 | 0.6994 | [0.6635, 0.7335] | ok |
| md_EC | 529/1058 | 384 | 0.7259 | [0.6938, 0.7580] | ok |
| das_E_simple | 529/1058 | 470 | 0.8885 | [0.8563, 0.9168] | ok |
| das_C_simple | 529/1058 | 456 | 0.8620 | [0.8242, 0.8942] | ok |
| md_E_simple | 529/1058 | 460 | 0.8696 | [0.8336, 0.9036] | ok |
| md_C_simple | 529/1058 | 459 | 0.8677 | [0.8261, 0.8960] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.1021 | [-0.1342, -0.0662] | 0.0066 | ok |
| das_EC | das_E | -0.0057 | [-0.0113, 0.0038] | 0.4531 | ok |
| das_EC | das_C | 0.0586 | [0.0321, 0.0926] | 0.0002 | ok |
| das_EC | das_E_simple | -0.1342 | [-0.1645, -0.1002] | 0.0002 | ok |
| das_EC | das_C_simple | -0.1078 | [-0.1361, -0.0699] | 0.0032 | ok |
| md_EC | baseline_simple | -0.1304 | [-0.1626, -0.0964] | 0.0001 | ok |
| md_EC | md_E | 0.0076 | [-0.0095, 0.0265] | 0.5847 | ok |
| md_EC | md_C | 0.0265 | [-0.0000, 0.0530] | 0.0980 | ok |
| md_EC | md_E_simple | -0.1437 | [-0.1758, -0.1134] | 0.0000 | ok |
| md_EC | md_C_simple | -0.1418 | [-0.1682, -0.1040] | 0.0000 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.5009 | 530 | 376 | 0.7094 |
| baseline_simple | 0.5189 | 549 | 466 | 0.8488 |
| baseline_random | 0.4820 | 510 | 365 | 0.7157 |
| das_E | 0.4953 | 524 | 400 | 0.7634 |
| das_C | 0.5000 | 529 | 368 | 0.6957 |
| das_EC | 0.5028 | 532 | 402 | 0.7556 |
| md_E | 0.4991 | 528 | 379 | 0.7178 |
| md_C | 0.5028 | 532 | 371 | 0.6974 |
| md_EC | 0.5028 | 532 | 386 | 0.7256 |
| das_E_simple | 0.4953 | 524 | 467 | 0.8912 |
| das_C_simple | 0.5104 | 540 | 462 | 0.8556 |
| md_E_simple | 0.5180 | 548 | 474 | 0.8650 |
| md_C_simple | 0.4991 | 528 | 458 | 0.8674 |

## tf / 目标 75%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 793/1058 | 522 | 0.6583 | [0.6242, 0.6911] | ok |
| baseline_simple | 793/1058 | 620 | 0.7818 | [0.7503, 0.8159] | ok |
| baseline_random | 793/1058 | 571 | 0.7201 | [0.6873, 0.7516] | ok |
| das_E | 793/1058 | 562 | 0.7087 | [0.6772, 0.7390] | ok |
| das_C | 793/1058 | 516 | 0.6507 | [0.6166, 0.6835] | ok |
| das_EC | 793/1058 | 562 | 0.7087 | [0.6772, 0.7402] | ok |
| md_E | 793/1058 | 532 | 0.6709 | [0.6368, 0.7037] | ok |
| md_C | 793/1058 | 519 | 0.6545 | [0.6192, 0.6873] | ok |
| md_EC | 793/1058 | 536 | 0.6759 | [0.6406, 0.7112] | ok |
| das_E_simple | 793/1058 | 642 | 0.8096 | [0.7818, 0.8373] | ok |
| das_C_simple | 793/1058 | 630 | 0.7945 | [0.7629, 0.8247] | ok |
| md_E_simple | 793/1058 | 637 | 0.8033 | [0.7692, 0.8323] | ok |
| md_C_simple | 793/1058 | 626 | 0.7894 | [0.7579, 0.8209] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.0731 | [-0.0971, -0.0492] | 0.0036 | ok |
| das_EC | das_E | -0.0000 | [-0.0050, 0.0050] | 1.0000 | ok |
| das_EC | das_C | 0.0580 | [0.0353, 0.0794] | 0.0000 | ok |
| das_EC | das_E_simple | -0.1009 | [-0.1248, -0.0782] | 0.0000 | ok |
| das_EC | das_C_simple | -0.0858 | [-0.1084, -0.0605] | 0.0006 | ok |
| md_EC | baseline_simple | -0.1059 | [-0.1274, -0.0857] | 0.0000 | ok |
| md_EC | md_E | 0.0050 | [-0.0076, 0.0177] | 0.5966 | ok |
| md_EC | md_C | 0.0214 | [-0.0000, 0.0416] | 0.0712 | ok |
| md_EC | md_E_simple | -0.1274 | [-0.1475, -0.1021] | 0.0000 | ok |
| md_EC | md_C_simple | -0.1135 | [-0.1375, -0.0921] | 0.0000 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.7448 | 788 | 517 | 0.6561 |
| baseline_simple | 0.7713 | 816 | 632 | 0.7745 |
| baseline_random | 0.7817 | 827 | 595 | 0.7195 |
| das_E | 0.7439 | 787 | 557 | 0.7078 |
| das_C | 0.7486 | 792 | 515 | 0.6503 |
| das_EC | 0.7457 | 789 | 559 | 0.7085 |
| md_E | 0.7316 | 774 | 517 | 0.6680 |
| md_C | 0.7420 | 785 | 512 | 0.6522 |
| md_EC | 0.7401 | 783 | 526 | 0.6718 |
| das_E_simple | 0.7647 | 809 | 655 | 0.8096 |
| das_C_simple | 0.7637 | 808 | 640 | 0.7921 |
| md_E_simple | 0.7505 | 794 | 637 | 0.8023 |
| md_C_simple | 0.7637 | 808 | 637 | 0.7884 |

## qa / 目标 25%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 130/523 | 61 | 0.4692 | [0.3692, 0.5615] | ok |
| baseline_simple | 130/523 | 72 | 0.5538 | [0.4613, 0.6308] | ok |
| baseline_random | 130/523 | 54 | 0.4154 | [0.3308, 0.5000] | ok |
| das_E | 130/523 | 60 | 0.4615 | [0.3692, 0.5538] | ok |
| das_C | 130/523 | 66 | 0.5077 | [0.4231, 0.5923] | ok |
| das_EC | 130/523 | 63 | 0.4846 | [0.4077, 0.5846] | ok |
| das_generation_adapter_E | 130/523 | 67 | 0.5154 | [0.4154, 0.5925] | ok |
| das_generation_adapter_C | 130/523 | 57 | 0.4385 | [0.3538, 0.5385] | ok |
| das_generation_adapter_EC | 130/523 | 63 | 0.4846 | [0.4000, 0.5769] | ok |
| md_E | 130/523 | 57 | 0.4385 | [0.3462, 0.5308] | ok |
| md_C | 130/523 | 66 | 0.5077 | [0.4231, 0.5923] | ok |
| md_EC | 130/523 | 58 | 0.4462 | [0.3538, 0.5462] | ok |
| md_generation_adapter_E | 130/523 | 63 | 0.4846 | [0.3923, 0.5846] | ok |
| md_generation_adapter_C | 130/523 | 59 | 0.4538 | [0.3538, 0.5462] | ok |
| md_generation_adapter_EC | 130/523 | 59 | 0.4538 | [0.3615, 0.5463] | ok |
| random_generation_adapter_E | 130/523 | 65 | 0.5000 | [0.4000, 0.5846] | ok |
| random_generation_adapter_C | 130/523 | 57 | 0.4385 | [0.3538, 0.5385] | ok |
| random_generation_adapter_EC | 130/523 | 62 | 0.4769 | [0.3923, 0.5769] | ok |
| das_E_simple | 130/523 | 71 | 0.5462 | [0.4538, 0.6308] | ok |
| das_C_simple | 130/523 | 71 | 0.5462 | [0.4615, 0.6385] | ok |
| das_generation_adapter_E_simple | 130/523 | 55 | 0.4231 | [0.3308, 0.5077] | ok |
| das_generation_adapter_C_simple | 130/523 | 69 | 0.5308 | [0.4462, 0.6231] | ok |
| md_E_simple | 130/523 | 73 | 0.5615 | [0.4692, 0.6462] | ok |
| md_C_simple | 130/523 | 71 | 0.5462 | [0.4615, 0.6385] | ok |
| md_generation_adapter_E_simple | 130/523 | 57 | 0.4385 | [0.3462, 0.5462] | ok |
| md_generation_adapter_C_simple | 130/523 | 70 | 0.5385 | [0.4538, 0.6308] | ok |
| random_generation_adapter_E_simple | 130/523 | 66 | 0.5077 | [0.4000, 0.5923] | ok |
| random_generation_adapter_C_simple | 130/523 | 70 | 0.5385 | [0.4538, 0.6308] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.0692 | [-0.1462, 0.0462] | 0.3356 | ok |
| das_EC | das_E | 0.0231 | [-0.0308, 0.1000] | 0.6072 | ok |
| das_EC | das_C | -0.0231 | [-0.0540, 0.0231] | 0.4531 | ok |
| das_EC | das_E_simple | -0.0615 | [-0.1385, 0.0538] | 0.3817 | ok |
| das_EC | das_C_simple | -0.0615 | [-0.1462, 0.0308] | 0.3581 | ok |
| das_generation_adapter_EC | baseline_simple | -0.0692 | [-0.1538, 0.0462] | 0.3620 | ok |
| das_generation_adapter_EC | das_generation_adapter_E | -0.0308 | [-0.0538, 0.0154] | 0.2891 | ok |
| das_generation_adapter_EC | das_generation_adapter_C | 0.0462 | [-0.0231, 0.1154] | 0.2379 | ok |
| das_generation_adapter_EC | das_generation_adapter_E_simple | 0.0615 | [-0.0308, 0.1538] | 0.4096 | ok |
| das_generation_adapter_EC | das_generation_adapter_C_simple | -0.0462 | [-0.1462, 0.0538] | 0.5446 | ok |
| md_EC | baseline_simple | -0.1077 | [-0.1923, -0.0000] | 0.1089 | ok |
| md_EC | md_E | 0.0077 | [-0.0308, 0.0538] | 1.0000 | ok |
| md_EC | md_C | -0.0615 | [-0.1385, 0.0154] | 0.1153 | ok |
| md_EC | md_E_simple | -0.1154 | [-0.2000, -0.0077] | 0.0817 | ok |
| md_EC | md_C_simple | -0.1000 | [-0.1923, -0.0077] | 0.1112 | ok |
| md_generation_adapter_EC | baseline_simple | -0.1000 | [-0.1846, 0.0077] | 0.1480 | ok |
| md_generation_adapter_EC | md_generation_adapter_E | -0.0308 | [-0.0692, 0.0077] | 0.1250 | ok |
| md_generation_adapter_EC | md_generation_adapter_C | -0.0000 | [-0.0538, 0.0692] | 1.0000 | ok |
| md_generation_adapter_EC | md_generation_adapter_E_simple | 0.0154 | [-0.0923, 0.1077] | 0.9022 | ok |
| md_generation_adapter_EC | md_generation_adapter_C_simple | -0.0846 | [-0.1769, 0.0077] | 0.2000 | ok |
| random_generation_adapter_EC | baseline_simple | -0.0769 | [-0.1692, 0.0462] | 0.3143 | ok |
| random_generation_adapter_EC | random_generation_adapter_E | -0.0231 | [-0.0769, 0.0615] | 0.6636 | ok |
| random_generation_adapter_EC | random_generation_adapter_C | 0.0385 | [-0.0462, 0.1231] | 0.4869 | ok |
| random_generation_adapter_EC | random_generation_adapter_E_simple | -0.0308 | [-0.1231, 0.0923] | 0.7275 | ok |
| random_generation_adapter_EC | random_generation_adapter_C_simple | -0.0615 | [-0.1615, 0.0462] | 0.4030 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.3002 | 157 | 71 | 0.4522 |
| baseline_simple | 0.2658 | 139 | 75 | 0.5396 |
| baseline_random | 0.2275 | 119 | 49 | 0.4118 |
| das_E | 0.2964 | 155 | 71 | 0.4581 |
| das_C | 0.3117 | 163 | 79 | 0.4847 |
| das_EC | 0.2925 | 153 | 73 | 0.4771 |
| das_generation_adapter_E | 0.3059 | 160 | 80 | 0.5000 |
| das_generation_adapter_C | 0.3021 | 158 | 69 | 0.4367 |
| das_generation_adapter_EC | 0.2868 | 150 | 74 | 0.4933 |
| md_E | 0.3021 | 158 | 68 | 0.4304 |
| md_C | 0.3136 | 164 | 79 | 0.4817 |
| md_EC | 0.2983 | 156 | 66 | 0.4231 |
| md_generation_adapter_E | 0.2906 | 152 | 70 | 0.4605 |
| md_generation_adapter_C | 0.3021 | 158 | 69 | 0.4367 |
| md_generation_adapter_EC | 0.3078 | 161 | 71 | 0.4410 |
| random_generation_adapter_E | 0.3021 | 158 | 72 | 0.4557 |
| random_generation_adapter_C | 0.3002 | 157 | 68 | 0.4331 |
| random_generation_adapter_EC | 0.3193 | 167 | 75 | 0.4491 |
| das_E_simple | 0.2696 | 141 | 76 | 0.5390 |
| das_C_simple | 0.2505 | 131 | 72 | 0.5496 |
| das_generation_adapter_E_simple | 0.3250 | 170 | 78 | 0.4588 |
| das_generation_adapter_C_simple | 0.2639 | 138 | 73 | 0.5290 |
| md_E_simple | 0.2620 | 137 | 76 | 0.5547 |
| md_C_simple | 0.2505 | 131 | 72 | 0.5496 |
| md_generation_adapter_E_simple | 0.2600 | 136 | 59 | 0.4338 |
| md_generation_adapter_C_simple | 0.2639 | 138 | 74 | 0.5362 |
| random_generation_adapter_E_simple | 0.3021 | 158 | 69 | 0.4367 |
| random_generation_adapter_C_simple | 0.2658 | 139 | 74 | 0.5324 |

## qa / 目标 50%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 261/523 | 107 | 0.4100 | [0.3525, 0.4714] | ok |
| baseline_simple | 261/523 | 123 | 0.4713 | [0.4023, 0.5287] | ok |
| baseline_random | 261/523 | 111 | 0.4253 | [0.3602, 0.4789] | ok |
| das_E | 261/523 | 110 | 0.4215 | [0.3602, 0.4828] | ok |
| das_C | 261/523 | 104 | 0.3985 | [0.3372, 0.4598] | ok |
| das_EC | 261/523 | 107 | 0.4100 | [0.3448, 0.4674] | ok |
| das_generation_adapter_E | 261/523 | 120 | 0.4598 | [0.3946, 0.5211] | ok |
| das_generation_adapter_C | 261/523 | 113 | 0.4330 | [0.3716, 0.4943] | ok |
| das_generation_adapter_EC | 261/523 | 115 | 0.4406 | [0.3793, 0.5057] | ok |
| md_E | 261/523 | 108 | 0.4138 | [0.3525, 0.4751] | ok |
| md_C | 261/523 | 104 | 0.3985 | [0.3372, 0.4598] | ok |
| md_EC | 261/523 | 108 | 0.4138 | [0.3487, 0.4713] | ok |
| md_generation_adapter_E | 261/523 | 111 | 0.4253 | [0.3678, 0.4904] | ok |
| md_generation_adapter_C | 261/523 | 103 | 0.3946 | [0.3333, 0.4522] | ok |
| md_generation_adapter_EC | 261/523 | 109 | 0.4176 | [0.3525, 0.4751] | ok |
| random_generation_adapter_E | 261/523 | 109 | 0.4176 | [0.3563, 0.4789] | ok |
| random_generation_adapter_C | 261/523 | 114 | 0.4368 | [0.3716, 0.4943] | ok |
| random_generation_adapter_EC | 261/523 | 108 | 0.4138 | [0.3563, 0.4789] | ok |
| das_E_simple | 261/523 | 118 | 0.4521 | [0.3793, 0.5057] | ok |
| das_C_simple | 261/523 | 124 | 0.4751 | — | infeasible_resamples |
| das_generation_adapter_E_simple | 261/523 | 127 | 0.4866 | [0.4291, 0.5517] | ok |
| das_generation_adapter_C_simple | 261/523 | 120 | 0.4598 | [0.3985, 0.5211] | ok |
| md_E_simple | 261/523 | 121 | 0.4636 | [0.3946, 0.5173] | ok |
| md_C_simple | 261/523 | 124 | 0.4751 | — | infeasible_resamples |
| md_generation_adapter_E_simple | 261/523 | 115 | 0.4406 | [0.3793, 0.5019] | ok |
| md_generation_adapter_C_simple | 261/523 | 113 | 0.4330 | [0.3640, 0.4943] | ok |
| random_generation_adapter_E_simple | 261/523 | 115 | 0.4406 | [0.3755, 0.4981] | ok |
| random_generation_adapter_C_simple | 261/523 | 120 | 0.4598 | [0.3985, 0.5211] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.0613 | [-0.1111, -0.0115] | 0.0519 | ok |
| das_EC | das_E | -0.0115 | [-0.0421, 0.0115] | 0.6072 | ok |
| das_EC | das_C | 0.0115 | [-0.0115, 0.0268] | 0.4531 | ok |
| das_EC | das_E_simple | -0.0421 | [-0.0843, 0.0077] | 0.1770 | ok |
| das_EC | das_C_simple | -0.0651 | — | 0.0430 | infeasible_resamples |
| das_generation_adapter_EC | baseline_simple | -0.0307 | [-0.0958, 0.0421] | 0.4841 | ok |
| das_generation_adapter_EC | das_generation_adapter_E | -0.0192 | [-0.0575, 0.0268] | 0.5224 | ok |
| das_generation_adapter_EC | das_generation_adapter_C | 0.0077 | [-0.0383, 0.0613] | 0.8877 | ok |
| das_generation_adapter_EC | das_generation_adapter_E_simple | -0.0460 | [-0.1073, 0.0115] | 0.2713 | ok |
| das_generation_adapter_EC | das_generation_adapter_C_simple | -0.0192 | [-0.0766, 0.0421] | 0.6752 | ok |
| md_EC | baseline_simple | -0.0575 | [-0.1073, -0.0077] | 0.0627 | ok |
| md_EC | md_E | -0.0000 | [-0.0268, 0.0230] | 1.0000 | ok |
| md_EC | md_C | 0.0153 | [-0.0115, 0.0383] | 0.3877 | ok |
| md_EC | md_E_simple | -0.0498 | [-0.0920, 0.0000] | 0.1048 | ok |
| md_EC | md_C_simple | -0.0613 | — | 0.0599 | infeasible_resamples |
| md_generation_adapter_EC | baseline_simple | -0.0536 | [-0.1073, -0.0038] | 0.1143 | ok |
| md_generation_adapter_EC | md_generation_adapter_E | -0.0077 | [-0.0498, 0.0192] | 0.8238 | ok |
| md_generation_adapter_EC | md_generation_adapter_C | 0.0230 | [-0.0153, 0.0536] | 0.3075 | ok |
| md_generation_adapter_EC | md_generation_adapter_E_simple | -0.0230 | [-0.0805, 0.0230] | 0.5764 | ok |
| md_generation_adapter_EC | md_generation_adapter_C_simple | -0.0153 | [-0.0728, 0.0307] | 0.6835 | ok |
| random_generation_adapter_EC | baseline_simple | -0.0575 | [-0.1149, 0.0115] | 0.1374 | ok |
| random_generation_adapter_EC | random_generation_adapter_E | -0.0038 | [-0.0421, 0.0421] | 1.0000 | ok |
| random_generation_adapter_EC | random_generation_adapter_C | -0.0230 | [-0.0536, 0.0230] | 0.3771 | ok |
| random_generation_adapter_EC | random_generation_adapter_E_simple | -0.0268 | [-0.0766, 0.0345] | 0.4944 | ok |
| random_generation_adapter_EC | random_generation_adapter_C_simple | -0.0460 | [-0.0996, 0.0077] | 0.2007 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.5220 | 273 | 112 | 0.4103 |
| baseline_simple | 0.5392 | 282 | 130 | 0.4610 |
| baseline_random | 0.4818 | 252 | 106 | 0.4206 |
| das_E | 0.5201 | 272 | 115 | 0.4228 |
| das_C | 0.5526 | 289 | 113 | 0.3910 |
| das_EC | 0.5507 | 288 | 116 | 0.4028 |
| das_generation_adapter_E | 0.5201 | 272 | 124 | 0.4559 |
| das_generation_adapter_C | 0.5048 | 264 | 115 | 0.4356 |
| das_generation_adapter_EC | 0.5258 | 275 | 122 | 0.4436 |
| md_E | 0.5277 | 276 | 114 | 0.4130 |
| md_C | 0.5526 | 289 | 113 | 0.3910 |
| md_EC | 0.5526 | 289 | 117 | 0.4048 |
| md_generation_adapter_E | 0.5182 | 271 | 118 | 0.4354 |
| md_generation_adapter_C | 0.5411 | 283 | 111 | 0.3922 |
| md_generation_adapter_EC | 0.5373 | 281 | 116 | 0.4128 |
| random_generation_adapter_E | 0.5335 | 279 | 117 | 0.4194 |
| random_generation_adapter_C | 0.4933 | 258 | 112 | 0.4341 |
| random_generation_adapter_EC | 0.5277 | 276 | 118 | 0.4275 |
| das_E_simple | 0.5048 | 264 | 118 | 0.4470 |
| das_C_simple | 0.5105 | 267 | 126 | 0.4719 |
| das_generation_adapter_E_simple | 0.5545 | 290 | 146 | 0.5034 |
| das_generation_adapter_C_simple | 0.5277 | 276 | 127 | 0.4601 |
| md_E_simple | 0.5258 | 275 | 126 | 0.4582 |
| md_C_simple | 0.5105 | 267 | 126 | 0.4719 |
| md_generation_adapter_E_simple | 0.5124 | 268 | 118 | 0.4403 |
| md_generation_adapter_C_simple | 0.5354 | 280 | 117 | 0.4179 |
| random_generation_adapter_E_simple | 0.5220 | 273 | 118 | 0.4322 |
| random_generation_adapter_C_simple | 0.5220 | 273 | 125 | 0.4579 |

## qa / 目标 75%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| baseline_simple | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| baseline_random | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| das_E | — | — | — | — | 调参无可行配置 |
| das_C | — | — | — | — | 调参无可行配置 |
| das_EC | — | — | — | — | 调参无可行配置 |
| das_generation_adapter_E | — | — | — | — | 调参无可行配置 |
| das_generation_adapter_C | — | — | — | — | 调参无可行配置 |
| das_generation_adapter_EC | — | — | — | — | 调参无可行配置 |
| md_E | — | — | — | — | 调参无可行配置 |
| md_C | — | — | — | — | 调参无可行配置 |
| md_EC | — | — | — | — | 调参无可行配置 |
| md_generation_adapter_E | — | — | — | — | 调参无可行配置 |
| md_generation_adapter_C | — | — | — | — | 调参无可行配置 |
| md_generation_adapter_EC | — | — | — | — | 调参无可行配置 |
| random_generation_adapter_E | — | — | — | — | 调参无可行配置 |
| random_generation_adapter_C | — | — | — | — | 调参无可行配置 |
| random_generation_adapter_EC | — | — | — | — | 调参无可行配置 |
| das_E_simple | — | — | — | — | 调参无可行配置 |
| das_C_simple | — | — | — | — | 调参无可行配置 |
| das_generation_adapter_E_simple | — | — | — | — | 调参无可行配置 |
| das_generation_adapter_C_simple | — | — | — | — | 调参无可行配置 |
| md_E_simple | — | — | — | — | 调参无可行配置 |
| md_C_simple | — | — | — | — | 调参无可行配置 |
| md_generation_adapter_E_simple | — | — | — | — | 调参无可行配置 |
| md_generation_adapter_C_simple | — | — | — | — | 调参无可行配置 |
| random_generation_adapter_E_simple | — | — | — | — | 调参无可行配置 |
| random_generation_adapter_C_simple | — | — | — | — | 调参无可行配置 |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | — | — | — | — |
| baseline_simple | — | — | — | — |
| baseline_random | — | — | — | — |
