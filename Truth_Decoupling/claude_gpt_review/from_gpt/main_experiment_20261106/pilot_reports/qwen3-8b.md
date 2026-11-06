# E26 同实际作答率方法比较

精确配额表是批量筛题评估；排序不看正确性。冻结门槛表另报真实测试作答率。
QA分母为固定实质参考题；不够配额标不可行，未决判分不出表。

PILOT ONLY — 线路检查，不作效果结论。

## tf / 目标 25%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| baseline_simple | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| baseline_random | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| das_E | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| das_C | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| das_EC | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| md_E | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| md_C | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| md_EC | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| das_E_simple | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| das_C_simple | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| md_E_simple | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |
| md_C_simple | 1/6 | 1 | 1.0000 | [1.0000, 1.0000] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_E | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_C | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_E_simple | 0.0000 | [0.0000, 0.0000] | 1.0000 | ok |
| das_EC | das_C_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | baseline_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_E | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_C | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_E_simple | 0.0000 | [0.0000, 0.0000] | 1.0000 | ok |
| md_EC | md_C_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.3333 | 2 | 2 | 1.0000 |
| baseline_simple | 0.5000 | 3 | 3 | 1.0000 |
| baseline_random | 0.1667 | 1 | 1 | 1.0000 |
| das_E | 0.3333 | 2 | 2 | 1.0000 |
| das_C | 0.3333 | 2 | 2 | 1.0000 |
| das_EC | 0.3333 | 2 | 2 | 1.0000 |
| md_E | 0.3333 | 2 | 2 | 1.0000 |
| md_C | 0.3333 | 2 | 2 | 1.0000 |
| md_EC | 0.3333 | 2 | 2 | 1.0000 |
| das_E_simple | 0.6667 | 4 | 4 | 1.0000 |
| das_C_simple | 0.5000 | 3 | 3 | 1.0000 |
| md_E_simple | 0.5000 | 3 | 3 | 1.0000 |
| md_C_simple | 0.5000 | 3 | 3 | 1.0000 |

## tf / 目标 50%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| baseline_simple | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| baseline_random | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| das_E | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| das_C | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| das_EC | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| md_E | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| md_C | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| md_EC | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| das_E_simple | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| das_C_simple | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| md_E_simple | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |
| md_C_simple | 3/6 | 3 | 1.0000 | [1.0000, 1.0000] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_E | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_C | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_E_simple | 0.0000 | [0.0000, 0.0000] | 1.0000 | ok |
| das_EC | das_C_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | baseline_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_E | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_C | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_E_simple | 0.0000 | [0.0000, 0.0000] | 1.0000 | ok |
| md_EC | md_C_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.5000 | 3 | 3 | 1.0000 |
| baseline_simple | 0.5000 | 3 | 3 | 1.0000 |
| baseline_random | 0.1667 | 1 | 1 | 1.0000 |
| das_E | 0.5000 | 3 | 3 | 1.0000 |
| das_C | 0.5000 | 3 | 3 | 1.0000 |
| das_EC | 0.5000 | 3 | 3 | 1.0000 |
| md_E | 0.5000 | 3 | 3 | 1.0000 |
| md_C | 0.5000 | 3 | 3 | 1.0000 |
| md_EC | 0.5000 | 3 | 3 | 1.0000 |
| das_E_simple | 0.6667 | 4 | 4 | 1.0000 |
| das_C_simple | 0.5000 | 3 | 3 | 1.0000 |
| md_E_simple | 0.5000 | 3 | 3 | 1.0000 |
| md_C_simple | 0.5000 | 3 | 3 | 1.0000 |

## tf / 目标 75%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| baseline_simple | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| baseline_random | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| das_E | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| das_C | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| das_EC | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| md_E | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| md_C | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| md_EC | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| das_E_simple | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| das_C_simple | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| md_E_simple | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |
| md_C_simple | 4/6 | 4 | 1.0000 | [1.0000, 1.0000] | ok |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_E | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_C | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| das_EC | das_E_simple | 0.0000 | [0.0000, 0.0000] | 1.0000 | ok |
| das_EC | das_C_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | baseline_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_E | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_C | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |
| md_EC | md_E_simple | 0.0000 | [0.0000, 0.0000] | 1.0000 | ok |
| md_EC | md_C_simple | -0.0000 | [-0.0000, -0.0000] | 1.0000 | ok |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.6667 | 4 | 4 | 1.0000 |
| baseline_simple | 0.6667 | 4 | 4 | 1.0000 |
| baseline_random | 0.1667 | 1 | 1 | 1.0000 |
| das_E | 0.5000 | 3 | 3 | 1.0000 |
| das_C | 0.5000 | 3 | 3 | 1.0000 |
| das_EC | 0.5000 | 3 | 3 | 1.0000 |
| md_E | 0.5000 | 3 | 3 | 1.0000 |
| md_C | 0.5000 | 3 | 3 | 1.0000 |
| md_EC | 0.5000 | 3 | 3 | 1.0000 |
| das_E_simple | 0.6667 | 4 | 4 | 1.0000 |
| das_C_simple | 0.6667 | 4 | 4 | 1.0000 |
| md_E_simple | 0.6667 | 4 | 4 | 1.0000 |
| md_C_simple | 0.6667 | 4 | 4 | 1.0000 |

## qa / 目标 25%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 0/3 | 0 | — | — | undefined_zero_answers |
| baseline_simple | 0/3 | 0 | — | — | undefined_zero_answers |
| baseline_random | 0/3 | 0 | — | — | undefined_zero_answers |
| das_E | 0/3 | 0 | — | — | undefined_zero_answers |
| das_C | 0/3 | 0 | — | — | undefined_zero_answers |
| das_EC | 0/3 | 0 | — | — | undefined_zero_answers |
| md_E | 0/3 | 0 | — | — | undefined_zero_answers |
| md_C | 0/3 | 0 | — | — | undefined_zero_answers |
| md_EC | 0/3 | 0 | — | — | undefined_zero_answers |
| das_E_simple | 0/3 | 0 | — | — | undefined_zero_answers |
| das_C_simple | 0/3 | 0 | — | — | undefined_zero_answers |
| md_E_simple | 0/3 | 0 | — | — | undefined_zero_answers |
| md_C_simple | 0/3 | 0 | — | — | undefined_zero_answers |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | — | — | 1.0000 | undefined_zero_answers |
| das_EC | das_E | — | — | 1.0000 | undefined_zero_answers |
| das_EC | das_C | — | — | 1.0000 | undefined_zero_answers |
| das_EC | das_E_simple | — | — | 1.0000 | undefined_zero_answers |
| das_EC | das_C_simple | — | — | 1.0000 | undefined_zero_answers |
| md_EC | baseline_simple | — | — | 1.0000 | undefined_zero_answers |
| md_EC | md_E | — | — | 1.0000 | undefined_zero_answers |
| md_EC | md_C | — | — | 1.0000 | undefined_zero_answers |
| md_EC | md_E_simple | — | — | 1.0000 | undefined_zero_answers |
| md_EC | md_C_simple | — | — | 1.0000 | undefined_zero_answers |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.0000 | 0 | 0 | — |
| baseline_simple | 0.0000 | 0 | 0 | — |
| baseline_random | 0.0000 | 0 | 0 | — |
| das_E | 0.0000 | 0 | 0 | — |
| das_C | 0.0000 | 0 | 0 | — |
| das_EC | 0.0000 | 0 | 0 | — |
| md_E | 0.0000 | 0 | 0 | — |
| md_C | 0.0000 | 0 | 0 | — |
| md_EC | 0.0000 | 0 | 0 | — |
| das_E_simple | 0.0000 | 0 | 0 | — |
| das_C_simple | 0.0000 | 0 | 0 | — |
| md_E_simple | 0.0000 | 0 | 0 | — |
| md_C_simple | 0.0000 | 0 | 0 | — |

## qa / 目标 50%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| baseline_simple | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| baseline_random | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| das_E | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| das_C | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| das_EC | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| md_E | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| md_C | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| md_EC | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| das_E_simple | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| das_C_simple | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| md_E_simple | 1/3 | 0 | 0.0000 | — | infeasible_resamples |
| md_C_simple | 1/3 | 0 | 0.0000 | — | infeasible_resamples |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | -0.0000 | — | 1.0000 | infeasible_resamples |
| das_EC | das_E | -0.0000 | — | 1.0000 | infeasible_resamples |
| das_EC | das_C | -0.0000 | — | 1.0000 | infeasible_resamples |
| das_EC | das_E_simple | 0.0000 | — | 1.0000 | infeasible_resamples |
| das_EC | das_C_simple | -0.0000 | — | 1.0000 | infeasible_resamples |
| md_EC | baseline_simple | -0.0000 | — | 1.0000 | infeasible_resamples |
| md_EC | md_E | -0.0000 | — | 1.0000 | infeasible_resamples |
| md_EC | md_C | -0.0000 | — | 1.0000 | infeasible_resamples |
| md_EC | md_E_simple | 0.0000 | — | 1.0000 | infeasible_resamples |
| md_EC | md_C_simple | -0.0000 | — | 1.0000 | infeasible_resamples |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.0000 | 0 | 0 | — |
| baseline_simple | 0.0000 | 0 | 0 | — |
| baseline_random | 0.3333 | 1 | 0 | 0.0000 |
| das_E | 0.0000 | 0 | 0 | — |
| das_C | 0.0000 | 0 | 0 | — |
| das_EC | 0.0000 | 0 | 0 | — |
| md_E | 0.0000 | 0 | 0 | — |
| md_C | 0.0000 | 0 | 0 | — |
| md_EC | 0.0000 | 0 | 0 | — |
| das_E_simple | 0.0000 | 0 | 0 | — |
| das_C_simple | 0.0000 | 0 | 0 | — |
| md_E_simple | 0.0000 | 0 | 0 | — |
| md_C_simple | 0.0000 | 0 | 0 | — |

## qa / 目标 75%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 95%区间 | 状态 |
|---|---:|---:|---:|---|---|
| baseline_commitment | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| baseline_simple | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| baseline_random | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| das_E | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| das_C | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| das_EC | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| md_E | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| md_C | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| md_EC | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| das_E_simple | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| das_C_simple | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| md_E_simple | — | — | — | — | 无法填满配额：insufficient_actual_answers |
| md_C_simple | — | — | — | — | 无法填满配额：insufficient_actual_answers |

组合与单方向／简单筛题的直接配对差；下表统一为“组合减对照”。

| 组合 | 对照 | 已回答准确率差 | 95%区间 | 固定交付精确p | 状态 |
|---|---|---:|---|---:|---|
| das_EC | baseline_simple | — | — | — | infeasible_point |
| das_EC | das_E | — | — | — | infeasible_point |
| das_EC | das_C | — | — | — | infeasible_point |
| das_EC | das_E_simple | — | — | — | infeasible_point |
| das_EC | das_C_simple | — | — | — | infeasible_point |
| md_EC | baseline_simple | — | — | — | infeasible_point |
| md_EC | md_E | — | — | — | infeasible_point |
| md_EC | md_C | — | — | — | infeasible_point |
| md_EC | md_E_simple | — | — | — | infeasible_point |
| md_EC | md_C_simple | — | — | — | infeasible_point |

冻结调参门槛后的实际测试表现（不强制达到目标配额）：

| 方法 | 实际作答率 | 实际回答数 | 答对数 | 已回答准确率 |
|---|---:|---:|---:|---:|
| baseline_commitment | 0.3333 | 1 | 0 | 0.0000 |
| baseline_simple | 0.0000 | 0 | 0 | — |
| baseline_random | 0.3333 | 1 | 0 | 0.0000 |
| das_E | 0.3333 | 1 | 0 | 0.0000 |
| das_C | 0.3333 | 1 | 0 | 0.0000 |
| das_EC | 0.3333 | 1 | 0 | 0.0000 |
| md_E | 0.3333 | 1 | 0 | 0.0000 |
| md_C | 0.3333 | 1 | 0 | 0.0000 |
| md_EC | 0.3333 | 1 | 0 | 0.0000 |
| das_E_simple | 0.0000 | 0 | 0 | — |
| das_C_simple | 0.0000 | 0 | 0 | — |
| md_E_simple | 0.0000 | 0 | 0 | — |
| md_C_simple | 0.0000 | 0 | 0 | — |
