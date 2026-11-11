# P4 的代码

撰写：Claude（赵磊审）。北京时间 2026-11-11 16:39；18:30 补 E 主结果复核的脚本（第 3 节表中 3.2 末一行）与真假判断早期 Unknown 版本的来源（1.5 一行）；11-12 00:30 快照里的 `e44_eval_extra.py` 换成 GPT 第 34 轮审修后的版本，补自检与 E + C 的运行脚本（第 2 节最后三行、第 3 节 3.2 末与 3.7 两行）。

**完整、可运行的代码树**在 [`claude_gpt_review/from_claude/code/`](../../../../claude_gpt_review/from_claude/code/)（`core/` 核心模块、`scripts/` 入口、`configs/` 提示与配置、`tests/` 单元测试）。本目录的 [`快照/`](快照/) 是 P4 用到的关键文件的副本，方便直接读；以完整代码树为准。服务器上跑的是同一份代码（GPT 第 28 轮审完的版本，提交 6bfca9f），从审完到跑完一个字节没改。

## 1. 整条流水线

```
冻结  四种问法（加不加如实句 × 必须作答 / 允许弃答）+ 对照问法，对每道题各生成一次回答，存下回答与每一步的输出分布
  ↓   scripts/e44_factorial_das.py freeze
训练  每层一条方向：把两种问法在提示末位的状态差，沿方向写进另一种问法，要求输出分布变得像目标问法（改这一格）或不变（别动另一格）
  ↓   scripts/e44_factorial_das.py train      （核心在 core/qa_factorial_das.py）
生成  带写入地生成回答：固定平移（力度 × 平均差 × 方向）、按题交换、均值差方向、随机方向、反向写入、"别动另一格"的检查
  ↓   scripts/e44_factorial_das.py generate
导出  把各条件的回答整理成裁判输入
  ↓   scripts/e44_export_judge.py
判分  两家裁判各判各的：AllenAI 的两个 7B 裁判（scripts/allenai_tqa_judge.py）+ 规则弃答；DeepSeek V4 Pro（scripts/review_checks/judge_two_axis_run.py llm）
  ↓
出表  判据程序按事先写死的判据逐条判（results_09_scripts/e44_round1_tables.py），其余是只读的分析脚本
```

## 2. 快照里的文件

| 文件（`快照/` 下） | 作用 | 谁写的 | 审核状态 |
|---|---|---|---|
| `code/core/qa_factorial_das.py` | 写入钩子（只写两种问法的差）、逐词 JSD 损失、训练一条方向、带写入的生成与打分 | Claude | GPT 第 28 轮审修 |
| `code/core/qa_prompt_das.py` | 提示渲染、目标构造、位置掩码、状态读取（上一版方案留下的公共部件，原样复用） | Claude | GPT 第 25–27 轮审过 |
| `code/scripts/e44_factorial_das.py` | 驱动：freeze / train / generate / mc / summarize；每个产物带实验身份与文件指纹，身份对不上就停 | Claude | GPT 第 28 轮审修 |
| `code/scripts/e43_qa_prompt_das.py` | 驱动用到的公共函数（身份、冻结文件的校验、封存） | Claude | GPT 第 25–27 轮审过 |
| `code/scripts/e44_export_judge.py` | 导出裁判输入 | Claude | GPT 第 28 轮审修 |
| `code/configs/prompts/e44_factorial_v1.json`、`default_v1.json` | 全部提示的原文（四种问法、如实句、句式对照句、大众误解句） | Claude | 同上 |
| `code/tests/test_qa_factorial_das.py` | 单元测试（玩具模型 + 真分词器），连同 GPT 的检查共 249 项通过 | Claude、GPT | — |
| `code/scripts/allenai_tqa_judge.py`、`code/scripts/review_checks/judge_two_axis_run.py` | 两家裁判的程序（Pro 的判分提示也在后者里） | GPT 交付、Claude 改 | GPT 审过 |
| `results_09_scripts/*.sh` | 运行脚本，只按参数调用上面的入口 | Claude | — |
| `results_09_scripts/e44_round1_tables.py`、`test_round1_tables.py` | **判据程序**：把事先写死的判据写成代码，带反例自测 | Claude | 红队 + GPT 第 30、31、33 轮挑错后的版本 |
| `results_09_scripts/` 里其余的 `.py` | 只读的分析脚本（训练损失表、方向夹角、写入量、标签变化的原文、阶段 B 把握的模拟等） | Claude | 未单独送审 |
| `code/scripts/e44_eval_extra.py`、`code/configs/prompts/e44_eval_extra_v1.json`、`code/tests/test_e44_eval_extra.py` | 只做评估、不训练的新脚本：E + C 两层同时写、只用于评估的新提示（如必须作答 + 完整句 conf_F）、把 E 的写入拆成平行 / 垂直于均值差的两份、只读的状态检查、导出给裁判 | Claude 写，GPT 改 | **GPT 第 34 轮审修**（提交 04ee956，修了 7 类问题；快照是修后的版本 `4438a282…`）；服务器 CPU 45 项测试通过；真模型自检通过（16 题、192 题；普通问法与完整句问法各一次） |
| `results_09_scripts/e44_extra_selfcheck.sh`、`e44_extra_EC.sh`、`chain_EC_F.sh`、`chain_split_nat.sh` | 运行脚本：真模型自检；E + C 网格与对照、完整句背景、拆分对照、只读检查（只按参数调用上面的入口） | Claude | 方案事先写进 `from_claude/36`、`37`、`33` 第 1.7 节 |
| `results_09_scripts/e44_extra_EC_check.py`、`e44_EC_describe.py` | 只读：单写臂与原程序逐词元比对、按规则弃答数选档；E + C 的描述表（按完整条件名取行，不是正式的组合分析器） | Claude | 未单独送审；描述表已请 GPT 审 |

## 3. 每张结果表 → 脚本 → 命令

服务器上的结果目录 `results_e44_factorial_das/llama3.1-8b`（实验身份 `ef17e8300ee3`）。下面的 `$RUN` 指这个目录，`$WORK` 指它下面的某个判分目录。

| P4 页的结果 | 跑的脚本 | 出表命令 | 表（`../tables/`） |
|---|---|---|---|
| 1.5 真假判断最早一版（可以答 Unknown 的问法） | 旧协议的评测脚本 [`code/scripts/e2b_h1_rescue.py`](../../../../claude_gpt_review/from_claude/code/scripts/e2b_h1_rescue.py)（第 111 行 `allow_idk=True`）；主结果用的强制版读法见 [`code/scripts/e23_knob_sweep.py`](../../../../claude_gpt_review/from_claude/code/scripts/e23_knob_sweep.py) 第 144、146、232 行 | 从本地结果文件 `results_tqa/E2b/llama3.1-8b/e2b_llama3.1-8b.json` 原样整理 | `tf_early_unknown_option_E2b_llama_tqa.md` |
| 3.1 第 0 步（只换问法） | `e44_step0.sh`、`e44_step0_pro.sh` | `e44_step0_two_judges.py` | `step0_two_judges_tune.md` |
| 3.2 E 在必须作答问法上（力度 1 / 2） | `e44_round1_all.sh` → `e44_round1_extra_gen.sh` → `chain_dose2.sh`（两家裁判） | `e44_round1_tables.py cells $RUN E $WORK --judge pro --layers 10 --base conf --main conf.w.p2 --strength 2` | `round1_cells_E_two_judges_conf_base_*.md`、`round1_dose_E_two_judges_L8_L10.md`、`round1_flips_E_L10_conf_strength2_pro.md` |
| 3.2 末 E 主结果的复核（再两个训练种子、再两条随机方向） | `chain_main_E_robust.sh` → `e44_main_E_robust.sh`（只调已审的入口：种子 = 独立目录 + 同一批冻结文件 + `train --seed n`；随机方向 = 独立目录 + 原方向文件 + `generate --seed n`）→ `e44_round1_pro.sh`（两家裁判） | `e44_round1_tables.py cells`（同上，`--layers 10 --base conf --main conf.w.p2 --strength 2`）；`e44_arms_table.py` | `mainE_seeds_cells_E_conf_strength2_two_judges_L10.md`、`mainE_seeds_arms_two_judges_L10.md`、`mainE_axes_two_judges_L10.md` |
| 3.2 末 完整句背景上的检查（conf_F） | `chain_EC_F.sh` → `e44_extra_EC.sh STAGE=genCF`（`e44_eval_extra.py generate --bases conf_F conf_E1_F --arms conf_F:E10.w.p1 …`）→ `STAGE=judgeF`（两家裁判） | `e44_EC_describe.py $WORK --judge pro` | `EC_F_background_describe_two_judges_*.md` |
| 3.7 E + C 同时用 | `e44_extra_selfcheck.sh`（自检）→ `e44_extra_EC.sh STAGE=gen1`（网格 + 单写比对，`e44_extra_EC_check.py` 选档）→ `STAGE=gen2 KC=0.25`（选中档的对照）→ `STAGE=judge`；完整句背景：`chain_EC_F.sh`（`STAGE=genF` → `judgeF`） | `e44_EC_describe.py $WORK --judge pro` / `--judge allenai` | `EC_x_describe_two_judges_*.md`、`EC_F_background_describe_two_judges_*.md` |
| 3.3 C 的力度扫描与按题交换 | 同上 + `chain_dose.sh` | `e44_round1_tables.py cells $RUN C $WORK --judge pro --layers 31`；`e44_arms_table.py $WORK --judge pro` | `round1_cells_C_two_judges_L19_L31.md`、`round1_dose_C_two_judges_L31_L19.md`、`round1_cross_and_L14_C_two_judges.md` |
| 3.4 E 在普通问法上 | 同上 | `e44_round1_tables.py screen $RUN E $WORK`；`cells … --layers 8 9 10` | `round1_screen_E_allenai.md`、`round1_cells_E_two_judges_x_strength1_L8_9_10.md`、`round1_cells_E_x_strength2_and_C_strength05_two_judges.md` |
| 3.5 训练层面的旁证 | — | `e44_train_table.py`、`e44_w_adjacent.py`、`e44_write_sizes.py`（只读方向文件） | `round1_training_losses.md`、`round1_direction_cosines.md`、`round1_write_sizes_learned_vs_mean_difference.md` |
| 3.6 三组旁支 | `e44_round1_all.sh`（同批）；旁支 M 的补臂与复核：`chain_side_M.sh`、`chain_side_M_robust2.sh` | `e44_round1_tables.py cells`（同上）；`e44_layer_profile.py`；`e44_stageB_power.py`（阶段 B 把握的模拟） | `side_M_cells_E_two_judges_L11_12_13_17_10.md`、`side_M_seeds_cells_E_two_judges_L11_L13.md`、`side_M_axes_E_two_judges_L11_L13.md` |

**训练与生成的固定参数**（所有运行一样，也是实验身份的一部分）：`--batch-size 8 --max-new-tokens 48 --max-target-tokens 24`，Adam、学习率 1e-3、10 轮、每层留出 16 道训练题不参与梯度、随机种子 42。

## 4. 审核记录（只链接）

- 方案：[`from_claude/25`](../../../../claude_gpt_review/from_claude/25_E43复核与新方案E44_两因子交换训练_自由问答2x2_请GPT审.md)；判据定稿：[`from_claude/30`](../../../../claude_gpt_review/from_claude/30_回复GPT第31轮_判据v4两栏定稿_随机轴不改代码的做法.md) 第 4 节
- GPT 的审核：`claude_gpt_review/from_gpt/review_20261110_28.md`（代码审修）、`_29.md`、`review_20261111_30.md`、`_31.md`（判据）、`_33.md`（首轮结果）、`_34.md`（评估代码审修与方法裁定）
- 送审中的新代码说明：[`from_claude/33`](../../../../claude_gpt_review/from_claude/33_评估代码送审_E加C两层同时写_完整句对照_拆分对照_身份检查_三组旁支的AllenAI结果.md)；GPT 审修后的核对与真模型自检：[`from_claude/36`](../../../../claude_gpt_review/from_claude/36_回复GPT第34轮_全部核对_修后评估代码过了真模型自检_E加C网格与选档规则事先写死.md)；E + C 的结果：[`from_claude/37`](../../../../claude_gpt_review/from_claude/37_E加C第一二步结果_描述不下结论_普通问法上作答数回到基线答对多29题_接着跑完整句背景检查.md)
