# 09 E44 两因子交换训练（Llama-3.1-8B）：第 0 步结果与首轮运行记录

方案 [`25`](../../25_E43复核与新方案E44_两因子交换训练_自由问答2x2_请GPT审.md) → 修订 [`26`](../../26_回复GPT第28轮_核验_更正_早层Pro结果与首轮计划.md) 第 6 节 → 臂与第 0 步 [`27`](../../27_回复GPT第29轮_两家裁判各判2x2的判据_首轮定稿_第0步结果.md) → **判据与选层规则以 [`28`](../../28_自由问答2x2的判据与选层规则_定稿v2_红队后重写.md) 加 [`29`](../../29_回复GPT第30轮_采纳与坚持_判据v3修订_防句式改用设计式对照.md) 的修订、[`30`](../../30_回复GPT第31轮_判据v4两栏定稿_随机轴不改代码的做法.md) 第 4 节的完整重述（v4）为准**。代码 = GPT 第 28 轮审修后的 6bfca9f，一个字节没改；实验身份 `ef17e8300ee3`。服务器结果目录 `results_e44_factorial_das/llama3.1-8b`（旁支在 `…_side_M`、`…_side_E2`、`…_side_initmd`）。

## 状态（北京时间）

| 步骤 | 状态 |
|---|---|
| 第 0 步：11 条提示冻结（训练 96 + 调参 192 题）、参考答案读数、AllenAI 判分 | 11-10 23:36–23:48 跑完 |
| 第 0 步的 DeepSeek Pro 判分（11 条提示，调参题） | 11-10 23:50 – 11-11 00:20 判完 |
| 首轮主假设：E、C 各 32 层，训练 → 生成 → 参考读数 → AllenAI | 11-11 00:00–03:15 跑完，无报错 |
| 首轮旁支：M、E2、MD（只训 E，各改一个因素） | 11-11 03:12 起，在跑 |
| 入围层与探索臂的两家裁判判分 | 11-11 03:21–04:25 完成；结果见 [`31`](../../31_E44首轮结果_两家裁判_C成立_E在必须作答提示上成立_普通提示上按冻结口径不成立.md) |

**首轮主假设的结果已出，读法见 [`31`](../../31_E44首轮结果_两家裁判_C成立_E在必须作答提示上成立_普通提示上按冻结口径不成立.md)。** 一句话：C 成立（双向调作答量、作答里的正确率不动）；E 在必须作答的提示上两家裁判都成立（第 10 层力度 2：Pro 判答对 70 → 106，作答数不变），在普通提示上按冻结口径不成立。

## `tables/`（第 0 步）

| 文件 | 内容 |
|---|---|
| `step0_allenai_rule_refusals_tune.md`、`…_train.md` | 11 条提示的 答对 / 答错 / 无信息 / 弃答（AllenAI，规则弃答先拆出来），调参 192 题与训练 93 题 |
| `step0_allenai_raw_tune.md`、`…_train.md` | AllenAI 单裁判原始 A / B / N 计数（弃答没拆；约一半 "I don't know." 被它判成 A，所以不能直接读） |
| `step0_frozen_answers_tune.md`、`…_train.md` | 不靠裁判的统计：规则弃答数、未写完数、句末标点、平均词数；成对提示之间回答相同的题数与弃答的交叉；前 6 题原文 |
| `step0_reference_readout.md` | 不写入时每条提示的参考答案读数（任一真答案排第一的比例、MC2）与 "I don't know." 的对数概率 |
| `step0_two_judges_tune.md` | **两家裁判并排**：11 条提示的四类计数；成对提示之间的配对变化（错→对、对→错、弃答进出、共同作答题）；conf → conf_E1 的原文抽样 |
| `step0_pro_raw_tune.md` | Pro 单裁判原始 A / B / C / H 计数 |
| `step0_length_strata_tune.md` | **按"回答有没有变长"拆开**的答对变化：句式对照的增益全在变长的回答上，如实句的增益有一半在没变长的回答上（`27` 第 10.3 节） |
| `e43_supervised_candidate_length_strata_tune.md` | 同一拆法用在 E43 金标方向第 0 / 3 / 10 / 18 层的已有判分上：第 3、10 层的增益全在变长的回答上（`27` 第 10.5 节） |
| `step0_same_length_v2_tune.md` | **按 `28` 的定义重算**（只看写入前后都实质作答、且至多多 1 个词的题）：提示对比与 E43 金标方向各层；`28` 第 5 节的校准表出自这里 |
| `step0_prompt_factors_under_criteria_v2_tune.md` | 提示因子本身按 `28` 的判据（没有随机对照那一条）：Pro 下四格都过；AllenAI 下两个保持格各差 1–3 题 |

读法见 `27` 第 5 节与第 10 节（修订 1）。**第 0 步里最要紧的一条**：只换句式（x_F），AllenAI 多判 24 题、Pro 也多判 16 题答对，作答数几乎不变——"答案变长"在两家裁判下都涨分，所以判据加了"没变长的题上也要涨"这一条。其余要点：两个提示因子按设计工作（必须作答档弃答 0–2，允许弃答档 27–37；如实句在必须作答档让答对 88 → 108）；"说大家普遍相信的"与如实句的两极差 44 题；如实句在允许弃答档会多带出弃答；只换句式（x_F）AllenAI 就多判 24 题答对，而参考读数没动。

## `tables/`（首轮，`round1_*`）

| 文件 | 内容 |
|---|---|
| `round1_screen_E_allenai.md`、`round1_screen_C_allenai.md` | 按冻结规则对全部 32 层的筛选与入围（AllenAI + 规则弃答） |
| `round1_cells_E_two_judges_x_strength1_L8_9_10.md` | **冻结口径**：E 在普通提示上、力度 1，两家裁判的 A / B 两栏 |
| `round1_cells_C_two_judges_L19_L31.md`、`round1_cross_and_L14_C_two_judges.md` | **冻结口径**：C 两家裁判；按题交换臂；诊断层第 14 层 |
| `round1_cells_E_two_judges_conf_base_exploratory_L8_9_10.md` | 探索：E 在必须作答提示上（固定平移 + 随机 / 均值差对照 + 按题交换），力度 1 |
| `round1_cells_E_two_judges_conf_base_strength2_exploratory_L10_L8.md` | 探索：同上，力度 2 |
| `round1_dose_E_two_judges_L8_L10.md`、`round1_dose_E_allenai_L8_L10.md` | 探索：E 的力度扫描与同力度对照（普通提示、完整句背景、必须作答提示、反向） |
| `round1_dose_C_two_judges_L31_L19.md`、`round1_dose_C_allenai_L19_L31.md` | 探索：C 的力度扫描 |
| `round1_cells_E_x_strength2_and_C_strength05_two_judges.md` | 探索：E 在普通提示上力度 2；C 力度 0.5 |
| `round1_layer_profile_E_allenai.md`、`round1_cross_E_allenai_L3_8_9_10_12_16_20.md` | E 各臂的 32 层剖面（AllenAI） |
| `round1_cells_E_allenai_L8_9_10_20_before_extra_arms.md` | 补对照臂之前的 E 表（留档） |
| `round1_training_losses.md` | 64 条方向的训练损失：训练题与留出题、不写入 / 训练出的方向 / 均值差方向（脚本 `e44_train_table.py`） |
| `round1_write_sizes_learned_vs_mean_difference.md` | 64 条方向：训练方向臂与均值差臂每单位力度各写多少（均值差臂 = 训练方向臂那一份 + 与它垂直的一份，量是 1.4–1.65 倍），以及按题交换写入的坐标逐题的散布（脚本 `e44_write_sizes.py`，只读方向文件） |
| `round1_flips_E_L10_conf_strength2_pro.md` | 必须作答问法、第 10 层力度 2：Pro 下标签变了的 58 道题的两版回答，附 AllenAI 的判法（脚本 `e44_flips.py`） |
| `round1_direction_cosines.md` | 64 条方向相邻层之间、同层 E 与 C 之间的夹角余弦（脚本 `e44_w_adjacent.py` 的完整输出） |
| `side_M_layer_profile_E_allenai.md`、`side_E2_layer_profile_E_allenai.md`、`side_initmd_layer_profile_E_allenai.md` | 三组旁支的 32 层（MD 为 3 层）剖面：各臂对自己基线的答对数变化（实质作答数变化），AllenAI（脚本 `e44_layer_profile.py`） |
| `side_M_screen_E_allenai.md` | M 旁支按冻结的筛选程序（`e44_round1_tables.py screen` 原样）得到的 32 层表与入围层 |
| `side_branch_direction_cosines.md` | 主实验与三组旁支的 E 方向逐层夹角余弦、各自的写入量（脚本 `e44_branch_cosines.py`，只读方向文件） |

## `scripts/`（运行脚本，只按参数调用审过的入口）

| 文件 | 作用 |
|---|---|
| `e44_step0.sh` | 第 0 步：冻结 → 参考读数 → 导出 → AllenAI |
| `e44_step0_pro.sh` | 第 0 步的 Pro 判分（复用 E43 两次试判的缓存） |
| `e44_round1.sh` | 首轮单个分支（或它的一半层）的 训练 → 生成 → 参考读数；`STAGE=judge` 导出并做 AllenAI 判分；`STAGE=pilot` 三题试跑 |
| `e44_round1_all.sh` | 四张卡的排队：主假设先跑，判分后接旁支；每个进程只用一张卡 |
| `e44_step0_look.py` | 只读：冻结回答的统计表（CPU，不加载模型） |
| `e44_step0_two_judges.py` | 只读：两家裁判并排表、成对提示的配对变化、原文抽样 |
| `e44_step0_length_strata.py` | 只读：按"回答有没有变长"拆开的答对变化与自举区间；`--pairs` 可用于任意导出里的两个条件（E43 的表就是这样出的） |
| `e44_round1_tables.py` | 只读：把 **`30` 第 4 节（v4）** 的判据与选层规则写成程序；每层输出 `A=`（筛选，点估计）与 `B=`（确认，区间）两个状态；判据冻结之后加了两个探索选项 `--base`、`--strength`（输出里标 EXPLORATORY，默认行为不变）（`screen` 全层筛并列入围层；`cells` 某家裁判下的两格、自举区间与每层一行 `KEY`；`cross` 其余臂对各自基线）。只用标准库。**随判据一起、先于任何写入臂的判分表提交** |
| `test_round1_tables.py` | 上一个脚本的合成数据自测：造已知大小的效果，检查入围名额与过 / 不过的判定；红队与 GPT 第 30、31 轮造的反例都在里面（本机 CPU 通过） |
| `e44_same_length_v2.py` | 只读：任意两个条件之间按 `28` 的定义算 G（都作答且没变长的题上的答对增益）与自举区间 |
| `e44_prompt_factor_cells.py` | 只读：把提示因子本身（加如实句；必须作答 ↔ 允许弃答）代进 `28` 的判据，看这条线有多严 |
| `e44_round1_pro.sh` | 入围层送 DeepSeek Pro（每层一个进程；基线复用第 0 步的 Pro 缓存；给了显卡号就把同一批格子也交 AllenAI 判）。还没运行 |
| `e44_round1_extra_gen.sh` | 入围层补生成：随机对照的另一个符号 `x.rand.m1`、防句式对照 `x_F.w.p1`（同一条 E 写入打在已要求完整句子的提示上）。还没运行 |
| `e44_judge_allenai_subset.sh` | 探索臂的 AllenAI 判分（任意臂的子集） |
| `e44_arms_table.py` | 只读：任意导出里每条臂对各自基线的一行表（力度扫描用） |
| `e44_w_shape.py`、`e44_w_init_share.py`、`e44_w_adjacent.py` | 只读（CPU）：已训练方向的形状——与初始向量的夹角（约 0）、与均值差方向的夹角（各层都约 0.70）、相邻层之间的夹角（E 0.55–0.93、C 0.41–0.94，越往后越高；此前写的 0.81–0.91 只是后段层，全表见 `tables/round1_direction_cosines.md`）、E 与 C 同层的夹角 |

这些脚本是 Claude 写的、没经 GPT 审。五个运行脚本没有新的计算逻辑，请抽查参数（臂名、层、`--e-on / --e-off`、批大小 8、最大生成 48 词元、目标 24 词元）；**`e44_round1_tables.py` 有新逻辑（四类映射、配对自举、入围名额、过 / 不过的判定），请 GPT 重点看**，它只读存盘产物、不加载模型。起跑前三题试跑抓到并修掉的两处问题记在 `27` 第 4.5 节。

四张卡各冻结一遍调参题，文件逐字节相同（sha256 前 16 位 `107b117d5e4befcc`）。
