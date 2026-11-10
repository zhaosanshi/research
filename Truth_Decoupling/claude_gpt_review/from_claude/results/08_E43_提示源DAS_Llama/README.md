# 08 E43（`21` 第四版，提示来源 + 钳制 + 整句交叉熵）Llama-3.1-8B 结果与复核表（2026-11-10）

正文与结论在 [`../../25_E43复核与新方案E44_两因子交换训练_自由问答2x2_请GPT审.md`](../../25_E43复核与新方案E44_两因子交换训练_自由问答2x2_请GPT审.md) 第 1–2 节。这里只放表和脚本。

- 跑的是什么：代码 = GPT 第 27 轮 e89b0d8 逐字节原样，单卡，身份指纹 `ca4cffc652cf`；服务器 `results_e43_qa_prompt_das/llama3.1-8b`（13:11 起的四卡跑已作废并归档，不用）。
- 进度：18:28 全部跑完（训练 64 条方向；调参 192 题与训练 96 题生成；随机对照臂），18:37 完整 AllenAI 判分完成。
- 判分：AllenAI 两个 7B 判官（审过的脚本原样），规则弃答单独拆开；Pro 只做了一个小试判（四条提示对照 + E 第 18 层主臂）。

## tables/

| 文件 | 内容 |
|---|---|
| `summary_from_driver.md` | 驱动自己的 `summarize` 输出（不靠裁判，全部 704 个生成文件） |
| `tune_judge_free_all.md`、`train_judge_free_all.md` | 不靠裁判：每层每臂的改动条数、弃答前后、与另一侧提示回答一致的条数、写入量 |
| `tune_allenai_all_arms.md`、`train_allenai_all_arms.md` | 32 层、全部臂（含随机对照）的 AllenAI 判分：答对 / 答错 / 真但无信息 / 规则弃答，与配对翻转。训练集有 3 题金标只有弃答、不判 |
| `prompt_controls_allenai.md`、`prompt_controls_single_allenai_summary.md` | 四条提示对照（调参集） |
| `E_tune_format_and_flips.md` | E 各臂：句号结尾条数、平均词数、否定词、每位置写入量对两端差；全部条件的翻转按"同文本 / 没变长 / 变长"三分；按金标含不含否定词分层 |
| `tune_surface_only_flips.md` | 只变标点时裁判标签变了多少（E、C 的主臂） |
| `directions_audit.md` | 64 条方向：读取位坐标、配对差、与均值差的夹角、损失曲线、E 与 C 的夹角、与作废那一跑的同层余弦 |
| `pair_audit.md` | 坐标是否跟行为走（C：弃答 AUC；E：指令改没改答案的 AUC） |
| `C_switch_pairs_ce.md` | C 的 18 对切换题逐对的训练后交叉熵 |
| `pro_trial_two_judges.md` | 第一个 Pro 小试判（四条提示对照 + E 第 18 层主臂）与 AllenAI 的对照 |
| `pro_trial2_two_judges_L0_L3_L10.md` | 第二个 Pro 小试判（四条提示对照 + E 主臂第 0 / 3 / 10 层；这三层是看了 AllenAI 表之后挑的） |

## scripts/

`e43_full.sh` 是这一跑的执行脚本，`e43_judge_allenai.sh` 是完整判分；`e43_judge_promptctl.sh`、`e43_judge_partial.sh`、`e43_judge_protrial.sh`、`e43_judge_protrial2.sh` 只是按不同文件子集调用审过的导出 / 判分脚本（单独工作目录，不碰最终的 `judge_tune/`）。其余 `e43_*.py` 是**只读**的表格脚本：Claude 写的、**没经 GPT 审**，只读存盘产物、CPU、不加载模型。请抽查。

**GPT 第 28 轮用自己的脚本从原始产物独立复算过（`../../../from_gpt/audit_20261110_round28/e43_numbers_review.md`）**：主要计数一致；要注意的口径——表里"句号结尾"统计的是以句号、问号或叹号结尾；三列的摘要省了"真但无信息"一类，完整四类看 `tune_allenai_all_arms.md`；"B→A / A→B"只是答错与答对之间的互换，不等于答对总数的变化。
