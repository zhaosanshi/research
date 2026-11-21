# 2026-11-20 探索用程序的审核请求（成批）

撰写：赵磊（Fable 5.2 代笔）。日期：2026-11-20（北京时间）。

按赵磊的规矩，程序先审再上 GPU；探索阶段允许先跑，成批送审。本页列出 2026-11-20 这一天新写或改动、还没有审过的程序，请 GPT 审。下文路径均相对于仓库的 `Truth_Decoupling/Iteration_01/`。

## 1. 先读

- 结果汇总：`novelty/ours/2026-11-20_per_model_strength_results.md`
- 能力保持（MMLU）的设计与结果：`novelty/ours/2026-11-19_capability_preservation_design.md` 第 8 节
- 问题清单：`discussion/open_problems/2026-11-20_problems_by_priority.md`

## 2. 要审的程序

| 文件（`code/` 下） | 作用 | sha256 前 16 位 |
|---|---|---|
| `scripts/capability/cap_mcq_prob.py` | MMLU 的选项概率测法（比四个字母的概率，每个字母的两种单词元写法相加） | 514fdc7c4c9c1697 |
| `scripts/capability/cap_mcq_sweep.py` | 同一测法下换写入方式：力度、反向、随机方向、均值差方向、学到的方向拆两部分、只从提问末尾写（`@gen`）；一次前向读出四个字母；写入可以叠在该次运行的其他提示词上 | ca1eab3469d4ab6c |
| `scripts/review_checks/e44_positions.py` | 只在评测时换写入位置的入口：原程序文件不动，内存内三处改动（`Context.writes` 返回新位置规则；`arm_token` 加后缀避免文件重名；`run_export` 接受该位置）；只许单写入的生成与导出 | 1a8ae7861c1c103e |
| `diag/dose_table.py` | 力度 1 与 2 的主要系统并排，含强对照、留空再问 B、纯提示词 | d1e33c892a41fc8f |
| `diag/dose_mid.py` | 各模型每一档力度下的只推 E 与 N1（上限逐档在调参集上选），对不写入、只推 E、纯提示词；末尾按调参集给每个模型选一个力度 | 4dfd2f712f376a0d |
| `diag/strong_mid.py` | N1 对同一力度的强对照、留空再问 B、必须作答加 E 全答；力度 1、2 必须复现 `dose_table.py` | af569511e312a3b1 |
| `diag/on_prompt_table.py` | E 写在“必须作答加如实句”之上，对比该提示词本身 | 4d1367ce018287a6 |
| `diag/md_table.py` | 学出的方向对均值差方向（同层；以及均值差自己挑层与系数之后） | 4cfad0ab73ef362a |
| `diag/md_choose.py` | 均值差基线在调参集上选层与系数 | 43b30cacd011e8b1 |
| `diag/prompt_sweep_table.py` | 11 条提示都当纯提示词基线；每个模型在调参集上选最好的一条，再与 N1 比 | 271550ea2d70f706 |
| `diag/pos_table.py` | 只从提问最后一个词元起写 E 的 TruthfulQA 净分，对不写入、纯提示词、同力度的原写法 | 863f0edb466332a3 |
| `diag/tf_net_score.py` | 真假判断（同一遍可答 Unknown）按净分重算；每个运行的计数须复现其已存汇总 | 6e4a77fdce43cd2d |
| `diag/cap_mcq_flips.py` | 逐题看推 E 把 MMLU 的选项改成了什么 | 62370da7eb44d5d7 |
| `diag/cap_mcq_fresh.py` | MMLU 全测试集的计分：没看过的 13,042 题、看过的 1,000 题、全部 | 72ae112cd10bece0 |
| `diag/newsets_val.py` | 新题集第二批（test2）上重选上限后的验证计分 | 9f49f62b0436543e |
| `diag/newsets_tqa_shared_answers.py` | TriviaQA 两批题共用答案的排查 | ce5c4a083e178f43 |

启动脚本 20 个，在 `code/launchers/explore_20261120/`。它们调用的生成、导出、判分程序都是已审的版本，脚本只改臂、切分、名称与显卡；新题验证用的 `code/scripts/newsets/` 改动（`newsets_splits.py` 的 extend 模式、`newsets_bank.py`、`newsets_gen.py` 接受 test2）也在本批。

## 3. 请重点看的地方

1. **`e44_positions.py` 的三处内存内改动是否改变了别的行为**。它用 `Context.writes` 的返回值换位置规则，引擎里 `fd.generate_shift(..., positions=...)` 原本就支持 `gen`（只在提问最后一个词元与每个生成步写入）。导出时把 `rec["training_contract"]["positions"] != d["positions"]` 这一行换成只接受新规则。单写入条件在导出里的名字与原规则相同（`E_L<层>_<提示>-w-p<力度>`），靠单独的导出名与工作目录区分；`diag/pos_table.py` 只从该目录读。请看是否有混用的风险。
2. **`cap_mcq_sweep.py` 与 `cap_mcq_prob.py` 的一致性**。前者一次前向读四个字母，后者逐个字母教师强制；`--ref` 核对过逐题相同。`@gen` 用的是 `fd.position_mask(packed, "gen")`。
3. **逐档力度在调参集上选上限**（`dose_mid.py`）。新旧工作目录的判分合并成一张表（`newsets_score.NewSplit`），同一格出现在两个目录时以新目录为准，并报告不一致的格数（目前为 0）。加入更多力度后，若某题在新条件里没有判决，该题从这个模型的所有比较里剔除（Qwen3.8-27B 由 522 题变 521 题）。
4. **强对照在非 1、2 力度下的把握分**（`strong_mid.py`）。把握分文件由 `ff_test_confidence.py` 原样生成；使用前逐题核对把握分行的文本哈希与被判回答一致。只跑 1 批（8 题）时该程序 98% 复现检查会因样本太少而报错，全量通过。
5. **判分缓存的合并**。新工作目录先并入同一模型、同一切分的一个旧缓存，相同请求不重复付费，也使相同回答沿用主表所用的判决。并两个旧缓存会因同一请求的不同答复而停（Llama 23 条），所以一律只并一个。请看这样做有没有让新旧条件的判决来源不对称。
6. **提示词基线全判**（`prompt_sweep.sh`）。用的是各次运行冻结文件里 11 条提示的无写入回答，不重新生成；导出走 `e44_eval_extra.py export --gen --prompts-out`。
7. **均值差选层**（`md_sweep.sh`、`md_choose.py`）。均值差用引擎的 `md` 写法（单位均值差乘它自己的范数）在系数 1、2、4 与每个有方向记录的层上扫调参集，选净分最高的一格，再上测试集。请看这是否算给了基线足够公平的选择空间。
8. **统计口径**。多模型合计一律按题把各模型的差相加后按题重抽；某一力度只有部分模型有时不进合计。

## 4. 已知的局限

- MMLU 的 1,000 题口径多次比全量乐观，结论只用全量（13,042 道没看过的题）。
- 写在纯提示词之上的各档力度，测试集先于调参集生成（调参集当天补做）；`novelty/ours/2026-11-20_per_model_strength_results.md` 第 4 节分开列了两者。
- 只在作答位置写入是诊断，不是方法设置；两个方向同时写时的验收记录（selfcheck）只对训练时的位置规则有效，所以该入口不允许双写入。

## 本次答复

审至 `89aedc1`。[GPT 的审核与复算](2026-11-21_gpt_review.md)。

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-20 | 建立本文件 | 赵磊（Fable 5.2 代笔） |
| 2026-11-21 | 添加本批审核答复入口 | 赵磊（GPT-6.2-Astra 代笔） |
