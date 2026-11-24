# 代码

撰写：赵磊（Opus 5.5 代笔）。起始日期：2026-11-16（北京时间）。

本目录存放从服务器迁入、供审核与讨论的程序。路径与服务器上的 `Iteration_01/code/`、`Iteration_01/launchers/`、`Iteration_01/diag/` 相同，以服务器上的文件为准。下表记录迁入时 sha256 的前 16 位。

| 文件 | 用途 | 状态 | sha256 前 16 位 |
|---|---|---|---|
| `scripts/review_checks/ff_ec_probe.py` | 机制探测：推 E 之后，下游层在 C 方向上的读数变化 | 按 GPT 审核意见修改（B1、S1–S3），CPU 测试通过，已在两个 12B 模型上运行（见 `discussion/mechanism_probe/2026-11-17_fix_report.md`）；修改前为 634b3c2361e4a91b | d1432b3a5359210a |
| `diag/doubt_signal.py` | 疑心能否分出对错："I don't know."对数概率在推 E 前后的变化对调参集标签的区分度（`novelty/ours/2026-11-17_e_doubt_c_commit.md` 第七节第 1 步） | 新写，单元测试与 Opus 子代理红队通过，已运行 | 780a287e62c62466 |
| `diag/doubt_policy.py` | 调参集上的疑心门槛策略，对半分选设置、另一半报告（同上第 2 步） | 新写，单元测试通过（与 `policy_select2.py` 逐题一致），待运行 | 6b42708b8e751780 |
| `diag/probe_vs_doubt.py` | 机制探测的逐题读数变化与疑心的相关、对标签的区分度（同上第 5 步） | 新写，已运行 | 04a9869e641522e0 |
| `launchers/ff_tune_pro_20261117.sh` | 给已有 AllenAI 判分的调参集目录补 DeepSeek Pro 判分（同上第 0 步），只调用现有判分脚本 | 新写，已运行 | 8b1f3fb550af58b0 |
| `launchers/ff_g3_fresh_pro_offpeak_20261118.sh` | 给两个 Gemma-3 模型测试集中 Pro 未判过的题补判，只在 DeepSeek 的空闲时段运行，到高峰时段前自动停止、之后续判；判完运行 `diag/confirm_fresh.py`。只调用 `ff_tune_pro_20261117.sh` | 新写（Opus 5.5 子代理），时间窗口与续判逻辑用假命令演练通过，2026-11-18 已启动 | 9d3e22a776afc518 |
| `launchers/ff_xu1_stage1_20261118.sh` | 三个模型（Gemma-4-31B、Gemma-3-12B、Gemma-3-27B）的加强谨慎问法运行：用 `make_e44_source.py --append-sentence` 生成源目录，再按普通问法运行的层表调用现成的两条链（`novelty/ours/2026-11-18_stronger_caution_prediction.md`） | 新写（Opus 5.5 子代理），2026-11-18 09:37 已启动 | 3352eaab26e80767 |
| `launchers/ff_xu1_stage1_after_20261118.sh` | 三个加强谨慎问法运行的链结束后，补 DeepSeek Pro 判分（调参集的 E+C 网格与测试集的博士 100 题），只在空闲时段运行；步骤照 `ff_gemma3_27b_after_v3.sh`。链有失败步骤的运行不判；只有 `sn_*`（模型对自己回答的自信度，排在两个网格之后、与网格无关）失败时照常判，并要求两个网格的 8 个步骤都有完成标记 | 新写（Opus 5.5 子代理）；失败步骤的规则由 Fable 5.2 改为现在的写法（第 2 版）。用假目录与假命令演练通过，2026-11-18 已启动 | aa4dc86d55cbd433 |
| `launchers/ff_ec_probe.sh` | 上一脚本的启动脚本 | 新写，待审，未运行 | 2c606c74ea6f06c1 |
| `diag/policy_select2.py` | 方法与两条提示词基线的正式评估，只读已判分的回答；另有"不写入""单独使用 C"两个对照 | 已对八个运行运行；报告部分包在 main() 中，可被其他脚本导入；Gemma 各行改由 Pro 判调参集选设置，原按 AllenAI 选定的行保留为"（tune AllenAI）"；2026-11-17 新增 Gemma-3-27B 一行，其余模型输出逐字未变 | 4bbbf285d4e6f97a |
| `diag/c_first.py` | "C 先放、E 后压"三种做法与现在的方法的比较；两类留空题上放出的结果（`novelty/ours/2026-11-17_silence_and_release.md` 第 4.3、4.4 节） | 新写（Fable 5.2），文件头有事先写下的预测，已运行 | aedd0968465a4925 |
| `diag/silence_table.py` | 谁被压住：各写入组把哪些已作答的题变为不答（同上第 4.1 节） | 新写（Opus 5.5 子代理），已知数核对通过，已运行 | 65762e55cd8f1b70 |
| `diag/e_gain_split.py` | E 的增益按改答案、压住、放出三项拆分（同上第 4.2 节）；`--test` 运行单元测试 | 新写（Opus 5.5 子代理），单元测试通过，已运行 | d5e43bfb78e9f520 |
| `diag/method_split.py` | 方法相对单独使用 E 的增益按两类留空题拆分（同上第 4.3 节） | 新写（Opus 5.5 子代理），与 `c_first.py` 的独立计算一致，已运行 | b0160b4fba332105 |
| `diag/law_lam1.py` | "0.45 倍"规律的复现与按答对 − 答错的重算（同上第 4.5 节） | 新写（Opus 5.5 子代理），原规律八个点复现无误，已运行 | 4bbfac586ec14f79 |
| `diag/survey_cp.py` | 清点"只推 C 往谨慎"的判分目录 | 新写（Opus 5.5 子代理），已运行 | ce9e4ead34f44245 |
| `diag/doubt_levels.py` | "I don't know."读数在不写入与推 E 后的水平（同上第 4.7 节） | 新写（Fable 5.2），已运行 | c0f3e045d09ab531 |
| `diag/gemma4_diagnosis.py` | Gemma-4 与其他模型不一致之处的五项离线统计：各写入放出原本不答的题、E 线逐层的放出、随机方向下判分的变化、以 "thought" 开头的回答、问法普查（`novelty/ours/2026-11-17_gemma4_diagnosis.md`） | 新写（Fable 5.2），已运行 | 7f5c1653d290b424 |
| `diag/release_gate.py` | 放出的回答能否按"是否稳定"筛选（`novelty/ours/2026-11-17_silence_and_release.md` 第 4.8 节） | 新写（Opus 5.5 子代理），已知数核对通过，已运行 | 069d0ddba6a5c390 |
| `diag/two_caps.py` | "两个上限"方案与现在的方法、单独使用 E、单独使用 C 的比较，含第二个裁判的辅助核对（同上第 4.9 节） | 新写（Fable 5.2），文件头有事先写下的预测，已运行 | 79cc5fc8db7287a1 |
| `diag/two_caps_vs_prompt.py` | "两个上限"与两条提示词基线的比较（同上第 4.9 节第 7 点） | 新写（Fable 5.2），已运行 | 72f520e79fe2d543 |
| `diag/release_origin.py` | 放出的回答按所需力度与"谁的答案"拆分，方案 N1，力度曲线（`novelty/ours/2026-11-17_silence_and_release.md` 第 4.10、4.11 节） | 新写（Fable 5.2），文件头有事先写下的预测，已运行 | c776978a6a6886a6 |
| `diag/n1_vs_prompt.py` | 方案 N1 与现在的方法、单独使用 C、两条提示词基线的比较（同上） | 新写（Fable 5.2），已运行 | 14786f62d406f9fc |
| `diag/n1_aux_allenai.py` | 方案 N1 用第二个裁判在测试集 523 题上的辅助核对（`novelty/ours/2026-11-17_silence_and_release.md` 第 8 节） | 新写（Fable 5.2），已运行 | 7b47a8dfaf881279 |
| `diag/confirm_fresh.py` | 方案在未参与设计的测试题上的确认（同上第 8 节）；`--judge allenai` 为程序试跑 | 新写（Fable 5.2），文件头有事先写下的预测，已用 AllenAI 判分试跑，Pro 判分完成后运行 | 6d7e61c9ce92daa2 |
| `diag/prompt_compare.py` | 加强谨慎问法的运行与普通问法的运行在相同题目上的配对比较；必须作答问法与普通问法的比较（`novelty/ours/2026-11-18_stronger_caution_prediction.md`） | 新写（Fable 5.2）；Opus 5.5 补入三个加强谨慎运行与对、错、含糊、不答的计数并运行（输出 `prompt_compare_20261118c.txt`；修改前依次为 4f5a68732dc05464、3f3d791e27553660） | 1b1b840b8969d511 |
| `diag/provenance_ablation.py` | 打乱来历的对照、否决原答案单独的作用、原本不答的题上的交互项、文字比较写法的敏感性（`novelty/ours/2026-11-17_silence_and_release.md` 第 4.12 节） | 新写（Fable 5.2），文件头有事先写下的预测，已运行 | 633e2856ee892465 |
| `diag/final_records.py` | 每道题最终交出的是哪条回答（条件名、原文、判分、来历、放出的档、否决与否），计分与打印个例共用；`--selftest` 用合成题检验否决、超上限、只在一方留空三种情况，`--check` 在五个模型的真实运行上与此前各计分程序逐题核对 | 新写（Opus 5.5），只读；自测 18 项与 133 种设置通过，真实运行 12 组划分全部一致（输出 `final_records_check_20261119.txt`） | ab1d0e41c99e76ac |
| `diag/b_vs_n1.py` | N1 与提示词基线 B 的逐题拆分：按两方各自的处理分组，各组之和等于总差（`novelty/ours/2026-11-17_silence_and_release.md` 第 8 节第 7 点） | 新写（Opus 5.5）；2026-11-19 改为读 `final_records.py` 的记录，改正个例打印（GPT 复核），合计与旧版逐项相同（输出 `b_vs_n1_20261119.txt`；修改前为 323f1c45165770fe） | 499d34370d1ead6b |
| `diag/main_table_same_n.py` | 相同作答数下的主表：N1 对"必须作答问法上只推 E、原模型、只推 C，各留最有把握的同样多条"，以及各系统在各自作答数上的净分、把握门槛由调参集选定的版本、博士 11-15 点名的一组、五个模型与正文三个模型的合计（`novelty/ours/2026-11-19_same_answer_count_main_table.md`） | 新写（Fable 5.2），只读，已运行；Opus 5.5 改为读 `final_records.py` 的记录（原有各行逐项相同），补博士点名的一组与三个模型的合计（输出 `main_table_same_n_20261119.txt`；修改前为 79c65e75f0b70849） | d1aa4ced836c6f20 |
| `diag/examples_for_dr.py` | 给博士的例子：按事先定死的取法（加 C 后放出的题按题号取前 3 道答对、前 3 道答错，再取前 2 道否决的题），列出 N1 实际交出的回答；`--md` 输出表格行（`doctor/2026-11-19_tables_for_dr.md` 第 4 节） | 新写（Opus 5.5），只读，已运行（输出 `examples_for_dr_20261119.txt`、`examples_for_dr_20261119_md.txt`） | 2b7d15357564e944 |
| `diag/gemma4_random_dose.py` | Gemma-4-12B 与 31B 的配置差别；随机方向写入量与回答变化的关系（`novelty/ours/2026-11-17_gemma4_diagnosis.md` 第 3.7 节） | 新写（Fable 5.2），已运行 | c8ce0a08b5e86985 |
| `diag/g27_prediction_check.py` | Gemma-3-27B 起跑前预测的核对（`novelty/ours/2026-11-16_gemma3_27b_prediction.md` 第 4 节） | 新写（Fable 5.2），已运行 | 710a046ae2f46560 |
| `promptladder/ladder_extra.py` | 提示词分档对比的辅助程序：原样调用 `scripts/e44_eval_extra.py`，在内存中改三处（新提示逐字读入、输出目录、导出目录），使同一个 E 写入换用新的系统提示（`novelty/ours/2026-11-18_prompt_ladder_vs_c_ladder.md`） | 新写（Opus 5.5 子代理）；按 GPT 第二轮意见（B1、S2、S3）修改，CPU 自测通过，未上 GPU，待 GPT 复核（`discussion/prompt_ladder/2026-11-18_fix_report.md`）；修改前为 acb2ef6ee5817b28 | 23b0eea22f4f1cba |
| `promptladder/ladder_run.sh` | 提示词分档对比的启动程序：空跑检查、8 题试跑、正式生成、写出与核对候选题、判分 | 同上（B1、B2、S1、S2、S3）；修改前为 c8313e0a89c404ce | 21e92e0a4488e83e |
| `promptladder/ladder_prompts_draft_20261118.json` | P1 至 P3 的文本 | 未改 | 6633a8d615fc91f2 |
| `promptladder/selftest_cpu.sh` | 只用 CPU 的自测，含 GPT 列出的反例 | 84 项全过；修改前为 fa04208a9ba39c2c | 24ad9a71bd4c9562 |
| `promptladder/selftest_fns.sh` | 启动程序中记账函数的单元测试 | 36 项全过；修改前为 46793e7357ac5101 | 34428fbc55f5d244 |
| `promptladder/ops/smoke_r2_chain.sh` | 按 GPT 复核放行的一步：五个模型依次做一次 8 题试跑（新目录 `promptladder/smoke_r2_20261118/`，只用 1 号卡）；只调用上面的启动程序 | 新写（Opus 5.5），2026-11-18 23:33 起跑；辅助程序随后要改，这些目录只用来核对设备布局 | eb721d9f313ad58f |
| `promptladder/ops/smoke_r2_check.py` | 试跑后的逐题核对：对照与三个新问法各 8 行、题号与顺序和父运行的前 8 行一致、对照文字与父运行相同（只读） | 新写（Opus 5.5） | 0029eaf0c25ef13e |
| `scripts/e44_eval_extra.py` | 评估驱动，探测脚本复用其 Context、写入与读取状态的部分 | 现行代码，供对照 | 4438a282d3e0c12c |
| `scripts/e44_factorial_das.py` | 方向训练与 gap 的定义 | 现行代码，供对照 | 9ef1759fe492337b |
| `scripts/review_checks/ff_explore_ec_confidence.py` | 置信度脚本，探测脚本沿用其 Context 参数 | 现行代码，供对照 | 231052bd8c8a6b64 |
| `core/qa_factorial_das.py` | 写入钩子 ShiftHook 与位置掩码 | 现行代码，供对照 | 7adfd9c0deef9e99 |
| `core/qa_prompt_das.py` | 挂钩子的 clamp、提示词渲染与打包 | 现行代码，供对照 | c63ac689be2e510b |
| `core/model_io.py` | 前向计算 forward_logits | 现行代码，供对照 | 69ebe708de33d8d2 |
| `scripts/review_checks/newmodel_e44.sh` | 单模型运行脚本；重学 C 的改动草稿见 `discussion/relearn_c/` | 现行代码，供对照 | bcb375feb7044142 |
| `scripts/review_checks/ff_test_eval.sh` | 测试集运行脚本 | 现行代码，供对照 | 92bdfb4996306ff1 |
| `scripts/review_checks/e44_split_test.py` | 允许评估程序在测试集上运行的放宽 | 现行代码，供对照 | ac5ab93f17ae1f90 |
| `scripts/review_checks/ff_test_subset_judge.sh` | 只判部分题目的判分程序 | 现行代码，供对照 | 6199d697ba94e4b6 |
| `scripts/review_checks/ff_explore_ec.sh` | E+C 网格运行脚本 | 现行代码，供对照 | a8803d6102e01e4b |
| `scripts/review_checks/ff_explore_ec_select.sh` | E+C 选择步骤运行脚本 | 现行代码，供对照 | 33aa833414317e88 |
| `configs/prompts/e44_factorial_v1.json` | 两因子设计的问法 | 现行代码，供对照 | c39172b39b550fa8 |
| `launchers/ff_ec_lines_chain.sh` | E 线与 C 线的链 | 现行代码，供对照 | 6f17d94d661fa62a |
| `launchers/ff_ec_test_chain_v3.sh` | 测试集 E+C 的链（规则 v3） | 现行代码，供对照 | d02654b8c702134e |
| `launchers/e45_newmodels_20261119.sh` | 两个新模型的真假判断、允许答 Unknown 时的 E+C（方案 `claude_gpt_review/from_claude/40` 原样）：从 `code_newmodels_20261112` 运行，方法程序与 11-12 逐字节相同（启动时核对），E、C 取新模型 2×2 的选择；`LIMIT=8` 为试跑（`novelty/ours/2026-11-19_tf_unknown_ec_new_models.md`） | 新写（Opus 5.5），只是启动参数；两个模型试跑通过，四次正式运行 2026-11-19 02:45 完成，结果表在 `Iteration_01/results/tf/` | 6efcf06450cb9098 |
| `scripts/newsets/newsets_splits.py` | TriviaQA 与 PopQA 的固定种子切分：测试 500、调参 192、训练 300，测试中再抽 Pro 判的 100 题；只输出题号。PopQA 去掉同名不同实体的歧义题、合并同一实体的多值题（`novelty/ours/2026-11-19_new_datasets_pilot_design.md` 第 3 节） | 新写（Opus 5.5），自测与独立核对通过，已运行；加 `extend`：从没用过的题里另抽一份新题（test2）（修改前为 94378c3e14d149dd） | c28c6c9e8b634bcf |
| `scripts/newsets/newsets_bank.py` | 由原始文件与题号清单生成中间题库（含题目原文，只存服务器 `Iteration_01/newsets/bank/`） | 新写，自测通过，已运行；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 c90fbc57429314b8）；可读 `extend` 产出的新题清单（修改前为 167d7dc28bd83a3f） | e71da39da8be35a6 |
| `scripts/newsets/alias_match.py` | 别名判分（辅助裁判），用 TriviaQA 官方的答案规范化 | 新写，自测通过；与数据自带的规范化别名在 9,960 题上逐条相同 | 803814f798bf49fa |
| `scripts/newsets/newsets_gen.py` | 用冻结的 TruthfulQA 方向在新题上生成：repro（复现检查）、gen（生成）、conf（把握分，调用现有计分函数）、export（判分输入）。核心与引擎都不改 | 新写，待 GPT 审；复现检查五个模型都 8 / 8 相同，新题各 8 题试跑正常；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 0fb24e28b5dffa02）；conf 加 `--score-batch 1` 补救（原把握分程序的规则，修改前为 0b6540ac4c29ef72）；划分名加 test2（修改前为 5a9aeb83aeec92f9） | ebc8c8461fcb89eb |
| `scripts/capability/cap_sets.py` | 能力保持的题号清单：MMLU 按学科分层 1,000 题，GSM8K 1,319 题（`novelty/ours/2026-11-19_capability_preservation_design.md`） | 新写，自测通过，已运行 | 0d427697b0c022f0 |
| `scripts/capability/cap_prompts.py` | MMLU、GSM8K 的问法与规则判分 | 新写，待 GPT 审；自测通过，8 题试跑取得到答案；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 006b371dd7b6f749） | b258e649a96f8a49 |
| `configs/newsets/frozen_models_20261119.json` | 五个模型冻结的 N1 设置：代码副本、解释器、运行目录、层、上限、单 C 的力度 | 新写，待 GPT 审 | c7306fe165f6739a |
| `configs/newsets/triviaqa_splits_20261119.json`、`popqa_splits_20261119.json` | 新题集切分的题号清单（服务器上在 `Iteration_01/newsets/splits/`） | 已生成 | 3fa2af2a0fb95e8b、992f2e1181eb50e2 |
| `configs/capability/mmlu_1000_20261119.json`、`gsm8k_1319_20261119.json` | 能力保持的题号清单（服务器上在 `Iteration_01/capability/sets/`） | 已生成 | 13fcbb997dcbb383、f8158017f1fcbdde |
| `launchers/newsets_c2c3_20261119.sh` | C2、C3 的启动脚本：生成、把握分、导出判分输入、DeepSeek Pro 判分（离高峰不足 40 分钟就不开始） | 新写，待 GPT 审；其中复现一步已跑通；补了 D2 的 gap 一步（修改前为 19c3b47c871e19e3）；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 191ebdfd2d6f006f）；把握分先每批 8、未过再每批 1；测试段去掉单 C 的把握分（修改前为 5146358125e0b3c7）；加 val、val_judge 两段（新题验证）（修改前为 de3cc887e46726f1） | 7ab6572bf162d2ed |
| `launchers/capability_f1_20261119.sh` | F1 的启动脚本：三个整套条件，再对推 E 后留空的题做 N1 的逐档再问 | 新写，待 GPT 审，未运行；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 41471c6dd6eeef7a） | baabd580b4041eca |
| `scripts/newsets/newsets_gap.py` | D2 最小版的差值读出：E 所在层、提示最后一个词元处的状态投影到 E 的写入方向，区分对错的 AUROC 减把握分的 AUROC（调参集，Pro 标签）；只读状态，不写入 | 新写，待 GPT 审；自测通过；Gemma-3-12B 用 8 题临时标签跑通了读状态这一步（数字不作数）；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 62e92df7602bd735） | c23bfd3d0da3ef30 |
| `scripts/newsets/newsets_votes.py` | Pro 判票的核验：判分输入未被改动、一格一票、票绑定当前回答（input_sha 与生成文字），拒判、空答、参考答案只剩拒答的题分开计数；计分、D2 与启动脚本共用 | 新写（GPT 审稿后），自测通过（两票、错配、漏票、未判完、判分输入被改都报错） | d5d63ad36e6a734c |
| `diag/newsets_score.py` | 新题集的计分：C2（五种问法、第 0 步、别名与 Pro 一致率、D2 差值）、C3（各系统、强形式、只重选上限的一行、五模型合计与闸口 1 的读法）。N1 等各系统直接调用 `final_records.py`，不改 | 新写，待 GPT 审；合成自测通过（否决、超上限、不是候选、放出）；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 9f43436d104312de） | 54c252130b5fcfa1 |
| `diag/cap_score.py` | 能力保持的规则计分：各条件的准确率、不答率、与不写入的差和区间；N1 直接调用 `final_records.py`（GSM8K 的条件名映射到 x） | 新写，待 GPT 审；合成自测通过；按 GPT 审稿（`discussion/validation_dataset/2026-11-19_gpt_review_pilot_capability.md`）修改，自测含其反例（修改前为 df3bc84903463d7f） | b774cee195c6beec |
| `diag/d2_register.py` | D2 预测登记：读十格差值与调参集收益，按事先的规则写出预测表与 `newsets/gap/d2_predictions.json`；已有任何测试生成或测试判分输入就拒绝运行 | 新写，待运行（调参判分之后） | 64d0a9a996aff923 |
| `diag/newsets_val.py` | 新题验证的计分：不写入、只推 E、重选上限的 N1（Pro 100 题与别名 500 题），五模型合计（`novelty/ours/2026-11-20_rechosen_caps_validation.md`） | 新写，未经 GPT 审 | 9f49f62b0436543e |
| `diag/newsets_tqa_shared_answers.py` | TriviaQA 两批测试题上只推 E 的差别，按答案是否与别的题共用拆开（新题排除共用答案是否改变了题的构成）；容器内运行 | 新写，未经 GPT 审 | ce5c4a083e178f43 |
| `configs/newsets/rechosen_caps_20261120.json` | 各题集调参集上重选、冻结的 N1 上限 | 新写 | 097709b6ee358227 |
| `configs/newsets/triviaqa_test2_20261120.json`、`popqa_test2_20261120.json` | 新题（test2）的题号清单（服务器上在 `Iteration_01/newsets/splits/`） | 已生成 | b51a9ffa4dfce1e9、18ea733ad9bc0a37 |
| `scripts/capability/cap_mcq_prob.py` | MMLU 的选项概率测法（ITI、LoFiT 所用的方式：比四个字母的概率，不能说不知道；每个字母的两种单词元写法相加）；同题、同问法、同写入，只换判分 | 新写（赵磊 2026-11-19 19:5x 要求），未经 GPT 审；8 题试跑后运行 | 514fdc7c4c9c1697 |
| `scripts/capability/cap_mcq_sweep.py` | MMLU 选项概率测法下换写入方式的对照：力度、反向、同样大小的随机方向、均值差方向、学到的方向拆成两部分、只从提问末尾写；一次前向读出四个字母，与 `cap_mcq_prob.py` 的结果逐题相同 | 新写（赵磊 2026-11-20 要求找原因），未经 GPT 审；16 题试跑后运行；同日加改：写入可以叠在该次运行的其他提示词上（例如必须作答加如实句），不写入的提示词本身也可单列一臂（修改前为 001bf8d5725f5818） | ca1eab3469d4ab6c |
| `diag/cap_mcq_flips.py` | 逐题看推 E 把 MMLU 的选项改成了什么（字母偏向、原选择对错、把握高低、否定选项、科目） | 新写，未经 GPT 审 | 62370da7eb44d5d7 |
| `diag/dose_table.py` | TruthfulQA 测试集上 E 力度 1 与力度 2 的主要系统并排（只读已有回答与判分，调用 `main_table_same_n.py`） | 新写，未经 GPT 审；同日补入与提示词基线的比较（留空再问 B、必须作答加如实句不写入）及正文三个模型的合计（修改前为 829257ae39f6c876） | d1e33c892a41fc8f |
| `diag/cap_mcq_fresh.py` | MMLU 全测试集的计分：没看过的 13,042 题、看过的 1,000 题、全部，各按 1 个百分点判据给标签 | 新写，未经 GPT 审；2026-11-20 晚加改：只有提示词、不写入的一臂也能列出；写在别的提示词上的臂另报对该提示词本身的变化（修改前为 fa36034a0e569c6e） | 72ae112cd10bece0 |
| `diag/dose_mid.py` | TruthfulQA 测试集上各模型在自己有的每一档 E 力度（1、1.25、1.5、1.75、2，个别模型另有 2.5 至 4）下的只推 E 与 N1：对不写入、对只推 E、对纯提示词（必须作答加如实句、不写入）；每档的两个上限都在调参集上选；末尾按调参集给每个模型选一个力度（只读已有回答与判分） | 新写（赵磊 2026-11-20：分模型列出、中间力度也试、各模型可用各自力度），未经 GPT 审；当晚加改：保留给 GPT 复算工具用的旧函数名（修改前为 bdec110dc0d82678） | da7f169c48bfc6db |
| `diag/on_prompt_table.py` | 把 E 写在“必须作答加如实句”这条提示词之上，对比该提示词本身（TruthfulQA，DeepSeek Pro，分模型与五模型合计；只读已有回答与判分） | 新写，未经 GPT 审 | 4d1367ce018287a6 |
| `diag/md_table.py` | 学出的 E 方向对均值差方向（同一层；同写入量与均值差自身的量两种），TruthfulQA 测试集，力度 1 与 2，分模型与五模型合计（只读已有回答与判分）；2026-11-21 08:3x：多模型合计改为只在每个模型都有判决的题上算（与主表一致），分模型各行不变 | 新写，未经 GPT 审；当晚加第二部分：均值差在调参集上自己挑层与系数后的对照（修改前为 9b706dc5ffe7d320） | f2ac0aaf1a2e2f08 |
| `diag/md_choose.py` | 给均值差基线在调参集上选它自己最好的层与系数（读 `md_sweep.sh` 的调参集判分，写出选择记录） | 新写，未经 GPT 审 | 43b30cacd011e8b1 |
| `diag/strong_mid.py` | N1 在每一档力度下，对同一力度的三个对照：必须作答加 E 按把握留同样多条（强对照）、留空再问 B、必须作答加 E 全答；力度 1、2 用旧文件（须与 `dose_table.py` 一致），其余力度用 `pm2_all.sh` 的回答与置信度文件 | 新写，未经 GPT 审；Qwen3.8-27B 力度 2 复现 +38 | eb05bbdea9c4a24e |
| `diag/prompt_sweep_table.py` | 把每个模型冻结回答里的 11 条提示都当作纯提示词基线（无写入）：调参集与测试集的净分；每个模型在调参集上选最好的一条，再与各档力度的 N1 在测试集上比较（只读判分） | 新写，未经 GPT 审 | 271550ea2d70f706 |
| `diag/pos_table.py` | 只从提问最后一个词元起写 E（`e44_positions.py` 的 gen 规则）在 TruthfulQA 测试集上的净分，对不写入、对纯提示词、对同力度的原写法（只读判分） | 新写，未经 GPT 审 | 863f0edb466332a3 |
| `diag/tf_net_score.py` | 真假判断（同一遍可答 Unknown 的 E45 运行）按自由回答的记分法重算：答对 +1、Unknown 0、答错 −1；529 对测试题，按题对重抽（只读已有的逐条概率） | 新写（赵磊 2026-11-20：真假判断与自由回答一致），未经 GPT 审；每个运行的计数须复现其已存汇总 | 6e4a77fdce43cd2d |
| `diag/cap_net_score.py` | 第一次 MMLU 测量（模型自己写答案、可以不答）按答对 +1、不答 0、答错 −1 重算，并列出推 E 把原本对 / 错的题改去了哪里（调用 `cap_score.py` 的读入与核对） | 新写，未经 GPT 审 | 54c7b5aebcc5649e |
| `diag/per_model_table.py` | 每个模型用自己力度的主表：MMLU 14,042 题对半分，“选力度”的一半上取掉分不超过预算的最大力度，另一半报 MMLU；该力度下 TruthfulQA 的原模型、只推 C、只推 E、N1、同力度强对照、纯提示词与各项差（只读已有回答与判分） | 新写（赵磊 2026-11-21：表按各模型各自力度摆），未经 GPT 审 | bd6d9a06eca1ef2d |
| `diag/ns_dose_table.py` | TriviaQA / PopQA 上各档 E 力度的只推 E、N1（上限在该题集调参集上选）与不带否决的诊断版；按调参集选力度后看测试集；别名匹配判分，不调用裁判 | 新写（赵磊 2026-11-21：新题集也按各自力度），未经 GPT 审；力度 2 的现有数据上试跑通过 | be103265544d17b4 |
| `diag/keep_table.py` | 训练题加入 MMLU 题后学出的 E，与现用 E 在 TruthfulQA 上逐档比较（只读判分） | 新写，未经 GPT 审 | d0a493f2836c7d84 |
| `diag/oncaut_table.py` | 把 N1 整套叠在“加强谨慎 + 如实句”这条提示上的计分（条件名里的提示名读作 x，记录函数不改）：提示本身、其上只推 E、其上的 N1 | 新写，未经 GPT 审 | 1b6bdcc8d443eba5 |
| `diag/ns_caut_table.py` | TriviaQA / PopQA 上让 C 的力度取谨慎方向：只推 C（谨慎）、E 加 C（谨慎）每题都写；各自在调参集上选力度，再比测试集 | 新写，未经 GPT 审 | 6af51448c1284165 |
| `diag/l16_table.py` | Llama 把 E 写在第 16 层（不是第 11 层）时的计分：各档力度的只推 E、N1（上限在调参集上选）与 MMLU 全量变化并排，附第 11 层各档作参照 | 新写，未经 GPT 审 | a6f727cdd26c2bd3 |
| `diag/main_table_layers.py` | 主表的换层版：每个模型的 E 层可以不止一个候选（登记在 `alt_layer.py`：Llama 第 11、16 层，Gemma-3-12B 第 22、24 层，Gemma-3-27B 第 26、28 层）。两步规则：每层取 MMLU“选力度”一半上掉分不超过预算的最大力度，再按该力度下 N1 的调参集净分选层；只有一层的模型与 `per_model_table.py` 逐行相同；2026-11-21 07:4x 加一行对照：每个模型在调参集上净分最高的无写入提示（沿用 `prompt_sweep_table.py` 的选法），原有各行逐行核对未变；08:0x 加一个取数接口（把选中设置下 N1 的逐题记录交给 `prompt_conf_table.py`），输出与加之前逐字相同 | 新写，未经 GPT 审 | 2a022e567e819c16 |
| `diag/alt_layer.py` | “换一层写 E”的通用计分（`l16_table.py` 的一般形式）：登记了 Llama 第 16 层与 Gemma-3-27B 第 28 层；各档力度的只推 E、N1、同层强对照、同层均值差对照与 MMLU 全量并排；`main_table_layers.py` 从这里取候选层 | 新写，未经 GPT 审 | 0e23a0b83badc780 |
| `diag/gsm8k_main_table.py` | GSM8K（1,319 题）在主表设置下的规则计分：不写入一遍、各模型在主表所用的层与力度上只推 E 一遍；准确率与不答率的变化带区间，判词沿用能力保持设计里的规则；不答按原因分列（拒答、到 512 词元上限被截断等），并另报“两遍都作答的题”上的准确率变化；`--cap N` 读另一种生成上限的两遍（校验文件里的上限与每批题数）；换了写入位置的一遍显示为“臂名@位置”；计分函数直接用 `cap_score.py` 的 | 新写，未经 GPT 审；2026-11-22 修一处：跳过再问留下的题号文件（原先当成回答文件读、报错退出），各模型数字不变 | a1cac4b73291d86a |
| `scripts/review_checks/ff_prompt_confidence.py` | 各条提示的无写入回答的把握（回答自身词元的平均对数概率）。把握计算程序 `ff_explore_ec_confidence.py` 原样运行，只在内存里把要算的回答集换成“某条提示的冻结回答、不写入”，并让模型只加载一次；每条提示一个文件；98% 的复现检查不放宽，没过的按原程序写明的办法改每批 1 题再算一次 | 新写，未经 GPT 审；Llama、Qwen3.5-9B、Gemma-3-12B 的 11 条提示在调参集与测试集上全部通过 | 32e22e7180bea944 |
| `scripts/newsets/ns_positions.py` | 探索：让 `newsets_gen.py` 把 E 写在另一种位置（核心里已有的 prompt = 只写题目与助手开头、生成的词都不写；gen = 从最后一个提示词元起写）。`newsets_gen.py` 原样运行，内存里三处改动（引擎的写入位置、`plan_arm` 的位置检查、输出文件名加“-at-位置”）；只许单个写入；23 时加两个环境开关：`NS_POS_TWO_WRITE=1` 放开两个写入（事实题的 N1 再问要用），`NS_POS_PLAIN_NAMES=1` 在专用目录 `gen_<位置>` 里用原文件名，让现成的表格程序能读 | 新写，未经 GPT 审；Llama 两批 GSM8K 试跑核对过：文件里记的位置是 prompt、回答与不写入不同 | 85d5ada9f9651440 |
| `scripts/review_checks/e44_positions2.py` | 探索：`e44_positions.py`（不改，导入）放开到两个写入的臂，用于在“只写题目位置”下生成 N1 的再问各臂；两个写入的验收记录另存为 `selfcheck_*-at-<位置>.json`，不覆盖训练时位置的记录 | 新写，未经 GPT 审 | fddbf7073d29ebf2 |
| `diag/posp_table.py` | 探索：TruthfulQA 上只写题目位置的只推 E 对原写法（同层），力度在调参集上选（规则先写在 `posp.sh` 文件头） | 新写，未经 GPT 审 | b74bfa57748dc51b |
| `diag/posp_n1_table.py` | 探索：TruthfulQA 上 N1 的两个写入都只写题目位置，对原写法的 N1（同层同力度 = 主表预算 1 点的设置）；两种写法各自在调参集上选上限；原写法这一行必须等于主表 | 新写，未经 GPT 审 | 300afdadd819fee7 |
| `diag/main_table_prompt.py` | 探索：主表的“只写题目”版（所有写入都用核心规则 prompt）。主表规则不变：每层取 MMLU 预算内的最大力度（两种写法 MMLU 相同），再按 N1 的调参集净分选层；只推 C 的力度与 N1 的上限都在调参集上选；同时打印原写法的 N1 作对照；两种写法缺判决的题不同时，都在共同的题上读，并打印题数。含只写题目下的强对照（必须作答 + 只写题目的 E，按把握留同样多题） | 新写，未经 GPT 审 | bebd64ab39d96ff6 |
| `diag/posp_check.py` | 探索：主表之外的某一组（模型、层、力度、标签）在只写题目下的 N1 对只推 E、不写入、纯提示词、原写法 N1（例如 Llama 第 16 层力度 4，标签 l16s4） | 新写，未经 GPT 审 | ce9e111d8d57688f |
| `diag/ns_prompt_table.py` | 探索：TriviaQA、PopQA 在只写题目、E 用主表层与力度下的表。两种读法先写在 `ns_prompt.sh` 文件头：F = 力度与上限全部沿用 TruthfulQA 的（不在新题集上选任何东西）；D = 在各题集自己的调参集上选力度与上限。另数再问放出的回答是原答案还是新答案、各对多少 | 新写；2026-11-22 应用 GPT 的补丁（用生成器自己的读取函数核合同，另核臂名），并把各题集自选的读法改成先在调参集齐全的力度里选、缺文件标待完成；改前改后表逐字相同 | ad287a3dae18b9cd |
| `diag/gsm8k_n1_prompt.py` | 探索：GSM8K 上完整 N1（只写题目）的实际读数。N1 只改动推 E 后规则判为留空的题；这些题用 E + C 逐档再问（`launchers/explore_20261120/gsm8k_n1_pos.sh`），上限用主表的；`ids` 模式列出要再问的题，`table` 模式出表 | 新写；2026-11-22 应用 GPT 的补丁（逐份回答文件严格核合同：模型、臂、位置、题号、方向文件、词元上限），改前改后三个模型数字相同；同日 05 时加两行平均（正文三个模型、五个模型，只推 E 与 N1 各一个），各模型的行不变 | 0eca5cb83f42de10 |
| `diag/posm_table.py` | 探索：只写题目下，学出的 E 对同层同写入量的均值差方向（TruthfulQA 测试集，Pro 判）；读 `launchers/explore_20261120/posm.sh` 的输出，N1 取 `main_table_prompt.py` 的同一套 | 新写；2026-11-22 按 GPT 的复审补核：两份生成文件同层、同力度、同写入量、同一次运行、同一批题，不写入的格缺失算错误；Llama 的判分未出，尚无读数 | 75dc93dabe8ae8d8 |
| `diag/nsc_prompt_table.py` | 探索：TriviaQA、PopQA 在只写题目下，同两个方向的另几种用法——A（E 与往谨慎的 C 一遍写入）、B（不写入与推 E 两遍一致才保留原回答）、B2、G（已答的题用 E + 往谨慎的 C 再问一遍，答案不变才保留）、U（全部候选里取调参集最好的，含两个单方向）。每个设置只在调参集上选；`--choose 模型` 给启动脚本列出要补到 test / test2 的臂（`discussion/work_queue/2026-11-22_queue.md` 第 2 节） | 新写，未经 GPT 审；两个模型的小试验已跑（`results/newsets/prompt_rule/`）；2026-11-22 子代理只读审过：没有会改变数字的问题，按它的意见补了两项文件核对（臂名、写入位置），并把只推 C 往作答也放进 U 的候选，改后小试验的表逐字相同 | 41560bf09815ee82 |
| `scripts/newsets/ns_task.py` | 探索：让 `newsets_gen.py`（文件不动）把一个新的题集当作 triviaqa / popqa 那样的题库任务来跑（内存里重编 `load_items` 与 `main` 里写死任务名的四处）；带写入位置时再交给 `ns_positions.py`（文件不动） | 新写，未经 GPT 审；(QA)² 小试验用它生成与导出 | f907ce4bdcbeb310 |
| `scripts/newsets/qaqa_bank.py` | 探索：从 (QA)² 评估集（570 题，下载数据，用 `python -I` 读）按固定种子、按题型比例抽 100 题做成题库（题目原文只留服务器），并与 TruthfulQA 的 817 题核重合 | 新写，未经 GPT 审；已运行：50 / 40 / 10，重合 0 | 86418287a5447dae |
| `diag/qaqa_pilot_table.py` | 探索：(QA)² 的表（只写题目，Pro 判；原模型、纯提示词、只推 C、只推 E、N1，按题型分开；容器里运行）。三种用法：默认 = 全部设置沿用 TruthfulQA（100 题小试验或 570 题整套，可多个模型合计）；`--tune` = 把 100 题当调参集，列出每档 E 力度、每档 C、每对上限的净分并选定（`--write-choice`）；`--choice` = 用调参集选定的设置读其余题（`--skip-ids`），`--list-arms` 列出整套题上还缺的臂；`--tune --fix-e` = E 用主表力度、只选上限；有把握分文件时自动加一行“纯提示词按把握留同样多题”的对照；有强对照的把握分文件时再加“必须作答 + E、按把握留下与 N1 同样多”和“必须作答、不写入、留同样多”两行（要在 `--work` 里给出它们的判分目录，否则只在标准错误里提示并略过） | 新写，未经 GPT 审；2026-11-22 子代理只读审过（数字可信），按其意见加了三道硬检查，改后四张表逐字相同；05 时加强对照两行，加之前先核过原来四张表逐字不变；2026-11-24 加可选开关 `DECLINED_OK`（默认关，行为不变；`falseqa_read.py` 打开它，把裁判拒判的格子记 0 分并记入清单），改前为 6d4939a768c0fd92 | d0decce9ffd8ae42 |
| `scripts/newsets/ns_conf_pos.py` | 探索：让 `newsets_gen.py` 的把握打分（文件不动）用于“写入只加在题目上”生成的回答。内存里三处：引擎报告的写入规则换成 prompt、`run_conf` 里要求 content 的那一处判断换成 prompt（必须恰好匹配一处）、打分时取写入掩码的函数在被要求 content 时返回 prompt 的掩码；回答文件必须记着 prompt 规则，输出文件名必须以 `-at-prompt.json` 结尾。98% 复现检查与“首个不同词元”检查照旧。新任务名按 `ns_task.py` 的办法登记 | 新写，未经 GPT 审；Llama 上跑通：复现 0.989，481 道不同的回答全部朝实际回答方向 | 087b2c59509aaf88 |
| `scripts/newsets/iti_baseline.py` | 探索：外部对照 ITI（Li 等 NeurIPS 2023）放进本项目自己的评测流程（同一个模型加载与对话渲染、同样的 48 词元贪心生成、同一份裁判输入）。五个阶段：`fit`（每个注意力头的探针、均值差方向与两种 σ）、`check`（自检：不加干预逐词元复现原回答；α = 0 不变；加在头上与加在注意力块输出上两种写法的首词元概率一致）、`gen`（调参集或测试集的回答）、`mmlu`（MMLU 选项概率）、`export`（裁判输入，题目与参考答案照抄同一模型同一批题的现成裁判输入）。做法逐项对照 `novelty/prior_work/2026-11-22_iti_method_spec.md` | 新写，未经 GPT 审；Llama 上自检通过（16 / 16 逐词元复现）；2026-11-22 加 `gsm8k` 阶段（不加干预时 32 / 32 逐词元复现平时的那一遍）；10 时半按只读审核加 `fit --faithful`（探针文本的答案末尾补句号、σ 那份每题抽一次另一题、头按官方 numpy 顺序；记在拟合文件里，旧拟合照旧）与不覆盖的原子写入；新版用旧拟合生成的回答与旧版逐词元相同 | 1e6c53d6eff1e8e3（2026-11-22 15:15 装到服务器的 `scripts/newsets/` 与 `scripts/newsets_stage/`；Llama 与 Gemma-3-12B 的忠实版文件都出自此前的 7f8620f0a595adeb，Gemma-3-27B 的全部出自这一版） |
| `diag/iti_table.py` | 探索：ITI 对照的三张表。`budget` = 各格点在 MMLU 全量上的变化（选择半 / 报告半，与主表同一条预算线；先核不加干预的一遍与平时的不写入文件逐题一致）；`tune` = 预算内各格点在调参集上的净分并选定（`--write-choice`）；`test` = 选定格点在测试集上对原模型、只推 E、N1 的成对差；`point` 供启动脚本提前停用；`--all` = 不看预算、整个网格里取调参集最好的格点（看过头三个读数后加的读法）；`stems` / `unjudged` 供无人值守的启动脚本列出预算内的格点与还没判的调参集回答 | 新写，未经 GPT 审；2026-11-22 09 时 `point` 改为只读两个文件（几秒而不是几分钟），在 Llama 的五个格点上与改前逐数相同；10 时按只读审核加三道检查（两个判分目录判分不一致就报错；`--ks / --alphas` 声明网格、缺格点就报错；不看预算的读法选到主读数同一格点时读主读数的判分目录），改后 Llama 的五张表逐字相同；11 时半应用 GPT 的补丁（只在声明的网格里选）；14 时：只因裁判拒判（Pro 在某一维返回 D）而缺判决的格点不再报“还没判”，改为单列（已判回答的净分、拒判条数、净分的可能范围）、不参与选择，并且只有当它把拒判的回答全算答对也仍低于选中格点时才出表，否则照旧报错；此前五张调参表逐字相同，这一处未经 GPT 审 | 420a10c8c7a1e3e4 |
| `diag/iti_period_check.py` | 探索：只读审核要的检查——答案末尾补句号重新拟合 ITI（`iti_baseline.py` 不动，只在内存里换掉文本函数），与现有拟合比选出的头、方向余弦与 σ | 新写，未经 GPT 审；结果：Llama 前 48 个头只有 18 个相同，Gemma-3-12B 24 个 | d0b6c992c71d49a0 |
| `diag/tq_same_n_decomp.py` | 探索：把 GPT 在 (QA)² 上做的拆分（`discussion/validation_dataset/2026-11-22_qaqa_replay_b.py` 的 `replay`，以及 `recovery_pool_b.py` 的“单 E 补同样多的空答”）原样搬到 TruthfulQA 测试集、主表设置、只写题目上：N1 减强单 E 的拆分（挑题 / 回答 / 边界）、换答文的两行、N1 减“两遍的单 E”（作答数相同）。只读现成判分；题目清单取主表的，第一行必须与 `main_table_prompt.py` 的“N1 减强对照”逐数相同（已核：五个模型 520 题 +128 [+58, +197]） | 新写；GPT 第三轮复审查出一处错（同作答数的汇总先在各模型自己的题单上排名、再取共同题，合计时单 E 多补一题），16 时半按它的补丁 `discussion/validation_dataset/2026-11-22c_equal_count.patch` 改成先取共同题再排：N1 减两遍的单 E 五个模型 +63 改为 +64 [+16, +110]，三个模型 +50 改为 +51 [+17, +86]，第一行（主表的数）与分模型各行不变；输出 `results/write_positions/tq_same_n_decomp_five_models_20261122.txt` | 8117f4de5f0649e2 |
| `diag/qaqa_gate_fair.py` | 探索：运行 GPT 的 `2026-11-22_recovery_pool_b.py`（文件不动，只在内存里改三处），让“单 E 拿到同一个识别器、补同样多空答”这一行在完美题型识别（题集官方题型）下也算出来 | 新写；GPT 第三轮复审更正一处标签（题型筛选后一共交了多少条，原先标成了新补了多少条），数字不变；输出 `results/newsets/qaqa/qaqa_gate_fair_five_models_20261122.txt`（N1 减它 +12 [−14, +38]） | d343939b16cdada7 |
| `diag/iti_faithful_input_checks.py` | 只读检查：忠实版 ITI 两个模型的回答文件（`baselines/iti/<模型>/faithful_seed42/gen/`）与判分输入（`judge_{tune,test}_iti_faithful_seed42*/judge_input.json`）逐个核三件事——题号不重复、行的题号顺序等于文件自己的 `question_ids`、等于同一切分的参照判分输入（Llama `pospromptl16`、Gemma-3-12B `pospromptl24`）；为装 `scripts/newsets/iti_baseline.py` 的补丁做准备。在宿主机上从项目根运行 | 新写；GPT 第三轮复审指出空目录或某一臂整批重复也能通过，16 时半按它的补丁 `2026-11-22c_iti_checks.patch` 加严（另核网格是否齐全、冻结选择、判分输入的源文件摘要与实际答案）；输出 `results/baselines/iti/iti_faithful_input_checks_20261122.txt`（两个模型的 66 个回答文件、14 个判分输入与 4 份参照全部满足，返回码 0；两类反例上返回码 1） | edde77163035f1a3 |
| `diag/iti_gsm8k.py` | 探索：ITI 各格点的 GSM8K（1,319 题，规则计分，与 `gsm8k_main_table.py` 同一套函数与判词），对平时的不写入那一遍；并列出我们自己的 E 在两种写入位置下的读数。容器里运行 | 新写，未经 GPT 审 | 6e36ea3bc5f90044 |
| `scripts/review_checks/ff_posrule_confidence.py` | 探索：只写题目下的强对照要用的把握。把握计算程序原样运行，内存里两处改动：要算的回答换成“必须作答 + 只写题目的 E”的那一遍；打分时的写入位置由 content 换成 prompt。98% 复现与第一处不同的检查都照旧 | 新写，未经 GPT 审 | 0813f8c16784518d |
| `diag/prompt_conf_table.py` | 主表缺的一个对照：纯提示词再按模型自己的把握挑题（什么都不写入）。P1 = 必须作答 + 如实句的提示留下与 N1 同样多的最有把握的回答；P2 = 提示与门槛都在调参集上选。N1 的逐题记录直接取自主表程序。规则先写在文件头再看数 | 新写，未经 GPT 审；挑题的几个函数用假题测过 | 2bb6bf53f7b11b45 |
| `diag/ns_stable_table.py` | 探索：TriviaQA / PopQA 上把两次写入当作对模型原回答的“稳定性检查”（N1 规则的镜像：只保留推 E 后不变、或加 C 后原样回来的原回答）；E 力度与上限在调参集上选，读测试集；2026-11-21 按 GPT 的补丁改两行：保留原回答时取原回答自己的判决（此前取的是探针那一遍的判决；现有表逐字不变） | 新写，未经 GPT 审 | 822bb02da36210e1 |
| `diag/ns_probe_table.py` | 探索：把任意一遍写入（E、C 往谨慎、C 往作答、两个一起写）或换问法当作“探针”，只保留探针下不变的原回答；各族在调参集上选成员，读测试集；含“换成必须作答问法”的对照 | 新写，未经 GPT 审 | c85c558756ec40ea |
| `diag/ns_choose.py` | 只用调参集，给出某模型某题集需要在测试集上补“再问”回答的 E 力度（N1 最好的一档、稳定性检查最好的一档） | 新写，未经 GPT 审 | 1a674775fff960e4 |
| `diag/ns_caut_confirm.py` | 探索：两套事实题上“E 加往谨慎推的 C”的确认——设置在调参集加第一批测试题（692 题）上选，只读没用过的第二批测试题（500 题）；`--choose` 给出要在第二批题上生成的三遍回答 | 新写，未经 GPT 审 | 2e4baaec3c43e3b4 |
| `diag/ns_pressure_pilot.py` | 探索小试验：两套事实题题后加一句用户看法（Sharma 等 ICLR 2024 的句子）之后，C 放出来的回答对多还是错多；只看调参集 | 新写，未经 GPT 审 | ba4b2e5964a5832a |
| `diag/ns_pressure_table.py` | 探索：同一种问法下，各系统的设置在调参集上选、读测试集；每个差值按“原题是否答对”拆开；`--choose` 给出要在测试集上生成的回答 | 新写，未经 GPT 审 | 20ac688d51789679 |
| `scripts/newsets/newsets_pressure_bank.py` | 造带用户看法句子的题库（否定句 / 肯定句，只用标准答案）；题号、标准答案、切分照抄，生成与判分程序原样可用；题目原文只留服务器 | 新写，未经 GPT 审 | 3f5416f240da0458 |
| `diag/train_loss_table.py` | 从各次运行的训练日志读出 E 方向在留出题上的损失：不写入、整个激活差、均值差方向、学出的方向 | 新写，未经 GPT 审 | 88ef7200f986cea5 |
| `scripts/review_checks/e44_positions.py` | 只在评测时换写入位置的入口：把训练时“从题目第一个词写到结尾”的方向，改成只从提问的最后一个词元起写（题目的词不动）；原程序文件不动，三处内存内改动；只允许单写入的生成与导出 | 新写（探索：MMLU 掉分来自题目词上的写入），未经 GPT 审；Llama 8 题试跑后运行 | 1a8ae7861c1c103e |
| `scripts/review_checks/make_keep_source.py` | 给 E 的训练题里加入多项选择题的源目录构造：原训练题加 160 道 MMLU 诊断题（调参集、测试集不动）；训练程序不改，照旧在训练题上学“写入后要与带如实句的提示输出一致” | 新写（探索：能否让 E 不伤 MMLU），未经 GPT 审；Llama 的源目录已构造并核对 | fd71b3337a121b18 |
| `diag/selfjudge_probe.py` | 探索：作答后自判。单 E 在 TruthfulQA 上已作答的题，让模型对自己的回答做真假判断（E45 的真假判断代码原样导入，不写入与写入真假 E 力度 1 / 2 / 3 四种条件），看“判为假 → 不答”能否加净分；`export` 导出题与答案（题目原文只留服务器），`score` 在容器里做前向（先对 E45 自己的测试句回归），`table` / `step2` 出第一步、第二步的表（R5a / R5b 规则，门槛只在 TruthfulQA 调参集上选）；`--sources` 给阶段一与事实题的其他答案来源打分 | 新写，探索；已按 GPT 第四、五轮的补丁修改（前向与读表的核对加严；启动器加占卡锁） | dd9e8bb20443bf61 |
| `diag/selfjudge_equal_n.py` | 探索：作答后自判的第三步，相同作答数的比较（只读，不调模型与裁判）：单 E + 自判对同样多去掉的单 E、按把握留同样多的强单 E；N1 + 自判对强单 E；TruthfulQA 与 (QA)² 各一套 | 新写，探索；已按 GPT 第四、五轮的补丁修改 | 4736e617c7bdb3e8 |
| `diag/selfjudge_stage1.py` | 探索：作答后自判的阶段一（只读）：强单 E 的回答、N1 放出的回答也交给自判；S1 / S2 / S3 三个系统各对同作答数的强单 E；门槛规则 P / P' 只在 TruthfulQA 调参集上选；`tableZ` 是规则 Z（不选门槛，判分差小于 0 即判假，否定前提开头的不判） | 新写，探索；已按 GPT 第四、五轮的补丁修改 | 46d71d1181cb9526 |
| `diag/selfjudge_gate.py` | 探索：题型开关。只在单 E 留空的题上决定放不放 N1 的回答：K0 = (QA)² 官方题型（只作诊断）、K1 = 模型自己判断这道题是否基于错误或无法核实的前提（固定提示，容器里前向）、K2 = N1 自己的来源信号；各对同作答数的强单 E 与带同一开关的单 E | 新写，探索；已按 GPT 第四、五轮的补丁修改（K1 读表核模型、提示、题单与分数摘要，分数须有限；打印改为不可比时标 n/c） | efe730b80918f904 |
| `diag/selfjudge_factqa.py` | 探索：事实题（PopQA、TriviaQA 两批测试题）上的作答后自判，规则 Z；只用别名匹配的判决（这些写法没有 Pro 判分）；S1 / S2 / S3 对单 E 与 N1 | 新写，探索；GPT 第四、五轮已审，文件未改（它导入的 `selfjudge_probe.py` 改了，表逐行相同） | 35abb91942303c2d |
| `diag/selfjudge_gate_k3.py` | 探索：第三个题型开关 K3——在不写入的提问末位读各层残差流，用 (QA)² 的 100 道调参题训一个线性探针判“陷阱题”（五折交叉验证选层）；`capture` 在容器里取特征，`fit` 拟合，`table` 出表（导入 `selfjudge_gate.py`） | 新写，探索，未经 GPT 审 | 3e745a3030bd06e7 |
| `diag/factqa_nonanswer_diag.py` | 只读诊断：事实题上别名匹配判错的回答里有多少其实是“不回答”（固定短语表，先写后看数），把它们记 0 后 N1 减单 E 变成多少；回答、判决与系统都取自 `ns_prompt_table.py` | 新写，探索，未经 GPT 审 | 20b2749072e34806 |
| `diag/selfjudge_arbitrate.py` | 探索：排期庚 G4。同一道题有两条不同的回答时（原模型的与 E 的；C 放出的与“必须作答 + E”的），两条各做作答后自判，取自判分高的一条；G4c 再加规则 Z；不写入与写入真假 E 两种条件。只读现成回答、判决与自判分 | 新写，探索；GPT 第六轮审过（分母 113 更正为 112） | 4bca9c4ccfb28b25 |
| `diag/ec_offline_checks.py` | 探索：排期庚的三项离线检查——G1 每题只生成一遍的 E + C；G2 两个补答上限 25 种组合的平均；G3 四档 C 的独有答对候选。只读现成回答与 Pro 判决 | 新写，探索；装了 GPT 第六轮补丁（同一模型、同一题、同一文字统一判决，G3 由 71 改为 66） | b110985ebf4cc021 |
| `diag/plain_honesty.py` | 探索：排期辛第 5 节。只加如实句、什么都不写入的普通问法（`x_E2`，记作 ⓪）在 TruthfulQA、(QA)² 570 题（调参 / 留出、分题型）与事实题四批上，对原模型、只推 E、N1 的差 | 新写，探索，未经 GPT 审 | 2fd34e98ff8f6f69 |
| `diag/plain_honesty_overlap.py` | 只读：TruthfulQA 上 ⓪ 与只推 E、N1 的逐题转移、逐题取较好一条的上限，以及纯提示词答两遍（⓪ 先答，留空的题用“必须作答 + 如实句”补，T2） | 新写，探索，未经 GPT 审 | 86c4e0e320d69a69 |
| `diag/plain_honesty_fill.py` | 只读：TruthfulQA 上几种补空的做法——⓪ 先答、留空用 N1 的回答补（F1）；N1 先答、留空用 ⓪ 补（F2）；留空用只推 E 补（FE）；T2 | 新写，探索，未经 GPT 审；F1 是看了数据后想到的做法 | 1483bba5fc4d82a5 |
| `diag/replacement_2x2.py` | 探索：排期辛的 2×2——“如实”用句子或 E 实现 × “作答”用必须作答的指令或 C 实现，四格在同一批题上比；`--with-zero` 加 ⓪ 一行 | 新写，探索，未经 GPT 审；GPT 第六轮的两个参照数逐数复现 | f69d0077891e2427 |
| `diag/n1_on_sentence.py` | 探索：排期壬第一步。主表整套做法叠在如实句（`x_E2`）上：P + E、P + E + C 按 N1 的规则组合，E 两档力度在调参集上选、测试集读；第 0 步必须用同一程序逐数复现主表的十个参照数 | 新写，探索，未经 GPT 审；主会话用原始判决另算核过 | f479640f41d97ff0 |
| `diag/selfjudge_two_answers.py` | 探索：排期癸。⓪ 的回答与 N1 的回答不同时，按作答后自判的分数取高的一条（A1），再加规则 Z（A1z）；对照 F1、T2 与纯提示词的同样做法；`score1` 把自判分改为一条一条算（成批算会随同批的其他句子起伏） | 新写，探索，未经 GPT 审 | acecb242b9113ec7 |
| `diag/caa_table.py` | 探索：外部对照 CAA（对比激活相加，ACL 2024）在本项目流程里的表：MMLU 预算、调参集上选层与倍数、测试集对原模型 / 只推 E / N1；结构与 `iti_table.py` 相同 | 新写，探索；方案经 GPT 第五、六轮审；2026-11-23 装了 GPT 晚间复审的 `caa_identity.patch`（改前为 5b1505dcef9b5c80） | f1648ff48783d9b6 |
| `diag/caa_gsm8k.py` | 探索：CAA 选中格点的 GSM8K（1,319 题，规则计分），对平时的不写入那一遍 | 新写，探索，未经 GPT 审 | e047860e8420646b |
| `scripts/newsets/caa_vector.py` | 探索：CAA 的向量——96 道训练题、409 个正误选项对、A/B 位置交换、先题内后题间平均、读在字母词元上；写成方向记录放进镜像运行目录；含验收（零倍复现、正负号检验） | 新写，探索；装了 GPT 第六轮补丁；2026-11-23 又装了 `caa_identity.patch`（改前为 ef185badca118054） | 92d9c382ec50a218 |
| `scripts/newsets/caa_run_guard.py` | CAA 启动脚本的输入检查（训练来源、镜像向量、续跑时的核验），来自 GPT 第六轮补丁 | GPT 写，原样装入；2026-11-23 装了 `caa_identity.patch`（改前为 ff7472d675e2dd30） | 99c67f3b50b66db0 |
| `scripts/newsets/n1s_check.py` | 排期壬的开跑前核对：零写入与不写入逐词元相同、两遍相同、写入记录与主表同层同族相同 | 新写，探索，未经 GPT 审 | 719d709e974ad400 |
| `scripts/review_checks/e44_positions_h22.py` | 只在评测时用的入口：原样调用各代码副本里的 `e44_positions2.py`，只把两写入的验收记录改名为 `*-h22-at-prompt.json`，不覆盖主表那一份 | 新写，探索，未经 GPT 审 | 570dfc48af6bf5f4 |
| `configs/prompts/e44_eval_extra_h22_v1.json` | 评测用的提示条件：原有两条照留，加 `x_E2` = 普通问法 + 如实句（与 `e44_promptdiag_v1.json` 的写法相同）；不进训练 | 新写 | 6fe6e437a18c9b41 |
| `configs/newsets/caa_grid_20261122.json` | CAA 网格的候选层与倍数（排期己第 9 节） | 新写 | ff21bd1b66a042b8 |
| `configs/selfjudge/gemma3-12b.json` | 作答后自判程序给 Gemma-3-12B 用的模型配置 | 新写 | ddf654d812780b6b |
| `configs/selfjudge/gemma3-27b.json` | 作答后自判程序给 Gemma-3-27B 用的模型配置 | 新写 | b4ea1b54b06cc963 |
| `launchers/explore_20261120/` | 2026-11-20 探索用的服务器启动脚本原样存档（2026-11-24 时 176 个，含 `selfjudge_1122.sh`（作答后自判的前向：`selfjudge_probe.py score`，只用 1 号卡）、`selfjudge_d_1122.sh`（(QA)² 上单 E 那一遍回答的把握分）、`selfjudge_gate_1122.sh`（K1 的前向与事实题自判的前向）、`selfjudge_gate_k3_1122.sh`（K3 的取特征与拟合）——前三个已按 GPT 第四、五轮的补丁加占卡锁，第四个照同样写法加锁、未经 GPT 审；含 `iti_faithful_1122.sh`（忠实版 ITI 在 Llama 与 Gemma-3-12B 上的拟合、自检、网格与两种读法）、`iti_faithful_g27_1122.sh`（同一套做法用到第三个普通注意力模型 Gemma-3-27B，每张卡一条网格支线；GPT 第三轮复审的启动保护补丁 `2026-11-22c_iti_launcher.patch` 已进仓库——先过自检再发任务、核卡号、加锁、清旧标记——服务器上 15:15 起在跑的是补丁前的版本，这一趟结束后再换）、`iti_recheck_1122.sh`（只读审核要的三项复查）；含 `iti_free.sh`（ITI 的第二种读法：整个网格不看预算取最好的格点、读测试集，再给两个选择各量一遍 GSM8K）、`e_hi_llama.sh`（Llama 的 E 在力度 5、7 下的 MMLU 与 GSM8K）；含 `iti_chain.sh`（ITI 的一份变体从补齐网格到读测试集的无人值守流程）；含 `iti_grid.sh`（ITI 对照：各格点的 MMLU 全量与调参集回答，不调裁判；某个 K 超预算太多时跳过更大的力度）、`iti_judge.sh`（ITI 回答的判分：题目与参考答案照抄现成裁判输入，先并入那份输入的裁判缓存）；含 `qaqa_strong.sh` / `qaqa_strong_1122.sh`（(QA)² 上相同作答数的强对照：必须作答 + 只写题目的 E 与必须作答不写入两遍的生成、判分与把握分）；含 `qaqa_conf.sh` / `qaqa_conf_1122.sh`（(QA)² 上纯提示词回答的把握分）、`qaqa_tables_1122.sh`（四张终表）；含 `qaqa_run.sh` / `qaqa_full_1122.sh`（(QA)² 五个模型 × 570 题，设置沿用 TruthfulQA）、`qaqa_tune.sh` / `qaqa_tune_1122.sh`（100 题当调参集，各档力度）、`qaqa_tuned_chain.sh`（调参集选定 → 补生成 → 出两种读法的表）；含 `qaqa_pilot.sh`（(QA)² 小试验：生成、导出、Pro 判）、`posm_rest_1122.sh`（其余四个模型的均值差对照）；含 `nsc_prompt.sh`（事实题只写题目下 C 往谨慎的小试验：调参集整格、选完只补选中的臂）、`night2b_1121.sh`（取代 `night2_1121.sh`：均值差对照先只做 Llama）、`night3_1121.sh`（小试验扩到其余三个模型）；含 `night2_1121.sh`（第二条夜间队列）、`gsm8k_n1_pos.sh`、`posm.sh`（同层同写入量的均值差方向，只写题目）；含 `ns_prompt.sh`（事实题只写题目）、`night_1121.sh`（11-21 夜间队列）；含 `gsm8k_pos_27b.sh`（两个 27B 的 GSM8K 只写题目）；其中 `pospall.sh`、`posp_batch.sh` 的说明文字 13:2x 按自查改过，批次跑完后（22 时）已同步到服务器；含 `posd.sh`、`posd_batch.sh`（只写题目下的强对照）；含 `pospall.sh`（某模型某层“只写题目”下主表要的整套）、`posp_batch.sh`（五个模型的排队）、`gsm8k_pos_q9.sh`；含 `posp2.sh`（N1 只写题目位置）、`g27_mm28.sh`（Gemma-3-27B 那组 MMLU 换到 2 号卡重跑）；含 `gsm8k_pos.sh`（GSM8K 换写入位置）、`posp.sh`（TruthfulQA 只写题目位置的只推 E）；含 `g27_gpu0.sh`（Gemma-3-27B：第 28 层叠最好提示各臂的 MMLU，再接 GSM8K）；含 `gsm8k_cap.sh`（GSM8K 换生成上限）、`prompt_conf.sh`（各条提示无写入回答的把握）、`lane27f.sh`（27B 队列当前版本）；含 `gsm8k_27b.sh`（两个 27B 的 GSM8K，排在各自占卡的任务之后）；含 `lane27e.sh`（27B 统一队列的当前版本；一步因显存不足失败后按“同卡不放两个大任务”重排）；含 `gsm8k_main.sh`（主表设置下的 GSM8K）；含 `oncaut_mm_l24.sh`；含 `lane27d.sh`（27B 统一队列的当前版本：Gemma-3-27B 的“叠在最好提示上”改在第 28 层做）；含 `alt_strong.sh`（换层后的同力度强对照与同层均值差对照，通用版）、`oncaut_mm_l16.sh`、`lane27c.sh`（27B 统一队列的当前版本）；含 `alt_chain.sh` 与 `g12_l24_launch.sh`（换层整套的通用版，Gemma-3-12B 第 24 层用它）、`g12map.sh`（Gemma-3-12B 的 MMLU 分层图）、`oncaut_mm_rest0.sh`、`oncaut_alt.sh` 与生成它的 `build_oncaut_alt.py`（叠在最好提示上的 N1，E 写在另一层）；含 `g27_l28.sh`（Gemma-3-27B 第 28 层的整套）与 `lane27b.sh`（27B 统一队列的当前版本，替换 `lane27.sh`）；含 `oncaut_mm_rest.sh`（Llama、Qwen3.5-9B 叠在最好提示上各档的 MMLU）；含 `l16_mdg.sh` 与 `l16_mdg_launch.sh`（第 16 层同层的均值差对照）；含 `l16_strong.sh` 与 `l16_strong_launch.sh`（第 16 层的同力度强对照）、`md27.sh`（两个 27B 的均值差自己挑层，分成两段占卡）、`oncaut2_g27b.sh`、`lane27.sh`（两个 27B 模型的统一队列，替换 ns2c / ns2d / ns2e）；含 `low_strong.sh`（低力度的同力度强对照）与 `ns2e_27b.sh`（27B 支线的当前版本）；含 `pressure_pilot.sh`、`pressure2.sh`、`pressure2_launch.sh`（题后加一句用户看法的试验）与 `oncaut_mm_g12.sh`（叠在最好提示上各档的 MMLU）；含 `l16b_all.sh`（第 16 层那条链改成判分与 MMLU 同时进行）与 `ns2d_27b.sh`（27B 支线前加 Gemma-3-27B 的 MMLU 分层图）；另有 01:2x 之后新增的 `ns2c_27b.sh`（两个 27B 模型的短清单）、`ns4_test2.sh` 与 `ns4_launch.sh`（第二批测试题上的确认）、`md_q9_fix.sh`（一步因撞卡失败的重跑）、`oncaut2_g27.sh`（改了起跑条件的 Gemma-3-27B 支线）；新增 Llama 第 16 层 `l16_all.sh`、叠在最好提示上的 `oncaut_all.sh` 与省判分的第二版 `oncaut2_all.sh`、新题集上 C 取谨慎方向的 `ns_caut.sh`、新题集各档力度的精简版 `ns2_all.sh` 与改排显卡并补测试集再问回答的 `ns2b_all.sh`；另含含 Llama 换层加力度的试探 `layer_probe.sh`、训练题加入 MMLU 题的试验 `keep_pilot.sh`、作答位置写入第二轮 `pos2_all.sh` 与 `pos2_judge.sh`、Llama 与 Gemma-3-27B 的低力度 `low_all.sh`、新题集换力度 `ns_dose_all.sh`；晚间新增 Gemma-3 重跑 `pm3_gemma.sh`、只在作答位置写入 `pos_all.sh` 与 `pos_judge.sh`、均值差选层 `md_sweep.sh`、提示词基线全判 `prompt_sweep.sh`）：新题验证 `val_all.sh`；MMLU 找原因与全量确认 `sweep_all.sh`、`sweep_g27.sh`、`full_all.sh`；中间力度 `dose_mid_all.sh`、`dose_mid_judge.sh`；提示词之上再写 E `idea_a.sh`、`idea_a2.sh`；高力度 `q9_high.sh`；补判 `extra_judge.sh`、`extra_judge2.sh`；下午两条队列 `pm_all.sh`、`pm_judge.sh`、`pm2_all.sh`、`pm2_judge.sh` | 均未经 GPT 审（探索阶段先跑，成批送审）；生成、导出、判分调用的都是已审程序，这些脚本只改臂、切分与名称 | 见各文件 |
| `diag/n1_on_sentence2.py` | 排期壬第 7 节：如实句叠法在 (QA)² 上的读数（7.1）、只加 C 的 SC（7.2）、事实题四批的 SE（7.4）、一遍到底的对照（7.5）；只读 | 新写（Opus 5.5），探索；GPT 11-23 复审（`discussion/validation_dataset/2026-11-23_gpt_review.md`）时在服务器上读过并独立复算 | e35681d6c5b577ea |
| `diag/n1_on_sentence3.py` | 排期壬第 8 节：如实句方法的 E 力度在四个调参集上选（s_sel 或 s_half = s_sel / 2）；`choose`、`fact`、`final` 三种用法；只读 | 新写，探索；GPT 晚间复审的 `2026-11-23_l_review/strength.py` 独立复算了五个模型的选择 | 7ed1b7656cae7ee6 |
| `diag/prompt_family.py` | 排期壬 7.6：各条不写入提示（含 x_H、x_E2_H、x_E1_E2）在 TruthfulQA 上的净分，调参集最好的提示 PB 与 SN1 / SE 的比较；只读 | 新写，探索；GPT 11-23 复审时在服务器上读过 | 2af1ea6bdc291eca |
| `diag/selfjudge_stack.py` | 排期癸第 6 节：如实句叠法 SN1 两边用同一自判（批大小 1）与规则 Z；`score` 是占卡的一步 | 新写（Opus 5.5），探索；GPT 的 `candidate_checks.patch` 延后（排期 L 第 7 节第 5 项） | e9e65d074922d586 |
| `diag/selfjudge_stack_pick.py` | 排期癸第 7、7a 节：叠法的回答与不带如实句的单 E 回答按模型自判二选一；也给 `fill_only.py`、`summary_table.py` 提供各系统的逐题格 | 新写（Opus 5.5），探索；`candidate_checks.patch` 同上延后；GPT 晚间复审的 `fill_scores.patch`（只加一个可选的自判分来源）由执行人甲验证使用，本条摘要是 2026-11-23 23 时仓库与服务器相同的版本 | 58c3b54d2a08fe6f |
| `diag/fill_only.py` | 排期 L 第 1–4 节：只补空（S0 已答的题原样交，只在 S0 按规则留空的题上交 SN1 / SE / SC / conf_E2 / 不带如实句的 N1）；用法 `check`、`missing`、`tqa`、`qaqa`、`fact`、`sec4`；只读 | 新写（Opus 5.5），探索；主会话 13:40 运行，第 3 节核对全过；GPT 晚间复审复现（`2026-11-23_evening_review.md`）；`fill_scores.patch` 同上，本条摘要是 23 时的版本 | 7ae947b4745bd098 |
| `diag/sentence_capability.py` | 排期壬 7.3 与第 9 节：如实句接在系统提示后的能力代价，MMLU（13,042 题，选项概率）与 GSM8K（1,319 题，上限 512）的 P、P + E(s_sel)，`--half` 加 P + E(s_half) 与截断数 | 新写，探索；GPT 晚间复审的 `2026-11-23_l_review/capability.py` 独立复算；`capability_reader.patch` 不用（排期 L 第 7 节第 5 项） | 4a024fe763dbbec8 |
| `diag/summary_table.py` | 排期 L 第 5 节（含 10.1 的两处更正）：一张总表——原模型、S0、单 E、N1、只补空、按调参集选力度的叠法 SN1m、整套叠法 SN1（ITI 三个模型单列）在 TruthfulQA 测试集、(QA)² 留出（分题型）、TriviaQA / PopQA 各两批上的净分与作答数，及 MMLU、GSM8K 相对原模型的变化；另有 P + E 两档相对 P 的小表。只 import 现有读表程序，开表前与一组现成数核对，对不上就不出表 | 新写（Opus 5.5，执行人丙），探索，未经 GPT 审；输出 `results/write_positions/summary_table_five_models_20261123.txt` | 6dd411aa64c60183 |
| `scripts/capability/cap_h22.py` | 排期壬 7.3：能力测量的入口，把如实句接在系统提示后（MMLU 经 `cap_mcq_sweep.py`，GSM8K 多一个提示名 `neutral_E2`）；原程序不改，只在内存中加提示 | 新写（Opus 5.5），探索；没查到 GPT 审读记录 | 819380f607e82f24 |
| `scripts/newsets/ns_h22.py` | 排期辛与壬 7.1 / 7.4：用 `configs/prompts/e44_eval_extra_h22_v1.json` 代替原提示文件运行 `newsets_gen.py`（内存中改一处），另有 `check`、`compare`、`parts` 三个核对 | 新写（Opus 5.5），探索；GPT 11-23 复审的 `2026-11-23_replay/stack_checks.patch` 涉及它（审核附件，未部署） | c059807998f46b3b |
| `scripts/review_checks/e44_pfam.py` | 排期壬 7.6：只在评测用的入口，把运行的提示配置里没有的句子（TruthfulQA 官方 PROF_PRIMER 的指令部分 H）在内存中加进去，再交给 `e44_positions2.py` | 新写（Opus 5.5），探索；没查到 GPT 审读记录 | 82c8428530c8b799 |
| `configs/prompts/e44_eval_extra_pfam_v1.json` | 排期壬 7.6 的三条不写入提示 x_H、x_E2_H、x_E1_E2（H 写在 `sentences_added`） | 新写 | ac485c4a207c6279 |
| `launchers/explore_20261120/caa_accept.sh` | 排期己第 4、7 节：一个模型的 CAA 验收（拟合向量、镜像、检查、同一设置两遍逐词元相同、小看一眼），不调裁判 | 原样存档（服务器 `logs/newsets_20261119/`，下同），探索；GPT 第六轮审读过（`discussion/validation_dataset/2026-11-22_gpt_review_f.md`） | 9cf6e06ea8c8a2a6 |
| `launchers/explore_20261120/caa_grid.sh` | 排期己第 1、7 节：一个模型的 CAA 正式网格（逐层 MMLU、提前停、调参集回答、判分输入），不调裁判 | 探索；GPT 第六轮审读过；排期 L 第 7 节记为审过的脚本 | 29c0f9ffb17653b1 |
| `launchers/explore_20261120/caa_chain.sh` | 排期己第 1、7、8 节：一个模型从网格到测试集与 GSM8K 读数的无人值守流程 | 探索；GPT 第七轮补丁 `caa_checks.patch` 已装（见变更记录 2026-11-23 01:59） | d774bcddd9614261 |
| `launchers/explore_20261120/caa_judge.sh` | CAA 一个判分目录的 DeepSeek Pro 判分：先并入常规运行的裁判缓存，避开高峰 | 同上 | 23da524974e04fc7 |
| `launchers/explore_20261120/caa_night_1122.sh` | 排期己第 9 节第 3 项：11-22 夜里不花判分钱的部分（各模型验收与网格）的排卡 | 探索；没查到 GPT 审读记录 | bd532872ad9dc013 |
| `launchers/explore_20261120/caa_night2_1122.sh` | 同上的第二版排卡（3 号卡 Llama-3.1-8B 后接 Qwen3.5-9B），替换前者 1 号卡与 2 / 3 号卡两条支线 | 探索；GPT 晚间复审的 `caa_identity.patch` 改它监控里的一处算术，由执行人乙验证部署；该脚本 2026-11-24 03:33 运行结束，08:00 服务器上的文件换成补丁后的版本（运行过的是补丁前的 6924686f6b760541） | 6da190917758be3d |
| `launchers/explore_20261120/cap_h22_model.sh` | 排期壬 7.3：一个模型一张卡上量如实句的能力（MMLU：x、x_E2、x_E2 + E(s_sel)；GSM8K：neutral、neutral_E2、neutral_E2 + E 只写题目） | 探索；没查到 GPT 审读记录 | c32a83fa714608ee |
| `launchers/explore_20261120/cap_h22_part.sh` | 同上，第三个参数只跑 MMLU 或 GSM8K（两个小模型的 MMLU 超过 1 号卡 65 GB 的规矩） | 同上 | 17deb221ee983a4f |
| `launchers/explore_20261120/cap_h22_1122.sh` | 7.3 五个模型的排卡 | 同上 | a236f128d66e42e5 |
| `launchers/explore_20261120/cap_h22_small_1123.sh` | 两个小模型 7.3 的排卡（GSM8K 在 1 号卡，MMLU 在大卡） | 同上 | 554b7e9d62588de5 |
| `launchers/explore_20261120/cap_h22_g27_restart.sh` | Gemma-3-27B 的 7.3 让卡后重启（7.6 排在 7.3 前） | 同上 | 3fddcd887341421b |
| `launchers/explore_20261120/cap_h22_g27_resume2.sh` | Gemma-3-27B 的 7.3 续跑（2 号卡的锁被抢先 14 秒后） | 同上 | 3066eca5044ec25e |
| `launchers/explore_20261120/cap_h22_g27_resume3.sh` | 同上，加 0 号卡（门槛 8 GB） | 同上 | a17be0e5bc1f77c6 |
| `launchers/explore_20261120/cap9_model.sh` | 排期壬第 9 节：E(s_half) 的能力代价，一个模型（Qwen3.5-9B 另补不带如实句的 GSM8K 参照） | 同上 | 10b9e24c138ac49d |
| `launchers/explore_20261120/cap9_1123.sh` | 第 9 节五个模型的排卡 | 同上 | a1bf79f9a0c51090 |
| `launchers/explore_20261120/cap9_g27gsm_1123.sh` | Gemma-3-27B 第 9 节的 GSM8K 在 0 号卡、与它的 MMLU 并行 | 同上 | a4466d42799bc97f |
| `launchers/explore_20261120/give_card_1123.sh` | 11-23 01:46 主会话定：2 / 3 号卡当前那一遍能力测量结束时让卡给排期癸的自判分一次，之后原样续跑 | 同上 | 6320f0689aa26758 |
| `launchers/explore_20261120/gsm8k_orig.sh` | 11-21 GSM8K 掉分排查：E 放回换层前的 TruthfulQA 选层（Llama-3.1-8B 第 11 层 0.75、Gemma-3-12B 第 22 层 1.25）时的 GSM8K | 同上 | 16f4efb3f55795b3 |
| `launchers/explore_20261120/h22_model.sh` | 排期辛：2×2 的第 4 格（如实句 + 只加 C 0.25，只写题目），一个模型 | 探索；GPT 11-23 复审指出续跑只看验收文件是否非空，补丁 `stack_checks.patch` 是审核附件、未部署 | ad09b6c5ff949236 |
| `launchers/explore_20261120/h22_1122.sh` | 第 4 格五个模型的排卡 | 探索；没查到 GPT 审读记录 | a90b24df2076faf1 |
| `launchers/explore_20261120/h22_q27_1122.sh` | Qwen3.8-27B 的第 4 格改排 3 号卡（替换 `h22_1122.sh` 的 D 支线） | 同上 | ac8c3964888e3139 |
| `launchers/explore_20261120/h22z_model.sh` | 排期辛第 5 节：第 0 格（如实句、不写入）在 (QA)² 与事实题四批上的生成与判分 | 同上 | 9bf6af84e6739012 |
| `launchers/explore_20261120/h22z_1122.sh` | 第 0 格五个模型的排卡 | 同上 | 19b2a9b40e5bd55e |
| `launchers/explore_20261120/n1s_model.sh` | 排期壬第一步：主表的 E / E + C 叠在如实句上（TruthfulQA，只写题目），tune、test 两段 | 探索；GPT 11-23 复审，同 `h22_model.sh` | 35da2df5d65c5abe |
| `launchers/explore_20261120/n1s_1122.sh` | 第一步五个模型的排卡 | 探索；没查到 GPT 审读记录 | 93e1ace5fa59514a |
| `launchers/explore_20261120/n1s_q27_1122.sh` | Qwen3.8-27B 的第一步改到 3 号卡 | 同上 | ec79dbcbf6dcf1a3 |
| `launchers/explore_20261120/n1s2_model.sh` | 排期壬第 7 节（7.1 (QA)²、7.2 只加 C、7.4 事实题）一个模型的生成与判分 | 同上 | 7b65a1dd4375bb2e |
| `launchers/explore_20261120/n1s2_1122.sh` | 第 7 节五个模型的排卡 | 同上 | 3313942484ea9445 |
| `launchers/explore_20261120/n1s2_qaqa_rejudge.sh` | 7.1 Llama-3.1-8B 与 Gemma-3-27B 的 (QA)² 判分停在缓存冲突检查，按主会话 00:02 的决定在新目录 `test_full570_h22n1` 只并 S0 的缓存重判 | 同上 | ade8b66885048a5d |
| `launchers/explore_20261120/n1s3_model.sh` | 排期壬第 8 节：在四个调参集上选力度所需的 s_half 生成，选中 s_half 时补测试集一遍 | 同上 | 96e4303e8242a4c1 |
| `launchers/explore_20261120/n1s3_1123.sh` | 第 8 节五个模型的排卡 | 同上 | 78e7f86b3f930c1f |
| `launchers/explore_20261120/n1s3_g27_1123.sh` | Gemma-3-27B 的第 8 节改在 0 号卡，其后接 Qwen3.8-27B 的第 9 节 | 同上 | a9ed541d6e66d95e |
| `launchers/explore_20261120/n1s3_g27b_1123.sh` | Qwen3.8-27B 的第 9 节换到 3 号卡，Gemma-3-27B 的第 8 节在这里接着走 | 同上 | cff95f6f9882f0db |
| `launchers/explore_20261120/pfam_model.sh` | 排期壬 7.6：三条不写入提示（x_H、x_E2_H、x_E1_E2）在 TruthfulQA tune / test 的生成与判分 | 同上 | 1a6f4db22df7b68f |
| `launchers/explore_20261120/pfam_1122.sh` | 7.6 五个模型的排卡 | 同上 | 25249f6b68181e34 |
| `launchers/explore_20261120/selfjudge_gpu0_1122.sh` | 排期戊（1d）与庚（G4）：`selfjudge_probe.py` 的 score / cfgcheck 在 0 号卡上逐个模型跑 | 同上 | f79207231e9d47f4 |
| `launchers/explore_20261120/selfjudge_two_answers_1122.sh` | 排期癸"两条回答挑一条"：`selfjudge_two_answers.py score` 一个模型一张卡（有真假 E 的模型先做 E45 全量回归） | 同上 | 0d5df6ca18b01f1f |
| `launchers/explore_20261120/selfjudge_two_answers_chain_1122.sh` | 排期癸：其余模型的自判分逐个排卡 | 同上；服务器上没找到同名文件 | 99dedacd9af8ff7d |
| `launchers/explore_20261120/selfjudge_two_answers_b1_chain_1122.sh` | 排期癸第 5 节：批大小 1 的自判分逐个模型 | 同上 | a385608de4bf48b4 |
| `launchers/explore_20261120/selfjudge_stack_1122.sh` | 排期癸第 6 节：`selfjudge_stack.py score` 一个模型一张卡 | 同上 | af482395b9d079ee |
| `launchers/explore_20261120/selfjudge_stack_chain_1122.sh` | 第 6 节逐个模型排卡 | 同上 | 5a84a2565e8978d7 |
| `launchers/explore_20261120/selfjudge_stack_pick_1122.sh` | 排期癸第 7 节：`selfjudge_stack_pick.py score` 一个模型一张卡 | 同上 | deb069dc2febaa95 |
| `launchers/explore_20261120/selfjudge_stack_pick_chain_1122.sh` | 第 7 节：先 export 再 score，逐个模型 | 同上 | 17b7cd9e2428a65f |
| `launchers/explore_20261120/selfjudge_stack_pick_small_1123.sh` | 第 7 节两个小模型的排卡（协调人 02:37） | 同上；服务器上没找到同名文件 | dbb6fb4e970a2681 |
| `launchers/explore_20261120/selfjudge_big_scores_1123.sh` | 排期癸第 6、7、7a 节：三个大模型的批大小 1 自判分在一张卡的一次占锁里做完 | 同上 | c1179b798b1c0f48 |
| `diag/fill_only2.py` | 排期 L 第 9 节（含 10.1 的三处修正）：同样条件的补空对照，补 EC 对"必须作答 + 如实句 + E、不加 C"的补答；`gencheck` 核对生成文件，`export` 写自判分的条目清单（不打分），其余用法只读出表 | 新写（Opus 5.5），探索，已运行；程序写成后未经 GPT 审 | 305edd3a9a5797ce |
| `diag/fill_only2_score.py` | 同上：`fill_only2.py export` 列出的回答的批大小 1 自判分（占卡的一步，原样调用 `selfjudge_two_answers.score1`） | 同上 | 9f4635af92dc69e2 |
| `launchers/explore_20261120/fillctl_gen.sh` | 排期 L 第 9 节：新的一遍 TE（必须作答 + 如实句 + E，力度 s_sel，只写题目），TruthfulQA；照 `n1s_model.sh` 的 SE 命令只改臂的前缀 | 原样存档（服务器 `logs/newsets_20261119/`，下同），探索；未经 GPT 审 | 9ba330c19c9eca77 |
| `launchers/explore_20261120/fillctl_chain.sh` | 同上：几个模型、几个阶段在一张卡上依次运行 `fillctl_gen.sh`，第一个失败即停 | 同上 | e370707b35023450 |
| `launchers/explore_20261120/fillctl_judge.sh` | 同上：TE 的 DeepSeek Pro 判分（非思考），判前并入一份旧的裁判缓存 | 同上 | 046d29412c7269c6 |
| `launchers/explore_20261120/fillctl_score.sh` | 同上：`fill_only2_score.py` 的占卡启动脚本 | 同上 | b8f30adfabdf7e54 |
| `scripts/newsets/falseqa_prep.py` | 排期 L 第 8、10.2、11 节：FalseQA（Hu 等，ACL 2023）取题建库。开发集 50 对取自 valid、留出集 300 对取自 test，种子 20261123，整对抽取，成对按行对齐；写题库（只在服务器）与题号文件 | 新写（Opus 5.5），探索，已运行；已请 GPT 审（`discussion/validation_dataset/2026-11-24_reply_evening_and_falseqa_request.md`），尚无回复 | 44aea1513ccd121c |
| `configs/newsets/falseqa_ids_20261123.json` | FalseQA 开发集与留出集的题号、成对关系与原文件行号（不含题目文字） | 同上 | 1a5c0df6b74c858e |
| `scripts/newsets/falseqa_judge_split.py` | 排期 L 第 11 节：把 700 题的判分输入按开发集 / 留出集拆开，使开发集能先判；其余内容原样保留 | 同上 | e1b95b481d471797 |
| `launchers/explore_20261120/falseqa_model.sh` | 排期 L 第 11 节：一个模型 22 臂的生成（经 `ns_h22.py`），带入口检查与写入记录核对 | 原样存档，探索，已运行；已请 GPT 审，尚无回复 | 70d4ef089fce6bb5 |
| `launchers/explore_20261120/falseqa_score.sh` | 同上：每条可能被交出的回答的批大小 1 自判分，一个模型一张卡 | 同上 | ccb95349ae8f71e9 |
| `diag/falseqa_score.py` | 同上：自判分占卡的一步（原样调用 `selfjudge_two_answers.score1`） | 同上 | 840d84d94ca72c93 |
| `launchers/explore_20261120/falseqa_judge.sh` | 同上：一个模型 22 臂的判分导出与 DeepSeek Pro 判分，一次一个集合（先开发集、后留出集），自带高峰时段的时钟保护，可续判 | 同上 | cac05b5e28dfd87a |
| `launchers/explore_20261120/falseqa_night_1124.sh` | 同上：11-24 夜里五个模型的生成与自判分排卡，只启动上面两个脚本 | 原样存档，探索（主会话写），已运行 | 4fdfe24adfe4f88f |
| `launchers/explore_20261120/falseqa_judge_night_1124.sh` | 同上：11-24 夜里五个模型的判分排队，只启动 `falseqa_judge.sh` | 同上 | e32767fe002510bf |
| `launchers/explore_20261120/falseqa_budget_guard_1124.sh` | FalseQA 判分的余额保护：每两分钟读一次余额，低于下限就停掉 FalseQA 的裁判进程（余额用尽后判分会静默卡住），停一次即结束 | 同上 | 22baba20940bc43b |
| `launchers/explore_20261120/caa_budget_guard_1124.sh` | CAA 判分的余额保护：每分钟读一次余额，低于下限期间每 20 秒停一次 CAA 的裁判进程，余额恢复后不再干预 | 同上 | 7dfbfdb794da018a |
| `launchers/explore_20261120/caa_after_1123.sh` | 排期 L 第 7 节第 1 项与 10.3：CAA 网格之后的后续启动（3 号卡 Qwen3.5-9B 整条流程、Qwen3.8-27B 验收与网格；0 号卡 Gemma-3-27B 网格结束后接整条流程），只启动审过的脚本、不改它们 | 原样存档，探索（Opus 5.5），运行中；未经 GPT 审 | 831ef8705ea0e3dc |
| `diag/falseqa_read.py` | 排期 L 第 11 节的读表程序：开发集 100 题上选六个上限，留出集读第 11 节的读数 1–5（分前提错误 / 前提成立 / 合计，分模型与五个模型合计，区间按对子重抽）；另有 `check`、`status`、`export-margins`、`check-margins`、`smoke`、`selftest` | 新写（Opus 5.5），探索；合成测试 18 项通过；2026-11-24 加一条规则"裁判拒判的格子记 0 分并在表头列出"（写于读留出集之前；改前为 d45879978304345a，开发集的输出逐字节未变）；已请 GPT 审，尚无回复；留出集的表还没有读 | 61737f2f7e650a9a |
| `diag/falseqa_dev_read.py` | 只读开发集 100 题的先看一眼：计算全部调用 `falseqa_read.py`，不打开留出集的判分目录 | 新写（Fable 5.2），探索，已运行（输出 `results/write_positions/falseqa_dev_first_look_20261124.txt`）；2026-11-24 加一行拒判格子的提示（改前为 ead7f1551165f98b，输出未变） | a7b181eaecacac5d |
| `diag/falseqa_table_dryrun.py` | 留出集读表的试跑：回答、留空与把握分都是真的，只把留出集每一格的判决换成假的（全部答对 / 全部答错 / 按题号与回答文字散列的伪随机），原样运行 `falseqa_read.py` 的 `run_table`；表的正文不输出，只报是否跑通与各项核对 | 新写（Fable 5.2），探索，已运行（五个模型、三种假判决全部通过） | 972f95bd45a4992b |
| `diag/n1_vs_single_e_tqa.py` | TruthfulQA 上分模型看 N1 减不带如实句的单 E、N1 减 S0（自然作答数与相同作答数）；读数程序原样复用，只读 | 新写（Fable 5.2），探索，已运行（排期 L 进度 2026-11-24 01:05） | ae810e2ebe53ad09 |
| `diag/n1_vs_single_e_qaqa.py` | 同上，(QA)² 留出 470 题，按题型分 | 同上 | d34c1f92acfeabf3 |
| `diag/c_row_layers.py` | 排期 M 第 1 节：自由回答 2×2 的 C 行，逐模型、逐层、逐臂的读数（只用 DeepSeek Pro）；只读 | 新写（Opus 5.5），探索，已运行；未经 GPT 审 | 00c607ba92e17703 |
| `diag/crow_pro_requests.py` | 同上：一批 Pro 判分之前数清要发多少请求（格数、不同的题与回答对数），不调接口 | 同上 | 05786796a7ac3479 |
| `diag/crow_sign.py` | 同上：某一层 C 写入的方向核对（往谨慎推不应比不写入答得更多），只读 | 同上 | d11b9ad23f6c3993 |
| `launchers/explore_20261120/crow_gen.sh` | 排期 M 第 1 节第一阶段：C 行逐层在调参集（192 题）上的生成 | 原样存档，探索（Opus 5.5），已运行；未经 GPT 审 | d3589ba6e001fd65 |
| `launchers/explore_20261120/crow_judge.sh` | 同上：C 行各臂的 DeepSeek Pro 判分（非思考），不占卡 | 同上 | cd1fdf78bed0a7ec |
| `launchers/explore_20261120/crow_night.sh` | 同上：11-24 夜里按卡排 `crow_gen.sh` | 同上 | 4b4bcf5357a4dabe |
| `launchers/explore_20261120/tfg3_model.sh` | 排期 M 第 2 节：两个 Gemma-3 的真假判断整条流程（E1 的 DAS 逐层训练、DAS 与均值差的扫描、对比、博士的三种方法表、C 的扫描、2×2），每 8 层一段占一次卡锁 | 原样存档，探索（Opus 5.5），运行中；未经 GPT 审 | 65150c9c3feec0dc |
| `diag/tfg3_tables.py` | 同上的读表程序：`smoke` 核对试跑，`tables` 出逐层表、三种方法表与 2×2；只读，只输出数字 | 新写（Opus 5.5），探索；未经 GPT 审 | 0416c732ff5ea2cf |
| `configs/capability/gsm8k_1319_20261119.json` | GSM8K 测试集 1,319 题的题号清单（不含题目文字） | 现行配置，补登记 | f8158017f1fcbdde |
| `configs/newsets/popqa_splits_20261119.json` | PopQA 的划分题号：拟合 300、调参 192、测试 500（其中 Pro 判的 100），种子 20261119 | 同上 | 992f2e1181eb50e2 |
| `configs/newsets/popqa_test2_20261120.json` | PopQA 第二批新题的题号：500 题（其中 Pro 判的 100），种子 20261120 | 同上 | 18ea733ad9bc0a37 |
| `keepfix_20261123/` | 排期 K（`discussion/work_queue/2026-11-23_queue_k.md`，训练侧加保持项，已停）的代码存档，共 10 个文件：`CHANGES.md` 说明改动；`scripts/e44_keepfix.py` 是加了保持项的 E 训练入口（开关默认关）；其余为保持集、读表程序、三个 Llama 启动脚本与模型配置 | 探索，已停；未经 GPT 审 | 见各文件 |

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-16 | 建立本文件，迁入机制探测脚本及其依赖 | 赵磊（Opus 5.5 代笔） |
| 2026-11-16 | 迁入重学 C 的改动所涉及的现行文件 | 赵磊（Opus 5.5 代笔） |
| 2026-11-17 | 更新 `diag/policy_select2.py`：加入"不写入""单独使用 C"两个对照与 Gemma-3-12B、Gemma-4-31B 两行，报告部分包在 main() 中 | 赵磊（Opus 5.5 代笔） |
| 2026-11-17 | 更新 `ff_ec_probe.py`（按 GPT 审核意见修改）；迁入 `doubt_signal.py`、`doubt_policy.py`、`probe_vs_doubt.py`、`ff_tune_pro_20261117.sh` | 赵磊（Opus 5.5 代笔） |
| 2026-11-17 | 更新 `diag/policy_select2.py`：Gemma 各行改由 Pro 判调参集选设置 | 赵磊（Opus 5.5 代笔） |
| 2026-11-17 | 更新 `diag/policy_select2.py`（新增 Gemma-3-27B 一行）；迁入离线检验的八个脚本：`c_first.py`、`silence_table.py`、`e_gain_split.py`、`method_split.py`、`law_lam1.py`、`survey_cp.py`、`doubt_levels.py`、`g27_prediction_check.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-17 | 迁入 `diag/gemma4_diagnosis.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-17 | 迁入 `diag/gemma4_random_dose.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-17 | 迁入 `diag/release_gate.py`、`diag/two_caps.py`、`diag/two_caps_vs_prompt.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-17 | 迁入 `diag/release_origin.py`、`diag/n1_vs_prompt.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | 迁入 `diag/n1_aux_allenai.py`、`diag/confirm_fresh.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | 迁入 `launchers/ff_g3_fresh_pro_offpeak_20261118.sh` | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | 迁入 `diag/prompt_compare.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | 迁入 `launchers/ff_xu1_stage1_20261118.sh` | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | 迁入 `diag/provenance_ablation.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | 迁入 `launchers/ff_xu1_stage1_after_20261118.sh` | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | `launchers/ff_xu1_stage1_after_20261118.sh` 改为第 2 版：只有 `sn_*` 步骤失败时照常判分（Gemma-4-12B 的两次运行都只在这一步失败过） | 赵磊（Fable 5.2 代笔） |
| 2026-11-18 | 迁入 `promptladder/` 五个文件（待 GPT 审核）与 `scripts/review_checks/e44_split_test.py`、`ff_test_subset_judge.sh`（供对照） | 赵磊（Opus 5.5 代笔） |
| 2026-11-18 | 迁入 `diag/b_vs_n1.py` | 赵磊（Opus 5.5 代笔） |
| 2026-11-18 | `promptladder/` 四个文件按 GPT 第二轮意见修改后重新迁入 | 赵磊（Opus 5.5 代笔） |
| 2026-11-18 | `diag/prompt_compare.py` 补入 Gemma-3-27B、Gemma-4-31B 与计数后重新迁入；迁入 `promptladder/ops/` 两个试跑文件 | 赵磊（Opus 5.5 代笔） |
| 2026-11-19 | 迁入 `diag/main_table_same_n.py`；提示词分档比较按当晚的决定暂停，`promptladder/` 各文件停在 2026-11-18 的版本 | 赵磊（Fable 5.2 代笔） |
| 2026-11-19 | 提示词分档比较由赵磊决定取消（除非博士要求）；`promptladder/` 保留作记录，不再维护。后续工作的排期见 `discussion/work_queue/2026-11-19_queue.md` | 赵磊（Fable 5.2 代笔） |
| 2026-11-19 | 迁入 `diag/final_records.py`、`diag/examples_for_dr.py`；`diag/b_vs_n1.py`、`diag/main_table_same_n.py` 改为读逐题最终记录后重新迁入（排期 A1、A2） | 赵磊（Opus 5.5 代笔） |
| 2026-11-19 | 迁入 `launchers/e45_newmodels_20261119.sh`（排期 B） | 赵磊（Opus 5.5 代笔） |
| 2026-11-19 | `launchers/e45_newmodels_20261119.sh` 的状态改为已运行完 | 赵磊（Opus 5.5 代笔） |
| 2026-11-19 | 迁入新题集试做与能力保持的程序、配置与题号清单（排期 C0、C1、F0），待 GPT 审 | 赵磊（Opus 5.5 代笔） |
| 2026-11-19 | 迁入 `scripts/newsets/newsets_gap.py`、`diag/newsets_score.py`、`diag/cap_score.py`（D2、C2、C3、F1 的读出与计分），待 GPT 审 | 赵磊（Opus 5.5 代笔） |
| 2026-11-19 | 按 GPT 审稿修改新题集与能力保持的八个程序，新增 `scripts/newsets/newsets_votes.py`；答复见 `discussion/validation_dataset/2026-11-19_response_to_gpt_review.md` | 赵磊（Opus 5.5 代笔） |
| 2026-11-19 | 迁入 `diag/d2_register.py`；能力保持 MMLU 的计分输出放进 `results/capability/mmlu/` | 赵磊（Opus 5.5 代笔） |
| 2026-11-20 | 迁入 `scripts/capability/cap_mcq_prob.py`；`newsets_gen.py` 与启动脚本的把握分补救；新题集 C2、C3 计分输出放进 `results/newsets/c2c3/` | 赵磊（Opus 5.5 代笔） |
| 2026-11-20 | 新题验证：迁入 `diag/newsets_val.py`、重选上限的配置与新题清单；切分、题库、生成程序与启动脚本相应扩展 | 赵磊（Opus 5.5 代笔） |
| 2026-11-20 | 新题验证结果放进 `results/newsets/val/`；迁入 `diag/newsets_tqa_shared_answers.py` | 赵磊（Opus 5.5 代笔） |
| 2026-11-20 | MMLU 找原因：迁入 `cap_mcq_sweep.py`、`diag/cap_mcq_flips.py`、`diag/dose_table.py`、`diag/cap_mcq_fresh.py`；结果在 `results/capability/dose/` 与 `results/capability/mmlu/` | 赵磊（Fable 5.2 代笔） |
| 2026-11-20 | `diag/dose_table.py` 补入提示词基线；总表换成 Fable 5.2 重做的版本 | 赵磊（Fable 5.2 代笔） |
| 2026-11-20 | 中间力度与分模型力度：迁入 `diag/dose_mid.py`、`diag/on_prompt_table.py`；`cap_mcq_sweep.py` 可叠在其他提示词上；当天的服务器启动脚本存档到 `launchers/explore_20261120/`；结果表放进 `results/capability/dose/` 与 `results/capability/mmlu/` | 赵磊（Fable 5.2 代笔） |
| 2026-11-20 | 晚间：迁入 `diag/md_table.py`、`diag/md_choose.py`、`diag/strong_mid.py`、`scripts/review_checks/e44_positions.py`；`cap_mcq_fresh.py`、`dose_mid.py` 小改；高力度、调参集、MMLU 全量与均值差对照的结果表放进 `results/capability/` | 赵磊（Fable 5.2 代笔） |
| 2026-11-20 | 夜间：迁入 `diag/prompt_sweep_table.py`、`diag/pos_table.py`、`diag/tf_net_score.py`；`md_table.py` 加第二部分；真假判断按净分重算的结果放进 `results/tf/` | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 凌晨：迁入 `diag/nsc_prompt_table.py`、`diag/posm_table.py`、`diag/gsm8k_n1_prompt.py`（Qwen3.8-27B 改用 1,024 词元的文件）；小试验结果放进 `results/newsets/prompt_rule/` | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 01 时：应用 GPT 的补丁 `discussion/validation_dataset/2026-11-22_night_fixes.patch`（`gsm8k_n1_prompt.py`、`ns_prompt_table.py`、`ns_positions.py` 与八个启动脚本的失败传递、标记、输入核对）；另改 `ns_prompt_table.py` 的选择与 `posm_table.py` 的核对。`launchers/explore_20261120/` 里的 `night_1121.sh`、`night2b_1121.sh`、`night3_1121.sh` 现在是补丁后的版本，今晚实际运行的文本见提交 0607e79 | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 01 时半：迁入 `scripts/newsets/ns_task.py`、`scripts/newsets/qaqa_bank.py`、`diag/qaqa_pilot_table.py` 与两个启动脚本；`results/newsets/qaqa/` 放小试验的题号与结果，`results/write_positions/posm_five_models_20261122.txt` 是五个模型的均值差对照 | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 02 时：`diag/qaqa_pilot_table.py` 加调参集选择与按选择读数两种用法；迁入 (QA)² 的五个启动脚本 | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 02 时 45 分：`diag/qaqa_pilot_table.py` 加按把握挑题的对照与 E 不动的读法；迁入把握分与终表的三个启动脚本 | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 03 时 40 分：`diag/qaqa_pilot_table.py` 按只读审查的意见加固（回答文本与判分绑定、未齐即失败、题号清单核对）；`qaqa_tuned_chain.sh` 的高峰时段判断改用北京时间 | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 05 时：GSM8K 收尾（`diag/gsm8k_n1_prompt.py` 加两行平均，`diag/gsm8k_main_table.py` 跳过再问的题号文件）；(QA)² 相同作答数的强对照：新增 `scripts/newsets/ns_conf_pos.py` 与两个启动脚本，`diag/qaqa_pilot_table.py` 加两行对照 | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 05 时半：新增外部对照 ITI 的程序 `scripts/newsets/iti_baseline.py` 与启动脚本 `iti_grid.sh`（探索） | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 16 时半：GPT 第三轮复审（`discussion/validation_dataset/2026-11-22_gpt_review_c.md`）。它的四个复算程序在服务器原样重跑，输出与它的逐字节相同（`results/gpt_replay_c_rerun_20261122/`）；三个补丁已应用：`diag/tq_same_n_decomp.py`（共同题上先取题再排）、`diag/qaqa_gate_fair.py`（标签）、`diag/iti_faithful_input_checks.py`（加严）装到服务器 `diag/`，`iti_faithful_g27_1122.sh` 的启动保护只进仓库。回归与答复见 `discussion/validation_dataset/2026-11-22_reply_c.md` | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 15 时一刻：`scripts/newsets/iti_baseline.py` 的导出检查补丁装到服务器（两个路径，sha 1e6c53d6eff1e8e3）；新增启动脚本 `iti_faithful_g27_1122.sh`，忠实版 ITI 在 Gemma-3-27B 上起跑（网格、选参规则、两种读法与前两个模型相同）；新增文献核对 `novelty/prior_work/2026-11-22_caa_method_spec.md`（CAA 原文与我们的同层均值差对照逐项对比） | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 15 时半（Opus 5.5）：新增 `diag/iti_faithful_input_checks.py`（忠实版 ITI 回答文件与判分输入的只读检查，80 个文件都满足）；`diag/tq_same_n_decomp.py`、`diag/qaqa_gate_fair.py` 的状态仍是未经 GPT 审 | 赵磊（Opus 5.5 代笔） |
| 2026-11-22 | 14 时：`diag/iti_table.py` 加“裁判拒判”的处理。起因：Llama 忠实版 ITI 不看预算的读法 13:56 停在出调参表那一步，程序报四个格点“还没判”；实际判分已完成，是这四格（48 个头的力度 25、30 与 64 个头的力度 25、30）输出成了乱码，裁判对其中 94 条回答在“有没有信息量”一维返回 D（拒判）。改法：这类格点单列、不参与选择，且只在它不可能影响选择时才出表。回归：此前五张调参表（第一版 96 题拟合的两种读法、288 题拟合、论文那种 σ、忠实版预算内）逐字相同；装上后重启失败的那一步，选中的仍是主读数的格点 | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 11 时半：应用 GPT 的两个补丁（`discussion/validation_dataset/2026-11-22_iti_input_checks.patch`：`diag/iti_table.py` 只在声明的网格里选、`scripts/newsets/iti_baseline.py` 导出时拒绝重复或错序的题号；`2026-11-22_qaqa_checks_b.patch`：`diag/qaqa_pilot_table.py` 的对照凑不够 N 条时标“不可比”等六处核对）。`qaqa_pilot_table.py` 另加一处小改：“每题都被迫作答”的两行只看两条臂是否齐全，不再跟着同作答数的那一格一起消失。回归：ITI 第一版四张表逐字相同；(QA)² 两张 470 题的表逐字节相同，570 题原样搬的表按补丁本意变了一格（Qwen3.8-27B 的不写入对照 568 对 569，标不可比）。服务器上 `iti_table.py`、`qaqa_pilot_table.py` 已装，`iti_baseline.py` 等忠实版 ITI 跑完再装。新增 `diag/tq_same_n_decomp.py`、`diag/qaqa_gate_fair.py` | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 10 时（Opus 5.5）：按只读审核改 `iti_baseline.py`（忠实拟合、原子写入）、`diag/iti_table.py`（三道检查）、`iti_free.sh`、`iti_chain.sh`、`iti_grid.sh`（日志名、`ITI_PROGRAM`）；新增 `diag/iti_period_check.py` 与两个启动脚本 | 赵磊（Opus 5.5 代笔） |
| 2026-11-22 | 09 时（Opus 5.5）：ITI 扩到 Gemma-3-12B（赵磊定：先扩一个模型）；新增启动脚本 `iti_free.sh`、`e_hi_llama.sh`；`diag/iti_table.py` 的 `point` 提速并加 `unjudged` | 赵磊（Opus 5.5 代笔） |
| 2026-11-22 | 08 时：ITI 补 `diag/iti_table.py`（预算表、调参集选择、测试集对比）、`diag/iti_gsm8k.py`、`iti_judge.sh`；`iti_baseline.py` 加 GSM8K 阶段；结果放 `results/baselines/iti/` | 赵磊（Fable 5.2 代笔） |
| 2026-11-22 | 复核时更新 ns_positions.py 的镜像摘要；与实验机现行文件相同 | 赵磊（GPT-6.2-Astra 代笔） |
| 2026-11-22 | 登记作答后自判与题型开关的七个程序（`diag/selfjudge_probe.py`、`selfjudge_equal_n.py`、`selfjudge_stage1.py`、`selfjudge_gate.py`、`selfjudge_factqa.py`、`selfjudge_gate_k3.py`、`factqa_nonanswer_diag.py`）与四个 `selfjudge_*.sh` 启动脚本；应用 GPT 第四、五轮的补丁（`discussion/validation_dataset/2026-11-22d_probe.patch`、`2026-11-22d_equal.patch`、`2026-11-22d_equal_stage1.patch`、`2026-11-22e_compat.patch`），K3 的启动脚本照同样写法加占卡锁 | 赵磊（Opus 5.5 代笔） |
| 2026-11-23 | 00 时：登记 2026-11-22 下午到夜里定稿的十八个文件（G4 与 G1–G3、如实句的几项对照、2×2、叠在如实句上的第一步、两条回答挑一条、CAA 的表与向量、配置）；仍在改的几个（`ns_h22.py`、`n1_on_sentence2.py`、`prompt_family.py`、`selfjudge_stack.py` 等）等这一批做完再登记 | 赵磊（Opus 5.5 代笔） |
| 2026-11-23 | 将四个自判脚本的摘要与本次核实的仓库和服务器现行版本对齐 | 赵磊（GPT-6.2-Astra 代笔） |
| 2026-11-23 | 01 时 59 分：应用 GPT 第七轮的 CAA 补丁 `discussion/validation_dataset/2026-11-23_replay/caa_checks.patch`（`diag/caa_table.py` 不再把 `L*.receipt.json` 当能力结果读——原先会在 Gemma-3-12B 的真实目录上报错；`caa_chain.sh`、`caa_judge.sh` 重验已有选择与来源、同模型逐层合并判分缓存），已装到服务器（摘要 73d454370960012a、d774bcddd9614261、23da524974e04fc7），预算表在已测的层上读得出 | 赵磊（Opus 5.5 代笔） |
| 2026-11-23 | 23 时 22 分（执行人丙）：登记此前未登记的 57 个文件：`diag/` 八个（`fill_only.py`、`n1_on_sentence2.py`、`n1_on_sentence3.py`、`prompt_family.py`、`selfjudge_stack.py`、`selfjudge_stack_pick.py`、`sentence_capability.py` 与新写的 `summary_table.py`）、`scripts/` 三个、`configs/prompts/e44_eval_extra_pfam_v1.json`、`launchers/explore_20261120/` 四十五个（每个一行，原有的目录条目不改）。摘要按仓库文件算，与服务器相同（服务器上没找到 `selfjudge_two_answers_chain_1122.sh`、`selfjudge_stack_pick_small_1123.sh` 两个同名文件）；今晚另三位执行人新建的 `fill_only2*`、`fillctl_*`、`caa_after_1123*`、`falseqa_*` 待其完成后再登记 | 赵磊（Opus 5.5 代笔） |
| 2026-11-24 | 08 时：登记 34 个文件与 `keepfix_20261123/` 目录（排期 L 第 9、11 节的补空对照与 FalseQA、两个余额保护、`caa_after_1123.sh`、分模型的 N1 对单 E 两个读数程序、排期 M 的 C 行逐层与两个 Gemma-3 的真假判断、三个题号配置）；新增 `diag/falseqa_table_dryrun.py`；`diag/falseqa_read.py`、`diag/falseqa_dev_read.py`、`diag/qaqa_pilot_table.py` 加裁判拒判格子的处理（留出集读表之前写定）；补上 2026-11-23 `caa_identity.patch` 后三个 CAA 程序没更新的摘要；`caa_night2_1122.sh` 在服务器上换成补丁后的版本；探索脚本目录的个数改为 176 | 赵磊（Fable 5.2 代笔） |
