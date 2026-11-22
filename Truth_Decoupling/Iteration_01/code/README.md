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
| `scripts/newsets/ns_positions.py` | 探索：让 `newsets_gen.py` 把 E 写在另一种位置（核心里已有的 prompt = 只写题目与助手开头、生成的词都不写；gen = 从最后一个提示词元起写）。`newsets_gen.py` 原样运行，内存里三处改动（引擎的写入位置、`plan_arm` 的位置检查、输出文件名加“-at-位置”）；只许单个写入；23 时加两个环境开关：`NS_POS_TWO_WRITE=1` 放开两个写入（事实题的 N1 再问要用），`NS_POS_PLAIN_NAMES=1` 在专用目录 `gen_<位置>` 里用原文件名，让现成的表格程序能读 | 新写，未经 GPT 审；Llama 两批 GSM8K 试跑核对过：文件里记的位置是 prompt、回答与不写入不同 | 247ad845addde168 |
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
| `diag/qaqa_pilot_table.py` | 探索：(QA)² 的表（只写题目，Pro 判；原模型、纯提示词、只推 C、只推 E、N1，按题型分开；容器里运行）。三种用法：默认 = 全部设置沿用 TruthfulQA（100 题小试验或 570 题整套，可多个模型合计）；`--tune` = 把 100 题当调参集，列出每档 E 力度、每档 C、每对上限的净分并选定（`--write-choice`）；`--choice` = 用调参集选定的设置读其余题（`--skip-ids`），`--list-arms` 列出整套题上还缺的臂；`--tune --fix-e` = E 用主表力度、只选上限；有把握分文件时自动加一行“纯提示词按把握留同样多题”的对照；有强对照的把握分文件时再加“必须作答 + E、按把握留下与 N1 同样多”和“必须作答、不写入、留同样多”两行（要在 `--work` 里给出它们的判分目录，否则只在标准错误里提示并略过） | 新写，未经 GPT 审；2026-11-22 子代理只读审过（数字可信），按其意见加了三道硬检查，改后四张表逐字相同；05 时加强对照两行，加之前先核过原来四张表逐字不变 | a4a518c5c3173762 |
| `scripts/newsets/ns_conf_pos.py` | 探索：让 `newsets_gen.py` 的把握打分（文件不动）用于“写入只加在题目上”生成的回答。内存里三处：引擎报告的写入规则换成 prompt、`run_conf` 里要求 content 的那一处判断换成 prompt（必须恰好匹配一处）、打分时取写入掩码的函数在被要求 content 时返回 prompt 的掩码；回答文件必须记着 prompt 规则，输出文件名必须以 `-at-prompt.json` 结尾。98% 复现检查与“首个不同词元”检查照旧。新任务名按 `ns_task.py` 的办法登记 | 新写，未经 GPT 审；Llama 上跑通：复现 0.989，481 道不同的回答全部朝实际回答方向 | 087b2c59509aaf88 |
| `scripts/newsets/iti_baseline.py` | 探索：外部对照 ITI（Li 等 NeurIPS 2023）放进本项目自己的评测流程（同一个模型加载与对话渲染、同样的 48 词元贪心生成、同一份裁判输入）。五个阶段：`fit`（每个注意力头的探针、均值差方向与两种 σ）、`check`（自检：不加干预逐词元复现原回答；α = 0 不变；加在头上与加在注意力块输出上两种写法的首词元概率一致）、`gen`（调参集或测试集的回答）、`mmlu`（MMLU 选项概率）、`export`（裁判输入，题目与参考答案照抄同一模型同一批题的现成裁判输入）。做法逐项对照 `novelty/prior_work/2026-11-22_iti_method_spec.md` | 新写，未经 GPT 审；Llama 上自检通过（16 / 16 逐词元复现）；2026-11-22 加 `gsm8k` 阶段（不加干预时 32 / 32 逐词元复现平时的那一遍） | 092970ca5a37376f |
| `diag/iti_table.py` | 探索：ITI 对照的三张表。`budget` = 各格点在 MMLU 全量上的变化（选择半 / 报告半，与主表同一条预算线；先核不加干预的一遍与平时的不写入文件逐题一致）；`tune` = 预算内各格点在调参集上的净分并选定（`--write-choice`）；`test` = 选定格点在测试集上对原模型、只推 E、N1 的成对差；`point` 供启动脚本提前停用；`--all` = 不看预算、整个网格里取调参集最好的格点（看过头三个读数后加的读法） | 新写，未经 GPT 审 | c45d1e32fefb69e1 |
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
| `launchers/explore_20261120/` | 2026-11-20 探索用的服务器启动脚本原样存档（106 个，含 `iti_grid.sh`（ITI 对照：各格点的 MMLU 全量与调参集回答，不调裁判；某个 K 超预算太多时跳过更大的力度）、`iti_judge.sh`（ITI 回答的判分：题目与参考答案照抄现成裁判输入，先并入那份输入的裁判缓存）；含 `qaqa_strong.sh` / `qaqa_strong_1122.sh`（(QA)² 上相同作答数的强对照：必须作答 + 只写题目的 E 与必须作答不写入两遍的生成、判分与把握分）；含 `qaqa_conf.sh` / `qaqa_conf_1122.sh`（(QA)² 上纯提示词回答的把握分）、`qaqa_tables_1122.sh`（四张终表）；含 `qaqa_run.sh` / `qaqa_full_1122.sh`（(QA)² 五个模型 × 570 题，设置沿用 TruthfulQA）、`qaqa_tune.sh` / `qaqa_tune_1122.sh`（100 题当调参集，各档力度）、`qaqa_tuned_chain.sh`（调参集选定 → 补生成 → 出两种读法的表）；含 `qaqa_pilot.sh`（(QA)² 小试验：生成、导出、Pro 判）、`posm_rest_1122.sh`（其余四个模型的均值差对照）；含 `nsc_prompt.sh`（事实题只写题目下 C 往谨慎的小试验：调参集整格、选完只补选中的臂）、`night2b_1121.sh`（取代 `night2_1121.sh`：均值差对照先只做 Llama）、`night3_1121.sh`（小试验扩到其余三个模型）；含 `night2_1121.sh`（第二条夜间队列）、`gsm8k_n1_pos.sh`、`posm.sh`（同层同写入量的均值差方向，只写题目）；含 `ns_prompt.sh`（事实题只写题目）、`night_1121.sh`（11-21 夜间队列）；含 `gsm8k_pos_27b.sh`（两个 27B 的 GSM8K 只写题目）；其中 `pospall.sh`、`posp_batch.sh` 的说明文字 13:2x 按自查改过，批次跑完后（22 时）已同步到服务器；含 `posd.sh`、`posd_batch.sh`（只写题目下的强对照）；含 `pospall.sh`（某模型某层“只写题目”下主表要的整套）、`posp_batch.sh`（五个模型的排队）、`gsm8k_pos_q9.sh`；含 `posp2.sh`（N1 只写题目位置）、`g27_mm28.sh`（Gemma-3-27B 那组 MMLU 换到 2 号卡重跑）；含 `gsm8k_pos.sh`（GSM8K 换写入位置）、`posp.sh`（TruthfulQA 只写题目位置的只推 E）；含 `g27_gpu0.sh`（Gemma-3-27B：第 28 层叠最好提示各臂的 MMLU，再接 GSM8K）；含 `gsm8k_cap.sh`（GSM8K 换生成上限）、`prompt_conf.sh`（各条提示无写入回答的把握）、`lane27f.sh`（27B 队列当前版本）；含 `gsm8k_27b.sh`（两个 27B 的 GSM8K，排在各自占卡的任务之后）；含 `lane27e.sh`（27B 统一队列的当前版本；一步因显存不足失败后按“同卡不放两个大任务”重排）；含 `gsm8k_main.sh`（主表设置下的 GSM8K）；含 `oncaut_mm_l24.sh`；含 `lane27d.sh`（27B 统一队列的当前版本：Gemma-3-27B 的“叠在最好提示上”改在第 28 层做）；含 `alt_strong.sh`（换层后的同力度强对照与同层均值差对照，通用版）、`oncaut_mm_l16.sh`、`lane27c.sh`（27B 统一队列的当前版本）；含 `alt_chain.sh` 与 `g12_l24_launch.sh`（换层整套的通用版，Gemma-3-12B 第 24 层用它）、`g12map.sh`（Gemma-3-12B 的 MMLU 分层图）、`oncaut_mm_rest0.sh`、`oncaut_alt.sh` 与生成它的 `build_oncaut_alt.py`（叠在最好提示上的 N1，E 写在另一层）；含 `g27_l28.sh`（Gemma-3-27B 第 28 层的整套）与 `lane27b.sh`（27B 统一队列的当前版本，替换 `lane27.sh`）；含 `oncaut_mm_rest.sh`（Llama、Qwen3.5-9B 叠在最好提示上各档的 MMLU）；含 `l16_mdg.sh` 与 `l16_mdg_launch.sh`（第 16 层同层的均值差对照）；含 `l16_strong.sh` 与 `l16_strong_launch.sh`（第 16 层的同力度强对照）、`md27.sh`（两个 27B 的均值差自己挑层，分成两段占卡）、`oncaut2_g27b.sh`、`lane27.sh`（两个 27B 模型的统一队列，替换 ns2c / ns2d / ns2e）；含 `low_strong.sh`（低力度的同力度强对照）与 `ns2e_27b.sh`（27B 支线的当前版本）；含 `pressure_pilot.sh`、`pressure2.sh`、`pressure2_launch.sh`（题后加一句用户看法的试验）与 `oncaut_mm_g12.sh`（叠在最好提示上各档的 MMLU）；含 `l16b_all.sh`（第 16 层那条链改成判分与 MMLU 同时进行）与 `ns2d_27b.sh`（27B 支线前加 Gemma-3-27B 的 MMLU 分层图）；另有 01:2x 之后新增的 `ns2c_27b.sh`（两个 27B 模型的短清单）、`ns4_test2.sh` 与 `ns4_launch.sh`（第二批测试题上的确认）、`md_q9_fix.sh`（一步因撞卡失败的重跑）、`oncaut2_g27.sh`（改了起跑条件的 Gemma-3-27B 支线）；新增 Llama 第 16 层 `l16_all.sh`、叠在最好提示上的 `oncaut_all.sh` 与省判分的第二版 `oncaut2_all.sh`、新题集上 C 取谨慎方向的 `ns_caut.sh`、新题集各档力度的精简版 `ns2_all.sh` 与改排显卡并补测试集再问回答的 `ns2b_all.sh`；另含含 Llama 换层加力度的试探 `layer_probe.sh`、训练题加入 MMLU 题的试验 `keep_pilot.sh`、作答位置写入第二轮 `pos2_all.sh` 与 `pos2_judge.sh`、Llama 与 Gemma-3-27B 的低力度 `low_all.sh`、新题集换力度 `ns_dose_all.sh`；晚间新增 Gemma-3 重跑 `pm3_gemma.sh`、只在作答位置写入 `pos_all.sh` 与 `pos_judge.sh`、均值差选层 `md_sweep.sh`、提示词基线全判 `prompt_sweep.sh`）：新题验证 `val_all.sh`；MMLU 找原因与全量确认 `sweep_all.sh`、`sweep_g27.sh`、`full_all.sh`；中间力度 `dose_mid_all.sh`、`dose_mid_judge.sh`；提示词之上再写 E `idea_a.sh`、`idea_a2.sh`；高力度 `q9_high.sh`；补判 `extra_judge.sh`、`extra_judge2.sh`；下午两条队列 `pm_all.sh`、`pm_judge.sh`、`pm2_all.sh`、`pm2_judge.sh` | 均未经 GPT 审（探索阶段先跑，成批送审）；生成、导出、判分调用的都是已审程序，这些脚本只改臂、切分与名称 | 见各文件 |

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
| 2026-11-22 | 08 时：ITI 补 `diag/iti_table.py`（预算表、调参集选择、测试集对比）、`diag/iti_gsm8k.py`、`iti_judge.sh`；`iti_baseline.py` 加 GSM8K 阶段；结果放 `results/baselines/iti/` | 赵磊（Fable 5.2 代笔） |
