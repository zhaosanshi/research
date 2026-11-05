# results

预研阶段公开的原始结果，共 133 个文件、约 11 MB。除两个汇总文本和一个词向量检查结果外，都是每行一条记录的 JSON。结论见 `../02_预研结论.md`。

## 没有上传的

- **中文试点的全部原始结果。** 对话材料仿照了我同学的一次私人会话，见 `../code/README.md`。
- **逐头归因的原始文件**，每个约 7 MB。上传的是汇总文本 `summary_heads_en_plain.txt` 和 `summary_lens_en_plain.txt`。
- **污染样本改写的两个英文结果文件。** 污染句用错了语言，见 `../code/README.md` 的已知问题第 4 条。
- **Gemma-3-4B 的 64k 上下文结果。** 输出退化，已作废。
- 运行日志、其他作废的结果。

记录里的模型路径只保留了模型目录名。早期带包装设定的记录里，污染样本的时间戳改成了中性值。

## 文件名的读法

格子名：`X20k_c03` 是"约 2 万 token、写代码内容、不带包装、最后 3 条回答被污染"。下划线后的 `c` 是写代码，`m` 是日常求助；中间的数字是带不带包装；最后的数字是被污染的回答条数。抽样文件名里的 `m03`、`c13` 是同一套写法。

| 文件 | 内容 | 设定 |
|---|---|---|
| `en_plain_cross_Llama-3.1-8B-Instruct.jsonl` | 主表：四个格子，5 个种子 | 论文设定 |
| `en_plain_e2e_{m00,m03,c00,c03}.jsonl` | 抽样：每格 5 段对话，每段抽 2 万次第一个字，外加 32 条完整回答 | 论文设定 |
| `en_plain_e2e_x_<模型>_<格子>.jsonl` | 同上，其他模型每段 2 万次。Llama-3.1-8B 的两个文件是每段 100 万次，抽中角色词的接着写 | 论文设定 |
| `en_plain_xmodel_<模型>.jsonl` | 跨模型探针，另外八个模型。Llama-3.1-8B 在上面的主表文件里 | 论文设定 |
| `en_plain_extra_Llama-3.1-8B-Instruct.jsonl` | 用户粘贴角色行（`U20k_*`）、专用工具角色（`T20k_*`） | 论文设定 |
| `en_plain_length_Llama-3.1-8B-Instruct.jsonl` | 上下文长度 8k / 32k / 64k | 论文设定 |
| `en_plain_logit_lens_*.jsonl` | 逐层解码，指令版和底座 | 论文设定 |
| `summary_heads_en_plain.txt`、`summary_lens_en_plain.txt` | 注意力头归因与消融、逐层解码的汇总。标题行的 P(user) 是算术平均，消融那几行是几何平均 | 论文设定 |
| `special_token_norms.txt` | 底座和指令版里回合头等专用符号的词向量大小 | 与对话无关 |
| `induction_*.json` | 每颗头的归纳分 | 与对话无关 |
| `variants_<模型>.jsonl` | 角色词变体，Qwen3-8B 51 个，Gemma-3-4B 44 个 | 论文设定 |
| `joint_{en,de,ja}_<模型>.jsonl` | 整词概率和喂整个词之后的续写 | 论文设定 |
| `rolewords_<语言>_<模型>.jsonl` | 英文对话，污染样本里的角色词换成七种语言的词 | 论文设定 |
| `lang_{de,ja}_user_<模型>.jsonl` | 整场德文或日文对话，角色词仍是 `user` | 论文设定 |
| `lang_{de,ja}_native_<模型>.jsonl` | 整场德文或日文对话，角色词用本语言的词 | 论文设定 |
| `doorsearch_<模型>_<项目>.jsonl` | 思考模式、用户粘贴、`Human`、`USER`，以及 Qwen3-8B 的 64k | 论文设定 |
| `dsv4_probe*.jsonl` | DeepSeek-V4-Flash：英文四格、专用 token 形式的污染、非模板标签、整场日文 | 论文设定 |
| `filler_en_*.jsonl`、`filler_de_general.jsonl`、`filler_ja_general.jsonl` | 填充用的问答。来源见 `../code/README.md` | |
| `en_cross_*`、`en_controls_*`、`en_length_*`、`en_logit_lens_*`、`en_selftalk_cross_*` | Llama-3.1-8B 的主表、对照组、长度、逐层解码、"AI 谈论自我"内容 | 早期设定：英文、带包装 |
| `en_e2e_{m10,m13,c10,c13}.jsonl`、`en_e2e_x_<模型>_<格子>.jsonl` | 抽样 | 早期设定：英文、带包装 |
| `en_xmodel_<模型>.jsonl`、`en_xmodel_ownvoice_*`、`en_xmodel_forceUser_*` | 跨模型探针、各模型用自己生成的填充、喂大写 `User` | 早期设定：英文、带包装 |

## 字段

探针记录（`turn_boundary_probe.py` 写出）：

| 字段 | 含义 |
|---|---|
| `condition`、`desc`、`kwargs` | 格子名、说明、搭对话用的参数 |
| `filler_seed` | 填充问答的打乱顺序，即"种子" |
| `n_ctx_tokens` | 上下文长度 |
| `top10` | 槽位上概率最高的 10 个 token |
| `p_role` | 所有角色类词的概率之和 |
| `top_role`、`p_role_exact` | 概率最高的五个角色类词，前者四舍五入，后者是精确值。**只有前五个。** `user` 不在里面时，只知道它低于第五名，不是零 |
| `p_contam_word` | 污染样本里用的角色词在槽位上的首 token 概率。多 token 的词会偏高 |
| `force_first` | 喂词续写：键是喂进去的词，值是模型接着写出的 40 个 token。多 token 的词只喂了第一个 token |
| `forced`、`unforced` | 禁止收声和不禁止收声时模型自己写的回答 |
| `rendered_tail` | 渲染后对话的结尾一段，用来核对 |

抽样记录（`e2e_sample.py` 写出）：`p_user` 是小写 `user` 在槽位上的精确概率；`n_first` 是每段对话抽第一个字的次数；`first_hits`、`first_hits_total` 是抽中角色类词的次数；`hit_continuations` 是抽中之后让模型接着写的文本，每段每个词最多 3 条；`full_samples`、`full_fake` 是完整回答及其中被自动判为假用户发言的条数。自动判定偏宽，见 `../code/README.md`。

整词概率记录（`joint_prob.py` 写出）：`word`、`tokens`、`p_first`（首 token 概率）、`p_joint`（整词概率）、`continuation`（喂整个词之后的续写）。

DeepSeek 记录（`dsv4_probe.py` 写出）：`p_user_sp` 是专用角色 token 的概率，`rank_user_sp` 是它在全词表里的名次，`p_words` 是 `user`、`User` 等普通词的概率，`force_first` 是喂词续写，`logprob_mass_check` 是全词表概率之和，用来确认读数完整。
