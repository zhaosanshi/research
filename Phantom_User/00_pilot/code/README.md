# code

预研阶段用到的全部脚本。结论和每张表对应的脚本见 `../02_预研结论.md`。

## 公开版说明

- **服务器路径换成了环境变量。** 批处理脚本要求设置 `PHANTOM_ROOT`（本目录）和 `MODEL_ROOT`（放模型权重的目录），没设置会直接报错退出。容器镜像也改成环境变量：`IMG` 是 PyTorch 镜像，`IMG_VLLM` 是 DeepSeek-V4 用的 vLLM 镜像。
- **中文试点的材料换成了中性文本。** 最早的中文试点（`PROBE_LANG=zh`，也是脚本的默认值）里，对话材料仿照了我同学的一次私人会话。公开版把其中的专有内容换掉了：文件名、笔记内容、用户那句原话、污染样本里的时间戳。中文试点的原始结果文件也没有上传。所以实验记录第 1 到 11 节的数字，不能用这里的代码逐位复现。
- **论文设定完整公开。** 英文通用材料、不带包装、污染样本为纯文本，即 `PROBE_LANG=en CONTAM_STYLE=plain`。代码、材料、结果文件都在。德文、日文的材料在 `lang/`。
- 早期带包装的英文设定（`CONTAM_STYLE=wrapped`）里，污染样本的时间戳在公开版的代码和结果里统一改成了中性值，其余不变。

## 环境

- 4 张 RTX PRO 5000（每张 72 GB）。8B 级模型单卡，32B 双卡，70B 三到四卡。
- `IMG`：NVIDIA NGC 的 PyTorch 26.06 镜像，transformers 4.57。全部用 bf16 推理，不改权重。
- `IMG_VLLM`：DeepSeek-V4-Flash 需要 vLLM 的开发版，并且要有针对这代显卡的 DeepSeek-V4 内核。公开的 `vllm/vllm-openai:v0.27.1` 在这代显卡上跑不起来，原因见实验记录第 16 节。另需 `DSV4_ENCODING_DIR` 指向 DeepSeek 官方的 `encoding` 目录，默认是 `$MODEL_ROOT/deepseek-v4-flash/encoding`。

## 怎么跑

脚本假定当前目录下有 `results/`。从本仓库复现时，先在本目录建一个链接：

```bash
ln -s ../results results
export PHANTOM_ROOT=$(pwd)
export MODEL_ROOT=/path/to/models
export IMG=<PyTorch 镜像>
```

主表（Llama-3.1-8B，四个格子，5 个种子）：

```bash
PROBE_LANG=en CONTAM_STYLE=plain python turn_boundary_probe.py \
    --model $MODEL_ROOT/Llama-3.1-8B-Instruct \
    --filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl \
    --force-first user \
    --conditions X20k_c00 X20k_c03 X20k_m00 X20k_m03 \
    --filler-seed 1 --out results/my_cross.jsonl
```

`--filler-seed` 换成 1 到 5 各跑一次。完整的队列见 `job_en_plain.sh`。

参数说明：`--force-first user` 是喂词，续写长度由 `--force-first-len` 控制，默认 40 个 token。`--max-forced` 管的是另一个字段（禁止收声时模型自己写的回答），与喂词无关。

格子名的读法：`X20k_c03` 是"约 2 万 token、写代码内容、不带包装、最后 3 条回答被污染"。下划线后的第一个字母 `c` 是写代码，`m` 是日常求助；中间的数字 0 或 1 是带不带包装；最后的数字是被污染的回答条数。

## 脚本一览

| 脚本 | 做什么 |
|---|---|
| `turn_boundary_probe.py` | 核心脚本。搭对话、套模板、读槽位上第一个 token 的概率、喂词续写。环境变量：`PROBE_LANG`（zh / en / de / ja）、`CONTAM_STYLE`（wrapped / plain）、`CONTAM_ROLE_WORD`（污染样本里用哪个角色词） |
| `make_filler.py` | 让模型生成填充用的问答，把对话拉长 |
| `e2e_sample.py` | 不硬塞字的抽样：大量抽第一个字，抽中角色词的再接着写，外加完整回答 |
| `logit_lens.py` | 逐层解码，看 `user` 在每一层的概率和名次 |
| `head_attrib.py` | 把 `user` 的得分拆到每颗注意力头，重算注意力，做消融 |
| `induction_test.py` | 给每颗头打归纳分，用来确认复制头不是通用归纳头 |
| `contam_variants.py` | 改写污染样本（只留包装、只留角色词、换换行数、换角色词） |
| `joint_prob.py` | 整词概率：多 token 的角色词，把各 token 的概率连乘，并喂整个词去续写 |
| `dsv4_probe.py` | DeepSeek-V4 的探针，读全词表概率 |
| `special_token_norms.py` | 读权重文件，比较底座和指令版里回合头等专用符号的词向量大小（只用 CPU） |
| `render_example.py` | 把渲染后的对话原文打印出来，方便核对 |
| `compare_xmodel.py`、`compare_zh_en.py`、`summarize_*.py` | 汇总成表 |
| `verify_fix.py`、`gemma27_smoke.py` | 排查多卡推理乱码问题时用的小脚本，见已知问题第 7 条 |
| `lang/de.json`、`lang/ja.json` | 德文、日文的整套对话材料 |
| `job_*.sh` | 批处理队列。文件头写了每个队列跑什么 |

## 批处理和结论的对应

| 结论 | 队列 |
|---|---|
| 主表、抽样、长度、机制 | `job_en_plain.sh`、`job_en_plain_xmodel.sh` |
| 用户粘贴角色行、专用工具角色 | `job_en_plain_extra.sh`：原始运行是手动启动的，这个脚本是事后按结果文件里记录的参数重建的 |
| 跨模型 | `job_en_plain_xmodel.sh` |
| 非模板标签、整词概率 | `job_variants.sh`、`job_joint.sh` |
| 七种语言的角色词 | `job_rolewords_multi.sh` |
| 整场德文、日文 | `job_lang_full.sh`、`job_ja_more.sh` |
| 思考模式、用户粘贴、其他角色词 | `job_doorsearch.sh` |
| DeepSeek-V4-Flash | `job_dsv4_alt.sh`、`job_dsv4_more.sh` |
| 早期带包装的英文设定 | `job_en.sh`、`job_en_xmodel.sh`、`job_en_xmodel_part2.sh` |
| 早期中文试点 | `job_overnight.sh`、`job_followup.sh`、`job_g3.sh`、`job_70b*.sh`、`job_big2.sh`、`job_lens.sh`、`job_heads.sh`、`job_lang2x2.sh`、`job_roleword.sh` |

填充对话的来源：英文的 `filler_en_coding.jsonl`、`filler_en_general.jsonl` 由 Llama-3.1-8B 生成，跨模型实验所有模型都用这两份；日文的 `filler_ja_general.jsonl` 由 Qwen3-8B 生成，日文实验所有模型都用它；`filler_en_general_<模型>.jsonl` 是各模型自己生成的，只用在早期带包装设定的对照里。

## 已知问题

1. **探针只记录前五个角色类词。** `p_role_exact` 字段只含概率最高的五个角色类词。`user` 没进前五时，这个字段里没有它，不等于概率为零。`compare_xmodel.py` 把缺失当成 0 处理，会产生误导。精确值可以从 `e2e_sample.py` 的 `p_user` 字段或 `joint_prob.py` 的输出里读。
2. **多 token 的角色词只喂了第一个 token。** `force_first_continuations` 只把词的第一个 token 接上，其余由模型自己写。模型接成别的词时，那条续写不算对这个词的测试。`joint_prob.py` 喂的是整个词。
3. **自动判定"是不是假用户发言"偏宽。** `compare_xmodel.py` 和 `e2e_sample.py` 里的判定只看"角色词单独成行"，后面接的是助手自己的回答也会被算进去，助手回合头漏出来也会触发。`../02_预研结论.md` 里的计数是逐条读过之后按 A、B、C 三类重数的。
4. **`contam_variants.py` 的污染句写死成了中文。** 在英文设定下跑这个脚本，污染句仍是中文那一句。组内对比的方向不受影响，绝对值需要改成英文句后重跑。
5. **两个环境变量的默认值指向不公开的中文试点设定**：`PROBE_LANG` 默认 zh，`CONTAM_STYLE` 默认 wrapped。跑论文设定时要显式设置。
6. **Gemma-3-4B 在 64k 上下文、单卡下输出退化。** 原因未查明，相关结果已作废，没有上传。
7. **多卡推理的一个坑。** 所用镜像里 PyTorch 的一个加速路径在多卡时会读错内存，导致长上下文输出乱码。`turn_boundary_probe.py` 的 `load_model` 里已经关掉。用别的环境时不需要这一步。
8. **概率是从半精度的 logit 算的。** 会出现不同 token 概率完全相等的情况，报数最多保留两位有效数字。
