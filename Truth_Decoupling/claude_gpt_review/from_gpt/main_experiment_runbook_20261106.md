# E26主实验与E27生成适配：运行说明

2026-11-06。面向Llama-3.1-8B、Qwen3-8B，运行真假陈述判断与真实问答两个任务。**E26完整主表尚未完成，本文不报告方法胜出或DAS突破。** 两模型E26／E27三题线路已在服务器退出0；正式日志与验证清单由本轮交付证据记录。E27已完成两模型、两档力度、三种初始化的调参；Qwen固定力度3后完成529题测试，另完成可选结束符的调参对照。结果和限制见[第18轮交付](review_20261106_18.md)。测试阶段仍是独立命令。

## 1. 表格到底比较什么

DAS（分布式对齐搜索 · 交换内部表示学习干预方向）与mean-diff（均值差 · 正确和错误样本内部表示的平均差）都保留。每种方向方法分别比较E、C、E＋C；原模型有承诺分数、简单分数、固定随机排序三个对照。E＋简单筛题、C＋简单筛题也各自独立调参，不继承另一评分方式选出的配置。

- **真假陈述：**输出True或False，按给定标签算对错；承诺分数是另一提示中首词Yes减No的原始输出分数。简单筛题使用真假判断的对数概率差绝对值。
- **真实问答：**自由生成答案，再由语义裁判判为正确、错误或未作答。承诺分数仍只是筛题依据；简单筛题用原模型对当前生成答案的平均词元对数概率，不取参考答案作为输入。
- **实际交付配额：**默认比较25%、50%、75%。对固定N题保留`floor(q×N)`条实质回答，其余实际交付固定拒答文本。自然拒答不能填配额；可用回答不足就标`infeasible`（不可行 · 无法回答要求的数量），不删除难题、更换分母或勉强补齐。
- **固定题目集合：**延用源运行的估计／调参／测试划分。真实问答主表仅含固定的实质参考题；只有拒答参考的题在原始判分中保留标记，所有方法统一排除这些题。任何未决判分都会阻止选择与报告。
- **区间：**bootstrap（自举重采样 · 反复抽题看结果波动）按问题成组抽取，真假陈述同题的一真一假一起抽；所有方法共用抽样，每次重新选择同一配额。不能填满的抽样明确记录，不只留下成功抽样。
- **另一张冻结门槛表：**只用调参分数确定门槛，测试时原样应用并报告实际作答率。它不保证测试恰好达到25%／50%／75%，与批量精确配额表分开解释。

当前C的具体定义是：在估计提示上计算首词Yes−No分数与隐藏状态的协方差方向，再投影掉**同一层**E的分量；强度按该方向的估计投影标准差缩放。这是同层正交化，不是跨层正交化，也不保证功能完全独立。E／C／E＋C在每个目标作答率下各自用调参题选择，测试只运行冻结配置。

## 2. 服务器环境与输入路径

以下命令在已具备模型、数据与裁判配置的实验容器内运行。每次只选一个模型，两个模型分别执行同一套步骤。没有凭证写在本文中。

```bash
set -euo pipefail

export TD_PROJECT_ROOT="${TD_PROJECT_ROOT:-/workspace/Truth_Decoupling}"
export CODE_DIR="${CODE_DIR:-$TD_PROJECT_ROOT/review_e26_release_20261106/claude_gpt_review/from_claude/code}"
export TD_DATA_DIR="${TD_DATA_DIR:-$TD_PROJECT_ROOT/code/data/processed}"
export TD_MODEL_ROOT="${TD_MODEL_ROOT:-$TD_PROJECT_ROOT/models}"
export RESULTS_ROOT="${RESULTS_ROOT:-$TD_PROJECT_ROOT/results_e26_codex_20261106}"
export JUDGE_CACHE="${JUDGE_CACHE:-$RESULTS_ROOT/judge_cache}"

set_model_paths() {
  export MODEL="$1"
  export QA_SOURCE="$TD_PROJECT_ROOT/results_qa_das_v1u/$MODEL/full"
  export TF_DAS_DIR="$TD_PROJECT_ROOT/results_das_tfce/E1/$MODEL"
  export TF_DAS_SWEEP="$TD_PROJECT_ROOT/results_das_tfce/E23/$MODEL/knob_sweep_das_tfce_full.json"
  export TF_MD_SWEEP="$TD_PROJECT_ROOT/results_das_tf/E23/$MODEL/knob_sweep_meandiff_full.json"
  export RUN="$RESULTS_ROOT/$MODEL/full"
  export ADAPT_ROOT="$TD_PROJECT_ROOT/results_e27_search_codex_20261106/$MODEL/ce_alpha1_ep3"
}

set_model_paths llama3.1-8b
# 切换第二个模型时执行：set_model_paths qwen3-8b
# 如需改源或输出，在此覆盖对应变量；所有后续命令均使用这些变量。

cd "$CODE_DIR"
test -f scripts/e26_matched_coverage.py
test -f scripts/e27_qa_generation.py
test -f "$QA_SOURCE/bank.json"
test -f "$QA_SOURCE/bank.npz"
test -f "$QA_SOURCE/training.json"
test -f "$QA_SOURCE/questions.json"
test -f "$QA_SOURCE/shortlist.json"
test -f "$QA_SOURCE/selection.json"
test -f "$TF_DAS_DIR/summary.json"
test -f "$TF_DAS_SWEEP"
test -f "${TF_DAS_SWEEP%.json}.npz"
test -f "$TF_MD_SWEEP"
test -f "${TF_MD_SWEEP%.json}.npz"
```

`QA_SOURCE`必须是已完成方向训练与代理入围的完整E25运行，不能指向旧三题试点。E26的`--pilot 3`只缩小本次线路检查的题目数，不把完整源bank替换成三题源。缺任一输入文件就先定位原有产物，不重造假目录或改用另一模型的文件。

E26默认配置为`configs/experiments/E26_matched_coverage.json`：每种E最多3个不同层候选，C最多1个估计层、两个正负强度，实际候选数由可用方向决定。模型的设备／精度等设置继续取源模型配置；GPU可通过容器已有的`CUDA_VISIBLE_DEVICES`分配。

## 3. E26三题线路复跑

这是小样本执行检查，不能据此判断效果。输出用独立目录；`prepare`已存在时只会核验相同请求，不会覆盖不同配置。

```bash
export PILOT_RUN="${RUN}_pilot3"

python3 scripts/e26_matched_coverage.py prepare \
  --model "$MODEL" --run-dir "$PILOT_RUN" \
  --qa-run "$QA_SOURCE" --tf-das-dir "$TF_DAS_DIR" \
  --tf-das-sweep "$TF_DAS_SWEEP" --tf-md-sweep "$TF_MD_SWEEP" \
  --pilot 3

python3 scripts/e26_matched_coverage.py collect-tune \
  --model "$MODEL" --run-dir "$PILOT_RUN"
python3 scripts/e26_matched_coverage.py judge-tune \
  --model "$MODEL" --run-dir "$PILOT_RUN" --judge-exp E4_qa --cache-dir "$JUDGE_CACHE"
python3 scripts/e26_matched_coverage.py freeze \
  --model "$MODEL" --run-dir "$PILOT_RUN"
python3 scripts/e26_matched_coverage.py collect-test \
  --model "$MODEL" --run-dir "$PILOT_RUN"
python3 scripts/e26_matched_coverage.py judge-test \
  --model "$MODEL" --run-dir "$PILOT_RUN" --judge-exp E4_qa --cache-dir "$JUDGE_CACHE"
python3 scripts/e26_matched_coverage.py report \
  --model "$MODEL" --run-dir "$PILOT_RUN"
```

三题25%配额可能取整为0，某些方法也可能因自然拒答无法填满；程序应如实标注，准确率未定义时不能用零代替。看日志里的逐题原输出，并核`report.md`和`delivered_answers.json`确有真实输出／拒答文本。

## 4. E26完整主实验

### 4.1 估计、调参、冻结

```bash
python3 scripts/e26_matched_coverage.py prepare \
  --model "$MODEL" --run-dir "$RUN" \
  --qa-run "$QA_SOURCE" --tf-das-dir "$TF_DAS_DIR" \
  --tf-das-sweep "$TF_DAS_SWEEP" --tf-md-sweep "$TF_MD_SWEEP"

python3 scripts/e26_matched_coverage.py collect-tune \
  --model "$MODEL" --run-dir "$RUN"
python3 scripts/e26_matched_coverage.py judge-tune \
  --model "$MODEL" --run-dir "$RUN" --judge-exp E4_qa --cache-dir "$JUDGE_CACHE"
python3 scripts/e26_matched_coverage.py freeze \
  --model "$MODEL" --run-dir "$RUN"
```

这一步只用调参题选配置。`selection.json`绑定来源数据、方向和评分文件；之后改动这些输入会被拒绝。已启动相同目录的运行应续接现有进程，不并发启动第二份写同一目录的任务。

### 4.2 独立执行测试并生成报告

```bash
python3 scripts/e26_matched_coverage.py collect-test \
  --model "$MODEL" --run-dir "$RUN"
python3 scripts/e26_matched_coverage.py judge-test \
  --model "$MODEL" --run-dir "$RUN" --judge-exp E4_qa --cache-dir "$JUDGE_CACHE"
python3 scripts/e26_matched_coverage.py report \
  --model "$MODEL" --run-dir "$RUN"
```

主产物为`report.md`、`report.json`、`delivered_answers.json`，原始输出和评分分别保留在`tf_{tune,test}.json`、`qa_{tune,test}.json`和`qa_{tune,test}_scored.json`。报告包含精确配额结果、组合减E／C／E_simple／C_simple／原模型简单筛题的直接配对差，以及冻结门槛下的实际测试作答率。

## 5. E27：让方向直接学习回答内容

这条方法称“DAS初始化的生成适配”：从既有方向起步，以完整正确答案的CE（交叉熵 · 提高正确答案各词的概率）训练同一个部署放大算子。底座冻结，只训练指定层的方向。它不是原始DAS交换目标，也不能把未来收益提前归因于DAS。

默认单层来自`QA_SOURCE/selection.json`已经冻结的DAS选择。`das`、`md`、`random`三个初始化都使用这个相同层、相同中心和同一预算；强度固定为1、训练3轮、排序损失权重0。要覆盖默认层，可在三种初始化的训练命令中统一加同一个`--layers`值；不要分别根据测试表现挑层。

### 5.1 三题线路检查

```bash
for INIT in das md random; do
  export ADAPT_RUN="${ADAPT_ROOT}_pilot3/$INIT"
  python3 scripts/e27_qa_generation.py train \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN" \
    --init "$INIT" --alpha 1 --epochs 3 --ranking-weight 0 --pilot 3
  python3 scripts/e27_qa_generation.py check \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN"
  python3 scripts/e27_qa_generation.py judge-check \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN" --judge-exp E4_qa
done
```

`train --pilot 3`限制训练题；`check`固定生成前三道调参题，展示原模型、原DAS和当前适配器的答案，并记录实际写入。判分默认就是`E4_qa`，上面显式写出以便核对。检查点保存实际初始化、中心、强度、训练目标和来源指纹；后续生成读取保存值，不会悄悄换强度。

### 5.2 完整92对训练与调参生成

当前服务器搜索目录就是`$ADAPT_ROOT/{das,md,random}`。已存在匹配检查点时，训练命令会验证请求相同并跳过重训；配置不同须使用另一个输出目录。

```bash
for INIT in das md random; do
  export ADAPT_RUN="$ADAPT_ROOT/$INIT"
  python3 scripts/e27_qa_generation.py train \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN" \
    --init "$INIT" --alpha 1 --epochs 3 --ranking-weight 0
  python3 scripts/e27_qa_generation.py check \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN"
  python3 scripts/e27_qa_generation.py judge-check \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN" --judge-exp E4_qa
  python3 scripts/e27_qa_generation.py generate-tune \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN"
  python3 scripts/e27_qa_generation.py judge-tune \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN" --judge-exp E4_qa
done
```

先检查调参逐题变化是否是真正改对内容，不能仅凭损失下降、出现参考答案关键词或少数裁判边界宣称突破。默认排序权重0；若之后尝试非零排序项，另建运行目录，其他变量不同时改变。

### 5.3 E27测试仍为独立命令

以下步骤没有包含在上面的训练／调参循环内。对已经确定的检查点，分别执行：

```bash
for INIT in das md random; do
  export ADAPT_RUN="$ADAPT_ROOT/$INIT"
  python3 scripts/e27_qa_generation.py generate-test \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN"
  python3 scripts/e27_qa_generation.py judge-test \
    --source-run "$QA_SOURCE" --run-dir "$ADAPT_RUN" --judge-exp E4_qa
done
```

E27输出包括`adapter.pt`、`adapter.json`、`check.json`、`tune.json`／`test.json`及相应`*_scored`摘要与逐题结果。这里的原模型／原DAS参考有相同提示、强度、中心和生成预算；三种适配初始化的结果应并列看，不只保留最好的一种。

## 6. 将生成适配器加入老师主表

必须新建E26运行。适配后的方向进入独立方法名，并按该方向重新估C、调参、冻结；不能在旧测试目录替换方向。所有适配器应来自同一个`QA_SOURCE`，正式主实验不能传三题pilot适配器。

若复用本轮已测的Qwen正向候选，先执行`set_model_paths qwen3-8b`，再将`ADAPT_ROOT`明确设为`$TD_PROJECT_ROOT/results_e27_search_codex_20261106/qwen3-8b/ce_alpha3_ep3`。第2节默认的`ce_alpha1_ep3`是另一档力度，不能拿其结果冒充力度3的候选。结束符对照另在`ce_eos_alpha3_ep3`，本轮未把它选作主结果。

```bash
export ADAPTED_RUN="${RUN}_with_generation_adapters"

python3 scripts/e26_matched_coverage.py prepare \
  --model "$MODEL" --run-dir "$ADAPTED_RUN" \
  --qa-run "$QA_SOURCE" --tf-das-dir "$TF_DAS_DIR" \
  --tf-das-sweep "$TF_DAS_SWEEP" --tf-md-sweep "$TF_MD_SWEEP" \
  --qa-adapters "$ADAPT_ROOT/das" "$ADAPT_ROOT/md" "$ADAPT_ROOT/random"

python3 scripts/e26_matched_coverage.py collect-tune \
  --model "$MODEL" --run-dir "$ADAPTED_RUN"
python3 scripts/e26_matched_coverage.py judge-tune \
  --model "$MODEL" --run-dir "$ADAPTED_RUN" --judge-exp E4_qa --cache-dir "$JUDGE_CACHE"
python3 scripts/e26_matched_coverage.py freeze \
  --model "$MODEL" --run-dir "$ADAPTED_RUN"

python3 scripts/e26_matched_coverage.py collect-test \
  --model "$MODEL" --run-dir "$ADAPTED_RUN"
python3 scripts/e26_matched_coverage.py judge-test \
  --model "$MODEL" --run-dir "$ADAPTED_RUN" --judge-exp E4_qa --cache-dir "$JUDGE_CACHE"
python3 scripts/e26_matched_coverage.py report \
  --model "$MODEL" --run-dir "$ADAPTED_RUN"
```

适配器额外方法只加入真实问答任务，真假陈述任务仍保留原来的DAS与均值差对照。若适配器的中心、提示、模型、来源或强度对不上，程序应拒绝，而不是自动改成“差不多”的设置。

## 7. 收工回传哪些文件

回传每个模型／运行的配置和来源、`selection.json`、完整逐题输出与判分、`report.json`／`report.md`、`delivered_answers.json`和执行日志。E27另回传适配器清单、训练轨迹与逐题改对／改错；保留原模型与原DAS参考。

进度记录只填已完成事实：哪个模型、哪个阶段、退出码、产物路径、未决判分数、是否尚未进入测试。不要把三题线路通过写成效果通过，不要把少答题、识别错题或选择题偏好提升写成自由回答内容已改善。

相关依据：[生成方法与八篇文献核验](qa_generation_literature_20261106.md)、[E26方法红队与修复复核](e26_method_redteam_20261106.md)。

## 8. 可选结束符监督（已有负结果，默认不启用）

新训练加`--include-eos`，明确监督所选模型的结束符；默认关闭。已有适配器在检查／生成时读取保存设置，不能悄悄切换；如分词器结束符不在模型停止集合内，程序明确拒绝，可用`--stop-token-id`指定集合内的真实ID。旧不含这些字段的适配器仍可读取和续跑，旧文件不会被改写。

Qwen同层、力度3、3轮的结束符对照已完成调参：DAS正确数66→60，未结束11→2，含替换字符32→35。因此不把这个可选项当作已修好乱码的默认方案，原非结束符测试结果保持原样。

```bash
export EOS_RUN="$ADAPT_ROOT/eos_trial_new_run"
python3 scripts/e27_qa_generation.py train \
  --source-run "$QA_SOURCE" --run-dir "$EOS_RUN" \
  --init das --alpha 3 --epochs 1 --pilot 3 --include-eos
python3 scripts/e27_qa_generation.py check \
  --source-run "$QA_SOURCE" --run-dir "$EOS_RUN"
python3 scripts/e27_qa_generation.py judge-check \
  --source-run "$QA_SOURCE" --run-dir "$EOS_RUN" --judge-exp E4_qa
```

三种初始化的“同预算”指新增生成适配的训练题、轮数、目标、层和力度相同；DAS初始化自身包含此前交换训练开销，不能写成三者总计算量完全相同。
