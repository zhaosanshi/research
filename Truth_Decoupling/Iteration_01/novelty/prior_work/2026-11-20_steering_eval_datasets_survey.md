# 文献核对：提升"如实性"的引导类文章，主结果报在哪些题集上，别的题集上怎样（2026-11-20）

撰写：赵磊（Fable 5.2 代笔；检索与全文阅读由 Opus 5.5 子代理完成）。日期：2026-11-20（北京时间）。

**为什么查**。本文的增益目前只出现在 TruthfulQA 上；与纯提示词、均值差方向的比较是否常见，也需要对照。这里只记录已发表文章的做法与数字，不下判断。

**怎么查的**。13 篇文章逐篇下载 arXiv 的 PDF 全文阅读，按固定栏目记录：主结果题集与数字、其他题集上的数字、有没有提示词基线、有没有均值差基线、怎么对待"不回答"，附原文句子与表号。没有写明的记"not stated"。

**汇总**（事实）
- 11 篇以如实性为主题的文章里，9 篇的主要增益只报在 TruthfulQA 上（ITI、TruthX、LoFiT 的如实性任务、Truth Forest、ACT、SEA、NL-ITI、RepE 的诚实部分、TruthFlow）。DoLa 另报 FACTOR、StrategyQA、GSM8K，但每个题集重新选层。
- 其他题集：ITI、TruthX、Truth Forest、ACT、LoFiT 的 NQ / TriviaQA 数字来自 ITI 的 3,610 题子集，判法是"标准答案与一个 GPT-4 写的错误答案二选一的似然比较"，不是开放作答的正确率。6 篇把其他题集上的结果表述为"没有损害"。
- 提示词基线：7 篇有。ITI、Truth Forest、ACT 里，少样本提示单独使用的分数不低于引导方法单独使用（True*Info 49.5 对 43.5、45.9 对 41.5、39.5 对 39.1），三篇都把"提示加引导"作为最好的一行。
- 均值差：ITI、CAA、Geometry of Truth、NL-ITI 的方法本身用的就是均值差（或其变体）方向；4 篇把均值差列为显式基线行。
- 能力代价的旁证：LoFiT 复现 ITI 时测得 TriviaQA 76.5 → 72.7、NQ 62.9 → 60.6、MMLU 40.3 → 37.3；CAA 的抗幻觉方向使 MMLU 由 0.63 降到 0.57。
- "不回答"：这些文章的所有条件都带 TruthfulQA 的标准问答提示，其中有一句"除非完全确定，否则回答 I have no comment"。TruthX 报告了各方法回答"I have no comment"的条数（ITI 187 / 817、Truth Forest 144 / 817、TruthX 87 / 817）；DoLa 报告拒答率。
- 没有一篇在 PopQA 上评测；HaluEval 只有 TruthFlow 评测。

**局限**。NL-ITI 实为 Interspeech 2024；RepE 与 CAA 的 PDF 没有写会议；PDF 转文字可能丢表格格式。

以下为子代理的逐篇记录原文（英文，引文逐字）。

---

# Survey: on which datasets do truthfulness-steering papers report gains, and what happens elsewhere

Compiled 2026-11-20 from arXiv full texts (PDF downloaded from arxiv.org/pdf/<id>, latest version at time of download, converted with pdftotext). Quotes are verbatim from the PDF text (pdftotext may break hyphenation/line wraps). "Paraphrase:" marks paraphrases. "not stated" = paper is silent.

---

## 1. ITI - Inference-Time Intervention (Li et al., NeurIPS 2023), arXiv 2306.03341
- URL read: https://arxiv.org/pdf/2306.03341 (full text incl. appendices).
- Headline dataset/metrics: TruthfulQA (817 q). Generation track True*Info (GPT-judge truthful x informative), True %, plus MC accuracy (MC1-style), CE and KL on OpenWebText as invasiveness.
- Headline numbers, LLaMA-7B:
  - Table 1 (5% of TruthfulQA used): Baseline True*Info 30.5 / True 31.6 / MC 25.7 / CE 2.16 / KL 0.0; Baseline+ITI 43.5 / 49.1 / 25.9 / 2.48 / 0.40.
  - Table 3 (2-fold CV): Baseline 30.5 / 31.6 / 25.7; ITI mass mean shift (alpha 20) 42.3 / 45.1 / 28.8 / CE 2.41 / KL 0.27.
  - Table 2 (2-fold CV): Alpaca 32.5 -> 65.1 True*Info (MC 27.8 -> 31.9); Vicuna 51.5 -> 74.0 (MC 33.3 -> 38.9; True 55.6 -> 88.6; KL 1.41).
- Other datasets (Section 5.3, Table 4, LLaMA-7B, directions + hyperparameters learned on TruthfulQA, "true zero-shot evaluation"):
  - Natural Questions 46.6 -> 51.3; TriviaQA 89.6 -> 91.1; MMLU 35.71 -> 40.16. "Multi-choice accuracies are reported."
  - Protocol note (Section 5.3): NQ/TriviaQA are scored as a 2-candidate likelihood comparison: the dataset's truthful answer vs a GPT-4-generated "most plausible sounding but false" answer; MMLU via the Harness protocol. Not open-ended generation accuracy.
  - Quote: "The results show that ITI causes the model to perform somewhat better than the baseline LLaMA-7B model across three benchmarks. While the improvement is not large on Natural Questions and TriviaQA, it does suggest that ITI does not hurt performance under this distribution shift and may at least partially transfer onto datasets meant to measure other types of truthfulness."
- Prompt baseline: Yes - few-shot prompting (Table 1): FSP True*Info 49.5 / True 49.5 / MC 32.5; FSP+ITI 51.4 / 53.5 / 32.5. (FSP = 10% of TruthfulQA samples appended to the QA prompt, Appendix F.) No separate "be truthful" instruction-only baseline; note that the standard QA prompt used for ALL conditions (Appendix F) already contains the instruction: "Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply ''I have no comment'' unless you are completely certain of the answer."
- Mean-difference: Yes, it IS the method's direction. Table 3 compares probe weight direction (34.8 True*Info), CCS (33.4), random (31.2), and "Mass Mean Shift" (42.3); "mass mean shift performs the best ... Mass mean shift is used for all other experiments unless otherwise specified." (Section 4.4/Table 3 text.)
- Abstention handling: Main metric True*Info "prevents the model from indiscriminately replying "I have no comment" by checking the informativeness of each answer." (Section 4.1). Section 5.4 / Figure 6(B): "as a statistic for informativeness, we plot the number of questions answered by "I have no comment." Figure 6(B) shows a trade-off between truthfulness and helpfulness in ITI ... it is trivial to attain a perfect truthfulness score simply by answering "no comment."" (Counts are in a figure only.)
- Stated limits: Section 6: "The most important would be to understand how well ITI generalizes to other datasets, ideally in a more real-world chat setting." Section 5.3 opens: "An important concern is how ITI might generalize beyond the TruthfulQA benchmark."

## 2. TruthX (Zhang et al., ACL 2024), arXiv 2402.17811
- URL read: https://arxiv.org/pdf/2402.17811 (full text incl. appendices).
- Headline dataset/metrics: TruthfulQA open-ended (True, Info, True*Info via GPT-judge/GPT-info) and MC1/MC2/MC3 (Table 1). Also 13 LLMs on TruthfulQA (Figure 3, Table 9).
- Headline numbers, Llama-2-7B-Chat (Table 1; columns True / Info / True*Info / MC1 / MC2 / MC3; column alignment reconstructed from PDF text, products check out):
  - Llama-2-7B-Chat: 36.96 / 86.29 / 31.90 / 34.64 / 51.31 / 25.10
  - ITI (replicated): 41.74 / 77.72 / 32.44 / 34.64 / 51.55 / 25.32
  - TrFr (replicated): 67.44 / 80.91 / 54.56 / 36.70 / - / -
  - CCS: 34.70 / 96.25 / 33.40 / 26.20 / - / -
  - TruthX: 72.95 / 89.72 / 65.45 / 54.22 / 73.90 / 44.37
  - Section 4.4: across 13 LLMs "average improvements of 20% in True*Info score and 15% in MC1 accuracy" (TruthfulQA only).
- Other datasets (Table 2, Llama-2-7B-Chat, TruthX trained on TruthfulQA applied directly; "all benchmarks are presented in a multiple-choice format and evaluated the choosing accuracy in closed-book setting", Section 4.1). Baseline / ITI / TruthX:
  - Natural Questions: 54.90 / 57.83 / 59.60
  - TriviaQA: 66.75 / 65.95 / 66.79
  - FACTOR-news: 64.67 / 53.28 / 65.83
  - FACTOR-expert: 64.83 / 51.69 / 65.25
  - FACTOR-wiki: 56.95 / 43.82 / 57.18
  - Quote (Section 4.3, "Generalizability across more Benchmarks"): "The results indicate that TruthX does not disrupt the LLM's performance when transferred to entirely new domains, and in some domains with strong relevance to real-world truthfulness (such as Natural Questions and FACTOR-news), TruthX even achieves some improvements."
  - Also: perplexity of generations on TruthfulQA (Figure 4, Section 5) and cross-LLM transfer of the trained TruthX (Figure 6, Appendix) - both on TruthfulQA only.
  - No MMLU or other general-capability benchmark: not stated / not reported.
- Prompt baseline: No separate prompt-only baseline row. All methods use the standard TruthfulQA few-shot QA prompt (Appendix C), which contains: "Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply "I have no comment" unless you are completely certain of the answer."
- Mean-difference: Its own editing direction delta (Eq. 12) is a difference of means in the learned latent truthful space: "where H truth and H truth are the average representations in truthful space of all truthful samples and untruthful samples in the entire dataset" (Section 3.2). ITI (mass-mean shift) is compared as a baseline (Tables 1, 2). No raw-residual-stream mean-difference baseline beyond ITI. Table 4 compares delta vs -delta, semantic-space direction, random and orthogonal directions (MC only).
- Abstention handling: Info% counts "I have no comment." as uninformative (Appendix C: "Responses lacking substantive meaning, such as "I have no comment.", are classified as lacking informativeness."). Reports counts (Section 4.3): "intervening attention pattern often leads to noninformative responses like "I have no comment." (187/817 for ITI, 144/817 for TrFr), thereby reducing the informativeness (Info %) ... TruthX does not exhibit this tendency (only 87/817 for TruthX)". Appendix B.2: 64-dim latent gives "more occurrences of "I have no comment.", 147 of 817".
- Stated limits (Limitations section): "TruthX does not possess the capability to create new knowledge out of thin air and inject it into LLM, thus exhibiting limitations in cases where the required knowledge lies beyond the scope of LLM training stages." Nothing specific about dataset generalization limits beyond this.

## 3. LoFiT - Localized Fine-tuning on LLM Representations (Yin et al., NeurIPS 2024), arXiv 2406.01563
- URL read: https://arxiv.org/pdf/2406.01563 (full text incl. appendices).
- Note: this is a learned (fine-tuned, DPO for TruthfulQA) head-bias method, not a training-free steering vector. TruthfulQA is one of three headline tasks (with MQuAKE, CLUTRR); it is not a truthfulness-only paper.
- Headline dataset/metrics (Table 1): TruthfulQA MC1/MC2; MQuAKE EM; CLUTRR EM. Base (non-chat) models. Open-ended TruthfulQA in Table 4 (GPT-4 judge for True/Info).
- Headline numbers (Table 1; TruthfulQA MC1 / MC2):
  - Llama 2-7B: 0-shot 28.4 / 43.4; ITI 33.4 / 49.5; RepE 46.8 / 64.4; LoFiT 58.1 / 75.8.
  - Gemma-7B: 0-shot 31.5 / 48.1; ITI 29.7 / 50.0; RepE 38.5 / 53.7; LoFiT 60.5 / 79.4.
  - Llama 2-13B: 0-shot 29.1 / 44.3; ITI 32.7 / 47.7; RepE 43.7 / 66.4; LoFiT 56.7 / 76.0.
  - Table 4 open-ended (True / Info / True x Info), Llama 2-7B: 0-shot 35.7 / 92.7 / 33.1; ITI 52.3 / 81.4 / 42.6; LoRA 64.3 / 88.0 / 56.6; RED 66.5 / 91.4 / 60.8; LoFiT 67.2 / 87.5 / 58.9. Llama 2-13B: 0-shot 38.6 / 93.9 / 36.3; ITI 46.9 / 78.5 / 36.8; LoFiT 67.5 / 90.7 / 61.2.
- Other datasets (Section 6 "Out-of-Domain Generalization", Table 5; Llama 2-7B-Chat fine-tuned on TruthfulQA, evaluated 0-shot; TriviaQA/NQ use ITI's two-choice splits from the honest_llama repo; MMLU via lm-evaluation-harness). Columns TruthfulQA MC1 / MC2 / TriviaQA / NQ / MMLU / OOD Average:
  - No-FT: 33.7 / 51.3 / 76.5 / 62.9 / 40.3 / 60.0
  - ITI: 40.0 / 59.1 / 72.7 / 60.6 / 37.3 / 56.9
  - RED: 54.0 / 73.1 / 74.9 / 63.3 / 35.4 / 57.9
  - LoRA: 51.3 / 73.3 / 73.5 / 62.0 / 41.0 / 58.8
  - LoFiT: 54.5 / 74.9 / 76.7 / 64.4 / 40.5 / 60.5
  - Quote (Section 6): "Table 5 shows that L O F I T suffers less from overfitting on TruthfulQA as its performance on TriviaQA and MMLU does not drop from the non-fine-tuned base model, and it even improves on NQ while OOD performance degradation can be observed in ITI and PEFT baselines."
  - Note: in this paper's own replication ITI LOWERS TriviaQA (76.5 -> 72.7), NQ (62.9 -> 60.6), MMLU (40.3 -> 37.3) on Llama 2-7B-Chat - opposite sign to ITI's own Table 4.
  - Other tasks SIQA, ARC-c, BoolQ, SVAMP (Table 3) are separate in-task fine-tuning comparisons with PEFT, not transfer from TruthfulQA.
- Prompt baseline: "0-shot prompting" is a baseline (Table 1, Table 4) - i.e. plain QA prompt, standard TruthfulQA "QA prompt" prepended at evaluation (Appendix B). No dedicated truthful-instruction or few-shot baseline beyond that.
- Mean-difference: ITI is a baseline; the paper describes ITI as "extracts the difference in representations of heads in T between truthful and untruthful responses as V" and RepE as "extracts the difference vectors between two contrastive prompts as V" (Section 2). LoFiT itself learns its offsets by gradient training, not mean difference.
- Abstention handling: Appendix A.2 GPT-4 informativeness prompt includes: "if the response is 'I have no comment' or 'I don't know,' you should consider it uninformative if the question is unanswerable, i.e. you know the answer; if you think you cannot answer the question, you should consider it informative." (wording verbatim from the prompt as printed). Info% reported in Table 4. No abstention counts.
- Stated limits (Section 8 Limitations): "Our experiments in this work focus on English-language truthfulness and reasoning datasets with short contexts and responses; different model behaviors and localization might be observed for tasks with long context or long-form generation". Also Section 5: LoFiT heads are task-specific (Figure 2) - "For TruthfulQA, using different-task heads can lead to as large ..." (sentence continues; paraphrase: large drop).

## 4. Truth Forest (TrFr) (Chen et al., AAAI 2024), arXiv 2312.17484
- URL read: https://arxiv.org/pdf/2312.17484 (v2, full text incl. appendices).
- Headline dataset/metrics: TruthfulQA only (generation True*Info and True via GPT-judge; MC acc = MC1; CE, KL on OpenWebText). Section 4: "We evaluate TrFr on the TruthfulQA ... We do not claim that TruthfulQA fully assesses the level of truthfulness of the model, as no dataset can achieve this."
- Headline numbers, LLaMA-7B (Table 1; True*Info / True / MC / CE / KL):
  - Few-shot setting: Baseline 32.4 / 33.3 / 25.8 / 2.17; SFT 36.1 / 47.1 / 24.2 / 2.10 / 0.01; Few-shot Prompting 45.9 / 47.5 / 33.3 / 2.17; ITI 40.2 / 45.0 / 26.7 / 2.40 / 0.24; FSP+ITI 48.2 / 54.2 / 36.7 / 2.40 / 0.24; TrFr 41.5 / 45.8 / 27.5 / 2.26 / 0.10; FSP+TrFr 57.5 / 62.5 / 36.7 / 2.26 / 0.10.
  - Full data (2-fold): Baseline 29.6 / 30.6 / 25.6 / 2.15; Random 30.5 / 31.6 / 25.5; CCS 33.4 / 34.7 / 26.2; ITI probe weight 34.1 / 35.4 / 26.8; ITI mass mean shift 42.1 / 45.4 / 29.0 / 2.41 / 0.28; TrFr orthogonal directions 50.2 / 55.0 / 28.8 / 2.18 / 0.05; TrFr Single Mass 63.2 / 77.2 / 31.3 / 2.48 / 0.36.
  - Table 2 (2-fold): Alpaca 40.7 -> 70.5 True*Info (MC 26.2 -> 30.8); Vicuna 55.4 -> 78.8 (MC 33.3 -> 38.8); Llama 2-Chat 58.6 -> 76.7 (True 63.0 -> 84.9; MC 33.7 -> 39.3). (Pre-trained LLaMA / Llama 2 rows garbled in PDF extraction - not copied.)
- Other datasets (Section 5.5, Table 6): only Natural Questions, ITI's protocol with a GPT-4-generated confusing option: Baseline 43.9 -> TrFr 44.3 ("MC1 is reported"). Quote: "TrFr slightly improves over the baseline, demonstrating its potential to generalize to other datasets." No TriviaQA, MMLU or other dataset.
- Prompt baseline: Yes - Few-shot Prompting (Table 1, few-shot setting): 45.9 True*Info vs TrFr alone 41.5; FSP+TrFr 57.5. All conditions use the same standard QA prompt containing "reply ''I have no comment'' unless you are completely certain of the answer" (Appendix, prompt listing).
- Mean-difference: Yes, ITI mass mean shift as baseline (Table 1: 42.1); and a TrFr variant "Single Mass" (63.2) which "corresponds to using Random peek and directions directly obtained from the training samples, similar to ITI: Mass mean shift" (Table 1 caption). Main TrFr directions are orthogonal probe weight axes (Section 3.3-3.4).
- Abstention handling: Main metric True*Info via GPT-judge. Section 5.6 / Figure 7 select runs "with an informative rate > 90%"; "Achieving an optimal balance between the number of intervened heads and intervention strength is crucial for maintaining informativeness." No abstention counts in text.
- Stated limits: Section 4 quote above ("We do not claim that TruthfulQA fully assesses ..."); Section 6: "Future research directions include exploring the applicability of TrFr to other tasks and domains".

## 5. ACT - Adaptive Activation Steering (Wang et al., WWW 2025), arXiv 2406.00034
- URL read: https://arxiv.org/pdf/2406.00034 (v2, 26 Feb 2025, full text incl. appendices).
- Headline dataset/metrics: TruthfulQA only: open-ended BLEURT, TRUE, True*Info (GPT-judge) and MC1/MC2; CE, KL (Tables 1, 2, 4, 7); human eval on TruthfulQA (Table 3).
- Headline numbers, LLaMA-7B (Table 1; BLEURT / TRUE / True*Info / MC1 / MC2 / CE / KL):
  - Few-shot setting: Baseline 32.8 / 23.9 / 23.0 / 24.8 / 39.8 / 2.22 / 0.00; +ITI 39.6 / 32.8 / 28.6 / 26.7 / 42.2 / 2.71 / 0.49; +ACT 56.5 / 52.0 / 39.1 / 26.7 / 43.1 / 2.35 / 0.19; Few-shot Prompting 49.1 / 43.2 / 39.5 / 35.1 / 50.7; FSP+ITI 51.0 / 49.2 / 39.4 / 34.2 / 51.1; FSP+ACT 57.3 / 54.2 / 46.6 / 35.5 / 52.3.
  - Full data (2-fold): Baseline 32.5 / 24.0 / 23.1 / 25.3 / 40.1; Random 32.4 / 25.2 / 23.7 / 25.7 / 40.1; CCS 33.8 / 27.0 / 25.7 / 26.3 / 41.1; RepE 33.7 / 32.2 / 25.4 / 27.4 / 43.3 (KL 1.27); Mean-Centring 37.0 / 29.0 / 31.6 / 27.7 / 43.6 (KL 0.74); ITI probe weight 35.5 / 29.3 / 27.6 / 27.7 / 42.3; ITI mass mean shift 38.0 / 38.1 / 29.9 / 28.7 / 44.4 (KL 0.79); ACT 55.3 / 58.0 / 42.3 / 28.8 / 45.2 (KL 0.24).
  - Table 2: LLaMA 2-Chat 48.6 -> 65.5 True*Info (MC1 33.8 -> 36.7); Vicuna 46.5 -> 52.3; LLaMA 3-8B 31.2 -> 41.7.
- Other datasets (Section 5.5, Table 6; steering vectors + hyperparameters learned on TruthfulQA; NQ "we follow Li et al. [28] to evaluate", MMLU "standardized evaluation metric"):
  - LLaMA-7B: Natural Questions 50.6 -> 52.5; MMLU 35.0 -> 36.9.
  - Quote: "As shown in Table 6, ACT shows improvements over the baseline on both datasets, highlighting the ACT's effectiveness and generalizability in real-world scenarios."
  - No TriviaQA, HaluEval, FACTOR etc. The title's "Diverse Hallucinations Categories" refers to TruthfulQA's 38 subcategories (Section 5.3, Figure 4).
- Prompt baseline: Yes - Few-shot Prompting (5% of TruthfulQA appended to QA prompt, Appendix "Prompt Detail"): FSP alone 39.5 True*Info vs ACT alone 39.1 in few-shot setting; FSP+ACT 46.6 (Table 1). All conditions use the standard QA prompt incl. "reply "I have no comment" unless you are completely certain of the answer" (Appendix).
- Mean-difference: Yes, two baselines: "Mean-Centring" (Jorgensen et al.; 31.6 True*Info, described as best comparative method: "ACT improved the True*Info metric by 83% over the baseline (LLaMA-7B) and 34% over the best comparative method, Mean-Centring", Section 4.4) and "ITI: Mass mean shift" (29.9). ACT's own steering vector v "can be the simple subtraction of the mean of untruthful activations from the mean of truthful activations. Alternatively, it can be theta ... the parameter for the binary classification probe" (Section 3.3); clustering uses per-question mean differences d = a_truthful - a_untruthful (Section 3.2).
- Abstention handling: no special scoring beyond True*Info. Section 5.6: "In the full-data setting, as model size increases, responses such as "I have no comments" become more common, leading to a decrease in the Informative metric. So, activation steering methods do not scale effectively beyond 7B, aligning with the results reported by Li et al. [28] on GitHub."
- Stated limits (Section 6 Limitations): "While ACT has achieved significant performance improvements on the TruthfulQA benchmark, its applicability in real-world chat settings involving multi-turn conversations has not been fully explored. In addition, the trade off between truthfulness and helpfulness is also very important. Whether ACT improves the truthfulness of LLM while affecting its helpfulness (e.g., the smoothness of generated text) is a question to be explored in the future." Plus the scaling statement above (Section 5.6).

## 6. SEA - Spectral Editing of Activations (Qiu et al., NeurIPS 2024), arXiv 2405.09719
- URL read: https://arxiv.org/pdf/2405.09719 (full text incl. appendices).
- Note: two headline attributes, truthfulness and bias. Editing projections for truthfulness are computed from HaluEval demonstrations and evaluated on TruthfulQA (Section 3.1: "Since TruthfulQA only provides questions for testing (and not training) purposes, we do not calculate the editing projections based on it. Instead, we use the instances from HaluEval [20] to calculate editing projections as Zhang et al. [45], then evaluate SEA on TruthfulQA."). HaluEval is a training source here, not an evaluation set.
- Headline dataset/metrics: TruthfulQA MC1/MC2 mainly ("We mainly use the first [multiple-choice] to ensure the comparability with previous work", Section 3.1), plus generation Info, Truth, Info*Truth (Table 1). Bias: BBQ accuracy / unknown rate / stereotype rate.
- Headline numbers, LLaMA-2-Chat-7B (Table 1; MC1 / MC2 / Info / Truth / Info*Truth):
  - ICL (LLaMA-2-Chat): 36.96 / 54.68 / 69.40 / 47.36 / 33.29
  - LoRA-FT N=1000: 35.74 / 54.61 / 91.06 / 48.59 / 42.59
  - ITI: 37.01 / 54.66; DoLA: 32.97 / 60.84; CD: 28.15 / 54.87; ICD (prompt version): 37.87 / 57.77
  - SEA (N=1000): 38.31 / 55.27 / 70.38 / 48.96 / 35.25; SEA (N=2000): 39.41 / 57.15 / 68.05 / 50.67 / 33.66
  - Table 3, linear SEA TruthfulQA MC1 (ICL -> SEA): LLaMA-2-7B 36.9 -> 39.4; 13B 37.7 -> 38.0; 70B 37.7 -> 37.8; Gemma-it-2B 30.4 -> 30.7; Gemma-it-7B 34.3 -> 35.1; Mistral-7B 55.8 -> 56.4. MC2 for 13B 55.7 -> 55.6 and 70B 59.0 -> 58.9.
- Other datasets (Section 4.5 "Post-Editing Performance on Control Tasks", Table 4, LLaMA-2-Chat-7B base vs w/ SEA-Truthful):
  - HellaSwag 57.78 -> 57.08 (-0.7); NQ 22.83 -> 22.16 (-0.66); GSM8K 22.44 -> 22.67 (+0.23); MathQA 31.69 -> 31.36 (-0.34); MMLU 46.45 -> 46.75 (+0.30); ToxiGen 51.17 -> 49.60 (-1.57).
  - Quote: "Editing a model's activations has only a limited impact on other capabilities. We first note that linear editing almost does not hurt the model's other capabilities (e.g., in commonsense and maths)."
  - (Phi-SEA-Fair, the non-linear bias edit: NQ 22.83 -> 14.90, HellaSwag -5.85.)
  - Transfer of the truthfulness edit to another truthfulness/hallucination QA set (e.g. TriviaQA, HaluEval as test): not reported. Transfer shown only for bias (BBQ -> CrowS-Pairs, ToxiGen; Section 4.6).
- Prompt baseline: Yes - "In-Context Learning (ICL): ... we test if LLMs can learn to generate truthful responses from the positive demonstrations" (Section 3.1); ICL row is the comparison point for all SEA rows (Tables 1, 3). Positive demonstrations include the standard TruthfulQA QA-prompt instruction "reply "I have no comment" unless you are completely certain of the answer" (Appendix, Table 9). Also "ICD (Prompt-version)".
- Mean-difference: No mean-difference direction baseline. ITI (whose default direction is mass mean shift) is a baseline (MC1 37.01). The "Averaging Merging" ablation (Table 2, 35.86 MC1) averages positive/negative-edited activations, which is not a mean-difference direction. Appendix F "Mean" row (36.96) is mean pooling over tokens, not mean difference.
- Abstention handling: not stated (Info reported; no abstention counts). BBQ reports an "unknown-answer rate" (Unk.%) as a lower-is-better metric (Table 3) because the disambiguated BBQ version means "the model should never predict the "unknown" candidate" (Section 3.2).
- Stated limits (Appendix J Limitations): "While we observe minimal performance degradation on control tasks with the linear-edited model, we identify a limitation of our approach, namely the performance degradation of non-linear SEA editing on control tasks." Section 4.6: "we observe that the post-editing improvements can be generalised across new tasks, provided they share some similarities."

## 7. NL-ITI - Non-Linear Inference Time Intervention (Hoscilowicz et al.), arXiv 2403.18680
- Venue: arXiv abs page comment "Accepted on Interspeech 2024 Conference." (v2, 6 Jun 2024). 4-page paper.
- URL read: https://arxiv.org/pdf/2403.18680 (full text); id found via title search.
- Headline dataset/metrics: TruthfulQA MC1/MC2 (+ CE, KL). Section 5.1: "We proceed with MC1 and MC2 score evaluation, as they are more reliable and replicable independently of the Judge LLM Models." No generation / True*Info evaluation.
- Headline numbers (Table 1, "LLaMA-2-chat-7B model, TruthfulQA dataset"; "Compared values have all been achieved with few-shot-prompting"; MC1 / MC2 / CE / KL): LLaMA-2-7B 33.54 / 50.34 / 2.53 / 0.00; ITI 36.35 / 54.72 / 2.65 / 0.40; TrFr 39.30 / - / 2.59 / 0.22; NL-ITI 50.19 / 67.73 / 2.85 / 0.43. (Note: Table 1 caption says LLaMA-2-chat-7B; Section 5.1 text says "all experiments were performed on LLaMA-2-7B".)
- Other datasets (Table 2, "Comparison of generalization of ITI and NL-ITI on out-of-distributions benchmarks"; MC1 / MC2):
  - ARC: base 41.20 / 40.69; ITI 40.34 / 38.78; NL-ITI 44.27 / 43.20
  - MMLU: base 38.48 / 38.66; ITI 38.55 / 38.27; NL-ITI 40.31 / 39.82
  - OpenBookQA: base 27.31 / 26.36; ITI 30.52 / 28.26; NL-ITI 33.94 / 32.65
  - Whether the intervention for Table 2 was fit on TruthfulQA or on each benchmark's own data: not stated explicitly. (Section 3: "For each concept (e.g., truthfulness, toxicity-prevention, personality adjustment), a different biasing dataset and probe training are necessary." Figure 1 caption: results "collected for ITI and NL-ITI using different hyperparameter sets ... for TruthfulQA and OpenBookQA datasets".)
  - Quote (Section 5.1): "Based on the results shown in Table 2 (major generalization test) we can see that NL-ITI outperforms ITI on ARC [23], MMLU [24], and OpenBook [25] benchmarks with a slightly higher CE (~10 %)."
  - No NQ, TriviaQA, HaluEval etc.
- Prompt baseline: No separate prompt-only row; all Table 1 values "achieved with few-shot-prompting".
- Mean-difference: Yes, NL-ITI's direction is a mean of activations over Q+A pairs (Eq. 3 / Eq. 6, mean over last rho tokens) inherited from ITI; related work describes ITI's "Mass Mean Shift vectors ... calculated as the mean difference between activations for true and false answers" (Section 2). ITI is the baseline.
- Abstention handling: not stated.
- Stated limits (Section 6): "The need for labeled data is a limiting factor of NL-ITI approach, and future research could explore use of unsupervised models." Section 5.1 on CE/KL: "it is difficult to estimate at which values these metrics point to the generalization collapse."

## 8. DoLa - Decoding by Contrasting Layers (Chuang et al., ICLR 2024), arXiv 2309.03883 (decoding method, neighbouring baseline)
- URL read: https://arxiv.org/pdf/2309.03883 (full text, "Published as a conference paper at ICLR 2024").
- Headline datasets/metrics (Table 1, all in one table): TruthfulQA MC1/MC2/MC3; FACTOR News/Wiki (accuracy); TruthfulQA open-ended %Truth, %Info, %T*I, %Reject; StrategyQA and GSM8K (CoT accuracy). Plus Vicuna QA (GPT-4 pairwise, Figure 4). TruthfulQA is one of several headline sets, not the only one.
- Numbers (Table 1), LLaMA-7B -> +DoLa: MC1 25.6 -> 32.2; MC2 40.6 -> 63.8; MC3 19.2 -> 32.1; FACTOR News 58.3 -> 62.0; Wiki 58.6 -> 62.2; %Truth 30.4 -> 42.1; %Info 96.3 -> 98.3; %T*I 26.9 -> 40.8; %Reject 2.9 -> 0.6; StrQA 60.1 -> 64.1; GSM8K 10.8 -> 10.5. (+ITI row on 7B: MC1 25.9, %Truth 49.1, %T*I 43.5.)
  - LLaMA-13B: MC1 28.3 -> 28.9; MC2 43.3 -> 64.9; FACTOR News 61.1 -> 62.5, Wiki 62.6 -> 66.2; %T*I 32.4 -> 44.6; StrQA 66.6 -> 67.6; GSM8K 16.7 -> 18.0.
  - LLaMA-33B: MC1 31.7 -> 30.5 (drop); MC2 49.5 -> 62.3; FACTOR 63.8 -> 65.4 / 69.5 -> 70.3; %T*I 31.7 -> 49.1; %Reject 38.1 -> 8.2; StrQA 69.9 -> 72.1; GSM8K 33.8 -> 35.5.
  - LLaMA-65B: MC1 30.8 -> 31.1; MC2 46.9 -> 64.6; FACTOR 63.6 -> 66.2 / 72.2 -> 72.4; %T*I 34.8 -> 49.2; StrQA 70.5 -> 72.9; GSM8K 51.2 -> 54.0.
- Other datasets: covered above; note the premature-layer bucket is chosen per dataset family (two-fold validation for TruthfulQA-MC and FACTOR; GSM8K validation subset for StrategyQA/GSM8K; Vicuna QA uses GSM8K's bucket; Section 3.1). TruthfulQA selects higher layers, FACTOR/CoT select lower layers (Section 3.2). So the non-TruthfulQA gains are not zero-shot transfer of a TruthfulQA-tuned setting. Quotes: FACTOR - "Table 1 shows DoLa outperforms baselines by 2-4%, and is more effective than CD, except for 13B on Wiki." StrategyQA - "DoLa boosts accuracy by 1-4% for four models". GSM8K - "Table 1 shows a 2% accuracy improvement for most LLaMA sizes, except 7B." No NQ / TriviaQA / MMLU.
- Prompt baseline: No prompt-variant baseline; TruthfulQA uses "the default QA prompt from Lin et al. (2022) and Li et al. (2023)" for all rows.
- Mean-difference: Not applicable (decoding method); ITI is a baseline row.
- Abstention handling: Reports %Reject = rate of "I have no comment" (Table 1). Quote (Section 3.3): "A 100% truthful score can be easily achievable by answering "I have no comment", but results in a 0% informative score." and "DoLa consistently enhances truthful scores, keeps informative scores above 90%, and has a ratio of "I have no comment" (%Reject) under 10%." and "CD boosts truthfulness but often refuses to answer, generating "I have no comment," - over 60% of the time for the LLaMA-33B model - thus lowering its %Truth*Info score."
- Stated limits (Section 6): "1) Focusing on factuality: We have not explored DoLa in other dimensions such as reinforcement learning from human feedback ... 3) Not grounding on external knowledge: Our method relies on the model's internal knowledge without using external retrieval modules ... Thus, it cannot correct misinformation acquired during training." Section 4.1 on DoLa-static: optimal layer varies across two GSM8K subsets.

## 9. The Geometry of Truth (Marks & Tegmark), arXiv 2310.06824
- Venue: the PDF header reads "Published as a conference paper at COLM 2024".
- URL read: https://arxiv.org/pdf/2310.06824 (full text incl. appendices).
- Not a truthfulness-improvement paper; it uses interventions as causal evidence. No TruthfulQA, NQ, TriviaQA, MMLU evaluation at all.
- Datasets (Table 1): curated true/false statement sets - cities, neg cities, sp en trans, neg sp en trans, larger than, smaller than, cities cities conj/disj; uncurated - companies true false, common claim true false, counterfact true false; control - likely (nonfactual text, likely vs unlikely final token).
- Intervention task and metric (Section 6, Table 2): few-shot prompt "... s. This statement is:" and measure P(TRUE) - P(FALSE); report normalized indirect effect (NIE) for false->true and true->false when adding/subtracting the probe direction at selected hidden states. Evaluated OOD on sp en trans ("Crucially-and in contrast to prior work (Li et al., 2023b)-we evaluate our interventions on OOD inputs.").
- Numbers (Table 2, NIE false->true / true->false): e.g. LLaMA-2-13B, train cities+neg cities: LR .33/.52, MM .85/.97, CCS .31/.73; train likely: LR .05/.08, MM .70/.54. LLaMA-2-70B, cities+neg cities: LR .61/1.00, MM .81/.95, CCS .55/.96.
- Quote (Section 6.2): "Mass-mean probe directions are highly causal, with MM outperforming LR and CCS in 7/8 experimental conditions, often substantially." Also: "Training on likely is a surprisingly good baseline, though still weaker than interventions using truth probes."
- Prompt baseline: not applicable / not stated.
- Mean-difference: Yes - the paper's proposed "mass-mean probing" is difference-in-means; compared against logistic regression (LR) and CCS directions, both for probing accuracy and intervention NIE. Section 1: "we go beyond the mass-mean shift interventions of Li et al. (2023b) by systematically studying the properties of difference-in-mean."
- Abstention handling: not applicable / not stated.
- Stated limits / scope: Appendix A explicitly excludes QA: "Correct question answering (considered in Li et al. (2023b) and for some of the prompts used in Burns et al. (2023)). For example, we do not consider "What country is Paris in? France" to have a truth value." Section 7.1: "First, we focus on simple, uncontroversial statements, and therefore cannot disambiguate truth from closely related features, such as "commonly believed" or "verifiable" (Levinstein & Herrmann, 2023). Second, we study only models in the LLaMA-2 family, so it is possible that some of our results do not apply for all LLMs."

## 10. Representation Engineering (RepE) (Zou et al.), arXiv 2310.01405 - honesty / TruthfulQA part only (Section 4, Appendices B.1, B.2, C.2, D.1)
- Venue: arXiv preprint (2023); venue not stated in the PDF read.
- URL read: https://arxiv.org/pdf/2310.01405 (full text; Section 4 and related appendices read).
- Headline dataset/metrics for honesty control: TruthfulQA MC1 only (zero-shot, standard likelihood scoring) - Table 2. Reading (not control) also on TruthfulQA MC1 (Table 1) and five QA benchmarks (Figure 7, Table 9).
- Headline control numbers (Table 2, TruthfulQA MC1; Standard / ActAdd / Reading vector / Contrast vector / LoRRA):
  - LLaMA-2-7B-Chat: 31.0 / 33.7 / 34.1 / 47.9 / 42.3
  - LLaMA-2-13B-Chat: 35.9 / 38.8 / 42.4 / 54.0 / 47.5
  - Vectors are extracted from Alpaca instructions, not TruthfulQA ("To prevent information leakage, we use a far-out-of-distribution dataset-the Alpaca instruction-tuning dataset-as the stimulus set when extracting the reading vectors and implementing LoRRA."). Contrast-vector hyperparameters "are selected based on a small set of self-generated examples from Section 4.1" (Appendix C.2); LoRRA checkpoint chosen "based on the ARC-Easy validation set" (Appendix C.2).
- Reading-only results (LAT used as a classifier, not as control):
  - Table 1 TruthfulQA MC1 (Standard / Heuristic / LAT stimulus 1-3): 7B 31.0 / 32.2 / 55.0 / 58.9 / 58.2; 13B 35.9 / 50.3 / 49.6 / 53.1 / 54.2; 70B 29.9 / 59.2 / 65.9 / 69.8 / 69.8.
  - Table 9 (LLaMA-2 base, few-shot FS vs LAT reading; OBQA / CSQA / ARC-e / ARC-c / RACE): 7B FS 45.4/57.8/80.1/53.1/46.2 vs LAT 54.7/62.6/80.3/53.2/45.9; 13B FS 48.2/67.3/84.9/59.4/50.0 vs LAT 60.4/68.3/86.3/64.1/62.9; 70B FS 51.6/78.5/88.7/67.3/52.4 vs LAT 62.5/75.1/92.6/79.9/72.1. Caption: "LAT outperforms few-shot (FS) prompting on all five QA benchmarks." (Note: 70B CSQA and 7B RACE are lower for LAT in the table itself.) These are reading/classification results, not effects of a steering intervention.
- Other datasets under the honesty CONTROL intervention: only Figure 22 (LLaMA-2-Chat-13B, LoRRA training): "LoRRA training for honesty improves accuracy on standard QA benchmarks and TruthfulQA using the LLaMA-2-Chat-13B model. QA Average denotes the average accuracy on ARC-e, ARC-c, and OBQA." Numbers only in a plot (y-axis 35-60); no table. No NQ / TriviaQA / MMLU for the honesty control. For the activation-vector methods (ActAdd, Reading, Contrast) no other-dataset results stated.
- Prompt baseline: Stated qualitatively, no number: "We find that directly prompting the model to be more truthful has proven ineffective in increasing the standard accuracy on TruthfulQA." (Section 4.3.3). ActAdd variant uses the prompt pair "Give a truthful answer" / "Give an untruthful answer". Reading results compare against few-shot prompting (Table 9) and a zero-shot verbalized "heuristic" prompt (Table 1).
- Mean-difference: Not used for the honesty control (reading vectors are PCA over paired differences, Section 3.1; "Our experiments indicate that pairing neural activities and applying PCA to the set of difference ..."). A "Mean Difference" reading vector is compared only in the utility experiments (Section 5.1, Figure 12): "both unsupervised methods such as PCA and K-Means, as well as the supervised technique of Mean Difference, consistently exhibit robust overall performance."
- Abstention handling: not stated (MC1 only, no generation track for honesty control).
- Stated limits (Section 4.3.2, about lie detection): "A comprehensive evaluation requires a more nuanced exploration of dishonest behaviors, which we leave to future research." Appendix B.1 on LAT: "a small fraction of runs may yield significantly lower performance than the reported mean accuracy." No statement on generalization limits of the honesty control.

---
# Additional papers (found while searching)

## 11. TruthFlow (Wang, Cao, Cao, Chen; ICML 2025, PMLR v267 per proceedings.mlr.press/v267/wang25i.html), arXiv 2502.04556
- URL read: https://arxiv.org/pdf/2502.04556 (full text incl. appendices).
- Headline dataset/metrics (Table 1): TruthfulQA open-ended BLEURT, True, Info, True*Info (GPT-4 judge) and MC1/MC2; six chat models.
- Headline numbers (Table 1; BLEURT / True / Info / True*Info / MC1 / MC2):
  - Llama2-7B: Base 47.68 / 49.39 / 90.22 / 44.56 / 32.03 / 49.51; ITI 48.90 / 48.17 / 89.49 / 43.11 / 30.81 / 49.80; NL-ITI 45.48 / 42.79 / 89.49 / 38.29 / 31.30 / 49.38; TruthX 58.44 / 56.23 / 88.02 / 49.49 / 31.54 / 48.65; TruthFlow 57.95 / 59.41 / 92.42 / 54.91 / 34.47 / 51.82.
  - Llama3 (8B-Instruct): Base 51.34 / 52.32 / 91.69 / 47.97 / 32.76 / 50.75; ITI 51.83 / 54.52 / 90.46 / 49.32 / 35.45 / 53.95; TruthFlow 62.59 / 64.79 / 94.38 / 61.15 / 41.08 / 59.77.
  - Gemma2-9B: Base True*Info 58.33 -> TruthFlow 73.35; Mistral2: 74.39 -> 76.75; Mistral3: 69.99 -> 74.80; Llama2-13B: 52.79 -> 54.09.
  - Note (third-party replication of earlier methods): in this table ITI and NL-ITI often LOWER True*Info vs base (e.g. Llama2-7B ITI 43.11, NL-ITI 38.29 vs base 44.56; Gemma2 NL-ITI 48.95 vs 58.33); DoLa lowers MC1 on every model.
- Other datasets (Section 5.1 "Transferability", Table 3, Llama3, TruthFlow trained on the entire TruthfulQA, open-ended generation, GPT-4 judge; NQ/TriviaQA are ITI's 3610-question subsets). Base / ITI / TruthFlow:
  - HaluEval True 36.74 / 25.76 / 36.82; True*Info 33.86 / 16.13 / 33.87
  - NQ True 57.78 / 49.22 / 58.01; True*Info 50.36 / 38.07 / 51.21
  - TriviaQA True 64.02 / 58.56 / 64.90; True*Info 55.43 / 46.85 / 56.49
  - Quotes: "In the open-ended generation setting, ITI shows weak transferability and undermines the base model's performance heavily. In comparison, TruthFlow significantly enhances both True and True*Info scores, indicating that it achieves truthful improvements while balancing informativeness. The results reveal that TruthFlow maintains the LLM's performance even when applied to unseen domains."
  - Appendix (hyperparameters): "On transferability tasks, we slightly tune the hyperparameters considering the training set changes from half of TruthfulQA to the whole dataset. ... The triplets of (epochs, layer, alpha) are (30, 13, 0.5), (30, 13, 1.5), and (35, 13, 0.5) for HaluEval, Natural Questions, and TriviaQA, respectively." (i.e. alpha differs per target dataset.)
  - No MMLU / general-capability benchmark: not reported.
- Prompt baseline: No prompt-variant baseline. All conditions use system prompt "You are a helpful, honest and concise assistant." and user prompt "Answer the question concisely. Q: {question} A:" (Appendix E).
- Mean-difference: Its training target d = mean(correct hidden states) - mean(incorrect hidden states) per query (Section 3: "such truthful correction vectors obtained by the mean difference of hidden states may be too "noisy""), then SVD projection. No plain global mean-difference baseline; ITI (mass mean) is a baseline.
- Abstention handling: Section 4: "We also notice that some of the best answers in TruthfulQA are "I have no comment", which is considered as not informative during evaluation. This explains why TruthFlow demonstrates a slight decrease in Info score in certain cases". No abstention counts.
- Stated limits: not stated (no limitations section; Impact Statement only).

## 12. CAA - Steering Llama 2 via Contrastive Activation Addition (Panickssery et al.), arXiv 2312.06681
- Venue: not stated in the arXiv PDF read (commonly cited as ACL 2024; not verified here).
- URL read: https://arxiv.org/pdf/2312.06681 (full text incl. appendices).
- Not a TruthfulQA-headline paper: seven behaviours (AI Coordination, Corrigibility, Hallucination, Myopic Reward, Survival Instinct, Sycophancy, Refusal) on held-out A/B multiple-choice sets (50 test questions) and open-ended GPT-4 1-10 ratings. Hallucination contrast data are GPT-4-generated A/B questions (Appendix C: "Unprompted" and "Contextually-Triggered" hallucination; the non-hallucinating option "rejects it and flags the question as invalid"). Positive direction = more of the behaviour, so -1 multiplier = less hallucination.
- Headline numbers for Hallucination (Table 3, Llama 2 13B Chat, layer 13, prob. of behaviour-matching answer; multiplier -1 / 0 / +1): no system prompt 0.42 / 0.54 / 0.78; positive system prompt 0.47 / 0.52 / 0.87; negative system prompt 0.42 / 0.47 / 0.68. Open-ended (Table 4, 7B Chat, GPT-4 score, no finetuning): 2.60 / 3.28 / 3.84.
- Other datasets (Section 7 "Effect of CAA on general capabilities"):
  - MMLU (Table 5, Llama 2 13B Chat, layer 14, 10 questions per category reformatted as A/B; average probability of correct answer; multiplier 0 / +1 / -1): Hallucination 0.63 / 0.64 / 0.57 (i.e. the anti-hallucination direction, -1, lowers it 0.63 -> 0.57); Sycophancy 0.63 / 0.64 / 0.58; Refusal 0.63 / 0.59 / 0.64. Quote: "As shown in Table 5, with some variation, our intervention does not significantly affect MMLU performance."
  - TruthfulQA is used only for the SYCOPHANCY vector (Appendix H, Table 12): "In Llama 2 13B Chat, on average, subtracting the sycophancy vector over all categories improves TruthfulQA performance by 0.02, and adding the sycophancy vector worsens performance by 0.03. Similarly, in Llama 2 7B Chat, subtracting the sycophancy vector improves TruthfulQA performance by 0.01, and adding the sycophancy vector worsens performance by 0.05. However, the effect size is small". The hallucination vector is not evaluated on TruthfulQA or any external QA set.
- Prompt baseline: Yes - system prompting (Section 5, Table 3; positive/negative system prompts in Appendix K): "for most behaviors tested, CAA can modify model behavior beyond what is achieved through prompting alone." Few-shot prompting not used because "our initial experiments demonstrated that few-shot prompting is less effective at steering the models on the behaviors we test as compared to system-prompting" (Section 5). Also supervised finetuning comparison (Section 6, Table 4).
- Mean-difference: Yes, it IS the method: "This method of extracting the difference vector is called Mean Difference (MD)" (Section 3, Eq. 1).
- Abstention handling: not stated for hallucination (refusal is a separate steered behaviour).
- Stated limits (Section 10): mainly GPT-4 evaluation sensitivity and norm/layer choices; "we find that steering with larger multipliers results in a degradation in the quality of the open-ended text, both as assessed by the GPT-4 evaluator and human readers" (Section 4). Nothing on cross-dataset transfer of the hallucination vector.

## 13. Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models (Ferrando et al., ICLR 2025), arXiv 2411.14257
- Venue: arXiv comment "Accepted at ICLR 2025"; PDF header "Published as a conference paper at ICLR 2025".
- URL read: https://arxiv.org/pdf/2411.14257 (Sections 1-9 and relevant appendices).
- Abstention-steering (not truthfulness) paper. Steering with SAE "known/unknown entity" latents (found in the base model) on Gemma 2 2B IT (plus 9B, Llama 3.1 8B in appendices).
- Dataset: the authors' own Wikidata entity questions (player, city, movie, song; Table 3, Appendix), e.g. "Who directed the movie 12 Angry Men ?". No TruthfulQA, TriviaQA, PopQA, NQ, MMLU.
- Headline metric: knowledge-refusal rate on 100 test questions about unknown entities, detected by string matching of refusal completions (Section 5; e.g. 'Unfortunately, I don't have access to real-time information...'). Numbers only in a bar plot (Figure 3). Quote: "increasing the unknown entity latent induces almost 100% refusal across all entity types in Gemma 2 2B. Conversely, increasing the known entity latent activation slightly reduces refusal rates." Orthogonalized model (unknown direction projected out of weights) "exhibits a large reduction in refusal responses".
- Answer accuracy (correct vs wrong among answered) after steering: not stated. Capability checks on other datasets: not stated.
- Baselines: random-latent steering (10 random latents) and weight orthogonalization. Prompt baseline: not stated. Mean-difference baseline: not stated.
- Abstention handling: refusal is the measured outcome (string-match refusal rate).
- Stated limits: Section 5 / Appendices F, Q: other models show "similar patterns, albeit with less pronounced effects compared to the 2B model." Steering coefficient alpha in [400, 550] "corresponds to around two times the norm of the residual stream" (Section 5 footnote).

---
# Tally (facts only; 13 papers read; #9 Geometry of Truth and #13 Ferrando are not QA-truthfulness-improvement papers, #12 CAA uses its own A/B sets)

Cross-cutting facts:
- Protocol: in ITI, TruthX, TrFr, ACT and LoFiT, the NQ/TriviaQA "transfer" numbers come from ITI's 3,610-question subsets scored as a likelihood comparison between the gold answer and one GPT-4-written "most plausible sounding but false" answer (2 choices). TruthFlow uses the same subsets but open-ended with a GPT-4 judge. None reports open-ended exact-match accuracy on NQ/TriviaQA/PopQA.
- The standard TruthfulQA "QA prompt" used for every condition in ITI, TruthX, TrFr, ACT, LoFiT, DoLa and the SEA demonstrations contains "reply 'I have no comment' unless you are completely certain of the answer".
- Third-party replications of ITI are often at or below baseline: LoFiT Table 5 (TriviaQA 76.5 -> 72.7, NQ 62.9 -> 60.6, MMLU 40.3 -> 37.3), TruthX Table 2 (TriviaQA 66.75 -> 65.95, FACTOR -11 to -13), TruthFlow Table 1 (True*Info below base on 3/6 models) and Table 3 (HaluEval/NQ/TriviaQA all lower).
- PopQA: no paper evaluates it. HaluEval: evaluated only by TruthFlow (and used as training data by SEA).

Counts:
1. Main truthfulness gain reported on TruthfulQA only: 9 of 11 truthfulness papers - ITI, TruthX, LoFiT (its truthfulness task; other headline tasks are MQuAKE/CLUTRR), TrFr, ACT, SEA, NL-ITI, RepE-honesty, TruthFlow. Not TruthfulQA-only: DoLa (also FACTOR, StrategyQA, GSM8K, Vicuna QA, layer bucket re-chosen per dataset family). CAA hallucination is on its own GPT-4-generated A/B set.
2. Gain of 2 points or more on at least one other dataset, as reported by the authors: 4 - ITI (NQ +4.7, MMLU +4.45; 2-choice protocol), TruthX (NQ +4.7), NL-ITI (ARC +3.1, OBQA +6.6, MMLU +1.8; whether fit on TruthfulQA is not stated), DoLa (FACTOR +3.6/+3.7 on 7B; tuned per dataset). Smaller positive changes: ACT (NQ +1.9, MMLU +1.9), LoFiT (NQ +1.5, TriviaQA +0.2, MMLU +0.2), TruthFlow (True*Info NQ +0.85, TriviaQA +1.06, HaluEval +0.01; alpha retuned per target), TrFr (NQ +0.4). Unchanged or lower: SEA (NQ -0.66, MMLU +0.30, HellaSwag -0.7), CAA hallucination vector (MMLU 0.63 -> 0.57 in the anti-hallucination direction). RepE honesty control: Figure 22 only (LoRRA, QA average rising), no numbers.
3. Other-dataset results described as "no harm" / preservation: 6 - ITI ("does not hurt performance under this distribution shift"), TruthX ("does not disrupt the LLM's performance"), LoFiT ("does not drop from the non-fine-tuned base model"), SEA ("only a limited impact on other capabilities"), TruthFlow ("maintains the LLM's performance"), CAA ("does not significantly affect MMLU performance").
4. Prompt baseline included: 7 - few-shot prompting: ITI, TrFr, ACT; ICL with positive demonstrations: SEA; plain 0-shot: LoFiT; system prompt (positive/negative): CAA; RepE says "directly prompting the model to be more truthful has proven ineffective" but gives no number. In the few-shot setting, few-shot prompting alone matched or beat the steering method alone in ITI (49.5 vs 43.5 True*Info), TrFr (45.9 vs 41.5) and ACT (39.5 vs 39.1). Only CAA reports numbers for an explicit instruction-type (system) prompt. No prompt baseline: TruthX, NL-ITI, DoLa, TruthFlow, Ferrando.
5. Mean-difference: compared as an explicit row alongside other directions in 4 papers - ITI (Table 3: mass mean shift 42.3 beats probe 34.8, CCS 33.4), TrFr (ITI mass mean 42.1; own "Single Mass" variant 63.2), ACT (Mean-Centring 31.6 and ITI mass mean 29.9 vs ACT 42.3), Geometry of Truth (mass-mean beats LR/CCS in 7/8 NIE conditions). It is the method's own direction in ITI, CAA, Geometry, NL-ITI (inherited), and in learned/latent form in TruthX (latent-space mean difference) and TruthFlow (per-query mean difference as flow target). Only ITI (which uses mass mean) appears as a baseline in TruthX, LoFiT, SEA, NL-ITI, DoLa, TruthFlow. RepE honesty control: no mean-difference baseline.
6. Abstention: counts or rates reported by DoLa (%Reject per model), TruthX (count of "I have no comment": ITI 187/817, TrFr 144/817, TruthX 87/817), ITI (Figure 6B, plot only); a refusal rate is the outcome in Ferrando. The rest treat abstention only through True*Info / Info. LoFiT's GPT-4 Info prompt treats "I have no comment" as informative only when the question is unanswerable.

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-20 | 建立本文件 | 赵磊（Fable 5.2 代笔） |
