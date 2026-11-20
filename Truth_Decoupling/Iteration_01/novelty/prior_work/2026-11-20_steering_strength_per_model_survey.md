# 文献核对：各模型用同一个力度，还是各自选（2026-11-20）

撰写：赵磊（Fable 5.2 代笔；检索与全文阅读由 Opus 5.5 子代理完成）。日期：2026-11-20（北京时间）。

**为什么查**。赵磊 2026-11-20 15:24 提出：五个模型一致的应是方法，力度可以各不相同。这里只核对已发表文章的做法，不下判断。

**怎么查的**。18 篇在两个以上模型上做引导（steering · 往模型内部状态里加一个方向来改变输出）的文章，逐篇打开 arXiv 全文，读方法、实验设置与附录，记录"力度旋钮叫什么、怎么选、各模型是否相同"，附原文句子与位置。没有写明的记"not stated"，不推断。

**汇总**（事实）
- 明确按模型（或更细）各自选的 6 篇：Arditi 等的拒答方向（13 个模型各自在验证集上选层与位置）、LoFiT（按模型乘任务网格搜索）、AxBench（按概念、方法、模型设置选系数）、SEA（按模型与任务）、Stolfo 等（系数按输入计算，层按模型乘指令选）、CAST（各模型系数 4 / 14 / 1.7 / 1.5 等）。
- 各模型同一个值或同一组取值的 6 篇：CAA、RepE、ICV、ReFT、Tan 等、Persona Vectors。其中 CAA、RepE、Tan、Persona 仍按模型各自选层。
- 只在一个主模型上调、其余模型怎么定没有写的 6 篇：ITI、TruthX、ACT、SADI、Truth Forest、ActAdd。

**局限**。正文由网页转成文字后阅读，个别表格单元可能丢失（LoFiT 表 6 末行、CAST 的表有截断，文内已标）；ActAdd 与 Persona Vectors 的会议没有从 arXiv 页面核实。

以下为子代理的逐篇记录原文（英文，引文逐字）。

---

# Survey: is steering strength kept identical across models or chosen per model?

Method: full text fetched from arxiv.org/html/<id> (converted to text, searched and read around method / setup / appendix). Quotes are verbatim from that text (math symbols may be flattened by HTML-to-text conversion). Paraphrase is marked "(paraphrase)".

---

## 1. Inference-Time Intervention (ITI) — Li et al., NeurIPS 2023
- arXiv 2306.03341; read: https://arxiv.org/html/2306.03341
- Models: LLaMA-7B (main), Alpaca-7B, Vicuna-7B (Table 2). (A LLaMA2-7B-chat edited model is linked, no experiments reported in the paper.)
- Knob: K (number of heads) and alpha (multiple of the per-head std along the truthful direction). Chosen values: K = 48, alpha = 15.
- How chosen: hyperparameter sweep on LLaMA-7B using 5% of TruthfulQA for training and 5% for validation, "considering multiple scores" (true*info, CE, KL). For direction-type comparisons (Table 3) and position-selection comparisons (Table 5), alpha is grid-searched separately per direction/method on LLaMA-7B.
- Same or per model: The sweep is reported only for LLaMA-7B. For Alpaca and Vicuna the paper does not state which K / alpha were used, nor whether they were re-tuned. -> not stated for the other models.
- Quotes:
  - Sec 4.1 / Fig 4 text: "In Figure 4, we sweep two hyperparameters controlling the strength of the intervention, using 5 % of randomly sampled questions for training and validation each. ... We choose the optimal hyperparameters K = 48 and α = 15 by considering multiple scores."
  - Sec 3.3: "we explore their effects experimentally and determine optimal values via a standard hyperparameter sweep. ... The α should be selected per need by the user via trial and error"
  - Sec 4.3 (IFT models): "In Table 2, we apply ITI on instruction finetuned models by finding and intervening in their truthful directions." (no hyperparameter values given for Alpaca/Vicuna)

## 2. Contrastive Activation Addition (CAA) — Rimsky/Panickssery et al., ACL 2024
- arXiv 2312.06681; read: https://arxiv.org/html/2312.06681
- Models: Llama 2 7B Chat, Llama 2 13B Chat (vectors also from Llama 2 7B base for transfer tests).
- Knob: steering "multiplier" (applied to behaviour-norm-normalized vectors) and layer. Values: layer sweep with multipliers -1 and +1; open-ended examples with multipliers 2 and -2; MMLU/TruthfulQA tests on 13B with multipliers 1 and -1.
- How chosen: layer chosen per model by sweeping all layers with a constant multiplier (+-1) on held-out test questions; then a range of multipliers tested at that layer. Vectors are normalized across behaviours, not across layers.
- Same or per model: Layer is reported per model (7B: layer 13; 13B: "usually 14 or 15"; some 13B tables use layer 14, Table 3 uses layer 13). The multiplier grid (-1/+1, and a limited range in figures) is the same set across the two models; no per-model multiplier tuning is described.
- Quotes:
  - Sec 4.1: "To find the optimal layer for steering, we sweep over all layers and perform CAA with multipliers of − 1 and 1 , assessing the effect size on the held-out test questions." / "In the 7B model, this corresponds to layer 13 and adjacent layers. The optimal layer in the 13B model is usually 14 or 15."
  - Limitations (Sec on norms): "In contrast, our approach to CAA hyperparameter search involves first finding an optimal layer using a constant multiplier and then testing a range of multipliers at the resultant best layer."
  - Same section: "We normalize steering vector magnitudes across all behaviors to standardize across behaviors before applying steering multipliers."

## 3. Representation Engineering (RepE) — Zou et al., 2023 (arXiv preprint)
- arXiv 2310.01405; read: https://arxiv.org/html/2310.01405
- Models (multi-model quantitative control result = TruthfulQA honesty control, Table 2): LLaMA-2-Chat-7B and LLaMA-2-Chat-13B. Other sections use LLaMA-2-Chat-13B, Vicuna-13B / Vicuna-33B-Uncensored mostly one model per demo.
- Knob: coefficient of the linear-combination operator + set of layers (Contrast Vector); LoRRA has alpha, beta (loss coefficients) and adapter layers.
- How chosen: "selected based on a small set of self-generated examples" maximizing log-probs of truthful responses.
- Same or per model: Contrast Vector coefficient is the SAME (0.25) for 7B and 13B; the LAYER set differs per model (7B range(8,32,3); 13B range(10,40,3)). LoRRA: alpha = 5, beta = 0 stated once; adapter layers differ per model (7B range(10,20,2); 13B range(10,40,3)).
- Quotes (Appendix C.2):
  - "For the 7B model, we apply a linear combination with a coefficient of 0.25 to layers range(8, 32, 3) . Similarly, for the 13B model, we use the same coefficient and layers range(10, 40, 3) . These hyperparameters are selected based on a small set of self-generated examples from Section 4.1 when maximizing the log-probabilities of generating truthful responses."
  - "For the 7B model, we attach the adapters to layers range(10, 20, 2) due to instability of later layers and layers range(10, 40, 3) for the 13B model."
  - Sec 3.2: "The control elements v can be scaled by coefficients, depending on the strength of the desired effect, which we omit for simplicity."

## 4. Refusal Is Mediated by a Single Direction — Arditi et al., NeurIPS 2024
- arXiv 2406.11717; read: https://arxiv.org/html/2406.11717
- Models (13 chat models): Qwen 1.8B/7B/14B/72B, Yi 6B/34B, Gemma 2B/7B, Llama-2 7B/13B/70B, Llama-3 8B/70B.
- Knob: no scalar coefficient. Directional ablation removes the full component (all layers, all positions); activation addition adds the raw (un-normalized) difference-in-means vector r^(l) with coefficient 1 at a single layer l*. The knobs that vary are the source layer l* and token position i*.
- How chosen: per model, select among |I| x L candidate difference-in-means vectors on validation sets (32 HarmBench harmful val + harmless val) by minimum bypass_score subject to induce_score > 0, kl_score < 0.1, l < 0.8L.
- Same or per model: PER MODEL (layer and token position). Strength coefficient itself is fixed (raw vector added; ablation is full), so the effective magnitude is the model's own diff-in-means norm.
- Per-model values (Table 5, i* ; l*/L): Qwen 1.8B -1; 15/24 | Qwen 7B -1; 17/32 | Qwen 14B -1; 23/40 | Qwen 72B -1; 62/80 | Yi 6B -5; 20/32 | Yi 34B -1; 37/60 | Gemma 2B -2; 10/18 | Gemma 7B -1; 14/28 | Llama-2 7B -1; 14/32 | Llama-2 13B -1; 26/40 | Llama-2 70B -1; 21/80 | Llama-3 8B -5; 12/32 | Llama-3 70B -5; 25/80.
- Quotes:
  - Sec 2.3: "We then select the single most effective vector 𝐫 i ∗ ( l ∗ ) from this set by evaluating each candidate vector over validation sets 𝒟 harmful (val) and 𝒟 harmless (val) ."
  - Sec 3: "For each model, we extract a single difference-in-means vector 𝐫 via the methodology described in § 2.3 ."
  - Sec 2.4: "Note that for activation addition, we intervene only at layer l , and across all token positions." (Eq. 3: x(l)' <- x(l) + r(l), no coefficient)

## 5. TruthX — Zhang et al., ACL 2024
- arXiv 2402.17811; read: https://arxiv.org/html/2402.17811
- Models: Llama-2-7B-Chat (main/all analyses) + 13 LLMs in Fig. 3 / Table 9: Llama-1-7B, Alpaca-7B, Llama-2-7B, Llama-2-7B-Chat, Llama-2-13B-Chat, Vicuna-7B, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1/v0.2, Baichuan2-7B-Base/Chat, Chatglm3-6B-Base/Chat.
- Knob: editing strength alpha (x_hat = x + alpha * Delta) and number of edited modules k (top-k by probing accuracy on validation set).
- Values: k = 10; alpha = 1.0 (open-ended generation) and alpha = 4.5 (multiple-choice). Table 6 sweeps k and alpha on Llama-2-7B-Chat.
- How chosen: "Based on the performance on validation set" (2-fold validation on TruthfulQA), described for the main Llama-2-7B-Chat setup. Choice of which k layers is per model (by that model's probing accuracy); alpha is per TASK (open-ended vs MC).
- Same or per model: For the 13-model experiment the paper does not state whether alpha / k were re-tuned per model -> not stated (the only stated values are the single k / alpha pair given in the Configuration paragraph). Per task: yes (1.0 vs 4.5).
- Quotes:
  - Sec 4 Configuration: "Based on the performance on validation set, we set the number of editing layers k = 10 and the editing strength α = 1.0 and α = 4.5 for the open-ended generation and multiple-choice task."
  - Sec 3: "In practice, TruthX edits the LLM's internal representations on the selected top k layers from all attention and FFN layers based on the probing accuracy of each layer on the validation set."
  - Sec 5.1 (multi-LLM): "To validate the effectiveness across various LLMs, we apply TruthX to 13 advanced LLMs and show the improvements on TruthfulQA benchmark in Figure 3 ." (no hyperparameters stated there)

## 6. LoFiT — Yin et al., NeurIPS 2024
- arXiv 2406.01563; read: https://arxiv.org/html/2406.01563
- Models: Gemma-7B, Llama 2-7B, Llama 2-13B (base versions); Llama 2-7B-Chat for OOD study.
- Knob: LoFiT has no inference-time scalar; the offset biases are learned (cross-entropy). The "how much" knobs are: K (number of heads), L1 weight lambda, learning rates. Baselines ITI and RepE have a scaling factor alpha.
- How chosen: LoFiT hyperparameters by grid search, per model and per task (Table 6). K fixed as 3% (or 10%) of heads, which gives different K per model. Baselines: alpha tuned on a validation set of each dataset; the resulting value was the same across settings (ITI alpha = 15, RepE alpha = 5).
- Same or per model: LoFiT: PER MODEL x PER TASK (grid search). K: same fraction (3%), different absolute K: Gemma-7B K=16, Llama 2-7B K=32, Llama 2-13B K=48. Baseline alpha: tuned per dataset, but one value (ITI 15, RepE 5) reported as working across all settings.
- Per-model values (Table 6, columns: head-selection lambda, head-selection LR, BT 3% LR, BT 10% LR): Gemma-7B TruthfulQA 5e-4, 5e-3, 2e-2, 8e-3; MQuAKE 5e-4, 5e-3, 8e-3, 8e-3; CLUTRR 5e-3, 5e-4, 1e-2, 1e-2 | Llama 2-7B TruthfulQA 5e-4, 5e-3, 1e-2, 5e-3; MQuAKE 1e-3, 5e-3, 1e-2, 5e-3; CLUTRR 5e-3, 5e-4, 1e-2, 5e-3 | Llama 2-13B TruthfulQA 1e-3, 1e-3, 2e-2, 5e-3; MQuAKE 1e-3, 1e-3, 8e-3, 5e-3; CLUTRR 1e-3, 1e-3, 1e-2, (cut off in extracted text).
- Quotes:
  - App. D.1: "Because LoFiT for different models involves different numbers of learned parameters, we did not use the same set of hyperparameters throughout; instead, we tuned the following hyperparameters with grid search."
  - Sec 4: "we select 3 % attention heads for each model for LoFiT and ITI: precisely, K = 16 for Gemma-7B, K = 32 for Llama 2-7B, and K = 48 for Llama 2-13B."
  - App. D (ITI baseline): "We tuned α on a validation set of each dataset and we found that α = 15 worked robustly across all settings." (RepE baseline): "we tuned the scaling factor α of the offset vectors on a validation set of each dataset and found that α = 5 worked best across settings."

## 7. ReFT / LoReFT — Wu et al., NeurIPS 2024
- arXiv 2404.03592; read: https://arxiv.org/html/2404.03592
- Models: LLaMA-1 7B/13B, Llama-2 7B, Llama-3 8B (commonsense / arithmetic / instruction following), RoBERTa-base / RoBERTa-large (GLUE).
- Knob: no inference-time scalar strength (interventions are trained). Analogous knobs: layers L, prefix/suffix positions p, s, tying, rank r, and training hyperparameters.
- How chosen: tuned on development sets (300 held-out GSM8K training examples; Alpaca-52K for instruction tuning; GLUE validation set per task).
- Same or per model: For LLaMA decoder models, SAME — hyperparameters selected on LLaMA-7B and transferred to LLaMA-13B (and the commonsense settings reused across benchmarks) without additional tuning. Instruction-following: tuned only with LLaMA-7B, then applied to Llama-2 7B. For Llama-2 7B / Llama-3 8B in commonsense the paper says it replicates the same setup; no separate tuning described (paraphrase of what is stated). For RoBERTa (GLUE): PER MODEL x PER TASK ("hyperparameter tuning with RoBERTa-base and RoBERTa-large for each task individually").
- Quotes:
  - App. D.1: "We select the hyperparameters using LLaMA-7B, and apply the same settings to LLaMA-13B without additional tuning."
  - App. D.1: "We finetune LLaMA-7B on Alpaca-52K [ Taori et al., 2023 ] to select hyperparameters."; Sec 4.4: "We do hyperparameter-tuning on the unseen instruction-following dataset Alpaca-52K [ Taori et al., 2023 ] with only LLaMA-7B to prevent test-set hill-climbing."
  - App. D.1: "We conduct hyperparameter tuning with RoBERTa-base and RoBERTa-large for each task individually."

## 8. AxBench — Wu et al., ICML 2025
- arXiv 2501.17148; read: https://arxiv.org/html/2501.17148
- Models: Gemma-2-2B-it (layers 10, 20) and Gemma-2-9B-it (layers 20, 31); 4 model x layer settings.
- Knob: "steering factor"; actual magnitude alpha = steering factor x maximal activation aggregated over the concept-detection evaluation set (for SAEs, Neuronpedia max activation per concept). Steering vectors unit-normed. Grid: {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0} (14 factors); SAE clamping grid {0.4 ... 100.0}.
- How chosen: for every concept separately and for each method separately, pick the factor with the best LLM-judge overall steering score on one half of the Alpaca-Eval instructions; evaluate on the other half. Training hyperparameters tuned per method on a 10-concept Concept10 set.
- Same or per model: PER CONCEPT x PER METHOD x PER model/layer setting (selection is done within each setting; Fig. 8 shows the distribution of optimal factors "across the 4 tasks"). Also the magnitude is scaled per concept by that concept's max activation.
- Quotes:
  - Sec 5.2: "We select the optimal steering factor for each method independently for every concept based on which factor achieves the highest overall steering score, as given by the LLM judge. Our actual steering magnitude (i.e., α , as described in Section 4 ) is the product of the steering factor and the maximal activations aggregated over the evaluation dataset for concept detection."
  - Sec 3: "To ensure a fair comparison, we partition our instructions into two equally sized sets, selecting the best factor from one set and evaluating it on the holdout set."
  - App. K: "For steering factors, we enumerate factors from { 0.2 , 0.4 , 0.6 , 0.8 , 1.0 , 1.2 , 1.4 , 1.6 , 1.8 , 2.0 , 2.5 , 3.0 , 4.0 , 5.0 } ."

## 9. Spectral Editing of Activations (SEA) — Qiu et al., NeurIPS 2024
- arXiv 2405.09719; read: https://arxiv.org/html/2405.09719
- Models: LLaMA-2-Chat-7B (main), plus LLaMA-2-Chat-13B, LLaMA-2-Chat-70B, Gemma-it-2B, Gemma-it-7B, Mistral-7B (Table 3; six LLMs).
- Knob: no additive scalar; edit strength is controlled by K (explained-variance ratio kept/removed by the spectral projections), L (number of top layers edited), N (number of demonstrations), and for Phi-SEA the feature function (and its alpha hyperparameter).
- How chosen: for LLaMA-2-Chat-7B truthfulness, "two-fold cross-validation on TruthfulQA following Li et al."; sweep of K in {95%, 99%, 99.5%, 99.99%} and L in {1..5} shown in App. Fig. 8 (LLaMA-2-Chat-7B). Other models: "best setting" reported per model (selection procedure for these not spelled out beyond "best").
- Same or per model: PER MODEL (and per task, per variant). Per-model values given only for fairness (App. H.2); truthfulness settings for the five non-7B models are not stated.
- Per-model values (App. H.1/H.2):
  - LLaMA-2-Chat-7B: truthfulness linear SEA K=99.8%, top 21 layers, N=2000; fairness linear K=99.9%, top 3 layers, N=1000; Phi-SEA fairness K=99.99%, top 2 layers (ELU: top 6 layers).
  - LLaMA-2-Chat-13B (fairness): linear K=99.9%, top 1 layer; Phi-SEA sq-exp, K=99.99%, top 2 layers.
  - LLaMA-2-Chat-70B (fairness): linear K=99.9%, top 1 layer; Phi-SEA tanh, K=99.9%, top 1 layer.
  - Gemma-it-2B (fairness): linear K=99.9%, top 3 layers; Phi-SEA ELU, K=99.99%, top 1 layer.
  - Gemma-it-7B (fairness): linear K=99.9%, top 2 layers; Phi-SEA sq-exp, K=99.99%, top 3 layers.
  - Mistral-7B (fairness): linear K=99.9%, top 1 layer; Phi-SEA sq-exp, K=99.99%, top 1 layer.
  (All fairness settings use N=1000.)
- Quotes:
  - App. H.1: "We set the hyperparameter K = 99.8 % and we edit the top 21 layers. All hyperparameters are determined with two-fold cross-validation on TruthfulQA following Li et al. [21] ."
  - App. H.2: "Gemma-it 7B Model. For the best linear SEA setting, we use 1000 pairs of demonstrations randomly sampled from our training split of BBQ to obtain the editing projections for fairness. We set the hyperparameter K = 99.9 % and we edit the top 2 layers."
  - App. H.2: "Mistral 7B Model. For the best linear SEA setting, ... We set the hyperparameter K = 99.9 % and we edit the top 1 layer."

## 10. In-context Vectors (ICV) — Liu et al., ICML 2024
- arXiv 2311.06668; read: https://arxiv.org/html/2311.06668
- Models: LLaMA-7B, LLaMA-13B, Falcon-7B, Vicuna-7B.
- Knob: scaling factor / step size lambda (h~ = h + lambda * h_ICV at every layer), plus an adaptive rescaling lambda := lambda * (1 + max{0, cos(h, -Delta h)}).
- Values: default lambda = 0.1 for all models and tasks "if not otherwise specified"; exceptions: an unpaired variant with lambda = 0.12; jailbreak on Vicuna-7b reported at lambda = 0.10 / 0.18 / 0.20; lambda varied in Fig. 2 / Fig. 4 analyses; vector-arithmetic demos (Falcon-7B) tune lambda per vector.
- How chosen: fixed default (no selection procedure described for the 0.1 default); sweeps shown as analyses.
- Same or per model: SAME default (lambda = 0.1) across Falcon-7B, LLaMA-7B, LLaMA-13B in the main tables (e.g. "Falcon-7b (ICV, λ = 0.1 )", "Llama-7b (ICV, λ = 0.1 )"). Per-vector tuning only in the qualitative arithmetic demos.
- Quotes:
  - App. (implementation): "We set the number of components for PCA as 1, and the scaling factor λ for the “in-context” vector as 0.1 if not otherwise specified."
  - Sec 3: "λ is the step size which is a hyperparameter that controls the strength of applying the task."
  - Fig. 3 caption (arithmetic demos): "The same Falcon-7B model is used across all examples, with no additional training. ... We tune the strength λ ’s for each vector."

## 11. Adaptive Activation Steering (ACT) — Wang et al., WWW 2025
- arXiv 2406.00034; read: https://arxiv.org/html/2406.00034
- Models: LLaMA-7B (primary), LLaMA 2-7B, Alpaca-7B, Vicuna-7B, LLaMA 2-7B-Chat, LLaMA 3-8B (Table 2); LLaMA-13B/33B/65B for scalability (Table 7).
- Knob: steering intensity alpha * (1 - p(a) + beta), i.e. base strength alpha, offset beta, adaptively scaled per head/per token by the probe output; Top-K heads; number of clusters C.
- Values (App. C, Table 8): Few-shot setting: alpha = 15, beta = 0.1, Top-K heads = 24, C = 2. Full-data setting: alpha = 12, beta = 0, Top-K = 24, C = 3.
- How chosen: not stated (the table just lists values "required to reproduce the experiments"). Heads chosen per cluster by probe accuracy on the validation set. Adaptive part is per-activation by design ("tuning-free" refers to this).
- Same or per model: A single hyperparameter table with one column per data setting (few-shot / full-data), no per-model columns. Whether the same alpha was used for every model is not explicitly stated; no per-model values are given. -> one reported set per data setting; per-model: not stated.
- Quotes:
  - App. C: "In this section, we provide the hyperparameters required to reproduce the experiments. For the 7B model, the experiments can be conducted using a single NVIDIA 3090 GPU." Table 8: "α 15 12 β 0.1 0 Top-K heads 24 24 C 2 3" (Few-shot / Full-data columns).
  - Sec 3.3: "Here, α and β are hyperparameters, and v c , l h is the steering vector."
  - Sec 3.3: "For every cluster, we meticulously select the top K heads based on the accuracy of the corresponding probes on the validation set."

## 12. SADI (Semantics-Adaptive Dynamic Intervention) — Wang et al., ICLR 2025
- arXiv 2410.12299; read: https://arxiv.org/html/2410.12299
- Models: LLaMA2-7b-chat (main, 11 tasks), BLOOMZ-7b, Mistral-7b, Falcon-7b-instruct (COPA, BoolQ, NLI; Table 5), BLOOMZ series smaller sizes (560m, 1b, ...; Table 6).
- Knob: delta (strength of intervention along the input's own masked activation), K (number of top elements / heads).
- Values: no numeric delta / K values found in the text (only Fig. 2 sweeps on LLaMA2-7b-chat).
- How chosen: "hyperparameter sweep"; paper states optimal settings vary markedly across tasks and recommends searching on each task's validation set.
- Same or per model: PER TASK stated (for LLaMA2-7b-chat). For the other three backbones: not stated whether delta/K were re-swept; no per-model values given.
- Quotes:
  - Sec 3: "We perform a hyperparameter sweep to empirically determine their optimal values. Detailed analysis of the hyperparameter selection is provided in Section 4.3 ."
  - Sec 4.3 "Hyperparameters are task-specific.": "Results indicate that optimal settings for these hyperparameters markedly vary across different tasks. ... For precise task performance optimization, it is recommended to search optimal hyperparameters using data from the validation sets specific to each task."
  - Sec 4 baselines: "We sweep the hyperparameters of the heads involved and the strength of intervention to optimize results." (ITI baseline)

## 13. Truth Forest (TrFr) — Chen et al., AAAI 2024
- arXiv 2312.17484; read: https://arxiv.org/html/2312.17484
- Models: LLaMA-7B (primary), Llama 2-7B, Alpaca-7B, Vicuna-7B, Llama 2-Chat-7B.
- Knob: intervention strength alpha (times per-head std sigma, ITI-style), Top-K heads, number of orthogonal directions.
- Values (App. E, Table 8): Few-shot: alpha = 15, Top-K heads = 48, Num. Dir. = 3; Full-data: alpha = 10, Top-K = 48, Num. Dir. = 3 (plus probe-training lambda 5e-2, mu 1e-4, LR 1e0, 20 iterations).
- How chosen: not explicitly stated for the main alpha; calibration via CE/KL "following Li et al." is mentioned; Table 7 reports "optimal α values" per direction type from 2-fold cross-validation (LLaMA-7B).
- Same or per model: One hyperparameter table with columns per data setting (few-shot / full-data), no per-model columns; per-model tuning not stated. -> one reported set per data setting; per-model: not stated.
- Quotes:
  - App. E: "In this section, we provide the hyperparameters required to reproduce the experiments." Table 8: "α 15 10 ... Top-K heads 48 48 Num. Dir. 3 3" (Few-shot / Full-data).
  - Table 7 caption: "Comparison of standard deviation for various intervention directions and their corresponding optimal α values. Different α are used in the Few-shot setting and complete data for Trfr, 15, and 10, respectively. The results above are obtained from 2-fold cross-validation."
  - Sec 5 Measuring: "Following ( Li et al. 2023b ) , we calibrate intervention strength using Cross Entropy ( CE ) and Kullback–Leibler divergence ( KL ) to measure deviation from the original generation distribution."

## 14. Improving Instruction-Following through Activation Steering — Stolfo et al., ICLR 2025
- arXiv 2410.12877; read: https://arxiv.org/html/2410.12877
- Models: Phi-3 Mini (instruct), Gemma 2 2B IT, Gemma 2 9B IT, Mistral 7B Instruct v0.1; base Gemma 2 2B / 9B for cross-model transfer.
- Knob: steering weight c (vector = c * unit direction u_l), and the layer l.
- How chosen:
  - Format instructions: c computed per input (Eq. 2: c = z_bar - x'^T u_l), i.e. maps the residual-stream projection to its mean on inputs that contain the instruction — no hand-set scalar.
  - Length instructions: c set manually over a range of values (shown as a dose-response, Fig. 5).
  - Word inclusion/exclusion: Eq. 2 plus a small grid search over neighbouring values on a held-out set.
  - Layer: per model, per instruction, per setting (with/without text instruction), by sweeping layers on a validation set with a perplexity-based quality filter; if no layer helps, no steering.
- Same or per model: PER MODEL x PER INSTRUCTION x PER SETTING (layer; Tables 8-9); c is computed per input from that model's own activations (format) or grid-searched (words).
- Per-model values (Table 8, extract, layer index, w/o Instr. / w/ Instr.): Capital Word Freq.: Phi-3 18/20, Gemma-2-2B 21/23, Mistral-7B 12/24, Gemma-2-9B 16/-; Capitalize: Phi-3 28/22, Gemma-2-2B 11/11, Mistral-7B 28/30, Gemma-2-9B 28/8.
- Quotes:
  - Sec 2: "For format instructions, we use a systematic scaling approach where the value of c is selected to ensure that the residual stream activations are mapped to their mean value on inputs that contain the instruction in question."
  - Sec 2: "Finally, for word-specific constraints, we compute the weight using Eq. 2 and additionally perform a small grid search over neighboring values on a held-out set of examples to fine-tune the steering effect."
  - Table 8 caption: "Layer indices used for steering across different models and evaluation settings (with and without explicit instructions) for each instruction in the format subset."
  - App. E: "Among the remaining configurations, we select the one with the highest validation accuracy."

## 15. Analyzing the Generalization and Reliability of Steering Vectors — Tan et al., NeurIPS 2024
- arXiv 2407.12404; read: https://arxiv.org/html/2407.12404
- Models: Llama-2-7b-Chat, Qwen-1.5-14b-Chat (main); appendix layer sweeps also on Llama-2-70B and Gemma-2-2B-IT.
- Knob: multiplier lambda (lambda * v_L added at the last token position of layer L); steering vectors are raw mean-differences (not unit-normed). Values: Lambda = {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5}; the evaluation metric (steerability) is the slope over this whole range, so no single best lambda is chosen.
- How chosen: multiplier range fixed by hand at (-1.5, 1.5), with an ablation over other ranges (App. D.8). Layer: sweep over all layers on the validation split, per model.
- Same or per model: Multiplier range SAME across models; layer PER MODEL (Llama-2-7b: 13; Qwen-1.5-14b: 21; appendix: Llama-2-70B 30; Gemma-2-2B-IT 14). Effective magnitude is the model's own mean-difference norm.
- Quotes:
  - Sec 4: "In order to determine the optimal layer, we sweep over all layers using the validation split. ... Thus, we fix layer 13 for Llama and layer 21 for Qwen for all subsequent experiments."
  - App. D.8: "In our experiments, we fix a range of ( − 1.5 , 1.5 ) within which we select multipliers to perform contrastive activation addition."
  - App. (aggregation): "we focus on mean-difference as it naturally results in steering vectors which have an appropriate magnitude relative to model activations , such that a steering multiplier of 1 has the same semantic meaning across different layers and different concepts."

---
# Additional multi-model papers

## 16. Conditional Activation Steering (CAST) — Lee et al., ICLR 2025 (Spotlight, per arXiv comments field)
- arXiv 2409.05907; read: https://arxiv.org/html/2409.05907
- Models (App. C.1 table): Qwen 1.5 Chat 1.8B, Qwen 1.5 Chat 32B, LLaMA 3.1 Inst, NeuralDaredevil, Hermes 2 Pro, OLMo SFT (further rows such as Zephyr are mentioned in text; table extraction may be truncated).
- Knob: behaviour-vector strength alpha + behaviour layers; condition threshold theta + condition layer + comparison direction.
- How chosen: behaviour alpha and layers searched manually; condition (theta, layer, direction) by grid search maximizing F1.
- Same or per model: PER MODEL (and per condition). Values (App. C.1, Behavior (Layers, Strength alpha) for "Add Refusal"): Qwen 1.5 1.8B (10-20, 4); Qwen 1.5 32B (36-61, 14); LLaMA 3.1 Inst (17-24, 1.7) and Remove Refusal (14-28, -1); NeuralDaredevil (15-31, 1.5); Hermes 2 Pro (15+17-24, 1.7); OLMo SFT (12-15+16-28 interval 2, 4). Example condition: Qwen 1.5 1.8B harmful (layer 8, > 0.031).
- Quotes:
  - Sec 3.1: "For the behavior vector, similar to other works in activation steering, we manually search for the appropriate intervention strength and layers."
  - Sec 3.1: "For the condition vector, we use a grid search (Appendix C.2 ) algorithm that determines the best threshold, layer, and comparison direction ( > or < )."
  - App.: "It is important to note that there isn't a universally applicable range for the grid search ... The cosine similarity values can vary drastically depending on the specific model architecture (more dependent) and the condition being explored (less dependent)."

## 17. Activation Addition (ActAdd) — Turner et al., "Steering Language Models With Activation Engineering" (venue not verified from the arXiv page)
- arXiv 2308.10248; read: https://arxiv.org/html/2308.10248
- Models: GPT-2-XL (main qualitative/perplexity), OPT-6.7B and LLaMA-3-8B (toxicity, sentiment), GPT-J-6B (perplexity appendix), Llama-1-13B (appendix examples).
- Knob: injection coefficient c and injection layer l.
- How chosen: "simple grid search, usually between c in [3, 20] and l in [6, 24]"; "hyperparameter tuning to select c and also the injection layer l".
- Same or per model: The paper says c and l are tuned, and that reused (l, c) works "for a given frozen model and level of abstraction in the task" (i.e. per model and per task). Per-model values for OPT / LLaMA-3 in Table 3 were not found in the extracted text -> per-model numbers not stated (as far as found).
- Quotes:
  - Sec 3: "It's typical for the intervention strength c to have a magnitude less than 15. We perform hyperparameter tuning to select c and also the injection layer l ."
  - App. (ActAdd hyperparameters): "This section does not have complete statistics . We perform simple grid search, usually between c ∈ [ 3 , 20 ] and l ∈ [ 6 , 24 ] ."
  - App.: "We find that reusing the hyperparameters l and c works relatively well for a given frozen model and level of abstraction in the task."

## 18. Persona Vectors — Chen et al., 2025 (venue not verified from the arXiv page)
- arXiv 2507.21509; read: https://arxiv.org/html/2507.21509
- Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.
- Knob: steering coefficient alpha (h <- h + alpha * v at layer l) and layer.
- How chosen: layer per model and per trait by steering at each layer "using the same coefficient" and picking the layer with the highest trait-expression score. Coefficient is mostly reported as a sweep (curves, one line per coefficient); some fixed values used in specific experiments (e.g. coefficient 1.5; alpha = 2.5 for SAE-feature tables; preventative steering 0.5 / 1.25).
- Same or per model: Layer PER MODEL x PER TRAIT (Qwen: layer 20 evil & sycophancy, layer 16 hallucination; Llama: layer 16 for all three). Coefficient: shown as the same sweep for both models; no per-model optimal coefficient stated.
- Quotes:
  - App. B.4: "To identify the most informative layer for each trait, we apply steering individually at each layer using the same coefficient, and select the layer that elicits the highest trait expression score."
  - App. B.4: "We select layer 20 for both evil and sycophancy , and layer 16 for hallucination . Results on Llama are shown in Figure 13 , where we find that layer 16 is the most informative for all three traits."
  - Limitations: "our method requires selecting an appropriate steering coefficient."

(Checked but excluded: "Steering Without Side Effects" arXiv 2406.15518 — experiments are essentially on Llama-2-7b-chat only.)

---
# Tally (18 papers; all multi-model; all full texts opened via arxiv.org/html)

Strength knob (coefficient / multiplier / alpha or its analogue):
- PER MODEL or finer, explicitly stated (6): Arditi refusal (no scalar; layer + token position per model on validation set, raw diff-in-means added), LoFiT (grid search per model x task), AxBench (per concept x method x model/layer), SEA (K and #layers per model and per task), Stolfo (c computed per input; layer per model x instruction; grid search for word constraints), CAST (alpha + layers per model; threshold per model x condition).
- ONE value / one grid for all models, explicitly (6): CAA (same multiplier grid; layer per model), RepE (coefficient 0.25 "same" for 7B and 13B; layers per model), ICV (lambda = 0.1 default), ReFT (tuned on LLaMA-7B, applied to LLaMA-13B unchanged; RoBERTa is per task), Tan et al. (fixed range -1.5..1.5; layer per model), Persona Vectors (same coefficient sweep; layer per model x trait).
- Tuned on one main model, values for the other models NOT STATED (6): ITI, TruthX (alpha per task), ACT (one set per data setting), SADI (per task), Truth Forest (one set per data setting), ActAdd.
- Single-model: none of the 18 (one candidate, Steering Without Side Effects, excluded as single-model).

Layer, separately: chosen per model in Arditi, CAA, RepE, Tan, Persona, Stolfo, CAST, SEA, AxBench (fixed two layers per model), TruthX/ITI/ACT/TrFr (heads/layers by each model's own probe accuracy).

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-20 | 建立本文件 | 赵磊（Fable 5.2 代笔） |
