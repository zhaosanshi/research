#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
out=json.loads((ROOT/'full_independent_recount.json').read_text())
allrows=[r for model in out.values() for r in model['comparisons']]
assert len(allrows)==96 and all(m['pilot_limit'] is None for m in out.values())
fields=['model','method','selector','coverage','k','e_layer','c_layer','c_alpha','order','base','E','C','EC','EC_minus_E','EC_minus_C','C_minus_base','ci_low','ci_high','pair_swap_p','holm48']
with (ROOT/'all_matched_counts.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in allrows:
  row={k:r[k] for k in ['model','method','selector','coverage','k','e_layer','c_layer','c_alpha','order','EC_minus_E','C_minus_base']}
  row.update(r['correct']);row['EC_minus_C']=r['correct']['EC']-r['correct']['C'];row['ci_low'],row['ci_high']=r['question_bootstrap_reselect_ci95_correct_count'];row['pair_swap_p']=r['pair_swap_p_fixed_selection'];row['holm48']=r['holm_within_48_prespecified_points'];w.writerow(row)
lines=['# E38：固定 E 的浅层 C、深层 C 完整结果','',
 '本轮实际完成两个模型各 192 对真假陈述、19 个预定条件。每个条件重新计算真正的真假答案和独立能力提示；没有复用未受干预答案。全部 25%、50%、75% 作答率、两种估计器、C 两层两符号都保留。', '',
 'DAS（分布式对齐搜索 · 用内部表示交换估计证据方向）与 mean-diff（配对均值差 · 用成对表示差估计方向）分别保留。能力筛题按 Yes−No，简单筛题按 True/False 置信度；主要解释按简单筛题。', '',
 '## 先看能支持什么结论','']
for selector,label in [('simple','简单筛题'),('commitment','能力筛题')]:
 rows=[r for r in allrows if r['selector']==selector]
 pos=sum(r['EC_minus_E']>0 for r in rows);zero=sum(r['EC_minus_E']==0 for r in rows);neg=len(rows)-pos-zero
 sig=[r for r in rows if r['holm_within_48_prespecified_points']<.05 and r['EC_minus_E']>0]
 lines += [f'{label}的 48 个预定 E+C 对固定 E 比较中：{pos} 个多答对、{zero} 个持平、{neg} 个少答对；差值范围 {min(r["EC_minus_E"] for r in rows):+d} 到 {max(r["EC_minus_E"] for r in rows):+d} 题。统一多重比较校正后，显著正收益格为 {len(sig)} 个。', '']
effects=json.loads((ROOT/'full_C_norm_and_gate_effects.json').read_text())
for model,entries in effects.items():
 ec=[r for r in entries if r['condition']['kind']=='EC']
 changes=[r['answer_changed'] for r in ec]
 swaps=[r['selected_sets_vs_reference']['simple']['0.5']['newly_selected'] for r in ec]
 lines += [f'{model}：与固定 E 相比，384 句中各组合只有 {min(changes)}–{max(changes)} 句改变真假判断；50% 简单筛题的 192 句交付集合中，只替换了 {min(swaps)}–{max(swaps)} 句。C 确实写入，能力分数也有变化，但这批配置对最终答案和筛题的增量很小。', '']
lines+=['这些数值是已经看过的调参题上的探索诊断，不是新的未见题确认。个别格正向不自动满足“组合胜过任何单方向”；完整表同时保留相同 C 单独组和原模型。','',
 '## 50% 实际作答率：每组都答 192 句','',
 '|模型|估计器|筛题|层序/符号|原模型|固定 E|对应 C|E+C|组合−E|按题重抽重选 95% 区间|校正 p|',
 '|---|---|---|---|---:|---:|---:|---:|---:|---|---:|']
for r in allrows:
 if r['coverage']!='0.5':continue
 c=r['correct'];lo,hi=r['question_bootstrap_reselect_ci95_correct_count'];layout=f'C{r["c_layer"]}→E{r["e_layer"]}' if r['order']=='C_before_E' else f'E{r["e_layer"]}→C{r["c_layer"]}'
 lines.append(f'|{r["model"]}|{r["method"]}|'+('简单' if r['selector']=='simple' else '能力')+f'|{layout} / {r["c_alpha"]:+g}|{c["base"]}|{c["E"]}|{c["C"]}|{c["EC"]}|{r["EC_minus_E"]:+d}|[{lo:g}, {hi:g}]|{r["holm_within_48_prespecified_points"]:.4g}|')
lines+=['','全部 96 个对比及 25%/75% 档位见 `all_matched_counts.csv`；不是只选这张表中的有利结果。','',
 '## 训练尺度校准与完整复核','',
 '|模型|估计器/层序|训练目标相对量|实际重算中位数|负号中位数|正号中位数|冻结系数|',
 '|---|---|---:|---:|---:|---:|---:|']
for model,j in out.items():
 for key,v in j['calibration_independently_recomputed'].items():
  lines.append(f'|{model}|{key}|{100*v["rho"]:.4f}%|{100*v["recomputed_median"]:.4f}%|{100*v["negative_median"]:.4f}%|{100*v["positive_median"]:.4f}%|{v["coefficient"]:.7g}|')
lines += ['',
 '每个模型都只在 192 条训练陈述/96 对问题的无干预答案状态上一次校准，保留原 BF16 类型；全体正负号合并确定共同目标。C 单独和组合共用同一系数，完整调参阶段不再改值。独立脚本直接加载原状态、以 FP32 计算改动并写回 BF16，逐层重算，与冻结目标均在预定误差内。', '',
 '独立核对两个模型各 114 个配额格的正确数与全部入选题号、每题两个上下文的层执行顺序、全部冻结源文件摘要。原模型、DAS E、均值差 E 的完整输出与旧原件重新对照。2000 次重抽以原问题为单位同时抽取真假两句，每次重新筛题；成组精确检验固定观察交付集合，二者口径分开。每种筛题方式的 48 个预定组合比较统一校正。', '',
 '## 结论边界','',
 '这补上了此前缺失的“同一个 E，C 在其前后”的成对比较。旧主表 44 个可比较位置中有 17 个同时换了 E；本轮所有 E 与对应组合中的 E 均完全一致。', '',
 '移动 C 同时改变层位与该层方向。即使训练中位写入量对齐，E 还会改变之后的残差，评估时每题实际相对量不一定完全相等。能力分数的作用也可能不同。因此这是有力度控制的层位布局诊断，不是没有混杂的抽象先后交换。', '',
 'Qwen 原简单筛题最优 E 在末层 L35，无法放置更深 C；本轮固定的是已存在的能力筛题 50% E（DAS L16、均值差 L21），不能冒称比较了末层最优 E 的后层组合。', '',
 '本轮结果仍不能证明所有未测试的层、方向构造与任务都无效；也不把探索中的偶然正格包装成论文贡献已达成。', '']
(ROOT/'results.md').write_text('\n'.join(lines))
print('REPORT WRITTEN',ROOT/'results.md')
