#!/usr/bin/env python3
"""Read-only inventory of R20 repaired TF/QA frozen layers; writes only R21."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def canonical(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def order(e, c):
    if not e or not c:
        return None
    assert 'layers' not in e, 'Multilayer E requires an explicit schedule, not a single order'
    return 'C_before_E' if c['layer'] < e['layer'] else 'E_before_C' if c['layer'] > e['layer'] else 'same_layer_E_then_C'


def eshow(e):
    return '—' if not e else 'L%d / %g' % (e['layer'], e['alpha'])


def cshow(c):
    return '—' if not c else 'L%d / %+g' % (c['layer'], c['alpha'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('/tmp/td_r20_20261107/Truth_Decoupling/claude_gpt_review'))
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    if a.output.resolve().is_relative_to(a.source.resolve()):
        raise ValueError('R20 source tree is read-only; output must be outside it')
    audit = a.source / 'from_gpt/audit_20261107_round20'
    sources, frozen, pools, anchors, e31 = {}, [], [], [], []
    for task in ('tf', 'qa'):
        for model in ('llama3.1-8b', 'qwen3-8b'):
            p = audit / f'{task}_ability/results/{model}/{task}_ability_full'
            for name in ('bank.json', 'bank.npz', 'data.json', 'selection.json', f'{task}_tune.json'):
                sources[str((p / name).relative_to(a.source))] = sha(p / name)
            bank, select, tune = read(p / 'bank.json'), read(p / 'selection.json'), read(p / f'{task}_tune.json')
            assert bank['model'] == model and bank['pilot'] is None
            assert bank['config']['evaluation_tasks'] == [task]
            assert bank['bank_sha256'] == sha(p / 'bank.npz') and bank['data_sha256'] == sha(p / 'data.json')
            for obj in (bank, select):
                assert obj['fingerprint'] == canonical({k: v for k, v in obj.items() if k != 'fingerprint'})
            assert select['bank_fingerprint'] == bank['fingerprint'] == tune['bank_fingerprint']
            assert tune['completed'] is True
            specs = {v['name']: v for v in bank['conditions'][task]}
            assert len(specs) == len(bank['conditions'][task])
            assert set(specs) == set(tune['rows'])
            for method in ('das', 'md'):
                ec = [v for v in specs.values() if v['method'] == method and v['kind'] == 'EC']
                counter = collections.Counter(order(v['e'], v['c']) for v in ec)
                pairs = sorted({(v['e']['layer'], v['c']['layer']) for v in ec})
                by_e = collections.defaultdict(set)
                for v in ec:
                    by_e[canonical(v['e'])].add(order(v['e'], v['c']))
                pools.append(dict(task=task, model=model, method=method, n_layers=bank['num_layers'],
                    e_choices=bank['e_choices'][task][method], c_layers=bank['c_layers'][task][method],
                    c_alphas=bank['config']['c_alphas'], ec_conditions=len(ec), layer_pairs=pairs,
                    order_counts={k: counter[k] for k in ('C_before_E', 'E_before_C', 'same_layer_E_then_C')},
                    same_e_has_both_cross_layer_orders=any({'C_before_E', 'E_before_C'} <= v for v in by_e.values())))
            for coverage, choices in select['tasks'][task]['choices'].items():
                for method in ('das', 'md'):
                    for selector, suffix in (('commitment', ''), ('simple', '_simple')):
                        selected = {kind: choices[f'{method}_{kind}{suffix}'] for kind in ('E', 'C', 'EC')}
                        s = {kind: specs[name] if name else None for kind, name in selected.items()}
                        e = s['E']['e'] if s['E'] else None
                        ec_e = s['EC']['e'] if s['EC'] else None
                        ec_c = s['EC']['c'] if s['EC'] else None
                        frozen.append(dict(task=task, model=model, method=method, selector=selector,
                            coverage=coverage, condition_names=selected, e_only=e,
                            c_only=s['C']['c'] if s['C'] else None, ec_e=ec_e, ec_c=ec_c,
                            ec_order=order(ec_e, ec_c), same_e_config=e == ec_e if e and ec_e else None,
                            infeasible=e is None or ec_e is None))
            if task == 'tf':
                for method in ('das', 'md'):
                    name = select['tasks'][task]['choices']['0.5'][f'{method}_E']
                    e = specs[name]['e']; l = e['layer']; last = bank['num_layers'] - 1
                    csites = [int(bank['num_layers'] * .25) - 1, int(bank['num_layers'] * .75) - 1]
                    assert csites[0] < l < csites[1]
                    for cl in csites:
                        assert bank['c_estimation'][task]['methods'][method][str(cl)]['available']
                    anchors.append(dict(model=model, method=method, e_condition=name, e=e,
                        source_rule='repaired TF frozen 50% commitment E-only, chosen before new data',
                        c_layers=csites, c_alphas=[-1,1], note='Each C uses its own saved layer vector and scale; E remains identical.'))
    for model in ('llama3.1-8b', 'qwen3-8b'):
        p = audit / f'causal_C/results/{model}/full192/plan.json'
        sources[str(p.relative_to(a.source))] = sha(p)
        j = read(p)
        e31.append({k: j[k] for k in ('model', 'protocol', 'candidate_answer_rule', 'claims_limit', 'layer_rule', 'all_layers', 'alphas')})
    for rel in ('from_claude/code/core/joint_steering.py', 'from_claude/code/core/intervene.py', 'from_claude/code/scripts/e26_matched_coverage.py'):
        sources[rel] = sha(a.source / rel)
    comparable = [r for r in frozen if not r['infeasible']]
    result = dict(protocol='r21_existing_e_c_layer_inventory_v1', source_root=str(a.source),
        layer_indexing='zero-based; L0 is the first decoder block',
        source_sha256=sources, candidate_pools=pools, frozen_comparisons=frozen,
        n_frozen=len(frozen), n_comparable=len(comparable),
        n_same_e=sum(r['same_e_config'] for r in comparable),
        n_changed_e=sum(not r['same_e_config'] for r in comparable),
        frozen_ec_order_counts=dict(collections.Counter(r['ec_order'] for r in comparable)),
        e31_scope=e31, proposed_tf_fixed_e_anchors=anchors)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'inventory_e_c_layers.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    fields = ['task','model','method','selector','coverage','E_only','C_only','EC_E','EC_C','EC_order','same_E','E_condition','C_condition','EC_condition']
    with (a.output / 'frozen_e_c_layers.csv').open('w', newline='') as f:
        w = csv.writer(f);w.writerow(fields)
        for r in frozen:
            w.writerow([r['task'],r['model'],r['method'],r['selector'],r['coverage'],eshow(r['e_only']),cshow(r['c_only']),eshow(r['ec_e']),cshow(r['ec_c']),r['ec_order'],r['same_e_config'],*[r['condition_names'][k] for k in ('E','C','EC')]])
    lines = ['# 第 21 轮：既有 E/C 写入层与顺序盘点', '',
        '本报告只读取第 20 轮修正能力提示后的完整 TF（真假判断）与 QA（真实问答）原件，没有修改旧结果，也未开始新实验。层号从 0 开始。', '',
        'DAS（分布式对齐搜索 · 通过交换内部表示估计 E 方向）；MD（配对均值差 · 用成对样本表示的均值之差估计 E 方向）。C 均是首词 Yes−No 分数相关方向，并在该层去掉 E 分量。', '',
        f'48 个冻结比较位置中，{len(comparable)} 个可比较；其中 {result["n_changed_e"]} 个 E+C 与 E 单独组使用了不同的 E 配置。这些是各自调参后方案的比较，不能全解释为“在同一 E 上增加 C”。Qwen QA 的 75% 作答率在调参集不可行，4 个位置保留不可行。', '',
        '## 全部候选覆盖', '',
        '|任务/模型|估计器|E 层及强度|C 层|C→E 条件数|E→C 条件数|同层|固定同一 E 的两种顺序|',
        '|---|---|---|---|---:|---:|---:|---|']
    for r in pools:
        n = r['order_counts']
        lines.append(f'|{r["task"]}/{r["model"]}|{r["method"]}|'+', '.join(eshow(e) for e in r['e_choices'])+f'|{r["c_layers"]}|{n["C_before_E"]}|{n["E_before_C"]}|{n["same_layer_E_then_C"]}|无|')
    lines += ['', '每个层对均已包含 C 强度 −1 与 +1；表中条件数因此是层对数的两倍。全部 8 个任务/模型/估计器单元都只保留 1 个 C 层；没有任何单元在同一固定 E 下覆盖 C 更浅和 C 更深。没有 E、C 同层候选。', '',
        '## 全部冻结 E 与 E+C', '',
        '能力筛题=按独立 Yes−No 能力提示分数排序；简单筛题=真假任务的 True/False 置信度、问答任务的输出平均词分数。每格写作“层 / 强度”。C 单独组与准确条件名另见 CSV。', '',
        '|任务/模型|估计器|筛题|作答率|E 单独|组合中的 E|组合中的 C|实际层序|同一 E|',
        '|---|---|---|---|---|---|---|---|---|']
    onames = {'C_before_E':'C→E','E_before_C':'E→C','same_layer_E_then_C':'同层 E→C',None:'不可行'}
    for r in frozen:
        lines.append('|'+ '|'.join([r['task']+'/'+r['model'], r['method'], '能力' if r['selector']=='commitment' else '简单', str(int(float(r['coverage'])*100))+'%', eshow(r['e_only']), eshow(r['ec_e']), cshow(r['ec_c']), onames[r['ec_order']], '—' if r['same_e_config'] is None else '是' if r['same_e_config'] else '**否**'])+'|')
    lines += ['', '## 代码中真正的时间顺序', '',
        '不同层按模型从浅到深的实际执行顺序写入，声明列表先放 E 再放 C 并不让深层 E 提前发生。同一层才按列表执行 E→C。真假判断中，E 与 C 都在输入末词写一次；答案提示与能力提示各做一次独立前向计算。', '',
        '问答中，C 只在读入问题时写一次，E 会在之后每步生成继续写入。因此表里的 C→E/E→C 只描述首次读入问题的顺序，不能把整段生成说成单次“先 C 后 E”。', '',
        '同层 E 使用沿自身方向放大，C 使用加法平移。若 C 与该层 E 严格正交，这两个运算在精确算术下可交换；只调换同层调用顺序预期主要检验数值舍入，不能直接当成两种不同的科学机制。不同层有中间网络变换，才有值得实际检验的先后效应。', '',
        '## E31 能回答什么', '',
        'E31（只改门控的层诊断）冻结原模型和 E 模型的候选答案，只在独立能力提示中改变 C 的写入层。它没有在真正判断 True/False 或生成答案的前向计算中改变 C 层；并且使用的是修正语义之前的方向库。它不能替代本轮答案写入层与先后顺序诊断。', '',
        '## 建议的真假任务最小成对诊断', '',
        '先沿用修正版主表 50% 能力筛题的 E 单独冻结配置，所有配置都在内部层，可在其前后各放一个 C。E 层、强度、向量和中心全程固定；C 浅层与深层分别取全模型 25% 和 75% 深度（层数乘比例后减 1），即 Llama 的 L7/L23、Qwen 的 L8/L26。此规则先写死，不按新结果改层。各层使用本层已保存的 C 向量和标准差尺度，不把一个层的向量直接搬去另一层。', '',
        '|模型|估计器|固定 E 层/强度|浅层 C→E|深层 E→C|C 强度|',
        '|---|---|---|---|---|---|']
    for r in anchors:
        lines.append(f'|{r["model"]}|{r["method"]}|{eshow(r["e"])}|L{r["c_layers"][0]}|L{r["c_layers"][1]}|−1、+1|')
    lines += ['',
        '每个估计器 10 组：原模型、固定 E、两层×两强度的 C 单独 4 组、相同 C 配置加固定 E 的 4 组。两个估计器共用原模型，合计每个模型 19 组；不扫更多层或强度。先 3 对真假样本核真实写入顺序与输出，再跑既有 192 对调参题，逐组保留，不只报赢家。', '',
        '同时报告全体真假判断正确数、相同实际作答数下的能力筛题与简单筛题四组表。逐层看 E+C−E，也同时看 C−原模型，避免把 C 单独作用当成组合协同。真假两句按原问题成对统计。新数据只作探索诊断，不能把已经看过的调参集或测试集重新称为未看过的确认集。', '',
        '边界必须写清：移动 C 会同时改变写入位置、该层向量和先后顺序；这能检验“换顺序/选层是否有用”，不能单独证明纯时间顺序就是因果来源。Qwen TF 的简单筛题最佳 E 在 L35（末层），它本来就没有更深的 C 位置；不应悄悄换掉 E 后仍声称是在同一主表 E 上验证。上述内部层 E 诊断须单独命名，并保留末层主表 E 结果作参照。', '',
        '现有结果只支持“目前已测配置未显示稳定组合优势”。尚未检验固定 E 的浅/深 C 成对对照，不能推出所有选层与顺序都无效。', '',
        '复核方式：运行 `python3 inventory_e_c_layers.py`；脚本核对源文件摘要、方向库及数据绑定、冻结选择身份与全部调参条件已完成，并在结束时确认所有读入源文件未变。完整源摘要在 JSON。', '']
    (a.output / 'inventory_e_c_layers.md').write_text('\n'.join(lines))
    for rel, digest in sources.items():
        assert sha(a.source / rel) == digest, f'Source changed during inventory: {rel}'
    print(json.dumps({k: result[k] for k in ('n_frozen','n_comparable','n_same_e','n_changed_e','frozen_ec_order_counts')}, ensure_ascii=False))
    print('R20 read-only verified; saved JSON, CSV, Chinese report under', a.output)

if __name__ == '__main__':
    main()
