#!/usr/bin/env python3
"""Freeze and report the explicitly exploratory equal-simple-selector extension.
All EC choices come from the original tuning rows; old primary results remain
unchanged. Every missing test condition must be complete before reporting.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import numpy as np

CODE = Path(__file__).resolve().parents[3] / 'from_claude/code/scripts/review_checks'
sys.path.insert(0, str(CODE))
from e26_independent_recount import (audit_run, choose, read, digest, fingerprint,
                                    load_rows, quota, exact, require, ordered)

MODELS = ('llama3.1-8b', 'qwen3-8b')


def build_manifest(root):
    res = {'protocol': 'E26_equal_simple_selector_extension_v1', 'exploratory': True,
           'selection_rule': 'tuning-only delivered correct, existing deterministic intervention-size tie rule',
           'source_commit': '63e3674b6d98729e453a4b259ee7724b9234711a',
           'scope': 'DAS and mean-difference, equal simple-score selector for baseline/E/C/EC; old primary analysis unchanged', 'runs': []}
    for model in MODELS:
        run = root / 'E26_main' / model; b = read(run/'bank.json'); d = read(run/'data.json'); s = read(run/'selection.json')
        row = {'model': model, 'sources': {x: digest(run/x) for x in ['bank.json', 'bank.npz', 'data.json', 'selection.json',
               'tf_tune.json', 'qa_tune_scored.json', 'qa_tune.json']}, 'bank_fingerprint': b['fingerprint'], 'tasks': {}}
        for task in ('tf', 'qa'):
            tune, spec = load_rows(run,b,d,task,'tune',s)
            filename = task + '_test' + ('_scored' if task == 'qa' else '') + '.json'; test = read(run/filename)
            choices = {}; missing = {}
            for q in map(str, b['config']['coverages']):
                choices[q] = {}
                for method in ('das', 'md'):
                    name = choose(tune,spec,method+'_EC_simple',q,b['config']['seed'])
                    choices[q][method+'_EC_simple'] = name
                    if name and name not in test['rows']:
                        missing[name] = spec[name]
            row['tasks'][task] = {'choices': choices, 'missing_condition_specs': list(missing.values()),
                                  'existing_test_source_sha256': digest(run/filename)}
        res['runs'].append(row)
    res['fingerprint'] = fingerprint(res)
    return res


def load_extension(folder, task, manifest_path, manifest_run, source_data, existing_baseline, prior_grade_rows=None):
    filename = task + '_test_extension.json'; raw = read(folder/filename)
    scored_file = 'qa_test_extension_scored.json' if task == 'qa' else filename
    scored = read(folder/scored_file)
    expected_specs = manifest_run['tasks'][task]['missing_condition_specs']; names = {s['name'] for s in expected_specs}
    cohort = source_data[task]['test']; ids = [str(x['pair_id'] if task == 'qa' else x['id']) for x in cohort]
    for artifact in (raw, scored):
        require(artifact['protocol'] == 'E26_equal_simple_selector_extension_v1' and artifact.get('exploratory') is True,
                'extension protocol/scope changed')
        require(artifact['completed'] and artifact['pilot_limit'] is None, 'extension is incomplete or pilot-only')
        require(artifact['cohort_sha256'] == fingerprint(cohort), 'extension full test cohort hash changed')
        require(artifact['model'] == manifest_run['model'] and artifact['bank_fingerprint'] == manifest_run['bank_fingerprint'],
                'extension model/bank changed')
        require(artifact['extension_manifest_sha256'] == digest(manifest_path), 'extension manifest changed')
        require(artifact['condition_specs'] == expected_specs and set(artifact['rows']) == names, 'extension conditions differ')
    if task == 'qa':
        require(scored['generation_sha256'] == digest(folder/filename), 'extension QA grading source changed')
    base = {r['id']: r for r in existing_baseline}
    prior_grades = {}
    for r in (prior_grade_rows or existing_baseline):
        key = (r['id'], r['output'])
        require(key not in prior_grades or prior_grades[key] == r['outcome'], 'prior identical answer grades conflict')
        prior_grades[key] = r['outcome']
    for name, rows in scored['rows'].items():
        require([x['id'] for x in rows] == ids and [x['id'] for x in raw['rows'][name]] == ids,
                'extension must contain every test row exactly once, in source order')
        for r, original, question in zip(rows, raw['rows'][name], cohort):
            require(r['pair_id'] == str(question['pair_id']) and r['outcome'] in ('correct','wrong','abstain'), 'extension row invalid')
            require(all(math.isfinite(r[k]) for k in ('score','simple_score')), 'extension nonfinite score')
            if task == 'qa':
                require(all(r[k] == original[k] for k in ('output','score','simple_score','finished','edits')), 'grading changed generation')
                require(r['outcome'] == r['grade']['label'] and r['grade']['rubric_id'] == 'qa_semantic_v3', 'wrong grading rule')
                require(r['gold_is_abstention'] == base[r['id']]['gold_is_abstention'], 'extension changes main cohort')
                key = (r['id'], r['output'])
                require(key not in prior_grades or prior_grades[key] == r['outcome'], 'extension changes grade for identical question/answer')
                prior_grades[key] = r['outcome']
            else:
                require(type(r['predicted_true']) is bool and r['output'] == ('True' if r['predicted_true'] else 'False'), 'TF output differs')
                require(r['outcome'] == ('correct' if r['predicted_true'] == question['label'] else 'wrong'), 'TF grade differs')
    return {n: [r for r in rs if not r.get('gold_is_abstention')] for n,rs in scored['rows'].items()}


def holm(ps):
    out = [None] * len(ps); running = 0
    for rank,i in enumerate(sorted(range(len(ps)), key=lambda i: ps[i])):
        running = max(running, min(1., (len(ps)-rank)*ps[i])); out[i] = running
    return out


def bootstrap_pair(a, b, q, seed, n_boot=2000):
    """Group-weighted rank reselection; any infeasible replicate suppresses CI."""
    base = sorted(a,key=lambda r:r['id']); index={r['id']:i for i,r in enumerate(base)}
    gs=sorted({r['pair_id'] for r in base}); gi={g:i for i,g in enumerate(gs)}
    rng=np.random.default_rng(seed); draws=rng.integers(0,len(gs),(n_boot,len(gs)))
    weights=np.array([np.bincount(d,minlength=len(gs)) for d in draws])[:,[gi[r['pair_id']] for r in base]]
    ns=weights.sum(1); f=Fraction(str(q)); ks=ns*f.numerator//f.denominator
    counts=[]; valid=[]
    for rows in (a,b):
        ranks=ordered(rows,'simple',seed); w=weights[:,[index[r['id']] for r in ranks]]
        valid.append(w.sum(1)>=ks)
        take=np.minimum(w,np.maximum(0,ks[:,None]-(w.cumsum(1)-w)))
        counts.append(take @ np.array([r['outcome']=='correct' for r in ranks],dtype=np.int64))
    failures=int((~(valid[0]&valid[1])).sum())
    ci=None if failures else np.percentile((counts[0]-counts[1])/ks,[2.5,97.5]).tolist()
    return {'n_resamples':n_boot,'infeasible_resamples':failures,'ci95_accuracy':ci,
            'rule':'resample whole question groups then reselect exact quota; fixed tuning choices'}


def report(root, extension_root, manifest_path, output):
    manifest=read(manifest_path)
    require(manifest==build_manifest(root),'frozen extension differs from independent tuning-only rederivation')
    cells=[];families={};run_sources=[]
    for mr in manifest['runs']:
        model=mr['model'];run=root/'E26_main'/model;folder=extension_root/model
        audit_run(run)
        bank=read(run/'bank.json');data=read(run/'data.json');sel=read(run/'selection.json');seed=bank['config']['seed']
        run_sources.append({'model':model,'extension_sha256':{n:digest(folder/n) for n in
            ('tf_test_extension.json','qa_test_extension.json','qa_test_extension_scored.json')}})
        for task in ('tf','qa'):
            old=read(run/(task+'_test'+('_scored' if task=='qa' else '')+'.json'))
            extra=load_extension(folder,task,manifest_path,mr,data,old['rows']['baseline'], [r for rs in old['rows'].values() for r in rs])
            rows={n:[r for r in rs if not r.get('gold_is_abstention')] for n,rs in old['rows'].items()}
            require(not (set(rows)&set(extra)),'extension overwrites a prior test condition');rows.update(extra)
            groups={r['id']:r['pair_id'] for r in rows['baseline']};n=len(groups)
            for q,picked in sel['tasks'][task]['choices'].items():
                for method in ('das','md'):
                    names={'base':'baseline','E':picked[method+'_E_simple'],'C':picked[method+'_C_simple'],
                           'EC':mr['tasks'][task]['choices'][q][method+'_EC_simple']}
                    deliveries={arm:quota(rows[name],q,'simple',seed) if name else None for arm,name in names.items()}
                    cell={'model':model,'task':task,'coverage':float(q),'method':method,'n':n,
                          'k':int(Fraction(q)*n),'conditions':names,'arms':{}}
                    for arm,d in deliveries.items():
                        cell['arms'][arm]=({'feasible':False} if d is None else {'feasible':True,'correct':sum(v=='correct' for v in d.values()),
                            'answered':sum(v!='abstain' for v in d.values()),'selected_ids':sorted(i for i,v in d.items() if v!='abstain')})
                        if d is not None:
                            rs=rows[names[arm]]; kept=[r for r in rs if d[r['id']]!='abstain']
                            cell['arms'][arm].update(native_answered=sum(r['outcome']!='abstain' for r in rs),
                                native_correct=sum(r['outcome']=='correct' for r in rs),
                                delivered_ufffd=sum('\ufffd' in r['output'] for r in kept),
                                delivered_unfinished=sum(not r['finished'] for r in kept),
                                delivered_correct_ufffd=sum(r['outcome']=='correct' and '\ufffd' in r['output'] for r in kept),
                                delivered_correct_unfinished=sum(r['outcome']=='correct' and not r['finished'] for r in kept))
                    for control in ('base','E','C'):
                        family=method+': EC_simple - '+control+'_simple'
                        rec={k:cell[k] for k in ('model','task','coverage','method','n','k')}
                        rec.update(a='EC',b=control)
                        if deliveries['EC'] is None or deliveries[control] is None:
                            rec.update(p=1.,delta_correct=None,note='infeasible; retain in 12-cell Holm family')
                        else:
                            ex=exact(deliveries['EC'],deliveries[control],groups)
                            rec.update(p=ex['p_value'],delta_correct=ex['delta_correct_count'],exact=ex,
                                bootstrap=bootstrap_pair(rows[names['EC']],rows[names[control]],q,seed,bank['config']['n_boot']))
                        families.setdefault(family,[]).append(rec)
                    cells.append(cell)
    for family,recs in families.items():
        require(len(recs)==12,'extension family incomplete')
        for r,p in zip(recs,holm([r['p'] for r in recs])):r['p_holm']=p
    out={'protocol':'E26_equal_simple_selector_extension_report_v1','exploratory':True,
         'reason':'Test set was previously seen and extension proposed after old primary results; no new confirmatory claim.',
         'eligibility_provenance':'Offline actual-answer eligibility uses the saved semantic judge abstention label (judge sees reference answers); correct/wrong labels do not enter ranking. Exact K is an offline comparison, not a judge-free online deployment policy.',
         'manifest_sha256':digest(manifest_path),'sources':run_sources,'cells':cells,'families':families}
    output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    write_markdown(out, output.with_suffix('.md'))
    for c in cells:
        print(c['model'],c['task'],c['coverage'],c['method'],{k:v.get('correct') for k,v in c['arms'].items()})
    return out



def write_markdown(result, path):
    sig_counts={m:sum(r['delta_correct'] is not None and r['delta_correct']>0 and r['p_holm']<.05
                     for r in result['families'][m+': EC_simple - E_simple']) for m in ('das','md')}
    conclusion='组合对 E，经校正显著为正的格子：DAS %d/12，配对均值差 %d/12。' % (sig_counts['das'],sig_counts['md'])
    lines = ['# 同一种简单筛题规则下的 E、C、E+C 补齐表', '',
             '**两个模型全部测试生成与判分已完成。'+conclusion+'**', '',
             '这是探索性补齐：原测试集早已看过；新增 E+C 配置只在原调参集选择，冻结后才补生成所缺的 11 个测试条件。旧主表与旧主检验保持不变。', '',
             '四臂统一使用简单筛题：真假判断为各自输出的真/假分数差绝对值，自由问答为原模型对该臂生成答案的平均对数概率。实质回答资格取离线语义裁判的拒答标签，裁判会看参考答案；答对/答错不参与排序。每一行的实际交付数相同，但这不是无需裁判的在线精确配额服务。', '',
             '表中为答对数；区间是已回答准确率之差，单位为百分点。Holm（多重比较校正 · 同时检验多个格子时收紧显著门槛）每类保留全部 12 格，不可行格按 p=1 纳入。', '']
    for method, title in [('das', 'DAS 方向'), ('md', '配对均值差方向')]:
        lines += ['## '+title, '', '| 模型 | 任务 | 目标 | 交付数/总数 | 原模型 | E | C | E+C | 组合减 E | 95%区间 | Holm p |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
        comparisons = {(r['model'],r['task'],r['coverage']):r for r in result['families'][method+': EC_simple - E_simple']}
        for c in result['cells']:
            if c['method']!=method:continue
            r=comparisons[(c['model'],c['task'],c['coverage'])]
            numbers=[str(c['arms'][a]['correct']) if c['arms'][a]['feasible'] else '不可行' for a in ('base','E','C','EC')]
            ci=r.get('bootstrap',{}).get('ci95_accuracy')
            interval='—' if ci is None else '[%+.2f, %+.2f]' % tuple(100*x for x in ci)
            delta='—' if r['delta_correct'] is None else '%+d' % r['delta_correct']
            lines.append('| %s | %s | %.0f%% | %d/%d | %s | %s | %s | %s | %s | %s | %.3g |' %
                         ('Llama' if c['model'].startswith('llama') else 'Qwen','真假' if c['task']=='tf' else '问答',
                          100*c['coverage'],c['k'],c['n'],*numbers,delta,interval,r['p_holm']))
        lines.append('')
    lines += ['## 判读边界', '', '- '+conclusion,
              '- 相比原模型的改进与相比 E 的额外改进是两个问题，不能用前者代替后者。',
              '- 不可行格说明实质回答数不足，不能把差值当成零。不显著也不等于证明无效。',
              '- 每个方法仍独立调参，这张表比较整套方法表现，不把差值全部归因于 C。纯粹检验加 C 的增量还需固定同一 E 配置。']
    for method in ('das','md'):
        for r in result['families'][method+': EC_simple - E_simple']:
            bs=r.get('bootstrap',{})
            if bs.get('infeasible_resamples',0):
                lines.append('- %s %s %s %.0f%%：%d 次自举中有 %d 次无法填满配额，相关区间留空，未丢弃失败抽样。' %
                    (method,r['model'],r['task'],100*r['coverage'],bs['n_resamples'],bs['infeasible_resamples']))
    lines += ['', '完整条件、选择清单、逐题交付编号、精确配对检验、输出质量计数与误差估计见同名 JSON；原始扩展产物保存在 extension/。', '']
    path.write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['freeze','report'])
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--extension-root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.phase=='freeze':
        value=build_manifest(a.source_root)
        if a.manifest.exists():require(read(a.manifest)==value,'existing frozen extension differs')
        else:a.manifest.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
        print('verified/frozen',a.manifest,digest(a.manifest))
    else:
        require(a.extension_root is not None and a.output is not None,'report requires extension-root and output')
        report(a.source_root,a.extension_root,a.manifest,a.output)
if __name__=='__main__':main()
