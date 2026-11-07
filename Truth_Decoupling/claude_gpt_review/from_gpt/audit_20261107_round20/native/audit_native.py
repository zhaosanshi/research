"""Independent native-delivery recount and standalone four-panel figures.

Does not import the production policy/metric functions. Inputs are the archived
raw and scored tuning records; all dose points remain in the output.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[2]
SOURCE = REVIEW / 'from_claude/results/03_E26同作答率_E27生成适配/E26_main'
MODELS = ('llama3.1-8b', 'qwen3-8b')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())


def tie(item, seed=42):
    return digest(json.dumps(['actual_coverage_v1', seed, item], ensure_ascii=False, separators=(',', ':')).encode())


def quota(rows, k, score, seed=42):
    eligible = [r for r in rows if r['outcome'] != 'abstain']
    if len(eligible) < k:
        return {'feasible': False, 'answered': None, 'correct': None, 'accuracy_answered': None, 'selected_ids': None}
    picked = sorted(eligible, key=lambda r: (-r[score], tie(r['id'], seed), r['id']))[:k]
    correct = sum(r['outcome'] == 'correct' for r in picked)
    return {'feasible': True, 'answered': k, 'correct': correct,
            'accuracy_answered': correct / k if k else None,
            'selected_ids': sorted(r['id'] for r in picked)}


def native(rows):
    picked = [r for r in rows if r['score'] > 0 and r['outcome'] != 'abstain']
    n, k = len(rows), len(picked)
    correct = sum(r['outcome'] == 'correct' for r in picked)
    return {'n': n, 'answered': k, 'actual_coverage': k/n if n else None,
            'correct': correct, 'wrong': k-correct, 'accuracy_answered': correct/k if k else None,
            'selected_ids': sorted(r['id'] for r in picked)}


def read_artifacts(raw_root, model):
    if raw_root is not None:
        byte_files = {name: (raw_root/model/'full192'/name).read_bytes() for name in
                      ('native_raw.json', 'native_scored.json', 'report.json')}
    else:
        with tarfile.open(HERE/'evidence/native_artifacts.tar.gz') as archive:
            byte_files = {name: archive.extractfile('native/'+model+'/full192/'+name).read() for name in
                          ('native_raw.json', 'native_scored.json', 'report.json')}
    return {k: json.loads(v) for k, v in byte_files.items()}, {k: digest(v) for k, v in byte_files.items()}


def run(raw_root=None):
    result = {'scope': 'previously exposed tuning set; descriptive, no confirmation test', 'models': {}}
    verified_arms = verified_controls = 0
    plotting = {}
    for model in MODELS:
        files, hashes = read_artifacts(raw_root, model)
        raw, scored, reported = (files[n] for n in ('native_raw.json', 'native_scored.json', 'report.json'))
        plan = raw['plan']
        assert raw['completed'] and scored['completed'] and scored['plan'] == plan == reported['plan']
        assert plan['split'] == 'tune' and plan['uses_test_outcomes'] is False and plan['limit'] is None
        assert plan['fingerprint'] == fingerprint({k:v for k,v in plan.items() if k != 'fingerprint'})
        assert scored['grading_inputs']['raw_sha256'] == hashes['native_raw.json']
        assert scored['grading_identity'] == fingerprint(scored['grading_inputs'])
        data_bytes = (SOURCE/model/'data.json').read_bytes()
        assert digest(data_bytes) == plan['source_data_sha256']
        data = json.loads(data_bytes)
        model_out = {'source_hashes': hashes, 'fixed_conditions': plan['conditions'], 'tasks': {}}
        plotting[model] = {}
        for task in ('tf', 'qa'):
            cohort = data[task]['tune']
            expected = [str(r['id'] if task == 'tf' else r['pair_id']) for r in cohort]
            names = {'baseline'}
            for method in plan['conditions'][task]:
                names.add(method+'_E')
                for family in ('C','EC'):
                    for alpha in plan['alphas']:
                        stem = method+'_'+family+'_a'+format(float(alpha),'.17g')
                        names.update(stem+'_'+scope for scope in plan['scopes'])
            assert set(scored['tasks'][task]) == set(raw['tasks'][task]) == set(reported['tables'][task]) == names
            arms, cells, shared_grades = {}, {}, {}
            for name in sorted(names):
                rows, originals = scored['tasks'][task][name], raw['tasks'][task][name]
                assert [r['id'] for r in rows] == [r['id'] for r in originals] == expected
                for r, original, q in zip(rows, originals, cohort):
                    assert r['pair_id'] == str(q['pair_id'])
                    assert all(r[k] == original[k] for k in ('output','score','simple_score','finished'))
                    if task == 'tf':
                        assert r['outcome'] == ('correct' if r['predicted_true'] == q['label'] else 'wrong')
                    else:
                        assert r['outcome'] == r['grade']['label'] and r['grade']['rubric_id'] == 'qa_semantic_v3'
                        assert r['gold_is_abstention'] is False  # This full tuning cohort has no reference-only refusals.
                        key = (r['id'], r['output'])
                        assert key not in shared_grades or shared_grades[key] == r['outcome']
                        shared_grades[key] = r['outcome']
                arms[name] = rows
                actual = native(rows)
                assert actual == reported['tables'][task][name]['native'], (model,task,name)
                quality = {'replacement_character':sum('\ufffd' in r['output'] for r in rows),
                           'unfinished':sum(not r['finished'] for r in rows)}
                assert quality == reported['tables'][task][name]['quality']
                cells[name] = {'native':actual,'quality':quality,'controls':{}}
                verified_arms += 1
            for name, row in cells.items():
                for control in ['baseline'] + [m+'_E' for m in plan['conditions'][task]]:
                    for score, suffix in [('score','gate_threshold'),('simple_score','simple_threshold')]:
                        key = control+'_'+suffix
                        actual = quota(arms[control], row['native']['answered'], score)
                        assert actual == reported['tables'][task][name]['same_actual_count_controls'][key], (model,task,name,key)
                        row['controls'][key] = actual
                        verified_controls += 1
            for method in plan['conditions'][task]:
                for family in ('C','EC'):
                    parent = 'baseline' if family == 'C' else method+'_E'
                    for alpha in plan['alphas']:
                        stem = method+'_'+family+'_a'+format(float(alpha),'.17g')
                        a,b = arms[stem+'_gate_only'],arms[stem+'_gate_and_answer']
                        assert [r['score'] for r in a] == [r['score'] for r in b]
                        for x,y in zip(a,arms[parent]):
                            assert (x['id'],x['output'],x['outcome']) == (y['id'],y['output'],y['outcome'])
            model_out['tasks'][task] = cells
            plotting[model][task] = (arms,plan)
        result['models'][model] = model_out
    result['verified_native_arms'] = verified_arms
    result['verified_same_k_controls'] = verified_controls
    result['all_match'] = True
    # ID lists are compared above in full; retain hashes instead of duplicating
    # megabytes of identical control IDs already present in the source reports.
    def compact(value):
        if isinstance(value,list):return [compact(v) for v in value]
        if isinstance(value,dict):return {('selected_ids_sha256' if k=='selected_ids' else k):
            (None if v is None else fingerprint(v)) if k=='selected_ids' else compact(v) for k,v in value.items()}
        return value
    (HERE/'independent_recount.json').write_text(json.dumps(compact(result), ensure_ascii=False, indent=2)+'\n')
    with (HERE/'all_native_doses.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['model','task','condition','n','actual_answers','correct','accuracy_answered',
                         'baseline_simple_correct','das_E_simple_correct','md_E_simple_correct'])
        for model,m in result['models'].items():
            for task,cells in m['tasks'].items():
                for name,c in cells.items():
                    v=c['native']
                    writer.writerow([model,task,name,v['n'],v['answered'],v['correct'],v['accuracy_answered']]+[
                        c['controls'][key]['correct'] if c['controls'][key]['feasible'] else 'INFEASIBLE'
                        for key in ('baseline_simple_threshold','das_E_simple_threshold','md_E_simple_threshold')])
    plot(plotting)
    print(json.dumps({k:v for k,v in result.items() if k != 'models'},ensure_ascii=False))
    return result


def plot(dataset, filename_prefix='native_commitment', title_suffix=''):
    fonts = font_manager.get_font_names()
    if 'Arial Unicode MS' in fonts:
        plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams.update({'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path','font.size':10})
    for method, title in [('das','DAS 方向'),('md','配对均值差方向')]:
        fig, axes = plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for i,model in enumerate(MODELS):
            for j,task in enumerate(('tf','qa')):
                ax = axes[i,j]; arms,plan = dataset[model][task]; n=len(arms['baseline'])
                for control,color,label in [('baseline','#777777','原模型＋简单筛题'),(method+'_E','#111111','固定 E＋简单筛题')]:
                    eligible = sum(r['outcome'] != 'abstain' for r in arms[control])
                    counts = [quota(arms[control],k,'simple_score') for k in range(1,eligible+1)]
                    ax.plot([r['answered']/n for r in counts],[1-r['accuracy_answered'] for r in counts],
                            color=color,lw=1.3,alpha=.75,label=label)
                for family,color in [('C','#2c7fb8'),('EC','#d95f0e')]:
                    for scope,style,marker,desc in [('gate_only','--','o','只改门控'),('gate_and_answer','-','s','门控＋答案')]:
                        points=[]
                        for alpha in plan['alphas']:
                            name=method+'_'+family+'_a'+format(float(alpha),'.17g')+'_'+scope
                            v=native(arms[name])
                            if v['answered']:points.append((alpha,v))
                        ax.plot([v['actual_coverage'] for _,v in points],[1-v['accuracy_answered'] for _,v in points],
                                linestyle=style,marker=marker,markersize=4,lw=1.5,color=color,label=family+'：'+desc)
                        if family=='EC' and scope=='gate_and_answer':
                            for a,v in points:
                                ax.annotate(str(a), (v['actual_coverage'],1-v['accuracy_answered']),
                                            xytext=(-20 if v['actual_coverage']>.95 else 3,5),
                                            textcoords='offset points',fontsize=7,color=color)
                layer=plan['conditions'][task][method]['c_layer']
                ax.set_title(('Llama' if i==0 else 'Qwen')+' · '+('真假判断' if task=='tf' else '自由问答')+' · C 层 '+str(layer))
                ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2)
                ax.set_xlabel('实际交付比例');ax.set_ylabel('已交付回答错误比例（越低越好）')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),frameon=False)
        fig.suptitle(title+'：固定 E 的原生承诺门控\n192 道调参题的探索性诊断；真假判断含 384 条陈述'+title_suffix,fontsize=14)
        fig.text(.5,.105,'点按预定剂量连接；橙色标注为 C 剂量。零交付点无错误率，不绘制；筛题对照不足额处不外推。',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.13,1,.93))
        for ext in ('svg','pdf','png'):
            fig.savefig(HERE/(filename_prefix+'_'+method+'.'+ext),dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path)
    args=parser.parse_args();run(args.raw_root)
