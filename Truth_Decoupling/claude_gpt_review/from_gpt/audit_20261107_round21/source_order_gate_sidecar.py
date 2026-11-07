"""Exploratory Yes-No ranking sidecar; leaves the declared primary score intact."""
import argparse,copy,math
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
LEGACY=HERE.parent/'audit_20261107_round20/refusal_guard'
sys.path.insert(0,str(LEGACY))
from audit_and_sidecars import read,save,digest,eval_arms,correct_arms,evaluate_coverage
from native_confidence import validate_generation
from e34_sidecars import quality_view
EXPECTED=['baseline','E_source','C_before_m1','EC_before_m1','C_before_p1','EC_before_p1',
          'C_after_m1','EC_after_m1','C_after_p1','EC_after_p1']
NEIGHBORS=['C_neighbor_m1','EC_neighbor_m1','C_neighbor_p1','EC_neighbor_p1']


def build(run,output,n_boot=2000):
    raw,grade,rp,gp=validate_generation(run,'tune')
    expected={
        'source_e_commitment_layer_order_v1':EXPECTED,
        'source_e_commitment_neighbor_extension_v1':EXPECTED+NEIGHBORS}.get(raw['protocol'])
    if (expected is None or len(raw['evaluations'])!=192 or raw['conditions']!=expected
            or (raw['protocol']=='source_e_commitment_neighbor_extension_v1' and raw['model']!='llama3.1-8b')):
        raise ValueError('requires complete fixed E37 tuning cohort')
    sp=run/'tune_simple_scores.json';scores=read(sp)
    ids=[r['pair_id'] for r in raw['evaluations']]
    if (not scores['completed'] or scores['identity']['source_generation_sha256']!=digest(rp)
            or scores['identity']['conditions']!=raw['conditions'] or scores['identity']['model']!=raw['model']
            or scores['identity'].get('uses_gold_in_score') is not False
            or [r['pair_id'] for r in scores['rows']]!=ids):raise ValueError('common score source mismatch')
    arms,questions=eval_arms(grade);guarded,guard=correct_arms(arms,questions);strict,changes=quality_view(guarded,raw)
    if any(r['gold_is_abstention'] for rows in arms.values() for r in rows):raise ValueError('fixed E37 substantive cohort changed')
    by_id={r['pair_id']:r for r in raw['evaluations']};sc={r['pair_id']:r['scores'] for r in scores['rows']}
    views={};gate_quality={}
    for name in expected:
        masses=[];outside=0
        for row in raw['evaluations']:
            gate=row['gate_readout'][name];probabilities=[gate['p_yes'],gate['p_no']]
            if (any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1 for v in probabilities)
                    or sum(probabilities)>1.000001):raise ValueError('invalid gate probability mass')
            top=gate['trace']['next_top1_id']
            if type(top) is not int:raise ValueError('missing real gate top-token identity')
            masses.append(sum(probabilities));outside+=top not in (gate['yes_id'],gate['no_id'])
        gate_quality[name]={'n':len(masses),'yes_no_probability_mean':sum(masses)/len(masses),
                            'yes_no_probability_min':min(masses),'yes_no_probability_max':max(masses),
                            'top1_outside_yes_no_count':outside,'top1_outside_yes_no_fraction':outside/len(masses)}
    for view,labels in (('R1',guarded),('R2',strict)):
        selected=copy.deepcopy(labels)
        for name,rows in selected.items():
            for r in rows:
                gate=by_id[r['id']]['gate_readout'][name];v=gate['yes_minus_no']
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or gate['yes_id']==gate['no_id']:
                    raise ValueError('invalid ability first-token readout')
                r['score']=v
        selected['E_source_common_score']=copy.deepcopy(labels['E_source'])
        for r in selected['E_source_common_score']:r['score']=sc[r['id']]['E_source']
        views[view]=evaluate_coverage(selected,[.25,.5,.75],n_boot=n_boot,seed=42)
    result={'protocol':'r21_exploratory_ability_score_sidecar_v1','model':raw['model'],'n_boot':n_boot,
        'added_after_full_generation_started':True,'changes_primary_plan':False,'exploratory_tuning_only':True,
        'score_scope':'C-only ability margin is a functional probe. E-containing gates exchange ability-base with QA-answer source; this is a cross-prompt exploratory score, not a trained E ability head.',
        'strong_control':'E_source_common_score reuses exact fixed E answers/labels and their unmodified-model average token log probabilities',
        'source_files':{p.name:digest(p) for p in (rp,gp,sp)},'generation_protocol':raw['protocol'],
        'conditions':expected,'guard_audit':guard,'quality_changes':changes,
        'gate_score_quality':gate_quality,'views':views}
    save(output/'report.json',result)
    lines=['# E/C层序：Yes−No筛题的探索补充','','这张表在完整生成开始后追加，不改变主要共同置信评分；所有十臂与两质量视图全部保留。含E的能力读数是跨提示诊断，必须同时对照固定E＋共同置信的强筛题结果，不能只击败较弱的E能力读数就宣布组合优势。','']
    for view,table in views.items():
        lines+=['## '+view,'','| 作答率 | 方法 | 实际交付 | 答对 |','|---|---|---:|---:|']
        for point in table['coverages']:
            for name,r in point['arms'].items():lines.append('| %g%% | %s | %s | %s |'%(100*point['target_coverage'],name,r['answered'] if r['feasible'] else '不可行',r['correct'] if r['feasible'] else '—'))
        lines.append('')
    lines += ['## 门控格式质量（不据此删题）','','| 方法 | Yes+No平均概率 | 最低概率 | 最高分词非Yes/No |','|---|---:|---:|---:|']
    for name,q in gate_quality.items():lines.append('| %s | %.6f | %.6f | %d/%d |'%(name,q['yes_no_probability_mean'],q['yes_no_probability_min'],q['top1_outside_yes_no_count'],q['n']))
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    for p in (rp,gp,sp):assert digest(p)==result['source_files'][p.name]
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();build(a.run_dir,a.output_dir)
