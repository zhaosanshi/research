"""All-arm guard/quality/native/common sensitivity for a completed source run."""
import argparse
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_and_sidecars import read,save,digest,eval_arms,correct_arms,natural,audit_source
from native_confidence import validate_generation,native_score
from e34_sidecars import quality_view,score_views


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--n-boot',type=int,default=2000)
    a=p.parse_args();raw,grade,rp,gp=validate_generation(a.run_dir,'tune')
    scorepath=a.run_dir/'tune_simple_scores.json';scores=read(scorepath)
    if (not scores.get('completed') or scores['identity']['source_generation_sha256']!=digest(rp)
            or scores['identity']['model']!=raw['model'] or scores['identity']['conditions']!=raw['conditions']
            or scores['identity'].get('uses_gold_in_score') is not False
            or [r['pair_id'] for r in scores['rows']]!=[r['pair_id'] for r in raw['evaluations']]):
        raise ValueError('common scoring does not bind this exact complete source generation')
    arms,q=eval_arms(grade);guarded,guard=correct_arms(arms,q)
    strict,changes=quality_view(guarded,raw)
    result=audit_source(a.run_dir,a.run_dir,a.n_boot)
    result.update(protocol='uniform_source_guard_quality_sensitivity_v1',conditions=raw['conditions'],
        original_generation_protocol=raw['protocol'],exploratory=True,n_boot=a.n_boot,
        quality_sensitivity={'natural':{n:natural(rs) for n,rs in strict.items()},'changes':changes,
            **score_views(strict,raw,scores,a.n_boot)},
        quality_definition='Complete-personal-refusal guard for all arms, then correct unfinished/U+FFFD responses count wrong; C and fixed denominator retained. No text/token manipulation.',
        source_script_sha256=digest(__file__))
    for old,new in [('native_policy_confidence_corrected','native_policy'),('common_unmodified_confidence_corrected','common_unmodified_model')]:
        for first,second in zip(result[old]['coverages'],result['quality_sensitivity'][new]['coverages']):
            for name in raw['conditions']:
                for key in ('selected_ids','answered','feasible'):
                    if first['arms'][name][key]!=second['arms'][name][key]:raise ValueError('quality view changed selection/eligibility')
    by_id={r['pair_id']:r for r in raw['evaluations']}
    result['quality_counts']={name:{'unfinished':sum(not r['finished'][name] for r in raw['evaluations']),
        'unicode_replacement':sum('\ufffd' in r['outputs'][name] for r in raw['evaluations']),
        'guarded_correct_unfinished':sum(r['outcome']=='correct' and not by_id[r['pair_id']]['finished'][name] for r in guarded[name])}
        for name in raw['conditions']}
    save(a.output_dir/'report.json',result)
    lines=['# 来源实验：全臂统一拒答守卫与质量侧表','',raw['model']+' / '+raw['protocol'],
        '已暴露调参题探索；原语义裁判仍可能错，不把以下数值当无误真值。共同原模型分与各自生成置信并列，不挑更好的读出。',
        'R1 为完整个人拒答守卫；R2 再将未自然结束/替换乱码的正确回答计失败，资格/分母/选中ID与R1保持一致。','',
        '| 视图 | 方法 | 自然答对/实答 | 共同分25/50/75%答对 | 各自置信25/50/75%答对 |','|---|---|---|---|---|']
    for view,nats,common,native in [('R1',result['natural_corrected'],result['common_unmodified_confidence_corrected'],result['native_policy_confidence_corrected']),
                                  ('R2',result['quality_sensitivity']['natural'],result['quality_sensitivity']['common_unmodified_model'],result['quality_sensitivity']['native_policy'])]:
        for name in raw['conditions']:
            cells=[]
            for scoring in (common,native):
                cells.append('/'.join(str(x['arms'][name]['correct']) if x['arms'][name]['feasible'] else '不可行' for x in scoring['coverages']))
            n=nats[name];lines.append('| %s | %s | %s/%s | %s | %s |'%(view,name,n['correct'],n['answered'],*cells))
    (a.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    if digest(rp)!=result['source_files']['tune.json'] or digest(gp)!=result['source_files']['tune_scored_items.json']:
        raise RuntimeError('source files changed during read-only analysis')
    print(raw['model'],len(raw['conditions']),'arms','guard_changes',guard['changed_record_occurrences'],'R2_changes',len(changes),flush=True)

if __name__=='__main__':main()
