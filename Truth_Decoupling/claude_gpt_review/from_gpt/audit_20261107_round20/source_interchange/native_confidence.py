"""Readout sensitivity from existing generation traces; no model/API calls.

Native-policy confidence and the common unmodified-model scorer are both shown.
The former can differ for identical text across conditions because the generating
policies differ. No readout is selected based on its advantage in these results.
"""
import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys

REVIEW=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REVIEW/'from_claude/code'))
from core.judge import QA_GRADER_VERSION
from core.qa_scoring import reference_answers
from core.selective_eval import evaluate_coverage,quota_select


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def native_score(row,name):
    ids=row['generation_token_ids'][name];trace=row['trace'][name];finished=row['finished'][name]
    text=row['outputs'][name]
    if (not isinstance(ids,list) or any(type(i) is not int or i<0 for i in ids)
            or not isinstance(trace,list) or len(trace)!=len(ids) or type(finished) is not bool
            or not isinstance(text,str)):
        raise ValueError('invalid token/trace/completion identity')
    if finished and not ids:raise ValueError('empty sequence cannot contain a natural stop token')
    probabilities=[]
    for i,(token,step) in enumerate(zip(ids,trace)):
        probability=step['next_top1_probability']
        if (step['step']!=i or step['next_top1_id']!=token or step['accepted_prefix_ids']!=ids[:i]
                or step.get('uses_reference_answer') is not False):
            raise ValueError('trace is not the actual reference-free generated token path')
        if isinstance(probability,bool) or not isinstance(probability,Real) or not math.isfinite(probability) or not 0<probability<=1:
            raise ValueError('chosen-token probability must be finite in (0,1]')
        probabilities.append(float(probability))
    # The audited greedy decoder sets finished only upon its declared natural
    # stop token. No token is inserted, repaired or removed from the generation.
    content=probabilities[:-1] if finished else probabilities
    reason='no_answer_tokens' if not content else 'empty_decoded_answer' if not text.strip() else None
    score=-1e6 if reason else math.fsum(math.log(p) for p in content)/len(content)
    return {'score':score,'scorable':reason is None,'reason':reason,'raw_tokens':len(ids),
            'answer_tokens':len(content),'natural_stop_excluded':finished,
            'natural_stop_token_id':ids[-1] if finished else None}


def validate_generation(run,stem):
    raw_path=run/(stem+'.json');grade_path=run/(stem+'_scored_items.json')
    raw,grade=read(raw_path),read(grade_path)
    if (not raw.get('completed') or not grade.get('completed') or not grade.get('grading_completed')
            or raw.get('split')!='tune' or grade.get('split')!='tune'
            or grade.get('grading_source_generation_sha256')!=digest(raw_path)
            or raw['model']!=grade['model'] or raw['identity']!=grade['identity']
            or raw['conditions']!=grade['conditions'] or raw['condition_specs']!=grade['condition_specs']):
        raise ValueError('generation/semantic grading identity mismatch')
    if (raw.get('protocol_fingerprint')!=fingerprint(raw['identity'])
            or raw.get('bank_protocol_fingerprint')!=fingerprint(raw['identity'])
            or raw['identity'].get('gold_used_at_inference') is not False
            or any(c.get('uses_reference_answer') is not False for c in raw['condition_specs'])):
        raise ValueError('unverified or reference-informed generation')
    names=raw['conditions'];ids=[r['pair_id'] for r in raw['evaluations']]
    if len(names)!=len(set(names)) or len(ids)!=len(set(ids)) or ids!=[r['pair_id'] for r in grade['evaluations']]:
        raise ValueError('duplicated or incomplete condition/question cohort')
    if stem=='tune' and (raw.get('pilot') or raw.get('wiring_pilot')):
        raise ValueError('pilot cannot enter full tuning sensitivity')
    for r,g in zip(raw['evaluations'],grade['evaluations']):
        for key in ('pair_id','question','gold','incorrect','best_answer','outputs','finished'):
            if r.get(key)!=g.get(key):raise ValueError('graded raw question/answer changed')
        refs,only=reference_answers(r['gold'])
        if g.get('judge_reference_answers')!=refs or g.get('gold_is_abstention') is not only:
            raise ValueError('grading reference partition changed')
        if any(set(r[k])!=set(names) for k in ('outputs','finished','generation_token_ids','trace')):
            raise ValueError('raw condition missing')
        if set(g.get('outcome',{}))!=set(names) or set(g.get('judge',{}))!=set(names):
            raise ValueError('semantic condition missing')
        same={}
        for name in names:
            label=g['outcome'][name]
            if label not in ('correct','wrong','abstain') or g['judge'][name].get('label')!=label or g['judge'][name].get('rubric_id')!=QA_GRADER_VERSION:
                raise ValueError('invalid semantic outcome/rubric')
            text=r['outputs'][name]
            if text in same and same[text]!=label:raise ValueError('same question/answer received conflicting grades')
            same[text]=label
    return raw,grade,raw_path,grade_path


def compare(run,output,common_run=None,stem='tune',n_boot=2000):
    run=Path(run).resolve();output=Path(output)
    common_run=None if common_run is None else Path(common_run).resolve()
    raw,grade,raw_path,grade_path=validate_generation(run,stem)
    native={name:[] for name in raw['conditions']};details=[]
    for r,g in zip(raw['evaluations'],grade['evaluations']):
        for name in raw['conditions']:
            value=native_score(r,name)
            details.append({'pair_id':r['pair_id'],'condition':name,**value})
            if g['gold_is_abstention']:continue
            if not value['scorable'] and g['outcome'][name]!='abstain':
                raise ValueError('empty/unscorable answer has a substantive semantic label')
            native[name].append({'id':r['pair_id'],'pair_id':r['pair_id'],'score':value['score'],'outcome':g['outcome'][name]})
    result={'protocol':'native_policy_confidence_sensitivity_v1','model':raw['model'],'conditions':raw['conditions'],
        'split':'tune','pilot':stem=='check','exploratory':True,
        'readout_definition':'mean log of actual chosen-token probabilities; exclude final natural stop only when finished; no generation changed',
        'unscorable_rule':'-1e6 sentinel; semantic C remains in the fixed denominator',
        'eligibility':'reference-aware offline semantic annotations; A and B equally eligible',
        'warning':'Both readouts reported, no post-hoc winner selection; these are exposed tuning questions, not confirmation.',
        'source_files':{raw_path.name:digest(raw_path),grade_path.name:digest(grade_path)},
        'trace_scores':details,'native_policy':evaluate_coverage(native,(.25,.5,.75),n_boot=n_boot,seed=42)}
    if common_run is not None:
        c_raw,c_grade,c_raw_path,c_grade_path=validate_generation(common_run,stem)
        score_path=common_run/(stem+'_simple_scores.json');scores=read(score_path)
        if (not scores.get('completed') or scores['identity'].get('protocol')!='source_generation_equal_simple_coverage_v1'
                or scores['identity']['source_generation_sha256']!=digest(c_raw_path)
                or scores['identity'].get('uses_gold_in_score') is not False
                or scores['identity']['conditions']!=c_raw['conditions']
                or scores['identity']['model']!=raw['model']
                or scores['identity']['source']!=raw['identity']['source']):
            raise ValueError('common-model score source mismatch')
        controls_subset=digest(c_raw_path)!=digest(raw_path)
        if controls_subset and c_raw['identity'].get('control_generation_sha256')!=digest(raw_path):
            raise ValueError('six-arm result does not bind these exact original five controls')
        ids=[r['pair_id'] for r in raw['evaluations']]
        if ids!=[r['pair_id'] for r in c_raw['evaluations']] or ids!=[r['pair_id'] for r in scores['rows']]:
            raise ValueError('common-score cohort differs')
        common={name:[] for name in raw['conditions']}
        for r,g,cr,cg,s in zip(raw['evaluations'],grade['evaluations'],c_raw['evaluations'],c_grade['evaluations'],scores['rows']):
            for name in raw['conditions']:
                if (r['question']!=cr['question'] or r['outputs'][name]!=cr['outputs'][name]
                        or g['outcome'][name]!=cg['outcome'][name]):
                    raise ValueError('a reused control answer or grade changed')
                if g['gold_is_abstention']:continue
                common[name].append({'id':r['pair_id'],'pair_id':r['pair_id'],'score':s['scores'][name],'outcome':g['outcome'][name]})
        result['common_unmodified_model']=evaluate_coverage(common,(.25,.5,.75),n_boot=n_boot,seed=42)
        result['common_score_source']={'generation_sha256':digest(c_raw_path),'grading_sha256':digest(c_grade_path),
                                      'score_sha256':digest(score_path),'controls_subset_reuse':controls_subset}
        official_path=common_run/(stem+'_matched_coverage.json')
        if official_path.exists():
            official=read(official_path)
            if any(official['source_files'].get(p.name)!=digest(p) for p in (c_raw_path,c_grade_path,score_path)):
                raise ValueError('E32 common report does not bind these exact raw/grade/score files')
            for a,b in zip(result['common_unmodified_model']['coverages'],official['matched_coverage']['coverages']):
                for name in raw['conditions']:
                    for key in ('feasible','answered','correct','selected_ids'):
                        if a['arms'][name][key]!=b['arms'][name][key]:raise ValueError('common score recount differs from E32')
            result['common_score_matches_E32_point_counts_and_ids']=True
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 生成置信读出敏感性', '',raw['model']+'：调参集探索；两种读出并列，不择优宣称成功。',
           '原生置信来自实际生成词元概率；只有自然结束时才在计算分数时排除最后停止词，原文和词元从未改动。',
           '实质回答仍按带参考的离线语义标注计数；拒答保留分母。','',
           '| 作答份额 | 方法 | 统一原模型分：答对/交付 | 各自生成置信：答对/交付 |','|---:|---|---:|---:|']
    def cell(x):return '不可行' if not x['feasible'] else '%d/%d'%(x['correct'],x['answered'])
    for i,point in enumerate(result['native_policy']['coverages']):
        for name in raw['conditions']:
            common_cell=cell(result['common_unmodified_model']['coverages'][i]['arms'][name]) if 'common_unmodified_model' in result else '未提供'
            lines.append('| %.0f%% | %s | %s | %s |'%(point['target_coverage']*100,name,common_cell,cell(point['arms'][name])))
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(raw['model'],len(raw['conditions']),'arms',len(raw['evaluations']),'questions','done',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--common-run',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stem',choices=('tune','check'),default='tune');p.add_argument('--n-boot',type=int,default=2000)
    a=p.parse_args();compare(a.run_dir,a.output_dir,a.common_run,a.stem,a.n_boot)
