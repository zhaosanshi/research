#!/usr/bin/env python3
"""E33: predeclared prompt-only test of truth/answerability gate confounding.

Four prompt conditions; frozen baseline TF predictions; tuning questions only.
Record full input/token IDs and full-vocabulary probabilities, not just a forced
Yes/No ratio. Question-level scores are computed once and shared by the T/F pair.
No direction is re-estimated and no previous E26 metadata/result is modified.
"""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import e26_matched_coverage as joint
from core.experiment_provenance import implementation_identity,require_unchanged
from core.joint_steering import fingerprint
from core.selective_eval import quota_select,random_score

PROTOCOL='e33_gate_prompt_diagnostic_v1'
NEUTRAL_SYSTEM='You assess whether you can determine the answer to a question. Follow the requested response format.'
EXPLICIT_USER=('Statement: {statement}\nCan you determine whether this statement is true or false? '
    'Reply Yes if you can determine either that it is true OR that it is false. '
    'Reply No only if you cannot determine which. This is NOT asking whether the statement itself is true. '
    'Reply with exactly one word: Yes or No.')
QUESTION_USER='Question: {question}\nCan you answer this question accurately? Reply with exactly one word: Yes or No.'
SOURCE_FILES=('bank.json','bank.npz','data.json','selection.json','tf_tune.json','qa_tune.json','qa_tune_scored.json')


def sources(args):
    ns=SimpleNamespace(run_dir=args.source_run,model=args.model)
    meta,bank,data=joint.load_run(ns);joint.frozen_selection(ns,meta)
    if meta['pilot'] or meta.get('generation_adapters'):raise ValueError('complete original E26 source required')
    tf=joint.scored_rows(ns,meta,'tf','tune')[0]['baseline']
    qa=joint.scored_rows(ns,meta,'qa','tune')[0]['baseline']
    return meta,data,tf,qa


def definitions(meta):
    return [
      {'name':'legacy_statement','system':meta['prompts']['statement']['system'],'user':meta['config']['tf_gate_user'],'input_kind':'statement'},
      {'name':'neutral_system_same_user','system':NEUTRAL_SYSTEM,'user':meta['config']['tf_gate_user'],'input_kind':'statement'},
      {'name':'explicit_either_truth_value','system':NEUTRAL_SYSTEM,'user':EXPLICIT_USER,'input_kind':'statement'},
      {'name':'shared_question_answerability','system':NEUTRAL_SYSTEM,'user':QUESTION_USER,'input_kind':'question'}]


def make_plan(args):
    if args.limit is not None and args.limit<1:raise ValueError('limit must be positive')
    meta,data,tf,qa=sources(args)
    questions=data['qa']['tune'][:args.limit];keep={str(r['pair_id']) for r in questions}
    statements=[r for r in data['tf']['tune'] if str(r['pair_id']) in keep]
    groups=defaultdict(list)
    for r in statements:groups[str(r['pair_id'])].append(bool(r['label']))
    if set(groups)!=keep or any(sorted(v)!=[False,True] for v in groups.values()):raise ValueError('need one T/F pair per question')
    plan={'protocol':PROTOCOL,'model':args.model,'split':'tune','exploratory':True,'pilot_limit':args.limit,
      'bank_fingerprint':meta['fingerprint'],'source_sha256':{n:joint.digest(args.source_run/n) for n in SOURCE_FILES},
      'implementation':implementation_identity(__file__),'batch_size':meta['config']['batch_size'],'seed':meta['config']['seed'],
      'precision':'fp32 output projection; native frozen backbone','conditions':definitions(meta),
      'input_rows':{'statement':[{'id':str(r['id']),'pair_id':str(r['pair_id']),'statement':r['text']} for r in statements],
                    'question':[{'id':str(r['pair_id']),'pair_id':str(r['pair_id']),'question':r['question']} for r in questions]},
      'question_mode':'One forward score per unique original question, copied to both TF rows; truth-label AUC=.5 is construction, not evidence of useful answerability.',
      'contrasts':'legacy vs neutral changes system only; neutral vs explicit changes user instructions only; question condition changes the information being assessed.',
      'readout':'canonical single-token Yes/No logits; also retain full-vocabulary probabilities, mass, top1 and top5; no post-hoc regrouping of variants',
      'evaluation':'frozen baseline TF answers, correctness and truth-label AUC, T/F paired margin; exact quotas and same-K simple/random controls; tuning diagnostics only'}
    plan['fingerprint']=fingerprint(plan);return plan


def prepare(args):
    plan=make_plan(args);args.run_dir.mkdir(parents=True,exist_ok=True);path=args.run_dir/'plan.json'
    if path.exists() and joint.read(path)!=plan:raise ValueError('existing plan changed')
    root=Path(__file__).resolve().parents[1]
    for rel,h in plan['implementation']['files'].items():
        target=args.run_dir/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and joint.digest(target)!=h:raise ValueError('snapshot changed')
        shutil.copyfile(root/rel,target)
        if joint.digest(target)!=h:raise ValueError('snapshot copy failed')
    joint.save(path,plan,immutable=True);print('E33 PREPARED',args.model,'questions',len(plan['input_rows']['question']),flush=True)


def load_plan(args):
    plan=joint.read(args.run_dir/'plan.json')
    if plan.get('protocol')!=PROTOCOL or plan.get('model')!=args.model or plan.get('split')!='tune':raise ValueError('wrong plan identity')
    if plan['fingerprint']!=fingerprint({k:v for k,v in plan.items() if k!='fingerprint'}):raise ValueError('plan changed')
    if args.limit is not None and args.limit!=plan['pilot_limit']:raise ValueError('limit differs from frozen plan')
    require_unchanged(plan['implementation'],__file__)
    ns=SimpleNamespace(**vars(args));ns.limit=plan['pilot_limit']
    if make_plan(ns)!=plan:raise ValueError('plan sources/definitions no longer match')
    for rel,h in plan['implementation']['files'].items():
        if joint.digest(args.run_dir/'source_snapshot'/rel)!=h:raise ValueError('source snapshot changed')
    return plan


def probability_record(logits,yes,no,tokenizer):
    a=np.asarray(logits,dtype=np.float64)
    if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or yes==no or not (0<=yes<len(a) and 0<=no<len(a)):raise ValueError('invalid logits/labels')
    mx=float(a.max());lse=mx+math.log(float(np.exp(a-mx).sum()))
    py,pn=math.exp(float(a[yes])-lse),math.exp(float(a[no])-lse)
    best=int(np.argmax(a));n_top=min(5,len(a));top=sorted(np.argpartition(a,-n_top)[-n_top:].tolist(),key=lambda i:(-a[i],i))
    decode=lambda i:tokenizer.decode([int(i)],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    return {'yes_token_id':yes,'no_token_id':no,'yes_logit':float(a[yes]),'no_logit':float(a[no]),
      'score':float(np.float32(a[yes])-np.float32(a[no])),'p_yes':py,'p_no':pn,'yes_no_probability_mass':py+pn,
      'p_yes_conditional_on_yes_no':float(np.exp(a[yes]-np.logaddexp(a[yes],a[no]))),
      'top1_token_id':best,'top1_text':decode(best),'top1_probability':math.exp(float(a[best])-lse),
      'top1_is_canonical_yes_or_no':best in (yes,no),
      'top5':[{'token_id':i,'text':decode(i),'probability':math.exp(float(a[i])-lse)} for i in top]}


def validate_rows(rows,inputs,partial=False):
    ids=[r['id'] for r in inputs]
    if len(rows)>len(ids) or [r['id'] for r in rows]!=(ids[:len(rows)] if partial else ids):raise ValueError('readout cohort/order changed')
    for r,q in zip(rows,inputs):
        if r['pair_id']!=q['pair_id'] or not math.isfinite(r['score']):raise ValueError('invalid score identity')
        if r['score']!=float(np.float32(r['yes_logit'])-np.float32(r['no_logit'])):raise ValueError('logit margin mismatch')
        if not math.isclose(r['yes_no_probability_mass'],r['p_yes']+r['p_no'],abs_tol=1e-12):raise ValueError('probability mass mismatch')
        probability_keys=('p_yes','p_no','yes_no_probability_mass','p_yes_conditional_on_yes_no','top1_probability')
        if any(not math.isfinite(r[k]) or not 0<=r[k]<=1+1e-12 for k in probability_keys):raise ValueError('invalid probability')
        if not math.isfinite(r['yes_logit']) or not math.isfinite(r['no_logit']):raise ValueError('invalid saved logits')
        expected=float(np.exp(r['yes_logit']-np.logaddexp(r['yes_logit'],r['no_logit'])))
        if not math.isclose(r['p_yes_conditional_on_yes_no'],expected,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('conditional probability mismatch')
        if r['yes_no_probability_mass']>1e-300 and not math.isclose(r['p_yes']/r['yes_no_probability_mass'],expected,rel_tol=1e-8,abs_tol=1e-12):raise ValueError('saved full probabilities disagree with logit odds')
        if type(r['top1_is_canonical_yes_or_no']) is not bool or r['top1_is_canonical_yes_or_no']!=(r['top1_token_id'] in (r['yes_token_id'],r['no_token_id'])):raise ValueError('top1 canonical status changed')
        if any(not math.isfinite(x['probability']) or not 0<=x['probability']<=1 for x in r['top5']):raise ValueError('invalid top5 probability')
        mask=r['attention_mask']
        if len(r['input_ids'])!=len(mask) or not mask or any(v not in (0,1) for v in mask) or mask[-1]!=1:raise ValueError('invalid token trace')


def collect(args,bundle=None):
    plan=load_plan(args);meta,data,tf,qa=sources(args)
    bundle=joint.open_bundle(meta) if bundle is None else bundle;joint.verify_tokens(bundle,meta)
    identity={'protocol':PROTOCOL,'model':args.model,'split':'tune','pilot_limit':plan['pilot_limit'],
              'plan_sha256':joint.digest(args.run_dir/'plan.json'),'implementation':plan['implementation']}
    path=args.run_dir/'readouts.json';out=joint.read(path) if path.exists() else {**identity,'completed':False,'rows':{}}
    if any(out.get(k)!=v for k,v in identity.items()):raise ValueError('readout resume source changed')
    if not set(out['rows']).issubset({c['name'] for c in plan['conditions']}):raise ValueError('unexpected condition')
    with joint.fp32_output_projection(bundle):
        for c in plan['conditions']:
            inputs=plan['input_rows'][c['input_kind']];rows=out['rows'].setdefault(c['name'],[]);validate_rows(rows,inputs,True)
            for start in range(len(rows),len(inputs),plan['batch_size']):
                chunk=inputs[start:start+plan['batch_size']]
                texts=[bundle.build_chat(c['system'],c['user'].format(**r)) for r in chunk]
                tokens=bundle.encode(texts);labelmap=joint.gate_map(bundle,texts[0],meta['config'])
                logits=joint.one_step(bundle,texts,[])
                for i,(r,text,logit) in enumerate(zip(chunk,texts,logits)):
                    rows.append({'id':r['id'],'pair_id':r['pair_id'],'prompt':text,
                      'input_ids':tokens['input_ids'][i].detach().cpu().tolist(),
                      'attention_mask':tokens['attention_mask'][i].detach().cpu().tolist(),
                      **probability_record(logit,labelmap['yes'],labelmap['no'],bundle.tokenizer)})
                validate_rows(rows,inputs,True);joint.save(path,out)
            validate_rows(rows,inputs)
            if c['name']=='legacy_statement':
                old={r['id']:r['score'] for r in tf};diff=max(abs(r['score']-old[r['id']]) for r in rows)
                out['legacy_source_score_max_abs_diff']=diff
                if plan['pilot_limit'] is None and diff>1e-5:raise RuntimeError('legacy full-batch score reproduction failed')
            print('E33 CONDITION',args.model,c['name'],len(rows),flush=True)
    require_unchanged(plan['implementation'],__file__);out['completed']=True;joint.save(path,out)


def auc(scores,labels):
    x=np.asarray(scores,dtype=float);y=np.asarray(labels,dtype=bool)
    if x.shape!=y.shape or x.ndim!=1 or not np.isfinite(x).all():raise ValueError('invalid AUC data')
    pos,neg=x[y],x[~y]
    if not len(pos) or not len(neg):return None
    return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))


def compact_quota(rows,q,seed):
    r=quota_select(rows,q,seed=seed)
    return {k:r[k] for k in ('feasible','n','k','answered','correct','wrong','accuracy_answered','selected_ids')}


def report(args):
    plan=load_plan(args);meta,data,tf,qa=sources(args);raw=joint.read(args.run_dir/'readouts.json')
    if (raw.get('protocol')!=PROTOCOL or raw.get('model')!=args.model or raw.get('split')!='tune' or not raw.get('completed')
            or raw.get('plan_sha256')!=joint.digest(args.run_dir/'plan.json') or raw.get('implementation')!=plan['implementation']
            or raw.get('pilot_limit')!=plan['pilot_limit']):raise ValueError('readout identity/incomplete')
    if set(raw['rows'])!={c['name'] for c in plan['conditions']}:raise ValueError('missing/extra readout condition')
    ids=[r['id'] for r in plan['input_rows']['statement']];base=[r for r in tf if r['id'] in set(ids)]
    if [r['id'] for r in base]!=ids:raise ValueError('frozen baseline cohort changed')
    truth={str(r['id']):bool(r['label']) for r in data['tf']['tune']};n=len(base);seed=plan['seed']
    controls={'base_simple':[{**r,'score':r['simple_score']} for r in base],
              'id_random':[{**r,'score':random_score(r['id'],seed)} for r in base],
              'pair_random':[{**r,'score':random_score(r['pair_id'],seed)} for r in base]}
    old={r['id']:r['score'] for r in base};d0=max(abs(r['score']-old[r['id']]) for r in raw['rows']['legacy_statement'])
    if d0!=raw.get('legacy_source_score_max_abs_diff') or (plan['pilot_limit'] is None and d0>1e-5):raise ValueError('legacy source mismatch')
    results={}
    for c in plan['conditions']:
        rows=raw['rows'][c['name']];validate_rows(rows,plan['input_rows'][c['input_kind']])
        lookup={r['id']:r for r in rows};expanded=[{**r,'score':lookup[r['pair_id'] if c['input_kind']=='question' else r['id']]['score']} for r in base]
        margins=[r['score'] for r in expanded];labels=[truth[r['id']] for r in expanded]
        paired=defaultdict(dict)
        for r in expanded:paired[r['pair_id']][truth[r['id']]]=r['score']
        differences=[v[True]-v[False] for v in paired.values()]
        yes=[r for r in expanded if r['score']>0];good=sum(r['outcome']=='correct' for r in yes);k=len(yes)
        native={'n':n,'answered':k,'correct':good,'accuracy_answered':good/k if k else None,
                'same_K_controls':{name:compact_quota(rs,Fraction(k,n),seed) for name,rs in controls.items()}}
        results[c['name']]={'n_tf':n,'n_unique_model_inputs':len(rows),
          'truth_label_auc':auc(margins,labels),'baseline_correctness_auc':auc(margins,[r['outcome']=='correct' for r in base]),
          'yes_share_true':float(np.mean([r['score']>0 for r in expanded if truth[r['id']]])),
          'yes_share_false':float(np.mean([r['score']>0 for r in expanded if not truth[r['id']]])),
          'paired_truth_minus_false':{'mean':float(np.mean(differences)),'mean_abs':float(np.mean(np.abs(differences))),
                                     'min':min(differences),'max':max(differences),'by_pair':dict(zip(paired,differences))},
          'probability_audit':{'mean_yes_no_mass':float(np.mean([r['yes_no_probability_mass'] for r in rows])),
                              'min_yes_no_mass':min(r['yes_no_probability_mass'] for r in rows),
                              'mass_below_half':sum(r['yes_no_probability_mass']<.5 for r in rows),
                              'top1_outside_canonical_yes_no':sum(not r['top1_is_canonical_yes_or_no'] for r in rows),
                              'top1_text_counts':dict(Counter(r['top1_text'] for r in rows))},
          'native_gate':native,'quotas':{str(q):compact_quota(expanded,q,seed) for q in (.25,.5,.75)}}
        if c['input_kind']=='question' and any(x!=0 for x in differences):raise ValueError('shared-question pair scores differ')
    result={'protocol':PROTOCOL,'model':args.model,'split':'tune','exploratory':True,'pilot_limit':plan['pilot_limit'],
      'plan_sha256':joint.digest(args.run_dir/'plan.json'),'readouts_sha256':joint.digest(args.run_dir/'readouts.json'),
      'conditions':results,'controls':{name:{'correctness_auc':auc([r['score'] for r in rs],[r['outcome']=='correct' for r in rs]),
                              'quotas':{str(q):compact_quota(rs,q,seed) for q in (.25,.5,.75)}} for name,rs in controls.items()},
      'limit':'Question-level truth AUC=.5 follows the shared-input construction; only correctness association and matched-count comparisons test usefulness. All outcomes are reused tuning diagnostics, not held-out evidence.'}
    require_unchanged(plan['implementation'],__file__);joint.save(args.run_dir/'report.json',result)
    print('E33 REPORT',args.model,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','collect','report'])
    p.add_argument('--model',required=True,choices=['llama3.1-8b','qwen3-8b']);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    {'prepare':prepare,'collect':collect,'report':report}[a.phase](a)
if __name__=='__main__':main()
