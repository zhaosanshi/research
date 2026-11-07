#!/usr/bin/env python3
"""Fixed old/explicit QA answerability gate diagnostic, without answer steering.

Three elementary yes/no pairs are verified by actual baseline answers first;
full generation reads only the original 192 tuning questions. Main candidate
answers/grades remain frozen. Report legacy and common full-refusal-guard views.
"""
import argparse,copy,json,math,re,shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

import common  # noqa: F401
import e26_matched_coverage as joint
from e33_gate_prompt_diagnostic import probability_record,validate_rows,auc,compact_quota
from core.qa_generation_adapter import implementation_identity
from core.qa_scoring import apply_full_personal_refusal_policy
from core.selective_eval import random_score,quota_select

PROTOCOL='e35_qa_ability_gate_v1'
ABILITY_SYSTEM='You assess whether you can determine the answer to a question. Follow the requested response format.'
ABILITY_USER=("Question: {question}\nThis is an answerability check, not a request to answer the question above. "
 "Can you provide an accurate, substantive answer to it? Reply Yes if you can provide such an answer. "
 "This includes a correct answer of 'No', correcting a false premise, or accurately explaining that the requested answer is objectively unknown, nonexistent, or not uniquely determined by the question. "
 "Merely expressing your own ignorance or declining without a substantive answer does not qualify. "
 "Reply No if you cannot provide an accurate substantive answer. Do not answer the original question here. "
 "Reply with exactly one word: Yes or No.")
PILOT_PAIRS=(('math','Is 2 + 2 equal to 4?','Is 2 + 2 equal to 5?'),
             ('capital','Is Paris the capital of France?','Is Paris the capital of Germany?'),
             ('earth','Is Earth a planet?','Is Earth a star?'))
SOURCE_FILES=('bank.json','bank.npz','data.json','selection.json','qa_tune.json','qa_tune_scored.json')


def sources(args):
    ns=SimpleNamespace(run_dir=args.source_run,model=args.model)
    meta,bank,data=joint.load_run(ns);joint.frozen_selection(ns,meta)
    if meta['pilot'] or meta.get('generation_adapters'):raise ValueError('complete original E26 source required')
    raw=joint.read(args.source_run/'qa_tune_scored.json')
    joint.scored_rows(ns,meta,'qa','tune')
    if len(data['qa']['tune'])!=192 or 'qa_gate_system' in meta['config']:
        raise ValueError('diagnostic requires the original 192-question legacy QA gate source')
    return meta,data,raw['rows']['baseline']


def synthetic_questions():
    return [{'id':name+'_'+label.lower(),'pair_id':name,'question':question}
            for name,yes,no in PILOT_PAIRS for label,question in [('Yes',yes),('No',no)]]


def make_plan(args):
    meta,data,base=sources(args)
    rows=(synthetic_questions() if args.pilot else
          [{'id':str(q['pair_id']),'pair_id':str(q['pair_id']),'question':q['question']} for q in data['qa']['tune']])
    plan={'protocol':PROTOCOL,'model':args.model,'pilot':args.pilot,'split':'synthetic_controls' if args.pilot else 'tune',
      'exploratory':True,'source_sha256':{n:joint.digest(args.source_run/n) for n in SOURCE_FILES},
      'bank_fingerprint':meta['fingerprint'],'implementation':implementation_identity(),
      'conditions':[{'name':'legacy_question_gate','system':meta['qa_config']['qa_system'],'user':meta['config']['qa_gate_user']},
                    {'name':'explicit_substantive_ability','system':ABILITY_SYSTEM,'user':ABILITY_USER}],
      'question_rows':rows,'batch_size':meta['config']['batch_size'],'seed':meta['config']['seed'],
      'pilot_expected_original_answer':{r['id']:('Yes' if r['id'].endswith('_yes') else 'No') for r in rows} if args.pilot else None,
      'pilot_expected_gate':'Yes for either original answer polarity, after confirming baseline can answer accurately',
      'inference_never_receives_expected_answers_or_references':True,
      'evaluation':'Same saved candidate answers; legacy and common full-personal-refusal-guard outcome views; actual-answer eligibility is offline.',
      'new_template_fixed_before_results':True}
    plan['fingerprint']=joint.fingerprint(plan);return plan


def prepare(args):
    plan=make_plan(args);args.run_dir.mkdir(parents=True,exist_ok=True);path=args.run_dir/'plan.json'
    if path.exists() and joint.read(path)!=plan:raise ValueError('existing diagnostic plan changed')
    root=Path(__file__).resolve().parents[1]
    for rel,h in plan['implementation']['files'].items():
        p=args.run_dir/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if joint.digest(root/rel)!=h:raise RuntimeError('source changed before copy')
        if p.exists() and joint.digest(p)!=h:raise ValueError('saved code snapshot changed')
        shutil.copyfile(root/rel,p)
    joint.save(path,plan,immutable=True);print('E35 PREPARED',args.model,'pilot',args.pilot,len(plan['question_rows']),flush=True)


def load_plan(args):
    plan=joint.read(args.run_dir/'plan.json')
    if plan!=make_plan(args):raise ValueError('frozen diagnostic plan/source/code differs')
    for rel,h in plan['implementation']['files'].items():
        if joint.digest(args.run_dir/'source_snapshot'/rel)!=h:raise ValueError('frozen source snapshot differs')
    return plan


def pilot_answers(bundle,meta,rows):
    result=[]
    for row in rows:
        text=bundle.build_chat(meta['qa_config']['qa_system'],meta['qa_config']['qa_user'].format(question=row['question']))
        inp=bundle.encode(text);n=inp['input_ids'].shape[1]
        with torch.no_grad():
            y=bundle.model.generate(**inp,do_sample=False,num_beams=1,use_cache=True,
                max_new_tokens=meta['config']['max_new_tokens'],pad_token_id=bundle.tokenizer.pad_token_id)
        ids=y[0,n:].tolist();eos=bundle.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,(tuple,list)) else [eos])
        answer=bundle.tokenizer.decode(ids,skip_special_tokens=True)
        match=re.match(r'^\s*(Yes|No)\b',answer,re.I)
        result.append({'id':row['id'],'question':row['question'],'prompt':text,'input_ids':inp['input_ids'][0].tolist(),
                       'output':answer,'token_ids':ids,'finished':any(i in eos for i in ids),
                       'literal_answer_polarity':match.group(1).capitalize() if match else None})
    return result


def collect(args,bundle=None):
    plan=load_plan(args);meta,data,base=sources(args);path=args.run_dir/'readouts.json'
    identity={'protocol':PROTOCOL,'model':args.model,'pilot':args.pilot,'plan_sha256':joint.digest(args.run_dir/'plan.json'),
              'implementation':plan['implementation'],'split':plan['split']}
    if path.exists():
        old=joint.read(path)
        if any(old.get(k)!=v for k,v in identity.items()) or not old.get('completed'):raise ValueError('existing readouts differ or are partial; preserve them')
        return old
    bundle=joint.open_bundle(meta) if bundle is None else bundle;joint.verify_tokens(bundle,meta)
    out={**identity,'completed':False,'rows':{}}
    with joint.fp32_output_projection(bundle):
        if args.pilot:
            out['baseline_original_answers']=pilot_answers(bundle,meta,plan['question_rows'])
            for r in out['baseline_original_answers']:print('E35 ORIGINAL_ANSWER',args.model,r['id'],repr(r['output']),flush=True)
        for c in plan['conditions']:
            rows=[];out['rows'][c['name']]=rows
            for start in range(0,len(plan['question_rows']),plan['batch_size']):
                chunk=plan['question_rows'][start:start+plan['batch_size']]
                texts=[bundle.build_chat(c['system'],c['user'].format(question=r['question'])) for r in chunk]
                tokens=bundle.encode(texts);labels=joint.gate_map(bundle,texts[0],meta['config'])
                logits=joint.one_step(bundle,texts,[])
                for i,(q,text,logit) in enumerate(zip(chunk,texts,logits)):
                    rows.append({'id':q['id'],'pair_id':q['pair_id'],'prompt':text,
                      'input_ids':tokens['input_ids'][i].tolist(),'attention_mask':tokens['attention_mask'][i].tolist(),
                      **probability_record(logit,labels['yes'],labels['no'],bundle.tokenizer)})
                validate_rows(rows,plan['question_rows'],True);joint.save(path,out)
            validate_rows(rows,plan['question_rows'])
            print('E35 READOUT',args.model,c['name'],len(rows),flush=True)
        if not args.pilot:
            reference={r['id']:r['score'] for r in base}
            delta=max(abs(r['score']-reference[r['id']]) for r in out['rows']['legacy_question_gate'])
            if delta>1e-5:raise RuntimeError('legacy QA gate does not reproduce with original batch geometry')
            out['legacy_source_score_max_abs_diff']=delta
    if implementation_identity()!=plan['implementation']:raise RuntimeError('source changed during diagnosis')
    out['completed']=True;joint.save(path,out);return out


def native(rows):
    chosen=[r for r in rows if r['score']>0 and r['outcome']!='abstain'];n=len(rows);k=len(chosen);good=sum(r['outcome']=='correct' for r in chosen)
    return {'n':n,'answered':k,'correct':good,'accuracy_answered':good/k if k else None,
            'actual_coverage':k/n,'selected_ids':[r['id'] for r in chosen]}


def summary(rows,seed):
    attempted=[r for r in rows if r['outcome']!='abstain']
    return {'n':len(rows),'n_available':len(attempted),'correctness_auc_all':auc([r['score'] for r in rows],[r['outcome']=='correct' for r in rows]),
            'correctness_auc_among_answers':auc([r['score'] for r in attempted],[r['outcome']=='correct' for r in attempted]) if attempted else None,
            'native':native(rows),'quotas':{str(q):compact_quota(rows,q,seed) for q in (.25,.5,.75)}}


def report(args):
    plan=load_plan(args);meta,data,base=sources(args);raw=joint.read(args.run_dir/'readouts.json')
    if (not raw.get('completed') or raw.get('protocol')!=PROTOCOL or raw.get('model')!=args.model or raw.get('pilot')!=args.pilot
            or raw.get('plan_sha256')!=joint.digest(args.run_dir/'plan.json') or raw.get('implementation')!=plan['implementation']
            or set(raw['rows'])!={c['name'] for c in plan['conditions']}):raise ValueError('readout identity differs')
    for rows in raw['rows'].values():validate_rows(rows,plan['question_rows'])
    result={'protocol':PROTOCOL,'model':args.model,'pilot':args.pilot,'split':plan['split'],'exploratory':True,
            'plan_sha256':joint.digest(args.run_dir/'plan.json'),'readouts_sha256':joint.digest(args.run_dir/'readouts.json'),'probability_audit':{}}
    for name,rows in raw['rows'].items():
        result['probability_audit'][name]={'mean_mass':float(np.mean([r['yes_no_probability_mass'] for r in rows])),
             'minimum_mass':min(r['yes_no_probability_mass'] for r in rows),'top1_outside':sum(not r['top1_is_canonical_yes_or_no'] for r in rows),
             'yes_count':sum(r['score']>0 for r in rows),'n':len(rows)}
    if args.pilot:
        baseline=raw['baseline_original_answers'];expected=plan['pilot_expected_original_answer']
        result['baseline_literal_expected_matches']=sum(r['literal_answer_polarity']==expected[r['id']] for r in baseline)
        result['baseline_original_answers']=baseline
        result['case_readouts']={name:[{'id':r['id'],'score':r['score'],'top1':r['top1_text'],'mass':r['yes_no_probability_mass']} for r in rows] for name,rows in raw['rows'].items()}
        result['interpretation']='Inspect each baseline original answer first; every correctly answered elementary control should receive ability Yes, including original No answers.'
    else:
        if [r['id'] for r in base]!=[r['id'] for r in plan['question_rows']]:raise ValueError('fixed candidate cohort differs')
        original={r['id']:r for r in base};questions={r['pair_id']:r['question'] for r in data['qa']['tune']}
        delta=max(abs(r['score']-original[r['id']]['score']) for r in raw['rows']['legacy_question_gate'])
        if delta!=raw['legacy_source_score_max_abs_diff'] or delta>1e-5:raise ValueError('legacy control mismatch')
        result['policies']={};result['correct_yes_no_cases']=[]
        for policy in ('legacy','full_personal_refusal_guard'):
            fixed=[]
            for r in base:
                if r.get('gold_is_abstention'):continue
                label=r['outcome'] if policy=='legacy' else apply_full_personal_refusal_policy(r['grade'],questions[r['id']],r['output'])['label']
                fixed.append({**r,'outcome':label})
            controls={'simple':[{**r,'score':r['simple_score']} for r in fixed],
                      'random':[{**r,'score':random_score(r['id'],plan['seed'])} for r in fixed]}
            pr={'controls':{name:summary(rows,plan['seed']) for name,rows in controls.items()},'conditions':{}}
            for name,rows in raw['rows'].items():
                scores={r['id']:r['score'] for r in rows};joined=[{**r,'score':scores[r['id']]} for r in fixed]
                sr=summary(joined,plan['seed']);k=sr['native']['answered'];sr['native_same_K_controls']={c:compact_quota(rs,Fraction(k,len(joined)),plan['seed']) for c,rs in controls.items()}
                pr['conditions'][name]=sr
            result['policies'][policy]=pr
        maps={name:{r['id']:r for r in rows} for name,rows in raw['rows'].items()}
        for r in base:
            match=re.match(r'^\s*(Yes|No)\b',r['output'],re.I)
            if r['outcome']=='correct' and match:
                result['correct_yes_no_cases'].append({'id':r['id'],'question':questions[r['id']],'output':r['output'],
                  'answer_prefix':match.group(1).capitalize(),'binary_question_form':bool(re.match(r'^(Is|Are|Can|Do|Does|Did|Was|Were|Has|Have|Will|Would|Should|Could)\b',questions[r['id']],re.I)),
                  'simple_score':r['simple_score'],'gates':{name:{'score':m[r['id']]['score'],'top1':m[r['id']]['top1_text'],'mass':m[r['id']]['yes_no_probability_mass']} for name,m in maps.items()}})
        result['interpretation']='Fixed old answers, no C training yet. Both outcome policies apply to both readouts and controls; offline refusal eligibility is not a gold-free online policy.'
    joint.save(args.run_dir/'report.json',result);print('E35 REPORT',args.model,'pilot',args.pilot,flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','collect','report'))
    p.add_argument('--model',required=True,choices=('llama3.1-8b','qwen3-8b'));p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--pilot',action='store_true');args=p.parse_args()
    {'prepare':prepare,'collect':collect,'report':report}[args.phase](args)
if __name__=='__main__':main()
