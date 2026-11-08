#!/usr/bin/env python3
"""Offline selection replay only. Never overwrites old raw picks or judgment files.
Uses fixed TF training-bank orientation and already-recorded candidate readouts.
Old-v5 labels are provisional evidence; unresolved cells give bounds, never fills.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--finals',type=Path,required=True)
 p.add_argument('--tf-bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 inputs={str(p):sha(p) for p in (a.raw,a.finals,a.tf_bank/'bank.json',a.tf_bank/'bank.npz',a.tf_bank/'data.json')}
 raw=read(a.raw);meta=read(a.tf_bank/'bank.json');data=read(a.tf_bank/'data.json')
 assert raw['completed'] and raw['identity']['phase']=='select' and len(raw['evaluations'])==192
 assert meta['sources']['qa_bank_fingerprint']==raw['identity']['source']['bank_fingerprint']
 assert meta['bank_sha256']==sha(a.tf_bank/'bank.npz') and meta['data_sha256']==sha(a.tf_bank/'data.json')
 finals={}
 for line in a.finals.read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line);assert r['cell_id'] not in finals;finals[r['cell_id']]=r
 evidence={};signs={}
 with np.load(a.tf_bank/'bank.npz',allow_pickle=False) as arrays:
  for family,layers in raw['identity']['direction_sha256'].items():
   signs[family]={};evidence[family]={}
   for layer,h in layers.items():
    w=arrays[('tf_das_E_L' if family=='dasce' else 'qa_das_E_L')+layer].astype(np.float32)
    assert fp(w.tolist())==h
    md=arrays['tf_md_E_L'+layer].astype(np.float64);v=w.astype(np.float64)
    cosine=float(np.sum(v*md)/np.sqrt(np.sum(v*v)*np.sum(md*md)))
    assert np.isfinite(cosine) and abs(cosine)>1e-8
    signs[family][layer]=1 if cosine>0 else -1
    evidence[family][layer]={'raw_direction_sha256_exact':h,'dot_to_TF_train_mean_difference':float(np.sum(v*md)),
      'cosine':cosine,'sign_multiplier':signs[family][layer],'TF_mean_difference_sha256':fp(md.tolist())}
 train_identity=[{k:r.get(k) for k in ('id','pair_id','text','label')} for r in data['tf']['train']]
 assert not {r['pair_id'] for r in data['tf']['train']}&{r['pair_id'] for r in raw['evaluations']}
 variants={key:{} for key in ('old_raw_mean','sign_only','zscore_only','sign_and_zscore')}
 per_question=[]
 for r in raw['evaluations']:
  pool=[c for c in raw['conditions'] if not r['rule_abstain'][c] and r['outputs'][c].strip()] or list(raw['conditions'])
  question={'pair_id':r['pair_id'],'pool':pool,'choices':{}}
  for mode in variants:
   chosen={}
   for family,layers in raw['identity']['direction_sha256'].items():
    matrix=np.array([[r['readout'][c][f'coord_{family}_L{l}'] for l in layers] for c in pool])
    if mode in ('sign_only','sign_and_zscore'):
     matrix*=np.array([signs[family][l] for l in layers])
    for j,l in enumerate(layers):chosen[f'coord_{family}_L{l}']=pool[int(matrix[:,j].argmax())]
    if mode in ('zscore_only','sign_and_zscore'):
     std=matrix.std(0);matrix=np.divide(matrix-matrix.mean(0),std,out=np.zeros_like(matrix),where=std>1e-12)
    chosen[f'coord_{family}_mean']=pool[int(matrix.mean(1).argmax())]
   if mode=='old_raw_mean':
    assert all(r['picks'][rule]==name for rule,name in chosen.items())
   question['choices'][mode]=chosen
   for rule,c in chosen.items():variants[mode].setdefault(rule,[]).append(c)
  per_question.append(question)
 def label_bounds(row,c):
  value=finals[f'{row["pair_id"]}|{c}']
  if value['status']!='final' or value['label'] is None:return 0,1
  return (1,1) if value['label']=='A' else (0,0)
 stats={};changed=[]
 for mode,rules in variants.items():
  stats[mode]={}
  for rule,choices in rules.items():
   old=variants['old_raw_mean'][rule];low=high=0;delta_low=delta_high=0;changed_ids=[];text_changed=[];pending=[]
   for row,new,previous in zip(raw['evaluations'],choices,old):
    lo,hi=label_bounds(row,new);low+=lo;high+=hi
    if lo!=hi:pending.append(row['pair_id'])
    if new!=previous:
     changed_ids.append(row['pair_id']);oldlo,oldhi=label_bounds(row,previous)
     delta_low+=lo-oldhi;delta_high+=hi-oldlo
     if row['outputs'][new].strip()!=row['outputs'][previous].strip():text_changed.append(row['pair_id'])
   stats[mode][rule]={'n':192,'candidate_index_changes':len(changed_ids),'output_text_changes':len(text_changed),
    'changed_question_ids':changed_ids,'changed_text_question_ids':text_changed,
    'known_old_v5_A':low,'selected_unresolved_count':high-low,'selected_unresolved_ids':pending,
    'old_v5_A_count_bounds':[low,high],'paired_A_change_bounds_vs_old_rule':[delta_low,delta_high]}
 baselines={}
 for rule in ('greedy','truthful_source_only','random','max_logp','verdict'):
  choices=[rule if rule in ('greedy','truthful_source_only') else r['picks'][rule] for r in raw['evaluations']]
  bounds=[label_bounds(r,c) for r,c in zip(raw['evaluations'],choices)]
  baselines[rule]={'old_v5_A_count_bounds':[sum(v[0] for v in bounds),sum(v[1] for v in bounds)],
    'selected_unresolved_count':sum(lo!=hi for lo,hi in bounds)}
 random_low=random_high=0.
 for r in raw['evaluations']:
  pool=[c for c in raw['conditions'] if not r['rule_abstain'][c] and r['outputs'][c].strip()] or raw['conditions']
  bounds=[label_bounds(r,c) for c in pool]
  random_low+=sum(v[0] for v in bounds)/len(pool);random_high+=sum(v[1] for v in bounds)/len(pool)
 baselines['expected_random']={'old_v5_A_count_bounds':[random_low,random_high],'definition':'uniform over the same frozen inference-time candidate pool, no outcome eligibility filter'}
 result={'protocol':'r22_offline_fixed_readout_repair_replay_v1','model':raw['model'],'n_questions':192,
   'new_model_generations':0,'new_judge_calls':0,'original_raw_unchanged':True,
   'label_limit':'Only frozen old-v5 machine labels; unresolved cells remain interval bounds. Not v6 truth, not a new held-out experiment.',
   'calibration_uses_labels_from':'only fixed TF estimation train statements, disjoint from replay questions',
   'TF_training_statement_identity_sha256':fp(train_identity),'TF_prompt_sha256':fp(meta['prompts']),
   'orientation_evidence':evidence,'source_sha256':inputs,'fixed_baselines':baselines,'results':stats,'per_question_choices':per_question,
   'select_pending_cells':sum(r['status']!='final' and r['status']!='excluded_abstention_gold' for r in finals.values())}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 for mode in stats:
  print(mode,{rule:{k:v[k] for k in ('candidate_index_changes','output_text_changes','old_v5_A_count_bounds','paired_A_change_bounds_vs_old_rule')} for rule,v in stats[mode].items() if rule.endswith('_mean')})
 assert all(sha(Path(p))==h for p,h in inputs.items())

if __name__=='__main__':main()
