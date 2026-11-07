#!/usr/bin/env python3
"""Independent E40 source/label inheritance, physical writes and quality audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

FIELDS=('outputs','finished','rule_abstain','generation_token_ids','trace','gate_readout')
NEW=('C_neighbor_m1','EC_neighbor_m1','C_neighbor_p1','EC_neighbor_p1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def audit(run,parent_run,pilot):
 stem='check' if pilot else 'tune';raw=read(run/(stem+'.json'));grade=read(run/(stem+'_scored_items.json'))
 parent=read(parent_run/'tune.json');pg=read(parent_run/'tune_scored_items.json');plan=read(run/'plan.json')
 n=3 if pilot else 192
 assert raw['completed'] and grade['grading_completed'] and len(raw['evaluations'])==n and len(grade['evaluations'])==n
 assert raw['identity']==plan and raw['protocol']=='source_e_commitment_neighbor_extension_v1'
 assert plan['fingerprint']==fp({k:v for k,v in plan.items() if k!='fingerprint'})
 assert plan['parent_generation_sha256']==sha(parent_run/'tune.json') and plan['parent_scored_sha256']==sha(parent_run/'tune_scored_items.json')
 assert plan['parent_reference_lock_audit_sha256']==sha(parent_run/'reference_lock_audit.json')
 assert plan['parent_reference_lock_manifest_sha256']==sha(parent_run/'reference_lock_manifest.json')
 assert raw['control_identity']==parent['identity'] and raw['conditions']==parent['conditions']+list(NEW)
 assert raw['condition_specs'][:10]==parent['condition_specs']
 assert raw['all_controls_freshly_generated'] is False and raw['all_controls_inherited_exactly'] is True
 assert plan['neighbor_c_layer']==14 and plan['e_layer']==15 and plan['max_new_tokens']==48
 assert plan['neighbor_c_coefficient']==plan['c_coefficients']['neighbor']>0
 assert grade['grading_source_generation_sha256']==sha(run/(stem+'.json'))
 assert grade['grading_judge_config_sha256']==pg['grading_judge_config_sha256']
 for rel,h in plan['implementation']['files'].items():assert sha(run/'code_snapshot'/rel)==h
 for rel,h in parent['identity']['implementation']['files'].items():
  if rel.startswith('core/') or rel in ('scripts/common.py','scripts/e29_qa_source_interchange.py'):assert sha(run/'code_snapshot'/rel)==h
 stats={c:{'correct':0,'wrong':0,'abstain':0,'unfinished':0,'replacement_character':0,'text_changes_from_control':0,
   'steps_checked':0,'gate_deltas':[],'C_actual_ratios':[]} for c in NEW}
 inherited=0;events=0
 for i,(row,g,old,oldg) in enumerate(zip(raw['evaluations'],grade['evaluations'],parent['evaluations'],pg['evaluations'])):
  assert row['pair_id']==plan['question_ids'][i]==old['pair_id'] and row['idx']==i
  assert {k:v for k,v in row.items() if k not in FIELDS}=={k:v for k,v in old.items() if k not in FIELDS}
  for field in FIELDS:
   assert {name:row[field][name] for name in parent['conditions']}==old[field]
   assert g[field]==row[field]
  for name in parent['conditions']:
   assert g['judge'][name]==oldg['judge'][name] and g['outcome'][name]==oldg['outcome'][name];inherited+=1
  for name in NEW:
   enabled=name.startswith('EC_');sign=-1 if name.endswith('m1') else 1;control='E_source' if enabled else 'baseline'
   tokens=row['generation_token_ids'][name];trace=row['trace'][name];assert len(tokens)==len(trace)<=48
   entry=stats[name];entry['steps_checked']+=len(trace)
   for step,t in enumerate(trace):
    assert t['accepted_prefix_ids']==tokens[:step] and t['next_top1_id']==tokens[step]
    assert t['source_forward_calls']==int(enabled) and t['source_c_writes']==0 and t['base_forward_calls']==1
    expected=([('C',14)] if step==0 else [])+([('E',15)] if enabled else [])
    assert [(e['operator'],e['layer']) for e in t['base_operation_events']]==expected
    assert len(t['c_writes'])==int(step==0);events+=len(t['c_writes'])
    for c in t['c_writes']:
     assert c['layer']==14 and c['alpha']==sign and c['scale']==plan['neighbor_c_coefficient']
     assert c['physical_position']==c['logical_position']==len(row['prompt_token_ids']['base'])-1
     assert math.isclose(c['ideal_delta_norm'],plan['neighbor_c_coefficient'],rel_tol=1e-5,abs_tol=1e-7)
     assert math.isclose(c['stored_delta_over_base'],c['stored_delta_norm']/c['base_norm'],rel_tol=1e-6,abs_tol=1e-8)
     entry['C_actual_ratios'].append(c['stored_delta_over_base'])
    if enabled:
     assert t['cache_storage_disjoint'] and t['source_predictors_captured']==1
     for w in t['writes']:
      assert w['ideal_coordinate_error']<=w['float32_tolerance'] and w['ideal_orthogonal_error']<=w['float32_tolerance']
   gate=row['gate_readout'][name];tr=gate['trace']
   assert [(e['operator'],e['layer']) for e in tr['base_operation_events']]==([('C',14),('E',15)] if enabled else [('C',14)])
   assert tr['source_c_writes']==0 and tr['source_forward_calls']==int(enabled)
   assert len(tr['c_writes'])==1 and tr['c_writes'][0]['physical_position']==len(row['gate_prompt_ids'])-1
   assert math.isclose(math.log(gate['p_yes']/gate['p_no']),gate['yes_minus_no'],abs_tol=5e-6)
   delta=gate['yes_minus_no']-row['gate_readout'][control]['yes_minus_no'];entry['gate_deltas'].append(delta)
   entry['text_changes_from_control']+=int(row['outputs'][name]!=row['outputs'][control])
   entry['unfinished']+=int(not row['finished'][name]);entry['replacement_character']+=int('\ufffd' in row['outputs'][name])
   assert g['outcome'][name]==g['judge'][name]['label'];entry[g['outcome'][name]]+=1
 for name,entry in stats.items():
  sign=-1 if name.endswith('m1') else 1
  entry['gate_expected_sign_count']=sum(sign*d>0 for d in entry['gate_deltas'])
  entry['gate_opposite_sign_count']=sum(sign*d<0 for d in entry['gate_deltas'])
  entry['gate_abs_change_median']=statistics.median(abs(x) for x in entry['gate_deltas'])
  entry['C_actual_ratio_median']=statistics.median(entry['C_actual_ratios'])
 if pilot:
  assert len(raw['pilot_control_reproduction'])==3
  for witness,old in zip(raw['pilot_control_reproduction'],parent['evaluations']):
   assert witness['pair_id']==old['pair_id'] and witness['zero_C_exact_parent']
   for name,value in witness['regenerated'].items():
    assert value['output']==old['outputs'][name] and value['token_ids']==old['generation_token_ids'][name]
    assert value['finished']==old['finished'][name] and value['trace']==old['trace'][name]
 return {'protocol':'e40_independent_inheritance_and_execution_audit_v1','pilot':pilot,'questions':n,
   'parent_grades_preserved_exactly':inherited,'new_C_prefill_writes_checked':events,
   'raw_sha256':sha(run/(stem+'.json')),'graded_sha256':sha(run/(stem+'_scored_items.json')),
   'plan_sha256':sha(run/'plan.json'),'all_assertions_passed':True,'new_conditions':stats}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 j=audit(a.run,a.parent,a.pilot);a.output.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
 print('E40 independent PASS',j['questions'],'parent grades',j['parent_grades_preserved_exactly'],'new C writes',j['new_C_prefill_writes_checked'])
