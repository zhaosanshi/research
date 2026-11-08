#!/usr/bin/env python3
"""Read-only reproductions against the reviewed snapshot. No network/model calls.
Assertions document observed unsafe behavior, not a claim that these are healthy tests.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
CODE=Path(os.environ.get('R22_REVIEWED_CODE',ROOT/'from_claude/code')).resolve()
sys.path[:0]=[str(CODE),str(CODE/'scripts'),str(CODE/'scripts/review_checks')]
import judge_two_axis_run as R
import e42_analysis as A
import e42_qa_oracle_and_select as E
from core import judge_two_axis as T
from core.io_utils import write_jsonl

CFG={'version':T.TWO_AXIS_VERSION,'allenai':{'threshold':.5},'audit':{'n':0,'seed':0},
     'pro':{'model':'old-model','temperature':0},'gpt':{'model':'gpt-fixture'}}

def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def raw(answer='Paris.',gold='Paris.',conditions=('method',),n=1):
 return {'completed':True,'model':'fixture','identity':{'phase':'oracle'},'conditions':list(conditions),
   'evaluations':[{'pair_id':f'q{i}','question':f'Capital question {i}?','gold':[gold],'incorrect':['Berlin.'],
       'outputs':{c:answer for c in conditions},'rule_abstain':{c:False for c in conditions},
       'finished':{c:True for c in conditions}} for i in range(n)]}

def build(p,g=None):
 dump(p/'tune.json',g or raw());R.build_cells(str(p),audit_n=0)
 return R._cells(str(p))

def ballots(p,cells,truth='T',info='S',unresolved_ids=()):
 aa=[];pro=[];gpt=[]
 for c in cells:
  sh=T.qa_sha(c['question'],c['prediction']);cid=c['cell_id']
  aa.append({'cell_id':cid,'qa_sha':sh,'truth_p_yes':.9 if truth=='T' else .1,'info_p_yes':.9 if info=='S' else .1})
  pro.append({'cell_id':cid,'qa_sha':sh,'rubric':T.TWO_AXIS_VERSION,'model':'old-model',
      'truth':'D' if cid in unresolved_ids else truth,'info':info})
  if cid in unresolved_ids:gpt.append({'cell_id':cid,'qa_sha':sh,'rubric':T.TWO_AXIS_VERSION,'model':'gpt-fixture','truth':'D','info':info})
 for name,rows in [('allenai',aa),('pro',pro),('gpt',gpt)]:write_jsonl(rows,str(p/'judge2'/f'{name}.jsonl'))

class StubJudge:
 def __init__(self,model='new-model'):
  self.calls=0;self.model=model;self.llm=SimpleNamespace(n_calls=0,n_cache_hits=0)
 def grade(self,question,gold,prediction,incorrect):
  self.calls+=1;self.llm.n_calls+=1
  return {'judge':'pro','model':self.model,'rubric':T.TWO_AXIS_VERSION,'qa_sha':T.qa_sha(question,prediction),'truth':'F','info':'S'}


def main():
 findings=[]
 def emit(id,observed):findings.append({'id':id,'reproduced':True,'observed':observed})
 with tempfile.TemporaryDirectory(prefix='r22-metrics-') as td:
  root=Path(td)
  def case(name):p=root/name;p.mkdir();return p
  p=case('stale_human');cells=build(p);ballots(p,cells)
  R.human_set(str(p),'q0|method','A',by='fixture-reviewer')
  cells=build(p,raw(answer='Berlin.'));ballots(p,cells,truth='F')
  summary,_=R.adjudicate(str(p),CFG);final=R._rows(str(p/'judge2/final.jsonl'))['q0|method']
  assert final['label']=='A' and final['source']=='human' and final['judges_label']=='B'
  emit('stale_human_applies_to_changed_answer',{'new_answer':'Berlin.','machine_verdict':'B','final':final['label'],'human_record':R._human_verdicts(str(p))})

  p=case('legacy_hash_backfill');cells=build(p,raw(answer='new incompatible answer'))
  old={'q0|method':{'cell_id':'q0|method','truth':'T','info':'S','rubric':T.TWO_AXIS_VERSION,'model':'old-model'}}
  accepted,stats=R._verify_rows(old,cells)
  assert accepted['q0|method']['qa_sha']==T.qa_sha(cells[0]['question'],cells[0]['prediction']) and stats['legacy_backfilled']==1
  emit('missing_identity_backfilled_without_evidence',{'accepted_old_grade':accepted,'statistics':stats})

  p=case('changed_model');cells=build(p);ballots(p,cells)
  judge=StubJudge();cfg=copy.deepcopy(CFG);cfg['pro']['model']='new-model'
  summary=R.run_llm(str(p),'pro',cfg,judge=judge)
  assert judge.calls==0 and R._rows(str(p/'judge2/pro.jsonl'))['q0|method']['model']=='old-model' and summary['model']=='new-model'
  emit('model_changed_but_old_judge_votes_reused',{'calls_to_new_model':judge.calls,'summary_model':summary['model'],'row_model':R._rows(str(p/'judge2/pro.jsonl'))['q0|method']['model']})

  p=case('changed_references');cells=build(p);ballots(p,cells)
  cells=build(p,raw(gold='Changed reference with a different truth claim'))
  judge=StubJudge('old-model');summary=R.run_llm(str(p),'pro',CFG,judge=judge)
  assert judge.calls==0
  emit('references_changed_but_old_truth_vote_reused',{'calls_after_reference_change':judge.calls,'current_reference':cells[0]['gold_refs']})

  p=case('stale_final');cells=build(p);ballots(p,cells);R.adjudicate(str(p),CFG)
  build(p,raw(answer='Berlin.'))
  loaded,rows,_=A.load(str(p),two_axis=True)
  assert rows[0]['outputs']['method']=='Berlin.' and rows[0]['outcome']['method']=='correct'
  emit('E42_accepts_old_final_after_cells_meta_rebuild',{'new_answer':rows[0]['outputs']['method'],'old_terminal_label_applied_as':rows[0]['outcome']['method']})

  p=case('conditional_denominators');cells=build(p,raw(conditions=('stable','pending'),n=2))
  ballots(p,cells,unresolved_ids=('q1|pending',))
  # Stable has one correct and one false; pending has one correct and one unresolved.
  pro=R._rows(str(p/'judge2/pro.jsonl'));aa=R._rows(str(p/'judge2/allenai.jsonl'))
  pro['q1|stable']['truth']='F';aa['q1|stable']['truth_p_yes']=.1
  write_jsonl(list(pro.values()),str(p/'judge2/pro.jsonl'));write_jsonl(list(aa.values()),str(p/'judge2/allenai.jsonl'))
  summary,_=R.adjudicate(str(p),CFG)
  assert summary['per_condition']['stable']['true_info_rate']==.5 and summary['per_condition']['pending']['true_info_rate']==1.
  blocked=False
  try:A.load(str(p),two_axis=True)
  except SystemExit:blocked=True
  assert blocked
  emit('progress_summary_uses_different_resolved_denominators',{'per_condition':summary['per_condition'],'E42_correctly_blocks_unresolved_main_analysis':blocked})

  p=case('false_hedge_answered');g=raw(answer='A false but irrelevant hedge',conditions=('baseline',));cells=build(p,g);ballots(p,cells,truth='F',info='H')
  summary,_=R.adjudicate(str(p),CFG);r,rs,outcomes=A.load(str(p),two_axis=True);report,_=A.oracle_report(r,rs,outcomes,20,0)
  assert summary['per_condition']['baseline']['info_rate']==0 and report['arms']['baseline']['answered']==1
  emit('E42_counts_false_uninformative_hedge_as_substantive_answer',{'two_axis_info_rate':0,'E42_answered':report['arms']['baseline']['answered'],'final':R._rows(str(p/'judge2/final.jsonl'))['q0|baseline']})

  p=case('duplicate_judge_rows');cells=build(p);cid=cells[0]['cell_id']
  write_jsonl([{'cell_id':cid,'truth':'T'},{'cell_id':cid,'truth':'F'}],str(p/'votes.jsonl'))
  accepted=R._rows(str(p/'votes.jsonl'));assert len(accepted)==1 and accepted[cid]['truth']=='F'
  emit('duplicate_judge_ids_silently_last_wins',{'input_rows':2,'retained_rows':len(accepted),'retained_truth':accepted[cid]['truth']})

  # Actual E42 selector claims a within-question z-score mean but uses raw coordinates.
  names=['a','b','c'];x=np.asarray([[100.,0.],[90.,10.],[0.,1.]])
  row={'rule_abstain':dict.fromkeys(names,False),'outputs':dict.fromkeys(names,'nonempty'),'readout':{
      name:{'mean_token_logp':0.,'verdict_margin':0.,'coord_dasce_L1':x[i,0],'coord_dasce_L2':x[i,1]} for i,name in enumerate(names)}}
  chosen=E.picks(row,names,{'dasce':{1:None,2:None}},[1,2],42)['coord_dasce_mean']
  claimed=names[int(((x-x.mean(0))/x.std(0)).mean(1).argmax())]
  assert chosen=='a' and claimed=='b'
  emit('E42_claimed_zscore_mean_is_raw_mean',{'coordinate_matrix':x.tolist(),'actual_selected':chosen,'within_question_zscore_selected':claimed})

  p=case('legacy_analysis_unbound');g=raw(conditions=('baseline',));dump(p/'tune.json',g)
  scored=copy.deepcopy(g);scored['grading_completed']=True;scored['evaluations'][0].update(outputs={'baseline':'different answer'},outcome={'baseline':'correct'},gold_is_abstention=False)
  dump(p/'tune_scored_items.json',scored);_,rows,_=A.load(str(p),False)
  assert rows[0]['outcome']['baseline']=='correct'
  emit('legacy_E42_accepts_grade_of_different_output',{'current_answer':rows[0]['outputs']['baseline'],'grade_answer':scored['evaluations'][0]['outputs']['baseline']})

 sources=[CODE/'scripts/review_checks/judge_two_axis_run.py',CODE/'scripts/review_checks/e42_analysis.py',CODE/'scripts/e42_qa_oracle_and_select.py']
 result={'scope':'CPU executions of existing production functions with synthetic artifacts/stub judge; no source modification, network or actual human judgment',
   'findings':findings,'n_reproduced':len(findings),'source_sha256':{'from_claude/code/'+str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
 out=Path(__file__).with_name('reproductions.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('REPRODUCED',len(findings));print('\n'.join(f['id'] for f in findings));print(out)

if __name__=='__main__':main()
