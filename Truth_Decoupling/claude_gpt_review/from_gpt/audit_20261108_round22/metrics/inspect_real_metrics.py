#!/usr/bin/env python3
"""Read-only audit of the two historical v5 runs; no judging or relabeling."""
import argparse
import hashlib
import json
from pathlib import Path

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 result={}
 for name in ('select-tune','oracle-tune'):
  p=a.runs/name;raw=read(p/'tune.json');finals=lines(p/'judge2/final.jsonl');byid={r['cell_id']:r for r in finals}
  assert len(byid)==len(finals)
  false_noninfo=[];quality=[];flagged=[]
  for row in raw['evaluations']:
   eligible=[c for c in raw['conditions'] if not row['rule_abstain'][c] and row['outputs'][c].strip()]
   final_pool=eligible or list(raw['conditions'])
   for condition in raw['conditions']:
    cell=byid[f'{row["pair_id"]}|{condition}'];pred=row['outputs'][condition]
    if cell.get('status')=='final' and cell.get('label')=='B' and cell.get('info')=='N':
     false_noninfo.append({'pair_id':row['pair_id'],'condition':condition,'prediction':pred,'info_subtype':cell.get('info_subtype')})
    if not row['finished'][condition] or '\ufffd' in pred:
     quality.append({'pair_id':row['pair_id'],'condition':condition,'finished':row['finished'][condition],
                     'unicode_replacement':'\ufffd' in pred,'v5_label':cell.get('label')})
    if name=='select-tune' and row['rule_abstain'][condition] and cell.get('status')=='final' and cell.get('label')=='A':
     flagged.append({'pair_id':row['pair_id'],'question':row['question'],'condition':condition,'prediction':pred,
       'lexically_flagged_but_v5_A':True,'actually_excluded_from_final_pool':condition not in final_pool,
       'whole_pool_fallback':not eligible,'unflagged_alternatives':eligible,
       'all_candidates':[{'condition':c,'prediction':row['outputs'][c],'rule_abstain':row['rule_abstain'][c],
                         'v5_label':byid[f'{row["pair_id"]}|{c}'].get('label')} for c in raw['conditions']],
       'old_picks':row.get('picks')})
  result[name]={'n_questions':len(raw['evaluations']),'n_conditions':len(raw['conditions']),'n_cells':len(finals),
    'conditions':raw['conditions'],'n_pending':sum(r['status']!='final' and r['status']!='excluded_abstention_gold' for r in finals),
    'false_noninformative_cells':false_noninfo,'false_noninformative_unique_questions':sorted({r['pair_id'] for r in false_noninfo}),
    'quality_flags':quality,'unfinished_count':sum(not r['finished'] for r in quality),
    'content_A_unfinished':sum(not r['finished'] and r['v5_label']=='A' for r in quality),
    'flagged_v5_A':flagged,'actually_excluded_v5_A_count':sum(r['actually_excluded_from_final_pool'] for r in flagged),
    'source_sha256':{f:sha(p/f) for f in ('tune.json','judge2/final.jsonl','judge2/meta.json','judge2/summary.json')}}
 a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 for name,r in result.items():print(name,{k:r[k] for k in ('n_cells','n_pending','unfinished_count','content_A_unfinished','actually_excluded_v5_A_count')})
