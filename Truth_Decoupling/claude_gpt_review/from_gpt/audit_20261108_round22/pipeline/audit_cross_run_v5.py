"""Read-only same-input conflicts in the two captured historical E42/v5 runs.
No label is chosen as correct and no old ballot is converted into a v6 ballot.
"""
import argparse,hashlib,json
from pathlib import Path

def read_rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root,output):
 index={};sources={}
 for phase in ('select-tune','oracle-tune'):
  base=root/phase;j=base/'judge2';meta=json.loads((j/'meta.json').read_text())
  assert meta['version']=='qa_two_axis_v5' and meta['generation_sha256']==sha(base/'tune.json')
  cells=read_rows(j/'cells.jsonl');fs=read_rows(j/'final.jsonl');final={r['cell_id']:r for r in fs}
  assert len(final)==len(fs)==len(cells) and len({c['cell_id'] for c in cells})==len(cells)
  sources[phase]={n:sha(j/n) for n in ('meta.json','cells.jsonl','final.jsonl')}
  for c in cells:
   key=json.dumps({k:c.get(k) for k in ('question','prediction','gold_refs','incorrect')},sort_keys=True,ensure_ascii=False)
   r=final[c['cell_id']];index.setdefault(key,[]).append({'run':phase,'cell_id':c['cell_id'],'label':r['label'],'status':r['status']})
 conflicts=[];shared=0
 for key,values in index.items():
  if len({r['run'] for r in values})<2:continue
  shared+=1;labels={r['label'] for r in values if r['status']=='final'}
  if len(labels)>1:conflicts.append({'input':json.loads(key),'outcomes':values})
 result={'scope':'Historical v5 same full semantic inputs across the two captured E42 runs; these are conflicts requiring consistent re-evaluation, not proof of which label is correct',
         'legacy_request_parameters_not_fully_recorded':True,'shared_requests':shared,'conflicts':conflicts,'source_sha256':sources,'new_calls':0,'new_human_labels':0}
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('shared semantic requests',shared,'conflicting finals',len(conflicts))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.inputs,a.out)
