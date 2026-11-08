from pathlib import Path
import json,hashlib,math,argparse
parser=argparse.ArgumentParser();parser.add_argument('--input-root',type=Path,default=Path('/tmp/td_r22_inputs/runs'));parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True)
report={'scope':'Pinned v5 historical artifacts only; not votes from repaired v6 prompts','phases':{}}
for phase in ['select-tune','oracle-tune']:
 root=args.input_root/phase/'judge2';data={}
 for name in ['cells','allenai','pro','gpt','final']:
  p=root/(name+'.jsonl');data[name]=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 cells={r['cell_id']:r for r in data['cells']};bad=[r for r in data['final'] if r.get('truth')=='F' and r.get('info')=='N' and r.get('label')=='B']
 invalid=[]
 for r in data['allenai']:
  for key in ['truth_p_yes','truth_p_no','info_p_yes','info_p_no']:
   x=r.get(key)
   if x is not None and (not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1):invalid.append([r['cell_id'],key])
 stats={'n':{k:len(v) for k,v in data.items()},'max_prediction_characters':max(len(r['prediction']) for r in data['cells']),
  'above_800_characters':sum(len(r['prediction'])>800 for r in data['cells']),'above_4000_characters':sum(len(r['prediction'])>4000 for r in data['cells']),
  'pro_D':sum(r.get('truth')=='D' or r.get('info')=='D' for r in data['pro']),
  'human_decisions':sum(r.get('source')=='human' for r in data['final']),
  'pro_non_single_letter_raw':sum(any(str(r.get(k,'')).strip() not in allowed for k,allowed in [('truth_raw',{'T','F','D'}),('info_raw',{'S','H','R','D'})]) for r in data['pro'] if not r.get('skipped_abstention_gold') and not r.get('empty_output')),
  'invalid_allenai_probabilities':invalid,
  'false_noninformative_B_count':len(bad),'false_noninformative_B':[{'cell_id':r['cell_id'],'prediction':cells[r['cell_id']]['prediction'],'final':r} for r in bad],
  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.jsonl'))}}
 report['phases'][phase]=stats
(out/'actual_scope.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({p:{k:v for k,v in s.items() if k not in ['source_sha256','false_noninformative_B']} for p,s in report['phases'].items()},ensure_ascii=False,indent=2))
