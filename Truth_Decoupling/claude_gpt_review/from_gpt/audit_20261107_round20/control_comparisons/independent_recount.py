"""Independent standard-library recount; does not import experiment modules."""
import argparse,hashlib,json,math
from pathlib import Path

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank_hash(pid):return hashlib.sha256(json.dumps(['actual_coverage_v1',42,pid],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def main(root,out):
 results=[];cells=tests=0
 for group in ('results_r20_norm_random_codex_20261107','results_source_question_mask_codex_20261107','results_e34_completion_codex_20261107'):
  for p in sorted((root/group).glob('*/*/tune_matched_coverage.json')):
   d=read(p);raw=read(p.with_name('tune.json'));grade=read(p.with_name('tune_scored_items.json'));scores=read(p.with_name('tune_simple_scores.json'))
   assert d['source_files']=={f:digest(p.with_name(f)) for f in ('tune.json','tune_simple_scores.json','tune_scored_items.json')}
   ids=[r['pair_id'] for r in raw['evaluations']];assert len(ids)==len(set(ids))==192
   assert all(not r['gold_is_abstention'] for r in grade['evaluations'])
   assert [r['pair_id'] for r in grade['evaluations']]==[r['pair_id'] for r in scores['rows']]==ids
   gs={r['pair_id']:r for r in grade['evaluations']};ss={r['pair_id']:r['scores'] for r in scores['rows']}
   for name in raw['conditions']:
    assert d['natural'][name]=={'n':192,'correct':sum(gs[i]['outcome'][name]=='correct' for i in ids),'answered':sum(gs[i]['outcome'][name]!='abstain' for i in ids)}
    assert d['quality'][name]=={'replacement_character':sum('\ufffd' in r['outputs'][name] for r in raw['evaluations']),'unfinished':sum(not r['finished'][name] for r in raw['evaluations'])}
   for point in d['matched_coverage']['coverages']:
    k={.25:48,.5:96,.75:144}[point['target_coverage']];delivery={}
    for name,row in point['arms'].items():
     available=[i for i in ids if gs[i]['outcome'][name]!='abstain'];ranked=sorted(available,key=lambda i:(-ss[i][name],rank_hash(i),i));cells+=1
     assert row['feasible']==(len(available)>=k)
     if len(available)<k:
      assert row['selected_ids'] is None and row['correct'] is None;continue
     selected=set(ranked[:k]);assert sorted(selected)==row['selected_ids']
     delivered={i:(gs[i]['outcome'][name] if i in selected else 'abstain') for i in sorted(ids)}
     assert delivered==row['delivered_outcomes'];assert row['correct']==sum(v=='correct' for v in delivered.values())
     assert row['answered']==k;delivery[name]=delivered
    for c in point['comparisons']:
     a,b=c['arm_a'],c['arm_b']
     if a not in delivery or b not in delivery:continue
     aa,bb=delivery[a],delivery[b];plus=sum(aa[i]=='correct' and bb[i]!='correct' for i in ids);minus=sum(bb[i]=='correct' and aa[i]!='correct' for i in ids);n=plus+minus
     prob=min(1.,2*sum(math.comb(n,j) for j in range(min(plus,minus)+1))/(2**n)) if n else 1.
     exact=c['exact_fixed_deliveries'];assert abs(exact['p_value']-prob)<2e-14
     assert exact['positive_discordant_rows']==plus and exact['negative_discordant_rows']==minus;tests+=1
   results.append({'run':str(p.parent.relative_to(root)),'raw_sha256':digest(p.with_name('tune.json')),'score_sha256':digest(p.with_name('tune_simple_scores.json')),'grade_sha256':digest(p.with_name('tune_scored_items.json')),'report_sha256':digest(p),'all_checks_passed':True})
 assert len(results)==6
 out.write_text(json.dumps({'runs':results,'quota_cells':cells,'feasible_fixed_delivery_exact_tests':tests,'score_recomputed_with_model':False,'bootstrap_recomputed_here':False,'scope':'Independent selection IDs, full delivered outcomes, natural/quality counts and exact paired probabilities from raw files; no experiment module imports'},indent=2)+'\n')
 print('verified',len(results),'runs',cells,'quota cells',tests,'exact tests')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
