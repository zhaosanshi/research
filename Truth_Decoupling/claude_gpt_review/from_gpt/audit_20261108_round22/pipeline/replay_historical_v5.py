"""Read-only aggregation audit of existing v5 ballots, NEVER a fresh v6 grade."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve();ROOT=HERE.parents[3];CODE=ROOT/'from_claude/code'
sys.path.insert(0,str(CODE))
from core.judge_two_axis import adjudicate_cell,cell_axes,parse_truth,parse_info,ADJUDICATION_VERSION

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def main(inputs,output):
    result={'scope':'Historical v5 ballot replay under the repaired aggregation/statistics only; no new API/human votes and not a valid v6 grading workspace',
        'ballot_rubric':'qa_two_axis_v5','aggregation_version':ADJUDICATION_VERSION,'new_model_calls':0,'new_human_decisions':0,'runs':{}}
    for phase in ('select-tune','oracle-tune'):
        run=inputs/phase;j=run/'judge2';paths=[run/'tune.json',j/'meta.json',*[j/(n+'.jsonl') for n in ('cells','allenai','pro','gpt','final')]]
        before={str(p.relative_to(inputs)):sha(p) for p in paths}
        cells=rows(j/'cells.jsonl');old={r['cell_id']:r for r in rows(j/'final.jsonl')}
        votes={n:{r['cell_id']:r for r in rows(j/(n+'.jsonl'))} for n in ('allenai','pro','gpt')}
        finals=[];parse_changes=[]
        for c in cells:
            if c['gold_is_abstention']:continue
            parsed={}
            for name in ('pro','gpt'):
                g=votes[name].get(c['cell_id'])
                if g is None:parsed[name]=None;continue
                g=dict(g)
                if not g.get('empty_output') and not g.get('skipped_abstention_gold'):
                    t,i=parse_truth(g.get('truth_raw')),parse_info(g.get('info_raw'))
                    if (t,i)!=(g.get('truth'),g.get('info')):parse_changes.append({'cell_id':c['cell_id'],'judge':name})
                    g.update(truth=t,info=i)
                parsed[name]=g
            rec=adjudicate_cell(c,allenai=votes['allenai'].get(c['cell_id']),pro=parsed['pro'],gpt=parsed['gpt'])
            rec.update(condition=c['condition'],pair_id=c['pair_id']);finals.append(rec)
        changes=[{'cell_id':r['cell_id'],'old':old[r['cell_id']]['label'],'replayed':r['label']} for r in finals if r['label']!=old[r['cell_id']]['label']]
        affected=[]
        for r in finals:
            if r['status']=='final' and r['label']=='B' and cell_axes(r)[1]=='N':
                c=next(c for c in cells if c['cell_id']==r['cell_id'])
                affected.append({'cell_id':r['cell_id'],'question':c['question'],'prediction':c['prediction'],'truth':r['truth'],'info':r['info'],'info_subtype':r['info_subtype']})
        conds={}
        for name in sorted({c['condition'] for c in cells}):
            subset=[r for r in finals if r['condition']==name];fin=[r for r in subset if r['status']=='final'];n=len(subset)
            a=sum(r['label']=='A' for r in fin);old_answered=sum(r['label'] in ('A','B') for r in fin);new_answered=sum(cell_axes(r)[1]=='S' for r in fin)
            conds[name]={'n':n,'n_final':len(fin),'A_unchanged':a,'old_A_plus_B_answered':old_answered,'explicit_info_S_answered':new_answered,
                         'fixed_A_rate_bounds':[a/n,(a+n-len(fin))/n] if n else None,
                         'old_resolved_subset_A_rate':a/len(fin) if fin else None}
        result['runs'][phase]={'source_sha256':before,'n_cells':len(cells),'n_active':len(finals),'statuses':dict(Counter(r['status'] for r in finals)),
            'parser_changes':parse_changes,'terminal_label_changes':changes,'false_noninformative_wrongly_answered':affected,'per_condition':conds}
        assert before=={str(p.relative_to(inputs)):sha(p) for p in paths}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({p:{'labels_changed':len(v['terminal_label_changes']),'wrongly_answered':len(v['false_noninformative_wrongly_answered']),'statuses':v['statuses']} for p,v in result['runs'].items()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.inputs,a.out)
