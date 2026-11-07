#!/usr/bin/env python3
"""Post-hoc diagnostic: fixed-E comparisons and answer/score recombination.
No condition is chosen on test performance. Match each frozen EC condition to
its saved E-only condition by its E specification. Four counterfactual tables
reuse saved answers and scores; these are an algebraic decomposition, not new
model interventions or confirmatory significance tests.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np

CODE = Path(__file__).resolve().parents[3] / 'from_claude/code/scripts/review_checks'
sys.path.insert(0, str(CODE))
from e26_independent_recount import quota, exact, read, require


def ranks(values):
    a = np.asarray(values); order = np.argsort(a, kind='stable'); result = np.zeros(len(a))
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def recombine(answer_rows, score_rows, coverage, seed, score_key='score'):
    s = {r['id']: r[score_key] for r in score_rows}
    require({r['id'] for r in answer_rows} == set(s), 'recombination cohorts differ')
    rows = [{**r, 'score': s[r['id']]} for r in answer_rows]
    d = quota(rows, coverage, 'score', seed)
    if d is None:
        return None, None
    return sum(v == 'correct' for v in d.values()), {i for i,v in d.items() if v != 'abstain'}


def contrast(old, new, coverage, seed):
    a = sorted(old, key=lambda r: r['id']); b = sorted(new, key=lambda r: r['id'])
    require([r['id'] for r in a] == [r['id'] for r in b], 'comparison cohort changed')
    f00, mask00 = recombine(a,a,coverage,seed); f01, mask01 = recombine(a,b,coverage,seed)
    f10, mask10 = recombine(b,a,coverage,seed); f11, mask11 = recombine(b,b,coverage,seed)
    r = {'n': len(a), 'old_native_answered': sum(x['outcome']!='abstain' for x in a),
         'new_native_answered': sum(x['outcome']!='abstain' for x in b),
         'outputs_changed': sum(x['output'] != y['output'] for x,y in zip(a,b)),
         'outcomes_changed': sum(x['outcome'] != y['outcome'] for x,y in zip(a,b)),
         'score_spearman': float(np.corrcoef(ranks([x['score'] for x in a]), ranks([x['score'] for x in b]))[0,1]),
         'score_shift_mean': float(np.mean([y['score']-x['score'] for x,y in zip(a,b)])),
         'old_answer_old_score': f00, 'old_answer_new_score': f01,
         'new_answer_old_score': f10, 'new_answer_new_score': f11}
    if all(x is not None for x in (f00,f01,f10,f11)):
        r.update(total_delta=f11-f00, score_component=((f01-f00)+(f11-f10))/2,
                 answer_component=((f10-f00)+(f11-f01))/2,
                 delivery_replacements=len(mask00^mask11)//2,
                 score_only_replacements=len(mask00^mask01)//2)
        d0,d1=quota(a,coverage,'score',seed),quota(b,coverage,'score',seed)
        r['exact_fixed_e_comparison']=exact(d1,d0,{x['id']:x['pair_id'] for x in a})
    return r


def audit(run):
    b=read(run/'bank.json');s=read(run/'selection.json');seed=b['config']['seed'];out=[]
    for task in ('tf','qa'):
        d=read(run/(task+'_test'+('_scored' if task=='qa' else '')+'.json'))
        rs={name:[x for x in rows if not x.get('gold_is_abstention')] for name,rows in d['rows'].items()}
        specs={x['name']:x for x in d['condition_specs']}
        for q,picks in s['tasks'][task]['choices'].items():
            for method in ('das','md'):
                e,ec,c=[picks.get(method+'_'+x) for x in ('E','EC','C')]
                rec={'model':b['model'],'task':task,'coverage':float(q),'method':method,
                     'frozen_E':e,'frozen_EC':ec,'frozen_C':c}
                if not e or not ec or not c:
                    rec['infeasible']=True;out.append(rec);continue
                matches=[name for name,sp in specs.items() if sp['method']==method and sp['kind']=='E' and sp['e']==specs[ec]['e']]
                require(len(matches)==1,'matched E-only test arm absent or ambiguous')
                me=matches[0];rec['matched_E_for_EC']=me;rec['same_E_as_primary']=e==me
                rec['C_vs_base']=contrast(rs['baseline'],rs[c],q,seed)
                rec['EC_vs_matched_E']=contrast(rs[me],rs[ec],q,seed)
                rec['EC_vs_independently_selected_E']=contrast(rs[e],rs[ec],q,seed)
                baseline_count,_=recombine(rs[e],rs[e],q,seed)
                matched_count,_=recombine(rs[me],rs[me],q,seed)
                rec['E_configuration_component']=matched_count-baseline_count
                # Same saved generation conditions, both ranked by simple confidence.
                counts={name:recombine(rs[name],rs[name],q,seed,'simple_score')[0] for name in (e,me,ec)}
                rec['same_saved_conditions_simple_score_correct']=counts
                out.append(rec)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-json',type=Path,required=True);p.add_argument('runs',type=Path,nargs='+');args=p.parse_args()
    rows=[r for run in args.runs for r in audit(run)]
    result={'exploratory':True,'selection':'frozen EC determines the matching E; no test-based search',
      'interpretation':'algebraic answer/score recombination, not a causal mediation claim; original E/EC tuning selections remain unchanged',
      'rows':rows}
    args.out_json.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for r in rows:
        if r['method']!='das':continue
        if r.get('infeasible'):print(r['model'],r['task'],r['coverage'],'infeasible');continue
        x=r['EC_vs_matched_E'];c=r['C_vs_base'];raw=r['EC_vs_independently_selected_E']
        print(r['model'],r['task'],r['coverage'],'primary',raw['total_delta'],'sameE',r['same_E_as_primary'],
              'E-config',r['E_configuration_component'],'fixedE',x['total_delta'],
              'score',x['score_component'],'answer',x['answer_component'],'C/base replaced',c['delivery_replacements'])
if __name__=='__main__':main()
