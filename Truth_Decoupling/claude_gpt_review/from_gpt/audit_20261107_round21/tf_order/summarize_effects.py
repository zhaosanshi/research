#!/usr/bin/env python3
"""Separate C's actual norm and gate/answer score effect from the fixed E."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
out={}
for model in ('llama3.1-8b','qwen3-8b'):
    run=ROOT/'results_calibrated'/model/'full192'
    plan=json.loads((run/'plan.json').read_text());raw=json.loads((run/'tune.json').read_text())
    report=json.loads((run/'report.json').read_text())
    assert raw['completed'] and plan['pilot_limit'] is None
    items=[]
    for c in plan['conditions']:
        if c['kind'] not in ('C','EC'):continue
        rows=raw['rows'][c['name']];reference=raw['rows']['baseline' if c['kind']=='C' else c['method']+'_E']
        assert [r['id'] for r in rows]==[r['id'] for r in reference]
        item={'condition':c,'n':len(rows),'C_contexts':{}}
        for context in ('answer','gate'):
            events=[next(w for w in r[context+'_writes'] if w['layer']==c['c']['layer']) for r in rows]
            item['C_contexts'][context]={key:{'min':float(np.min([v[key] for v in events])),
                 'median':float(np.median([v[key] for v in events])),
                 'max':float(np.max([v[key] for v in events]))} for key in ('requested_delta_l2','actual_delta_l2','actual_delta_over_residual')}
        for key in ('score','answer_margin'):
            differences=np.asarray([r[key]-b[key] for r,b in zip(rows,reference)])
            item[key+'_change']={'mean':float(np.mean(differences)),'median':float(np.median(differences)),
                'min':float(differences.min()),'max':float(differences.max()),'positive_count':int((differences>0).sum()),'negative_count':int((differences<0).sum())}
        item['selected_sets_vs_reference']={}
        refname='baseline' if c['kind']=='C' else c['method']+'_E'
        for selector in ('simple','commitment'):
            item['selected_sets_vs_reference'][selector]={}
            for q in ('0.25','0.5','0.75'):
                current=set(report['conditions'][c['name']]['quotas'][selector][q]['selected_ids'])
                previous=set(report['conditions'][refname]['quotas'][selector][q]['selected_ids'])
                item['selected_sets_vs_reference'][selector][q]={'same_selected':len(current&previous),'newly_selected':len(current-previous),'deselected':len(previous-current)}
        item['answer_changed']=sum(r['output']!=b['output'] for r,b in zip(rows,reference))
        item['gate_yes_count']=sum(r['score']>0 for r in rows)
        item['gate_label_mass_min']=min(r['gate_p_yes']+r['gate_p_no'] for r in rows)
        items.append(item)
    out[model]=items
(ROOT/'full_C_norm_and_gate_effects.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('Saved separate C norm/gate effects for',sum(map(len,out.values())),'conditions')
