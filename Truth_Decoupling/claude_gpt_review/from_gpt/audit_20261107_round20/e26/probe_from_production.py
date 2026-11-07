#!/usr/bin/env python3
"""Recount alpha +/-1 dose summaries from existing production tuning margins.
These already used E26 fp32 output projection and batch size 8. No new model
execution or label-dependent selection; use first 64 rows exactly as old probe.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
CHECKS=ROOT/'from_claude/code/scripts/review_checks';sys.path.insert(0,str(CHECKS))
from e26_independent_recount import digest
from e26_c_dose_probe import spearman


def main():
    root=ROOT/'from_claude/results/03_E26同作答率_E27生成适配';out=[]
    for model in ('llama3.1-8b','qwen3-8b'):
        old=json.loads((root/'探测'/('c_dose_probe_'+model+'.json')).read_text())
        for task in ('tf','qa'):
            path=root/'E26_main'/model/(task+'_tune.json');d=json.loads(path.read_text());base=d['rows']['baseline'][:64]
            m0=np.array([r['score'] for r in base]);sd=float(np.std(m0))
            for spec in d['condition_specs']:
                if spec['kind']!='C':continue
                name=spec['name'];rows=d['rows'][name][:64];m=np.array([r['score'] for r in rows]);a=spec['c']['alpha']
                method=spec['method']+'_L'+str(spec['c']['layer']);previous=old['tasks'][task]['methods'][method][str(a)]
                new={'mean_shift':float((m-m0).mean()),'shift_in_sd':float((m-m0).mean()/sd),
                     'yes_share':float((m>0).mean()),'flip_share':float(((m>0)!=(m0>0)).mean()),'rank_spearman_vs_base':spearman(m,m0)}
                out.append({'model':model,'task':task,'method':method,'alpha':a,'n':len(rows),
                            'production_source_sha256':digest(path),'production_batch_size':8,
                            'baseline_yes_share':float((m0>0).mean()),'baseline_margin_sd':sd,
                            'corrected_summary':new,'old_probe_v1_summary':previous,
                            'production_margins':m.tolist(),'production_baseline_margins':m0.tolist()})
    result={'protocol':'e26_c_probe_production_recount_v1','n':64,'alphas':[-1.,1.],
            'source':'saved production E26 tuning margins; no further model execution or data-dependent selection',
            'notes':'Old v1 +/-2 points have not been recomputed at 64 inputs; keep them as legacy protocol, not this result.', 'rows':out}
    target=Path(__file__).with_name('probe_production64_from_saved_rows.json')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for r in out:
        if not r['method'].startswith('das'):continue
        a=r['corrected_summary'];b=r['old_probe_v1_summary']
        print(r['model'],r['task'],r['alpha'],'shift',a['mean_shift'],'rho',a['rank_spearman_vs_base'],
              'old_shift',b['mean_shift'],'old_rho',b['rank_spearman_vs_base'])

if __name__=='__main__':main()
