#!/usr/bin/env python3
"""Independent saved-output audit of the five frozen original-DAS controls."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import tarfile

import numpy as np

LABELS=('correct','wrong','abstain')


def sha(data):return hashlib.sha256(data).hexdigest()


def exact(corrected,harmed):
    n=corrected+harmed
    return min(1.,2*sum(math.comb(n,k) for k in range(min(corrected,harmed)+1))/2**n) if n else 1.


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=here/'evidence/pure_das_layer_results.tar.gz');p.add_argument('--manifest',type=Path,default=here/'evidence/pure_das_layer_manifest.json');p.add_argument('--source',type=Path,default=here.parents[2]/'from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u/llama3.1-8b');p.add_argument('--out',type=Path,default=here/'pure_das_layer_comparison.json');a=p.parse_args()
    with tarfile.open(a.archive) as t:
        entries={x.name:x for x in t.getmembers() if x.isfile()}
        manifest=json.loads(a.manifest.read_text());expected={r['path']:r for r in manifest}
        assert set(entries)==set(expected)
        for name,m in entries.items():
            b=t.extractfile(m).read();assert len(b)==expected[name]['bytes'] and sha(b)==expected[name]['sha256']
        def content(name):return t.extractfile(name).read()
        def read(name):return json.loads(content(name))
        source_questions=(a.source/'questions.json').read_bytes();questions=json.loads(source_questions)
        source_rows=json.loads((a.source/'tune_scored_items.json').read_text())['evaluations']
        source_by={r['pair_id']:r for r in source_rows}
        result={'archive_sha256':sha(a.archive.read_bytes()),'verified_archive_files':len(entries),'runs':{}}
        for folder in ('llama_pure_das_pilot3','llama_pure_das_tune192'):
            pref=folder+'/'
            raw=read(pref+'tune.json');items=read(pref+'tune_scored_items.json');summary=read(pref+'tune_scored.json');identity=read(pref+'identity.json')
            assert raw['completed'] and items['completed'] and summary['completed']
            assert raw['identity']==items['identity']==identity
            assert identity['training_called_by_this_script'] is False and identity['extra_content_adaptation'] is False
            assert identity['split']=='tune' and identity['alpha']==3 and identity['layers']==[10,29]
            assert content(pref+'questions.json')==source_questions
            for name,h in identity['code_sha256'].items():assert sha(content(pref+'code_snapshot/'+name))==h
            assert summary['source_generation_sha256']==sha(content(pref+'tune.json'))
            assert raw['generation_context_sha256']==sha(content(pref+raw['generation_context_file']))
            assert all(x['exact_text_equal'] and x['finished_equal'] for x in raw['source_reproduction'])
            rows=items['evaluations'];ids=[r['pair_id'] for r in rows]
            assert ids==identity['selected_ids'] and len(ids)==len(set(ids))
            assert not set(ids)&{r['pair_id'] for r in questions['splits']['test']}
            assert not set(ids)&{r['pair_id'] for r in questions['e_train']}
            assert [r['outputs'] for r in rows]==[r['outputs'] for r in raw['evaluations']]
            assert len(rows)==(3 if raw['pilot'] else 192)
            base=np.array([r['outcome']['baseline'] for r in rows]);base_correct=base=='correct';n=len(rows)
            boot=np.random.default_rng(0).integers(0,n,(2000,n));table={};changed=[]
            for name in items['conditions']:
                lab=np.array([r['outcome'][name] for r in rows]);assert all(x in LABELS for x in lab)
                assert all(r['outcome'][name]==r['judge'][name]['label'] for r in rows)
                counts={s:int((lab==s).sum()) for s in LABELS}
                for s,v in counts.items():assert abs(summary['rates'][name]['rates'][s]-v/n)<1e-12
                trans={x+'->'+y:int(((base==x)&(lab==y)).sum()) for x in LABELS for y in LABELS}
                diff=(lab=='correct').astype(int)-base_correct.astype(int)
                gain=int((diff==1).sum());harm=int((diff==-1).sum())
                table[name]={**counts,'answered':counts['correct']+counts['wrong'],'transitions':trans,
                    'corrected':gain,'harmed':harm,'net_correct':gain-harm,'delta_accuracy':float(diff.mean()),
                    'percentile_ci95':np.percentile(diff[boot].mean(1),[2.5,97.5]).tolist(),
                    'exact_two_sided_p':exact(gain,harm),
                    'text_changed_vs_baseline':sum(r['outputs'][name]!=r['outputs']['baseline'] for r in rows),
                    'replacement_characters':sum('\ufffd' in r['outputs'][name] for r in rows),
                    'unfinished':sum(not r['finished'][name] for r in rows)}
            controls={}
            for name in ('baseline',identity['source_reference_condition']):
                controls[name]={'text_mismatches':[],'label_mismatches':[],'finished_mismatches':[]}
                for row in rows:
                    old=source_by[row['pair_id']]
                    for field,target in [('outputs','text_mismatches'),('outcome','label_mismatches'),('finished','finished_mismatches')]:
                        if row[field][name]!=old[field][name]:controls[name][target].append(row['pair_id'])
            assert all(not r['text_mismatches'] and not r['finished_mismatches'] for r in controls.values())
            for row in rows:
                if len(set(row['outcome'].values()))>1:
                    changed.append({k:row[k] for k in ('pair_id','question','gold','incorrect','outputs','outcome','finished')})
            result['runs'][folder]={'n':n,'pilot':raw['pilot'],'generation_context_questions':raw['generation_context_questions'],
                'source_questions_bytes_identical':True,'all_source_control_reproduction_checks_passed':True,
                'source_control_comparison':controls,'base_and_vectors_unchanged':raw['base_parameters_and_source_directions_unchanged'],
                'conditions':table,'label_change_cases':changed}
        a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    full=result['runs']['llama_pure_das_tune192'];print('n',full['n'],'source controls',full['source_control_comparison'])
    for name,r in full['conditions'].items():print(name,{k:r[k] for k in ('correct','wrong','abstain','corrected','harmed','net_correct','text_changed_vs_baseline','replacement_characters','unfinished','exact_two_sided_p')})
    for row in full['label_change_cases']:print('CHANGE',row['pair_id'],row['question'],row['outputs'],row['outcome'])


if __name__=='__main__':main()
