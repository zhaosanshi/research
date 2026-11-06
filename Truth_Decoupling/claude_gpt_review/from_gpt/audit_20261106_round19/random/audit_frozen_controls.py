#!/usr/bin/env python3
"""Independently audit the executed frozen-random control archive."""
import argparse
from collections import Counter
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import tarfile

import numpy as np
import torch


def sha(b):
    return hashlib.sha256(b).hexdigest()


def tsha(x):
    x = x.detach().float().cpu().contiguous()
    return sha(json.dumps({'shape': list(x.shape), 'dtype': 'float32'}, sort_keys=True).encode() + x.numpy().tobytes())


def exact(a, b):
    plus, minus = int((a & ~b).sum()), int((~a & b).sum()); n = plus + minus
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(plus, minus)+1)) / 2**n) if n else 1.
    return {'corrected': plus, 'harmed': minus, 'net': plus-minus, 'exact_two_sided_p': p}


def run(archive, manifest, old_archive):
    checks = []
    def ck(name, value):
        checks.append({'name': name, 'pass': bool(value)})
        if not value:
            raise ValueError(name)
    rows = json.loads(Path(manifest).read_text())
    rows = rows['files'] if isinstance(rows, dict) else rows
    expected = {x['path']: x for x in rows}
    result = {'archive_sha256': sha(Path(archive).read_bytes()), 'models': {}}
    with tarfile.open(archive) as tar, tarfile.open(old_archive) as old:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        ck('manifest_paths', set(members) == set(expected))
        for name, member in members.items():
            b = tar.extractfile(member).read()
            ck('archive_sha/'+name, sha(b) == expected[name]['sha256'])
        for model in ('llama3.1-8b', 'qwen3-8b'):
            adapters = {init: torch.load(io.BytesIO(old.extractfile(model+'/ce_alpha3_ep3/'+init+'/adapter.pt').read()),
                                        map_location='cpu', weights_only=True) for init in ('das', 'random')}
            cached = {init: json.load(old.extractfile(model+'/ce_alpha3_ep3/'+init+'/tune.json')) for init in ('das','random')}
            cached_scored = {init: json.load(old.extractfile(model+'/ce_alpha3_ep3/'+init+'/tune_scored_items.json')) for init in ('das','random')}
            old_by_id = {i: {r['pair_id']: r for r in d['evaluations']} for i,d in cached.items()}
            old_grade_by_id = {i: {r['pair_id']: r for r in d['evaluations']} for i,d in cached_scored.items()}
            for phase in ('pilot3','full192'):
                prefix = model+'/'+phase+'/'
                raw_bytes = tar.extractfile(prefix+'tune.json').read(); raw=json.loads(raw_bytes)
                scored=json.load(tar.extractfile(prefix+'tune_scored_items.json'))
                summary=json.load(tar.extractfile(prefix+'tune_scored.json'))
                ck(prefix+'complete',raw['completed'] and summary['completed'] and scored['completed'])
                ck(prefix+'source_generation',summary['source_generation_sha256']==sha(raw_bytes))
                ck(prefix+'no_reproduction_mismatch',not raw['reproduction_mismatches'])
                ck(prefix+'reproduced_reference_count',len(raw['cached_reproduction'])==4*len(raw['evaluations']))
                ck(prefix+'reproduced_reference_flags',all(x['exact_text_equal'] and x['finished_equal'] for x in raw['cached_reproduction']))
                ck(prefix+'base_unchanged',raw['base_parameter_versions_and_flags_unchanged'])
                for rel,h in raw['identity']['code_sha256'].items():
                    ck(prefix+'code/'+rel,sha(tar.extractfile(prefix+'code_snapshot/'+rel).read())==h)
                for init in ('das','random'):
                    ck(prefix+'adapter_hash/'+init,raw['identity'][init+'_adapter_sha256']==sha(old.extractfile(model+'/ce_alpha3_ep3/'+init+'/adapter.pt').read()))
                    path=model+'/ce_alpha3_ep3/'+init+'/tune.json'
                    ck(prefix+'cached_source/'+init,raw['identity']['cached_reference_generations'][init]['sha256']==sha(old.extractfile(path).read()))
                layer=raw['identity']['layers'][0]; l=str(layer)
                vectors={'original_das':adapters['das']['initial_W'][l],
                         'frozen_random_initial':adapters['random']['initial_W'][l],
                         'trained_random':adapters['random']['W'][l], 'trained_das':adapters['das']['W'][l]}
                for name,w in vectors.items(): ck(prefix+'vector/'+name,raw['identity']['vector_sha256'][name][l]==tsha(w))
                ids=[x['pair_id'] for x in raw['evaluations']]
                ck(prefix+'ordered_ids',[x['pair_id'] for x in scored['evaluations']]==ids==[x['pair_id'] for x in cached['random']['evaluations']][:len(ids)])
                for r,g in zip(raw['evaluations'],scored['evaluations']):
                    for field in ('question','gold','incorrect','outputs','finished'):
                        ck(prefix+r['pair_id']+'/'+field,r[field]==g[field])
                    for name in raw['conditions']:
                        ck(prefix+r['pair_id']+'/'+name+'/derived',g['outcome'][name]==g['judge'][name]['label'])
                        ck(prefix+r['pair_id']+'/'+name+'/tokens_present',bool(r['prompt_token_ids'][name]) and bool(r['generation_token_ids'][name]))
                        if name=='frozen_random_initial':continue
                        init,oldname=('das','das_generation_adapter') if name=='trained_das' else ('random','random_generation_adapter' if name=='trained_random' else name)
                        ck(prefix+r['pair_id']+'/'+name+'/old_text',r['outputs'][name]==old_by_id[init][r['pair_id']]['outputs'][oldname])
                        ck(prefix+r['pair_id']+'/'+name+'/old_grade',g['outcome'][name]==old_grade_by_id[init][r['pair_id']]['outcome'][oldname])
                if phase!='full192':continue
                ck(prefix+'192',len(ids)==192 and summary['n_resolved_all_conditions']==192)
                labs={name:np.array([r['outcome'][name] for r in scored['evaluations']]) for name in raw['conditions']}
                rng=np.random.default_rng(0);boot=rng.integers(0,192,(2000,192))
                out={'n':192,'scores':{},'comparisons':{},'frozen_random_changed_rows':[]}
                for name,a in labs.items():
                    count=dict(Counter(a));b=raw['evaluations']
                    count.update(answered=int((a!='abstain').sum()),bad_characters=sum('\ufffd' in r['outputs'][name] for r in b),
                                 unfinished=sum(not r['finished'][name] for r in b))
                    for label in ('correct','wrong','abstain'):
                        ck(prefix+name+'/rate/'+label,abs((a==label).mean()-summary['rates'][name]['rates'][label])<1e-12)
                    out['scores'][name]=count
                for first,second in itertools.combinations(raw['conditions'],2):
                    a,b=labs[first]=='correct',labs[second]=='correct';delta=a.astype(int)-b.astype(int)
                    out['comparisons'][first+'_minus_'+second]={**exact(a,b),'delta':float(delta.mean()),
                        'ci95':np.percentile(delta[boot].mean(1),[2.5,97.5]).tolist()}
                for rawrow,g in zip(raw['evaluations'],scored['evaluations']):
                    if g['outcome']['baseline']!=g['outcome']['frozen_random_initial']:
                        out['frozen_random_changed_rows'].append({'pair_id':g['pair_id'],'question':g['question'],
                            'baseline':rawrow['outputs']['baseline'],'frozen_random':rawrow['outputs']['frozen_random_initial'],
                            'before':g['outcome']['baseline'],'after':g['outcome']['frozen_random_initial']})
                result['models'][model]=out
    result['checks']={'n':len(checks),'failures':[x for x in checks if not x['pass']]}
    result['scope']='Original E27 cache. Do not merge these counts with release rerun/dose-probe caches.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--old-archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();d=run(args.archive,args.manifest,args.old_archive)
    args.out.write_text(json.dumps(d,ensure_ascii=False,indent=2))
    print(json.dumps({'checks':d['checks'],'models':{m:v['scores'] for m,v in d['models'].items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
