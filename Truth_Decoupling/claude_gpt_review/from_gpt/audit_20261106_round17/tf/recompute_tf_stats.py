#!/usr/bin/env python3
"""Independent TF audit: raw NPZ -> exact choices, SciPy-rank AUC and pair CIs.
No production statistics/selection function is imported. No model inference.
"""
import hashlib
import json
import re
from pathlib import Path
import os

import numpy as np
from scipy.stats import rankdata

ROOT = Path(os.environ.get('TD_AUDIT_CLAUDE_ROOT', str(Path(__file__).resolve().parents[3] / 'from_claude')))
SEARCH = ROOT / 'results/02_问答DAS_v1/第16轮_DAS正结果搜索'
OUT = Path(os.environ.get('TD_AUDIT_OUT', str(Path(__file__).resolve().parent))); OUT.mkdir(parents=True, exist_ok=True)
OLD = Path(os.environ.get('TD_AUDIT_OLD_TF', str(Path(__file__).resolve().parent / 'old_inputs')))
KEYS = ('ans', 'J', 'AUROC', 'B')
MODELS = ('llama3.1-8b', 'qwen3-8b', 'gemma3-4b')

def metric(a, p, y):
    a, p, y = np.atleast_2d(a), np.atleast_2d(p), np.atleast_2d(y).astype(bool)
    pred = p > .5
    pos, neg = y.sum(axis=1), (~y).sum(axis=1)
    auc = ((rankdata(p, method='average', axis=1) * y).sum(axis=1) - pos*(pos+1)/2) / (pos*neg)
    return np.column_stack((a.mean(axis=1), (pred*y).sum(axis=1)/pos-(pred*(~y)).sum(axis=1)/neg, auc, pred.mean(axis=1)))

def scalar_metrics(a,p,y):
    return dict(zip(KEYS,map(float,metric(a,p,y)[0])))

def summary(point, draws):
    return {k:{'delta':float(point[i]),'ci95':np.percentile(draws[:,i],[2.5,97.5]).tolist()} for i,k in enumerate(KEYS)}

class Sweep:
    def __init__(self,p):
        self.path=Path(p); self.j=json.loads(self.path.read_text())
        self.z=dict(np.load(self.path.with_suffix('.npz'),allow_pickle=True))
        self.y=self.z['labels'].astype(bool); self.tune=self.z['half_a'].astype(bool); self.test=~self.tune
        self.pairs=np.array([i.rsplit(':',1)[0] for i in self.z['ids']]); self.names=list(self.j['conds'])
        self.raw={}; self.boot_cache={}
        self.error=0.; self.disagreements=[]
        for n in self.names:
            a,p=self.arr(n)
            assert np.isfinite(p).all()
            self.raw[n]={}
            for split,mask in [('A',self.tune),('B',self.test),('all',np.ones(len(self.y),dtype=bool))]:
                vals=scalar_metrics(a[mask],p[mask],self.y[mask]); self.raw[n][split]=vals
                for k,v in vals.items():
                    d=abs(v-self.j['conds'][n][split][k]);self.error=max(d,self.error)
                    if d>1e-12:self.disagreements.append([n,split,k,v,self.j['conds'][n][split][k]])
        groups=[np.where((self.pairs==p)&self.test)[0] for p in sorted(set(self.pairs[self.test]))]
        assert all(len(g)==2 and self.y[g].sum()==1 for g in groups)
        groups=np.array(groups)
        rng=np.random.default_rng(0)
        sampled=np.array([rng.choice(len(groups),len(groups),replace=True) for _ in range(2000)])
        self.boot=groups[sampled].reshape(2000,-1)
        self.meta={'json':str(self.path),'npz_sha256':hashlib.sha256(self.path.with_suffix('.npz').read_bytes()).hexdigest(),
                   'n_conditions':len(self.names),'tune_pairs':len(set(self.pairs[self.tune])),
                   'test_pairs':len(groups),'max_json_metric_error':self.error,'disagreements':self.disagreements,
                   'tune_test_pair_overlap':len(set(self.pairs[self.tune])&set(self.pairs[self.test]))}
    def arr(self,n):return self.z[n+'__answer'].astype(bool),self.z[n+'__ptrue'].astype(float)
    def pick(self,kind='E',direction=None):
        mask=self.tune;y=self.y[mask];npos=int(y.sum());nneg=int((~y).sum());N=len(y)
        ba,bp=self.arr('baseline');bpred=bp[mask]>.5
        bapos=int(bpred[y].sum());baneg=int(bpred[~y].sum());baN=int(ba[mask].sum());bpN=int(bpred.sum())
        candidates=[]
        for n in self.names:
            if not n.startswith(kind+'_'):continue
            if direction and '_'+direction+'_' not in n:continue
            a,p=self.arr(n);pred=p[mask]>.5
            jnum=(int(pred[y].sum())-bapos)*nneg-(int(pred[~y].sum())-baneg)*npos
            da=int(a[mask].sum())-baN;db=int(pred.sum())-bpN
            if kind=='E':
                if abs(da)<=.03*N+1e-9:candidates.append((n,jnum))
            else:
                sign=-1 if direction=='abst' else 1
                if abs(jnum)<=.02*npos*nneg+1e-9 and abs(db)<=.05*N+1e-9 and sign*da>0:candidates.append((n,sign*da))
        return (max(candidates,key=lambda r:r[1])[0] if candidates else None),len(candidates)
    def values(self,n):
        if n not in self.boot_cache:
            a,p=self.arr(n)
            self.boot_cache[n]=metric(a[self.boot],p[self.boot],self.y[self.boot])
        a,p=self.arr(n)
        point=metric(a[self.test],p[self.test],self.y[self.test])[0]
        return point,self.boot_cache[n]
    def delta(self,n):
        p,b=self.values(n);p0,b0=self.values('baseline');return p-p0,b-b0
    def selected(self,kind='E',direction=None):
        ch,nok=self.pick(kind,direction)
        key=kind if direction is None else kind+'_'+direction
        return {'chosen':ch,'feasible':nok,'recorded':(self.j.get('selection',{}).get(key) or {}).get('chosen'),
                'metrics':summary(*self.delta(ch)) if ch else None}

def baseline_check(a,b):
    keys=['ids','labels','half_a','baseline__answer','baseline__ptrue']
    return {k:bool(np.array_equal(a.z[k],b.z[k])) for k in keys}

def compare(a,b):
    chk=baseline_check(a,b);assert all(chk[k] for k in ('ids','labels','half_a'))
    ca,_=a.pick();cb,_=b.pick();pa,ba=a.delta(ca);pb,bb=b.delta(cb)
    return {'baseline_identity':chk,'chosen_a':ca,'chosen_b':cb,'paired_difference':summary(pa-pb,ba-bb),
            'a':summary(pa,ba),'b':summary(pb,bb),'b_meta':b.meta}

def whole_scan(e,c):
    def d(sw,n,k):return sw.raw[n]['B'][k]-sw.raw['baseline']['B'][k]
    en=[n for n in e.names if re.fullmatch(r'E_L\d+_a[\d.]+',n)]
    ea=np.array([d(e,n,'ans') for n in en]);ej=np.array([d(e,n,'J') for n in en]);eu=np.array([d(e,n,'AUROC') for n in en])
    ans={'E':{'n':len(en),'ans_within_0.03':int((abs(ea)<=.03).sum()),'dJ_pos':int((ej>0).sum()),
              'dAUROC_pos':int((eu>0).sum()),'max_abs_dans':float(abs(ea).max())}}
    for kind in ['Cp','Cs']:
        names=[n for n in c.names if n.startswith(kind+'_L')]
        a=np.array([d(c,n,'ans') for n in names]);j=np.array([d(c,n,'J') for n in names]);u=np.array([d(c,n,'AUROC') for n in names]);mv=abs(a)>=.10
        ans[kind]={'n':len(names),'movers_10pts':int(mv.sum()),'J_within_0.02':int((mv&(abs(j)<=.02)).sum()),
                   'J_within_0.04':int((mv&(abs(j)<=.04)).sum()),'AUROC_within_0.02':int((mv&(abs(u)<=.02)).sum()),
                   'dJ_movers_median':float(np.median(j[mv])),'both_J_AUROC_within_0.02':int((mv&(abs(j)<=.02)&(abs(u)<=.02)).sum())}
    return ans

def main():
    output={'commit':'244b597535fe15b7734210006a4bd277c95f1f2c','n_boot':2000,'seed':0,'implementation':'NumPy exact-count selections; scipy.stats.rankdata independent AUC','models':{}}
    for m in MODELS:
        p=SEARCH/'tfce'/m/'E23';e=Sweep(p/'knob_sweep_das_tfce_full.json');c=Sweep(p/'knob_sweep_das_tfce_C.json')
        rr={'E_metadata':e.meta,'C_metadata':c.meta,'E_C_baselines':baseline_check(e,c),'baseline':e.raw['baseline']['B'],
            'selected':{'E':e.selected()},'whole_scan':whole_scan(e,c),'comparisons':{}}
        for kind in ['Cp','Cs']:
            for direction in ['abst','ans']:rr['selected'][kind+'_'+direction]=c.selected(kind,direction)
        for tag in ['das_tf_full','meandiff_full']:
            op=OLD/m/('knob_sweep_'+tag+'.json')
            rr['comparisons'][tag]=compare(e,Sweep(op)) if op.exists() and op.with_suffix('.npz').exists() else {'status':'raw NPZ absent; paired CI not independently recomputed'}
        if m=='llama3.1-8b':
            one=Sweep(SEARCH/'tfce1'/m/'E23/knob_sweep_das_tfce1_full.json')
            rr['comparisons']['oneway_minus_bidir']=compare(one,e)
            rr['comparisons']['oneway_minus_kl']=compare(one,Sweep(OLD/m/'knob_sweep_das_tf_full.json'))
        output['models'][m]=rr
        (OUT/'recomputed.json').write_text(json.dumps(output,indent=2))
        print(m,'done',rr['selected']['E']['chosen'],rr['E_metadata']['max_json_metric_error'],rr['C_metadata']['max_json_metric_error'],flush=True)

if __name__=='__main__':main()
