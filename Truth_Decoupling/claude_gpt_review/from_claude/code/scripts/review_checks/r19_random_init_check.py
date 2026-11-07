#!/usr/bin/env python3
"""Verify the four archived random initializations; non-match exits 1, bad input 2.

Future checker only: no training, no grading, no artifact mutation. Restricted
weights-only loading is mandatory; unsafe pickle fallback is never attempted.
"""
import argparse,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from r19_validation import checkpoint,layer_vectors,cosine


def check(root):
    mismatches=0
    for model in ('llama3.1-8b','qwen3-8b'):
        for run,alpha in (('ce_alpha1_ep3',1.),('ce_alpha3_ep3',3.)):
            rp=root/model/run/'random/adapter.pt';dp=root/model/run/'das/adapter.pt'
            rnd,das=checkpoint(rp),checkpoint(dp)
            for value,path in ((rnd,rp),(das,dp)):
                if value.get('model',model)!=model or float(value.get('alpha',alpha))!=alpha:
                    raise ValueError('model/alpha differs from requested artifact: '+str(path))
            initial=layer_vectors(rnd,'initial_W',rp);trained=layer_vectors(rnd,'W',rp);dinitial=layer_vectors(das,'initial_W',dp)
            if set(initial)!=set(trained) or set(initial)!=set(dinitial):raise ValueError('initial/trained/DAS layer sets differ')
            for layer,init in sorted(initial.items()):
                g=torch.Generator(device='cpu').manual_seed(42+layer)
                expected=F.normalize(torch.randn(init.numel(),generator=g,dtype=torch.float32),dim=0)
                exact=init.dtype==expected.dtype and torch.equal(init,expected)
                mismatches+=not exact
                diff=float((init.double()-expected.double()).abs().max())
                print('%-12s %-14s L%-3d alpha %-4s exact %-5s max|diff| %.1e | cos(init,DAS) %+.5f | cos(init,trained) %+.4f'%
                      (model,run,layer,alpha,exact,diff,cosine(init,dinitial[layer]),cosine(init,trained[layer])))
    return 1 if mismatches else 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    try:return check(a.root)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print('INVALID INPUT: '+str(exc),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
