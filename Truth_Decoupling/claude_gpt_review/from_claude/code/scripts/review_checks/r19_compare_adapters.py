#!/usr/bin/env python3
"""Read-only tensor comparison: differences exit 1, invalid inputs 2.

Use --report-only only for an explicitly descriptive comparison that expects
nonidentical tensors. Restricted checkpoint loading never executes pickle code.
"""
import argparse,sys
from pathlib import Path
import torch
from r19_validation import checkpoint


def flat(value,path=()):
    if isinstance(value,torch.Tensor):
        if value.layout!=torch.strided:raise ValueError('only dense checkpoint tensors are supported')
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite checkpoint tensor at '+repr(path))
        yield path,value
    elif isinstance(value,dict):
        for k,v in value.items():yield from flat(v,path+(('key',type(k).__name__,repr(k)),))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):yield from flat(v,path+((type(value).__name__,i),))


def tensors(path):
    out={}
    for key,value in flat(checkpoint(path)):
        if key in out:raise ValueError('tensor traversal path collision')
        out[key]=value
    if not out:raise ValueError('checkpoint contains no tensors: '+str(path))
    return out


def max_difference(a,b):
    if not a.is_floating_point() and not a.is_complex():
        return max(abs(int(x)-int(y)) for x,y in zip(a.reshape(-1).tolist(),b.reshape(-1).tolist()))
    dtype=torch.complex128 if a.is_complex() else torch.float64
    return float((a.to(dtype)-b.to(dtype)).abs().max())


def compare(early_root,release_root):
    differences=0
    for model in ('qwen3-8b','llama3.1-8b'):
        for early,release in (('das','das_s42'),('md','md_s42'),('random','random_s42')):
            a=tensors(early_root/model/'ce_alpha3_ep3'/early/'adapter.pt');b=tensors(release_root/model/'ce_alpha3_ep3'/release/'adapter.pt')
            same=0;diffs=[]
            for key in sorted(set(a)|set(b),key=repr):
                if key not in a or key not in b:diffs.append((key,'missing in '+('early' if key not in a else 'release')))
                elif a[key].shape!=b[key].shape or a[key].dtype!=b[key].dtype:
                    diffs.append((key,'shape/dtype differ'))
                elif torch.equal(a[key],b[key]):same+=1
                else:diffs.append((key,'max |diff| %.3e'%max_difference(a[key],b[key])))
            differences+=len(diffs)
            print('%-12s %-7s vs %-11s tensors %d, bit-identical %d%s'%(model,early,release,len(set(a)|set(b)),same,
                '' if not diffs else '; differ: '+'; '.join('%s (%s)'%(repr(k),v) for k,v in diffs[:6])))
    return differences


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('early',type=Path);p.add_argument('release',type=Path);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    try:diff=compare(a.early,a.release)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        print('INVALID INPUT: '+str(exc),file=sys.stderr);return 2
    if diff:print('NONIDENTICAL TENSORS:',diff)
    return 0 if not diff or a.report_only else 1
if __name__=='__main__':sys.exit(main())
