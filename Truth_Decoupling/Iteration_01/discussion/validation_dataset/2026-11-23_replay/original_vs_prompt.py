"""Offline read of the original combination versus the ordinary honesty prompt, no new scores.
Run from Truth_Decoupling on the scoring host. Reuse the batch-one verifier audit from the same directory.
"""
from pathlib import Path
import random
import runpy
import numpy as np


def compare(api):
    models = api["MODELS"]
    data = {m: api["load"](m) for m in models}
    qs = sorted(set.intersection(*(set(x) for x in data.values())))
    if len(qs) != 520:
        raise ValueError("wrong test roster")
    rng = random.Random(20261116)
    weights = np.array([np.bincount([rng.randrange(len(qs)) for _ in qs], minlength=len(qs)) for _ in range(4000)])
    for harmonize in (False, True):
        deltas = []
        plain_nets = []
        n1_nets = []
        counts = []
        natural = np.zeros(4, dtype=int)
        for model in models:
            rows = [data[model][q] for q in qs]
            candidates = [[r[a] for r in rows] for a in (0,1)]
            actual = [sum(v[1] for v in side) for side in candidates]
            k = min(actual)
            scores = []
            for j, side in enumerate(candidates):
                net = np.array([v[0] for v in side], dtype=int)
                if harmonize and j == 1:
                    for i,(p,n,_) in enumerate(rows):
                        if p[1] and n[1] and p[2]==n[2]:
                            net[i]=p[0]
                order = sorted((i for i,v in enumerate(side) if v[1]), key=lambda i: (side[i][3]["baseline"],i))
                omit = order[:len(order)-k]
                if len(order)-len(omit)!=k:
                    raise ValueError("answer count mismatch")
                natural[2*j] += int(sum(net))
                natural[2*j+1] += actual[j]
                net[omit]=0
                scores.append(net)
            counts.append(k)
            plain_nets.append(int(sum(scores[0])))
            n1_nets.append(int(sum(scores[1])))
            deltas.append(scores[1]-scores[0])
        if not harmonize and natural.tolist()!=[706,2269,729,2398]:
            raise ValueError("natural totals do not match the published table")
        v=np.sum(deltas, axis=0)
        b=np.sort(weights@v)
        print("identical issued answers share the prompt verdict",harmonize)
        print("matched per model",counts,"total",sum(counts))
        print("prompt",plain_nets,sum(plain_nets),"original E+C",n1_nets,sum(n1_nets))
        print("paired difference",int(v.sum()),"interval",int(b[100]),int(b[3899]),"per model",[int(s.sum()) for s in deltas])


if __name__=="__main__":
    compare(runpy.run_path(str(Path(__file__).with_name("arbitration.py"))))
