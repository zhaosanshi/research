"""Compare ordinary E, the ordinary E+C policy N1 and the no-write honesty sentence S0.
CPU only. Uses the same singleton no-write self-verifier and exactly the same per-model quota for all three.
"""
import sys
sys.path[:0]=["Iteration_01/diag","Iteration_01/code/scripts/newsets","Iteration_01/code/scripts/review_checks"]
import fill_only as F


def keep(rows, ids, name, n):
    cells=[rows[q][name] for q in ids]
    available=sorted((i for i,x in enumerate(cells) if x[1]),key=lambda i:(cells[i][2],i))
    if len(available)<n:
        raise ValueError("quota exceeds issued answers")
    dropped=set(available[:len(available)-n])
    return [0 if i in dropped else x[0] for i,x in enumerate(cells)]


def main():
    models=F.ORDER
    margins=F.margins()
    data={}
    for m in models:
        ids,items,info=F.SKP.qaqa(m)["qaqa"]
        held=sorted(q for q in ids if not info[q][1])
        if len(held)!=470 or len(set(held))!=len(held):
            raise ValueError("held-out split")
        if data and list(next(iter(data.values())))!=held:
            raise ValueError("model rosters differ")
        data[m]={q:F.SKP.systems(items[q],"baseline",margins[m]) for q in held}
    for group in (models[:3],models):
        for names in (("S0","N1"),("S0","E","N1"),("S0z","Ez","N1z")):
            counts=[]; gaps={}
            totals=[0 for _ in names]
            for m in group:
                per=data[m]
                k=min(sum(per[q][key][1] for q in held) for key in names)
                vals=[keep(per,held,key,k) for key in names]
                counts.append(k)
                for j,v in enumerate(vals):
                    totals[j]+=sum(v)
                    for i in range(j):
                        key=(names[j], names[i])
                        byq=[a-b for a,b in zip(v,vals[i])]
                        gaps[key]=[x+y for x,y in zip(gaps.get(key,[0]*len(held)),byq)]
            print("models",list(group),"arms",names,"quota",counts,"total",sum(counts),"scores",totals)
            for key,d in gaps.items(): print(" ",key,F.ci_j(d))
if __name__=="__main__":
    main()
