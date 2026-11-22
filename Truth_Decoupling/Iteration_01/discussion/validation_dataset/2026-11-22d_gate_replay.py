"""Offline replay of queue_e 1g with archived ordinary E, N1, no-write and forced-E answers.
Run from the data root in the scoring container, next to the two imported review scripts.
"""
import runpy
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,"Iteration_01/diag")
import policy_select2 as ps
import newsets_score as ns

base = Path(__file__).parent
credit = runpy.run_path(str(base/"2026-11-22_blank_credit_replay_c.py"))
selfcheck = runpy.run_path(str(base/"2026-11-22d_fair_selfcheck.py"))
for label, read in (("tq",credit["read_tqa"]),("qq",credit["read_qa2"])):
    banks = read()
    for num in (2,3,5):
        names=list(credit["MODELS"][:num])
        common=set.intersection(*(set(banks[m]) for m in names))
        ids=sorted(common) if label=="tq" else [q for q in banks[names[0]] if q in common]
        for gate in (("K2",) if label=="tq" else ("K2","K0")):
            ds=[]; nsent=[]; restored=[]
            for m in names:
                rr=banks[m]
                mask={q for q in ids if not rr[q][1] and (not rr[q][7] if gate=="K2" else rr[q][8]!="no_failure")}
                released={q for q in mask if rr[q][3]}
                eligible=[q for q in ids if rr[q][5] and rr[q][6] is not None]
                ranking=sorted(eligible,key=lambda q:-rr[q][6])
                k=len(released); n=k+sum(rr[q][1] for q in ids); old=set(ranking[:n]); ranked=[q for q in ranking if q in mask]
                if len(ranked)<k or len(old)!=n: raise ValueError((m,gate,k,"unreachable"))
                fill=set(ranked[:k])
                delta=[]
                for q in ids:
                    e,_,v,_,f,*_=rr[q]
                    g=v if q in released else e
                    delta.append((g-(f if q in old else 0),g-(f if q in fill else e)))
                ds.append(delta); nsent.append(n);restored.append(k)
            print(label,gate,num,"common",len(ids),"issued",sum(nsent),"releases",restored)
            for j,k in enumerate(("G-B","G-Bg")):
                vec=[[x[j] for x in m] for m in ds]
                print(" ",k,ps.ci(np.sum(vec,axis=0).tolist()) if label=="tq" else ns.joint_ci(vec))
# Exactly the same E+Z prefix is available to the gated method and its comparator.
# Isolate what this shared check actually does on normally premised questions.
per=[]
for m in credit["MODELS"][:2]:
    ids,_,ordinary,_,_=selfcheck["sources"](m,"qq")
    deltas=[-ordinary[q]["net"] if ordinary[q]["qtype"]=="no_failure" and ordinary[q]["answered"] and not selfcheck["passes"](ordinary[q],0) else 0 for q in ids]
    drops=sum(ordinary[q]["qtype"]=="no_failure" and ordinary[q]["answered"] and not selfcheck["passes"](ordinary[q],0) for q in ids)
    print("qq normal",m, "drops",drops,"S1-E",sum(deltas))
    per.append(deltas)
print("qq normal two-model S1-E (all 470 clusters)",ns.joint_ci(per))

# Among normally premised questions, inspect C as a competing candidate on the
# already released pool, before interpreting the gate's forced switch to E alone.
pair = runpy.run_path(str(base/"2026-11-22d_candidate_pair.py"))
masks = []
for m in credit["MODELS"][:2]:
    root=Path(selfcheck["sp"].OUT)/m
    c=pair["rows_for"](root,"qq_release_","QA2")[1]
    f=pair["rows_for"](root,"qq_forced_","QA2")[1]
    doc=selfcheck["load"](root,m,"qaqa")
    ids=sorted(q for (_,q),r in doc.items() if not r["pilot"] and r["qtype"]=="no_failure")
    masks.append(pair["evaluate"](ids,c,f,None))
rows=np.sum(masks,axis=0)
print("qq normal C-released same-question candidate pairs",sum(rows[:,0]),"C/forced/native-choice/written-choice",rows[:,[2,3,4,5]].sum(0).tolist(),
      "written-native",pair["interval"](rows[:,5]-rows[:,4]),"written-C",pair["interval"](rows[:,5]-rows[:,2]))
# A different role for C on ordinary questions: supply one counterproposal only when
# the first E answer was actually submitted but the shared Z check rejected it.
# Candidate availability only. No claim that the best available text can be recognized.
from collections import Counter
all_retries=Counter()
for m in credit["MODELS"][:2]:
    ids,_,ordinary,_,_=selfcheck["sources"](m,"qq")
    Q=selfcheck["qa"]
    raw=Q.Raw(m,"qaqafull",selfcheck["sp"].QAQA_BANK,("test_full570","test_full570_tuned","test_full570_strong"))
    s=Q.SET[m][1]; initial=raw.e(s); c=raw.ec(s,"0.25"); forced="conf:"+initial.split(":",1)[1]
    for arm in (initial,c,forced):
        raw.need(arm)
    counts=Counter()
    for q in ids:
        r=ordinary[q]
        if r["qtype"]!="no_failure" or not r["answered"] or selfcheck["passes"](r,0):
            continue
        counts["old_checked"]+=1
        counts["old_right"]+=int(r["net"]==1)
        good={}
        for a,arm in (("C",c),("F",forced)):
            fresh=not raw.blank(arm,q) and Q.ro.norm(raw.text(arm,q))!=Q.ro.norm(r["answer"])
            good[a]=fresh and raw.cell(arm,q)==1
            counts[a+"_new"]+=int(fresh)
            counts[a+"_correct"]+=int(good[a])
            counts[a+"_wrong"]+=int(fresh and raw.cell(arm,q)==-1)
        counts["exclusively_C_correct"]+=int(good["C"] and not good["F"])
        counts["old_wrong_and_exclusively_C_correct"]+=int(r["net"]==-1 and good["C"] and not good["F"])
        counts["exclusively_F_correct"]+=int(good["F"] and not good["C"])
    print("qq normal already answered but rejected, candidate availability",m,dict(counts))
    all_retries.update(counts)
print("qq normal candidate availability, two models",dict(all_retries))
