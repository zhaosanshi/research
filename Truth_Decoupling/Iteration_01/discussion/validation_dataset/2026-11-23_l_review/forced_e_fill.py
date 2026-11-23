"""CPU comparison on the same prompt-blank pool with the previously generated forced-answer E.
Run from Truth_Decoupling in the scoring container; no new answer or judge call.
"""
from pathlib import Path
import json
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import fill_only as fill
import tq_same_n_decomp as tq
import qaqa_pilot_table as qa


def forced(model, task, ids):
    display, layer, strength = tq.A.KEY[model]
    if task == "TruthfulQA":
        _, run, _ = tq.MP.usual(display, 1)
        scored = tq.forced(run, layer, strength, ids)
        work = tq.MP.work(run, "test", f"pospromptdl{layer}")
        texts = tq.MP.fr.raw_texts(work)
        tag = f"E_L{layer}_conf-w-p{strength}"
        return {q: (*scored[q], texts[q,tag].strip()) for q in ids}
    root = Path(qa.S.NS)
    r = qa.Raw(model,"qaqafull", str(root/"bank/qaqa_full_bank_20261122.jsonl"),
               ("test_full570","test_full570_tuned","test_full570_strong"))
    arm = "conf:" + r.e(qa.SET[model][1]).split(":")[1]
    r.need(arm)
    conf = r.confidence("qaqafull",arm)
    return {q:(r.cell(arm,q),not r.blank(arm,q),conf[q],r.text(arm,q).strip()) for q in ids}


def first_rung(model, task, ids):
    display, layer, _ = tq.A.KEY[model]
    ch = fill.N2.choice(model)
    if task == "TruthfulQA":
        strength = ch["strength"]
        sp,text = fill.N.build(model,1,"x_E2",strength,fill.N.restricted(model,1))
        arm=sp.cEC(strength,"0-25")
        return {q:(tq.V[sp.st[q,arm]], not sp.blank[q,arm], text.ec[q,arm]) for q in ids}
    root=Path(qa.S.NS)
    bank=str(root/"bank/qaqa_full_bank_20261122.jsonl")
    r=qa.Raw(model,"qaqafull",bank,("test_full570_h22n","test_full570_h22n1","test_full570_h22z"))
    r.gdir=root/"gen_prompt/h22/qaqafull"/model/"test"
    arm="x_E2:"+r.ec(ch["strength_dot"],"0.25").split(":",1)[1]
    r.need(arm)
    return {q:(r.cell(arm,q),not r.blank(arm,q),r.text(arm,q)) for q in ids}


def replay(task, data, qs, group):
    forced_rows = {m:forced(m,task,qs) for m in group}
    first = {m:first_rung(m,task,qs) for m in group}
    ci = fill.ci_tq if task=="TruthfulQA" else fill.ci_j
    print(task, len(qs), "questions", list(group))
    for share_first_e,one_try in ((False,False),(True,False),(True,True)):
        per_model=[]
        for m in group:
            rows=data[m][1]
            rival=forced_rows[m]
            blank=[q for q in qs if not rows[q]["S0"][1] and (not share_first_e or not rows[q]["SE"][1])]
            chosen = first[m] if one_try else {q: rows[q]["SN1"] for q in blank}
            submitted={q for q in blank if chosen[q][1]}
            available=[q for q in blank if rival[q][1] and rival[q][2] is not None]
            if len(available)<len(submitted):
                print(m,"unreachable",len(submitted),len(available))
                break
            kept=set(sorted(available,key=lambda q:-rival[q][2])[:len(submitted)])
            scores={}
            paired={}
            equal_text_conflicts=0
            for q in qs:
                if q not in blank:
                    scores[q]=paired[q]=0
                    continue
                ec=chosen[q]
                f=rival[q]
                # Both algorithms keep the same first-pass evidence. On the fill pool
                # follow fill_only: a native blank keeps its judge value; a trimmed answer gets 0.
                fv=f[0] if (q in kept or not f[1]) else 0
                scores[q]=ec[0]-fv
                both=q in submitted and f[1]
                if both and ec[2].strip()==f[3] and ec[0]!=f[0]:
                    equal_text_conflicts+=1
                paired[q]=ec[0]-f[0] if both and ec[2].strip()!=f[3] else 0
            per_model.append((scores,paired,len(submitted),sum(q in submitted and rival[q][1] for q in blank)))
            print(m,"single C retry",one_try,"shared SE",share_first_e,"pool",len(blank),"equal fills",len(submitted),
                  "delta",sum(scores.values()),"both answer",per_model[-1][3],"duplicate text label conflicts",equal_text_conflicts)
        if len(per_model)==len(group):
            for i,label in ((0,"matched fills using forced E own confidence"),(1,"both answered; identical text one verdict")):
                print(" ",label,"n",sum(x[i+2] for x in per_model),
                      "EC minus forced E",ci([sum(x[i][q] for x in per_model) for q in qs]),flush=True)


def main():
    caps={m:fill.sc_cap(m) for m in fill.ORDER}
    for task in ("TruthfulQA","QA2"):
        if task=="TruthfulQA":
            data={m:fill.tq(m,1,caps[m]) for m in fill.ORDER}
            ids=fill.common_of(data)
        else:
            data={m:fill.qq(m) for m in fill.ORDER}
            ids=[q for q in data[fill.ORDER[0]][0] if not data[fill.ORDER[0]][2][q][1]]
        assert len(ids)==(520 if task=="TruthfulQA" else 470)
        same=[m for m in fill.ORDER if fill.N2.choice(m)["strength_dot"]==qa.SET[m][1]]
        for group in (fill.ORDER,same):
            replay(task,data,ids,group)
if __name__=="__main__":
    main()
