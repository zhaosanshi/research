"""CPU audit of applying the same E to two prompts. Run from Truth_Decoupling in the scoring container.
All four cells must use the same question. Use only models with matching E layer and strength.
"""
import collections
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import ns_prompt_table as fact
import qaqa_pilot_table as qa
import n1_on_sentence as stack
import selfjudge_probe as judge

ROOT = Path(qa.S.NS)
QA_BANK = str(ROOT / "bank/qaqa_full_bank_20261122.jsonl")
PILOT = set(json.loads((ROOT/"splits/qaqa_pilot_20261122.json").read_bytes())["ids"])


def qa_rows(model, s):
    base = qa.Raw(model, "qaqafull", QA_BANK, ("test_full570","test_full570_tuned","test_full570_strong"))
    prompted = qa.Raw(model, "qaqafull", QA_BANK, ("test_full570_h22n","test_full570_h22n1","test_full570_h22z"))
    prompted.gdir = ROOT/"gen_prompt/h22/qaqafull"/model/"test"
    spec = base.e(s).split(":",1)[1]
    arms = ((base,"x"),(base,"x:"+spec),(prompted,"x_E2"),(prompted,"x_E2:"+spec))
    for loader,arm in arms:
        loader.need(arm)
    parts=[]
    for loader,arm in (arms[1],arms[3]):
        g=json.loads((loader.gdir/(qa.G.arm_token(arm)+".json")).read_bytes())
        part,=g["parts"] if "parts" in g else g["contract"]["parts"]
        parts.append(part)
    for key in ("family","layer","kind","amount","direction_file_sha256"):
        if key not in parts[0] or parts[0][key] != parts[1].get(key):
            raise ValueError(("differing QA2 write",key,model))
    if base.ids != prompted.ids or len(base.ids) != 570 or len(PILOT) != 100 or not PILOT.issubset(base.ids):
        raise ValueError("QA2 roster mismatch")
    return {q:(base.qtype[q],[(loader.cell(arm,q),not loader.blank(arm,q),loader.text(arm,q).strip())
                              for loader,arm in arms]) for q in base.ids if q not in PILOT}


def fact_rows(model, task, split, s):
    bank,path = fact.bank(task,split)
    el = qa.SET[model][0]
    spec = f"E{el}.w.p{s}"
    arms = (("","x"),("","x:"+spec),("h22","x_E2"),("h22","x_E2:"+spec))
    out = {q:("fact",[]) for q in bank}
    parts = []
    for parent,arm in arms:
        p=ROOT/"gen_prompt"/parent/task/model/split/(fact.G.arm_token(arm)+".json")
        g=fact.G.load_generation(p,SimpleNamespace(task=task,split=split,model=model,items=path),list(bank))
        contract = g["contract"]
        if contract["arm"]!=arm or contract.get("positions") != ("prompt" if ":" in arm else None):
            raise ValueError(str(p))
        if ":" in arm:
            part,=g["parts"] if "parts" in g else contract["parts"]
            parts.append(part)
        for row in g["rows"]:
            q=row["id"]; text=row["text"].strip(); issued=not row["rule_abstain"] and bool(text)
            verdict=fact.AM.verdict(text,bank[q]["gold"],issued)
            out[q][1].append((int(verdict=="R")-int(verdict=="W"),issued,text))
    for key in ("family","layer","kind","amount","direction_file_sha256"):
        if not parts[0].get(key) == parts[1].get(key) or key not in parts[0]:
            raise ValueError(("differing write",key,model,task,split))
    return out


def summarize(name, per_model):
    ids=sorted(set.intersection(*(set(x) for x in per_model.values())))
    for label in (("no_failure","failure","unverifiable") if name=="qa2" else ("fact",)):
        qs=[q for q in ids if next(iter(per_model.values()))[q][0]==label]
        totals=[]; recoveries=collections.Counter()
        for model, rows in per_model.items():
            net=[]
            for q in qs:
                kind,cells=rows[q]
                if kind!=label:
                    raise ValueError("type mismatch")
                x,e,p,s = cells
                diff=(s[0]-p[0])-(e[0]-x[0])
                net.append([x[0],e[0],p[0],s[0],diff,diff if all(c[1] for c in cells) else 0,
                            int(all(c[1] for c in cells))])
                if x[1] and x[0]==1:
                    recoveries["original_correct"]+=1
                    if p[1] and p[0]==1:
                        recoveries["both_original_and_prompt_correct"]+=1
                    else:
                        recoveries["prompt_disagrees_with_original_correctness"]+=1
                if p[1] and p[0]==1 and s[1] and s[0]==-1:
                    recoveries["P_right_to_PE_wrong"]+=1
                    if x[1] and x[0]==1:
                        recoveries["covered_by_x_right"]+=1
                    if judge.NEG_RE.match(s[2]):
                        recoveries["PE_wrong_negation_opening"]+=1
            sums=np.sum(net,axis=0)
            print(name,label,model,"cells x E P PE, interaction",sums[:5].tolist())
            totals.append(net)
        sums=np.sum(totals,axis=0)
        rng=random.Random(20261119)
        w=np.array([np.bincount([rng.randrange(len(qs)) for _ in qs],minlength=len(qs)) for _ in range(4000)])
        sims=np.sort(w @ sums[:,4:6],axis=0)
        print(name,label,len(qs),"models",len(totals),"pooled",sums.sum(axis=0).tolist(),
              "intervals (all, all-four-issued)",[(int(a),int(b)) for a,b in zip(sims[100],sims[3899])],
              dict(recoveries),flush=True)


if __name__=="__main__":
    models={}
    for m,(_,s,*_) in qa.SET.items():
        selected=json.load(open(stack.CHOICE % m,encoding="utf-8"))["strength_dot"]
        if s == selected:
            models[m]=s
    print("Matched strengths",models)
    summarize("qa2",{m:qa_rows(m,s) for m,s in models.items()})
    for task in ("triviaqa","popqa"):
        for split in ("test","test2"):
            summarize(task+" "+split,{m:fact_rows(m,task,split,s) for m,s in models.items()})
