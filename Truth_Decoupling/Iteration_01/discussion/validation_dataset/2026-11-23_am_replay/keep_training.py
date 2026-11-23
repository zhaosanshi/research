"""Audit K from archived tune judgements and generated facts. CPU only, no judge call.
Run at Truth_Decoupling root in the scoring container. No text is printed."""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import ns_prompt_table as NP
import factqa_nonanswer_diag as diag

B = Path("Iteration_01")
NS = B / "newsets"
M = "llama3.1-8b"
BASE = Path("results_e44_factorial_das/llama3.1-8b_side_E2")
NEW = B / "results/keepfix/llama3.1-8b/e44"
TAGS = ("old","half","w0p03","w0p1")
LAD = ("0-25","0-5","1","2")


def read(p):
    return json.loads(Path(p).read_bytes())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def cells(work):
    gen=read(work/"judge_input.json")
    src={(r["pair_id"],c):text.strip() for r in gen["evaluations"] for c,text in r["outputs"].items()}
    raw={}
    for line in (work/"judge2/cells.jsonl").read_text().splitlines():
        r=json.loads(line)
        key=r["pair_id"],r["condition"]
        assert r["prediction"].strip()==src[key]
        raw[r["cell_id"]]=r
    verdict={}
    for line in (work/"judge2/single_pro.jsonl").read_text().splitlines():
        v=json.loads(line)
        r=raw.get(v["cell_id"])
        if r is None:continue
        assert r['input_sha']==v['input_sha']
        score={"A":1,"B":-1,"C":0,"H":0}[v["label"]] if v["status"]=="scored" else 0
        if v["status"]!="scored": assert v.get("basis")=="empty_output_no_judge_vote"
        verdict[r["pair_id"],r["condition"]]=score, not (r["rule_abstain"] or not r["prediction"].strip()),r["prediction"].strip()
    assert len(verdict)==len(raw)
    return verdict


def candidates(tasks, ids, s, suffix, prefix):
    data={}
    for q in ids:
        key=prefix+q
        e=tasks[key,"E_L16_x_E2-w-p"+s]
        base=tasks[key,"baseline_x_E2"]
        cs=[tasks[key,"X_x_E2--E16-w-p"+s+"-and-C19-w-m"+k+suffix] for k in LAD]
        data[q]=base,e,cs
    return data


def policy(dataset):
    qs=sorted(dataset)
    values=[]
    for c1,c2 in itertools.product(range(5), repeat=2):
        answers=[]
        for q in qs:
            orig,e,cs=dataset[q]
            final=e
            if not e[1]:
                cap=c1 if orig[1] else c2
                for candidate in cs[:cap]:
                    if candidate[1]:
                        if not orig[1] or norm(candidate[2])!=norm(orig[2]):final=candidate
                        break
            answers.append(final)
        values.append((sum(v[0] for v in answers),(c1,c2),answers))
    _,caps,ans=max(values,key=lambda row:row[0])
    return np.array([(v[0],v[1]) for v in ans],dtype=int),caps


def facts(task,tag,s):
    folder="keepfix_main" if tag in ("old","half") else "keepfix_"+tag
    simple=s.replace("-",".")
    gen=read(NS/f"gen_prompt/{folder}/{task}/{M}/tune/x_E2--E16-w-p{s}.json")
    rows={q["id"]:q for q in gen["rows"]}
    bank_ids=read(NS/f"splits/{task}_splits_20261119.json")["tune"]
    bank={q["id"]:q["gold"] for q in (json.loads(l) for l in (NS/f"bank/{task}_bank_20261119.jsonl").read_text().splitlines())}
    assert len(rows)==len(gen["rows"])==len(bank_ids)==192 and set(rows)==set(bank_ids)
    assert gen["contract"]["arm"]==f"x_E2:E16.w.p{simple}" and gen["contract"]["positions"]=="prompt" and gen["completed"]
    part,=gen["contract"]["parts"]
    if tag not in ("old","half"):
        wpath=NEW/f"llama3.1-8b_keepfix_{tag}/dirs/E_L16.pt"
        assert part["direction_file_sha256"]==digest(wpath) and abs(part["amount"]-5.02640218)<0.0001
    values=[]
    suspect=0
    for q in sorted(bank_ids):
        row=rows[q]
        active=bool(row["text"].strip()) and not row["rule_abstain"]
        state=NP.AM.verdict(row["text"],bank[q],active)
        suspect += int(state=="W" and diag.nonanswer(row["text"]))
        values.append((int(state=="R")-int(state=="W"),active))
    print(task,tag,"wrong alias labels matching established no-answer list",suspect)
    return np.array(values,dtype=int)


def collect():
    source=BASE/"eval_extra"
    newbase=NS/"judge/qaqafull"/M
    tq0=cells(source/"judge_tune_h22c4_tune")
    tqmain=cells(source/"judge_tune_n1s_tune")
    qq0=cells(newbase/"test_full570_h22z")
    qqmain=cells(newbase/"test_full570_h22n1")
    qq_main_se={}
    tqids=read(BASE/"frozen_tune.json")["question_ids"]
    qaids=read(NS/"splits/qaqa_pilot_20261122.json")["ids"]
    assert len(set(tqids))==len(tqids)==192 and len(set(qaids))==len(qaids)==100
    out={}
    for tag in TAGS:
        if tag=="old":t,s,q=tqmain,"3-5",{**qqmain,**qq_main_se}
        elif tag=="half":t,s,q=tqmain,"1-75",cells(newbase/"keepfix_main_pilot100")
        else:
            s=(NS/f"keepfix/aligned_s_{M}_{tag}.txt").read_text().strip().replace(".","-")
            t=cells(NEW/f"llama3.1-8b_keepfix_{tag}/eval_extra/judge_tune_kf_tune")
            q=cells(newbase/f"keepfix_{tag}_pilot100")
        t,tc=policy(candidates({**tq0,**t},tqids,s,"-at-prompt",""))
        q,qc=policy(candidates({**qq0,**q},qaids,s,"","qaqafull:"))
        out[tag]=[t,q,facts("triviaqa",tag,s),facts("popqa",tag,s)]
        print(tag,"tq/qa caps as number of rungs",tc,qc,"score/issued/R/W",[(int(z[:,0].sum()),int(z[:,1].sum()),int(sum(z[:,0]==1)),int(sum(z[:,0]==-1))) for z in out[tag]],flush=True)
    return out


def train():
    md=read(NS/f"keepfix/keep_ids_{M}.json")
    print("kept",[ (t,v["keep_both"],v["keep_x_only"]) for t,v in md["tasks"].items()])
    base=torch.load(BASE/"dirs/E_L16.pt",weights_only=False,map_location="cpu")
    for tag in ("l0","l1","l0p03","l0p1","l0p3","w0","w0p03","w0p1"):
        name="llama3.1-8b_keepfix"+("" if tag=="l1" else "_"+tag)
        path=NEW/name/"dirs/E_L16.pt";r=torch.load(path,weights_only=False,map_location="cpu")
        k=r['keep']
        assert k["keep_set"]["file_sha256"]==md["keep_set_sha256"]
        print("direction",tag,"lambda",k['lambda'],"gap",r['gap'],"cos",float(torch.nn.functional.cosine_similarity(base['w'][None],r['w'][None])),"signed keep steps",r["training"]["history"][-1]["keep_sign_negative_steps"],"same side",k["sign_check"]["same_side"], "per-epoch count of negative sign steps",[x["keep_sign_negative_steps"] for x in r["training"]["history"]])


def analyze(rows):
    # Independent resampling streams per dataset, paired across methods. Caps fixed at observed tune choices.
    rng=random.Random(20261123)
    w=[]
    for vals in rows["old"]:
        n=len(vals)
        w.append(np.array([np.bincount([rng.randrange(n) for _ in range(n)],minlength=n) for _ in range(4000)]))
    for tag in TAGS[1:]:
        sims=np.zeros(4000,int)
        delta=0
        common_n=0;common_delta=0
        for i,(base,alt) in enumerate(zip(rows["old"],rows[tag])):
            diff=alt[:,0]-base[:,0];n=len(diff);delta+=int(sum(diff));sims+=w[i]@diff
            sel=(alt[:,1]==1)&(base[:,1]==1)
            d=int(sum(diff[sel])); common_n+=int(sum(sel));common_delta+=d
            print("both issued",tag,i,int(sum(sel)),d,"newly issued",int(sum((alt[:,1]==1)&(base[:,1]==0))),"blocked",int(sum((alt[:,1]==0)&(base[:,1]==1))))
        print("pooled dev",tag,delta,sorted(sims)[100],sorted(sims)[3899],"both issued",common_n,common_delta)
    for ref in ("old","half"):
        candidates=[]
        for tag in TAGS[2:]:
            sims=sum(w[i]@(x[:,0]-y[:,0]) for i,(x,y) in enumerate(zip(rows[tag],rows[ref])))
            candidates.append(sims); mean=sum(int((x[:,0]-y[:,0]).sum()) for x,y in zip(rows[tag],rows[ref]))
            print("four-set net",tag,"minus",ref,mean,"paired interval",np.sort(sims)[[100,3899]].tolist())
            base,alt=rows[ref][3],rows[tag][3]
            print("PopQA transitions",ref,tag,dict(sorted(collections.Counter((int(a),int(b)) for a,b in zip(base[:,0],alt[:,0])).items())))
        best=np.maximum(*candidates)
        print("better of two in each bootstrap sample versus",ref,np.sort(best)[[100,3899]].tolist())
if __name__=="__main__":
    train(); analyze(collect())
