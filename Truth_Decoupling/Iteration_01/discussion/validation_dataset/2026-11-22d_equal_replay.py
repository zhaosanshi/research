"""Read-only from Truth_Decoupling root, in the scoring container. Uses no model or external judge."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import numpy as np

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import tq_same_n_decomp as tq
import qaqa_pilot_table as qq

ROOT = Path("Iteration_01/newsets")
MODELS = ("llama3.1-8b", "qwen3.5-9b")
VALUE = {"R": 1, "W": -1, "H": 0, "X": 0}
ARMS = ("baseline", "E_a3")


def read(p):
    return json.loads(p.read_bytes())


def digest(s):
    return hashlib.sha1(s.strip().encode()).hexdigest()


def check(condition, what):
    if not condition:
        raise ValueError(what)


def load_margins(model, dataset):
    d = ROOT / "selfjudge" / model
    prefix = "qaqa_" if dataset == "qa2" else ""
    p = d / (prefix + "items.json")
    meta = read(d / (prefix + "probe.json"))
    items = read(p)
    check(meta["model"] == items["model"] == model and meta["items_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest(), "item signature")
    rx = re.compile(items["neg_re"], re.I)
    seen = set()
    answered = []
    for r in items["items"]:
        q = r["id"]
        check(q not in seen and digest(r["answer"]) == r["answer_sha1"] and bool(rx.match(r["answer"])) == r["neg"], (dataset, q))
        seen.add(q)
        if r["answered"]:
            answered.append(r)
    with np.load(d / (prefix + "probe.npz")) as z:
        check(list(z["ids"]) == [r["id"] for r in answered] and list(z["answer_sha1"]) == [r["answer_sha1"] for r in answered], "probe row identity")
        if not prefix:
            check(list(z["split"]) == [r["split"] for r in answered], "probe split")
        stmts = ['The answer to the question "{}?" is: {}.'.format(r["question"].strip().rstrip("?").strip(), r["answer"].strip().rstrip(".")) for r in answered]
        check(list(z["statement_sha1"]) == list(map(digest, stmts)), "statement signature")
        margins = {}
        for arm in ARMS:
            m = z[arm + "__lpT"] - z[arm + "__lpF"]
            check(np.array_equal(m, z[arm + "__margin"]) and np.isfinite(m).all(), "raw label probabilities")
            margins[arm] = dict(zip(z["ids"], m))
    return items, margins


def threshold(xs, m):
    vals = sorted(set(m[r["id"]] for r in xs))
    candidates = [-math.inf, *[(a + b) / 2 for a, b in zip(vals, vals[1:])], math.inf]
    def drops(t):
        return [VALUE[r["state"]] for r in xs if m[r["id"]] < t]
    t = max(candidates, key=lambda x: -sum(drops(x)))
    return t, len(drops(t)), -sum(drops(t))


def sources(model, dataset, doc):
    cache = {r["id"]:r for r in doc["items"] if r.get("split", "test") == "test"}
    if dataset == "tqa":
        ids, model_data = tq.load_model(model)
        _, layer, s, _, run, es, ea, n, na, f = model_data
        gen = Path(doc["splits"]["test"]["generation"])
        check(hashlib.sha256(gen.read_bytes()).hexdigest() == doc["splits"]["test"]["generation_sha256"], "ordinary E generation")
        texts = {r["pair_id"]:r["text"] for r in read(gen)["rows"]}
        forced_condition = f"E_L{layer}_conf-w-p{s}"
        forced_texts = tq.MP.fr.raw_texts(tq.MP.work(run,"test",f"pospromptdl{layer}"))
        rec = {q:(es[q], ea[q], n[q], na[q], *f[q], digest(texts[q])==digest(forced_texts[q, forced_condition])) for q in ids}
        check(list(cache) == ids and all((cache[q]["answer_sha1"],VALUE[cache[q]["state"]],cache[q]["answered"]) == (digest(texts[q]),es[q],ea[q]) for q in ids), "native TQA ordinary")
        return rec, set()
    bank = ROOT / "bank/qaqa_full_bank_20261122.jsonl"
    pilot = set(read(ROOT / "splits/qaqa_pilot_20261122.json")["ids"])
    choice = read(ROOT / "splits/qaqa_tune_choice_fixedE_20261122.json")[model]
    work = ["test_full570","test_full570_tuned","test_full570_strong"]
    r = qq.Raw(model, "qaqafull", str(bank), work)
    e = r.e(choice["n1_strength"])
    forced = "conf:" + e.split(":")[1]
    r.need(e)
    r.need(forced)
    n, na, _ = r.n1_system(choice["n1_strength"],choice["n1_cap_e"],choice["n1_cap_own"])
    cf = r.confidence("qaqafull",forced)
    cached = read(ROOT / "selfjudge" / model / "qaqa_equal_n.json")
    check(cached["model"]==model and cached["choice"]==choice and cached["bank_sha256"]==hashlib.sha256(bank.read_bytes()).hexdigest()
          and list(cache)==r.ids==[x["id"] for x in cached["rows"]] and len(set(r.ids))==570 and len(pilot)==100, "native QA2 roster")
    rec = {}
    for x in cached["rows"]:
        q=x["id"]
        cell=(r.cell(e,q),not r.blank(e,q),n[q],na[q],r.cell(forced,q),not r.blank(forced,q),cf[q] if not r.blank(forced,q) else None)
        check(cell==tuple(x[k] for k in ("e","eans","n1","issued","f","fans","conf")), ("native/dump",q))
        check(cache[q]["pilot"]==(q in pilot) and cache[q]["answer_sha1"]==x["e_sha1"]==digest(r.text(e,q)), ("native/probe",q))
        rec[q]=(*cell, digest(r.text(e,q))==digest(r.text(forced,q)))
    return rec, pilot


def differences(ds, data, ids):
    rec, (doc,margins,thr), _ = data
    cache = {r["id"]:r for r in doc["items"]}
    original = {q for q in ids if rec[q][1]}
    other = [q for q in ids if q in original and not cache[q]["neg"]]
    flags = {a:{q for q in other if margins[a][q]<thr[a]} for a in ARMS}
    drop=flags["E_a3"]
    bdrop=set(sorted(other, key=margins["baseline"].get)[:len(drop)])
    forced=sorted([q for q in ids if rec[q][5]], key=lambda q:-rec[q][6])
    kept=original-drop
    n2=sum(rec[q][3] for q in ids)-len(drop)
    top,top2=set(forced[:len(kept)]),set(forced[:n2])
    check(len(top)==len(kept) and len(top2)==n2, "coverage")
    reference=set(forced[:sum(rec[q][3] for q in ids)])
    fill_count=n2-len(original)
    blank_pool=[q for q in forced if q not in original]
    c2=set(blank_pool[:fill_count]) if 0<=fill_count<=len(blank_pool) else None
    both=kept.intersection(forced)
    matched=set(forced[:len(both)])
    restricted=[q for q in other if q in forced]
    reduced=drop.intersection(restricted)
    conf_drop=set(sorted(restricted, key=lambda q:rec[q][6])[:len(reduced)])
    print(ds,"both-available for same-text ranking (origin:forced conf)",len(restricted),"drops",len(reduced),
          "S1-only kept vs A-only",len(drop^bdrop)//2,"S1-B same text and shared-issued",sum(rec[q][7] for q in kept&top),len(kept&top))
    table=[]
    c2_diff=0
    for q in ids:
        e,ei,n,ni,f,fi,confidence,identical=rec[q]
        w=0 if q in drop else e
        a=0 if q in bdrop else e
        b=f if q in top else 0
        w2=0 if q in drop else n
        b2=f if q in top2 else 0
        which=f*(int(q in both)-int(q in matched))
        content=(e-f) if q in both else 0
        if c2 is not None:
            c2_diff+=w2-(f if q in c2 else e)
        table.append([w-a,w-b,w2-b2,
                      -e if q in flags["E_a3"] else 0,-e if q in flags["baseline"] else 0,
                      which,content,w-b-which-content,
                      (int(q in conf_drop)-int(q in reduced))*e,
                      e if not ei else 0,n if not ni else 0,
                      (e-f) if q in both and identical else 0,
                      a-b,n-(f if q in reference else 0)])
    print(ds,"S2-C2", c2_diff if c2 is not None else "unreachable", "S1/A/B",len(kept),
          "S2/B2",n2,"N1",sum(rec[q][3] for q in ids))
    return np.array(table,dtype=int)


def main():
    banks={"tqa":{},"qa2":{}}
    for m in MODELS:
        td,tm=load_margins(m,"tqa")
        train=[r for r in td["items"] if r["split"]=="tune" and r["answered"]]
        for group,sub in (("all",train),("other",[r for r in train if not r["neg"]])):
            pars={a:threshold(sub,tm[a]) for a in ARMS}
            test=[r for r in td["items"] if r["split"]=="test" and r["answered"] and (group=="all" or not r["neg"])]
            print(m,"tqa",group,"(threshold,tune drops/gain,test drops/gain)",{a:(*pars[a],sum(tm[a][r["id"]]<pars[a][0] for r in test),-sum(VALUE[r["state"]] for r in test if tm[a][r["id"]]<pars[a][0])) for a in ARMS})
        th={a:p[0] for a,p in pars.items()}
        for task in banks:
            doc,margins=(td,tm) if task=="tqa" else load_margins(m,"qa2")
            rows,pilot=sources(m,task,doc)
            banks[task][m]=(rows,(doc,margins,th),pilot)
    for task, bank in banks.items():
        roster=set.intersection(*(set(r[0]) for r in bank.values()))
        first=bank[MODELS[0]]
        groups={"test":sorted(roster)} if task=="tqa" else {"470":[q for q in first[0] if q not in first[2]],"570":list(first[0])}
        for name,ids in groups.items():
            totals=[]
            for m,data in bank.items():
                t=differences(f"{task} {name} {m}",data,ids)
                totals.append(t)
                print(task,name,m,"point estimates",t.sum(axis=0).tolist())
            total=sum(totals)
            n=len(ids)
            rng=random.Random(20261116 if task=="tqa" else 20261119)
            weights=np.array([np.bincount([rng.randrange(n) for _ in range(n)],minlength=n) for _ in range(4000)])
            draws=np.sort(weights@total,axis=0)
            print(task,name,n,"two models","totals [95% paired questions]")
            for col,label in enumerate(("S1-A","S1-B","S2-B2","step2 written improvement","step2 baseline improvement",
                                         "S1-B forced text frozen: question selection","S1-B selected eligible: answer text","S1-B boundary",
                                         "E answers frozen: self-judge vs forced-answer confidence on shared eligible others",
                                         "leftover E score on rule blanks","leftover N1 score on rule blanks","same text judge residual in content",
                                         "A-B no-write judge vs forced confidence","reference N1-strong")):
                print(label,int(total[:,col].sum()),[int(draws[100,col]),int(draws[3899,col])])
    print("completed both models and both banks")


if __name__=="__main__":
    main()
