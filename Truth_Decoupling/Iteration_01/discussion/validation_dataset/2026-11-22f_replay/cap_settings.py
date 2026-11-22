"""Queue G, G1/G2. Run in the scoring container from the Truth_Decoupling root.
Read-only; output is aggregate only. Each (question, model) retains its original
submission flag alongside its independently judged score. No new inference.
"""
from itertools import product
import json
from pathlib import Path
import random
import sys
from typing import NamedTuple

import numpy as np

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import qaqa_pilot_table as qa
import tq_same_n_decomp as tq

MODELS = tq.ORDER
DOSES = qa.LAD
CAPS = tuple(product(range(5), repeat=2))
PROTOCOL = {"TQA": (20261116, 520, 521), "QA2": (20261119, 470, 470)}


class Candidate(NamedTuple):
    score: int
    submitted: bool
    repeats_original: bool


class Item(NamedTuple):
    e: int
    answered: bool
    own_blank: bool
    candidates: tuple
    forced: tuple


def load_tqa(model):
    name, layer, strength = tq.A.KEY[model]
    source, = [entry for entry in tq.ML.sources(name)[2] if entry[0] == layer]
    splits = []
    for i, main_roster in enumerate((source[1], source[3])):
        split, error = tq.MP.prompt_split(name, layer, i, strength)
        if not split:
            raise ValueError(error)
        sp, text, _, all_ids = split
        allowed = set(main_roster.qs)
        ids = [q for q in all_ids if q in allowed]
        if ids != all_ids:
            split, err = tq.MP.prompt_split(name, layer, i, strength, qs=ids)
            if not split or split[3] != ids:
                raise ValueError(err or "wrong overlap")
            sp, text = split[:2]
        _, run, _ = tq.MP.usual(name, i)
        forced = tq.forced(run, layer, strength, ids) if i else {}
        forced_key = f"E_L{layer}_conf-w-p{strength}"
        forced_text = tq.MP.fr.raw_texts(tq.MP.work(run, "test", f"pospromptdl{layer}")) if i else {}
        e = sp.cE(strength)
        arms = [sp.cEC(strength, k.replace(".", "-")) for k in DOSES]
        rows = {}
        for q in ids:
            original = tq.MP.ro.norm(text.ec[q, "baseline_x"])
            candidates = tuple(Candidate(tq.V[sp.st[q, arm]], not sp.blank[q, arm],
                                         tq.MP.ro.norm(text.ec[q, arm]) == original) for arm in arms)
            f = (*forced[q], tq.MP.ro.norm(forced_text[q, forced_key]) == original) if i else None
            rows[q] = Item(tq.V[sp.st[q, e]], not sp.blank[q, e], sp.blank[q, "baseline_x"], candidates, f)
        splits.append(rows)
    return splits


def load_qa2(model):
    root = Path(qa.S.NS)
    split_file = json.loads((root / "splits/qaqa_pilot_20261122.json").read_bytes())
    if qa.G.sha_file(qa.TUNE_BANK) != split_file["bank_sha256"]:
        raise ValueError("foreign tune bank")
    pilot = set(split_file["ids"])
    sources = (("qaqa", qa.TUNE_BANK, ("test_tune100",)),
               ("qaqafull", str(root / "bank/qaqa_full_bank_20261122.jsonl"),
                ("test_full570", "test_full570_tuned", "test_full570_strong")))
    splits = []
    for i, (task, bank, works) in enumerate(sources):
        raw = qa.Raw(model, task, bank, works)
        strength = qa.SET[model][1]
        e = raw.e(strength)
        arms = [raw.ec(strength, k) for k in DOSES]
        f = "conf:" + e.split(":", 1)[1]
        for arm in ("x", e, *arms):
            raw.need(arm)
        if i:
            raw.need(f)
            conf = raw.confidence(task, f)
        ids = [q for q in raw.ids if not i or q not in pilot]
        if (not i and set(ids) != pilot) or len(ids) != (470 if i else 100):
            raise ValueError("foreign test/tune roster")
        rows = {}
        for q in ids:
            original = qa.ro.norm(raw.text("x", q))
            candidates = tuple(Candidate(raw.cell(arm,q), not raw.blank(arm,q), qa.ro.norm(raw.text(arm,q)) == original)
                               for arm in arms)
            forced = (raw.cell(f,q), not raw.blank(f,q), conf[q], qa.ro.norm(raw.text(f,q)) == original) if i else None
            rows[q] = Item(raw.cell(e,q), not raw.blank(e,q), raw.blank("x",q), candidates, forced)
        splits.append(rows)
    return splits


def submit(row, caps):
    if not row.answered:
        for candidate in row.candidates[:caps[int(row.own_blank)]]:
            if candidate.submitted:
                if row.own_blank or not candidate.repeats_original:
                    return candidate.score, True
                break
    return row.e, row.answered


def weights(n, seed):
    rng = random.Random(seed)
    return np.asarray([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n) for _ in range(4000)])


def bounds(draws):
    return tuple(round(float(x), 2) for x in np.sort(draws)[[100,3899]])


def evaluate(rows, scores, answered, twopass=False):
    n = len(rows)
    ranked = sorted((i for i,row in enumerate(rows) if row.forced[1] and row.forced[2] is not None),
                    key=lambda i: -rows[i].forced[2])
    if twopass:
        if any(row.answered and not ans for row,ans in zip(rows,answered)):
            raise ValueError("N1 drops original E answer")
        ranked = [i for i in ranked if not rows[i].answered]
        target = sum(answered) - sum(r.answered for r in rows)
        base = np.asarray([r.e for r in rows])
    else:
        target = sum(answered)
        base = np.zeros(n,int)
    if target > len(ranked):
        return None
    for i in ranked[:target]:
        base[i] = rows[i].forced[0]
    return np.asarray(scores) - base


def strict_source(rows, w, cap, own):
    """Both sides use the original-answer veto and exactly the same quota per source.
    When forced E cannot reach N1's target, omit exactly the deficit from N1.
    Return extreme observed and resampled deltas across all admissible omissions.
    """
    ids = [i for i,r in enumerate(rows) if not r.answered and r.own_blank == own]
    released = [i for i in ids if submit(rows[i],(cap,cap))[1]]
    ranked = sorted([i for i in ids if rows[i].forced[1] and rows[i].forced[2] is not None
                     and (own or not rows[i].forced[3])], key=lambda i: -rows[i].forced[2])
    k = min(len(released), len(ranked))
    deficit = len(released) - k
    gains = np.asarray([submit(rows[i], (cap,cap))[0]-rows[i].e for i in released], dtype=int)
    delta = np.zeros(len(rows),int)
    delta[released] = gains
    for i in ranked[:k]:
        delta[i] -= rows[i].forced[0]-rows[i].e
    baseline = w @ delta
    if not deficit:
        return baseline, baseline, 0, k
    omitted = np.sort(w[:,released]*gains,axis=1)
    return baseline-omitted[:,-deficit:].sum(axis=1), baseline-omitted[:,:deficit].sum(axis=1), deficit, k


def report(task, data, count):
    models = MODELS[:count]
    common = set.intersection(*(set(data[m][1]) for m in models))
    ids = sorted(common) if task == "TQA" else [q for q in data[models[0]][1] if q in common]
    seed,n5,n3 = PROTOCOL[task]
    if len(ids) != (n3 if count==3 else n5):
        raise ValueError("wrong common cohort")
    w = weights(len(ids),seed)
    w0 = np.vstack((np.ones(len(ids),int),w))
    one = {}
    cells = {}
    strict = {}
    for m in models:
        tune,test = data[m]
        rows = [test[q] for q in ids]
        nets = [sum(r.candidates[j].score for r in tune.values()) for j in range(4)]
        best = int(np.argmax(nets))
        for j in range(4):
            scores=[r.candidates[j].score for r in rows]
            answered=[r.candidates[j].submitted for r in rows]
            candidate=(sum(answered),evaluate(rows,scores,answered))
            one[m,j]=candidate
            if j==best:
                one[m,"selected"]=candidate
        for caps in CAPS[1:]:
            scores,submitted=zip(*(submit(r,caps) for r in rows))
            cells[m,caps]=(sum(submitted),tuple(zip(scores,submitted)),
                           evaluate(rows,scores,submitted), evaluate(rows,scores,submitted,True))
        print(task, m, "chosen dose", DOSES[best], "development nets",nets,
              "unique counts / score and submission masks among 24",
              len({cells[m,p][0] for p in CAPS[1:]}),len({cells[m,p][1] for p in CAPS[1:]}))
        for own in (False,True):
            for cap in range(1,5):
                strict[m,own,cap]=strict_source(rows,w0,cap,own)
    print(task,count,len(ids),"common questions")
    for j in ("selected",0,1,2,3):
        ns=[one[m,j][0] for m in models]
        vals=[one[m,j][1] for m in models]
        if any(v is None for v in vals):
            print("G1",j,"infeasible for",[m for m,v in zip(models,vals) if v is None])
        else:
            combined=sum(vals)
            print("G1",j,"answers",sum(ns),"delta",combined.sum(),bounds(w@combined))
    for col,lab in ((2,"one-pass"),(3,"two-pass")):
        vectors=[]
        for p in CAPS[1:]:
            if any(cells[m,p][col] is None for m in models):
                print("G2",lab,"24-cell statistic not reachable",p)
                break
            vectors.append(sum(cells[m,p][col] for m in models))
        else:
            mean=np.mean(vectors,axis=0)
            cell_sums=np.asarray(vectors).sum(axis=1)
            print("G2",lab,"mean",round(mean.sum(),2),bounds(w@mean),"positive / zero / negative",
                  sum(cell_sums>0),sum(cell_sums==0),sum(cell_sums<0),"min/max",cell_sums.min(),cell_sums.max())
            if lab=="two-pass":
                print("  per policy (row cap for E-induced blanks, col cap for native blanks)",
                      cell_sums.tolist())
    losses=[]; low=np.zeros(len(w0)); high=np.zeros(len(w0))
    for p in CAPS[1:]:
        part=[strict[m,own,cap] for m in models for own,cap in zip((False,True),p) if cap]
        losses.append(sum(entry[2] for entry in part))
        low+=sum(entry[0] for entry in part)
        high+=sum(entry[1] for entry in part)
    low/=24; high/=24
    print("G2 with matched veto and source quotas: policies meeting original N",losses.count(0),
          "mean/max count reduction",round(sum(losses)/24,3),max(losses),
          "nearest matched count net bounds",round(low[0],2),round(high[0],2),
          "paired outer range",bounds(low[1:])[0],bounds(high[1:])[1],flush=True)


if __name__=="__main__":
    for task,loader in (("TQA",load_tqa),("QA2",load_qa2)):
        data={m:loader(m) for m in MODELS}
        for count in (3,5):
            report(task,data,count)
