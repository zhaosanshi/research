"""G4a feasibility audit, no new model runs. From Truth_Decoupling root in the scoring container.
Inspect only the two models whose verifier outputs were already explored. Missing scores stay missing.
"""
from pathlib import Path
import collections
import hashlib
import json
import sys
import numpy as np
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import selfjudge_probe as sp
import tq_same_n_decomp as tq
import qaqa_pilot_table as qa


def cache(model, task):
    root = Path(sp.OUT)/model
    parts = ("", "tq_release_", "tq_forced_") if task == "tq" else ("qaqa_", "qq_release_", "qq_forced_")
    found = {}
    for part in parts:
        p = root/(part+"items.json")
        doc = json.loads(p.read_bytes())
        meta = json.loads((root/(part+"probe.json")).read_bytes())
        if doc["model"] != model or meta["model"] != model or meta["items_sha256"] != hashlib.sha256(p.read_bytes()).hexdigest():
            raise ValueError("foreign input or verifier")
        active = [r for r in doc["items"] if r["answered"]]
        with np.load(root/(part+"probe.npz"), allow_pickle=False) as scores:
            if [r["id"] for r in active] != list(scores["ids"]):
                raise ValueError("roster order")
            for i,r in enumerate(active):
                if task == "tq" and r.get("split",r["id"].split("|")[0]) != "test":
                    continue
                statement = sp.sha1(sp.statement(r["question"],r["answer"]))
                if statement != str(scores["statement_sha1"][i]) or sp.sha1(r["answer"]) != str(scores["answer_sha1"][i]):
                    raise ValueError("score for a different answer")
                f = (str(r["id"]).split("|")[-1], statement)
                m = tuple(float(scores[k+"__margin"][i]) for k in ("baseline","E_a3"))
                if not all(np.isfinite(m)):
                    raise ValueError("non-finite margin")
                found.setdefault(f,m)
    return found


def original_answers(model,task):
    if task == "tq":
        display,l,s = tq.A.KEY[model]
        ids,data = tq.load_model(model)
        (b,texts,_,_),error = tq.MP.prompt_split(display,l,1,s)
        if error:
            raise ValueError(error)
        doc=json.loads((Path(sp.OUT)/model/"items.json").read_bytes())
        qq={r["id"]:r["question"] for r in doc["items"] if r["split"]=="test"}
        e=b.cE(s)
        return {q:(qq[q],[(not b.blank[q,a],tq.V[b.st[q,a]],texts.ec[q,a].strip()) for a in ("baseline_x",e)]) for q in ids}
    root=Path(qa.S.NS)
    raw=qa.Raw(model,"qaqafull",sp.QAQA_BANK,("test_full570","test_full570_tuned","test_full570_strong"))
    pilot=set(json.loads(Path(sp.QAQA_PILOT).read_bytes())["ids"])
    items=[json.loads(x) for x in Path(sp.QAQA_BANK).read_text().splitlines()]
    qs={r["id"]:r["question"] for r in items}
    e=raw.e(qa.SET[model][1])
    for a in ("x",e):
        raw.need(a)
    return {q:(qs[q],[(not raw.blank(a,q),raw.cell(a,q),raw.text(a,q).strip()) for a in ("x",e)]) for q in raw.ids if q not in pilot}


def main():
    for task in ("tq","qq"):
        data={m:original_answers(m,task) for m in ("llama3.1-8b","qwen3.5-9b")}
        ids=sorted(set.intersection(*(set(v) for v in data.values())))
        if len(ids) != (523 if task=="tq" else 470):
            raise ValueError("roster mismatch")
        pooled=collections.Counter()
        for m in data:
            scores=cache(m,task)
            out=collections.Counter()
            for q in ids:
                question,(a,e)=data[m][q]
                if not (a[0] and e[0]) or a[2]==e[2]:
                    continue
                out["eligible"]+=1
                xo=a[1]; xe=e[1]
                out["original_right_E_wrong"]+=int(xo==1 and xe==-1)
                out["E_right_original_wrong"]+=int(xe==1 and xo==-1)
                out["perfect_chooser_over_E"]+=max(0,xo-xe)
                lookup=lambda t:scores.get((q,sp.sha1(sp.statement(question,t))))
                mo,me=lookup(a[2]),lookup(e[2])
                if me is None:
                    raise ValueError("missing ordinary E scores")
                if mo is None:
                    out["missing_original"]+=1
                else:
                    out["cached_both"]+=1
                    for j,k in enumerate(("native","written")):
                        out[k+"_over_E_cached"]+=(xo-xe) if mo[j]>me[j] else 0
            pooled.update(out)
            print(task,m,dict(out))
        print(task,"both",dict(pooled))
if __name__ == "__main__":
    main()
