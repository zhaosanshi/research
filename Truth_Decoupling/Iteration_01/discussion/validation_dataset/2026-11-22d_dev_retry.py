"""CPU-only, from Truth_Decoupling on the data host. Read already judged Llama tune answers.
The two audited stronger passes precede this review. No test result, re-verification or fit.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, "Iteration_01/diag")
import numpy as np
import selfjudge_probe as sp
import tq_same_n_decomp as tq

model = "llama3.1-8b"
root = Path(sp.OUT)/model
doc = json.loads((root/"items.json").read_bytes())
if sp.sha256_file(root/"items.json") != json.loads((root/"probe.json").read_bytes())["items_sha256"]:
    raise ValueError("foreign self-check inputs")
active = [i for i in doc["items"] if i["answered"]]
with np.load(root/"probe.npz", allow_pickle=False) as z:
    if list(z["answer_sha1"]) != [sp.sha1(x["answer"]) for x in active] or list(z["ids"]) != [x["id"] for x in active]:
        raise ValueError("self-check cache identity")
    tune = [(float(v),r) for v,r in zip(z["baseline__margin"],active) if r["split"]=="tune" and not r["neg"]]
threshold = sp.choose_thresholds([({"baseline":m}, sp.V[r["state"]]) for m,r in tune], ("baseline",))["baseline"][0]
name, layer, _ = tq.A.KEY[model]
_, run, _ = tq.MP.usual(name,0)
for rule, t in (("P",threshold),("Z",0)):
    flagged = [r for m,r in tune if m < t]
    for dose, work in (("4","pospromptl16s4"), ("5","pospromptl16")):
        c = f"E_L{layer}_x-w-p{dose}"
        path = tq.MP.work(run,"tune",work)
        states, blank = tq.ps.load(path,"pro")
        texts = tq.MP.fr.raw_texts(path)
        generation = json.loads((Path(run)/"eval_extra/gen"/f"x--E{layer}-w-p{dose}-at-prompt_tune.json").read_bytes())
        if generation["positions"] != "prompt" or generation["split"] != "tune" or not generation["completed"]:
            raise ValueError("wrong stronger-E generation")
        native = {r["pair_id"]: r["text"].strip() for r in generation["rows"]}
        cells = []
        for row in flagged:
            q = row["id"]
            if texts[q,c].strip() != native[q] or states[q,c]=="?":
                raise ValueError("text or verdict")
            if not blank[q,c] and tq.MP.ro.norm(texts[q,c]) != tq.MP.ro.norm(row["answer"]):
                cells.append((sp.V[states[q,c]], sp.V[row["state"]]))
        print(rule, dose, "flagged", len(flagged), "eligible changed answers", len(cells),
              "right/wrong/other", [sum(x==v for x,y in cells) for v in (1,-1,0)],
              "wrong-to-right", sum(x==1 and y==-1 for x,y in cells),
              "right-to-wrong", sum(x==-1 and y==1 for x,y in cells))
