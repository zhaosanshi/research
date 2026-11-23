"""CPU-only replay in Truth_Decoupling scoring container. No inference or judge calls."""
from pathlib import Path
import collections
import hashlib
import json
import random
import sys
import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import cap_score as scoring
import sentence_capability as table

ROOT = Path("Iteration_01")
CAP = ROOT / "capability"
MODELS = table.ORDER
LAYERS = {m: table.N.A.KEY[m][1] for m in MODELS}
NET = {"R": 1, "W": -1, "X": 0}


def read(p):
    return json.loads(p.read_bytes())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(ok, message):
    if not ok:
        raise ValueError(message)


def bootstrap(v, seed=20261119):
    v = np.asarray(v)
    unique, freq = np.unique(v, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(len(v), freq/len(v), size=4000) @ unique
    return tuple(round(float(x), 2) for x in (sum(v), np.sort(draws)[100], np.sort(draws)[3899]))


def main():
    gfile = CAP / "sets/gsm8k_1319_20261119.json"
    ids = read(gfile)["ids"]
    gold = scoring.gold_of("gsm8k")
    mfile = CAP / "sets/mmlu_14042_20261120.json"
    mids = read(mfile)["ids"]
    seen = set(read(CAP / "sets/mmlu_1000_20261119.json")["ids"])
    assert len(set(ids)) == len(ids) == 1319 and len(set(mids)) == len(mids) == 14042 and seen <= set(mids) and len(seen)==1000
    keepidx = [i for i,k in enumerate(mids) if k not in seen]
    gm_gold_sha = digest(Path(scoring.DATA["gsm8k"]))
    mm_gold_sha = digest(Path(scoring.DATA["mmlu"]))
    mm_gold = scoring.gold_of("mmlu")
    sums = collections.Counter()
    capability = collections.defaultdict(list)
    for m in MODELS:
        n1 = read(ROOT / "newsets/h22" / f"n1s_choice_{m}.json")
        s = n1["strength_dot"]
        cfg = n1["cfg"]
        own_cap = cfg[3] if cfg[0] == "N1" else None
        news = read(ROOT / "newsets/h22" / f"n1s3_choice_{m}.json")["chosen"]
        assert float(news) in (float(s), float(s)/2)
        layer = LAYERS[m]
        gg = {}
        for label, arm in (("base","neutral"),("P","neutral_E2"),("PE",f"neutral_E2:E{layer}.w.p{s}"),
                           ("half",f"neutral_E2:E{layer}.w.p{float(s)/2:g}")):
            fp = CAP / "gen_main" / ("gsm8k" if label == "base" else "gsm8k_h22") / m / "b8"
            token = arm.replace(":","--").replace(".","-") + ("-at-prompt" if ":" in arm else "")
            obj = read(fp/(token+".json"))
            ct = obj["contract"]; rows = obj["rows"]
            validate(obj["completed"] and ct["arm"]==arm and ct["model"]==m and ct["task"]=="gsm8k" and
                     ct["max_new_tokens"]==512 and ct["batch_size"]==8 and ct["full_pass"] and
                     ct["items_file_sha256"]==gm_gold_sha and ct["ids_file_sha256"]==digest(gfile) and
                     ct["ids"]==ids and [x["id"] for x in rows]==ids,(m,arm,"generation identity or order"))
            gg[label]=rows
        st = {k:[scoring.judge_row("gsm8k",x,gold(x["id"])) for x in rows] for k,rows in gg.items()}
        def blank(x):
            return bool(x["rule_abstain"]) or not x["text"].strip()
        P = gg["P"]; E=gg["PE"]
        ix = [i for i,row in enumerate(P) if blank(row)]
        invalid = [i for i,row in enumerate(P) if st["P"][i][0]=="X" and not blank(row)]
        diff = collections.Counter()
        net_gain = right_gain = unknown = 0
        for i in ix:
            state, why = st["P"][i]
            assert state == "X"
            candidate, _ = st["PE"][i]
            if blank(E[i]) and own_cap is not None:
                unknown += 1
                diff["PE blank; C output unavailable"]+=1
            else:
                net_gain += NET[candidate]
                right_gain += int(candidate=="R")
                diff["final "+candidate]+=1
        p_right = sum(s0=="R" for s0, _ in st["P"])
        p_net = sum(NET[s0] for s0, _ in st["P"])
        print("GSM",m,"P net/right",p_net,p_right,"P rule blanks",len(ix), "empty text",sum(not row["text"].strip() for row in P),
              "P all unfinished",sum(not row['finished'] for row in P),"all judge-zero, rule-nonblank",len(invalid),
              "of which truncated", sum(st['P'][i][1]=='truncated' for i in invalid), "own C cap",own_cap,
              "fill on rule blanks",dict(diff),"relative net bounds",(net_gain-unknown,net_gain+unknown),
              "relative right-count bounds",(right_gain,right_gain+unknown))
        sums.update(blank=len(ix),unknown=unknown,known_net=net_gain,known_right=right_gain,P_right=p_right,P_net=p_net,
                    p_issued=len(ids)-len(ix))
        sel = "PE" if float(news)==float(s) else "half"
        for key in ("P","PE","half",sel+" selected"):
            ss=sel if key.endswith(" selected") else key
            ds=[int(x[0]=="R")-int(y[0]=="R") for x,y in zip(st[ss],st["base"])]
            capability["GSM","selected" if key.endswith(" selected") else key].append(ds)
        files = {code: read(CAP / folder / (m+".json")) for code,folder in [('M','full_h22'),('H','full_h22s')]}
        choices={}
        for tag,(src,arm) in dict(base=('M','x'),P=('M','x_E2'),
                                  PE=('M',f'x_E2:E{layer}.w.p{s}'), half=('H',f'x_E2:E{layer}.w.p{float(s)/2:g}')).items():
            d=files[src]; rows=d['arms'][arm]['rows']
            validate(d['model']==m and d['n']==14042 and d['ids_file_sha256']==digest(mfile) and d['items_file_sha256']==mm_gold_sha
                     and [x['id'] for x in rows]==mids and all(x['gold']==mm_gold(x['id']) and x['pick'] in ('A','B','C','D') for x in rows),
                     (m,arm,"MMLU identity, option or order"))
            choices[tag]=[rows[i]['gold']==rows[i]['pick'] for i in keepidx]
        for key in ("P","PE","half",sel+" selected"):
            kk=sel if key.endswith(' selected') else key
            capability["MMLU","selected" if key.endswith(' selected') else key].append([int(y)-int(x) for x,y in zip(choices['base'],choices[kk])])
        print("I8 chosen", m,news,sel,"GSM, MMLU correct counts",sum(x[0]=='R' for x in st[sel]),sum(choices[sel]),flush=True)
    print("GSM pooled",dict(sums),"relative net bounds",(sums['known_net']-sums['unknown'],sums['known_net']+sums['unknown']),
          "relative correct-count bounds",(sums['known_right'],sums['known_right']+sums['unknown']))
    for (task,key),vs in capability.items():
        vv=np.asarray(vs)
        print(task,key,"mean accuracy change pp from no-write",bootstrap(vv.sum(axis=0)*100/vv.size))

if __name__=="__main__":
    main()
