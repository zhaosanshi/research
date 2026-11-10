import json, sys, torch
from pathlib import Path
run=Path(sys.argv[1])
fz=json.loads((run/"frozen_train.json").read_text()); fq={r["pair_id"]:r for r in fz["questions"]}
for L in (5,9,14,16,17,20,24,28):
    r=torch.load(run/"dirs"/("C_L%d.pt"%L), map_location="cpu", weights_only=False)
    pp={x["pair_id"]:x for x in r["per_pair_ce_after"]}
    sw=[i for i in r["pair_ids"] if fq[i]["answers"]["x_caut"]["refusal_rule"] and not fq[i]["answers"]["x_conf"]["refusal_rule"]]
    plus=sorted(pp[i]["ce_plus"] for i in sw); minus=sorted(pp[i]["ce_minus"] for i in sw)
    ntok=[fq[i]["answers"]["x_caut"]["n_tokens"]+1 for i in sw]
    print("C L%d switch pairs n=%d, target tokens (incl. eot) %s" % (L, len(sw), sorted(set(ntok))))
    print("   CE+ (base x_conf -> target refusal): " + " ".join("%.2f"%v for v in plus))
    print("   CE- (base x_caut -> target answer) : " + " ".join("%.2f"%v for v in minus))
