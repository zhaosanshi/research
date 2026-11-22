"""Verify comparator D on the 470 held-out questions. CPU only, from the scoring container root."""
import hashlib
import json
import math
from pathlib import Path
import runpy

audit = runpy.run_path(str(Path(__file__).with_name("2026-11-22d_fair_selfcheck.py")))
qa, sp = audit["qa"], audit["sp"]
both = []
for model in audit["MODELS"]:
    ids, _, ordinary, _, _ = audit["sources"](model, "qq")
    r = qa.Raw(model, "qaqafull", sp.QAQA_BANK, ("test_full570", "test_full570_tuned", "test_full570_strong"))
    arm = r.e(qa.SET[model][1])
    token = qa.G.arm_token(arm)
    data = json.loads((Path(sp.OUT)/model/(token+"-at-prompt.json")).read_bytes())
    if (data["model"], data["split"], data["task"], data["arm"]) != (model, "test", "qaqafull", arm):
        raise ValueError("arm identity")
    if not data["first_difference_check"] or [x["id"] for x in data["rows"]] != r.ids:
        raise ValueError("first-difference check or bank order")
    for key, name in (("generation_sha256",token),("reference_sha256","x")):
        if data[key] != hashlib.sha256((r.gdir/(name+".json")).read_bytes()).hexdigest():
            raise ValueError("wrong source")
    conf = {x["id"]: x for x in data["rows"]}
    for q in ids:
        if conf[q]["text_sha1"] != sp.sha1(ordinary[q]["answer"]):
            raise ValueError("wrong answer")
    active = [q for q in ids if ordinary[q]["answered"]]
    if any(not math.isfinite(conf[q]["mean_logp"]) or conf[q]["mean_logp"] > 0 for q in active):
        raise ValueError("invalid confidence")
    drops = {q for q in active if not audit["passes"](ordinary[q], 0.)}
    diffs = []
    for allow_neg in (True,False):
        pool = [q for q in active if allow_neg or not ordinary[q]["neg"]]
        chosen = set(sorted(pool,key=lambda q: conf[q]["mean_logp"])[:len(drops)])
        if len(chosen) != len(drops):
            raise ValueError("different count")
        diffs.append([ordinary[q]["net"]*(int(q in chosen)-int(q in drops)) for q in ids])
    if both and ids != roster:
        raise ValueError("roster mismatch")
    roster = ids
    both.append(diffs)
    print(model, "issued",len(active)-len(drops), "S1-D all, then other", [sum(d) for d in diffs],
          "scoring batch",data["scoring_batch_size"],"reproduction",data["greedy_reproduction"])
for i in range(2):
    print("joint", "all" if i==0 else "other", qa.S.joint_ci([v[i] for v in both]))
