"""Single counterfactual: keep gate-on behavior fixed, replace only OFF base answer by OFF single E."""

import hashlib
import json
import math
import sys
import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import gate_sim_1126 as g
import falseqa_read as f
import iti_table as a
import qaqa_pilot_table as qa
import ns_prompt_table as nprompt
import newsets_score as ns
from collections import defaultdict


def tab(model, task, ids, orig, steered, issues_o, issues_e, expected, groups=None):
    assert (
        len(ids) == len(set(ids))
        and (sum(orig.values()), sum(steered.values())) == expected
    )
    d = g.OUT / model
    s = json.loads((d / "fit_B_gate_1126.json").read_bytes())
    assert (
        s["model"] == model
        and s["pool"] == "B"
        and s["chosen_layer"] == s["refit"]["chosen"]["layer"]
        and s["capture_sha256"]
        == hashlib.sha256((d / "capture_gate_1126.json").read_bytes()).hexdigest()
    )
    prob = s["probs"]["chosen"][task]
    assert all(q in prob and math.isfinite(prob[q]) and 0 <= prob[q] <= 1 for q in ids)
    off = [q for q in ids if prob[q] <= 0.5]
    delta = {q: (steered[q] - orig[q] if prob[q] <= 0.5 else 0) for q in ids}
    issue = sum(int(issues_e[q]) - int(issues_o[q]) for q in off)
    bins = defaultdict(int)
    for q in ids:
        bins[groups[q] if groups else q] += delta[q]
    vals = np.array(list(bins.values()))
    draws = np.random.default_rng(20261126).integers(
        0, len(vals), size=(5000, len(vals))
    )
    l, h = np.quantile(vals[draws].sum(1), [0.025, 0.975], method="inverted_cdf")
    print(
        model,
        task,
        len(ids),
        len(off),
        "off E minus O",
        sum(delta.values()),
        int(l),
        int(h),
        "delta answered",
        issue,
        flush=True,
    )


ok, bank = f.check_bank()
assert ok
bygrp = {r["id"]: r["pair_id"] for r in bank}
dev = {r["id"] for r in bank if r["set"] == "dev"}
held = {r["id"] for r in bank if r["set"] == "heldout"}
pilot = {json.loads(ll)["id"] for ll in open(g.QAQA_PILOT) if ll.strip()}
models = ns.models()
for m in g.ORDER:
    st = f.setting(m)
    tr, tr2 = f.load_rows(m, dev)
    caps, _ = f.choose(tr, tr2, st)
    r, r2 = f.load_rows(m, held)
    ids = r.ids
    ss = f.systems_of(r, r2, st, caps, ids)
    maps = [
        {q: ss[name][q][index] for q in ids} for index in (0, 1) for name in ("O", "E")
    ]
    tab(m, "fqa_held", ids, *maps, g.K_FQA[m][:2], bygrp)
    display, layer, s = a.KEY[m]
    tt, err = a.MP.prompt_split(display, layer, 1, s)
    common = set(next(x[3] for x in a.ML.sources(display)[2] if x[0] == layer).qs)
    ids = [q for q in tt[3] if q in common]
    if ids != tt[3]:
        tt, err = a.MP.prompt_split(display, layer, 1, s, qs=ids)
    sp, tx, _, ids = tt
    maps = []
    iss = []
    for arm in [("base",), ("E", s)]:
        states = a.S.system(sp, tx, arm)[0]
        maps.append(
            dict(zip(ids, [{"R": 1, "W": -1, "X": 0, "H": 0}[x] for x in states]))
        )
        iss.append(dict(zip(ids, [x in ("R", "W") for x in states])))
    tab(m, "tq_test", ids, *maps, *iss, g.K_TQ[m][1:3])
    rr = qa.Raw(m, "qaqafull", g.QAQA_BANK, ["test_full570"])
    ids = [q for q in rr.ids if q not in pilot]
    o = rr.arm_system("x")
    e = rr.arm_system(rr.e(g.STRENGTH[m][0]))
    maps = [{q: v[q] for q in ids} for v in [o[0], e[0], o[1], e[1]]]
    tab(m, "qaqa:held", ids, *maps, g.K_QQ[m][:2])
    for task, split in g.FACT:
        sp, tx = nprompt.load(task, m, models[m], split)[:2]
        ids = [q.split(":", 1)[1] for q in sp.qs]
        maps = []
        iss = []
        for arm in [("base",), ("E", nprompt.dash(g.STRENGTH[m][0]))]:
            states, ans = ns.system(sp, tx, arm)[:2]
            maps.append(dict(zip(ids, ns.vals(states))))
            iss.append(dict(zip(ids, ans)))
        tab(m, task + "_" + split, ids, *maps, *iss, g.K_FACT[task, split][m])
