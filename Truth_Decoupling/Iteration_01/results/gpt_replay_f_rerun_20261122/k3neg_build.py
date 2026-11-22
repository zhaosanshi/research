"""Build fake Truth_Decoupling roots (symlink mirrors) for the K3 negative tests of k3_checks.patch.
Real caches are only read; every modified file is a fresh copy inside the fake root.
usage: python3 k3neg_build.py OUTDIR LINKPREFIX
  OUTDIR     where the fake roots are made (host path)
  LINKPREFIX the real root as seen by the process that will run (host /data/... or container /workspace/...)
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

REAL = "/data/sanshizhao/Truth_Decoupling"
OUT, PREFIX = sys.argv[1], sys.argv[2]
M = "llama3.1-8b"
SJ = "Iteration_01/newsets/selfjudge"
SETS = ("qaqa", "tq_test", "popqa_test", "popqa_test2", "triviaqa_test", "triviaqa_test2")


def sha(p):
    h = hashlib.sha256()
    if os.path.islink(p) and os.readlink(p).startswith(PREFIX + "/"):
        p = REAL + os.readlink(p)[len(PREFIX):]
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def payload_digest(doc):
    fields = {k: v for k, v in doc.items() if k != "payload_sha256"}
    raw = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def mirror(rel, keep_real):
    """fake/rel is a real dir; its entries are symlinks to PREFIX/rel/entry, except keep_real (made by caller)."""
    os.makedirs(os.path.join(root, rel), exist_ok=True)
    for e in sorted(os.listdir(os.path.join(REAL, rel))):
        if e in keep_real:
            continue
        os.symlink(os.path.join(PREFIX, rel, e), os.path.join(root, rel, e))


def build(name, copies):
    global root
    root = os.path.join(OUT, name)
    if os.path.exists(root):
        shutil.rmtree(root)
    mirror("", {"Iteration_01"})
    mirror("Iteration_01", {"newsets"})
    mirror("Iteration_01/newsets", {"selfjudge"})
    mirror(SJ, {M})
    mirror(os.path.join(SJ, M), set(copies))
    for c in copies:
        shutil.copy2(os.path.join(REAL, SJ, M, c), os.path.join(root, SJ, M, c))
    return os.path.join(root, SJ, M)


def new_format(d, fit_mut=None, drop_fit=False):
    """Turn the copied legacy capture/fit into the patch's new format (npz digests, payload digest)."""
    cap_p, fit_p = os.path.join(d, "k3_capture.json"), os.path.join(d, "k3_fit.json")
    cap = json.load(open(cap_p))
    for n in SETS:
        cap["sets"][n]["npz_sha256"] = sha(os.path.join(d, "k3_%s.npz" % n))
    json.dump(cap, open(cap_p, "w"), ensure_ascii=False, indent=1)
    if drop_fit:
        os.remove(fit_p)
        return
    doc = json.load(open(fit_p))
    doc["npz_sha256"] = {n: sha(os.path.join(d, "k3_%s.npz" % n)) for n in SETS}
    if fit_mut:
        fit_mut(doc)
    doc["training_inputs"] = [sha(os.path.join(REAL, "Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl")),
                              sha(os.path.join(REAL, "Iteration_01/newsets/splits/qaqa_pilot_20261122.json"))]
    doc["capture_sha256"] = sha(cap_p)
    doc["payload_sha256"] = payload_digest(doc)
    json.dump(doc, open(fit_p, "w"), indent=0)


report = []
# control_new: new-format files, nothing corrupted (must be accepted, same table)
d = build("control_new", ["k3_capture.json", "k3_fit.json"])
new_format(d)
report.append("control_new: new-format capture/fit, unmodified data")

# neg_text: same id, different question text (legacy files untouched; the K1 list copy altered)
d = build("neg_text", ["k1_items.json"])
p = os.path.join(d, "k1_items.json")
k1 = json.load(open(p))
i = next(j for j, x in enumerate(k1["items"]) if x["key"].startswith("qaqa|"))
k1["items"][i]["question"] = k1["items"][i]["question"].rstrip() + " (altered copy)"
json.dump(k1, open(p, "w"), ensure_ascii=False)
report.append("neg_text: k1_items.json copy, key %s question text changed" % k1["items"][i]["key"])

# neg_dup: a development (tune) row duplicated in the qaqa capture (key, question hash and features of one tune row
# copied over another tune row), new-format capture digest written for it; for the table a new-format fit, for fit none
pilot = set(json.load(open(os.path.join(REAL, "Iteration_01/newsets/splits/qaqa_pilot_20261122.json")))["ids"])
for name, drop in (("neg_dup_table", False), ("neg_dup_fit", True)):
    d = build(name, ["k3_capture.json", "k3_fit.json", "k3_qaqa.npz"])
    z = dict(np.load(os.path.join(d, "k3_qaqa.npz"), allow_pickle=False))
    keys = [str(k) for k in z["keys"]]
    tune = [j for j, k in enumerate(keys) if k.split("|", 1)[1] in pilot]
    a, b = tune[0], tune[1]
    for f in ("keys", "question_sha1", "resid", "argmax", "first_token"):
        z[f] = z[f].copy()
        z[f][b] = z[f][a]
    os.remove(os.path.join(d, "k3_qaqa.npz"))
    np.savez(os.path.join(d, "k3_qaqa.npz"), **z)
    new_format(d, drop_fit=drop)
    report.append("%s: k3_qaqa.npz copy, tune row %d (%s) overwritten by tune row %d (%s); unique keys %d of %d; fit %s"
                  % (name, b, keys[b], a, keys[a], len(set(z["keys"].tolist())), len(keys),
                     "removed (fit step)" if drop else "new-format, digests of the copy"))


# neg_prob: one probability of the selected layer corrupted, in a new-format fit with a valid payload digest
def bump(doc):
    l = str(doc["selected_layer"])
    k = sorted(doc["probs"][l])[0]
    doc["probs"][l][k] = 1.5
    report.append("neg_prob: k3_fit.json copy, probs[layer %s][%s] = 1.5 (payload digest recomputed)" % (l, k))


d = build("neg_prob", ["k3_capture.json", "k3_fit.json"])
new_format(d, fit_mut=bump)


# neg_prob_legacy: legacy fit, one probability set to NaN (no digest recomputed)
d = build("neg_prob_legacy", ["k3_fit.json"])
p = os.path.join(d, "k3_fit.json")
doc = json.load(open(p))
l = str(doc["selected_layer"])
k = sorted(doc["probs"][l])[0]
doc["probs"][l][k] = float("nan")
json.dump(doc, open(p, "w"), indent=0)
report.append("neg_prob_legacy: legacy k3_fit.json copy, probs[layer %s][%s] = NaN" % (l, k))
print("\n".join(report))
