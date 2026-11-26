"""Refit only each frozen selected B layer, using streamed uncompressed .npz features and no GPU or new models."""

import os

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import struct
import zipfile
import json
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, "Iteration_01/diag")
import gate_fit_1126 as F


def feature_file(path):
    # mmap just the chosen layer, rather than expand every model layer into RAM.
    with zipfile.ZipFile(path) as z:
        info = z.getinfo("feats.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("unexpected compressed input")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        if h[:4] != b"PK\x03\x04":
            raise ValueError("bad archive")
        nm, xtra = struct.unpack_from("<HH", h, 26)
        f.seek(nm + xtra, 1)
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, (major, minor))
        if fortran or dtype != np.float32 or len(shape) != 3:
            raise ValueError("unexpected features")
        return np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape)


from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path[:0] = [
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/capability",
]
import gate_capture_1126 as C
import gate_sim_1126 as G
import qaqa_pilot_table as Q


def compare(who, old, new, y):
    oo = old > 0.5
    nn = new > 0.5
    co = oo == y
    cn = nn == y
    print(
        who,
        "old_nright_auc_open",
        int(co.sum()),
        round(roc_auc_score(y, old), 6),
        int(oo.sum()),
        "new_nright_auc_open",
        int(cn.sum()),
        round(roc_auc_score(y, new), 6),
        int(nn.sum()),
        "on->off",
        int((oo & ~nn).sum()),
        "off->on",
        int((~oo & nn).sum()),
        "right->wrong",
        int((co & ~cn).sum()),
        "wrong->right",
        int((~co & cn).sum()),
        "new opens for 235 problematic/235 valid",
        int(nn[y == 1].sum()),
        int(nn[y == 0].sum()),
        flush=True,
    )


def main():
    with threadpool_limits(limits=1):
        for mi, m in enumerate(F.MODELS):
            capfile = F.OUT / m / "capture_gate_1126.json"
            fit = json.loads((F.OUT / m / "fit_B_gate_1126.json").read_bytes())
            assert (
                fit["model"] == m
                and fit["pool"] == "B"
                and F.sha_file(capfile) == fit["capture_sha256"]
            )
            layer = fit["chosen_layer"]
            assert layer == fit["refit"]["chosen"]["layer"]
            pilot = F.pilot()
            meta = F.load_meta(m)
            rows = F.pool_rows(meta, "B", pilot)
            left = [r for r in rows if r[0] != "qaqa"]
            assert (
                len(rows) == 4546
                and len(left) == 4446
                and len(
                    {
                        meta["qaqa"]["keys"][ix]
                        for name, ix, l, grp in rows
                        if name == "qaqa"
                    }
                    & set(pilot)
                )
                == 100
            )
            F.check_disjoint(meta, left, pilot)
            sel = F.select(meta, "qaqa", "held", pilot)
            ids = [meta["qaqa"]["keys"][i] for i in sel]
            y = np.array([meta["qaqa"]["labels"][i] for i in sel])
            assert (
                len(ids) == len(set(ids)) == 470
                and not set(ids) & set(pilot)
                and np.sum(y) == 235
            )
            probs = fit["probs"]["chosen"]["qaqa:held"]
            assert set(probs) == set(ids)
            old = np.array([probs[q] for q in ids])
            feats = {
                name: feature_file(F.npz(m, name))
                for name in {r[0] for r in rows} | {"qaqa"}
            }
            train = np.stack(
                [feats[name][idx, layer] for name, idx, _, _ in left]
            ).astype(np.float64)
            orig_y = np.array([r[2] for r in left])
            clf = F.make().fit(train, orig_y)
            new = clf.predict_proba(feats["qaqa"][sel, layer].astype(np.float64))[:, 1]
            assert np.all(np.isfinite(new))
            compare(m + " layer " + str(layer), old, new, y)
            _, strength, _, c1, c2, _ = Q.SET[m]
            qa = Q.Raw(m, "qaqafull", G.QAQA_BANK, ["test_full570"])
            o = qa.arm_system("x")[0]
            n1 = qa.n1_system(strength, c1, c2)[0]
            vo = np.array([o[q] for q in ids])
            vn = np.array([n1[q] for q in ids])
            assert (sum(vo), sum(vn)) == (G.K_QQ[m][0], G.K_QQ[m][2])
            print(
                m,
                "net at natural coverage old gate,new gate,always N1",
                int(np.where(old > 0.5, vn, vo).sum()),
                int(np.where(new > 0.5, vn, vo).sum()),
                int(vn.sum()),
                flush=True,
            )
            if mi == 0:
                sets = C.items_of(m, json.load(open(C.FROZEN_MODELS))["models"][m])
                readtext = [sets["qaqa"]["items"][i][1] for i in sel]
                pri = []
                for rr in [rows, left]:
                    assert all(
                        meta[n]["keys"][i] == sets[n]["items"][i][0]
                        and meta[n]["text"][i] == C.tsha(sets[n]["items"][i][1])
                        for n, i, _, _ in rr
                    )
                    model = make_pipeline(
                        TfidfVectorizer(
                            ngram_range=(1, 2), min_df=2, sublinear_tf=True
                        ),
                        LogisticRegression(C=1.0, solver="liblinear", random_state=0),
                    )
                    model.fit(
                        [sets[n]["items"][i][1] for n, i, _, _ in rr],
                        [r[2] for r in rr],
                    )
                    pri.append(model.predict_proba(readtext)[:, 1])
                compare("surface lexical model", pri[0], pri[1], y)


if __name__ == "__main__":
    main()
