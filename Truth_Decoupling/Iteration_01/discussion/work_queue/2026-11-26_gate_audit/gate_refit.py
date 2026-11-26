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


def main():
    with threadpool_limits(limits=1):
        for m in F.MODELS:
            cap_path = F.OUT / m / "capture_gate_1126.json"
            fit = json.loads((F.OUT / m / "fit_B_gate_1126.json").read_bytes())
            cap = json.loads(cap_path.read_bytes())
            if fit["capture_sha256"] != F.sha_file(cap_path):
                raise ValueError("wrong capture")
            meta = F.load_meta(m)
            pil = F.pilot()
            rows = F.pool_rows(meta, "B", pil)
            F.check_disjoint(meta, rows, pil)
            layer = fit["chosen_layer"]
            if layer != min(
                c["layer"]
                for c in fit["cv"]
                if c["mean"] == max(x["mean"] for x in fit["cv"])
            ):
                raise ValueError("selection")
            feats = {name: feature_file(F.npz(m, name)) for name in F.FAMILY}
            x = np.stack(
                [feats[name][index, layer] for name, index, _, _ in rows]
            ).astype(np.float64)
            yy = np.array([row[2] for row in rows])
            clf = F.make().fit(x, yy)
            dif = []
            flips = 0
            n = 0
            for name, part in F.READ:
                idx = F.select(meta, name, part, pil)
                key = name if part == "all" else name + ":" + part
                keys = [meta[name]["keys"][i] for i in idx]
                scores = fit["probs"]["chosen"][key]
                if set(scores) != set(keys) or len(keys) != len(set(keys)):
                    raise ValueError("roster")
                old = np.array([scores[q] for q in keys])
                value = clf.predict_proba(feats[name][idx, layer].astype(np.float64))[
                    :, 1
                ]
                if not np.isfinite(old).all() or not np.all((old >= 0) & (old <= 1)):
                    raise ValueError("invalid score")
                dif.append(np.max(np.abs(value - old)))
                flips += int(np.sum((value > 0.5) != (old > 0.5)))
                n += len(old)
            d = float(max(dif))
            print(
                m,
                layer,
                len(rows),
                n,
                "max refit probability difference",
                d,
                "routing disagreements",
                flips,
                flush=True,
            )
            if d > 1e-7 or flips:
                raise ValueError("failed re-fit")


if __name__ == "__main__":
    main()
