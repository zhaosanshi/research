"""Atomic I/O, seeding, experiment naming and resumable checkpoints.

Conventions inherited from the v12 code base:
  * results are JSON with a top-level `evaluations` list + `summary`;
  * partial results are flushed every N items so long runs can resume;
  * writes are atomic (tmp file + os.replace) so a killed job never leaves
    a half-written JSON behind.
"""
import hashlib
import json
import os
import random
import time
from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def atomic_json_dump(obj, path, indent=2):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    """Make numpy / torch scalars and arrays serialisable."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError("not JSON serialisable: %r" % type(o))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    os.replace(tmp, path)


def append_jsonl(row, path):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def stable_hash(*parts, length=12):
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=_json_default).encode("utf-8"))
    return h.hexdigest()[:length]


def exp_name(prefix, **kv):
    """Deterministic experiment name, e.g. e2_qwen3-8b_L23_r1_T8_e10_s42."""
    parts = [prefix]
    for k, v in kv.items():
        if v is None or v is False:
            continue
        if v is True:
            parts.append(k)
        else:
            parts.append("%s%s" % (k, v))
    return "_".join(str(p) for p in parts)


class ResumableResults:
    """Minimal resumable result store: a JSON file with `evaluations` list.

    Usage:
        rr = ResumableResults(path, header={...})
        for i, item in enumerate(items):
            if rr.done(i): continue
            rr.add(i, record)          # flushes every `flush_every`
        rr.finalize(summary)
    """

    def __init__(self, path, header=None, flush_every=10):
        self.path = path
        self.flush_every = flush_every
        if os.path.exists(path):
            self.data = load_json(path)
            self.data.setdefault("evaluations", [])
        else:
            self.data = dict(header or {})
            self.data.setdefault("started", now_iso())
            self.data["evaluations"] = []
        self._done = {e.get("idx") for e in self.data["evaluations"] if "idx" in e}
        self._since_flush = 0

    def done(self, idx):
        return idx in self._done

    def add(self, idx, record):
        record = dict(record)
        record["idx"] = idx
        self.data["evaluations"].append(record)
        self._done.add(idx)
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self):
        self.data["last_flush"] = now_iso()
        atomic_json_dump(self.data, self.path)
        self._since_flush = 0

    def finalize(self, summary=None):
        if summary is not None:
            self.data["summary"] = summary
        self.data["completed"] = now_iso()
        self.flush()
        return self.data

    @property
    def evaluations(self):
        return self.data["evaluations"]


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self):
        return time.time() - self.t0
