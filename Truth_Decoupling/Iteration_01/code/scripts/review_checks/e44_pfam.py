# -*- coding: utf-8 -*-
"""Evaluation-only entry (Claude Opus 5.5, 2026-11-22, queue I section 7.6): no-write prompt controls whose system text
uses a sentence the run's prompt configuration does not have (H, the TruthfulQA PROF_PRIMER instruction).
The model's own engine (e44_eval_extra.py) and e44_positions2.py are imported from the working directory's code copy and
run unchanged, with ONE in-memory change: extra_systems receives a deep copy of the run's prompt configuration with the
sentences of the prompt file's "sentences_added" added (names that already exist are refused), so the run's own
configuration and its identity hash are untouched. Then e44_positions2.py takes over (test-split widening, generate /
export). The system texts of the prompt file are printed once.
Run from the root of the model's code copy:
  python <this file> prompt generate|export --split tune|test <engine arguments> --prompts-extra <pfam file> --bases ...
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
CWD = Path(os.getcwd()).resolve()
for p in (CWD / "scripts", CWD / "scripts" / "review_checks"):
    sys.path.insert(0, str(p))
import e44_eval_extra as Y        # noqa: E402
import e44_positions2 as P2       # noqa: E402

if Path(Y.__file__).resolve().parent != CWD / "scripts" or Path(P2.__file__).resolve().parent != CWD / "scripts" / "review_checks":
    raise SystemExit("imported the engine or e44_positions2.py from outside the working directory's code copy")
pf = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--prompts-extra"), "")
if not pf.endswith("e44_eval_extra_pfam_v1.json"):
    raise SystemExit("pass --prompts-extra .../e44_eval_extra_pfam_v1.json")
ADDED = json.loads(Path(pf).read_text(encoding="utf-8"))["sentences_added"]
_original = Y.extra_systems
_shown = []


def extra_systems(meta, cfg, systems, args):
    cfg2 = copy.deepcopy(cfg)
    if any(k in cfg2["sentences"] for k in ADDED):
        raise SystemExit("an added sentence name already exists in the run's configuration")
    cfg2["sentences"].update(ADDED)
    out = _original(meta, cfg2, systems, args)
    if not _shown:
        for k, v in out.items():
            print("[pfam] system text of %s: %s" % (k, v), flush=True)
        _shown.append(1)
    return out


Y.extra_systems = extra_systems
print("[pfam] sentences added in memory: %s; engine sha256 %s; this entry sha256 %s" % (
    ", ".join(ADDED), hashlib.sha256(Path(Y.__file__).read_bytes()).hexdigest()[:16],
    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), flush=True)

if __name__ == "__main__":
    P2.main()
