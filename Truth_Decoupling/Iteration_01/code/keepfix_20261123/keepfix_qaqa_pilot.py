# -*- coding: utf-8 -*-
"""Queue K (Claude Opus 5.5, 2026-11-23; discussion/work_queue/2026-11-23_queue_k.md, section 3 item 3): the judge input
of the (QA)^2 100 tune (pilot) questions only. The arms are generated over the whole 570-question bank (the batches of
the existing x_E2 passes, so that a pilot question sees the same neighbours as in the passes it is compared with) and
exported by newsets_gen's own export (through ns_h22.py); this program keeps the evaluation rows of the 100 pilot ids
(newsets/splits/qaqa_pilot_20261122.json, "ids") in the export's order and changes nothing else in them.
usage (CPU, from /workspace/Truth_Decoupling): python -I keepfix_qaqa_pilot.py FULL_EXPORT.json OUT.json
"""
import hashlib
import json
import sys
from pathlib import Path

PILOT = Path("/workspace/Truth_Decoupling/Iteration_01/newsets/splits/qaqa_pilot_20261122.json")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit("%s exists; never overwrite a judge input" % out)
    d = json.loads(src.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))["ids"]
    want = {"qaqafull:%s" % i for i in pilot}
    if d.get("completed") is not True or d.get("task") != "qaqafull" or len(d.get("evaluations", [])) != 570:
        raise SystemExit("%s is not a complete 570-question qaqafull export" % src)
    rows = [r for r in d["evaluations"] if r["pair_id"] in want]
    if len(rows) != 100 or len(want) != 100:
        raise SystemExit("expected the 100 pilot questions, found %d" % len(rows))
    keep = dict(d)
    keep["evaluations"] = rows
    keep["question_ids"] = [r["pair_id"] for r in rows]
    keep["subset"] = {"of": src.name, "of_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "ids_file": PILOT.name,
                      "ids_file_sha256": hashlib.sha256(PILOT.read_bytes()).hexdigest(), "n": len(rows),
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".new")
    tmp.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.rename(out)
    print("pilot subset: %d of %d questions x %d conditions -> %s" % (len(rows), len(d["evaluations"]), len(d["conditions"]), out), flush=True)


if __name__ == "__main__":
    main()
