"""Isolated completion of existing FalseQA C2 batch-one self-scores. No answers generated or judged.
Run in the archive container from the project root; pass prepare, score MODEL smoke|missing, or extract.
The launcher must hold the project's device locks for each GPU operation.
"""

from pathlib import Path
import hashlib
import json
import os
import sys
import random
import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
    "Iteration_01/gpt_tools_20261126/gate_audit",
]
import falseqa_read as F
import falseqa_gate2_1127 as H
import selfjudge_probe as SP
import selfjudge_two_answers as TA
import selfjudge_stack_pick as SK
import fill_only2

WORK = Path("Iteration_01/gpt_tools_20261128/n1_falseqa_c2_verifier_0134")
MODELS = {"llama3.1-8b": 103, "gemma3-12b": 89}
TAG = {"smoke": "td28n_c2_smoke", "missing": "td28n_c2_missing"}


def write(path, data):
    with path.open("x", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=1)


def originals(m):
    SP.SOURCES.update(F.SRC["smoke"] + F.SRC["full"] + H.sources_all())
    extras = []
    for p in sorted((Path(SP.OUT) / m).glob("*_probe.json")):
        name = p.name.removesuffix("_probe.json")
        if (
            name not in SK.REUSE
            and name != "sp_items"
            and json.loads(p.read_bytes()).get("batch_size") == 1
        ):
            assert name in SP.SOURCES, name
            extras.append(name)
    return SK.lookup(m, extra_sources=tuple(extras))


def prepare():
    ok, bank = F.check_bank()
    assert ok
    rows = [r for r in bank if r["set"] == "heldout"]
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids) == 600
    for m, expected in MODELS.items():
        root = WORK / m
        root.mkdir()
        archived = (Path(SP.OUT) / m).resolve()
        for f in [
            "items.json",
            "probe.json",
            "probe.npz",
            "tj_s0_items.json",
            "tj_s0_probe.json",
            "tj_s0_probe.npz",
        ]:
            original = archived / f
            assert original.is_file()
            (root / f).symlink_to(original)
        r = F.FRaw(m, set(ids))
        assert r.ids == ids
        arm = F.roster(m)["C2"]
        assert arm == r.c("2")
        r.need(arm)
        scores = originals(m)
        known = []
        missing = []
        maps = {}
        for row in rows:
            q = row["id"]
            question = row["question"]
            answer = r.text(arm, q)
            if r.blank(arm, q):
                continue
            sh = F.statement_sha(question, answer)
            assert sh not in maps
            maps[sh] = q
            rec = dict(
                id=q,
                split="falseqa heldout C2",
                question=question,
                answer=answer,
                answer_sha1=SP.sha1(answer),
                value=None,
                answered=True,
                neg=bool(SP.NEG_RE.match(answer)),
            )
            (known if sh in scores else missing).append((sh, rec))
        assert len(missing) == expected
        smoke = random.Random(20261128).sample(known, 16)
        records = {"smoke": smoke, "missing": missing}
        for stage, l in records.items():
            tag = TAG[stage]
            write(
                root / (tag + "_items.json"),
                dict(model=m, source=tag, items=[d for _, d in l]),
            )
        detail = {
            "model": m,
            "arm": arm,
            "bank_sha256": SP.sha256_file(F.BANK),
            "generation_sha256": SP.sha256_file(
                r.gdir / (F.G.arm_token(arm) + ".json")
            ),
            "issued": len(known) + len(missing),
            "expected": {h: scores[h] for h, _ in smoke},
            "sources": {
                st: SP.sha256_file(str(root / (TAG[st] + "_items.json")))
                for st in records
            },
            "missing_q_and_hash": [(item["id"], sh) for sh, item in missing],
        }
        write(root / "contract.json", detail)
        print(
            "prepared",
            m,
            arm,
            "missing",
            len(missing),
            "known smoke",
            len(smoke),
            flush=True,
        )


def validate_artifact(m, stage):
    root = WORK / m
    contract = json.loads((root / "contract.json").read_bytes())
    tag = TAG[stage]
    assert (
        SP.sha256_file(str(root / (tag + "_items.json"))) == contract["sources"][stage]
    )
    SP.SOURCES.update(TAG.values())
    original = SP.OUT
    SP.OUT = str(WORK)
    try:
        doc, meta, z = SP.read_probe(m, source=tag)
    finally:
        SP.OUT = original
    assert meta["batch_size"] == 1
    stat = {
        str(q): {k: float(z[k + "__margin"][j]) for k in meta["conditions"]}
        for j, q in enumerate(z["statement_sha1"])
    }
    if stage == "smoke":
        assert stat.keys() == contract["expected"].keys()
        assert all(
            stat[h][k] == contract["expected"][h][k] for h in stat for k in stat[h]
        ), "archived batch-one margin differs"
        print(
            "exact batch-one smoke",
            m,
            len(stat),
            meta["conditions"],
            "all equal",
            flush=True,
        )
    else:
        assert list(zip(z["ids"].tolist(), z["statement_sha1"].tolist())) == [
            tuple(x) for x in contract["missing_q_and_hash"]
        ]
        assert len(stat) == MODELS[m]
        print("validated missing", m, len(stat), flush=True)
    return stat


def score(m, stage):
    import torch

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    if stage == "missing":
        for other in MODELS:
            validate_artifact(other, "smoke")
    old = SP.OUT
    SP.SOURCES.update(TAG.values())
    SP.OUT = str(WORK)
    try:
        TA.score1(m, [TAG[stage]])
    finally:
        SP.OUT = old
    vals = validate_artifact(m, stage)
    if stage == "missing":
        write(
            WORK / m / "verified_missing.json",
            {key: rec["baseline"] for key, rec in vals.items()},
        )


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        for m in MODELS:
            validate_artifact(m, "missing")
