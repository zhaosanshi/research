"""Isolated completion of existing TruthfulQA Gemma-3-12B batch-one self-scores. No answers generated or judged.
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

WORK = Path("Iteration_01/gpt_tools_20261128/n1_tq_g12_double_blank_0141")
MODELS = {"gemma3-12b"}
TAG = {"smoke": "td28n_tq_smoke", "missing": "td28n_tq_missing"}


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


def read_targets():
    import tqa_gate2b_1127 as tq

    m = "gemma3-12b"
    b = tq.base(m)
    ids = b.ids
    assert len(ids) == len(set(ids)) == 522 and (b.el, b.cl, b.sd) == (24, 25, "2-5")
    root = Path(SP.OUT) / m / "items.json"
    items = {
        r["id"]: r["question"]
        for r in json.loads(root.read_bytes())["items"]
        if r["split"] == "test"
    }
    questions = {q: items[q] for q in ids}

    def cells(cond):
        return {
            q: (
                tq.VAL[tq.G[b.sp.st[q, cond]]],
                not b.sp.blank[q, cond],
                b.tx.ec[q, cond],
            )
            for q in ids
        }

    e = cells(b.sp.cE(b.sd))
    o = cells("baseline_x")
    ec = cells(b.sp.cEC(b.sd, "1"))
    both = {q: (e[q] if e[q][1] or o[q][1] else ec[q]) for q in ids}
    arms = {
        "C1": cells(b.sp.cC("1")),
        "C05": cells(b.sp.cC("0-5")),
        "both_blank_EC1": both,
    }
    assert (
        sum(x[1] for x in both.values()) == 358
        and sum(x[0] for x in both.values() if x[1]) == 128
    )
    return ids, questions, arms, e, root


def prepare():
    m = "gemma3-12b"
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
    ids, questions, arms, e, questions_path = read_targets()
    scores = originals(m)
    extra = (
        Path("Iteration_01/gpt_tools_20261128/n1_falseqa_c2_verifier_0134")
        / m
        / "verified_missing.json"
    )
    for key, val in json.loads(extra.read_bytes()).items():
        assert key not in scores or scores[key]["baseline"] == val
        scores[key] = {"baseline": val}
    known = {}
    missing = {}
    provenance = {}
    manifest = {}
    counts = {}
    for name, arm in arms.items():
        roster = []
        absent = 0
        for q in ids:
            _, issued, txt = arm[q]
            sha = F.statement_sha(questions[q], txt)
            roster.append((q, issued, SP.sha1(questions[q]), SP.sha1(txt), sha))
            if not issued:
                continue
            raw_id = hashlib.sha256(
                json.dumps(
                    [q, questions[q], txt], ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest()
            rec = dict(
                id=raw_id,
                split="TruthfulQA frozen test; Gemma 12B",
                question=questions[q],
                answer=txt,
                answer_sha1=SP.sha1(txt),
                value=None,
                answered=True,
                neg=bool(SP.NEG_RE.match(txt)),
            )
            dest = known if sha in scores else missing
            assert raw_id not in dest or dest[raw_id] == rec
            dest[raw_id] = rec
            if sha not in scores:
                absent += 1
                provenance.setdefault(sha, []).append([q, name, raw_id])
        manifest[name] = hashlib.sha256(
            json.dumps(roster, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        counts[name] = {"issued": sum(x[1] for x in arm.values()), "missing": absent}
    assert not set(known) & set(missing)
    smoke = random.Random(20261128).sample(list(known), 16)
    for tag, src in [
        ("smoke", [known[h] for h in smoke]),
        ("missing", list(missing.values())),
    ]:
        write(
            root / (TAG[tag] + "_items.json"), dict(model=m, source=TAG[tag], items=src)
        )
    detail = {
        "model": m,
        "n": len(ids),
        "frozen_roster_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "question_source_sha256": SP.sha256_file(str(questions_path)),
        "arms_manifest_sha256": manifest,
        "counts": counts,
        "expected": {
            F.statement_sha(known[h]["question"], known[h]["answer"]): scores[
                F.statement_sha(known[h]["question"], known[h]["answer"])
            ]
            for h in smoke
        },
        "sources": {
            st: SP.sha256_file(str(root / (TAG[st] + "_items.json"))) for st in TAG
        },
        "missing_q_and_hash": [
            (h, F.statement_sha(d["question"], d["answer"])) for h, d in missing.items()
        ],
        "provenance": provenance,
    }
    write(root / "contract.json", detail)
    print(
        "prepared",
        counts,
        "union missing",
        len(missing),
        "known smoke",
        len(smoke),
        "rows",
        len(ids),
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
    for h in set(z["statement_sha1"]):
        ii = [i for i, x in enumerate(z["statement_sha1"]) if x == h]
        assert all(
            np.all(z[c + "__margin"][ii] == z[c + "__margin"][ii[0]])
            for c in meta["conditions"]
        )
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
        assert len(stat) == len(contract["provenance"])
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
