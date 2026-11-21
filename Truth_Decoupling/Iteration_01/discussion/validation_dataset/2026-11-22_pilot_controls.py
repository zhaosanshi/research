"""Read-only CPU replay for the two archived 0607e79 pilot models.

From the experiment's Truth_Decoupling root run python3 -B PATH. Candidates are
chosen on tune only. Outputs are alias scores. The confidence-matched comparisons
use a common external, label-free ranking on the first test batch; test2 lacks it.
"""
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import alias_match as am
import newsets_score as ns
import ns_prompt_table as nt
import nsc_prompt_table as pilot
import release_origin as ro

MODELS = ("llama3.1-8b", "gemma3-12b")


def require(ok, message):
    if not ok:
        raise ValueError(message)


@lru_cache(maxsize=None)
def sha(path):
    return ns.sha_file(path)


def loader(task, model, split):
    bank, path = nt.bank(task, split)
    ids = list(bank)
    manifest = Path(ns.NS) / "splits" / (task + ("_test2_20261120.json" if split == "test2" else "_splits_20261119.json"))
    require(ids == json.loads(manifest.read_text())[split] and len(ids) == (192 if split == "tune" else 500),
            "wrong pilot cohort")
    config = ns.models()[model]
    parent = Path(config["run_dir"])
    frozen = parent / "frozen_tune.json"
    identity = json.loads(frozen.read_text())["identity_sha256"]
    bank_sha = sha(path)

    @lru_cache(maxsize=None)
    def take(arm):
        file = Path(ns.NS) / "gen_prompt" / task / model / split / (ns.tok(arm) + ".json")
        if not file.is_file():
            return None
        doc = json.loads(file.read_text())
        k = doc["contract"]
        expect = dict(gen_version=2, task=task, model=model, split=split, arm=arm,
                      full_pass=True, positions="prompt" if ":" in arm else None,
                      ids=ids, items_file_sha256=bank_sha, identity_sha256=identity,
                      frozen_file_sha256=sha(frozen))
        rows = doc["rows"]
        require(doc.get("completed") is True and all(k.get(f) == v for f, v in expect.items())
                and all(doc.get(f) == v for f, v in k.items()) and [r["id"] for r in rows] == ids,
                f"incomplete or foreign generation: {file}")
        parts = arm.partition(":")[2].split("+") if ":" in arm else []
        require(len(parts) == len(k["parts"]), f"wrong writes: {file}")
        for spec, part in zip(parts, k["parts"]):
            layer, beta = spec.split(".w.", 1)
            fam, level = layer[0], int(layer[1:])
            run = Path(config.get("c_run_dir") or parent) if fam == "C" else parent
            sign = -1 if beta.startswith("m") else 1
            require(part["family"] == fam and part["layer"] == level and part["kind"] == "w"
                    and part["beta"] == sign * float(beta[1:]) and part["direction_run"] == run.name
                    and part["direction_file_sha256"] == sha(run / "dirs" / f"{fam}_L{level}.pt"),
                    f"wrong trained tensor/dose in {file}")
        result = {}
        for row in rows:
            q, text = row["id"], row["text"]
            require(isinstance(row["rule_abstain"], bool), f"missing routing marker: {file}")
            ans = bool(text.strip()) and not row["rule_abstain"]
            state = am.verdict(text, bank[q]["gold"], ans)
            result[q] = ((state == "R") - (state == "W"), int(ans), ro.norm(text))
        return result
    return ids, take


def confirm(first, second):
    if first is None or second is None:
        return None
    require(list(first) == list(second), "two passes cover different items")
    out = {}
    for q, (value, answered, text) in first.items():
        keep = bool(answered and second[q][1] and text == second[q][2])
        out[q] = (value if keep else 0, int(keep), text)
    return out


def enact(take, key):
    if key[0] == "arm":
        return take(key[1])
    return confirm(take(key[2] if key[0] == "gate" else "x"), take(key[1]))


def select(task, model):
    _, take = loader(task, model, "tune")
    families = pilot.families(model, ns.models()[model]["cl"])
    u = families[pilot.ALL_FAM]
    es, cs = families["single E"], families["single C, either sign"]
    # Offer the single-direction controls both direct answering and checking the original.
    ecandidates = es + [("check", a) for _, a in es]
    ccandidates = cs + [("check", a) for _, a in cs]
    controls = dict(U=u, E=es, C=cs, Echeck=ecandidates, Ccheck=ccandidates,
                    joint=[key for key in u if "+" in key[1]])
    picks = {}
    for label, keys in controls.items():
        scores = []
        for key in keys:
            x = enact(take, key)
            require(x is not None, f"missing tune candidate: {task} {model} {key}")
            scores.append(sum(r[0] for r in x.values()))
        winners = [k for k, s in zip(keys, scores) if s == max(scores)]
        picks[label] = winners[0]
        if label == "U":
            # Judge the sensitivity of a development winner without inspecting
            # either test split. Preserve exactly the published tie convention.
            matrix = np.array([[r[0] for r in enact(take, k).values()] for k in keys])
            left_out = np.argmax(np.asarray(scores)[:, None] - matrix, axis=0)
            print(json.dumps(dict(task=task, model=model, diagnostic="leave_one_development_item_out",
                                  questions=matrix.shape[1], identical_winner=int(np.sum(left_out == scores.index(max(scores)))),
                                  uses_both_axes=sum("+" in keys[i][1] for i in left_out))))
        print(json.dumps(dict(task=task, model=model, family=label, candidates=len(keys),
                              score=max(scores), best=picks[label], ties=winners)))
    return picks


def no_write_rank(task, model, split):
    g, _ = ns.gen_file(task, model, split, "conf")
    text = {task + ":" + r["id"]: r["text"] for r in g["rows"]}
    path = Path(ns.NS) / "conf" / task / model / split / "conf.json"
    src = json.loads(path.read_text())
    require([r["id"] for r in src["rows"]] == [r["id"] for r in g["rows"]], "confidence IDs changed")
    conf = ns.conf_rows(task, model, split, "conf", text)
    require(set(conf) == set(text), "partial confidence")
    return {p.split(":", 1)[1]: r["mean_logp"] for p, r in conf.items()}


def outputs(task, model, split, picks):
    ids, take = loader(task, model, split)
    rows = {f: enact(take, key) for f, key in picks.items()}
    rows["separate"] = confirm(take(picks["E"][1]), take(picks["C"][1]))
    rows["x"] = take("x")
    ranking = None
    if split == "test":
        ranking = no_write_rank(task, model, split)
        require(set(ranking) == set(ids) and all(math.isfinite(v) for v in ranking.values()), "invalid ranks")
        doc, _ = ns.gen_file(task, model, split, "conf")
        bank, _ = nt.bank(task, split)
        other = {}
        for r in doc["rows"]:
            q, text = r["id"], r["text"]
            a = not r["rule_abstain"] and bool(text.strip())
            state = am.verdict(text, bank[q]["gold"], a)
            other[q] = ((state == "R") - (state == "W"), int(a), ro.norm(text))
        rows["must_answer"] = other
        rows["change_prompt"] = confirm(rows["x"], other)
    return ids, rows, ranking


def report(task, split, archives, selector, controls, draws, matched):
    families = (selector, *controls)
    ids = archives[0][0]
    by_model, counts = [], []
    for qs, rows, rank in archives:
        require(qs == ids, "models have different ordered IDs")
        if any(rows[k] is None for k in families):
            print(json.dumps(dict(task=task, split=split, method=selector, controls=controls, status="missing arm")))
            return
        n = min(sum(r[1] for r in rows[k].values()) for k in families)
        vals, taken = {}, {}
        for k in families:
            r = rows[k]
            require(list(r) == ids, "lost questions")
            candidates = [q for q in ids if r[q][1]]
            if matched:
                kept = set(sorted(candidates, key=lambda q: -rank[q])[:n])
                vals[k] = np.array([r[q][0] if q in kept else 0 for q in ids])
                taken[k] = len(kept)
            else:
                vals[k] = np.array([r[q][0] for q in ids])
                taken[k] = len(candidates)
        by_model.append(vals)
        counts.append(taken)
    for c in controls:
        diff = sum(z[selector] - z[c] for z in by_model)
        bs = np.sort(diff[draws].sum(axis=1))
        print(json.dumps(dict(task=task, split=split, method=selector, control=c,
                              common_confidence_filter=matched, counts=counts,
                              net=int(diff.sum()), ci=[int(bs[100]), int(bs[3899])])))


def main():
    rng = random.Random(ns.SEED)
    draws = np.array([[rng.randrange(500) for _ in range(500)] for _ in range(4000)])
    for task in ns.TASKS:
        picks = [select(task, m) for m in MODELS]
        for split in ("test", "test2"):
            data = [outputs(task, m, split, p) for m, p in zip(MODELS, picks)]
            for method in ("U", "joint", "separate"):
                for c in ("E", "C", "Echeck", "Ccheck"):
                    report(task, split, data, method, [c], draws, False)
            if split == "test":
                for method in ("U", "separate"):
                    for controls in (["E", "C"], ["Echeck"], ["Ccheck"], ["x"], ["must_answer"], ["change_prompt"]):
                        report(task, split, data, method, controls, draws, True)


if __name__ == "__main__":
    main()
