"""Same-question replay of archived candidate texts and self-verdict scores. CPU only.
Run from the Truth_Decoupling data root, without a model or a judge. No question or answer text is printed.
"""
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path("Iteration_01/newsets/selfjudge")
MODELS = ("llama3.1-8b", "qwen3.5-9b")
SOURCES = {"TQA": ("", "tq_release_", "tq_forced_"), "QA2": ("qaqa_", "qq_release_", "qq_forced_")}
MARGINS = ("baseline__margin", "E_a3__margin")
NET = {"R": 1, "W": -1, "H": 0, "X": 0}


def digest(s):
    return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()


def rows_for(directory, prefix, task):
    path = directory / (prefix + "items.json")
    meta = json.loads((directory / (prefix + "probe.json")).read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["items_sha256"]:
        raise ValueError(f"{path} differs from the scored input")
    records = json.loads(path.read_text())["items"]
    roster, out = set(), {}
    with np.load(directory / (prefix + "probe.npz"), allow_pickle=False) as z:
        names = [r["id"] for r in records if r["answered"]]
        if names != z["ids"].tolist() or len(names) != len(set(names)):
            raise ValueError(f"unexpected row IDs in {path}")
        by_id = {name: i for i, name in enumerate(names)}
        for r in records:
            rid = r["id"]
            if task == "TQA":
                is_test = r.get("split", rid.split("|", 1)[0]) == "test"
            else:
                is_test = not r["pilot"]
            if not is_test:
                continue
            q = rid.split("|", 1)[-1]
            if q in roster:
                raise ValueError(f"duplicate question in {path}")
            roster.add(q)
            if not r["answered"]:
                continue
            i = by_id[rid]
            question = r["question"].strip().rstrip("?").strip() + "?"
            answer = r["answer"].strip()
            stmt = f'The answer to the question "{question}" is: {answer.rstrip(".")}.'
            text_id, stmt_id = digest(answer), digest(stmt)
            if text_id != r["answer_sha1"] or text_id != z["answer_sha1"][i] or stmt_id != z["statement_sha1"][i]:
                raise ValueError(f"answer or statement differs at {rid}")
            v = r["value"] if "value" in r else NET[r["state"]]
            margins = tuple(float(z[name][i]) for name in MARGINS)
            if v not in (-1, 0, 1) or not np.all(np.isfinite(margins)):
                raise ValueError(f"invalid verdict or score at {rid}")
            out[q] = (digest(question), text_id, stmt_id, v, margins, r["neg"])
    return roster, out


@lru_cache(maxsize=2)
def draws(n):
    rng = random.Random(20261116)
    return np.array([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n) for _ in range(4000)])


def interval(column):
    distribution = np.sort(draws(len(column)) @ column)
    return "%+d [%+d,%+d]" % (sum(column), distribution[100], distribution[3899])


def evaluate(ids, incumbent, forced, cutoff):
    # One paired contestant per method and model. No answer wording, oracle scores or
    # reference text enters the nomination, switching or confidence-band decisions.
    rows = np.zeros((len(ids), 12), dtype=np.int64)
    rank, matched_k = [], 0
    for i, q in enumerate(ids):
        if q not in incumbent or q not in forced:
            continue
        ia, fa = incumbent[q], forced[q]
        if ia[0] != fa[0]:
            raise ValueError("candidate answers belong to different questions")
        a, f = ia[3], fa[3]
        same_text = ia[1] == fa[1]
        if same_text:
            f = a  # same answer, one common verdict; remove cross-cache judge noise
        m0, m3 = (y-x for x,y in zip(ia[4],fa[4]))
        usable = ia[2] != fa[2] and (cutoff is None or min(abs(m0), abs(m3)) > cutoff)
        pick0 = usable and m0 > 0
        pick3 = usable and m3 > 0
        chosen0, chosen3 = (f if take else a for take in (pick0, pick3))
        if usable:
            rank.append((i, m0))
            matched_k += pick3
        discordant = a * f == -1 and ia[2] != fa[2]
        rows[i] = (1, int(usable), a, f, chosen0, chosen3, a, int(pick3),
                   int(discordant), int(discordant and chosen0 == 1),
                   int(discordant and chosen3 == 1), int(same_text and ia[3] != fa[3]))
    # Give the native verifier exactly as many switches as the written verifier per
    # model, ordered by the native difference in scores. A tie keeps roster order.
    for i, _ in sorted(rank, key=lambda x: -x[1])[:matched_k]:
        rows[i, 6] = rows[i, 3]
    return rows


def duplicate_check(task, model, sources, ids):
    # Apply one and the same threshold to scores of the identical question+answer in
    # two files; derive both thresholds from the ordinary E tune data, as in R5b.
    path = ROOT/model
    records = json.loads((path/"items.json").read_text())["items"]
    train = [r for r in records if r["answered"] and not r["neg"] and r["split"] == "tune"]
    with np.load(path/"probe.npz", allow_pickle=False) as z:
        ix = dict(zip(z["ids"].tolist(), range(len(z["ids"]))))
        cuts = []
        for cond in MARGINS:
            pairs = [(float(z[cond][ix[r["id"]]]), NET[r["state"]]) for r in train]
            unique = sorted({m for m, _ in pairs})
            choices = [-np.inf] + [(a+b)/2 for a,b in zip(unique,unique[1:])] + [np.inf]
            cuts.append(max(choices, key=lambda t: -sum(v for m,v in pairs if m<t)))
    e, c, f = [s[1] for s in sources]
    for name, arm in (("E",e),("C",c)):
        shared = [(arm[q],f[q]) for q in ids if q in arm and q in f and arm[q][:2] == f[q][:2]]
        for j,t in enumerate(cuts):
            differences = [abs(y[4][j]-x[4][j]) for x,y in shared]
            flips = sum(not a[5] and ((a[4][j]<t) != (b[4][j]<t)) for a,b in shared)
            print(task,model,name,MARGINS[j],"identical",len(shared), "abs>.02",sum(v>.02 for v in differences),
                  "max",round(max(differences,default=0),3),"same-threshold flips",flips)


def main():
    for task, prefixes in SOURCES.items():
        all_data = [[rows_for(ROOT/m, s, task) for s in prefixes] for m in MODELS]
        ids = sorted(set.intersection(*(model_data[0][0] for model_data in all_data)))
        if len(ids) != (523 if task == "TQA" else 470):
            raise ValueError("unexpected common evaluation roster")
        for m, sources in zip(MODELS,all_data):
            duplicate_check(task,m,sources,ids)
        for group in ("E", "C", "all"):
            for cutoff in (None,.375,1.75):
                if group != "C" and cutoff is not None:
                    continue
                blocks = []
                for m,sources in zip(MODELS,all_data):
                    e, c, f = [s[1] for s in sources]
                    if e.keys() & c.keys():
                        raise ValueError("C nominated a question E had answered")
                    incumbent = e if group == "E" else c if group == "C" else {**e, **c}
                    rows = evaluate(ids,incumbent,f,cutoff)
                    blocks.append(rows)
                    if group == "C":
                        print(task,m,group,cutoff,"pairs",rows[:,0].sum(),"eligible",rows[:,1].sum(),
                              "incumbent/native/written",rows[:,[2,4,5]].sum(0).tolist(),
                              "written-native",interval(rows[:,5]-rows[:,4]),
                              "written-matched native",interval(rows[:,5]-rows[:,6]))
                table = sum(blocks)
                print(task,"together",group,cutoff,"pairs",table[:,0].sum(),"eligible",table[:,1].sum(),
                      "incumbent/forced/native/written",table[:,[2,3,4,5]].sum(0).tolist(),
                      "written-native",interval(table[:,5]-table[:,4]),
                      "written-incumbent",interval(table[:,5]-table[:,2]),
                      "written-native matched switches",interval(table[:,5]-table[:,6]),
                      "opposite verdicts/native picks right/written picks right",table[:,[8,9,10]].sum(0).tolist(),
                      "equal-text inconsistent labels",table[:,11].sum())


if __name__ == "__main__":
    main()
