"""Read completed ITI evaluations in the scoring container, from Truth_Decoupling.
Run with python3 -B PATH. No fit, inference, writes or judge calls.
Prints the natural paired net difference and the finite-sample equal-count range:
after both systems retain exactly min(A, B) of their issued answers, how far
apart can they score under the best/worst possible choices of rows to retain?
All omitted rows score zero in that comparison.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/review_checks"]
import iti_table as archive
from judge_two_axis_run import load_verified_single

V = {"R": 1, "W": -1, "X": 0, "H": 0}
LABEL = {"A": "R", "B": "W", "C": "X", "H": "H"}
PATHS = (("train_seed42", "endq"), ("train_seed42", "pairs"), ("traintune_seed42", "endq"),
         ("faithful_seed42", "endq"))


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def retained_bounds(states, flags, k):
    eligible = sorted(V[s] for s, f in zip(states, flags) if f)
    if not 0 <= k <= len(eligible):
        raise ValueError("Requested number of retained answers is unreachable")
    return sum(eligible[:k]), sum(eligible[len(eligible)-k:])


def iti_inputs(model, run, fit, sigma, unbounded, questions):
    folder = Path(archive.BASE) / model
    trial = folder / fit
    suffix = "_sp" if sigma == "pairs" else ""
    ch = trial / ("choice" + ("_all" if unbounded else "") + suffix + ".json")
    if not ch.exists():
        return None
    choice = read(ch)
    if (choice["model"], choice["fit"], choice["sigma"]) != (model, fit, sigma):
        raise ValueError("Choice belongs to a different model, fit or sigma")
    if (choice["budget"] is None) != unbounded:
        raise ValueError("Choice belongs to a different budget")
    scores = choice["candidates"]
    if len({(k, a) for k,a,_ in scores}) != len(scores):
        raise ValueError("Duplicate grid point in saved choice")
    best = max(scores, key=lambda row: (row[2], -row[0], -row[1]))
    if best != [choice["k"],choice["alpha"],choice["tune_net"]]:
        raise ValueError("Saved choice disagrees with its development table")
    work = Path(archive.MP.work(run, "test", "iti_" + fit + suffix + ("_all" if unbounded else "")))
    if unbounded and not (work / "judge2/single_pro.jsonl").exists():
        main = trial / ("choice" + suffix + ".json")
        if main.exists() and read(main)["stem"] == choice["stem"]:
            work = Path(archive.MP.work(run, "test", "iti_" + fit + suffix))
    if not (work / "judge2/single_pro.jsonl").exists():
        return None
    gen_path = trial / "gen" / ("test_" + choice["stem"] + ".json")
    ref_path = work / "judge_input.json"
    gen, exported = read(gen_path),read(ref_path)
    cond = gen["condition"]
    if gen["model"] != model or gen["split"] != "test" or not gen["completed"] or gen["identity_sha256"] != exported["identity_sha256"]:
        raise ValueError("Unexpected test generation")
    if exported["source_files"][cond] != {"file":gen_path.name,"sha256":digest(gen_path)}:
        raise ValueError("Vote source differs from the selected generation")
    setting = gen["setting"]
    if (setting["k"], setting["alpha"], setting["sigma"]) != (choice["k"], choice["alpha"], sigma):
        raise ValueError("Write differs from selected grid point")
    if setting["fit_file_sha256"] != digest(folder / ("fit_" + fit + ".pt")):
        raise ValueError("Write belongs to a different set of fitted directions")
    fit_meta = read(folder / ("fit_" + fit + ".pt.json"))
    expected_heads = [h[:2] for h in fit_meta["ranking"][:choice["k"]]]
    if [h[:2] for h in setting["heads"]] != expected_heads:
        raise ValueError("The generation used a different head ranking from the committed fit")
    preds = {r["pair_id"]:r for r in gen["rows"]}
    if len(preds) != len(gen["rows"]) or list(preds) != gen["question_ids"] or gen["question_ids"] != exported["question_ids"]:
        raise ValueError("Unexpected question cohort or row order")
    _,cells,votes = load_verified_single(str(work), "pro", generation_path=str(ref_path), require_complete=False)
    incoming = {c["pair_id"]:(c,votes[cid]) for cid,c in cells.items() if c["condition"] == cond}
    states, flags = [], []
    for q in questions:
        c,r = incoming[q]
        if c["prediction"] != preds[q]["text"] or c["rule_abstain"] != preds[q]["rule_abstain"]:
            raise ValueError("Mismatched output text or abstention flag")
        if r["status"] == "scored":
            states.append(LABEL[r["label"]])
        elif not c["prediction"].strip() and r.get("basis") == "empty_output_no_judge_vote":
            states.append("X")
        else:
            raise ValueError("Unresolved ITI verdict")
        flags.append(not (c["rule_abstain"] or not c["prediction"].strip()))
    return states, flags, choice


def main():
    for model in ("llama3.1-8b","gemma3-12b"):
        display,layer,strength=archive.KEY[model]
        tune,err=archive.MP.prompt_split(display,layer,0,strength)
        if tune is None:
            raise ValueError(err)
        test,err=archive.MP.prompt_split(display,layer,1,strength)
        if test is None:
            raise ValueError(err)
        sp,tx,_,ids=test
        tsp,tt,_,_=tune
        _,run,_=archive.MP.usual(display,1)
        n1=archive.ML.n1_pick(tsp,tt,strength)
        systems={k:archive.S.system(sp,tx,cfg)[:2] for k,cfg in (("E",("E",strength)),("N1",n1))}
        for fit,sigma in PATHS:
            for unbounded in (False,True):
                row=iti_inputs(model,run,fit,sigma,unbounded,ids)
                if row is None:
                    print(model,fit,sigma,"grid" if unbounded else "budget", "no completed comparison")
                    continue
                iti,ifa,ch=row
                out={"model":model,"fit":fit,"sigma":sigma,"budget":ch["budget"],"setting":[ch["k"],ch["alpha"]],
                     "questions":len(ids),"iti_net":sum(V[s] for s in iti),"iti_answers":sum(ifa)}
                for key,(st,ans) in systems.items():
                    n=min(sum(ifa),sum(ans))
                    sl,sh=retained_bounds(st,ans,n);il,ih=retained_bounds(iti,ifa,n)
                    out[key]={"natural_minus_iti":archive.ps.ci([V[a]-V[b] for a,b in zip(st,iti)]),
                              "matched_answers":n,"matched_delta_min":sl-ih,"matched_delta_max":sh-il}
                print(json.dumps(out),flush=True)


if __name__=="__main__":
    main()
