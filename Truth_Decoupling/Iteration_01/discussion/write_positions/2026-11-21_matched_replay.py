"""Read-only replay of the 89aedc1 budget-one prompt-position table.

Run with python3 -B PATH_TO_THIS_FILE in the scoring container, from the
Truth_Decoupling directory. No generation or judge calls. Requires the archived
server inputs and the project's native single-judge verifier. The target N is
computed within the actual evaluation cohort for each model. Confidence ranks
are computed again on that same cohort; ties use its frozen question order.
Intervals follow the main table: paired question resampling after selection,
with the models sharing a question always resampled together. In the output,
strong = must-answer + E at the prompt rule; prompt = no write with the
truthfulness sentence; forced_base = no write with the plain must-answer prompt.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Iteration_01/diag").resolve()))
sys.path.insert(0, str(Path("Iteration_01/code/scripts/review_checks").resolve()))
import main_table_prompt as mp
import prompt_conf_table as pc
from judge_two_axis_run import load_verified_single

# All candidate layers investigated for this particular table, including those
# that lost development selection. Stop rather than silently skip a missing one.
LAYERS = {
    "Llama-3.1-8B": [11, 16], "Qwen3.5-9B": [15], "Qwen3.8-27B": [28],
    "Gemma-3-12B": [22, 24], "Gemma-3-27B": [26, 28],
}
MAIN = tuple(LAYERS)[:3]
ALL = tuple(LAYERS)


def read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(state):
    if state not in ("R", "W", "H", "X"):
        raise ValueError(f"unknown verdict: {state}")
    return int(state == "R") - int(state == "W")


def verify_grid(run, split, tag):
    work = Path(mp.work(run, split, tag))
    path = work / "judge_input.json"
    load_verified_single(work, "pro", generation_path=path, require_complete=False)
    doc = read(path)
    ids = doc["question_ids"]
    if len(set(ids)) != len(ids) or not doc["completed"] or doc["split"] != split:
        raise ValueError(f"Wrong or incomplete export: {path}")
    if not doc["sources"]["gen"]:
        raise ValueError(f"Missing generation sources: {path}")
    for name in doc["sources"]["gen"]:
        src = (Path(run) / "eval_extra" / "gen" / name).resolve()
        if not name.endswith(f"-at-prompt_{split}.json") or digest(src) != doc["sources"]["sha256"][str(src)]:
            raise ValueError(f"Wrong generation source: {src}")
        g = read(src)
        if (g["positions"] != "prompt" or not g["completed"] or g["split"] != split
                or g["identity_sha256"] != doc["identity_sha256"]
                or [r["pair_id"] for r in g["rows"]] != ids):
            raise ValueError(f"Wrong execution, scope or cohort: {src}")
    return doc


def confidence_rows(report, target_name, texts, ids, wanted_source, identity, frozen):
    d = read(report)
    if d["partial"] or d["split"] != "test" or d["identity_sha256"] != identity or d["frozen_file_sha256"] != frozen:
        raise ValueError(f"Foreign or partial confidence: {report}")
    target = d["targets"][target_name]
    if target["greedy_reproduction"]["rate"] < 0.98:
        raise ValueError(f"Greedy replay below 98%: {report}")
    if wanted_source is None:
        prompt = target_name.split(":", 1)[1]
        if (target["base_prompt"] != prompt or target["source"] != {"kind": "frozen", "prompt": prompt}
                or target["writes"]):
            raise ValueError(f"Wrong prompt-only source: {report}")
    else:
        source = target["source"]
        if source["file"] != wanted_source.name or source["file_sha256"] != digest(wanted_source):
            raise ValueError(f"Confidence is bound to another generation: {report}")
    rows = {r["pair_id"]: r for r in target["rows"]}
    if len(rows) != len(target["rows"]):
        raise ValueError(f"Duplicate confidence rows: {report}")
    for q in ids:
        expected = hashlib.sha1(texts[q].encode("utf-8")).hexdigest()
        if rows[q]["text_sha1"] != expected:
            raise ValueError(f"{report}: different answer for {q}")
        score = rows[q]["mean_logp"]
        if score is not None and not math.isfinite(score):
            raise ValueError(f"{report}: non-finite confidence for {q}")
    return {q: rows[q]["mean_logp"] for q in ids}


def matched_tune_prompt(run, tune, tune_text, cfg):
    """Exploratory check: which available no-write prompt wins at N1's tune answer count?"""
    _, answered, _ = mp.S.system(tune, tune_text, cfg)
    n = sum(answered)
    prompts = pc.judged(run, "tune", tune.qs)
    scores = {}
    for p in pc.PROMPTS:
        if p not in prompts:
            continue
        states, take, texts = prompts[p]
        conf, _ = pc.confidence(run, "tune", p, tune.qs, texts)
        if conf is None:
            continue
        if any(t and conf[q] is not None and not math.isfinite(conf[q]) for q, t in zip(tune.qs, take)):
            raise ValueError(f"Non-finite confidence in tune prompt {p}")
        rank = pc.ranked(take, conf, tune.qs)
        if len(rank) >= n:
            scores[p] = sum(value(states[i]) for i in rank[:n])
    winner = min(scores, key=lambda p: (-scores[p], p != "conf_E2", p))
    return dict(n=n, eligible=len(scores), winner=winner, tune_net=scores[winner])


def load_model(model):
    _, original, sources = mp.ML.sources(model)
    if [r[0] for r in sources] != LAYERS[model]:
        raise ValueError(f"Incomplete list of candidate layers for {model}")
    run = original.m["ec"][1].split("/eval_extra/")[0]
    baseline = sources[0][6]["0"]
    sel, rep = mp.PM.halves(list(baseline))
    for split in ("tune", "test"):
        verify_grid(run, split, "pospromptc")
    candidates = []
    for layer, tsc, ttc, xsc, xtc, strengths, accuracy, _ in sources:
        if accuracy["0"] != baseline:
            raise ValueError(f"MMLU baseline differs for {model} layer {layer}")
        eligible = [s for s in strengths if s in accuracy and
                    100 * sum(accuracy[s][q] - baseline[q] for q in sel) / len(sel) >= -1]
        if not eligible:
            continue
        strength = max(eligible, key=lambda s: float(mp.DM.show(s)))
        args = []
        for i, old in enumerate((tsc, xsc)):
            split = ("tune", "test")[i]
            for prefix in ("pospromptl", "posprompt2l"):
                verify_grid(run, split, prefix + str(layer))
            data, why = mp.prompt_split(model, layer, i, strength, qs=old.qs)
            if data is None:
                raise ValueError(why)
            args.append(data)
        tune, ttext, single_c_tune, _ = args[0]
        cfg = mp.ML.n1_pick(tune, ttext, strength)
        score = mp.ps.score(mp.S.system(tune, ttext, cfg)[0], 1)
        candidates.append((score, layer, strength, cfg, single_c_tune, args, accuracy))
    if not candidates:
        raise ValueError(f"No budget-eligible setting for {model}")
    # Python max keeps the earlier candidate on an exact tie.
    _, layer, strength, cfg, single_c_tune, data_pair, accuracy = max(candidates, key=lambda r: r[0])
    chosen_tune, chosen_tune_text, _, _ = data_pair[0]
    sp, txt, single_c_test, ids = data_pair[1]
    c_strength = max((k for k in mp.ps.LAD if k in single_c_tune),
                     key=lambda k: mp.ps.score(single_c_tune[k][0], 1))
    choice, take, _ = mp.S.system(sp, txt, cfg)
    c_states, c_take = single_c_test[c_strength]
    work = Path(mp.work(run, "test", "pospromptdl" + str(layer)))
    doc = verify_grid(run, "test", "pospromptdl" + str(layer))
    states, blanks = mp.ps.load(str(work), "pro")
    text = mp.fr.raw_texts(str(work))
    cond = f"E_L{layer}_conf-w-p{strength}"
    gen = Path(run) / "eval_extra" / "gen" / f"conf--E{layer}-w-p{strength}-at-prompt_test.json"
    file = Path(run) / "eval_extra" / f"confidence_pospromptdl{layer}_test.json"
    frozen = digest(Path(run) / "frozen_test.json")
    e_prob = confidence_rows(file, "conf_E" + mp.DM.show(strength),
                             {q: text[q, cond].strip() for q in ids}, ids,
                             gen, doc["identity_sha256"], frozen)
    pw = Path(mp.work(run, "test", "promptsweep"))
    load_verified_single(pw, "pro", generation_path=pw / "judge_input.json", require_complete=False)
    judged = pc.judged(run, "test", ids)
    pools = {}
    for key, prompt in (("prompt", "conf_E2"), ("forced_base", "conf")):
        p_states, p_take, p_text = judged[prompt]
        file = Path(run) / "eval_extra" / f"confidence_prompts_test_{prompt}.json"
        if not file.exists():
            file = file.with_name(file.stem + "_b1.json")
        prob = confidence_rows(file, "prompt:" + prompt, p_text, ids, None, doc["identity_sha256"], frozen)
        pools[key] = {q: (value(p_states[i]), p_take[i], prob[q]) for i, q in enumerate(ids)}
    mmlu_ci = mp.PM.boot([accuracy[strength][i] - baseline[i] for i in rep])
    matched_tune = matched_tune_prompt(run, chosen_tune, chosen_tune_text, cfg)
    print(json.dumps(dict(model=model, layer=layer, strength=strength, caps=cfg, C=c_strength,
                          matched_tune_prompt=matched_tune,
                          questions=len(ids), mmlu_selection=len(sel), mmlu_report=len(rep), mmlu_pp=mmlu_ci)))
    return {q: dict(n1=(value(choice[i]), take[i]), single_c=(value(c_states[i]), c_take[i]),
                    strong=(value(states[q, cond]), not blanks[q, cond], e_prob[q]),
                    prompt=pools["prompt"][q], forced_base=pools["forced_base"][q]) for i, q in enumerate(ids)}


def matched(ids, data, key):
    n = sum(data[q]["n1"][1] for q in ids)
    eligible = [q for q in ids if data[q][key][1] and data[q][key][2] is not None]
    if len(eligible) < n:
        raise ValueError(f"{key}: {len(eligible)} available, {n} required")
    kept = set(sorted(eligible, key=lambda q: -data[q][key][2])[:n])
    return n, [data[q]["n1"][0] - (data[q][key][0] if q in kept else 0) for q in ids]


def report(data, models):
    ids = sorted(set.intersection(*(set(data[m]) for m in models)))
    for key in ("strong", "prompt", "forced_base"):
        total = [0] * len(ids)
        counts = []
        for m in models:
            n, delta = matched(ids, data[m], key)
            counts.append(n)
            total = [x + y for x, y in zip(total, delta)]
        print(json.dumps(dict(models=models, control=key, common_questions=len(ids),
                              matched_answers=counts, paired_net=mp.ps.ci(total))))


def main():
    data = {m: load_model(m) for m in ALL}
    for model, rows in data.items():
        report(data, (model,))
        # Both sides keep K actual answers. Give the competitor the best K and
        # N1 the worst K by gold verdict. A positive bound holds for any filter
        # of the two fixed pools on this observed sample.
        n1 = sorted(r["n1"][0] for r in rows.values() if r["n1"][1])
        c = sorted((r["single_c"][0] for r in rows.values() if r["single_c"][1]), reverse=True)
        k = min(len(n1), len(c))
        print(json.dumps(dict(model=model, control="single_C_ideal_bound", common_answers=k,
                              lower_bound=sum(n1[:k]) - sum(c[:k]))))
    report(data, MAIN)
    report(data, ALL)


if __name__ == "__main__":
    main()
