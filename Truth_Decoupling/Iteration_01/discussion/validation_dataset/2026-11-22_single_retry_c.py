"""CPU replay from Truth_Decoupling root. Use tqa on the data host, qa2 in the scoring container.
The optional symmetric veto rejects a retry that repeats an issued no-write answer;
both sides then receive the same original-answer read. No model or judge call.
Both systems keep ordinary E answers, then generate once for every E refusal.
"""
import argparse
import math
from pathlib import Path
import sys

MODELS = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
sys.path.insert(0, str(Path("Iteration_01/diag").resolve()))


def truthful(model):
    import tq_same_n_decomp as t
    display, layer, strength = t.A.KEY[model]
    split, error = t.MP.prompt_split(display, layer, 1, strength)
    if error:
        raise ValueError(error)
    sp, tx, _, ids = split
    source, = [s for s in t.ML.sources(display)[2] if s[0] == layer]
    source_ids = set(source[3].qs)
    ids = [q for q in ids if q in source_ids]
    if model == MODELS[0]:
        tune, err = t.MP.prompt_split(display, layer, 0, strength)
        if err or ids != sp.qs or list(tune[0].qs) != list(source[1].qs):
            raise ValueError("Llama cost example roster differs from the main table")
        cfg = t.ML.n1_pick(tune[0], tune[1], strength)
        records = t.MP.fr.records(sp, cfg, tx)
        n = len(records)
        blanks = sum(r["cand"] is not None for r in records)
        with_cached_origin = sum(t.S.n1_generations(records, cfg[2], cfg[3]))
        print(model, "full policy calls on native test roster, N/E-blanks/C-attempts/with-cache/without-cache/E-fill",
              n, blanks, with_cached_origin - n, with_cached_origin, with_cached_origin + blanks, n + blanks)
    _, run, _ = t.MP.usual(display, 1)
    _, error = t.MP.strong_prompt(run, layer, strength, ids, 0)
    if error:
        raise ValueError(error)
    forced = t.forced(run, layer, strength, ids)
    e, c = sp.cE(strength), sp.cEC(strength, "0-25")
    forced_text = t.MP.fr.raw_texts(t.MP.work(run, "test", f"pospromptdl{layer}"))
    forced_key = f"E_L{layer}_conf-w-p{strength}"
    original = {q: t.MP.ro.norm(tx.ec[q, "baseline_x"]) for q in ids if not sp.blank[q, "baseline_x"]}
    return {q: (t.V[sp.st[q, e]], not sp.blank[q, e], t.V[sp.st[q, c]], not sp.blank[q, c],
                *forced[q], q in original and t.MP.ro.norm(tx.ec[q, c]) == original[q],
                q in original and t.MP.ro.norm(forced_text[q, forced_key]) == original[q]) for q in ids}


def presuppositions(model):
    import json
    import qaqa_pilot_table as t
    root = Path(t.S.NS)
    bank = root / "bank/qaqa_full_bank_20261122.jsonl"
    pilot = set(json.loads((root / "splits/qaqa_pilot_20261122.json").read_text())["ids"])
    r = t.Raw(model, "qaqafull", str(bank), ("test_full570", "test_full570_tuned", "test_full570_strong"))
    if len(pilot) != 100 or len(r.ids) != len(set(r.ids)) or len(r.ids) != 570 or not pilot.issubset(r.ids):
        raise ValueError("bank or pilot roster")
    s = t.SET[model][1]
    e, c = r.e(s), r.ec(s, "0.25")
    f = "conf:" + e.split(":")[1]
    for arm in ("x", e, c, f):
        r.need(arm)
    conf = r.confidence("qaqafull", f)
    original = {q: t.ro.norm(r.text("x", q)) for q in r.ids if not r.blank("x", q)}
    return {q: (r.cell(e, q), not r.blank(e, q), r.cell(c, q), not r.blank(c, q),
                r.cell(f, q), not r.blank(f, q), conf[q],
                q in original and t.ro.norm(r.text(c, q)) == original[q],
                q in original and t.ro.norm(r.text(f, q)) == original[q]) for q in r.ids if q not in pilot}


def evaluate(rows, ids, veto=False):
    blanks = [q for q in ids if not rows[q][1]]
    issued = {q for q in blanks if rows[q][3] and not (veto and rows[q][7])}
    eligible = [q for q in blanks if rows[q][5] and not (veto and rows[q][8])]
    if any(rows[q][6] is None or not math.isfinite(rows[q][6]) for q in eligible):
        raise ValueError("an issued forced answer lacks finite confidence")
    if len(eligible) < len(issued):
        raise ValueError("matched count unreachable")
    picked = set(sorted(eligible, key=lambda q: -rows[q][6])[:len(issued)])
    paired = issued.intersection(eligible)
    out = {}
    for q in ids:
        e, _, c, _, f, _, _ = rows[q][:7]
        cs = c if q in issued else e
        fs = f if q in picked else e
        out[q] = (cs, fs, cs - fs, int(q in issued), int(q in picked), int(q in paired),
                  c - f if q in paired else 0)
    return out, len(ids) + (1 + int(veto)) * len(blanks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("tqa", "qa2"))
    task = parser.parse_args().task
    import policy_select2 as ps
    loader = truthful if task == "tqa" else presuppositions
    data = {m: loader(m) for m in MODELS}
    for n in (3, 5):
        names = MODELS[:n]
        common = set.intersection(*(set(data[m]) for m in names))
        ids = sorted(common) if task == "tqa" else [q for q in data[names[0]] if q in common]
        expected = (521 if n == 3 else 520) if task == "tqa" else 470
        if len(ids) != expected:
            raise ValueError("unexpected evaluation roster")
        for veto in (False, True):
            per = []
            calls = 0
            for m in names:
                out, cost = evaluate(data[m], ids, veto)
                per.append(out)
                calls += cost
                print(task, "veto", veto, n, m, "matched releases, calls/side, net difference", sum(v[3] for v in out.values()),
                      cost, sum(v[2] for v in out.values()))
            cols = [[sum(d[q][i] for d in per) for q in ids] for i in range(7)]
            if sum(cols[3]) != sum(cols[4]):
                raise ValueError("release totals differ")
            print(task, "veto", veto, n, "common questions", len(ids), "generation calls per side", calls,
                  "equal releases", sum(cols[3]), "net C / E", sum(cols[0]), sum(cols[1]),
                  "delta", ps.ci(cols[2]), "both answered on retry", sum(cols[5]),
                  "same-question text difference", ps.ci(cols[6]), flush=True)


if __name__ == "__main__":
    main()
