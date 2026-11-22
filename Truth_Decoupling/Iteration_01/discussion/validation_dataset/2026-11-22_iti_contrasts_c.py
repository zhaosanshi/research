"""CPU-only (no judging or model). Run from Truth_Decoupling in the scoring container.
Place next to 2026-11-22_iti_replay_b.py. Unit: right minus wrong, rescoring each question.
"""
import runpy
import sys
from pathlib import Path

PREVIOUS = Path(__file__).with_name("2026-11-22_iti_replay_b.py")
previous = runpy.run_path(str(PREVIOUS))
archive, load_iti, val = (previous[k] for k in ("archive", "iti_inputs", "V"))
import cap_score as math_score


def main():
    for model in ("llama3.1-8b", "gemma3-12b"):
        display, layer, strength = archive.KEY[model]
        tune, err = archive.MP.prompt_split(display, layer, 0, strength)
        if tune is None:
            raise ValueError(err)
        test, err = archive.MP.prompt_split(display, layer, 1, strength)
        if test is None:
            raise ValueError(err)
        sp, tx, _, ids = test
        _, run, _ = archive.MP.usual(display, 1)
        fit = "faithful_seed42"
        statuses, flags, choice = load_iti(model, run, fit, "endq", False, ids)
        n1 = archive.ML.n1_pick(tune[0], tune[1], strength)
        for policy, cfg in (("E", ("E", strength)), ("N1", n1)):
            states, issued, _ = archive.S.system(sp, tx, cfg)
            idx = [i for i in range(len(ids)) if issued[i] and flags[i]]
            a, b = [val[states[i]] for i in idx], [val[statuses[i]] for i in idx]
            print(model, "both issued", policy, len(idx), sum(a), sum(b),
                  archive.ps.ci([x - y for x, y in zip(a, b)]), flush=True)

        from iti_gsm8k import read, IDS, ROOT
        ids = read(IDS)["ids"]
        gold = math_score.gold_of("gsm8k")
        def scores(path):
            data = read(path)
            if not data["completed"] or [r["id"] for r in data["rows"]] != ids:
                raise ValueError(path)
            return [val[math_score.judge_row("gsm8k", r, gold(r["id"]))[0]] for r in data["rows"]]

        it = scores(Path(archive.BASE) / model / fit / "gsm8k" / (choice["stem"] + ".json"))
        for suffix in ("-at-prompt", ""):
            filename = f"neutral--E{layer}-w-p{strength}{suffix}.json"
            e = scores(Path(ROOT) / model / "b8" / filename)
            paired = math_score.ci([x - y for x, y in zip(e, it)])
            print(model, "GSM8K", filename, "E minus ITI, net with interval",
                  tuple(round(x * len(ids)) for x in paired), flush=True)


if __name__ == "__main__":
    main()
