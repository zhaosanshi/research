"""Audit the three available replacement cells. Run at Truth_Decoupling in the scoring container with CPU only."""
import collections
import json
from pathlib import Path
import sys
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import qaqa_pilot_table as Q

ROOT = Path(Q.S.NS)
PILOT = set(json.loads((ROOT / "splits/qaqa_pilot_20261122.json").read_text())["ids"])
BANK = ROOT / "bank/qaqa_full_bank_20261122.jsonl"
WORK = ("test_full570", "test_full570_tuned", "test_full570_strong")


def replay():
    validators = []
    origin = collections.defaultdict(collections.Counter)
    for model, (_, strength, *_) in Q.SET.items():
        raw = Q.Raw(model, "qaqafull", str(BANK), WORK)
        e = raw.e(strength)
        forced = "conf:" + e.split(":")[1]
        joint = raw.ec(strength, "0.25")
        honest = "conf_E2"
        pilot_run = Q.Raw(model, "qaqa", Q.TUNE_BANK, ["test_tune100"])
        options = [raw.ec(strength,k) for k in Q.LAD]
        for arm in (e,*options):
            pilot_run.need(arm)
        if set(pilot_run.ids) != PILOT:
            raise ValueError("pilot roster mismatch")
        def dev_hits(arm):
            return sum(pilot_run.blank(e,q) and not pilot_run.blank(arm,q) and pilot_run.cell(arm,q)==1 for q in pilot_run.ids)
        chosen = max(options, key=dev_hits)
        for arm in ("x","conf",chosen):
            raw.need(arm)
        cells = {}
        for arm in (e, forced, joint, honest):
            raw.need(arm)
        if len(set(raw.ids)) != 570 or len(PILOT) != 100 or not PILOT.issubset(raw.ids):
            raise ValueError("wrong evaluation bank")
        for q in raw.ids:
            records = [(not raw.blank(arm,q), raw.cell(arm,q), raw.text(arm,q).strip()) for arm in (forced, honest, joint)]
            if raw.blank(e,q):
                selected = raw.text(chosen,q).strip()
                orig = raw.text("x",q).strip()
                if raw.blank("x",q):
                    label = "original also abstained"
                elif raw.cell("x",q) == -1:
                    label = "original issued wrong"
                else:
                    label = "original issued right or neutral"
                counts = origin[("pilot" if q in PILOT else "heldout",label)]
                counts["attempts"] += 1
                verdicts = {}
                for a in (e,"x","conf",forced,chosen):
                    verdicts.setdefault(raw.text(a,q).strip(), raw.cell(a,q))
                def correct(a):
                    return not raw.blank(a,q) and verdicts[raw.text(a,q).strip()] == 1
                cc, fc = correct(chosen),correct(forced)
                counts["C only vs forced E"] += int(cc and not fc)
                counts["forced E only vs C"] += int(fc and not cc)
                counts["C only vs all three"] += int(cc and not any(correct(a) for a in (forced,"x","conf")))
                counts["C repeats issued original"] += int(not raw.blank("x",q) and not raw.blank(chosen,q) and selected==orig)
            shared = all(x[0] for x in records)
            f,h,c = records
            # Preserve archived verdicts in the main reproduction. Separately collapse identical text to the
            # first verdict in the fixed order (must-answer+E, pure prompt, ordinary E+C) as a sensitivity.
            values = [x[1] for x in records]
            cache = {}
            tied = [cache.setdefault(x[2], x[1]) for x in records]
            cells[q] = (int(shared), f[1]-h[1] if shared else 0, c[1]-f[1] if shared else 0,
                        tied[0]-tied[1] if shared else 0, tied[2]-tied[0] if shared else 0,
                        int(shared and f[2] != c[2]), int(shared and f[2] != h[2]))
        validators.append(cells)
    for label, counts in sorted(origin.items()):
        print("E blank", label,dict(counts))
    if any(list(d) != list(validators[0]) for d in validators):
        raise ValueError("model rosters differ")
    for pilot in (True, False):
        ids = [q for q in validators[0] if (q in PILOT) == pilot]
        summed = [[sum(d[q][j] for d in validators) for q in ids] for j in range(7)]
        print("pilot" if pilot else "heldout",len(ids),"shared answers",sum(summed[0]))
        for j in range(1,5):
            print(("F-H","C-F","F-H same text unified","C-F same text unified")[j-1],Q.S.joint_ci([summed[j]]))
        print("difference of the two conditional contrasts",Q.S.joint_ci([[a-b for a,b in zip(summed[1],summed[2])]]))
        print("different texts F/C, F/H",sum(summed[5]),sum(summed[6]))

if __name__ == "__main__":
    replay()
