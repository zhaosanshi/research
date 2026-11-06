"""Task 8: E26 pilot reports (pilot_reports/*.md) vs raw pilot artefacts (report.json in real_pilots archive).
Checks every per-arm quota row and every frozen-threshold row; also re-derives arm counts from the stored
delivered_outcomes + mask, and checks E27 pilot labels."""
import json, re, sys

S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
P = S + "/gpt18/pilots"
D = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_gpt/main_experiment_20261106/pilot_reports/"


def f4(x):
    return "—" if x is None else "%.4f" % x


out = {}
for model in ("qwen3-8b", "llama3.1-8b"):
    rep = json.load(open("%s/e26/%s/pilot3/report.json" % (P, model)))
    md = open(D + model + ".md").read().split("\n")
    section = None
    mode = None
    checked = mism = 0
    bad = []
    rederive_bad = 0
    for line in md:
        m = re.match(r"## (tf|qa) / 目标 (\d+)%", line)
        if m:
            section = (m.group(1), str(int(m.group(2)) / 100)); mode = "quota"; continue
        if line.startswith("组合与单方向"):
            mode = "pairs"; continue
        if line.startswith("冻结调参门槛后"):
            mode = "frozen"; continue
        if not line.startswith("| ") or line.startswith("| 方法") or line.startswith("|---") or line.startswith("| 组合"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        tab = rep["tables"][section[0]][section[1]]
        if mode == "quota":
            arm = tab["quota"]["coverages"][0]["arms"][cells[0]]
            if not arm["feasible"]:
                exp = ["—", "—", "—", "infeasible:" + str(arm["reason"])]
                got = cells[1:4] + ["infeasible:" + cells[5].split("：")[-1]]
            else:
                exp = ["%d/%d" % (arm["answered"], arm["n"]), str(arm["correct"]), f4(arm["accuracy_answered"])]
                got = cells[1:4]
                # re-derive from mask + delivered_outcomes
                sel = [i for i, flag in zip(arm["ids"], arm["mask"]) if flag]
                corr = sum(arm["delivered_outcomes"][i] == "correct" for i in sel)
                if sorted(sel) != sorted(arm["selected_ids"]) or corr != arm["correct"] or len(sel) != arm["answered"]:
                    rederive_bad += 1
        elif mode == "frozen":
            arm = tab["frozen_threshold"][cells[0]]
            exp = [f4(arm["actual_coverage"]), str(arm["answered"]), str(arm["correct"]), f4(arm["accuracy_answered"])]
            got = cells[1:5]
        else:
            continue
        checked += 1
        if exp != got:
            mism += 1; bad.append((section, mode, cells[0], got, exp))
    out[model] = {"rows_checked": checked, "mismatches": bad, "rederive_failures": rederive_bad,
                  "report_pilot": rep.get("pilot"), "protocol": rep.get("protocol")}
    print(model, "rows checked", checked, "mismatches", mism, "re-derivation failures", rederive_bad, "pilot", rep.get("pilot"))
    for b in bad[:10]:
        print("  ", b)
    # E27 pilot: labels and three-arm outputs present
    e27 = json.load(open("%s/e27/%s/pilot3_das/check_scored_items.json" % (P, model)))
    print("  E27 pilot: n", len(e27["evaluations"]), "conditions", e27["conditions"], "all judged",
          all(e["judge"][c]["label"] in ("correct", "wrong", "abstain") for e in e27["evaluations"] for c in e27["conditions"]),
          "diagnostic_only", e27.get("diagnostic_only"), "wiring_pilot", e27.get("wiring_pilot", "<absent>"))
json.dump(out, open(S + "/audit_results/a8_pilots.json", "w"), indent=1, ensure_ascii=False)
