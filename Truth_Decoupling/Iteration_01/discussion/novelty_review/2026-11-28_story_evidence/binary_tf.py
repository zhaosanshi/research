"""On the balanced 529+529 archive infer full binary right-minus-wrong from the published three-decimal Youden J.
There is exactly one possible integer count per cell, checked by re-rounding. No commitment self-report applied."""

import subprocess

base = "7cede9b:Truth_Decoupling/claude_gpt_review/from_claude/results/05_E41同作答率_固定C全层E/"
for m in ("llama3.1-8b", "qwen3-8b", "gemma3-4b"):
    doc = subprocess.check_output(
        ["git", "show", base + m + "/e41_das_analysis.md"], text=True
    )
    t = doc.split("## Primary")[0].strip().splitlines()
    lines = [l.split("|")[1:4] for l in t if l.startswith("|")][2:]
    scores = []
    for cond, a, j in lines:
        cand = [2 * k for k in range(-529, 530) if f"{k / 529:+.3f}" == j.strip()]
        assert len(cand) == 1
        scores.append(cand[0])
    print(
        m,
        "nets O/E/Cabst/Canswer/ECabst/ECanswer",
        scores,
        "EC-E",
        [s - scores[1] for s in scores[4:]],
        "EC-C",
        [s - t for s, t in zip(scores[4:], scores[2:4])],
    )
