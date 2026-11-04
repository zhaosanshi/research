"""TruthfulQA as true/false statements (the ITI / TruthX style statement-level use of TruthfulQA).

For every question we build one TRUE statement from the best answer and one FALSE statement from the
first incorrect answer, phrased as a declarative sentence so the generic statement prompt applies:

    The answer to the question "<question>" is: <answer>

pair_id = the TruthfulQA question id, so pairs stay together across splits (unit = pair).
Questions whose best answer is a non-answer ("I have no comment", "Nothing happens", ...) are kept:
they are exactly the misconception items where the model's knowledge and its output may diverge.
Built on top of the `truthfulqa` adapter (same raw source), so no new download.
"""
from . import truthfulqa
from .base import record


def build(cfg, raw_dir):
    qa_recs = truthfulqa.build(cfg, raw_dir)
    out = []
    for r in qa_recs:
        q = r["text"].strip().rstrip("?").strip() + "?"
        best = (r["meta"].get("best_answer") or (r["gold"][0] if r["gold"] else "")).strip()
        wrong = [a for a in r.get("incorrect", []) if a.strip()]
        if not best or not wrong:
            continue
        template = cfg.get("template", 'The answer to the question "{q}" is: {a}')
        pid = r["id"]
        meta = {"category": r["meta"].get("category"), "type": r["meta"].get("type"), "question": q}
        out.append(record(pid + ":T", "tf", "truthfulqa_tf", template.format(q=q, a=best.rstrip(".") + "."), label=True,
                          pair_id=pid, meta=dict(meta, answer=best)))
        out.append(record(pid + ":F", "tf", "truthfulqa_tf", template.format(q=q, a=wrong[0].rstrip(".") + "."), label=False,
                          pair_id=pid, meta=dict(meta, answer=wrong[0])))
    return out
