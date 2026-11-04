"""TruthfulQA as a multiple-choice statement set, for MC1 / MC2-style scoring with the statement readout (E14).

For every question we build one statement per answer: the best answer and the other correct answers (label True)
and every incorrect answer (label False), all with the truthfulqa_tf template

    The answer to the question "<question>" is: <answer>

pair_id = the question id; meta.role in {best, correct, incorrect}. No calib/dev split is made here: the E14
evaluation drops the questions whose pairs sit in truthfulqa_tf calib/dev (the direction / probe estimation set).
MC1 (ITI / TruthfulQA convention): the best answer must out-score every incorrect answer.
MC2: normalised score mass on the correct answers.
"""
from . import truthfulqa
from .base import record


def build(cfg, raw_dir):
    qa_recs = truthfulqa.build(cfg, raw_dir)
    template = cfg.get("template", 'The answer to the question "{q}" is: {a}')
    out = []
    for r in qa_recs:
        q = r["text"].strip().rstrip("?").strip() + "?"
        best = (r["meta"].get("best_answer") or "").strip()
        groups = (("best", [best], True), ("correct", [a for a in r["gold"] if a.strip() != best], True), ("incorrect", r.get("incorrect", []), False))
        seen = set()
        meta = {"category": r["meta"].get("category"), "type": r["meta"].get("type"), "question": q}
        for role, answers, label in groups:
            for k, a in enumerate(answers):
                a = a.strip()
                if not a or a.lower() in seen:
                    continue
                seen.add(a.lower())
                out.append(record("%s:%s%d" % (r["id"], role[0].upper(), k), "tf", "truthfulqa_mc", template.format(q=q, a=a.rstrip(".") + "."),
                                  label=label, pair_id=r["id"], meta=dict(meta, answer=a, role=role)))
    return out
