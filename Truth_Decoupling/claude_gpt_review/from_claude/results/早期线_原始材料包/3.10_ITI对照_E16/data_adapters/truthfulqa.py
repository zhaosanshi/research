"""TruthfulQA (Lin et al., ACL 2022), 817 questions.

Primary source: Hugging Face dataset `truthfulqa/truthful_qa` (config generation, split validation,
817 rows) fetched through the datasets-server rows API (stdlib only, 100 rows per call).
Fallback: the GitHub CSV https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv,
which at the time of writing holds only 790 rows (verified 2026-10-25) -> a loud warning is printed.

Produces qa records with gold = [best_answer] + correct_answers, incorrect = incorrect_answers.
"I have no comment" counts as truthful in the original metric; we report abstention separately
(see 文献/notes/2109.07958_truthfulqa.md).
"""
import os
import sys

from .base import download, read_csv, read_json, read_jsonl, record, write_jsonl

ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset=truthfulqa%2Ftruthful_qa"
            "&config=generation&split=validation&offset={offset}&length=100")
CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
EXPECTED_N = 817


def _fetch_hf_rows(raw_dir):
    dest = os.path.join(raw_dir, "truthfulqa", "truthful_qa_generation_validation.jsonl")
    if os.path.exists(dest):
        return dest
    rows, offset = [], 0
    while True:
        tmp = download(ROWS_API.format(offset=offset), os.path.join(raw_dir, "truthfulqa", "_rows_%d.json" % offset), overwrite=True)
        batch = [r["row"] for r in read_json(tmp)["rows"]]
        os.remove(tmp)
        rows.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
    if len(rows) != EXPECTED_N:
        raise ValueError("HF rows API returned %d rows, expected %d" % (len(rows), EXPECTED_N))
    write_jsonl(rows, dest)
    return dest


def _split(s):
    return [x.strip() for x in (s or "").split(";") if x.strip()]


def build(cfg, raw_dir):
    src = cfg.get("source", {})
    out = []
    if src.get("prefer", "hf") == "hf":
        try:
            path = _fetch_hf_rows(raw_dir)
            for i, r in enumerate(read_jsonl(path)):
                best = (r.get("best_answer") or "").strip()
                correct = [a.strip() for a in r.get("correct_answers", []) if a.strip()]
                out.append(record("tqa:%d" % i, "qa", "truthfulqa", r["question"].strip(),
                                  gold=[best] + [a for a in correct if a != best],
                                  incorrect=[a.strip() for a in r.get("incorrect_answers", []) if a.strip()],
                                  meta={"type": r.get("type"), "category": r.get("category"), "source": r.get("source"),
                                        "best_answer": best, "source_used": "hf_rows_api"}))
            return out
        except Exception as e:  # noqa: BLE001
            print("  WARNING truthfulqa: HF rows API failed (%r); falling back to GitHub CSV (790 rows, not 817)" % (e,), file=sys.stderr)
    path = download(src.get("url", CSV_URL), os.path.join(raw_dir, "truthfulqa", "TruthfulQA.csv"))
    rows = read_csv(path)
    if len(rows) != EXPECTED_N:
        print("  WARNING truthfulqa: CSV has %d rows, official set has %d" % (len(rows), EXPECTED_N), file=sys.stderr)
    for i, r in enumerate(rows):
        best = r["Best Answer"].strip()
        out.append(record("tqa:%d" % i, "qa", "truthfulqa", r["Question"].strip(),
                          gold=[best] + [a for a in _split(r["Correct Answers"]) if a != best],
                          incorrect=_split(r["Incorrect Answers"]),
                          meta={"type": r.get("Type"), "category": r.get("Category"), "source": r.get("Source"),
                                "best_answer": best, "source_used": "github_csv"}))
    return out
