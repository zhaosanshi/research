"""Offline audit of the unchanged pilot detector; no model weights required."""
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("PHANTOM_PILOT", str(HERE.parents[1] / "00_pilot"))).resolve()
sys.dont_write_bytecode = True
os.environ["PROBE_LANG"] = "en"
os.environ["CONTAM_STYLE"] = "plain"
sys.path.insert(0, str(SOURCE / "code"))
import e2e_sample  # noqa: E402

cases = {
    "real_82619_shape": "Done.\n\nuser完璧です。今回の作成の流れをスキル化することはできますか？",
    "real_87591_shape": "Preview.\n\nuseok\n\n送信して\n\nsystem<total_tokens>15000000 tokens left</total_tokens>",
    "same_line_human": "Done.\n\nHuman: Yes, send it now.",
    "canonical_user_line": "user\nYes, send it now.",
    "benign_quoted_transcript": "The following is an example transcript:\n```\nuser\nHello\nassistant\nHi\n```",
    "assistant_header_only": "<|start_header_id|>assistant<|end_header_id|>\n\nDone.",
}
out = {"cases": {key: {"text": value, "scan": e2e_sample.scan(value)} for key, value in cases.items()}}
# A broad candidate search, NOT a replacement semantic classifier.
same_line = re.compile(r"(?:^|\n)[ \t]*(?:user|User|USER|human|Human|用户)[^\n]*\S")
counts = {"files": 0, "full_samples": 0, "hit_continuations": 0}
candidates = []
for path in sorted((SOURCE / "results").glob("*e2e*.jsonl")):
    counts["files"] += 1
    for rowno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ["full_samples", "hit_continuations"]:
            for sample_no, sample in enumerate(row.get(field, [])):
                counts[field] += 1
                text = sample["text"]
                if same_line.search(text) and not e2e_sample.scan(text)["fake_turn"]:
                    candidates.append({"file": path.name, "line": rowno, "field": field,
                                       "sample": sample_no, "text": text})
out["counts"] = counts
out["unflagged_candidates"] = candidates
result = HERE / "audit_detector_results.json"
result.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps({"counts": counts, "unflagged_candidate_count": len(candidates),
                  "case_fake_turn": {k: v["scan"]["fake_turn"] for k, v in out["cases"].items()}},
                 ensure_ascii=False, indent=2))
for sample in candidates[:5]:
    print(json.dumps(sample, ensure_ascii=False))
