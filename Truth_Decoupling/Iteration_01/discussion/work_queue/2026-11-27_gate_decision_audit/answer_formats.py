"""Check only the public benchmark's user-facing answer-format instructions, with no labels."""

import json
import sys

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/capability",
    "Iteration_01/code/scripts/newsets",
]
import gate_capture_1126 as capture
import gate_fit_1126 as fit


def constrained(text):
    markers = (
        "Answer with the letter of the correct option only.",
        'At the end, write "The answer is" followed by the final number.',
    )
    return any(text.strip().endswith(m) for m in markers)


for model in ("qwen3.5-9b", "gemma3-27b"):
    config = json.loads(capture.FROZEN_MODELS.read_bytes())["models"][model]
    sets = capture.items_of(model, config)
    meta = fit.load_meta(model)
    for family, part in fit.READ:
        source = sets[family]["items"]
        indices = fit.select(meta, family, part, fit.pilot())
        assert all(
            meta[family]["keys"][i] == source[i][0]
            and meta[family]["text"][i] == capture.tsha(source[i][1])
            for i in indices
        )
        flagged = sum(constrained(source[i][1]) for i in indices)
        print(model, family, part, len(indices), "explicit answer format", flagged)
