"""Readouts: map model outputs to labels.

Statement level: compare next-token log-probabilities of candidate label words
(True / False / optional Unknown). Each label may have several surface variants
("True", " True", "true"); the first sub-token of each variant in the prompt
context is used and the max over variants is taken. This is more robust than
parsing free generation and is the standard protocol for binary truth probes.

QA level: strict abstention regexes (rule layer); an LLM judge supplies the
second opinion (see judge.py). Rule + judge double-check is part of the plan.
"""
import math
import re

import numpy as np

DEFAULT_LABEL_VARIANTS = {
    "True": ["True", " True", "true", " true", "TRUE"],
    "False": ["False", " False", "false", " false", "FALSE"],
    "Abstain": ["Unknown", " Unknown", "unknown", " unknown", "Uncertain", " Uncertain"],
}


def first_token_ids(tokenizer, prompt_text, variants):
    """First new token id of each variant when appended to prompt_text.

    Tokenising prompt+word and diffing against prompt alone handles tokenizers
    whose word pieces depend on the preceding context (SentencePiece vs BPE).
    """
    base_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    ids = []
    for v in variants:
        full = tokenizer(prompt_text + v, add_special_tokens=False)["input_ids"]
        # find first divergence
        k = 0
        while k < min(len(base_ids), len(full)) and base_ids[k] == full[k]:
            k += 1
        if k < len(full):
            ids.append(int(full[k]))
    return sorted(set(ids))


def build_label_token_map(tokenizer, prompt_text, label_variants=None):
    label_variants = label_variants or DEFAULT_LABEL_VARIANTS
    return {lab: first_token_ids(tokenizer, prompt_text, vs) for lab, vs in label_variants.items()}


def label_logprobs_from_logits(logits_last, label_token_map, use_abstain=True):
    """logits_last: 1-D array/tensor over vocab. Returns dict label -> log-prob mass (logsumexp over variants)."""
    try:
        import torch
        if isinstance(logits_last, torch.Tensor):
            logits_last = logits_last.detach().float().cpu().numpy()
    except ImportError:
        pass
    logits_last = np.asarray(logits_last, dtype=np.float64)
    logz = _logsumexp(logits_last)
    out = {}
    for lab, ids in label_token_map.items():
        if lab == "Abstain" and not use_abstain:
            continue
        if not ids:
            out[lab] = -np.inf
            continue
        out[lab] = float(_logsumexp(logits_last[ids]) - logz)
    return out


def _logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def decide_label(label_logprobs, abstain_margin=0.0):
    """Argmax over labels; `abstain_margin` > 0 requires Abstain to beat the best
    answer by that many nats (default: plain argmax)."""
    labs = list(label_logprobs.keys())
    vals = np.array([label_logprobs[l] for l in labs])
    best = labs[int(np.argmax(vals))]
    if best == "Abstain" and abstain_margin > 0:
        others = [label_logprobs[l] for l in labs if l != "Abstain"]
        if others and label_logprobs["Abstain"] - max(others) < abstain_margin:
            best = labs[int(np.argmax([label_logprobs[l] if l != "Abstain" else -np.inf for l in labs]))]
    return best


def p_true_given_answer(label_logprobs):
    """P(True | answered) from True/False log-probs (renormalised)."""
    a, b = label_logprobs.get("True", -np.inf), label_logprobs.get("False", -np.inf)
    if a == -np.inf and b == -np.inf:
        return float("nan")
    return float(1.0 / (1.0 + math.exp(b - a))) if a != -np.inf else 0.0


# --------------------------------------------------------------------------- #
# QA abstention rules
# --------------------------------------------------------------------------- #

ABSTAIN_PATTERNS = [
    r"\bi don'?t know\b", r"\bi do not know\b", r"\bi'?m not sure\b", r"\bnot sure\b",
    r"\bi am not certain\b", r"\bi'?m not certain\b", r"\bcannot (be )?determine", r"\bcan'?t (be )?determine",
    r"\bunable to (determine|answer|verify|confirm)", r"\bno (reliable |definitive |sufficient )?information\b",
    r"\binsufficient information\b", r"\bnot enough information\b", r"\bunknown\b", r"\bunclear\b",
    r"\bi (do not|don'?t) have (enough |sufficient |that |this )?(information|knowledge|data)",
    r"\bno way to know\b", r"\bi have no (comment|information|idea)\b", r"\bcannot answer\b", r"\bcan'?t answer\b",
    r"\bnot (publicly )?(known|available|documented)\b", r"\bi'?m unable\b", r"\bi am unable\b",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)


def rule_abstains(text, max_chars=400):
    """True if the (beginning of the) answer contains an explicit abstention phrase."""
    if text is None:
        return False
    head = text.strip()[:max_chars]
    return bool(_ABSTAIN_RE.search(head))


def strict_tf_parse(text):
    """Strict parse for generated statement answers: exactly one of True/False/Unknown as first word."""
    if text is None:
        return None
    m = re.match(r"\s*\**\s*(true|false|unknown|uncertain)\b", text, re.IGNORECASE)
    if not m:
        return None
    w = m.group(1).lower()
    return {"true": "True", "false": "False", "unknown": "Abstain", "uncertain": "Abstain"}[w]
