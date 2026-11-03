"""Answer-context preparation and causally aligned teacher-forced scoring.

This module does not generate answers, fit directions, or choose hyperparameters.
Scores are summed conditional answer-token log probabilities, not official MC1.
"""
import re

import torch
import torch.nn.functional as F

from .qa_scoring import is_personal_refusal_gold


def _answer_key(text):
    """Deduplicate spelling/case/whitespace and terminal punctuation variants."""
    return " ".join(text.casefold().split()).rstrip(".!? ")


def _unusable_answer(text):
    if not isinstance(text, str):
        raise ValueError("answer candidates must be strings")
    if not text.strip():
        return "empty"
    if is_personal_refusal_gold(text):
        return "personal_refusal"
    bare = re.sub(r"[\W_]+", " ", text.casefold()).strip()
    if bare in {"unknown", "uncertain"}:
        return "bare_unknown_or_uncertain"
    return None


def _answer_list(record, key):
    value = record.get(key)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list or tuple of answer strings" % key)
    return list(value)


def prepare_e_pairs(questions):
    """Choose one substantive positive/negative answer per input question.

    The preferred positive is best_answer, then the gold list in input order.
    Personal refusals and bare Unknown/Uncertain are unsuitable in either side
    of the E contrast. Complete facts such as 'The origin is unknown' remain.
    Invalid questions are excluded with an audit; no replacement questions are
    fetched. IDs, input order, and original record fields are preserved.
    """
    retained, details, excluded, seen_ids = [], [], [], set()
    for record in questions:
        if not isinstance(record, dict):
            raise ValueError("questions must contain dictionaries")
        pid = record.get("pair_id")
        if pid is None or not str(pid).strip():
            raise ValueError("every question needs a nonempty pair_id")
        id_key = str(pid)
        if id_key in seen_ids:
            raise ValueError("duplicate pair_id: %r" % pid)
        seen_ids.add(id_key)
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question %r needs nonempty question text" % pid)

        positive_candidates = []
        if record.get("best_answer") is not None:
            positive_candidates.append(("best_answer", record["best_answer"]))
        positive_candidates += [("gold[%d]" % i, answer)
                                for i, answer in enumerate(_answer_list(record, "gold"))]
        negative_candidates = [("incorrect[%d]" % i, answer)
                               for i, answer in enumerate(_answer_list(record, "incorrect"))]

        def select(candidates, positive=None):
            chosen, chosen_source, seen, checks = None, None, set(), []
            for source, raw in candidates:
                reason = _unusable_answer(raw)
                answer = raw.strip()
                key = _answer_key(answer)
                if reason is None and key in seen:
                    reason = "duplicate"
                seen.add(key)
                if reason is None and positive is not None and key == _answer_key(positive):
                    reason = "same_as_positive"
                if reason is None and chosen is None:
                    chosen, chosen_source = answer, source
                    status = "selected"
                else:
                    status = reason or "eligible_not_selected"
                checks.append({"source": source, "answer": answer, "status": status})
            return chosen, chosen_source, checks

        positive, positive_source, positive_checks = select(positive_candidates)
        negative, negative_source, negative_checks = select(negative_candidates, positive)
        detail = {"pair_id": pid, "positive_candidates": positive_checks,
                  "negative_candidates": negative_checks}
        reasons = []
        if positive is None:
            reasons.append("no_substantive_positive")
        if negative is None:
            reasons.append("no_substantive_distinct_negative")
        if reasons:
            detail.update({"status": "excluded", "reasons": reasons})
            excluded.append({"pair_id": pid, "reasons": reasons})
        else:
            result = dict(record)
            result.update({"positive_answer": positive, "negative_answer": negative})
            retained.append(result)
            detail.update({"status": "retained", "positive_source": positive_source,
                           "negative_source": negative_source})
        details.append(detail)
    return retained, {
        "n_input": len(details), "n_retained": len(retained), "n_excluded": len(excluded),
        "retained_ids": [record["pair_id"] for record in retained],
        "excluded_ids": [item["pair_id"] for item in excluded],
        "excluded": excluded, "records": details,
    }


def _token_ids(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, (list, tuple)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("single-text tokenization must return a flat list of nonnegative token IDs")
    return list(ids)


def encode_answer(bundle, system, question, answer,
                  user_template="Question: {question}\nAnswer:"):
    """Append raw answer text directly to the rendered generation prompt.

    Prefix/full tokenizations must preserve the entire prefix exactly. A token
    spanning the prompt/answer boundary is rejected rather than assigned to an
    arbitrary side. No assistant-message terminator or EOS token is appended.
    Special IDs are allowed in the chat prompt, but not in the answer span.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a nonempty string")
    user = user_template.format(question=question)
    prefix_text = bundle.build_chat(system, user, add_generation_prompt=True)
    full_text = prefix_text + answer
    prefix_ids = _token_ids(bundle.tokenizer, prefix_text)
    full_ids = _token_ids(bundle.tokenizer, full_text)
    if not prefix_ids:
        raise ValueError("the generation prompt has no tokens")
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("prompt/answer boundary retokenizes the prefix; refusing ambiguous token positions")
    answer_ids = full_ids[len(prefix_ids):]
    if not answer_ids:
        raise ValueError("answer tokenization has no answer tokens")
    special = set(getattr(bundle.tokenizer, "all_special_ids", []) or [])
    invalid = sorted(set(answer_ids) & special)
    if invalid:
        raise ValueError("answer contains special token IDs: %s" % invalid)
    return {
        "prefix_ids": prefix_ids, "full_ids": full_ids, "answer_ids": answer_ids,
        "prompt_length": len(prefix_ids), "answer_length": len(answer_ids),
        "prefix_text": prefix_text, "full_text": full_text, "answer_text": answer,
        "text_audit": {"system": system, "question": question, "user_text": user,
                       "construction": "generation_prompt_plus_raw_answer",
                       "prefix_token_ids_unchanged": True, "appended_eos": False},
    }


def pack_answers(encoded, pad_token_id, device):
    """Left-pad encoded sequences and mark target/content vs predictor positions.

    For an unpadded prompt of P tokens and answer of M tokens, target/answer
    positions are P..P+M-1; predictors are P-1..P+M-2. The final prompt token
    therefore participates in first-answer-token prediction.
    """
    encoded = list(encoded)
    if not encoded:
        raise ValueError("cannot pack an empty answer batch")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    for item in encoded:
        p, m = item["prompt_length"], item["answer_length"]
        ids = item["full_ids"]
        if type(p) is not int or type(m) is not int or p < 1 or m < 1:
            raise ValueError("prompt and answer lengths must be positive integers")
        if len(ids) != p + m or ids != item["prefix_ids"] + item["answer_ids"]:
            raise ValueError("encoded token IDs and span lengths disagree")
        if len(item["prefix_ids"]) != p or len(item["answer_ids"]) != m:
            raise ValueError("encoded prefix/answer lengths disagree")
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("encoded token IDs must be nonnegative integers")
    batch, width = len(encoded), max(len(item["full_ids"]) for item in encoded)
    ids = torch.full((batch, width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((batch, width), dtype=torch.long, device=device)
    target = torch.zeros((batch, width), dtype=torch.bool, device=device)
    predictor = torch.zeros_like(target)
    prompt_end = torch.zeros_like(target)
    for row, item in enumerate(encoded):
        p, m = item["prompt_length"], item["answer_length"]
        start = width - (p + m)
        ids[row, start:] = torch.tensor(item["full_ids"], dtype=torch.long, device=device)
        attention[row, start:] = 1
        target[row, start + p:start + p + m] = True
        predictor[row, start + p - 1:start + p + m - 1] = True
        prompt_end[row, start + p - 1] = True
    return {
        "inputs": {"input_ids": ids, "attention_mask": attention},
        "target_mask": target, "predictor_mask": predictor,
        "prompt_end_mask": prompt_end, "answer_mask": target.clone(),
    }


def answer_logprobs(logits, packed):
    """Return [B] sums of log P(answer token | prompt, previous answer tokens).

    Only answer targets count. Prefix tokens, padding, and the position after
    the last answer do not contribute. Gradients remain available to callers.
    """
    ids = packed["inputs"]["input_ids"]
    target = packed["target_mask"]
    attention = packed["inputs"]["attention_mask"]
    if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(ids.shape):
        raise ValueError("logits must have shape [B,T,V] matching the packed inputs")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating-point")
    if target.shape != ids.shape or attention.shape != ids.shape or target.dtype != torch.bool:
        raise ValueError("packed target/attention masks must match input IDs; target mask must be boolean")
    if target[:, 0].any() or not target.any(dim=1).all() or (target & ~attention.bool()).any():
        raise ValueError("each answer needs predictable, unpadded target positions")
    shifted_mask = target[:, 1:].to(logits.device)
    targets = ids[:, 1:].to(logits.device)[shifted_mask]
    if (targets < 0).any() or (targets >= logits.shape[-1]).any():
        raise ValueError("answer target IDs are outside the logits vocabulary")
    selected = logits[:, :-1, :][shifted_mask]
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    token_logps = F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
    rows = shifted_mask.nonzero(as_tuple=False)[:, 0]
    return token_logps.new_zeros(logits.shape[0]).index_add(0, rows, token_logps)
