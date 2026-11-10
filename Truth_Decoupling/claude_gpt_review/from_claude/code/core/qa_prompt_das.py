"""Prompt-source rank-one interchange training with whole-answer teacher forcing.

Plan 21 (fourth version, route "乙", GPT round 26). One frozen model, one learned
unit direction w per layer. Nothing here calls a judge or reads test labels.

Source coordinate  z_src = h_L(source prompt, last prompt token) . w
Write (clamp)      h'   = h + (z_src - h . w) w   at the base sequence's content
                   positions: first user-content token .. end of sequence (user
                   text, assistant header, answer tokens). BOS, system segment and
                   the user role header are never written. Training and inference
                   use the same position rule; at inference the write covers the
                   prefill content positions and every decode step.
Objective          mean per-token cross-entropy of the whole target under teacher
                   forcing, two directions trained together (0.5 each).
Targets            complete targets end with the model's own end-of-turn token;
                   targets clipped to the token cap get NO terminator.

Everything about which prompts/targets form a pair lives in the driver script
(scripts/e43_qa_prompt_das.py); this module only knows prompts, token spans,
source capture, the clamp hook, the training loop and clamped generation.
"""
import contextlib
import hashlib
import json
import math
import time

import torch

from .model_io import capture_hooks, forward_logits, greedy_generate
from .qa_context import answer_logprobs, pack_answers

PROTOCOL = "qa_prompt_source_clamp_wholeanswer_ce_v2"


# ----------------------------------------------------------------------------- small helpers
def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def unit(w):
    w = torch.as_tensor(w).float()
    n = float(w.norm())
    if w.ndim != 1 or not math.isfinite(n) or n <= 0:
        raise ValueError("direction must be a finite nonzero 1-D vector")
    return w / n


def _flat_ids(ids):
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, (list, tuple)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("token IDs must be a flat list of nonnegative integers")
    return list(ids)


def stop_ids(bundle):
    stops = getattr(bundle.model.generation_config, "eos_token_id", None)
    stops = stops if isinstance(stops, (list, tuple)) else [stops]
    stops = [s for s in stops if type(s) is int and s >= 0]
    if not stops:
        raise ValueError("model must declare generation stop token IDs")
    return set(stops)


def turn_terminator_ids(bundle):
    """Special token(s) the chat template emits after an assistant message.

    Rendered from the template itself (user turn + assistant turn), so it is the
    terminator the model was trained to produce, not a guessed EOS. Non-special
    trailing tokens (a bare newline in some templates) are dropped.
    """
    tok = bundle.tokenizer
    marker = "zqxjv"
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": marker}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if text.count(marker) != 1:
        raise ValueError("cannot locate the assistant content in the rendered chat template")
    suffix = text[text.index(marker) + len(marker):]
    ids = _flat_ids(tok(suffix, add_special_tokens=False)["input_ids"])
    special = set(getattr(tok, "all_special_ids", []) or [])
    ids = [i for i in ids if i in special]
    if not ids:
        raise ValueError("chat template has no special end-of-turn token after the assistant message")
    if not set(ids) <= stop_ids(bundle):
        raise ValueError("end-of-turn token(s) %s are not generation stop IDs %s" % (ids, sorted(stop_ids(bundle))))
    return ids


# ----------------------------------------------------------------------------- prompts and spans
def render_prompt(bundle, system, user_template, question):
    """Chat prompt (with generation header) plus the index of the first user-content token."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    user_text = user_template.format(question=question)
    prefix_text = bundle.build_chat(system, user_text, add_generation_prompt=True)
    if prefix_text.count(user_text) != 1:
        raise ValueError("user text must occur exactly once in the rendered prompt")
    if not getattr(bundle.tokenizer, "is_fast", False):
        raise ValueError("content-position mapping needs a fast tokenizer with offset mappings")
    enc = bundle.tokenizer(prefix_text, add_special_tokens=False, return_offsets_mapping=True)
    prefix_ids = _flat_ids(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if len(offsets) != len(prefix_ids):
        raise ValueError("offset mapping does not align with token IDs")
    char = prefix_text.index(user_text)
    content_start = None
    for i, (a, b) in enumerate(offsets):
        if a <= char < b or a == char:
            content_start = i
            break
    if content_start is None or content_start >= len(prefix_ids):
        raise ValueError("cannot map the user-content start to a token")
    boundary_clean = offsets[content_start][0] == char
    first_token = bundle.tokenizer.decode(prefix_ids[content_start:content_start + 1])
    return {
        "system": system, "user_text": user_text, "prefix_text": prefix_text, "prefix_ids": prefix_ids,
        "prompt_length": len(prefix_ids), "content_start": content_start, "boundary_clean": boundary_clean,
        "content_first_token": first_token, "read_position": len(prefix_ids) - 1,
        "read_token": bundle.tokenizer.decode(prefix_ids[-1:]),
    }


def build_target(bundle, prompt, *, answer_text=None, answer_ids=None, finished=True,
                 max_target_tokens=24, eot_ids=None):
    """Attach a target (gold text or frozen own-answer IDs) to a rendered prompt.

    Returns an encoded dict for pack_answers plus content_start / truncation audit.
    Text answers are re-tokenized as prompt + raw answer; a token spanning the
    boundary is rejected. Frozen answers are appended as their generated IDs.
    Complete targets end with the end-of-turn token(s); clipped ones do not.
    """
    if (answer_text is None) == (answer_ids is None):
        raise ValueError("give exactly one of answer_text or answer_ids")
    if type(max_target_tokens) is not int or max_target_tokens < 1:
        raise ValueError("max_target_tokens must be a positive integer")
    eot_ids = _flat_ids(eot_ids or [])
    if not eot_ids:
        raise ValueError("build_target needs the chat end-of-turn token IDs")
    tok = bundle.tokenizer
    special = set(getattr(tok, "all_special_ids", []) or [])
    prefix_ids = list(prompt["prefix_ids"])
    if answer_text is not None:
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError("answer text must be nonempty")
        full_ids = _flat_ids(tok(prompt["prefix_text"] + answer_text, add_special_tokens=False)["input_ids"])
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("prompt/answer boundary retokenizes the prefix")
        body = full_ids[len(prefix_ids):]
        if not body:
            raise ValueError("answer tokenization produced no tokens")
        source_kind = "text"
    else:
        body = _flat_ids(answer_ids)
        if not body:
            raise ValueError("empty frozen answer cannot be a target")
        source_kind = "ids"
    if set(body) & special:
        raise ValueError("answer span contains special token IDs")
    truncated = len(body) > max_target_tokens
    body = body[:max_target_tokens]
    complete = bool(finished) and not truncated
    target_ids = body + (eot_ids if complete else [])
    return {
        "prefix_ids": prefix_ids, "answer_ids": target_ids, "full_ids": prefix_ids + target_ids,
        "prompt_length": len(prefix_ids), "answer_length": len(target_ids),
        "content_start": prompt["content_start"], "answer_body_tokens": len(body),
        "truncated": truncated, "terminator_appended": complete, "source_kind": source_kind,
        "answer_text": answer_text if answer_text is not None else tok.decode(body, skip_special_tokens=True),
    }


def pack_with_content_mask(encoded, pad_token_id, device):
    """pack_answers plus a boolean [B, T] mask of the content positions (user content .. end)."""
    packed = pack_answers(encoded, pad_token_id, device)
    width = packed["inputs"]["input_ids"].shape[1]
    content = torch.zeros_like(packed["target_mask"])
    for row, item in enumerate(encoded):
        start = width - (item["prompt_length"] + item["answer_length"])
        c = item["content_start"]
        if type(c) is not int or not 0 <= c < item["prompt_length"]:
            raise ValueError("content_start must index a prompt token")
        content[row, start + c:] = True
    if (content & ~packed["inputs"]["attention_mask"].bool()).any():
        raise RuntimeError("content mask touches padding")
    if not (packed["target_mask"] & ~content).sum() == 0:
        raise RuntimeError("every target position must be a content position")
    packed["content_mask"] = content
    return packed


def prompt_content_mask(attention_mask, prompt_lengths, content_starts):
    """Content mask for a left-padded prompt-only batch (prefill of generation)."""
    B, W = attention_mask.shape
    mask = torch.zeros((B, W), dtype=torch.bool, device=attention_mask.device)
    for b in range(B):
        p, c = int(prompt_lengths[b]), int(content_starts[b])
        if int(attention_mask[b].sum()) != p or not 0 <= c < p:
            raise ValueError("prompt length / content start disagree with the attention mask")
        mask[b, W - p + c:] = True
    return mask


# ----------------------------------------------------------------------------- source capture
def capture_prompt_end(bundle, texts, layer, batch_size=8):
    """[N, hidden] float32 block outputs of `layer` at the last prompt token (left padding)."""
    acts = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = bundle.encode(chunk)
        with capture_hooks(bundle, [layer], ("resid",), position="last") as store:
            forward_logits(bundle, inputs, grad=False)
        h = store[("resid", layer)]
        if not torch.isfinite(h).all():
            raise RuntimeError("nonfinite source activation at layer %d" % layer)
        acts.append(h)
    return torch.cat(acts, dim=0)


# ----------------------------------------------------------------------------- the clamp hook
@torch.no_grad()
def _stored_norm_matched_random(before, reference, random_direction):
    """Match a fixed random axis to the reference's *stored* norm at this state.

    This borrows only the learned edit's scalar size, not its orientation. In
    reduced precision, equal intended magnitudes need not survive casting
    equally. Use the existing project's declared cast tolerance; a discrete
    rounding grid that cannot realize that tolerance is an explicit failure.
    """
    b = before.float()
    r = unit(random_direction).to(b.device)
    wanted = (reference.float() - b).norm(dim=-1)

    def candidate(amplitude):
        stored = (b + amplitude[:, None] * r).to(before.dtype)
        return stored, (stored.float() - b).norm(dim=-1)

    low = torch.zeros_like(wanted)
    high = 2 * wanted
    _, upper_norm = candidate(high)
    for _ in range(12):
        grow = upper_norm < wanted
        if not bool(grow.any()):
            break
        high = torch.where(grow, high * 2, high)
        _, upper_norm = candidate(high)
    if bool((upper_norm < wanted).any()):
        raise RuntimeError("could not bracket stored random write magnitude")
    for _ in range(28):
        mid = (low + high) / 2
        _, actual = candidate(mid)
        below = actual < wanted
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    lv, ln = candidate(low)
    hv, hn = candidate(high)
    use_low = (ln - wanted).abs() <= (hn - wanted).abs()
    stored = torch.where(use_low[:, None], lv, hv)
    actual = (stored.float() - b).norm(dim=-1)
    tolerance = .01 * wanted + .0001 * b.norm(dim=-1)
    if not bool(torch.isfinite(stored).all()) or bool(((actual - wanted).abs() > tolerance + 1e-7).any()):
        raise RuntimeError("stored random write norm cannot match the reference within declared cast tolerance")
    return stored


class ClampHook:
    """Forward hook: pin the coordinate along `direction` to a target at selected positions.

    direction       [hidden] tensor or Parameter; normalized inside (gradients flow).
    targets         [B] tensor of target coordinates, or None when source_hidden is given.
    source_hidden   [B, hidden] detached source states; targets = source_hidden . u.
    mask            [B, T] bool for a forward whose sequence width is T (training, or the
                    prefill of generation). None selects every position.
    decode_all      when a later call has T == 1 (decode step), write every row.
    random_direction  if given, match the actual stored clamp magnitude at the same
                    current state along this fixed unit vector (within cast tolerance).
    """

    def __init__(self, direction, *, targets=None, source_hidden=None, mask=None, decode_all=True,
                 random_direction=None, record=True):
        if (targets is None) == (source_hidden is None):
            raise ValueError("give exactly one of targets or source_hidden")
        self.direction = direction
        self.targets = None if targets is None else torch.as_tensor(targets).float().reshape(-1)
        self.source_hidden = None if source_hidden is None else torch.as_tensor(source_hidden).detach().float()
        self.mask = None if mask is None else torch.as_tensor(mask).bool()
        self.decode_all = bool(decode_all)
        self.random_direction = None if random_direction is None else unit(random_direction)
        self.record = record
        self.active_rows = None     # generation updates this before each decode forward
        self.calls = 0
        self.applied = 0
        self.prefill = None          # per-row stats of the first (masked) call
        self.decode_gap_abs = []     # per decode step: [B] |target - z|
        self.decode_rel = []         # per decode step: [B] |delta| / |h|
        self.decode_stored_norm = []
        self.decode_reference_norm = []

    def _u(self, device):
        d = self.direction
        if not torch.is_tensor(d):
            d = torch.as_tensor(d)
        d = d.to(device)
        if d.dtype != torch.float32:
            d = d.float()
        n = d.norm()
        if d.ndim != 1 or not bool(torch.isfinite(d).all()) or not bool(torch.isfinite(n)) or float(n.detach()) <= 0:
            raise ValueError("clamp direction must be a finite nonzero vector")
        return d / n

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.ndim != 3:
            raise ValueError("layer output must be [batch, sequence, hidden]")
        B, T, _ = h.shape
        self.calls += 1
        if self.mask is not None and T == self.mask.shape[1] and B == self.mask.shape[0]:
            sel = self.mask.to(h.device)
            stage = "masked"
        elif self.mask is not None and T == 1 and self.decode_all:
            sel = torch.ones((B, 1), dtype=torch.bool, device=h.device)
            stage = "decode"
        elif self.mask is None:
            sel = torch.ones((B, T), dtype=torch.bool, device=h.device)
            stage = "all"
        else:
            raise RuntimeError("clamp hook saw sequence width %d; mask width %d" % (T, self.mask.shape[1]))
        if stage == "decode" and self.active_rows is not None:
            active = torch.as_tensor(self.active_rows, device=h.device, dtype=torch.bool)
            if active.shape != (B,):
                raise ValueError("active generation rows do not match the hidden batch")
            sel = sel & active[:, None]
        if not bool(sel.any()):
            return out
        u = self._u(h.device)
        hf = h.float()
        z = hf @ u                                              # [B, T]
        if self.source_hidden is not None:
            zt = self.source_hidden.to(h.device) @ u            # [B]
        else:
            zt = self.targets.to(h.device)
        if zt.numel() == 1:
            zt = zt.expand(B)
        if zt.numel() != B:
            raise ValueError("need one clamp target per batch row")
        if not bool(torch.isfinite(zt).all()) or not bool(torch.isfinite(h[sel]).all()):
            raise ValueError("clamp target and selected states must be finite")
        gap = torch.where(sel, zt[:, None] - z, torch.zeros_like(z))
        delta = gap[..., None] * u
        reference = torch.where(sel[..., None], (hf + delta).to(h.dtype), h)
        h_new = reference
        if self.random_direction is not None:
            if torch.is_grad_enabled() and (u.requires_grad or h.requires_grad):
                raise ValueError("norm-matched random control is for evaluation, not direction training")
            selected = _stored_norm_matched_random(h[sel], reference[sel], self.random_direction)
            h_new = h.clone()
            h_new[sel] = selected
        self.applied += 1
        if self.record:
            with torch.no_grad():
                n_sel = sel.sum(1).clamp_min(1).float()
                gap_abs = (gap.abs() * sel).sum(1) / n_sel
                stored_norm = (h_new.float() - hf).norm(dim=-1)
                reference_norm = (reference.float() - hf).norm(dim=-1)
                rel = torch.where(sel, stored_norm / hf.norm(dim=-1).clamp_min(1e-12), 0).sum(1) / n_sel
                stored_mean = torch.where(sel, stored_norm, 0).sum(1) / n_sel
                reference_mean = torch.where(sel, reference_norm, 0).sum(1) / n_sel
                if stage in ("masked", "all") and self.prefill is None:
                    self.prefill = {"gap_abs": gap_abs.cpu(), "rel": rel.cpu(),
                                    "stored_norm": stored_mean.cpu(), "reference_stored_norm": reference_mean.cpu(),
                                    "positions": sel.sum(1).cpu(), "sequence_width": T}
                elif stage == "decode":
                    self.decode_gap_abs.append(gap_abs.cpu())
                    self.decode_rel.append(rel.cpu())
                    self.decode_stored_norm.append(stored_mean.cpu())
                    self.decode_reference_norm.append(reference_mean.cpu())
        if isinstance(out, tuple):
            return (h_new,) + tuple(out[1:])
        return h_new

    def stats(self):
        result = {"calls": self.calls, "applied": self.applied,
                  "write_measurement": "stored_delta_after_residual_dtype_roundtrip"}
        if self.random_direction is not None:
            result.update(random_match_scope="reference clamp at the same current state; trajectories may differ",
                          learned_direction_used_for_size_only=True,
                          fully_information_free_random_control=False)
        if self.prefill is not None:
            result["prefill"] = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in self.prefill.items()}
        if self.decode_gap_abs:
            result["decode_steps"] = len(self.decode_gap_abs)
            result["decode_gap_abs_steps"] = torch.stack(self.decode_gap_abs).tolist()   # [steps][B]
            result["decode_rel_steps"] = torch.stack(self.decode_rel).tolist()
            result["decode_stored_norm_steps"] = torch.stack(self.decode_stored_norm).tolist()
            result["decode_reference_stored_norm_steps"] = torch.stack(self.decode_reference_norm).tolist()
        return result


@contextlib.contextmanager
def clamp(bundle, layer, hook):
    handle = bundle.layer(layer).register_forward_hook(hook)
    try:
        yield hook
    finally:
        handle.remove()


# ----------------------------------------------------------------------------- training
def teacher_forced_loss(bundle, layer, w, items, source_hidden):
    """Mean per-token CE over `items` (encoded base sequences) with the clamp installed.

    items: list of encoded targets (same device batch); source_hidden: [len(items), hidden].
    Returns (loss scalar, per-row mean CE tensor, hook stats).
    """
    packed = pack_with_content_mask(items, bundle.tokenizer.pad_token_id, bundle.device)
    hook = ClampHook(w, source_hidden=source_hidden, mask=packed["content_mask"], decode_all=False)
    with clamp(bundle, layer, hook):
        logits = forward_logits(bundle, packed["inputs"], grad=torch.is_tensor(w) and w.requires_grad)
    if hook.applied != 1:
        raise RuntimeError("clamp hook must fire exactly once per teacher-forced forward")
    logps = answer_logprobs(logits, packed)
    n_tokens = packed["target_mask"].sum(1).float()
    per_row = -logps / n_tokens
    return per_row.mean(), per_row.detach(), hook.stats()


def train_direction(bundle, layer, pairs, *, epochs=10, lr=1e-3, clip=1.0, seed=42, log=None):
    """Learn one unit direction at `layer` from question pairs.

    pairs: list of {"pair_id", "plus": {"encoded", "source_hidden"}, "minus": {...}}.
    One optimizer step per question per epoch on the two-row batch [plus, minus],
    loss = 0.5 * (CE_plus + CE_minus). Adam, gradient clipping, w renormalized
    after every step. The model is frozen; only w has gradients.
    """
    if not pairs:
        raise ValueError("training needs at least one pair")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in (lr, clip)):
        raise ValueError("learning rate and clipping norm must be finite and positive")
    if bundle.model.training or any(p.requires_grad for p in bundle.model.parameters()):
        raise ValueError("DAS direction training requires an eval-mode frozen model")
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + int(layer))
    w0 = unit(torch.randn(bundle.hidden_size, generator=g))
    w = torch.nn.Parameter(w0.to(bundle.device))
    opt = torch.optim.Adam([w], lr=lr)
    if torch.cuda.is_available():
        # On shared GPUs with little free memory, blocks the caching allocator kept from the source capture
        # (batched prefill logits) can starve cuBLAS handle creation in autograd's backward threads
        # (CUBLAS_STATUS_ALLOC_FAILED, 11-10 13:08 first full run). Return them before the first backward.
        torch.cuda.empty_cache()
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(pairs), generator=g).tolist()
        sum_plus = sum_minus = 0.0
        grad_norms = []
        for idx in order:
            pair = pairs[idx]
            items = [pair["plus"]["encoded"], pair["minus"]["encoded"]]
            src = torch.stack([torch.as_tensor(pair["plus"]["source_hidden"]).float(),
                               torch.as_tensor(pair["minus"]["source_hidden"]).float()]).to(bundle.device)
            opt.zero_grad(set_to_none=True)
            loss, per_row, _ = teacher_forced_loss(bundle, layer, w, items, src)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss at layer %d" % layer)
            loss.backward()
            if w.grad is None or not torch.isfinite(w.grad).all():
                raise RuntimeError("direction received no finite gradient at layer %d" % layer)
            gn = float(torch.nn.utils.clip_grad_norm_([w], clip))
            if gn == 0.0:
                raise RuntimeError("zero gradient at layer %d: the write has no path to the target logits" % layer)
            grad_norms.append(gn)
            opt.step()
            with torch.no_grad():
                w.data = w.data / w.data.norm()
            sum_plus += float(per_row[0]); sum_minus += float(per_row[1])
        history.append({"epoch": epoch + 1, "ce_plus": sum_plus / len(pairs), "ce_minus": sum_minus / len(pairs),
                        "grad_norm_mean": sum(grad_norms) / len(grad_norms), "grad_norm_max": max(grad_norms),
                        "seconds": time.time() - t0})
        if log is not None:
            log(json.dumps({"layer": layer, **history[-1]}, ensure_ascii=False))
    return w.detach().float().cpu(), {"history": history, "seed": seed, "lr": lr, "clip": clip, "epochs": epochs,
                                      "n_pairs": len(pairs), "init_sha256": fingerprint(w0.tolist()),
                                      "seconds": time.time() - t0}


@torch.no_grad()
def baseline_ce(bundle, layer, w, pairs):
    """Per-pair CE with the trained direction, for the record (no gradients)."""
    rows = []
    for pair in pairs:
        items = [pair["plus"]["encoded"], pair["minus"]["encoded"]]
        src = torch.stack([torch.as_tensor(pair["plus"]["source_hidden"]).float(),
                           torch.as_tensor(pair["minus"]["source_hidden"]).float()]).to(bundle.device)
        _, per_row, stats = teacher_forced_loss(bundle, layer, w.to(bundle.device), items, src)
        rows.append({"pair_id": pair["pair_id"], "ce_plus": float(per_row[0]), "ce_minus": float(per_row[1]),
                     "write_rel_mean": stats["prefill"]["rel"]})
    return rows


# ----------------------------------------------------------------------------- endpoints / controls
def coordinates(source_hidden, w):
    u = unit(w)
    z = torch.as_tensor(source_hidden).float() @ u
    return {"mean": float(z.mean()), "std": float(z.std(unbiased=False)) if z.numel() > 1 else 0.0,
            "min": float(z.min()), "max": float(z.max()), "values": z.tolist()}


def mean_difference_direction(h_source, h_base):
    diff = torch.as_tensor(h_source).float().mean(0) - torch.as_tensor(h_base).float().mean(0)
    return unit(diff)


def fixed_random_direction(hidden_size, seed, layer):
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 7919 * int(layer))
    return unit(torch.randn(hidden_size, generator=g))


# ----------------------------------------------------------------------------- clamped generation
def _cut_new_tokens(row_ids, stops):
    """(content ids before the first stop token, finished flag, the observed stop id or None)."""
    ids = _flat_ids(row_ids)
    for i, t in enumerate(ids):
        if t in stops:
            return ids[:i], True, t
    return ids, False, None


@torch.no_grad()
def generate_prompts(bundle, prompts, *, max_new_tokens, batch_size=8, layer=None, direction=None,
                     targets=None, random_direction=None):
    """Greedy generation for rendered prompts, optionally with the clamp installed.

    targets: None (no write), a float (shared endpoint) or a list with one value per prompt.
    Returns one record per prompt: text, new token IDs, finished flag, write stats.
    """
    stops = stop_ids(bundle)
    results = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        inputs = bundle.encode([p["prefix_text"] for p in chunk])
        for b, p in enumerate(chunk):
            got = inputs["input_ids"][b][inputs["attention_mask"][b].bool()].tolist()
            if got != p["prefix_ids"]:
                raise RuntimeError("batched prompt tokens differ from the single-prompt rendering")
        hook = None
        if targets is not None:
            if layer is None or direction is None:
                raise ValueError("clamped generation needs layer and direction")
            if isinstance(targets, (int, float)):
                t = torch.full((len(chunk),), float(targets))
            else:
                t = torch.as_tensor([targets[i + b] for b in range(len(chunk))]).float()
            mask = prompt_content_mask(inputs["attention_mask"], [p["prompt_length"] for p in chunk],
                                       [p["content_start"] for p in chunk])
            hook = ClampHook(direction, targets=t, mask=mask, decode_all=True, random_direction=random_direction)
        if hook is None:
            texts, out = greedy_generate(bundle, inputs, max_new_tokens)
            stats = None
        else:
            active = torch.ones(len(chunk), dtype=torch.bool, device=bundle.device)
            def track_active_rows(model, args, kwargs):
                # The actual previous generated token (after generation's logits
                # processors), not raw argmax logits, tells us which rows ended.
                ids = kwargs.get("input_ids", args[0] if args else None)
                if hook.calls and ids is not None:
                    if ids.ndim != 2 or ids.shape[0] != len(chunk):
                        raise ValueError("clamped generation requires an unexpanded greedy batch")
                    stopped = torch.zeros(len(chunk), dtype=torch.bool, device=ids.device)
                    for token in stops:
                        stopped |= ids[:, -1] == token
                    active.logical_and_(~stopped.to(active.device))
                    hook.active_rows = active
            pre = bundle.model.register_forward_pre_hook(track_active_rows, with_kwargs=True)
            try:
                with clamp(bundle, layer, hook):
                    texts, out = greedy_generate(bundle, inputs, max_new_tokens)
            finally:
                pre.remove()
            if hook.prefill is None or hook.prefill["sequence_width"] != inputs["input_ids"].shape[1]:
                raise RuntimeError("the clamp did not fire on the prefill")
            stats = hook.stats()
        prompt_len = inputs["input_ids"].shape[1]
        for b, p in enumerate(chunk):
            ids, finished, stop_id = _cut_new_tokens(out[b][prompt_len:], stops)
            rec = {"text": bundle.tokenizer.decode(ids, skip_special_tokens=True).strip(), "token_ids": ids,
                   "finished": finished, "stop_id": stop_id, "n_tokens": len(ids)}
            if stats is not None:
                # Decode steps that produced THIS row's tokens: token 1 comes from the prefill, tokens 2..m from
                # m-1 decode steps, and the stop token (if any) from one more. Later steps only pad a finished row.
                own = len(ids) if finished else max(len(ids) - 1, 0)
                gaps = [step[b] for step in stats.get("decode_gap_abs_steps", [])][:own]
                rels = [step[b] for step in stats.get("decode_rel_steps", [])][:own]
                stored = [step[b] for step in stats.get("decode_stored_norm_steps", [])][:own]
                references = [step[b] for step in stats.get("decode_reference_stored_norm_steps", [])][:own]
                rec["write"] = {"prefill_gap_abs": stats["prefill"]["gap_abs"][b],
                                "prefill_rel": stats["prefill"]["rel"][b],
                                "prefill_stored_norm": stats["prefill"]["stored_norm"][b],
                                "prefill_reference_stored_norm": stats["prefill"]["reference_stored_norm"][b],
                                "prefill_positions": stats["prefill"]["positions"][b],
                                "decode_steps_total": stats.get("decode_steps", 0), "decode_steps_own": len(gaps),
                                "decode_gap_abs_mean": (sum(gaps) / len(gaps)) if gaps else None,
                                "decode_rel_mean": (sum(rels) / len(rels)) if rels else None,
                                "decode_stored_norm_mean": (sum(stored) / len(stored)) if stored else None,
                                "decode_reference_stored_norm_mean": (sum(references) / len(references)) if references else None,
                                "write_measurement": stats["write_measurement"]}
                if random_direction is not None:
                    rec["write"].update(random_match_scope=stats["random_match_scope"],
                                        learned_direction_used_for_size_only=True,
                                        fully_information_free_random_control=False)
            results.append(rec)
    return results


# ----------------------------------------------------------------------------- verification helpers
class CoordinateProbe:
    """Second forward hook (registered after ClampHook) that records coordinates along u and state norms."""

    def __init__(self, direction):
        self.u = unit(direction)
        self.z = []
        self.norms = []
        self.states = []

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float()
        self.z.append((hf @ self.u.to(h.device)).detach().cpu())
        self.norms.append(hf.norm(dim=-1).detach().cpu())
        self.states.append(h.detach().cpu().clone())
        return None


def verify_clamp(bundle, layer, items, source_hidden, w):
    """Check the stored coordinates after the write: masked positions equal the target
    (within the residual dtype's rounding), unmasked positions are untouched."""
    packed = pack_with_content_mask(items, bundle.tokenizer.pad_token_id, bundle.device)
    u = unit(w)
    src = torch.as_tensor(source_hidden).float()
    zt = src @ u
    before = CoordinateProbe(u)
    hb = bundle.layer(layer).register_forward_hook(before)
    try:
        forward_logits(bundle, packed["inputs"], grad=False)
    finally:
        hb.remove()
    hook = ClampHook(u.to(bundle.device), targets=zt, mask=packed["content_mask"], decode_all=False)
    after = CoordinateProbe(u)
    h1 = bundle.layer(layer).register_forward_hook(hook)
    h2 = bundle.layer(layer).register_forward_hook(after)
    try:
        forward_logits(bundle, packed["inputs"], grad=False)
    finally:
        h1.remove(); h2.remove()
    z0, z1 = before.z[0], after.z[0]
    norms = after.norms[0]
    sel = packed["content_mask"].cpu()
    att = packed["inputs"]["attention_mask"].bool().cpu()
    target = zt[:, None].expand_as(z1)
    masked_err = (z1 - target).abs()[sel]
    gap0 = (z0 - target).abs()[sel]
    # Declared pass criterion: after the write each masked coordinate sits at the target up to the residual
    # dtype's rounding (bf16: 8 mantissa bits -> about 2^-8 of the state norm per component; a projection onto a
    # unit vector keeps that order), or within 5% of the gap it had to close; untouched positions are bit-exact.
    rounding = (2.0 ** -7) * norms[sel] + 1e-3 if next(bundle.model.parameters()).dtype != torch.float32 else 1e-4 * (1 + norms[sel])
    tolerance = torch.maximum(rounding, 0.05 * gap0)
    untouched = (after.states[0].float() - before.states[0].float()).abs()[att & ~sel]
    untouched_exact = torch.equal(after.states[0][att & ~sel], before.states[0][att & ~sel])
    passed = bool((masked_err <= tolerance).all()) and untouched_exact
    return {"passed": passed, "masked_positions": int(sel.sum()), "masked_abs_error_max": float(masked_err.max()),
            "masked_abs_error_mean": float(masked_err.mean()), "tolerance_max": float(tolerance.max()),
            "state_norm_mean_at_masked": float(norms[sel].mean()), "gap_closed_mean": float(gap0.mean()),
            "untouched_positions": int((att & ~sel).sum()),
            "untouched_full_state_equal": untouched_exact,
            "untouched_abs_change_max": float(untouched.max()) if untouched.numel() else 0.0,
            "target_values": zt.tolist(), "base_coordinate_mean_at_masked": float((z0 * sel).sum() / sel.sum()),
            "residual_dtype": str(next(bundle.model.parameters()).dtype)}


def gradient_probe(bundle, layer, items, source_hidden, seed=0):
    """Gradient norm on a fresh random direction after one teacher-forced loss (nonzero expected)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.nn.Parameter(unit(torch.randn(bundle.hidden_size, generator=g)).to(bundle.device))
    loss, per_row, _ = teacher_forced_loss(bundle, layer, w, items, torch.as_tensor(source_hidden).float())
    loss.backward()
    return {"loss": float(loss.detach()), "per_row_ce": per_row.tolist(), "grad_norm": float(w.grad.norm()),
            "grad_finite": bool(torch.isfinite(w.grad).all())}
