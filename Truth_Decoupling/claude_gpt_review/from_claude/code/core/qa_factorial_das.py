"""Factorial two-variable interchange training for free-form QA (E44; plan 25).

What E43 (core/qa_prompt_das.py) got wrong, measured on its full Llama run, and what changes here:

  E43                                             here
  ----------------------------------------------  -------------------------------------------------------
  clamp every content position to the source's    add the source-minus-base DIFFERENCE along w:
  absolute last-token coordinate; the write was     h' = h + ((h_src - h_base) . w) w
  6-10x the endpoint gap and nearly the same for  read both states at the same place (last prompt token);
  both endpoints                                  identical prompts give a write of exactly zero
  hard-label CE; for E one target was the gold    soft matching (per-token JSD) to the next-token
  reference text, so the direction learned the    distribution of the COUNTERFACTUAL PROMPT itself; every
  reference answers' surface format               target is the model's own answer under some prompt
  every row had base type != source type and      "cause" rows (only this family's factor differs: behave
  nothing penalised a write that ignores the      like the source prompt) and "isolate" rows (only the
  source                                          OTHER factor differs: behave like the base prompt)

One frozen model, one learned unit direction w per layer and family. Which prompts form cause / isolate
rows is decided by the driver (scripts/e44_factorial_das.py); this module knows the hook, the losses,
the training loop, shifted generation and candidate scoring. Prompt rendering, target building, masks
and source capture are reused from core/qa_prompt_das.py unchanged. Nothing here calls a judge.
"""
import json
import math
import time

import torch
import torch.nn.functional as F

from . import qa_prompt_das as pd
from .model_io import forward_logits, greedy_generate

PROTOCOL = "qa_factorial_shift_v1"
LOSSES = ("jsd", "ce")
POSITIONS = ("content", "gen", "prompt")


# ----------------------------------------------------------------------------- the shift hook
class ShiftHook:
    """Forward hook: add a per-row vector at selected positions.

    Exactly one of
      amounts     [B] (or scalar) fixed coordinates c_b: the write is c_b * v          (inference)
      difference  [B, hidden] detached read-position differences d_b = h_src - h_base:
                  c_b = d_b . u, the write is c_b * v; gradients reach `direction`      (training)
      vectors     [B, hidden] vectors written as they are (reference: the whole difference)
    with u = direction / |direction| and v = u unless `write_direction` (a fixed unit vector for the
    random control: same coordinate, other axis) is given.

    mask         [B, T] bool for a forward whose width is T (training, or the prefill of generation).
                 None selects every position.
    decode_all   when a later call has T == 1 (a decode step), write every still-active row; False = decode
                 steps are left alone (prompt-only writes).
    """

    def __init__(self, direction=None, *, amounts=None, difference=None, vectors=None, mask=None,
                 decode_all=True, write_direction=None, record=True):
        if sum(x is not None for x in (amounts, difference, vectors)) != 1:
            raise ValueError("give exactly one of amounts, difference or vectors")
        if vectors is None and direction is None:
            raise ValueError("amounts / difference need a direction")
        if vectors is not None and write_direction is not None:
            raise ValueError("vectors are written as they are; no write_direction")
        self.direction = direction
        self.amounts = None if amounts is None else torch.as_tensor(amounts).float().reshape(-1)
        self.difference = None if difference is None else torch.as_tensor(difference).detach().float()
        self.vectors = None if vectors is None else torch.as_tensor(vectors).detach().float()
        for name in ("amounts", "difference", "vectors"):
            value = getattr(self, name)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError("%s must be finite" % name)
        self.mask = None if mask is None else torch.as_tensor(mask).bool()
        self.decode_all = bool(decode_all)
        self.write_direction = None if write_direction is None else pd.unit(write_direction)
        self.record = record
        self.active_rows = None      # generation updates this before each decode forward
        self.calls = 0
        self.applied = 0
        self.prefill = None
        self.decode_rel = []
        self.decode_stored_norm = []
        self.last_amounts = None     # [B] coordinates actually used (detached), for the record

    def _unit(self, device):
        d = self.direction
        if not torch.is_tensor(d):
            d = torch.as_tensor(d)
        d = d.to(device)
        if d.dtype != torch.float32:
            d = d.float()
        n = d.norm()
        if d.ndim != 1 or not bool(torch.isfinite(d).all()) or float(n.detach()) <= 0:
            raise ValueError("shift direction must be a finite nonzero vector")
        return d / n

    def _select(self, h):
        B, T, _ = h.shape
        if self.mask is not None and T == self.mask.shape[1] and B == self.mask.shape[0]:
            return self.mask.to(h.device), "masked"
        if self.mask is not None and T == 1:                          # a decode step of cached generation
            if not self.decode_all:
                return torch.zeros((B, 1), dtype=torch.bool, device=h.device), "decode"
            sel = torch.ones((B, 1), dtype=torch.bool, device=h.device)
            if self.active_rows is not None:
                active = torch.as_tensor(self.active_rows, device=h.device, dtype=torch.bool)
                if active.shape != (B,):
                    raise ValueError("active generation rows do not match the hidden batch")
                sel = sel & active[:, None]
            return sel, "decode"
        if self.mask is None:
            return torch.ones((B, T), dtype=torch.bool, device=h.device), "all"
        raise RuntimeError("shift hook saw sequence width %d; mask width %d" % (T, self.mask.shape[1]))

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.ndim != 3:
            raise ValueError("layer output must be [batch, sequence, hidden]")
        B = h.shape[0]
        self.calls += 1
        sel, stage = self._select(h)
        if not bool(sel.any()):
            return out
        if self.vectors is not None:
            write = self.vectors.to(h.device)                                  # [B, hidden]
            amounts = write.norm(dim=-1)
        else:
            u = self._unit(h.device)
            if self.difference is not None:
                amounts = self.difference.to(h.device) @ u                     # [B], depends on the direction
            else:
                amounts = self.amounts.to(h.device)
                if amounts.numel() == 1:
                    amounts = amounts.expand(B)
            v = u if self.write_direction is None else self.write_direction.to(h.device)
            write = amounts[:, None] * v                                       # [B, hidden]
        if write.shape[0] != B:
            raise ValueError("need one shift per batch row")
        hf = h.float()
        shifted = (hf + write[:, None, :]).to(h.dtype)
        h_new = torch.where(sel[..., None], shifted, h)
        self.applied += 1
        self.last_amounts = amounts.detach().float().cpu()
        if self.record:
            with torch.no_grad():
                n_sel = sel.sum(1).clamp_min(1).float()
                stored = (h_new.float() - hf).norm(dim=-1)                      # [B, T] after the dtype round trip
                rel = torch.where(sel, stored / hf.norm(dim=-1).clamp_min(1e-12), torch.zeros_like(stored)).sum(1) / n_sel
                stored_mean = torch.where(sel, stored, torch.zeros_like(stored)).sum(1) / n_sel
                if stage in ("masked", "all") and self.prefill is None:
                    self.prefill = {"rel": rel.cpu(), "stored_norm": stored_mean.cpu(), "positions": sel.sum(1).cpu(),
                                    "amount": amounts.detach().float().cpu(), "sequence_width": h.shape[1]}
                elif stage == "decode":
                    self.decode_rel.append(rel.cpu())
                    self.decode_stored_norm.append(stored_mean.cpu())
        if isinstance(out, tuple):
            return (h_new,) + tuple(out[1:])
        return h_new

    def stats(self):
        result = {"calls": self.calls, "applied": self.applied,
                  "write_measurement": "stored_delta_after_residual_dtype_roundtrip"}
        if self.prefill is not None:
            result["prefill"] = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in self.prefill.items()}
        if self.decode_rel:
            result["decode_steps"] = len(self.decode_rel)
            result["decode_rel_steps"] = torch.stack(self.decode_rel).tolist()               # [steps][B]
            result["decode_stored_norm_steps"] = torch.stack(self.decode_stored_norm).tolist()
        return result


def position_mask(packed, positions):
    """[B, T] write mask for a packed teacher-forced batch.

    content  first user-content token .. end (the E43 rule)
    gen      last prompt token .. end (the read position and every answer position; the question is untouched)
    prompt   first user-content token .. last prompt token (answer positions / decode steps are untouched)
    """
    if positions == "content":
        return packed["content_mask"]
    if positions == "prompt":
        return packed["content_mask"] & ~packed["target_mask"]
    if positions == "gen":
        mask = packed["prompt_end_mask"] | packed["target_mask"]
        if (mask & ~packed["content_mask"]).any():
            raise RuntimeError("generation positions must be content positions")
        return mask
    raise ValueError("positions must be one of %s" % (POSITIONS,))


# ----------------------------------------------------------------------------- losses
def answer_token_logprobs(logits, packed):
    """Full-vocabulary log-probs at the predictor position of every answer token.

    Returns (logp [N, V] float32, row [N], target_ids [N]) in row-major order; N = all answer tokens.
    """
    ids = packed["inputs"]["input_ids"]
    target = packed["target_mask"]
    if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(ids.shape):
        raise ValueError("logits must have shape [B, T, V] matching the packed inputs")
    if target[:, 0].any() or not target.any(dim=1).all():
        raise ValueError("each row needs predictable target positions")
    shifted = target[:, 1:].to(logits.device)
    selected = logits[:, :-1, :][shifted].float()
    rows = shifted.nonzero(as_tuple=False)[:, 0]
    return F.log_softmax(selected, dim=-1), rows, ids[:, 1:].to(logits.device)[shifted]


def jsd(logp, logq):
    """Jensen-Shannon divergence per row of two [N, V] log-probability tensors (nats; 0 .. ln 2)."""
    if logp.shape != logq.shape:
        raise ValueError("distributions to compare must have the same shape")
    logm = torch.logsumexp(torch.stack([logp, logq]), dim=0) - math.log(2.0)
    kl_p = (logp.exp() * (logp - logm)).sum(-1)
    kl_q = (logq.exp() * (logq - logm)).sum(-1)
    return 0.5 * (kl_p + kl_q)


@torch.no_grad()
def reference_logprobs(bundle, encoded, batch_size=8):
    """Clean (no hook) teacher-forced log-probs of each sequence's answer tokens.

    encoded: list of build_target() dicts (target prompt + that prompt's own answer).
    Returns one float32 CPU tensor [answer_length, V] per item: the distribution the intervened base must match.
    """
    out = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i:i + batch_size]
        packed = pd.pack_with_content_mask(chunk, bundle.tokenizer.pad_token_id, bundle.device)
        logits = forward_logits(bundle, packed["inputs"], grad=False)
        logp, rows, _ = answer_token_logprobs(logits, packed)
        if not bool(torch.isfinite(logp).all()):
            raise RuntimeError("nonfinite reference distribution")
        for b, item in enumerate(chunk):
            piece = logp[rows == b]
            if piece.shape[0] != item["answer_length"]:
                raise RuntimeError("reference rows do not match the answer length")
            out.append(piece.cpu())
    return out


def rows_loss(bundle, layer, rows, *, w=None, mode="direction", loss="jsd", positions="content", grad=False):
    """Per-row loss of a batch of interchange rows.

    rows  list of {"encoded": base prompt + the target prompt's answer, "difference": [hidden] h_src - h_base
          at the read position (detached), "ref_logp": [answer_length, V] target-prompt distribution (jsd only)}.
    mode  "direction": write (difference . w) w        (the trained operator)
          "vectors":   write the whole difference      (reference: how much any direction could transfer)
          "none":      no write                        (reference: the gap to close; 0 for isolate rows)
    Returns (per_row [B] tensor, hook stats or None).
    """
    if loss not in LOSSES:
        raise ValueError("loss must be one of %s" % (LOSSES,))
    packed = pd.pack_with_content_mask([r["encoded"] for r in rows], bundle.tokenizer.pad_token_id, bundle.device)
    hook = None
    if mode == "none":
        logits = forward_logits(bundle, packed["inputs"], grad=False)
    else:
        diff = torch.stack([torch.as_tensor(r["difference"]).float() for r in rows]).to(bundle.device)
        mask = position_mask(packed, positions)
        if mode == "direction":
            if w is None:
                raise ValueError("mode 'direction' needs w")
            hook = ShiftHook(w, difference=diff, mask=mask, decode_all=False)
        elif mode == "vectors":
            hook = ShiftHook(vectors=diff, mask=mask, decode_all=False)
        else:
            raise ValueError("unknown mode %r" % mode)
        with pd.clamp(bundle, layer, hook):
            logits = forward_logits(bundle, packed["inputs"], grad=grad)
        if hook.applied != 1:
            raise RuntimeError("shift hook must fire exactly once per teacher-forced forward")
    logp, row_index, target_ids = answer_token_logprobs(logits, packed)
    if loss == "jsd":
        ref = torch.cat([r["ref_logp"] for r in rows], dim=0).to(logp.device)
        if ref.shape != logp.shape:
            raise RuntimeError("reference distributions do not align with the answer tokens")
        per_token = jsd(logp, ref)
    else:
        per_token = -logp.gather(1, target_ids[:, None]).squeeze(1)
    n_tokens = packed["target_mask"].sum(1).float().to(per_token.device)
    per_row = per_token.new_zeros(len(rows)).index_add(0, row_index, per_token) / n_tokens
    return per_row, (hook.stats() if hook is not None else None)


def question_loss(per_row, kinds):
    """Equal weight for the cause part and the isolate part of one question's rows."""
    cause = [i for i, k in enumerate(kinds) if k == "cause"]
    isolate = [i for i, k in enumerate(kinds) if k == "isolate"]
    if not cause:
        raise ValueError("a training question needs at least one cause row")
    total = per_row[cause].mean()
    if isolate:
        total = total + per_row[isolate].mean()
    return total


# ----------------------------------------------------------------------------- training
def train_direction(bundle, layer, questions, *, epochs=10, lr=1e-3, clip=1.0, seed=42, init=None, loss="jsd",
                    positions="content", val_questions=(), log=None):
    """Learn one unit direction at `layer`.

    questions      list of {"pair_id", "rows": [row, ...]}; row = {"kind": "cause" | "isolate", "encoded",
                   "difference", "ref_logp"}. One optimizer step per question on all of its rows.
    val_questions  same structure, never used for a gradient; their loss is recorded after every epoch.
    init           None = random unit vector from the seed; or a vector to start from (e.g. the mean difference).
    Adam, learning rate decayed linearly to zero, gradient clipping, w renormalized after every step.
    The model is frozen; only w has gradients.
    """
    if not questions:
        raise ValueError("training needs at least one question")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (lr, clip)):
        raise ValueError("learning rate and clipping norm must be finite and positive")
    if bundle.model.training or any(p.requires_grad for p in bundle.model.parameters()):
        raise ValueError("direction training requires an eval-mode frozen model")
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + int(layer))
    w0 = pd.unit(torch.randn(bundle.hidden_size, generator=g)) if init is None else pd.unit(init)
    w = torch.nn.Parameter(w0.clone().to(bundle.device))
    opt = torch.optim.Adam([w], lr=lr)
    total_steps = epochs * len(questions)
    step = 0
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(questions), generator=g).tolist()
        sums = {"cause": 0.0, "isolate": 0.0}
        counts = {"cause": 0, "isolate": 0}
        grad_norms = []
        for idx in order:
            q = questions[idx]
            kinds = [r["kind"] for r in q["rows"]]
            for group in opt.param_groups:
                group["lr"] = lr * (1.0 - step / total_steps)
            opt.zero_grad(set_to_none=True)
            per_row, _ = rows_loss(bundle, layer, q["rows"], w=w, loss=loss, positions=positions, grad=True)
            value = question_loss(per_row, kinds)
            if not torch.isfinite(value):
                raise RuntimeError("nonfinite training loss at layer %d" % layer)
            value.backward()
            if w.grad is None or not torch.isfinite(w.grad).all():
                raise RuntimeError("direction received no finite gradient at layer %d" % layer)
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_([w], clip)))
            opt.step()
            with torch.no_grad():
                w.data = w.data / w.data.norm()
            step += 1
            for k, v in zip(kinds, per_row.detach().tolist()):
                sums[k] += v
                counts[k] += 1
        record = {"epoch": epoch + 1,
                  "train_cause": sums["cause"] / max(1, counts["cause"]),
                  "train_isolate": (sums["isolate"] / counts["isolate"]) if counts["isolate"] else None,
                  "grad_norm_mean": sum(grad_norms) / len(grad_norms), "grad_norm_max": max(grad_norms),
                  "zero_grad_steps": sum(1 for x in grad_norms if x == 0.0), "seconds": time.time() - t0}
        if val_questions:
            record.update({"val_" + k: v for k, v in mean_losses(bundle, layer, val_questions, w=w.detach(), loss=loss,
                                                                 positions=positions).items()})
        history.append(record)
        if log is not None:
            log(json.dumps({"layer": layer, **record}, ensure_ascii=False))
    return w.detach().float().cpu(), {"history": history, "seed": seed, "lr": lr, "lr_schedule": "linear_to_zero",
                                      "clip": clip, "epochs": epochs, "n_questions": len(questions),
                                      "n_val_questions": len(val_questions), "loss": loss, "positions": positions,
                                      "init": "random" if init is None else "given",
                                      "init_sha256": pd.fingerprint(w0.tolist()), "seconds": time.time() - t0}


@torch.no_grad()
def mean_losses(bundle, layer, questions, *, w=None, mode="direction", loss="jsd", positions="content"):
    """Mean cause / isolate loss over questions (no gradients). Keys: cause, isolate (None if there are none)."""
    sums = {"cause": 0.0, "isolate": 0.0}
    counts = {"cause": 0, "isolate": 0}
    for q in questions:
        per_row, _ = rows_loss(bundle, layer, q["rows"], w=w, mode=mode, loss=loss, positions=positions, grad=False)
        for r, v in zip(q["rows"], per_row.tolist()):
            sums[r["kind"]] += v
            counts[r["kind"]] += 1
    return {k: (sums[k] / counts[k]) if counts[k] else None for k in ("cause", "isolate")}


# ----------------------------------------------------------------------------- endpoints / controls
def difference_stats(H_plus, H_minus, w):
    """Coordinates of the per-question differences h_plus - h_minus along w (both read at the last prompt token)."""
    d = torch.as_tensor(H_plus).float() - torch.as_tensor(H_minus).float()
    z = d @ pd.unit(w)
    return {"mean": float(z.mean()), "std": float(z.std(unbiased=False)) if z.numel() > 1 else 0.0,
            "min": float(z.min()), "max": float(z.max()), "share_positive": float((z > 0).float().mean()),
            "values": z.tolist()}


def mean_difference(H_plus, H_minus):
    """(unit direction, norm) of the mean difference vector."""
    d = (torch.as_tensor(H_plus).float() - torch.as_tensor(H_minus).float()).mean(0)
    n = float(d.norm())
    if not math.isfinite(n) or n <= 0:
        raise ValueError("mean difference is zero or nonfinite")
    return d / n, n


def fixed_random_direction(hidden_size, seed, layer):
    return pd.fixed_random_direction(hidden_size, seed, layer)


# ----------------------------------------------------------------------------- shifted generation
def _prefill_mask(inputs, prompts, positions):
    content = pd.prompt_content_mask(inputs["attention_mask"], [p["prompt_length"] for p in prompts],
                                     [p["content_start"] for p in prompts])
    if positions in ("content", "prompt"):
        return content
    if positions == "gen":
        mask = torch.zeros_like(content)
        mask[:, -1] = True                      # left padding: the last column is every row's last prompt token
        return mask
    raise ValueError("positions must be one of %s" % (POSITIONS,))


def _row_amounts(amount, n):
    """None, one number for every prompt, or one number per prompt -> list of n floats (or None)."""
    if amount is None:
        return None
    values = [amount] * n if isinstance(amount, (int, float)) and not isinstance(amount, bool) else list(amount)
    if len(values) != n or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("shift amounts must be finite numbers, one for all prompts or one per prompt")
    return [float(v) for v in values]


@torch.no_grad()
def generate_shift(bundle, prompts, *, max_new_tokens, batch_size=8, layer=None, direction=None, amount=None,
                   vectors=None, positions="content"):
    """Greedy generation for rendered prompts, optionally with a write at the write positions of the prefill and
    at every decode step of a still-active row:
      amount   a number (same for all prompts) or one number per prompt: add amount * direction
      vectors  [N, hidden], one vector per prompt, added as it is (the whole per-question difference)
    Neither = no write. Same record layout as pd.generate_prompts (text, token_ids, finished, stop_id, n_tokens, write)."""
    if amount is not None and vectors is not None:
        raise ValueError("give amount or vectors, not both")
    amounts = _row_amounts(amount, len(prompts))
    if vectors is not None:
        vectors = torch.as_tensor(vectors).float()
        if vectors.ndim != 2 or vectors.shape[0] != len(prompts) or not bool(torch.isfinite(vectors).all()):
            raise ValueError("vectors must be finite [number of prompts, hidden]")
    stops = pd.stop_ids(bundle)
    results = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        inputs = bundle.encode([p["prefix_text"] for p in chunk])
        for b, p in enumerate(chunk):
            got = inputs["input_ids"][b][inputs["attention_mask"][b].bool()].tolist()
            if got != p["prefix_ids"]:
                raise RuntimeError("batched prompt tokens differ from the single-prompt rendering")
        if amounts is None and vectors is None:
            _, out = greedy_generate(bundle, inputs, max_new_tokens)
            stats = None
        else:
            if layer is None or (vectors is None and direction is None):
                raise ValueError("shifted generation needs a layer, and a direction unless whole vectors are given")
            mask = _prefill_mask(inputs, chunk, positions)
            decode = positions != "prompt"
            if vectors is not None:
                hook = ShiftHook(vectors=vectors[i:i + len(chunk)], mask=mask, decode_all=decode)
            else:
                hook = ShiftHook(direction, amounts=torch.tensor(amounts[i:i + len(chunk)]), mask=mask, decode_all=decode)
            active = torch.ones(len(chunk), dtype=torch.bool, device=bundle.device)

            def track_active_rows(model, args, kwargs, hook=hook, active=active, n=len(chunk)):
                # The previously generated token (after generation's own processing) says which rows ended.
                ids = kwargs.get("input_ids", args[0] if args else None)
                if hook.calls and ids is not None:
                    if ids.ndim != 2 or ids.shape[0] != n:
                        raise ValueError("shifted generation requires an unexpanded greedy batch")
                    stopped = torch.zeros(n, dtype=torch.bool, device=ids.device)
                    for token in stops:
                        stopped |= ids[:, -1] == token
                    active.logical_and_(~stopped.to(active.device))
                    hook.active_rows = active

            pre = bundle.model.register_forward_pre_hook(track_active_rows, with_kwargs=True)
            try:
                with pd.clamp(bundle, layer, hook):
                    _, out = greedy_generate(bundle, inputs, max_new_tokens)
            finally:
                pre.remove()
            if hook.prefill is None or hook.prefill["sequence_width"] != inputs["input_ids"].shape[1]:
                raise RuntimeError("the shift did not fire on the prefill")
            stats = hook.stats()
        prompt_len = inputs["input_ids"].shape[1]
        for b, p in enumerate(chunk):
            ids, finished, stop_id = pd._cut_new_tokens(out[b][prompt_len:], stops)
            rec = {"text": bundle.tokenizer.decode(ids, skip_special_tokens=True).strip(), "token_ids": ids,
                   "finished": finished, "stop_id": stop_id, "n_tokens": len(ids)}
            if stats is not None:
                own = len(ids) if finished else max(len(ids) - 1, 0)       # decode steps that produced this row's tokens
                rels = [s[b] for s in stats.get("decode_rel_steps", [])][:own]
                rec["write"] = {"amount": stats["prefill"]["amount"][b], "prefill_rel": stats["prefill"]["rel"][b],
                                "prefill_stored_norm": stats["prefill"]["stored_norm"][b],
                                "prefill_positions": stats["prefill"]["positions"][b],
                                "decode_steps_total": stats.get("decode_steps", 0), "decode_steps_own": len(rels),
                                "decode_rel_mean": (sum(rels) / len(rels)) if rels else None,
                                "write_measurement": stats["write_measurement"]}
            results.append(rec)
    return results


# ----------------------------------------------------------------------------- candidate scoring (judge-free)
@torch.no_grad()
def score_answers(bundle, prompt, answers, *, eot_ids, max_target_tokens=48, batch_size=16, layer=None,
                  direction=None, amount=None, vector=None, positions="content"):
    """Teacher-forced log-probability of each candidate answer text after `prompt`, optionally with the shift
    (amount * direction, or one whole `vector` [hidden] for this question).

    The end-of-turn token is appended to complete candidates (so the hidden states see the same layout as
    training) but is NOT scored. Returns one record per candidate:
    {"sum_logp", "n_tokens", "truncated", "skipped"}; skipped holds the reason when a candidate cannot be encoded.
    """
    records = [None] * len(answers)
    items, index = [], []
    for i, text in enumerate(answers):
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty candidate")
            items.append(pd.build_target(bundle, prompt, answer_text=text.strip(), max_target_tokens=max_target_tokens,
                                         eot_ids=eot_ids))
            index.append(i)
        except ValueError as exc:
            records[i] = {"sum_logp": None, "n_tokens": 0, "truncated": False, "skipped": str(exc)}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        packed = pd.pack_with_content_mask(chunk, bundle.tokenizer.pad_token_id, bundle.device)
        if amount is None and vector is None:
            logits = forward_logits(bundle, packed["inputs"], grad=False)
        else:
            if amount is not None and vector is not None:
                raise ValueError("give amount or vector, not both")
            if layer is None or (vector is None and direction is None):
                raise ValueError("shifted scoring needs a layer, and a direction unless a whole vector is given")
            mask = position_mask(packed, positions)
            if vector is not None:
                hook = ShiftHook(vectors=torch.as_tensor(vector).float().reshape(1, -1).expand(len(chunk), -1), mask=mask,
                                 decode_all=False)
            else:
                hook = ShiftHook(direction, amounts=torch.full((len(chunk),), float(amount)), mask=mask, decode_all=False)
            with pd.clamp(bundle, layer, hook):
                logits = forward_logits(bundle, packed["inputs"], grad=False)
            if hook.applied != 1:
                raise RuntimeError("shift hook must fire exactly once per scoring forward")
        logp, rows, target_ids = answer_token_logprobs(logits, packed)
        token_logp = logp.gather(1, target_ids[:, None]).squeeze(1)
        for b, item in enumerate(chunk):
            own = token_logp[rows == b]
            body = item["answer_body_tokens"]
            if own.shape[0] != item["answer_length"] or body < 1:
                raise RuntimeError("scored tokens do not match the candidate length")
            records[index[start + b]] = {"sum_logp": float(own[:body].sum()), "n_tokens": body,
                                         "truncated": bool(item["truncated"]), "skipped": None}
    return records


def choice_metrics(true_scores, false_scores):
    """MC-style readouts from candidate log-probabilities (lists of floats; None entries are ignored).

    margin  best true minus best false (sum of token log-probs, the standard MC1 comparison)
    mc1     1 if margin > 0
    mc2     normalised probability mass on the true candidates
    """
    t = [s for s in true_scores if s is not None]
    f = [s for s in false_scores if s is not None]
    if not t or not f:
        return None
    margin = max(t) - max(f)
    top = max(t + f)
    mass_t = sum(math.exp(s - top) for s in t)
    mass_f = sum(math.exp(s - top) for s in f)
    return {"margin": margin, "mc1": 1 if margin > 0 else 0, "mc2": mass_t / (mass_t + mass_f)}
