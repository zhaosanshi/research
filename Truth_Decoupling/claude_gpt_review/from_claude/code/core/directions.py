"""Direction estimators.

All estimators return a unit-norm direction (or an orthonormal [d, r] basis for
rank r > 1) plus a metadata dict. Three estimators, mirroring the plan:

  1. mean_diff_direction      mean(pos) - mean(neg)            (lower bound; numpy)
  2. logreg_direction         logistic-regression weight vector (ITI/SAPLMA style; numpy IRLS)
  3. train_das                rank-r Distributed Alignment Search, multi-step
                              teacher-forced KL objective ported from v12
                              (das_experiment_v4_multistep.py) but implemented
                              with a plain forward hook instead of pyvene.

The DAS maths is identical to v12:
    z_base = h_base W,  z_src = h_src W            (W in R^{d x r}, orthonormal)
    h'_base = h_base + (z_src - z_base) W^T
    L = (1/T) sum_t KL( p_src(. | y^s_<t) || p_int(. | y^s_<t) )
with Adam(lr=1e-3), grad-norm clip 1.0, T=8, 10 epochs, rank 1 by default.

Variants of the DAS objective (2026-10-31; all default to the v12 behaviour above):
    loss="jsd"          Jensen-Shannon divergence over the full vocabulary instead of forward KL
                        (CDAS, ICLR 2026, Table 8: JSD 0.860 > reverse KL 0.840 >> forward KL 0.560)
    bidirectional=True  also clamp the SOURCE with the base's projection and match the base's own
                        distribution (CDAS D+ and D-), so the same w must work in both directions
    positions="all"     clamp every position of the base sequence with the source's last-token
                        projection (CDAS one-to-many) instead of the last position only
    pair_weights        per-pair loss weights (balanced by category x length cell, see balanced_pair_weights)
    trim                trimmed-mean fraction for the mean projection gap
    anchor="prompt"     read the source projection and write the base at the ORIGINAL prompts' last token at every
                        teacher-forced step (the position the evaluation intervenes at); the appended answer prefix
                        only observes the downstream effect. anchor="current" is v12: follow the growing sequence's
                        last position, i.e. from step 2 on the swap happens on already-emitted answer tokens
                        (GPT review round 2, 2026-11-01).
"""
import math
import re

import numpy as np


# --------------------------------------------------------------------------- #
# numpy helpers
# --------------------------------------------------------------------------- #

def unit(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / max(n, eps)


def cosine(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b))


def trimmed_mean(X, trim=0.0):
    """Column-wise trimmed mean (trim fraction removed from each tail, per coordinate)."""
    X = np.asarray(X, dtype=np.float64)
    if trim <= 0:
        return X.mean(axis=0)
    n = X.shape[0]
    k = int(math.floor(trim * n))
    if k == 0 or 2 * k >= n:
        return X.mean(axis=0)
    Xs = np.sort(X, axis=0)
    return Xs[k:n - k].mean(axis=0)


def mean_diff_direction(H_pos, H_neg, trim=0.0):
    """Unit direction from pos to neg class means. Returns (w, meta)."""
    mu_pos, mu_neg = trimmed_mean(H_pos, trim), trimmed_mean(H_neg, trim)
    diff = mu_pos - mu_neg
    gap = float(np.linalg.norm(diff))
    w = unit(diff)
    proj_pos = (np.asarray(H_pos) - (mu_pos + mu_neg) / 2) @ w
    proj_neg = (np.asarray(H_neg) - (mu_pos + mu_neg) / 2) @ w
    return w, {
        "estimator": "mean_diff", "gap": gap, "trim": trim,
        "mu_pos": mu_pos, "mu_neg": mu_neg, "center": (mu_pos + mu_neg) / 2,
        "proj_std_pos": float(proj_pos.std()), "proj_std_neg": float(proj_neg.std()),
        "n_pos": int(len(H_pos)), "n_neg": int(len(H_neg)),
    }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_logreg(X, y, l2=1.0, max_iter=30, tol=1e-6, standardize=True):
    """Ridge-penalised logistic regression by IRLS in the dual (n x n) form.

    Suitable for n << d (hundreds of samples, thousands of features).
    Returns dict with w (d,), b, mean, std, converged.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-6
    else:
        mean = np.zeros(d)
        std = np.ones(d)
    Xs = (X - mean) / std
    Xb = np.hstack([Xs, np.ones((n, 1))])  # bias column (penalised with the same l2; fine for a direction)
    K = Xb @ Xb.T
    sign = 2.0 * y - 1.0

    def reg_loss(w):
        eta = Xb @ w
        return float(np.sum(np.logaddexp(0.0, -sign * eta)) + 0.5 * l2 * np.sum(w * w))

    w_full = np.zeros(d + 1)
    cur = reg_loss(w_full)
    converged = False
    for it in range(max_iter):
        eta = Xb @ w_full
        p = _sigmoid(eta)
        s = np.clip(p * (1 - p), 1e-4, None)
        z = eta + (y - p) / s
        # Newton step via the dual (n x n) system: w_newton = Xb^T (K + l2 diag(1/s))^{-1} z
        a_new = np.linalg.solve(K + l2 * np.diag(1.0 / s), z)
        direction = Xb.T @ a_new - w_full
        # backtracking line search keeps IRLS from diverging on (near-)separable data
        t = 1.0
        while t > 1e-4:
            cand = w_full + t * direction
            cand_loss = reg_loss(cand)
            if cand_loss <= cur:
                break
            t *= 0.5
        step = t * direction
        w_full = w_full + step
        cur = min(cur, cand_loss)
        if np.linalg.norm(step) / (np.linalg.norm(w_full) + 1e-9) < tol:
            converged = True
            break
    w_std, b = w_full[:-1], w_full[-1]
    # map back to raw feature space: eta = ((x - mean)/std) . w_std + b
    w_raw = w_std / std
    b_raw = b - float(np.dot(mean, w_raw))
    return {"w": w_raw, "b": b_raw, "w_std": w_std, "mean": mean, "std": std,
            "converged": converged, "iters": it + 1}


def logreg_predict_proba(fit, X):
    X = np.asarray(X, dtype=np.float64)
    return _sigmoid(X @ fit["w"] + fit["b"])


def logreg_direction(H_pos, H_neg, l2=1.0, standardize=True):
    """Direction = logistic-regression weight vector (in raw activation space)."""
    X = np.vstack([H_pos, H_neg])
    y = np.concatenate([np.ones(len(H_pos)), np.zeros(len(H_neg))])
    fit = fit_logreg(X, y, l2=l2, standardize=standardize)
    w = unit(fit["w"])
    from .metrics import auroc
    auc_train = auroc(logreg_predict_proba(fit, X), y)
    return w, {"estimator": "logreg", "l2": l2, "standardize": standardize, "auc_train": auc_train,
               "bias": fit["b"], "converged": fit["converged"], "center": X.mean(axis=0),
               "n_pos": int(len(H_pos)), "n_neg": int(len(H_neg))}


def cv_auc(H_pos, H_neg, k=5, l2=1.0, seed=0, standardize=True):
    """k-fold cross-validated AUROC of a logistic probe (used by LA-Score `sep`)."""
    from .metrics import auroc
    X = np.vstack([H_pos, H_neg])
    y = np.concatenate([np.ones(len(H_pos)), np.zeros(len(H_neg))])
    n = len(y)
    rng = np.random.default_rng(seed)
    # stratified folds: unstratified splits give below-chance AUC on null layers (class-imbalance artefact)
    pos_idx = rng.permutation(len(H_pos))
    neg_idx = len(H_pos) + rng.permutation(len(H_neg))
    folds = [np.concatenate([p, q]) for p, q in zip(np.array_split(pos_idx, k), np.array_split(neg_idx, k))]
    idx = np.arange(n)
    scores = np.zeros(n)
    for f in folds:
        train = np.setdiff1d(idx, f)
        fit = fit_logreg(X[train], y[train], l2=l2, standardize=standardize)
        scores[f] = logreg_predict_proba(fit, X[f])
    return auroc(scores, y), scores


# --------------------------------------------------------------------------- #
# torch: random controls and DAS
# --------------------------------------------------------------------------- #

def random_direction(d, rank=1, seed=0, device="cpu", dtype=None):
    """Haar-uniform random orthonormal [d, rank] basis (QR with sign correction), as in v12."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(d, rank, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A, mode="reduced")
    Q = Q * torch.sign(torch.diagonal(R, 0)).unsqueeze(0)
    err = (Q.T @ Q - torch.eye(rank)).abs().max().item()
    assert err < 1e-4, "random basis not orthonormal: %.2e" % err
    Q = Q.to(device)
    if dtype is not None:
        Q = Q.to(dtype)
    return Q


class OrthoBasis:
    """Trainable [d, r] basis kept orthonormal (r=1: normalisation; r>1: QR)."""

    def __init__(self, d, rank, seed=0, device="cpu"):
        import torch
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.raw = (torch.randn(d, rank, generator=g, dtype=torch.float32) / math.sqrt(d)).to(device)
        self.raw.requires_grad_(True)
        self.rank = rank

    def parameters(self):
        return [self.raw]

    def __call__(self):
        import torch
        if self.rank == 1:
            return self.raw / self.raw.norm().clamp(min=1e-8)
        Q, R = torch.linalg.qr(self.raw, mode="reduced")
        return Q * torch.sign(torch.diagonal(R, 0)).unsqueeze(0)

    def detached(self):
        return self().detach().float().cpu()


def _interchange_hook_factory(W_fn, src_act_holder, positions="last", write_pos=-1):
    """Forward hook: replace the W-subspace component of the base with the source's value.

    W_fn returns the current orthonormal basis [d, r] (float32, requires grad).
    src_act_holder["h"] is the source activation at its read position, [B, d] (float32, same device).
    positions="last": only one base position is changed, index `write_pos` (-1 = the sequence's last token, v12;
                      the original prompt's last token under anchor="prompt").
    positions="all" : every position of the base sequence takes the source's projection
                      (CDAS-style one-to-many clamp). Pairs are encoded one at a time, so there is no padding.
    """
    import torch

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        W = W_fn()
        src = src_act_holder["h"].to(h.device).float()    # [B, d]
        z_src = src @ W                                   # [B, r]
        if positions == "last":
            h_pos = h[:, write_pos, :].float()            # [B, d]
            delta = (z_src - h_pos @ W) @ W.T             # [B, d]  == ((src - h_pos) @ W) @ W.T
            h_new = h.clone()
            h_new[:, write_pos, :] = (h_pos + delta).to(h.dtype)
        else:
            hf = h.float()                                # [B, T, d]
            delta = (z_src.unsqueeze(1) - hf @ W) @ W.T   # [B, T, d]
            h_new = (hf + delta).to(h.dtype)
        if isinstance(out, tuple):
            return (h_new,) + tuple(out[1:])
        return h_new
    return hook


def teacher_tokens(bundle, source_inputs, T):
    """Greedy T tokens from the (un-intervened) source prompt; teacher-forcing targets."""
    import torch
    from .model_io import greedy_generate
    with torch.no_grad():
        _, out = greedy_generate(bundle, source_inputs, T)
    return out[0, source_inputs["input_ids"].shape[1]:]  # [<=T]


def _teacher_cache_key(bundle, inputs, T):
    """Identity of a cached teacher sequence: the T it was generated for, the model object and the exact prompt token ids
    (review round 14 D4: another T; round 15 §3.5: the same T with another prompt or another model must not reuse the cache).
    id(model) is unique only among LIVE objects: a caller that releases a model and loads another while keeping the same pair
    dicts must use fresh pair dicts (the production flow keeps one bundle alive for the whole run)."""
    ids = inputs["input_ids"]
    return (int(T), id(getattr(bundle, "model", bundle)), tuple(int(x) for x in ids.reshape(-1).tolist()))


def _append_prefix(inputs, prefix_ids):
    import torch
    if prefix_ids.numel() == 0:
        return inputs
    p = prefix_ids.view(1, -1).to(inputs["input_ids"].device)
    return {
        "input_ids": torch.cat([inputs["input_ids"], p], dim=1),
        "attention_mask": torch.cat([inputs["attention_mask"], torch.ones_like(p, dtype=inputs["attention_mask"].dtype)], dim=1),
    }


def _divergence(int_logits, src_logits, kind="kl"):
    """Divergence between the intervened-base next-token distribution and the fixed source one (batch mean).

    kind="kl" : forward KL(p_src || p_int), the v12 objective.
    kind="jsd": Jensen-Shannon, 0.5 KL(p_int || m) + 0.5 KL(p_src || m), m = (p_int + p_src) / 2 (CDAS objective).
    Both over the full vocabulary. F.kl_div(input=log q, target=log p, log_target=True) == KL(p || q).
    """
    import torch
    import torch.nn.functional as F
    logp_int = F.log_softmax(int_logits, dim=-1)
    logp_src = F.log_softmax(src_logits, dim=-1)
    if kind == "kl":
        return F.kl_div(logp_int, logp_src, log_target=True, reduction="batchmean")
    if kind == "jsd":
        logm = torch.logsumexp(torch.stack([logp_int, logp_src]), dim=0) - math.log(2.0)
        return 0.5 * (F.kl_div(logm, logp_int, log_target=True, reduction="batchmean")
                      + F.kl_div(logm, logp_src, log_target=True, reduction="batchmean"))
    raise ValueError("unknown divergence %s" % kind)


def multistep_kl_loss(bundle, basis, pair, layer, y_src, T, loss="kl", positions="last",
                      base_key="base", source_key="source", anchor="current", trace=None):
    """One pair's teacher-forced multi-step divergence between p_int(base <- source) and p_src.

    y_src are the source's own greedy tokens (teacher forcing). base_key / source_key let the caller swap
    the roles of the two prompts (the D- term of the bidirectional objective). Name kept for compatibility;
    loss="kl", anchor="current" reproduces v12 exactly.

    anchor="prompt" : read the source and write the base at their ORIGINAL prompts' last token (index
                      len(prompt) - 1, fixed while the teacher prefix grows) -- the position the evaluation
                      intervenes at. The next-token distributions compared are still those at the current last
                      position, so steps >= 1 measure the downstream effect of that one intervention.
    anchor="current": read/write at the growing sequence's last token (from step 1 on: an answer token).
    trace: optional list; one dict per step with lengths and read/write indices (acceptance check).
    """
    from .model_io import capture_hooks, forward_logits

    total, n_steps = 0.0, 0
    holder = {}
    len_b, len_s = int(pair[base_key]["input_ids"].shape[1]), int(pair[source_key]["input_ids"].shape[1])
    for t in range(min(T, int(y_src.numel()))):
        base_t = _append_prefix(pair[base_key], y_src[:t])
        src_t = _append_prefix(pair[source_key], y_src[:t])
        if anchor == "prompt":
            pos_s, pos_b = len_s - 1, len_b - 1
        else:
            pos_s, pos_b = int(src_t["input_ids"].shape[1]) - 1, int(base_t["input_ids"].shape[1]) - 1
        if trace is not None:
            trace.append({"t": t, "len_base": int(base_t["input_ids"].shape[1]), "len_src": int(src_t["input_ids"].shape[1]),
                          "read_src_pos": pos_s, "write_base_pos": pos_b})
        with capture_hooks(bundle, [layer], ("resid",), position="all") as store:
            src_logits = forward_logits(bundle, src_t, grad=False)[:, -1, :].float()
        holder["h"] = store[("resid", layer)][:, pos_s, :].detach().float().to(bundle.device)
        handle = bundle.layer(layer).register_forward_hook(_interchange_hook_factory(basis, holder, positions, write_pos=pos_b))
        try:
            int_logits = forward_logits(bundle, base_t, grad=True)[:, -1, :].float()
        finally:
            handle.remove()
        total = total + _divergence(int_logits, src_logits, loss)
        n_steps += 1
    return total / max(n_steps, 1), n_steps


def train_das(bundle, pairs, layer, rank=1, T=8, epochs=10, lr=1e-3, seed=42, clip=1.0,
              log_every=1, log_fn=print, loss="kl", bidirectional=False, positions="last",
              pair_weights=None, trim=0.0, anchor="current"):
    """Rank-r DAS at `layer` on `pairs` (each: {"base": inputs, "source": inputs}, batch size 1 each).

    Defaults reproduce v12 (forward KL, base <- source only, current last position, unweighted, plain mean gap).
    loss / bidirectional / positions / pair_weights / trim / anchor: see the module docstring.
    Returns dict: W [d, r] float32 cpu, losses per epoch, mean_gap (mean of z_src - z_base over training
    pairs at t=0; the natural unit for source-free steering), gap_std, n_pairs, the variant settings and
    `position_trace` (lengths and read/write indices of the first pair's first steps; with anchor="prompt"
    the indices must stay fixed while the lengths grow).
    """
    import torch
    from .io_utils import set_seed
    from .model_io import capture_hooks, forward_logits

    if pair_weights is not None and len(pair_weights) != len(pairs):
        raise ValueError("pair_weights has %d entries for %d pairs" % (len(pair_weights), len(pairs)))
    for p in pairs:
        if p["base"]["input_ids"].shape[0] != 1 or p["source"]["input_ids"].shape[0] != 1:
            raise ValueError("DAS pairs must be encoded one sequence at a time (no padding): position indices assume batch size 1")
    wts = [1.0] * len(pairs) if pair_weights is None else [float(x) for x in pair_weights]
    set_seed(seed)
    basis = OrthoBasis(bundle.hidden_size, rank, seed=seed, device=bundle.device)
    opt = torch.optim.Adam(basis.parameters(), lr=lr)

    # Teacher tokens once per pair, cached on the pair dict and shared across seeds / variants. Each cached teacher carries the
    # identity it was made for (T, model object, exact prompt token ids; _teacher_cache_key) and is regenerated when any of them
    # differs (review round 14 D4: another T; round 15 §3.5: same T, other prompt or model). A stale SOURCE cache drops both
    # teachers, so a later bidirectional call cannot pick up a y_base made for the old T. A caller that pre-fills y_src / y_base
    # without a stamp manages its own cache: the pre-filled teacher is adopted and stamped from here on.
    for p in pairs:
        k_src = _teacher_cache_key(bundle, p["source"], T)
        if p.get("_teacher_key_src", k_src) != k_src:
            p.pop("y_src", None)
            p.pop("y_base", None)
            p.pop("_teacher_key_base", None)
        if "y_src" not in p:
            p["y_src"] = teacher_tokens(bundle, p["source"], T)
        p["_teacher_key_src"] = k_src
        if bidirectional:
            k_base = _teacher_cache_key(bundle, p["base"], T)
            if p.get("_teacher_key_base", k_base) != k_base:
                p.pop("y_base", None)
            if "y_base" not in p:
                p["y_base"] = teacher_tokens(bundle, p["base"], T)
            p["_teacher_key_base"] = k_base

    losses = []
    position_trace = []
    for ep in range(epochs):
        ep_loss, ep_steps = 0.0, 0
        for i, (p, wt) in enumerate(zip(pairs, wts)):
            opt.zero_grad()
            tr = position_trace if (ep == 0 and i == 0) else None      # acceptance trace: first pair, first epoch
            # D+: base <- source, match the source's distribution. Backward per term keeps peak memory at one term.
            l_fwd, n_steps = multistep_kl_loss(bundle, basis, p, layer, p["y_src"], T, loss=loss, positions=positions,
                                               anchor=anchor, trace=tr)
            (wt * (0.5 if bidirectional else 1.0) * l_fwd).backward()
            loss_val = float(l_fwd.item())
            if bidirectional:
                # D-: source <- base, match the base's own distribution (teacher-forced on the base's tokens)
                l_rev, n_rev = multistep_kl_loss(bundle, basis, p, layer, p["y_base"], T, loss=loss, positions=positions,
                                                 base_key="source", source_key="base", anchor=anchor)
                (wt * 0.5 * l_rev).backward()
                loss_val = 0.5 * (loss_val + float(l_rev.item()))
                n_steps += n_rev
            torch.nn.utils.clip_grad_norm_(basis.parameters(), max_norm=clip)
            opt.step()
            ep_loss += loss_val
            ep_steps += n_steps
            if tr is not None:
                # acceptance check (GPT review round 2) -- independent of logging, so it also runs with log_fn=None
                if anchor == "prompt" and len({(d["write_base_pos"], d["read_src_pos"]) for d in tr}) != 1:
                    raise RuntimeError("anchor=prompt but read/write indices moved across steps: %s" % tr)
                if log_fn:
                    tok = getattr(bundle, "tokenizer", None)      # show the teacher tokens (one-word answer + EOS => ~2 steps)
                    teacher = repr(tok.decode(p["y_src"], skip_special_tokens=False)) if tok is not None else "n/a"
                    log_fn("  [DAS L%d anchor=%s] position trace, first pair: %s | teacher tokens (%d): %s" % (
                        layer, anchor, "; ".join("t=%d len(base,src)=(%d,%d) write_base=%d read_src=%d" % (
                            d["t"], d["len_base"], d["len_src"], d["write_base_pos"], d["read_src_pos"]) for d in tr[:3]),
                        int(p["y_src"].numel()), teacher))
        avg = ep_loss / max(len(pairs), 1)
        losses.append(avg)
        if log_fn and (ep % log_every == 0 or ep == epochs - 1):
            log_fn("  [DAS L%d r%d %s%s%s anchor=%s] epoch %d/%d loss=%.6f (avg %.1f teacher steps)" % (
                layer, rank, loss, "+bidir" if bidirectional else "", "+allpos" if positions == "all" else "", anchor,
                ep, epochs, avg, ep_steps / max(len(pairs), 1)))

    W = basis.detached()  # [d, r]
    # mean projection gap at t=0 (for source-free steering units); optionally a trimmed mean
    gaps = []
    with torch.no_grad():
        for p in pairs:
            with capture_hooks(bundle, [layer], ("resid",)) as sb:
                forward_logits(bundle, p["base"], grad=False)
            with capture_hooks(bundle, [layer], ("resid",)) as ss:
                forward_logits(bundle, p["source"], grad=False)
            hb, hs = sb[("resid", layer)], ss[("resid", layer)]
            gaps.append(((hs - hb) @ W).squeeze(0))
    gaps = torch.stack(gaps)  # [n, r]
    mean_gap = trimmed_mean(gaps.numpy(), trim).tolist()
    return {"W": W, "losses": losses, "mean_gap": mean_gap, "gap_std": gaps.std(0).tolist(),
            "n_pairs": len(pairs), "layer": layer, "rank": rank, "T": T, "epochs": epochs, "lr": lr, "seed": seed,
            "loss_kind": loss, "bidirectional": bool(bidirectional), "positions": positions, "trim": float(trim),
            "weighted": pair_weights is not None, "anchor": anchor, "position_trace": position_trace}


def principal_cosines(W1, W2):
    """Cosines of principal angles between two orthonormal bases (rank 1 -> |cos|)."""
    import torch
    W1 = torch.as_tensor(np.asarray(W1), dtype=torch.float64).reshape(len(np.asarray(W1)), -1)
    W2 = torch.as_tensor(np.asarray(W2), dtype=torch.float64).reshape(len(np.asarray(W2)), -1)
    s = torch.linalg.svdvals(W1.T @ W2)
    return s.tolist()


# --------------------------------------------------------------------------- #
# DAS helpers: balanced weights, seed averaging
# --------------------------------------------------------------------------- #

def length_bins(lengths, n_bins=3):
    """Quantile bin index (0..n_bins-1) of each length; ties go to the lower bin."""
    x = np.asarray(lengths, dtype=np.float64)
    if len(x) == 0:
        return []
    edges = np.quantile(x, [i / n_bins for i in range(1, n_bins)])
    return [int(np.searchsorted(edges, v, side="left")) for v in x]


def balanced_pair_weights(keys):
    """Inverse-frequency weights, rescaled to mean 1, so every key (e.g. (category, length bin)) carries equal
    total weight in the DAS loss. The plan's 'balanced weighting by category and length'."""
    from collections import Counter
    cnt = Counter(keys)
    w = np.array([1.0 / cnt[k] for k in keys], dtype=np.float64)
    return (w / w.mean()).tolist()


def das_file_name(tag, layer, rank, seed, variant=""):
    return "das_%s_L%d_r%d_s%d%s.pt" % (tag, layer, rank, seed, ("_" + variant) if variant else "")


DAS_PROTOCOL_KEYS = ("anchor", "loss_kind", "positions", "rank", "n_pairs", "T", "bidirectional", "epochs", "lr", "trim", "weighted")
# Fields that legitimately differ between seeds of one training run. Every OTHER stored field (protocol, model / data / template
# identity, anything else the trainer recorded) must agree before seeds are merged (review round 15 §3.1-3.2). A trainer that
# stores per-seed bookkeeping (timings, traces) puts it under "per_seed" or in one of these keys.
DAS_PER_SEED_KEYS = frozenset({"W", "losses", "mean_gap", "gap_std", "seed", "position_trace", "per_seed"})


def _plain(v):
    """torch / numpy values as plain Python, so checkpoint fields can be compared with ==. A non-finite float becomes its string
    (NaN != NaN would otherwise report a field as conflicting with itself; the string is also JSON-safe)."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    return v


def load_das_direction(das_dir, tag, layer, variant="", align_to=None, rank=1, min_seed_cos=0.5):
    """Load the rank-1 DAS direction of (tag, layer, variant); several seeds are merged by a fixed, sign-insensitive rule.

    Files: das_<tag>_L<layer>_r<rank>_s<seed>[_<variant>].pt (torch.save of train_das' dict). variant="" matches only
    files WITHOUT a suffix. Returns (unit np.float64 [d], meta) or (None, None) when no file matches.

    Review round 14 (2026-11-04) fixes:
      D2  rank must be 1. This interface returns ONE direction; taking the first column of a higher-rank basis silently
          dropped trained dimensions. A higher-rank checkpoint needs a subspace loader, not this function.
      D5  a multi-seed merge needs the full protocol (DAS_PROTOCOL_KEYS) in EVERY file, and every stored field outside
          DAS_PER_SEED_KEYS must agree with the first file key by key (a key only one side has is a conflict). Round 15
          §3.1-3.2: comparing the key intersection let a file lacking `anchor` hide a conflict between two other seeds, and
          model / data identity fields were not compared at all. A single seed loads with whatever fields it has
          (meta["protocol_complete"] says whether the protocol is fully recorded).
      D1  seeds are sign-aligned to the FIRST seed (a reference inside the subspace), never to an external reference:
          aligning two near-identical seeds to a near-orthogonal `align_to` could cancel them to a zero vector or rotate
          the merge to a direction nearly orthogonal to both. Seeds must agree with the first to |cos| >= min_seed_cos and
          the mean must have norm >= 1e-6, otherwise the function refuses to average them.
      D3  the merged mean_gap is the mean of the aligned per-seed gaps divided by the norm of the un-normalised mean; this is
          exact for untrimmed gaps only, so for several seeds trained with trim > 0 the gap is reported as None
          (meta["mean_gap_available"] False) instead of a wrong number (round 15 §3.4). One seed returns its stored gap
          bit for bit (round 15 §4).
    Round 15 §3.3: directions and gaps must be finite, per file and after the merge; NaN / Inf pass a `norm < eps` test.
    `align_to` (e.g. the mean-difference direction) only chooses the SIGN of the merged direction and gap for display, so
    true = positive. With one seed the direction and the gap are identical to the previous implementation.
    """
    import glob
    import os
    import torch
    if rank != 1:
        raise ValueError("load_das_direction returns one direction; rank=%d checkpoints need a subspace loader (review round 14, D2)" % rank)
    pat = re.compile(r"^das_%s_L%d_r%d_s(\d+)%s\.pt$" % (re.escape(tag), layer, rank, ("_" + re.escape(variant)) if variant else ""))
    files = sorted(f for f in glob.glob(os.path.join(das_dir, "das_%s_L%d_r%d_s*.pt" % (tag, layer, rank))) if pat.match(os.path.basename(f)))
    if not files:
        return None, None
    ws, gaps, losses, seeds, n_pairs, protocols, fields = [], [], [], [], [], [], []
    for f in files:
        name = os.path.basename(f)
        res = torch.load(f, map_location="cpu", weights_only=False)
        W = np.asarray(res["W"], dtype=np.float64)
        if W.ndim != 2 or W.shape[1] != 1:
            raise ValueError("%s holds a basis of shape %s, not a rank-1 direction" % (name, list(W.shape)))
        if not np.all(np.isfinite(W)):
            raise ValueError("%s holds a non-finite direction" % name)
        if np.linalg.norm(W[:, 0]) < 1e-6:
            raise ValueError("%s holds a zero direction" % name)
        g = float(np.asarray(res["mean_gap"], dtype=np.float64).reshape(-1)[0])
        if not math.isfinite(g):
            raise ValueError("%s records a non-finite mean_gap (%r)" % (name, g))
        final_loss = float(res["losses"][-1])
        if not math.isfinite(final_loss):
            raise ValueError("%s records a non-finite final training loss (%r): the training diverged" % (name, final_loss))
        ws.append(unit(W[:, 0]))
        gaps.append(g)
        losses.append(final_loss)
        seeds.append(int(res.get("seed", -1)))
        n_pairs.append(int(res.get("n_pairs", -1)))
        protocols.append({k: _plain(res[k]) for k in DAS_PROTOCOL_KEYS if k in res})
        fields.append({k: _plain(v) for k, v in res.items() if k not in DAS_PER_SEED_KEYS})
    protocol_complete = all(k in protocols[0] for k in DAS_PROTOCOL_KEYS)
    if len(files) > 1:                                                      # D5: full protocol in every file, then every field key by key
        for f, p in zip(files, protocols):
            missing = [k for k in DAS_PROTOCOL_KEYS if k not in p]
            if missing:
                raise ValueError("%s lacks the protocol fields %s; a multi-seed merge with an unverified protocol is refused "
                                 "(load that seed alone, or retrain with the fields recorded)" % (os.path.basename(f), missing))
        for f, fl in zip(files[1:], fields[1:]):
            for k in sorted(set(fl) | set(fields[0])):
                if k not in fl or k not in fields[0] or fl[k] != fields[0][k]:
                    raise ValueError("DAS seeds %s and %s differ in %s (%r vs %r); refusing to merge them"
                                     % (os.path.basename(files[0]), os.path.basename(f), k, fields[0].get(k, "<missing>"), fl.get(k, "<missing>")))
    signs = [1.0 if float(np.dot(w, ws[0])) >= 0 else -1.0 for w in ws]          # D1: align to the first seed, inside the subspace
    aligned = [s * w for s, w in zip(signs, ws)]
    cos_first = [float(np.dot(a, aligned[0])) for a in aligned]
    if min(cos_first) < min_seed_cos:
        raise ValueError("DAS seeds at %s L%d disagree (cos with the first seed %s < %.2f); refusing to average them"
                         % (tag, layer, ["%.3f" % c for c in cos_first], min_seed_cos))
    mean_vec = np.mean(aligned, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm < 1e-6:
        raise ValueError("DAS seeds at %s L%d average to a zero vector" % (tag, layer))
    w_mean = mean_vec / norm
    if not np.all(np.isfinite(w_mean)):
        raise ValueError("DAS seeds at %s L%d merge to a non-finite direction" % (tag, layer))
    trim = float(protocols[0].get("trim", 0.0) or 0.0)
    if len(ws) == 1:
        gap = signs[0] * gaps[0]                                            # one seed: the stored gap, bit for bit
        gap_note = "single seed: the stored mean_gap"
    elif trim > 0:
        gap = None                                                          # D3 (round 15 §3.4): trimming is not linear
        gap_note = ("unavailable: %d seeds trained with trim=%g; per-seed trimmed gaps cannot be projected onto the merged "
                    "direction, recompute the trimmed gap from activations along it" % (len(ws), trim))
    else:
        gap = float(np.mean([s * g for s, g in zip(signs, gaps)])) / norm    # D3: the gap along the merged unit direction (exact: untrimmed)
        gap_note = "mean of aligned per-seed gaps / norm of the un-normalised mean (exact for untrimmed gaps)"
    if gap is not None and not math.isfinite(gap):
        raise ValueError("DAS seeds at %s L%d merge to a non-finite mean_gap" % (tag, layer))
    flip = 1.0
    if align_to is not None:                                                # display sign only: true = positive
        flip = 1.0 if float(np.dot(w_mean, unit(align_to))) >= 0 else -1.0
        w_mean = flip * w_mean
        gap = flip * gap if gap is not None else None
    meta = {"n_seeds": len(ws), "seeds": seeds, "final_losses": losses, "n_pairs": n_pairs, "mean_gap": gap,
            "mean_gap_available": gap is not None, "mean_gap_note": gap_note,
            "cos_seed_vs_mean": [float(np.dot(flip * a, w_mean)) for a in aligned],
            "cos_with_align_to": float(np.dot(w_mean, unit(align_to))) if align_to is not None else None,
            "protocol": protocols[0], "protocol_complete": protocol_complete,
            "merge": "seeds sign-aligned to the first seed, averaged, renormalised; align_to sets the final sign only; "
                     "all non-per-seed checkpoint fields must agree; mean_gap: see mean_gap_note",
            "files": [os.path.basename(f) for f in files]}
    return w_mean, meta
