"""LA-Score: training-free layer selection (plan Sec. 3.3).

For each layer l, from cached last-token activations of the two contrast
conditions (pos / neg), compute

  dact(l)  = || mean(H_pos) - mean(H_neg) || / mean ||h||        normalised residual gap
  dattn(l) = same on the attention-output stream (optional)
  sep(l)   = k-fold logistic-probe AUROC                           how readable the contrast is
  mass(l)  = fraction of samples whose projection on the mean-diff direction
             lies within +-eps*sigma of the class midpoint          how much is "movable"

  S(l) = z(dact) + z(dattn) + sep(l) + lambda * z(mass(l))

Read layer  R* = argmax sep   (best place to *read* "does the model know")
Write layer W* = argmax S     (predicted best place to *intervene*)
Delta = |R* - W*| is reported as a finding either way.

Everything here is numpy; activations come from model_io.capture_last_token.
"""
import numpy as np

from .directions import cv_auc, unit


def _zscore(values):
    v = np.asarray(values, dtype=np.float64)
    m = np.isfinite(v)
    out = np.full_like(v, np.nan)
    if m.sum() >= 2 and np.nanstd(v[m]) > 0:
        out[m] = (v[m] - v[m].mean()) / v[m].std()
    elif m.sum() >= 1:
        out[m] = 0.0
    return out


def boundary_mass(H_pos, H_neg, eps_sigma=0.5):
    H_pos, H_neg = np.asarray(H_pos, dtype=np.float64), np.asarray(H_neg, dtype=np.float64)
    mu_p, mu_n = H_pos.mean(0), H_neg.mean(0)
    w = unit(mu_p - mu_n)
    center = (mu_p + mu_n) / 2
    z = np.concatenate([(H_pos - center) @ w, (H_neg - center) @ w])
    sigma = z.std() + 1e-12
    return float(np.mean(np.abs(z) < eps_sigma * sigma)), float(sigma)


def layer_scores(acts_pos, acts_neg, attn_pos=None, attn_neg=None, eps_sigma=0.5, lam=1.0,
                 k_folds=5, l2=1.0, seed=0):
    """acts_*: dict layer -> array [n, d]. attn_*: optional dicts with the same keys.

    Returns dict with per-layer rows and the selected read/write layers.
    """
    layers = sorted(set(acts_pos) & set(acts_neg))
    rows = {}
    for l in layers:
        Hp, Hn = np.asarray(acts_pos[l], dtype=np.float64), np.asarray(acts_neg[l], dtype=np.float64)
        scale = np.linalg.norm(np.vstack([Hp, Hn]), axis=1).mean() + 1e-12
        dact = float(np.linalg.norm(Hp.mean(0) - Hn.mean(0)) / scale)
        if attn_pos is not None and attn_neg is not None and l in attn_pos and l in attn_neg:
            Ap, An = np.asarray(attn_pos[l], dtype=np.float64), np.asarray(attn_neg[l], dtype=np.float64)
            ascale = np.linalg.norm(np.vstack([Ap, An]), axis=1).mean() + 1e-12
            dattn = float(np.linalg.norm(Ap.mean(0) - An.mean(0)) / ascale)
        else:
            dattn = float("nan")
        sep_raw, _ = cv_auc(Hp, Hn, k=k_folds, l2=l2, seed=seed)
        # held-out AUC below chance on no-signal layers is the known leave-out bias of CV estimates
        # (training-fold class means shift away from held-out samples); floor at chance = "no decodable signal".
        sep = max(float(sep_raw), 0.5)
        mass, sigma = boundary_mass(Hp, Hn, eps_sigma)
        rows[l] = {"dact": dact, "dattn": dattn, "sep": sep, "sep_raw": float(sep_raw), "mass": mass, "proj_sigma": sigma}

    z_dact = _zscore([rows[l]["dact"] for l in layers])
    z_dattn = _zscore([rows[l]["dattn"] for l in layers])
    z_mass = _zscore([rows[l]["mass"] for l in layers])
    for i, l in enumerate(layers):
        zd = 0.0 if np.isnan(z_dattn[i]) else float(z_dattn[i])
        rows[l]["z_dact"], rows[l]["z_dattn"], rows[l]["z_mass"] = float(z_dact[i]), zd, float(z_mass[i])
        rows[l]["S"] = float(z_dact[i]) + zd + rows[l]["sep"] + lam * float(z_mass[i])

    read_layer = max(layers, key=lambda l: rows[l]["sep"])
    write_layer = max(layers, key=lambda l: rows[l]["S"])
    return {"layers": layers, "rows": rows, "read_layer": read_layer, "write_layer": write_layer,
            "delta_read_write": abs(read_layer - write_layer), "lam": lam, "eps_sigma": eps_sigma}


def top_k_layers(scores, k=3, key="S"):
    rows = scores["rows"]
    return sorted(scores["layers"], key=lambda l: rows[l][key], reverse=True)[:k]


def selection_error(predicted_layer, oracle_gains):
    """|predicted - argmax oracle| in layers, plus Spearman(S, gain) is computed by caller."""
    best = max(oracle_gains, key=lambda l: oracle_gains[l])
    return abs(predicted_layer - best), best
