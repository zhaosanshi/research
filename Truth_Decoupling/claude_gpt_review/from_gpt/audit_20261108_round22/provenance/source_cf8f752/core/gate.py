"""Per-input gating (plan Sec. 3.5).

A knowledge probe p(x) = sigma(theta . h_R*(x) + b) trained at the read layer
predicts whether the model will answer x correctly. Inputs are then routed:

  p >= tau_hi            known          -> no intervention
  tau_lo <= p < tau_hi   partly known   -> evidence direction, alpha_E(x) = aE_max (tau_hi - p)/(tau_hi - tau_lo)
  p <  tau_lo            unknown        -> commitment direction, alpha_C(x) = aC_max (tau_lo - p)/tau_lo

Thresholds are chosen on dev under constraints (defaults from the plan):
  delta accuracy on known items >= -1 point, delta answer rate <= +5 points,
maximising the overall correct rate.

The gate itself is model-free numpy; the probe uses directions.fit_logreg.
"""
import numpy as np

from .directions import fit_logreg, logreg_predict_proba


class KnowledgeProbe:
    def __init__(self, l2=1.0, standardize=True):
        self.l2, self.standardize, self.fit_ = l2, standardize, None

    def fit(self, H, correct):
        self.fit_ = fit_logreg(np.asarray(H), np.asarray(correct, dtype=float), l2=self.l2, standardize=self.standardize)
        return self

    def predict_proba(self, H):
        if self.fit_ is None:
            raise RuntimeError("probe not fitted")
        return logreg_predict_proba(self.fit_, np.asarray(H))

    def direction(self):
        w = self.fit_["w"]
        return w / (np.linalg.norm(w) + 1e-12)


def gate_alphas(p, tau_lo, tau_hi, alpha_e_max=1.0, alpha_c_max=1.0):
    """Vectorised routing. Returns (alpha_E, alpha_C, zone) arrays; zone in {known, partial, unknown}."""
    p = np.asarray(p, dtype=np.float64)
    if not (0.0 < tau_lo < tau_hi <= 1.0):
        raise ValueError("need 0 < tau_lo < tau_hi <= 1, got %r %r" % (tau_lo, tau_hi))
    alpha_e = np.zeros_like(p)
    alpha_c = np.zeros_like(p)
    zone = np.full(p.shape, "known", dtype=object)
    mid = (p >= tau_lo) & (p < tau_hi)
    low = p < tau_lo
    alpha_e[mid] = alpha_e_max * (tau_hi - p[mid]) / (tau_hi - tau_lo)
    alpha_c[low] = alpha_c_max * (tau_lo - p[low]) / tau_lo
    zone[mid] = "partial"
    zone[low] = "unknown"
    return alpha_e, alpha_c, zone


def nearest_level(values, levels):
    """Map continuous alphas to the nearest pre-computed grid level (for dev simulation)."""
    levels = np.asarray(sorted(levels), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    idx = np.abs(values[:, None] - levels[None, :]).argmin(axis=1)
    return levels[idx]


def simulate_gate(p, grid_outcomes, base_outcomes, tau_lo, tau_hi, alpha_e_max, alpha_c_max,
                  e_levels, c_levels):
    """Simulate the gate on dev from pre-computed outcomes.

    grid_outcomes[(aE, aC)] -> list of QA outcomes ("correct"/"wrong"/"abstain") for every item
    at that (alpha_E, alpha_C) setting; (0,0) must be present and equal base_outcomes.
    Returns the simulated outcome list.
    """
    alpha_e, alpha_c, zone = gate_alphas(p, tau_lo, tau_hi, alpha_e_max, alpha_c_max)
    ae = nearest_level(alpha_e, e_levels)
    ac = nearest_level(alpha_c, c_levels)
    out = []
    for i in range(len(p)):
        key = (float(ae[i]), float(ac[i]))
        if key not in grid_outcomes:
            raise KeyError("grid missing %r" % (key,))
        out.append(grid_outcomes[key][i])
    return out


def select_thresholds(p, grid_outcomes, base_outcomes, alpha_e_max, alpha_c_max, e_levels, c_levels,
                      tau_grid=None, max_known_drop_pts=1.0, max_answer_rate_rise_pts=5.0):
    """Grid-search (tau_lo, tau_hi) on dev under the plan's constraints."""
    from .metrics import QA_CORRECT, qa_rates, known_unknown_split
    p = np.asarray(p, dtype=np.float64)
    base_rates = qa_rates(base_outcomes)
    known, _ = known_unknown_split(base_outcomes)
    base_known_acc = np.mean([base_outcomes[i] == QA_CORRECT for i in known]) if known else float("nan")
    tau_grid = tau_grid or [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    best, table = None, []
    for lo in tau_grid:
        for hi in tau_grid:
            if not lo < hi:
                continue
            sim = simulate_gate(p, grid_outcomes, base_outcomes, lo, hi, alpha_e_max, alpha_c_max, e_levels, c_levels)
            rates = qa_rates(sim)
            known_acc = np.mean([sim[i] == QA_CORRECT for i in known]) if known else float("nan")
            d_known = 100.0 * (known_acc - base_known_acc) if known else 0.0
            d_rans = 100.0 * (rates["answer_rate"] - base_rates["answer_rate"])
            feasible = (d_known >= -max_known_drop_pts) and (d_rans <= max_answer_rate_rise_pts)
            row = {"tau_lo": lo, "tau_hi": hi, "correct": rates["correct"], "wrong": rates["wrong"],
                   "abstain": rates["abstain"], "d_known_acc_pts": d_known, "d_answer_rate_pts": d_rans, "feasible": feasible}
            table.append(row)
            if feasible and (best is None or row["correct"] > best["correct"] or
                             (row["correct"] == best["correct"] and row["wrong"] < best["wrong"])):
                best = row
    return best, table
