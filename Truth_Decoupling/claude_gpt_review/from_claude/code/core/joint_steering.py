"""E26 joint interventions. No labels/references enter an inference-time score.

C is estimated from the continuous first-token Yes-minus-No logit margin on
estimation prompts, then made orthogonal to the same-layer E of each method.
This is within-layer orthogonalization, not an independence theorem.
"""
import hashlib
import json
import math

import numpy as np

from .intervene import SteerSpec


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def unit(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-12:
        raise ValueError("direction must be a finite nonzero vector")
    return a / np.linalg.norm(a)


def commitment_direction(states, margins, evidence):
    """Continuous score covariance, projected off E; scale = projection SD.

    Correctness labels are deliberately absent from this function. Positive C
    follows a higher estimation-set Yes-No margin. No signal -> unavailable.
    """
    h, m, e = np.asarray(states, float), np.asarray(margins, float), unit(evidence)
    if h.ndim != 2 or m.shape != (len(h),) or h.shape[1] != len(e) or len(h) < 2:
        raise ValueError("need at least two aligned estimation states/margins")
    if not np.isfinite(h).all() or not np.isfinite(m).all():
        raise ValueError("nonfinite commitment estimation inputs")
    x, y = h - h.mean(0), m - m.mean()
    if np.std(y) < 1e-10:
        return None, {"available": False, "reason": "constant_yes_no_margin"}
    raw = x.T @ (y / np.std(y)) / len(h)
    projected = raw - np.dot(raw, e) * e
    if np.linalg.norm(projected) < 1e-10:
        return None, {"available": False, "reason": "no_component_orthogonal_to_E"}
    w = unit(projected)
    coordinate = x @ w
    scale = float(np.std(coordinate))
    if scale < 1e-10:
        return None, {"available": False, "reason": "zero_projection_variation"}
    corr = float(np.corrcoef(coordinate, m)[0, 1])
    if not math.isfinite(corr):
        raise ValueError("nonfinite C-score correlation")
    return w, {"available": True, "scale": scale, "score_correlation": corr,
               "cos_after": float(w @ e), "n_estimation": len(h),
               "definition": "covariance(hidden, standardized first Yes-No logit margin), projected off same-layer E"}


def candidate_conditions(task, e_choices, c_layers, c_alphas):
    """All four intervention families; E-only and C-only are tuned separately."""
    if task not in ("tf", "qa"):
        raise ValueError("unknown task")
    out = [{"name": "baseline", "kind": "base", "task": task, "method": "base",
            "e": None, "c": None}]
    for method in sorted(e_choices):
        choices = e_choices[method]
        for i, e in enumerate(choices):
            out.append({"name": method + "_E_%d" % i, "kind": "E", "task": task,
                        "method": method, "e": dict(e), "c": None})
        for layer in c_layers.get(method, []):
            for alpha in c_alphas:
                if not math.isfinite(float(alpha)) or float(alpha) == 0:
                    raise ValueError("C alpha must be finite and nonzero")
                c = {"layer": int(layer), "alpha": float(alpha)}
                suffix = "L%d_a%g" % (layer, alpha)
                out.append({"name": method + "_C_" + suffix, "kind": "C", "task": task,
                            "method": method, "e": None, "c": c})
                for i, e in enumerate(choices):
                    out.append({"name": method + "_EC_%d_" % i + suffix, "kind": "EC",
                                "task": task, "method": method, "e": dict(e), "c": c})
    if len({c['name'] for c in out}) != len(out):
        raise ValueError("duplicate condition names")
    return out


def intervention_specs(bank, condition, context="answer"):
    """Construct E/C specs; model layer order determines physical write order.

    Listing E first only controls registration order if both specs share a
    layer. At different layers, the shallower intervention executes first.
    C writes prompt end only; QA E may write each step.

    Gate prompts have their own estimation-set centres. The E vector and C
    vector are the same in gate/answer contexts; their efficacy is empirical.
    """
    if context not in ("answer", "gate"):
        raise ValueError("invalid write context")
    task, method = condition['task'], condition['method']
    result = []
    if condition.get('e'):
        e = condition['e']
        layers = e.get('layers', [e.get('layer')])
        if len(set(layers)) != len(layers) or any(type(l) is not int or l < 0 for l in layers):
            raise ValueError("invalid E layers")
        alpha = float(e['alpha'])
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("invalid E alpha")
        mode = e.get('mode', 'continuous' if task == 'qa' else 'prefill')
        if mode not in ('prefill', 'continuous'):
            raise ValueError("invalid E write mode")
        for layer in layers:
            key = '%s_%s_E_L%d' % (task, method, layer)
            centre = '%s_%s_center_L%d' % (task, 'gate' if context == 'gate' else 'prompt', layer)
            continuous = task == 'qa' and context == 'answer' and mode == 'continuous'
            result.append(SteerSpec(unit(bank[key]), layer, mode='amplify', alpha=alpha,
                center=bank[centre], decode_center=bank['qa_decode_center_L%d' % layer] if continuous else None,
                max_calls=None if continuous else 1, name=condition['name'] + '_E_L%d' % layer))
    if condition.get('c'):
        c = condition['c']; layer = c['layer']
        prefix = '%s_%s_C_L%d' % (task, method, layer)
        result.append(SteerSpec(unit(bank[prefix]), layer, mode='add', alpha=float(c['alpha']),
            coef=[float(np.asarray(bank[prefix + '_scale']).item())], max_calls=1,
            name=condition['name'] + '_C_L%d' % layer))
    return result


def choose_conditions(rows_by_condition, conditions, coverages, seed=42):
    """Choose within each (method, family) using tuning outcomes only.

    For each target quota, maximize delivered correct count; ties prefer less
    intervention (absolute amplitudes, then number/layer indices, then name).
    An infeasible family remains explicit None, never disappears from reports.
    """
    from .selective_eval import quota_select, validate_arms
    validate_arms(rows_by_condition)
    specs = {c['name']: c for c in conditions}
    if set(specs) != set(rows_by_condition):
        raise ValueError("candidate outputs and definitions differ")
    methods = sorted({c['method'] for c in conditions if c['kind'] != 'base'})
    families = [(method, kind) for method in methods for kind in ('E', 'C', 'EC')]
    result = {}
    for coverage in coverages:
        point = {}
        for method, kind in families:
            feasible = []
            for name, spec in specs.items():
                if (spec['method'], spec['kind']) != (method, kind):
                    continue
                scored = quota_select(rows_by_condition[name], coverage, seed=seed)
                if not scored['feasible']:
                    continue
                e, c = spec.get('e') or {}, spec.get('c') or {}
                layers = e.get('layers', [e['layer']] if 'layer' in e else []) + ([c['layer']] if c else [])
                key = (-scored['correct'], abs(e.get('alpha', 0)) + abs(c.get('alpha', 0)),
                       len(layers), tuple(layers), name)
                feasible.append((key, name))
            point[method + '_' + kind] = min(feasible)[1] if feasible else None
        result[str(coverage)] = point
    return result
