"""Patient bootstrap and prespecified clinical utility summaries."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, confusion_matrix, roc_curve)


def sensitivity_threshold(y, p, target):
    """Highest threshold meeting sensitivity target on inner OOF predictions only."""
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    return float(np.max(thresholds[tpr >= target]))


def scores(y, p, decisions=None):
    y, p = np.asarray(y), np.asarray(p)
    d = p >= .5 if decisions is None else np.asarray(decisions)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else np.nan
    return {'auc': roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            'average_precision': average_precision_score(y, p) if y.sum() else np.nan,
            'brier': brier_score_loss(y, p), 'log_loss': log_loss(y, p, labels=[0, 1]),
            'sensitivity': div(tp, tp + fn), 'specificity': div(tn, tn + fp),
            'ppv': div(tp, tp + fp), 'npv': div(tn, tn + fn),
            'flagged_fraction': float(d.mean())}


def calibration(y, p, bins=10):
    """Descriptive OOF calibration slope/intercept; never used to recalibrate OOF."""
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(x, y)
    edges = np.linspace(0, 1, bins + 1)
    groups = np.minimum(np.digitize(p, edges) - 1, bins - 1)
    rows = []
    for i in range(bins):
        keep = groups == i
        if keep.any():
            rows.append({'bin': i, 'n': int(keep.sum()), 'predicted': float(p[keep].mean()),
                         'observed': float(y[keep].mean())})
    ece = sum(r['n'] * abs(r['predicted'] - r['observed']) for r in rows) / len(y)
    return {'calibration_intercept': float(m.intercept_[0]),
            'calibration_slope': float(m.coef_[0, 0]), 'ece': float(ece)}, rows


def decision_curve(y, p, thresholds=(.05, .1, .15, .2, .3)):
    rows = []
    for t in thresholds:
        d = p >= t
        nb = np.mean(d & (y == 1)) - np.mean(d & (y == 0)) * t / (1 - t)
        rows.append({'threshold': t, 'net_benefit': float(nb),
                     'treat_all': float(y.mean() - (1 - y.mean()) * t / (1 - t)),
                     'treat_none': 0.})
    return rows


def bootstrap(y, p, decisions, repeats=2000, seed=2026, costs=None, thresholds=(.05, .1, .2, .3)):
    """Ordinary patient bootstrap, conditional on fixed cross-fitted predictions.

    One row per patient is enforced at loading. Threshold decisions stay fixed.
    Draws missing a class yield undefined AUC, retained for other metrics.
    """
    rng = np.random.default_rng(seed)
    draws = {}
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        s = scores(y[idx], p[idx], decisions[idx])
        for row in decision_curve(y[idx], p[idx], thresholds):
            s[f"net_benefit_at_{row['threshold']:g}"] = row['net_benefit']
        if len(np.unique(y[idx])) == 2:
            cal, _ = calibration(y[idx], p[idx])
            s.update(cal)
        else:
            s.update({k: np.nan for k in ['calibration_intercept', 'calibration_slope', 'ece']})
        if costs is not None:
            s['mean_cost'] = float(costs[idx].mean())
        for key, value in s.items():
            draws.setdefault(key, []).append(value)
    return {key: {'low': float(np.nanquantile(v, .025)),
                  'high': float(np.nanquantile(v, .975)),
                  'valid_draws': int(np.isfinite(v).sum())} for key, v in draws.items()}


def paired_bootstrap(y, a, b, repeats=2000, seed=2026):
    """Difference B minus A, identical resampled patient indices for both models."""
    rng = np.random.default_rng(seed)
    metrics = ['auc', 'average_precision', 'brier']
    differences = {k: [] for k in metrics}
    point_a, point_b = scores(y, a), scores(y, b)
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        sa, sb = scores(y[idx], a[idx]), scores(y[idx], b[idx])
        for key in metrics:
            differences[key].append(sb[key] - sa[key])
    result = {}
    for key, values in differences.items():
        result.update({f'{key}_difference_b_minus_a': point_b[key] - point_a[key],
                       f'{key}_low': float(np.nanquantile(values, .025)),
                       f'{key}_high': float(np.nanquantile(values, .975))})
    return result
