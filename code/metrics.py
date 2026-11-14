"""NumPy binary metrics with exact tie handling; no sklearn runtime dependency."""
import numpy as np


def auc_roc(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != scores.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected equally sized nonempty binary vectors")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.empty(len(y), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def binary_metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    auc = auc_roc(y, p)
    pred = p >= threshold
    tp, tn = int(((y == 1) & pred).sum()), int(((y == 0) & ~pred).sum())
    fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"auc": auc, "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "confusion_matrix": [[tn, fp], [fn, tp]], "count": len(y)}


def summarize(conditions, expected_names):
    if set(conditions) != set(expected_names):
        raise ValueError("Selection requires every validation condition exactly once")
    aucs = [conditions[name]["auc"] for name in expected_names]
    if any(a is None or not np.isfinite(a) for a in aucs):
        raise ValueError("Selection AUC undefined; validation must contain both classes")
    clean = conditions["clean"]["auc"]
    degraded = float(np.mean([conditions[name]["auc"] for name in expected_names if name != "clean"]))
    colour_names = [n for n in expected_names if n.startswith(("brightness_", "contrast_"))]
    colour_gap = clean - float(np.mean([conditions[n]["auc"] for n in colour_names]))
    return {"mean_auc": float(np.mean(aucs)), "minimum_auc": float(np.min(aucs)), "clean_auc": clean,
            "mean_degraded_auc": degraded, "clean_degraded_gap": clean - degraded, "colour_gap": colour_gap}
