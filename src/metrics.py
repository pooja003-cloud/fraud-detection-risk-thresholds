"""Evaluation metrics for imbalanced fraud detection.

PR-AUC (average precision) is the primary metric: with ~0.5% fraud, ROC-AUC
and accuracy look excellent even for weak models, because true negatives
dominate. Precision and recall describe what a fraud team actually feels:
how many alerts are real, and how much fraud is caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y, score) -> float:
    return float(average_precision_score(y, score))


def roc_auc(y, score) -> float:
    return float(roc_auc_score(y, score))


def at_threshold(y, score, amt, threshold: float, n_days: int) -> dict:
    """Operational metrics when alerting on score >= threshold."""
    y = np.asarray(y).astype(bool)
    alert = np.asarray(score) >= threshold
    amt = np.asarray(amt, dtype=float)
    tp = int((alert & y).sum()); fp = int((alert & ~y).sum())
    fn = int((~alert & y).sum()); tn = int((~alert & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "alerts": tp + fp,
        "alerts_per_day": (tp + fp) / n_days,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fraud_caught_usd": float(amt[alert & y].sum()),
        "fraud_missed_usd": float(amt[~alert & y].sum()),
        "dollar_recall": float(amt[alert & y].sum() / amt[y].sum()) if y.any() else 0.0,
        "false_alerts_per_day": fp / n_days,
    }


def best_f1_threshold(y, score) -> float:
    """Threshold that maximises F1 (used only as a neutral operating point for comparing models)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def threshold_for_alert_volume(score, n_alerts: int) -> float:
    """Lowest threshold that produces at most n_alerts alerts."""
    s = np.sort(np.asarray(score))[::-1]
    if n_alerts <= 0:
        return float(np.inf)
    if n_alerts >= len(s):
        return float(s[-1])
    # alerts are score >= t; step just above the (n+1)-th score so ties do not overflow
    return float(np.nextafter(s[n_alerts], np.inf)) if s[n_alerts] == s[n_alerts - 1] else float(s[n_alerts - 1])


def n_days(ts: pd.Series) -> int:
    return int(pd.to_datetime(ts).dt.normalize().nunique())


def card_bootstrap_indices(cards: pd.Series, n_boot: int, seed: int) -> list[np.ndarray]:
    """Cluster bootstrap: resample whole cards with replacement.

    Frauds arrive in bursts on the same card, so rows are not independent.
    Resampling rows would understate uncertainty; resampling cards does not.
    """
    codes, uniques = pd.factorize(cards)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniques))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[i] for i in pick]))
    return out


def bootstrap_matrix(y, scores: dict[str, np.ndarray], boot_idx: list[np.ndarray]) -> pd.DataFrame:
    """PR-AUC of every model on every bootstrap resample (rows = resamples, paired across models)."""
    y = np.asarray(y)
    return pd.DataFrame({name: [average_precision_score(y[i], np.asarray(s)[i]) for i in boot_idx]
                         for name, s in scores.items()})


def summarise_bootstrap(y, scores: dict[str, np.ndarray], boot: pd.DataFrame) -> pd.DataFrame:
    """Point PR-AUC plus a 95% card-bootstrap interval per model."""
    return pd.DataFrame({
        "model": list(scores),
        "pr_auc": [pr_auc(y, s) for s in scores.values()],
        "ci_low": [np.percentile(boot[m], 2.5) for m in scores],
        "ci_high": [np.percentile(boot[m], 97.5) for m in scores],
    })


def paired_diff(boot: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """95% interval for PR-AUC(a) - PR-AUC(b) from paired bootstrap resamples."""
    d = boot[a] - boot[b]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
