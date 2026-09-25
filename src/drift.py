"""Drift and performance monitoring: PSI, and monthly production-style metrics.

PSI (Population Stability Index) compares the distribution of a variable in a
monitoring window with a reference window:

    PSI = sum_over_bins (actual% - expected%) * ln(actual% / expected%)

Common industry reading: < 0.10 stable, 0.10-0.25 moderate shift (investigate),
> 0.25 major shift (act). Our alert trigger uses 0.20 as a slightly earlier
warning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    ref = reference[~np.isnan(reference)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def psi(reference, actual, n_bins: int = 10, categorical: bool | None = None) -> float:
    """PSI with quantile bins from the reference; missing values get their own bin.

    Low-cardinality variables (<= 12 distinct values, e.g. flags, hour bands,
    categories) are compared category by category instead of by quantile.
    """
    ref = pd.Series(reference)
    act = pd.Series(actual)
    if categorical is None:
        categorical = (not pd.api.types.is_numeric_dtype(ref)) or ref.nunique(dropna=True) <= 12
    if categorical:
        e = ref.astype("string").fillna("__missing__").value_counts(normalize=True)
        a = act.astype("string").fillna("__missing__").value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=0).to_numpy(), a.reindex(idx, fill_value=0).to_numpy()
    else:
        r, x = ref.to_numpy(dtype=float), act.to_numpy(dtype=float)
        edges = _bin_edges(r, n_bins)
        e = np.append(np.histogram(r[~np.isnan(r)], edges)[0], np.isnan(r).sum()) / len(r)
        a = np.append(np.histogram(x[~np.isnan(x)], edges)[0], np.isnan(x).sum()) / len(x)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_tail(reference, actual, quantiles=(0.90, 0.95, 0.99, 0.995, 0.999)) -> float:
    """PSI on the high-score tail only: bins at upper quantiles of the reference score.

    Standard 10-bin PSI puts every alerted transaction (top ~1%) in one bin, so it cannot see
    changes among the scores that actually generate alerts. This version can.
    """
    ref = np.asarray(reference, dtype=float); act = np.asarray(actual, dtype=float)
    edges = np.concatenate([[-np.inf], np.quantile(ref, quantiles), [np.inf]])
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(act, edges)[0] / len(act)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "major shift"


def monthly_monitoring(df: pd.DataFrame, reference: pd.DataFrame, score_col: str, threshold: float,
                       features: list[str], period: str = "M") -> pd.DataFrame:
    """One row per calendar period with volume, fraud rate, alerting and drift metrics.

    In production, fraud labels arrive late (chargebacks take weeks), so the
    label-based columns (fraud rate, precision, recall) would only be final
    for periods older than the label-maturity window. Score PSI, feature PSI,
    alert volume and missing rates are available immediately.
    """
    rows = []
    d = df.copy()
    d["period"] = d["ts"].dt.to_period(period)
    for p, g in d.groupby("period"):
        alert = g[score_col] >= threshold
        y = g["is_fraud"] == 1
        n_days = g["ts"].dt.normalize().nunique()
        tp = int((alert & y).sum()); fp = int((alert & ~y).sum()); fn = int((~alert & y).sum())
        row = {
            "period": str(p),
            "transactions": len(g),
            "days": n_days,
            "fraud_rate_pct": 100 * y.mean(),
            "alerts": int(alert.sum()),
            "alerts_per_day": alert.sum() / n_days,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "psi_score": psi(reference[score_col], g[score_col]),
            "psi_score_tail": psi_tail(reference[score_col], g[score_col]),
            "missing_rate_pct": 100 * g[features].isna().any(axis=1).mean(),
        }
        for f in features:
            row[f"psi_{f}"] = psi(reference[f], g[f])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_triggers(mon: pd.DataFrame, triggers: dict) -> pd.DataFrame:
    """Flag each period against the alert triggers. Returns one row per (period, trigger) that fired."""
    fired = []
    for _, r in mon.iterrows():
        psi_cols = [c for c in mon.columns if c.startswith("psi_") and c not in ("psi_score", "psi_score_tail")]
        checks = {
            "Score PSI > {psi_score}".format(**triggers): r["psi_score"] > triggers["psi_score"],
            "Score-tail PSI > {psi_score}".format(**triggers): r.get("psi_score_tail", 0) > triggers["psi_score"],
            "Any feature PSI > {psi_feature}".format(**triggers): any(r[c] > triggers["psi_feature"] for c in psi_cols),
            "Precision < {precision_min:.0%}".format(**triggers): r["precision"] < triggers["precision_min"],
            "Recall < {recall_min:.0%}".format(**triggers): r["recall"] < triggers["recall_min"],
            "Alerts/day > capacity ({capacity})".format(**triggers): r["alerts_per_day"] > triggers["capacity"],
            "Missing rate > {missing_pct}%".format(**triggers): r["missing_rate_pct"] > triggers["missing_pct"],
        }
        for name, hit in checks.items():
            if hit:
                detail = ""
                if name.startswith("Any feature"):
                    detail = ", ".join(c.replace("psi_", "") for c in psi_cols if r[c] > triggers["psi_feature"])
                fired.append({"period": r["period"], "trigger": name, "detail": detail})
    return pd.DataFrame(fired, columns=["period", "trigger", "detail"])
