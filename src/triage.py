"""Alert triage: map model scores to operational queues.

Bands (highest score first):
  auto-hold        score >= t_hold      transaction held; customer contacted to confirm
  priority review  t_review <= score < t_hold   reviewed first, same-day SLA
  standard review  t_standard <= score < t_review   reviewed with spare capacity
  monitor only     t_monitor <= score < t_standard  no alert; logged for look-back and drift analysis
  no action        score < t_monitor

The band cut-offs are chosen on validation (notebook 04) from precision
targets and the cost/capacity analysis in notebook 03.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = ["Auto-hold", "Priority review", "Standard review", "Monitor only", "No action"]


def assign_band(score, cutoffs: dict) -> np.ndarray:
    s = np.asarray(score)
    return np.select(
        [s >= cutoffs["hold"], s >= cutoffs["review"], s >= cutoffs["standard"], s >= cutoffs["monitor"]],
        BANDS[:4], default=BANDS[4],
    )


def lowest_threshold_with_precision(y, score, target: float, min_alerts: int = 20) -> float:
    """Lowest threshold at which the precision of everything above it is still >= target."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score)
    order = np.argsort(-s)
    cum_prec = np.cumsum(y[order]) / np.arange(1, len(s) + 1)
    ok = np.where((cum_prec >= target) & (np.arange(1, len(s) + 1) >= min_alerts))[0]
    if len(ok) == 0:
        return float(s.max())
    return float(s[order][ok.max()])


def band_table(df: pd.DataFrame, band_col: str, n_days: int) -> pd.DataFrame:
    g = df.groupby(band_col)
    out = pd.DataFrame({
        "transactions": g.size(),
        "per_day": g.size() / n_days,
        "frauds": g["is_fraud"].sum(),
        "fraud_rate_in_band": g["is_fraud"].mean(),
        "fraud_usd": g.apply(lambda x: x.loc[x.is_fraud == 1, "amt"].sum(), include_groups=False),
    }).reindex(BANDS).fillna(0)
    out.index.name = "band"
    out["share_of_all_fraud"] = out["frauds"] / out["frauds"].sum()
    return out
