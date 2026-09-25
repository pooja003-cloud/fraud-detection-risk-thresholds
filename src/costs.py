"""Business-cost framework for choosing an alert threshold.

Two cost definitions are computed:

  fp_only    Total = FN_loss + FP x review_cost
             (only false alerts are counted as waste; the textbook version)

  all_alerts Total = FN_loss + (TP + FP) x review_cost          <- PRIMARY
             (every alert is reviewed by an analyst, including real frauds,
              so every alert costs analyst time)

FN_loss = sum over missed frauds of amount x (1 - recovery_rate) x loss_multiplier.

Assumption: a fraud that is alerted is stopped before loss (the score is
used at authorisation to hold the transaction for review). Caught fraud
therefore costs only the review. This is optimistic for post-authorisation
review queues, and it is documented as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def sweep(y, score, amt, n_days: int, thresholds=None, review_cost: float = config.REVIEW_COST,
          recovery_rate: float = config.RECOVERY_RATE, loss_multiplier: float = 1.0) -> pd.DataFrame:
    """One row per threshold with volumes, errors, dollars and both cost variants.

    Vectorised with cumulative sums over descending scores, so thousands of
    thresholds on ~370k rows are evaluated in milliseconds.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    amt = np.asarray(amt, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(score, np.linspace(0, 1, 2001)))
    thresholds = np.asarray(thresholds, dtype=float)

    order = np.argsort(-score, kind="stable")
    s_sorted = score[order]
    y_sorted = y[order]
    fraud_amt_sorted = np.where(y_sorted, amt[order], 0.0)
    cum_tp = np.concatenate([[0], np.cumsum(y_sorted)])
    cum_fraud_amt = np.concatenate([[0.0], np.cumsum(fraud_amt_sorted)])

    # number of rows with score >= t  (s_sorted is descending)
    n_alert = np.searchsorted(-s_sorted, -thresholds, side="right")
    tp = cum_tp[n_alert]
    fp = n_alert - tp
    total_fraud = int(y.sum())
    fn = total_fraud - tp
    caught_usd = cum_fraud_amt[n_alert]
    total_fraud_usd = float(amt[y].sum())
    missed_usd = total_fraud_usd - caught_usd
    loss = missed_usd * (1 - recovery_rate) * loss_multiplier

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(n_alert > 0, tp / np.maximum(n_alert, 1), np.nan)
    recall = tp / total_fraud if total_fraud else np.zeros_like(tp, dtype=float)

    out = pd.DataFrame({
        "threshold": thresholds,
        "alerts": n_alert,
        "alerts_per_day": n_alert / n_days,
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision,
        "recall": recall,
        "fraud_caught_usd": caught_usd,
        "fraud_missed_usd": missed_usd,
        "expected_fraud_loss_usd": loss,
        "review_cost_all_alerts_usd": n_alert * review_cost,
        "review_cost_fp_only_usd": fp * review_cost,
    })
    out["total_cost_usd"] = out["expected_fraud_loss_usd"] + out["review_cost_all_alerts_usd"]          # primary
    out["total_cost_fp_only_usd"] = out["expected_fraud_loss_usd"] + out["review_cost_fp_only_usd"]
    return out.sort_values("threshold").reset_index(drop=True)


def cost_minimising(table: pd.DataFrame, col: str = "total_cost_usd") -> pd.Series:
    return table.loc[table[col].idxmin()]


def capacity_constrained(table: pd.DataFrame, capacity_per_day: float = config.DAILY_REVIEW_CAPACITY) -> pd.Series:
    """Fill the review queue: the lowest threshold (highest recall) whose alert volume fits within capacity.

    This is how operations teams usually run a fixed-size team: analysts are a
    fixed cost, so any spare capacity is spent reviewing the next-riskiest
    transactions.
    """
    feasible = table[table["alerts_per_day"] <= capacity_per_day]
    return feasible.loc[feasible["threshold"].idxmin()]


def recall_target(table: pd.DataFrame, target: float = config.RECALL_TARGET) -> pd.Series:
    """Highest threshold (fewest alerts) that still reaches the recall target."""
    ok = table[table["recall"] >= target]
    return ok.loc[ok["threshold"].idxmax()]


def no_model_cost(amt, y, recovery_rate: float = config.RECOVERY_RATE, loss_multiplier: float = 1.0) -> float:
    """Cost if nothing is alerted: every fraud is lost."""
    y = np.asarray(y).astype(bool)
    return float(np.asarray(amt)[y].sum() * (1 - recovery_rate) * loss_multiplier)


def workload(table_rows: pd.DataFrame, n_analysts: int = config.N_ANALYSTS,
             per_analyst: int = config.REVIEWS_PER_ANALYST_PER_DAY) -> pd.DataFrame:
    """Alerts per analyst per day versus capacity, for selected operating points."""
    out = table_rows.copy()
    out["analysts"] = n_analysts
    out["alerts_per_analyst_per_day"] = out["alerts_per_day"] / n_analysts
    out["capacity_per_analyst_per_day"] = per_analyst
    out["utilisation_pct"] = 100 * out["alerts_per_analyst_per_day"] / per_analyst
    out["false_alerts_per_analyst_per_day"] = out["FP"] / out["alerts"].clip(lower=1) * out["alerts_per_analyst_per_day"]
    out["analysts_needed"] = np.ceil(out["alerts_per_day"] / per_analyst).astype(int)
    out["within_capacity"] = out["alerts_per_day"] <= n_analysts * per_analyst
    return out


def sensitivity(y, score, amt, n_days: int, thresholds, multipliers=(0.5, 1.0, 1.5)) -> pd.DataFrame:
    """Re-optimise the cost-minimising threshold when fraud loss or review cost moves +/-50%."""
    rows = []
    for lm in multipliers:
        for rm in multipliers:
            t = sweep(y, score, amt, n_days, thresholds, review_cost=config.REVIEW_COST * rm, loss_multiplier=lm)
            best = cost_minimising(t)
            rows.append({"fraud_loss_multiplier": lm, "review_cost_multiplier": rm,
                         "review_cost_usd": config.REVIEW_COST * rm,
                         "optimal_threshold": best["threshold"], "alerts_per_day": best["alerts_per_day"],
                         "precision": best["precision"], "recall": best["recall"],
                         "total_cost_usd": best["total_cost_usd"]})
    return pd.DataFrame(rows)


def threshold_grid(score) -> np.ndarray:
    """Candidate thresholds: 2,001 quantiles overall plus 2,001 inside the top 5% of scores,
    where operating points actually live (fine resolution at low alert volumes)."""
    score = np.asarray(score, dtype=float)
    top = np.quantile(score, np.linspace(0.95, 1.0, 2001))
    return np.unique(np.concatenate([np.quantile(score, np.linspace(0, 1, 2001)), top]))
