"""Time-based train / validation / test split.

Why time-based? A fraud model is always trained on the past and used on the
future. A random split lets the model see transactions from the same fraud
burst (same card, same hour) in both train and test, which inflates results
and hides drift.
"""
from __future__ import annotations

import pandas as pd

from . import config


def compute_cutoffs(ts: pd.Series, fractions=config.SPLIT_FRACTIONS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (validation_start, test_start), snapped to midnight.

    The cut-offs are the 60th and 80th percentiles of transaction time, rounded
    to the nearest day boundary so that no calendar day straddles two sets.
    """
    q1 = ts.quantile(fractions[0])
    q2 = ts.quantile(fractions[0] + fractions[1])
    return q1.round("D"), q2.round("D")


def assign_split(df: pd.DataFrame, val_start: pd.Timestamp, test_start: pd.Timestamp) -> pd.Series:
    split = pd.Series("train", index=df.index)
    split[df["ts"] >= val_start] = "validation"
    split[df["ts"] >= test_start] = "test"
    return split


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split", sort=False)
    out = pd.DataFrame({
        "start": g["ts"].min().dt.date,
        "end": g["ts"].max().dt.date,
        "days": g["ts"].apply(lambda s: s.dt.normalize().nunique()),
        "transactions": g.size(),
        "frauds": g["is_fraud"].sum(),
        "fraud_rate_pct": (g["is_fraud"].mean() * 100).round(3),
        "fraud_amount_usd": g.apply(lambda x: x.loc[x.is_fraud == 1, "amt"].sum(), include_groups=False).round(0),
    })
    out["share_of_rows_pct"] = (out["transactions"] / out["transactions"].sum() * 100).round(1)
    return out.loc[["train", "validation", "test"]]
