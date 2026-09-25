"""Independent re-computation of SQL window features, used to prove there is no look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd


def recompute_velocity(df: pd.DataFrame, cards: list[str]) -> pd.DataFrame:
    """Recompute prior-1h / prior-24h counts and prior average amount in NumPy.

    Uses only transactions strictly earlier (by whole seconds) than the current
    one, which is the same definition as the SQL frames
    'RANGE BETWEEN n PRECEDING AND 1 PRECEDING'.
    """
    rows = []
    for card in cards:
        g = df[df["cc_num"] == card].sort_values("ts")
        t = ((g["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy()
        amt = g["amt"].to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(amt)])
        n_before = np.searchsorted(t, t, side="left")          # strictly earlier seconds
        c1h = n_before - np.searchsorted(t, t - 3600, side="left")
        c24h = n_before - np.searchsorted(t, t - 86400, side="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(n_before > 0, csum[n_before] / np.maximum(n_before, 1), np.nan)
        rows.append(pd.DataFrame({
            "trans_num": g["trans_num"].to_numpy(),
            "chk_n_txn_prior_1h": c1h,
            "chk_n_txn_prior_24h": c24h,
            "chk_cust_avg_amt_prior": avg,
        }))
    return pd.concat(rows, ignore_index=True)


def compare_with_sql(df: pd.DataFrame, cards: list[str]) -> pd.DataFrame:
    chk = recompute_velocity(df, cards).merge(
        df[["trans_num", "n_txn_prior_1h", "n_txn_prior_24h", "cust_avg_amt_prior"]], on="trans_num")
    return pd.DataFrame({
        "feature": ["n_txn_prior_1h", "n_txn_prior_24h", "cust_avg_amt_prior"],
        "rows_checked": len(chk),
        "mismatches": [
            int((chk["chk_n_txn_prior_1h"] != chk["n_txn_prior_1h"]).sum()),
            int((chk["chk_n_txn_prior_24h"] != chk["n_txn_prior_24h"]).sum()),
            int((~np.isclose(chk["chk_cust_avg_amt_prior"], chk["cust_avg_amt_prior"], equal_nan=True)).sum()),
        ],
    })
