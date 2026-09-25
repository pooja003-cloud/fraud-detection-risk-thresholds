"""Leakage tests for the SQL feature pipeline and the time split.

1. Truncation test: features computed with and without the future must be
   identical for past rows. A perturbation test also changes every later
   amount and checks that earlier features don't move. A mutation check was
   run: a deliberately leaky window (RANGE ... FOLLOWING) fails these tests.
2. Label test: flipping fraud labels must not change any feature (no feature
   may be built from labels).
3. Independent recomputation: the NumPy re-implementation matches SQL.
4. Split test: the split is strictly ordered in time.
"""
import csv
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src import data, features, leakage, split

HEADER = ["", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last", "gender",
          "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num",
          "unix_time", "merch_lat", "merch_long", "is_fraud"]


def _toy_rows(seed=0, n=400, days=10):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n):
        card = f"{1000 + i % 5}"
        ts = start + pd.Timedelta(seconds=int(rng.integers(0, days * 86400)))
        rows.append(["0", ts.strftime("%Y-%m-%d %H:%M:%S"), card, f"fraud_M{i % 7}",
                     ["grocery_pos", "shopping_net", "travel"][i % 3], f"{rng.lognormal(3, 1):.2f}", "A", "B",
                     "F", "1 St", "Town", "NY", "10001", "40.7", "-74.0", "1000", "Job", "1980-01-01",
                     f"t{i:05d}", "0", "40.8", "-74.1", str(int(rng.random() < 0.05))])
    return rows


def _write(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def _features(tmp_path, rows, name):
    a, b = tmp_path / f"{name}_a.csv", tmp_path / f"{name}_b.csv"
    _write(a, rows[:300]); _write(b, rows[300:])
    con = duckdb.connect()
    data.run_pipeline(con, a, b)
    out = con.sql("SELECT * FROM features").df().set_index("trans_num").sort_index()
    con.close()
    return out


FEATURE_COLS = [c for c in features.NUMERIC_FEATURES]


def test_features_are_identical_whether_or_not_the_future_exists(tmp_path):
    """Compute features on the full data and on data truncated at time T.

    For every transaction before T the two must match exactly. Any window
    that peeks forward (even by one second) sees different data in the two
    runs and fails. The toy data is dense (about 40 transactions per card per
    day), so many rows have future transactions within 1 hour.
    """
    rows = _toy_rows(days=2)
    full = _features(tmp_path, rows, "full")
    cutoff = full["ts"].sort_values().iloc[len(full) // 2]
    truncated_rows = [r for r in rows if pd.Timestamp(r[1]) < cutoff]
    trunc = _features(tmp_path, truncated_rows, "trunc")
    assert len(trunc) > 100
    pd.testing.assert_frame_equal(full.loc[trunc.index, FEATURE_COLS], trunc[FEATURE_COLS])


def test_changing_a_future_transaction_does_not_change_past_features(tmp_path):
    rows = _toy_rows(days=2)
    base = _features(tmp_path, rows, "base")
    cutoff = base["ts"].sort_values().iloc[len(base) // 2]
    rows2 = [r.copy() for r in rows]
    for r in rows2:
        if pd.Timestamp(r[1]) >= cutoff:
            r[5] = "99999.99"                       # every future amount becomes huge
    pert = _features(tmp_path, rows2, "pert")
    earlier = base.index[base["ts"] < cutoff]
    pd.testing.assert_frame_equal(base.loc[earlier, FEATURE_COLS], pert.loc[earlier, FEATURE_COLS])


def test_labels_do_not_influence_features(tmp_path):
    rows = _toy_rows()
    base = _features(tmp_path, rows, "base")
    flipped = [r.copy() for r in rows]
    for r in flipped:
        r[22] = str(1 - int(r[22]))
    pert = _features(tmp_path, flipped, "flip")
    pd.testing.assert_frame_equal(base[FEATURE_COLS], pert[FEATURE_COLS])


def test_sql_windows_match_independent_recomputation(tmp_path):
    feats = _features(tmp_path, _toy_rows(), "chk").reset_index()
    feats["ts"] = pd.to_datetime(feats["ts"])
    res = leakage.compare_with_sql(feats, feats["cc_num"].unique().tolist())
    assert res["mismatches"].sum() == 0


def test_same_second_transactions_are_not_visible_to_each_other(tmp_path):
    rows = _toy_rows(n=300)
    same = rows[0].copy(); same[18] = "twin"; same[5] = "5000.00"   # same card and second as rows[0]
    feats = _features(tmp_path, rows + [same], "tie")
    assert feats.loc["twin", "n_txn_prior_1h"] == feats.loc[rows[0][18], "n_txn_prior_1h"]


def test_time_split_is_strictly_ordered():
    ts = pd.Series(pd.date_range("2020-01-01", periods=1000, freq="h"))
    df = pd.DataFrame({"ts": ts, "is_fraud": 0, "amt": 1.0})
    v, t = split.compute_cutoffs(df["ts"])
    df["split"] = split.assign_split(df, v, t)
    g = df.groupby("split")["ts"]
    assert g.max()["train"] < g.min()["validation"] <= g.max()["validation"] < g.min()["test"]
    assert v == v.normalize() and t == t.normalize()


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/processed/splits.parquet").exists(),
                    reason="needs the processed data (run notebook 01)")
def test_real_split_has_no_overlap():
    from src import config
    df = data.load_features().merge(pd.read_parquet(config.DATA_PROCESSED / "splits.parquet"), on="trans_num")
    g = df.groupby("split")["ts"]
    assert g.max()["train"] < g.min()["validation"]
    assert g.max()["validation"] < g.min()["test"]
