"""Run the DuckDB SQL pipeline and load the engineered feature table."""
from __future__ import annotations

import duckdb
import pandas as pd

from . import config


def _read_sql(name: str, **params) -> str:
    text = (config.SQL_DIR / name).read_text()
    for k, v in params.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(config.DUCKDB_PATH), read_only=read_only)


def build_database(force: bool = False) -> None:
    """Execute sql/01 and sql/02 and export the feature table to Parquet."""
    if config.FEATURES_PARQUET.exists() and not force:
        return
    for p in (config.RAW_TRAIN, config.RAW_TEST):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download the Kaggle 'Credit Card Transactions Fraud "
                "Detection Dataset' (kartik2112) and place fraudTrain.csv / fraudTest.csv "
                "in data/raw/ (see README)."
            )
    con = connect()
    run_pipeline(con, config.RAW_TRAIN, config.RAW_TEST)
    con.execute(f"COPY features TO '{config.FEATURES_PARQUET.as_posix()}' (FORMAT PARQUET)")
    con.close()


def run_pipeline(con: duckdb.DuckDBPyConnection, raw_train, raw_test) -> None:
    """Run the cleaning and feature SQL on any DuckDB connection (also used by tests)."""
    con.execute("SET threads TO 2")
    con.execute("SET enable_progress_bar = false")
    con.execute(_read_sql("01_load_and_clean.sql",
                          raw_train=str(raw_train).replace("\\", "/"),
                          raw_test=str(raw_test).replace("\\", "/")))
    con.execute(_read_sql("02_behavioural_features.sql"))


def run_query(sql: str) -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.sql(sql).df()
    finally:
        con.close()


def run_sql_file(name: str) -> dict[str, pd.DataFrame]:
    """Run a file of named SELECT queries separated by '-- name: <x>' markers."""
    text = (config.SQL_DIR / name).read_text()
    out: dict[str, pd.DataFrame] = {}
    con = connect(read_only=True)
    try:
        for block in text.split("-- name:")[1:]:
            qname, _, body = block.partition("\n")
            out[qname.strip()] = con.sql(body).df()
    finally:
        con.close()
    return out


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(config.FEATURES_PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values(["ts", "trans_num"]).reset_index(drop=True)
