"""Project-wide configuration: paths, random seed and business assumptions.

Every business assumption used in the cost analysis lives here so that it is
documented once, can be changed once, and is reported in the README.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DASHBOARD_DATA = ROOT / "dashboard" / "data"

RAW_TRAIN = DATA_RAW / "fraudTrain.csv"
RAW_TEST = DATA_RAW / "fraudTest.csv"
DUCKDB_PATH = DATA_PROCESSED / "fraud.duckdb"
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- time split
# Earliest 60% of transactions -> train, next 20% -> validation, latest 20% -> test.
# Cut-offs are snapped to whole calendar days so that no day is split across sets.
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)

# ---------------------------------------------------------------- business assumptions
# Fraud loss for a missed fraud = the transaction amount x (1 - recovery rate).
# Default: no recovery. Sensitivity analysis covers +/-50% of the loss.
RECOVERY_RATE = 0.0

# Cost of one manual review, in dollars. Default reasoning: ~8 minutes of a
# fraud analyst at a fully loaded cost of ~$40/hour = ~$5.
REVIEW_COST = 5.0

# Review capacity. Default: 2 analysts x 50 reviews per analyst per day.
N_ANALYSTS = 2
REVIEWS_PER_ANALYST_PER_DAY = 50
DAILY_REVIEW_CAPACITY = N_ANALYSTS * REVIEWS_PER_ANALYST_PER_DAY

# Recall target used for the "recall-target" threshold option.
RECALL_TARGET = 0.80

for _p in (DATA_PROCESSED, MODELS_DIR, FIGURES, DASHBOARD_DATA):
    _p.mkdir(parents=True, exist_ok=True)
