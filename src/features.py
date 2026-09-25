"""Feature lists and train-only preprocessing.

Design decisions (each one is defensible in an interview):

* Only features available at authorisation time are used. All behavioural
  features come from backward-looking SQL windows (sql/02_behavioural_features.sql).
* Direct identifiers (card number, merchant name, trans_num) are not features:
  they would let the model memorise specific cards or merchants that
  appear in the fraud simulation, which does not generalise.
* State is not a model feature: there are ~1,000 customers across 50+ states,
  so state is close to a customer identifier and would be memorised. It is
  kept for segment analysis.
* Gender and age are protected characteristics. The production-candidate
  model excludes them; notebook 04 runs an ablation that adds them back, to
  measure what they contribute and to support the fairness discussion.
* All imputers, encoders and scalers are fitted on the training split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "log_amt",
    "hour",
    "day_of_week",
    "is_night",
    "log_city_pop",
    "dist_home_merchant_km",
    "n_txn_prior_1h",
    "amt_sum_prior_1h",
    "n_txn_prior_24h",
    "amt_sum_prior_24h",
    "n_txn_prior_7d",
    "cust_n_prior_txn",
    "amt_to_cust_avg",
    "amt_zscore_cust",
    "amt_to_cust_avg_7d",
    "amt_to_cust_category_avg",
    "hours_since_last_txn",
    "is_first_merchant_for_cust",
    "is_first_category_for_cust",
]
CATEGORICAL_FEATURES = ["category"]
DEMOGRAPHIC_FEATURES = ["cust_age", "gender_is_male"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plain-English names used in charts, SHAP plots and the case report.
FEATURE_LABELS = {
    "log_amt": "Transaction amount (log)",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "is_night": "Night-time (22:00-04:00)",
    "log_city_pop": "Customer city population (log)",
    "dist_home_merchant_km": "Distance home to merchant (km)",
    "n_txn_prior_1h": "Card transactions in prior 1h",
    "amt_sum_prior_1h": "Card spend in prior 1h ($)",
    "n_txn_prior_24h": "Card transactions in prior 24h",
    "amt_sum_prior_24h": "Card spend in prior 24h ($)",
    "n_txn_prior_7d": "Card transactions in prior 7d",
    "cust_n_prior_txn": "Card history length (txns)",
    "amt_to_cust_avg": "Amount / customer's historical average",
    "amt_zscore_cust": "Amount z-score vs customer history",
    "amt_to_cust_avg_7d": "Amount / customer's 7-day average",
    "amt_to_cust_category_avg": "Amount / customer's average in this category",
    "hours_since_last_txn": "Hours since card's previous transaction",
    "is_first_merchant_for_cust": "First time at this merchant",
    "is_first_category_for_cust": "First time in this category",
    "category": "Merchant category",
    "cust_age": "Customer age",
    "gender_is_male": "Customer gender (male)",
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender_is_male"] = (df["gender"] == "M").astype(int)
    return df


# ------------------------------------------------------------------ tree models
def tree_frame(df: pd.DataFrame, categories: list[str], demographics: bool = False) -> pd.DataFrame:
    """Frame for LightGBM: raw numerics (NaN kept, handled natively) + category dtype.

    `categories` must come from the training split so that category codes are
    identical in validation, test and production.
    """
    cols = NUMERIC_FEATURES + (DEMOGRAPHIC_FEATURES if demographics else [])
    X = df[cols].astype(float).copy()
    X["category"] = pd.Categorical(df["category"], categories=categories)
    return X


def onehot_frame(df: pd.DataFrame, categories: list[str], demographics: bool = False) -> pd.DataFrame:
    """Frame for sklearn trees: numerics + fixed one-hot columns; NaN -> -1 sentinel.

    A sentinel is safe for trees (they can split it off) and keeps "no history"
    distinguishable from real values.
    """
    X = tree_frame(df, categories, demographics)
    dummies = pd.get_dummies(X.pop("category"), prefix="cat", dtype=float)
    return pd.concat([X.fillna(-1.0), dummies], axis=1)


# ------------------------------------------------------------------ linear model
def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def linear_preprocessor(demographics: bool = False) -> ColumnTransformer:
    """Imputation (+ missing flags) -> signed log -> standard scaling; one-hot categories.

    Signed log compresses the heavy-tailed ratio and velocity features so that
    a handful of extreme values do not dominate the logistic-regression fit.
    """
    num_cols = NUMERIC_FEATURES + (DEMOGRAPHIC_FEATURES if demographics else [])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(_signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
