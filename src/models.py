"""Model training helpers. All fitting uses the training split only; the
validation split is used for hyper-parameter choice and early stopping.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .features import linear_preprocessor, onehot_frame, tree_frame
from .metrics import pr_auc


# ------------------------------------------------------------------ logistic regression
def fit_logreg(train: pd.DataFrame, C: float, demographics: bool = False) -> Pipeline:
    """Class-weighted logistic regression.

    class_weight='balanced' up-weights each fraud by ~(legit/fraud) so the
    loss is not dominated by the 99.4% legitimate class.
    """
    model = Pipeline([
        ("prep", linear_preprocessor(demographics)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                   random_state=config.SEED)),
    ])
    return model.fit(train, train["is_fraud"])


# ------------------------------------------------------------------ random forest
def fit_random_forest(X: pd.DataFrame, y: pd.Series, min_samples_leaf: int, n_estimators: int = 150,
                      max_samples: float = 0.25) -> RandomForestClassifier:
    """Random forest with per-tree balanced class weights.

    max_samples bootstraps a quarter of the training rows per tree. That keeps
    training tractable on 1.1M rows and adds diversity between trees.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features="sqrt",
        class_weight="balanced_subsample", max_samples=max_samples,
        n_jobs=-1, random_state=config.SEED,
    )
    return rf.fit(X, y)


# ------------------------------------------------------------------ LightGBM
BASE_LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    n_estimators=3000,               # upper bound; early stopping picks the real number
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=config.SEED,
    n_jobs=-1,
    verbose=-1,
)


def fit_lgbm(X_tr, y_tr, X_val, y_val, **overrides) -> lgb.LGBMClassifier:
    params = {**BASE_LGB_PARAMS, **overrides}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
    )
    return model


def resample(X: pd.DataFrame, y: pd.Series, method: str, ratio: float = 0.1):
    """Resample TRAINING data only. ratio = minority / majority after resampling.

    - 'undersample': randomly drop legitimate rows until fraud:legit = ratio.
    - 'smote': synthesise frauds by interpolating between fraud neighbours
      (needs a fully numeric frame without NaN, so it uses the one-hot frame).
    """
    if method == "undersample":
        return RandomUnderSampler(sampling_strategy=ratio, random_state=config.SEED).fit_resample(X, y)
    if method == "smote":
        return SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=config.SEED).fit_resample(X, y)
    raise ValueError(method)


def imbalance_experiment(train: pd.DataFrame, val: pd.DataFrame, categories: list[str]) -> tuple[pd.DataFrame, dict]:
    """Same LightGBM hyper-parameters, four ways of handling class imbalance."""
    y_tr, y_val = train["is_fraud"], val["is_fraud"]
    Xt_tr, Xt_val = tree_frame(train, categories), tree_frame(val, categories)
    Xo_tr, Xo_val = onehot_frame(train, categories), onehot_frame(val, categories)
    neg_pos = float((y_tr == 0).sum() / (y_tr == 1).sum())

    variants = {
        "No weighting": lambda: fit_lgbm(Xt_tr, y_tr, Xt_val, y_val),
        "Class weights (scale_pos_weight)": lambda: fit_lgbm(Xt_tr, y_tr, Xt_val, y_val, scale_pos_weight=neg_pos),
        "Random undersampling 1:10": lambda: fit_lgbm(*resample(Xt_tr, y_tr, "undersample"), Xt_val, y_val),
        "SMOTE to 1:10": lambda: fit_lgbm(*resample(Xo_tr, y_tr, "smote"), Xo_val, y_val),
    }
    rows, fitted = [], {}
    for name, fit in variants.items():
        t0 = time.time()
        m = fit()
        X_eval = Xo_val if name.startswith("SMOTE") else Xt_val
        s = m.predict_proba(X_eval)[:, 1]
        rows.append({"treatment": name, "val_pr_auc": pr_auc(y_val, s),
                     "best_iteration": m.best_iteration_, "fit_seconds": round(time.time() - t0, 1)})
        fitted[name] = m
    return pd.DataFrame(rows), fitted
