"""Rule-based baseline: the kind of transparent rules a fraud operations team runs today.

Each rule has one or two cut-offs. Cut-offs are chosen on the TRAIN split only,
by grid search over round numbers a fraud analyst could state out loud
("over $500 at night"). The objective per rule is F1 on train, a neutral
balance of catching fraud and keeping the rule's hit rate usable.

The baseline alerts if ANY rule fires. Its "score" (for PR curves) is the
number of rules that fired, 0-4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Rule:
    name: str
    description: str                      # template with {} placeholders for cut-offs
    fn: Callable[[pd.DataFrame, tuple], np.ndarray]
    grid: list[tuple]
    cutoff: tuple | None = None
    train_stats: dict = field(default_factory=dict)

    def fire(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.fn(df, self.cutoff), dtype=bool)

    def text(self) -> str:
        return self.description.format(*self.cutoff)


def _f1(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


# Rule logic lives in named module-level functions so fitted rules can be pickled.
def _r1_night_amount(d, c):
    return (d["is_night"] == 1) & (d["amt"] >= c[0])


def _r2_spend_spike(d, c):
    return (d["amt_sum_prior_24h"] + d["amt"]) >= c[0]


def _r3_above_norm(d, c):
    return (d["amt_to_cust_avg"] >= c[0]) & (d["cust_n_prior_txn"] >= 30)


def _r4_category_amount(d, c):
    return d["category"].isin(list(c[0])) & (d["amt"] >= c[1])


def make_rules(high_risk_categories: list[str]) -> list[Rule]:
    amt_grid = [100, 200, 250, 300, 400, 500, 750, 1000]
    return [
        Rule("R1 high amount at night", "Amount >= ${} between 22:00 and 04:00",
             _r1_night_amount, [(a,) for a in amt_grid]),
        Rule("R2 24h spend spike", "Card spend in last 24h (incl. this txn) >= ${}",
             _r2_spend_spike, [(s,) for s in [500, 750, 1000, 1250, 1500, 2000, 2500, 3000]]),
        Rule("R3 far above customer norm",
             "Amount >= {}x the customer's historical average (min. 30 prior txns)",
             _r3_above_norm, [(r,) for r in [2, 3, 4, 5, 6, 8, 10, 15]]),
        Rule("R4 high-risk category, large amount", "Category in {} and amount >= ${}",
             _r4_category_amount, [(tuple(high_risk_categories), a) for a in amt_grid]),
    ]


def high_risk_categories(train: pd.DataFrame, k: int = 3) -> list[str]:
    """Top-k categories by fraud rate, computed on the training split only."""
    return train.groupby("category")["is_fraud"].mean().sort_values(ascending=False).head(k).index.tolist()


def fit_rules(train: pd.DataFrame) -> list[Rule]:
    y = train["is_fraud"].to_numpy().astype(bool)
    rules = make_rules(high_risk_categories(train))
    for rule in rules:
        best = None
        for c in rule.grid:
            f1, p, r = _f1(y, np.asarray(rule.fn(train, c), dtype=bool))
            if best is None or f1 > best[0]:
                best = (f1, p, r, c)
        rule.cutoff = best[3]
        rule.train_stats = {"f1": best[0], "precision": best[1], "recall": best[2]}
    return rules


def rule_table(rules: list[Rule], df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy().astype(bool)
    rows = []
    for rule in rules:
        pred = rule.fire(df)
        f1, p, r = _f1(y, pred)
        rows.append({"rule": rule.name, "definition": rule.text(), "split": split_name,
                     "alerts": int(pred.sum()), "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)


def rule_score(rules: list[Rule], df: pd.DataFrame) -> np.ndarray:
    """Number of rules fired (0-4); alert = score >= 1."""
    return np.sum([rule.fire(df) for rule in rules], axis=0).astype(float)
