"""Unit tests for the business-cost framework, using hand-computable toy data."""
import numpy as np
import pytest

from src import costs

# 6 transactions: scores, labels and amounts chosen so every number can be checked by hand.
SCORE = np.array([0.95, 0.90, 0.60, 0.40, 0.20, 0.10])
Y = np.array([1, 0, 1, 0, 1, 0])
AMT = np.array([500.0, 50.0, 200.0, 20.0, 100.0, 10.0])


def test_counts_and_dollars_at_threshold():
    t = costs.sweep(Y, SCORE, AMT, n_days=2, thresholds=[0.5], review_cost=5.0).iloc[0]
    # alerts: 0.95 (fraud), 0.90 (legit), 0.60 (fraud)
    assert (t.alerts, t.TP, t.FP, t.FN) == (3, 2, 1, 1)
    assert t.alerts_per_day == 1.5
    assert t.precision == pytest.approx(2 / 3)
    assert t.recall == pytest.approx(2 / 3)
    assert t.fraud_caught_usd == 700.0
    assert t.fraud_missed_usd == 100.0


def test_both_cost_variants():
    t = costs.sweep(Y, SCORE, AMT, n_days=1, thresholds=[0.5], review_cost=5.0).iloc[0]
    assert t.review_cost_all_alerts_usd == 15.0          # 3 alerts x $5
    assert t.review_cost_fp_only_usd == 5.0              # 1 false alert x $5
    assert t.total_cost_usd == 100.0 + 15.0              # missed fraud + all reviews (primary)
    assert t.total_cost_fp_only_usd == 100.0 + 5.0


def test_recovery_rate_and_loss_multiplier_scale_only_fraud_loss():
    t = costs.sweep(Y, SCORE, AMT, n_days=1, thresholds=[0.5], review_cost=5.0,
                    recovery_rate=0.2, loss_multiplier=1.5).iloc[0]
    assert t.expected_fraud_loss_usd == pytest.approx(100.0 * 0.8 * 1.5)
    assert t.review_cost_all_alerts_usd == 15.0


def test_extreme_thresholds():
    t = costs.sweep(Y, SCORE, AMT, n_days=1, thresholds=[0.0, 1.0], review_cost=5.0).set_index("threshold")
    everything, nothing = t.loc[0.0], t.loc[1.0]
    assert everything.alerts == 6 and everything.FN == 0 and everything.total_cost_usd == 30.0
    assert nothing.alerts == 0 and nothing.TP == 0 and nothing.total_cost_usd == 800.0
    assert nothing.total_cost_usd == costs.no_model_cost(AMT, Y)


def test_ties_are_alerted_together():
    s = np.array([0.5, 0.5, 0.1])
    t = costs.sweep([1, 0, 0], s, [10, 10, 10], n_days=1, thresholds=[0.5]).iloc[0]
    assert t.alerts == 2


def test_threshold_choosers():
    table = costs.sweep(Y, SCORE, AMT, n_days=1, thresholds=[0.05, 0.15, 0.3, 0.5, 0.7, 0.92, 1.0], review_cost=5.0)
    best = costs.cost_minimising(table)
    assert best.threshold == 0.15 and best.total_cost_usd == 25.0     # catches all fraud with 5 reviews
    cap = costs.capacity_constrained(table, capacity_per_day=3)
    assert cap.alerts <= 3 and cap.threshold == 0.5
    rt = costs.recall_target(table, target=0.6)
    assert rt.recall >= 0.6 and rt.threshold == 0.5


def test_monotonicity_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.02
    s = np.clip(y * 0.3 + rng.random(5000), 0, 1)
    a = rng.lognormal(3, 1, 5000)
    t = costs.sweep(y, s, a, n_days=10)
    assert np.all(np.diff(t.alerts.to_numpy()) <= 0)       # higher threshold -> fewer alerts
    assert np.all(np.diff(t.recall.to_numpy()) <= 1e-12)
    assert np.allclose(t.TP + t.FN, y.sum())
    assert np.allclose(t.fraud_caught_usd + t.fraud_missed_usd, a[y].sum())
