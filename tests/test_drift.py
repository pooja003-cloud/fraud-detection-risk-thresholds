import numpy as np

from src import drift


def test_psi_is_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=10000)
    assert drift.psi(x, x) < 1e-9


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=20000)
    small = drift.psi(ref, rng.normal(0.1, 1, 20000))
    large = drift.psi(ref, rng.normal(1.0, 1, 20000))
    assert small < 0.10 < 0.25 < large


def test_psi_handles_missing_and_categorical():
    ref = np.array([1.0, 2.0, np.nan, 3.0] * 100)
    act = np.array([np.nan, np.nan, np.nan, 3.0] * 100)
    assert drift.psi(ref, act) > 0.25                      # jump in missing share is detected
    cats_ref = np.array(["a", "b", "c"] * 100, dtype=object)
    cats_act = np.array(["a"] * 300, dtype=object)
    assert drift.psi(cats_ref, cats_act) > 0.25


def test_psi_treats_string_columns_as_categorical():
    import pandas as pd
    ref = pd.Series(["grocery", "travel", "home"] * 200 + [f"c{i}" for i in range(20)] * 5)
    assert drift.psi(ref, ref) < 1e-9


def test_tail_psi_sees_changes_among_high_scores():
    rng = np.random.default_rng(3)
    ref = rng.beta(0.3, 30, 50000)
    act = ref.copy()
    top = act > np.quantile(ref, 0.99)
    act[top] = act[top] * 3          # only the alerting tail moves
    assert drift.psi(ref, act) < 0.10          # standard PSI barely notices
    assert drift.psi_tail(ref, act) > drift.psi(ref, act)
