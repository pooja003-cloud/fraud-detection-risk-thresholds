"""Explainability helpers: SHAP for tree models, grouped one-hot columns, readable labels.

SHAP values for LightGBM are in log-odds units: each value is how much a
feature pushed this transaction's score up (+) or down (-) relative to the
average score. They explain the MODEL, not the world. A large SHAP value
means the model relies on the feature, not that the feature causes fraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import FEATURE_LABELS


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values (n_rows x n_features) and the expected value (log-odds)."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                 # older SHAP returns [neg, pos]
        sv = sv[1]
    ev = explainer.expected_value
    ev = float(ev[1] if np.ndim(ev) else ev)
    return np.asarray(sv), ev


def group_onehot(sv: np.ndarray, columns: list[str], prefix: str = "cat_", group_name: str = "category") -> tuple[np.ndarray, list[str]]:
    """Sum SHAP values of one-hot columns back into one feature (additivity makes this exact)."""
    cols = list(columns)
    oh = [i for i, c in enumerate(cols) if c.startswith(prefix)]
    if not oh:
        return sv, cols
    keep = [i for i in range(len(cols)) if i not in oh]
    grouped = np.column_stack([sv[:, keep], sv[:, oh].sum(axis=1)])
    return grouped, [cols[i] for i in keep] + [group_name]


DISPLAY_OVERRIDES = {"log_amt": "Transaction amount", "log_city_pop": "Customer city population"}


def label(f: str) -> str:
    return FEATURE_LABELS.get(f, f)


def display_label(f: str) -> str:
    """Label for local explanations, where log features are shown in original units."""
    return DISPLAY_OVERRIDES.get(f, label(f))


def display_values(row: pd.Series) -> pd.Series:
    """Raw row with log-transformed features converted back to readable units."""
    out = row.copy()
    if "log_amt" in out and "amt" in out:
        out["log_amt"] = f"${row['amt']:,.2f}"
    if "log_city_pop" in out:
        out["log_city_pop"] = f"{np.expm1(row['log_city_pop']):,.0f}"
    return out


def global_importance(sv: np.ndarray, columns: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["label"] = imp.feature.map(label)
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def top_drivers(sv_row: np.ndarray, columns: list[str], values: pd.Series, k: int = 5) -> pd.DataFrame:
    """The k features that moved one transaction's score most, with their values and direction."""
    order = np.argsort(-np.abs(sv_row))[:k]
    return pd.DataFrame({
        "feature": [columns[i] for i in order],
        "description": [display_label(columns[i]) for i in order],
        "value": [values.get(columns[i], np.nan) for i in order],
        "shap_log_odds": [sv_row[i] for i in order],
        "direction": ["raises risk" if sv_row[i] > 0 else "lowers risk" for i in order],
    })


def feature_groups(columns: list[str], prefix: str = "cat_", group_name: str = "category") -> dict[str, list[str]]:
    """Map each original feature to its model columns (one-hot columns grouped together)."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(group_name if c.startswith(prefix) else c, []).append(c)
    return groups


def grouped_permutation_importance(predict, X: pd.DataFrame, y, groups: dict[str, list[str]],
                                   n_repeats: int = 3, seed: int = 42) -> pd.DataFrame:
    """Drop in validation PR-AUC when a feature (or a whole one-hot block) is shuffled.

    Shuffling the one-hot block as a unit keeps every row a valid category,
    which per-column permutation would break.
    """
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = average_precision_score(y, predict(X))
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - average_precision_score(y, predict(Xp)))
        rows.append({"feature": name, "label": label(name), "pr_auc_drop": float(np.mean(drops)),
                     "pr_auc_drop_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)


def waterfall(ax, drivers: pd.DataFrame, base_logodds: float, final_score: float, title_text: str, subtitle: str,
              pos_color: str, neg_color: str, text_color: str) -> None:
    """Horizontal bar chart of the top SHAP contributions for one transaction."""
    d = drivers.iloc[::-1]
    labels = [f"{r.description} = {_fmt(r.value)}" for r in d.itertuples()]
    ax.barh(labels, d.shap_log_odds, color=[pos_color if v > 0 else neg_color for v in d.shap_log_odds], height=0.6)
    for i, v in enumerate(d.shap_log_odds):
        ax.text(v, i, f" {v:+.2f} ", va="center", ha="left" if v > 0 else "right", fontsize=8, color=text_color)
    ax.axvline(0, color=text_color, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP contribution (log-odds; + raises risk)")
    ax.set_title(title_text, loc="left", pad=22, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color="#52514e", va="bottom")
    lo, hi = ax.get_xlim(); pad = 0.25 * (hi - lo); ax.set_xlim(lo - pad * (lo < 0), hi + pad)


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "n/a (no history)"
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}"
