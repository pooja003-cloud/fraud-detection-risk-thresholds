"""Collect the headline numbers quoted in the README and reports from saved outputs.

Every number in the documentation comes from reports/headline_metrics.json,
which this module builds from the CSV/JSON files written by notebooks 01-05.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config


def build() -> dict:
    R = config.REPORTS
    facts = pd.read_csv(R / "dataset_facts.csv", index_col=0)["value"].to_dict()
    split = pd.read_csv(R / "split_summary.csv", index_col=0)
    test = pd.read_csv(R / "model_comparison_test.csv", index_col=0)
    val = pd.read_csv(R / "model_comparison_validation.csv", index_col=0)
    opts = pd.read_csv(R / "threshold_options_val_vs_test.csv").set_index(["option", "split"])
    same = pd.read_csv(R / "model_vs_rules_same_volume.csv").set_index(["split", "approach"])
    wl = pd.read_csv(R / "fp_workload_table.csv").set_index(["option", "split"])
    sens = pd.read_csv(R / "sensitivity_analysis.csv")
    decision = json.loads((R / "threshold_decision.json").read_text())
    meta = json.loads((R / "model_metadata.json").read_text())

    cm = opts.loc[("Cost-minimising", "test")]
    cm_val = opts.loc[("Cost-minimising", "validation")]
    cap_name = [o for o in opts.index.get_level_values(0).unique() if o.startswith("Capacity")][0]
    rec_name = [o for o in opts.index.get_level_values(0).unique() if o.startswith("Recall")][0]
    rules_te = same.loc[("test", "Rules baseline")]
    lgb_same = same.loc[("test", "LightGBM @ same alert volume")]
    rules_wl = wl.loc[("Rules baseline (current process)", "test")]

    h = {
        "dataset": facts,
        "split": split.reset_index().to_dict("records"),
        "model_selection": meta,
        "test_pr_auc": test["PR-AUC"].to_dict(),
        "test_pr_auc_ci": test["PR-AUC 95% CI"].to_dict(),
        "val_pr_auc": val["PR-AUC"].to_dict(),
        "test_at_f1_threshold": test[["precision", "recall", "TP", "FP", "FN", "alerts_per_day"]].to_dict("index"),
        "recommended_threshold": decision["thresholds"]["Cost-minimising"],
        "threshold_options": {o: {s: opts.loc[(o, s)].to_dict() for s in ["validation", "test"]}
                              for o in ["Cost-minimising", cap_name, rec_name]},
        "rules_test": rules_te.to_dict(),
        "lgbm_same_volume_test": lgb_same.to_dict(),
        "headline": {
            "test_days": int(decision["test_days"]),
            "lgbm_alerts_per_day": cm.alerts_per_day,
            "rules_alerts_per_day": rules_te.alerts_per_day,
            "alert_reduction_pct": 100 * (1 - cm.alerts_per_day / rules_te.alerts_per_day),
            "lgbm_recall": cm.recall, "rules_recall": rules_te.recall,
            "lgbm_precision": cm.precision, "rules_precision": rules_te.precision,
            "lgbm_fp": cm.FP, "rules_fp": rules_te.FP,
            "fp_reduction_pct": 100 * (1 - cm.FP / rules_te.FP),
            "lgbm_tp": cm.TP, "rules_tp": rules_te.TP,
            "lgbm_fraud_missed_usd": cm.fraud_missed_usd, "rules_fraud_missed_usd": rules_te.fraud_missed_usd,
            "lgbm_total_cost_usd": cm.total_cost_usd, "rules_total_cost_usd": rules_te.total_cost_usd,
            "cost_saving_usd": rules_te.total_cost_usd - cm.total_cost_usd,
            "cost_saving_pct": 100 * (1 - cm.total_cost_usd / rules_te.total_cost_usd),
            "lgbm_val_alerts_per_day": cm_val.alerts_per_day, "lgbm_val_precision": cm_val.precision, "lgbm_val_recall": cm_val.recall,
            "same_volume_extra_frauds": lgb_same.TP - rules_te.TP,
            "same_volume_lgbm_recall": lgb_same.recall,
            "capacity_utilisation_pct": wl.loc[("Cost-minimising", "test"), "utilisation_pct"],
            "rules_false_alert_share_pct": 100 * (1 - rules_wl.precision),
            "sensitivity_alerts_per_day_range": [sens.alerts_per_day.min(), sens.alerts_per_day.max()],
            "sensitivity_recall_range": [sens.recall.min(), sens.recall.max()],
        },
    }
    for name in ["segment_error_analysis", "demographic_ablation", "realism_off_pattern_fraud", "realism_label_noise",
                 "monitoring_stress_tests", "feature_importance", "triage_band_table"]:
        p = R / f"{name}.csv"
        if p.exists():
            h[name] = pd.read_csv(p).to_dict("records")
    for name in ["monitoring_summary", "triage_bands"]:
        p = R / f"{name}.json"
        if p.exists():
            h[name] = json.loads(p.read_text())
    (R / "headline_metrics.json").write_text(json.dumps(h, indent=2, default=float))
    return h
