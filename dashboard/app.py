"""Fraud-monitoring dashboard.

Run from the repo root:
    streamlit run dashboard/app.py

It reads only the small CSV outputs written by notebooks 02-05 (dashboard/data and
reports/), so it works without the raw data or the trained model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard" / "data"
REPORTS = ROOT / "reports"

BLUE, ORANGE, AQUA, VIOLET, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#a3a29c", "#e34948"

st.set_page_config(page_title="Fraud Risk Monitoring", page_icon="🛡️", layout="wide")


@st.cache_data
def load():
    d = {
        "sweep_validation": pd.read_csv(DATA / "threshold_sweep_validation.csv"),
        "sweep_test": pd.read_csv(DATA / "threshold_sweep_test.csv"),
        "monthly": pd.read_csv(DATA / "monitoring_monthly.csv"),
        "weekly": pd.read_csv(DATA / "monitoring_weekly.csv"),
        "fired": pd.read_csv(DATA / "monitoring_triggers_fired.csv"),
        "stress": pd.read_csv(DATA / "monitoring_stress_tests.csv", index_col=0),
        "bands": pd.read_csv(DATA / "triage_band_table.csv"),
        "queue": pd.read_csv(DATA / "alert_queue_test.csv", parse_dates=["ts"]),
        "models": pd.read_csv(REPORTS / "model_comparison_test.csv"),
        "assumptions": pd.read_csv(REPORTS / "assumptions.csv"),
        "ops": pd.read_csv(REPORTS / "operational_comparison_test.csv", index_col=0),
        "savings": pd.read_csv(REPORTS / "operational_savings.csv"),
        "decision": json.loads((REPORTS / "threshold_decision.json").read_text()),
        "triage": json.loads((REPORTS / "triage_bands.json").read_text()),
    }
    return d


def layout(fig, title, height=380):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height, margin=dict(l=10, r=10, t=50, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


d = load()
thr_default = d["decision"]["thresholds"]

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Assumptions")
split = st.sidebar.radio("Evaluation period", ["test", "validation"], index=0,
                         help="Thresholds are chosen on validation; test is the unseen later period.")
review_cost = st.sidebar.number_input("Review cost per alert ($)", 0.5, 50.0, 5.0, 0.5)
loss_pct = st.sidebar.slider("Fraud loss on a missed fraud (% of amount)", 10, 150, 100, 5)
analysts = st.sidebar.number_input("Analysts", 1, 20, 2)
per_analyst = st.sidebar.number_input("Reviews per analyst per day", 10, 200, 50, 5)
capacity = analysts * per_analyst
st.sidebar.caption("Defaults are the documented project assumptions. Changing them re-optimises the threshold live.")

sw = d[f"sweep_{split}"].copy()
n_days = int(round(sw.alerts.max() / sw.alerts_per_day.max()))
sw["loss"] = sw.fraud_missed_usd * loss_pct / 100
sw["review"] = sw.alerts * review_cost
sw["total"] = sw.loss + sw.review

# threshold options are always chosen on VALIDATION with the current assumptions
sv = d["sweep_validation"].copy()
sv["total"] = sv.fraud_missed_usd * loss_pct / 100 + sv.alerts * review_cost
opt = {
    "Cost-minimising": float(sv.loc[sv.total.idxmin(), "threshold"]),
    f"Capacity fill ({capacity}/day)": float(sv[sv.alerts_per_day <= capacity].threshold.min()),
    "Recall target (80%)": float(sv[sv.recall >= 0.80].threshold.max()),
}

st.title("Fraud risk monitoring")
st.caption("LightGBM fraud model on the Sparkov synthetic card-transaction dataset. All figures come from notebook outputs.")

tab1, tab2, tab3, tab4 = st.tabs(["Threshold & cost", "Monitoring", "Alert queue", "Models"])

# ------------------------------------------------------------------ tab 1
with tab1:
    choice = st.radio("Operating point", list(opt) + ["Custom"], horizontal=True)
    if choice == "Custom":
        t = st.slider("Score threshold", 0.0001, 0.99, float(opt["Cost-minimising"]), 0.0001, format="%.4f")
    else:
        t = opt[choice]
    row = sw.iloc[(sw.threshold - t).abs().idxmin()]
    c = st.columns(6)
    c[0].metric("Threshold", f"{row.threshold:.4f}")
    c[1].metric("Alerts / day", f"{row.alerts_per_day:,.1f}", f"{row.alerts_per_day / capacity:.0%} of capacity", delta_color="off")
    c[2].metric("Precision", f"{row.precision:.1%}")
    c[3].metric("Recall", f"{row.recall:.1%}")
    c[4].metric("Fraud $ missed", f"${row.loss:,.0f}")
    c[5].metric("Total cost", f"${row.total:,.0f}", help="Missed-fraud loss + review cost for every alert (TP + FP)")
    if row.alerts_per_day > capacity:
        st.warning(f"This threshold needs {row.alerts_per_day:,.0f} reviews/day but capacity is {capacity}. "
                   f"About {int(np.ceil(row.alerts_per_day / per_analyst))} analysts would be needed.")

    s = sw[(sw.alerts_per_day > 0.3) & (sw.alerts_per_day <= max(capacity * 1.4, 60))]
    fig = go.Figure()
    fig.add_scatter(x=s.alerts_per_day, y=s.total, name="Total cost", line=dict(color=BLUE, width=2.5))
    fig.add_scatter(x=s.alerts_per_day, y=s.loss, name="Missed-fraud loss", line=dict(color=ORANGE, width=1.5))
    fig.add_scatter(x=s.alerts_per_day, y=s.review, name="Review cost", line=dict(color=AQUA, width=1.5))
    fig.add_vline(x=capacity, line=dict(color=GREY, width=1), annotation_text="capacity")
    fig.add_scatter(x=[row.alerts_per_day], y=[row.total], mode="markers", name="Selected",
                    marker=dict(size=13, color=VIOLET, line=dict(color="white", width=2)))
    ymax = float(s.total.quantile(0.9)) if len(s) else None
    fig.update_yaxes(title="Cost over period ($)", range=[0, ymax * 1.3] if ymax else None)
    fig.update_xaxes(title="Alerts per day")
    st.plotly_chart(layout(fig, f"Cost vs alert volume ({split}, {n_days} days)"), width="stretch")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Workload")
        wl = pd.DataFrame({
            "Metric": ["Alerts / analyst / day", "False alerts / analyst / day", "Utilisation", "Analysts needed"],
            "Value": [f"{row.alerts_per_day / analysts:,.1f}", f"{row.FP / n_days / analysts:,.1f}",
                      f"{row.alerts_per_day / capacity:.0%}", f"{int(np.ceil(row.alerts_per_day / per_analyst))}"],
        })
        st.dataframe(wl, hide_index=True, width="stretch")
    with cols[1]:
        st.subheader("Confusion counts")
        st.dataframe(pd.DataFrame({"": ["Fraud caught (TP)", "False alerts (FP)", "Fraud missed (FN)"],
                                   "Count": [int(row.TP), int(row.FP), int(row.FN)],
                                   "$": [f"${row.fraud_caught_usd:,.0f}", "", f"${row.fraud_missed_usd:,.0f}"]}),
                     hide_index=True, width="stretch")
    with st.expander("Documented assumptions"):
        st.dataframe(d["assumptions"], hide_index=True, width="stretch")

    st.subheader("Operational simulation (test period)")
    st.caption("The threshold explorer above uses the idealised cost model (every alerted fraud is stopped). The operational simulation "
               "(notebook 06) is stricter: only auto-hold stops a transaction; reviews after authorisation block the card after a 12h delay.")
    o = d["ops"]
    show_o = o[["fraud_prevented_pct", "fraud_usd_lost", "reviews_per_day", "holds_per_day", "wrong_holds", "total_cost_usd"]].copy()
    st.dataframe(show_o.style.format({"fraud_prevented_pct": "{:.0%}", "fraud_usd_lost": "${:,.0f}", "reviews_per_day": "{:.1f}",
                                      "holds_per_day": "{:.1f}", "wrong_holds": "{:,.0f}", "total_cost_usd": "${:,.0f}"}),
                 width="stretch")
    sv0 = d["savings"].iloc[0]
    st.caption(f"Saving vs rules under the same auto-hold policy: ${sv0.saving_usd:,.0f} over {int(sv0.test_days)} days "
               f"(95% card-bootstrap interval ${sv0.ci95_low:,.0f} to ${sv0.ci95_high:,.0f}). Synthetic data; illustrative.")

# ------------------------------------------------------------------ tab 2
with tab2:
    m = d["monthly"].copy()
    m["month"] = pd.PeriodIndex(m.period, freq="M").to_timestamp()
    live = m[m.window != "reference"]
    k = st.columns(4)
    last = m.iloc[-1]
    k[0].metric(f"Alerts/day ({last.period})", f"{last.alerts_per_day:.1f}")
    k[1].metric("Precision", f"{last.precision:.0%}", help="Provisional: labels < 60 days old" if not last.labels_mature else None)
    k[2].metric("Recall", f"{last.recall:.0%}")
    k[3].metric("Score PSI", f"{last.psi_score:.3f}")
    if not last.labels_mature:
        st.info("Label-based metrics for the latest months are **provisional**: fraud labels take ~60 days to mature (chargebacks).")

    c1, c2 = st.columns(2)
    f1 = go.Figure()
    f1.add_scatter(x=m.month, y=m.precision, name="Precision", mode="lines+markers", line=dict(color=BLUE))
    f1.add_scatter(x=m.month, y=m.recall, name="Recall", mode="lines+markers", line=dict(color=ORANGE))
    f1.add_hline(y=0.40, line=dict(color=BLUE, width=1, dash="dot"), annotation_text="precision trigger")
    f1.add_hline(y=0.90, line=dict(color=ORANGE, width=1, dash="dot"), annotation_text="recall trigger")
    f1.update_yaxes(range=[0, 1.05])
    c1.plotly_chart(layout(f1, "Precision & recall at the fixed threshold"), width="stretch")
    f2 = go.Figure()
    f2.add_bar(x=m.month, y=m.alerts_per_day, marker_color=[GREY if w == "reference" else BLUE for w in m.window], name="Alerts/day")
    f2.add_hline(y=capacity, line=dict(color=RED, width=1), annotation_text="capacity")
    c2.plotly_chart(layout(f2, "Alerts per day (grey = reference period)"), width="stretch")

    c3, c4 = st.columns(2)
    psi_cols = [c for c in m.columns if c.startswith("psi_")]
    f3 = go.Figure(go.Heatmap(z=m[psi_cols].T.values, x=m.period, y=[c.replace("psi_", "") for c in psi_cols],
                              colorscale="Blues", zmin=0, zmax=0.25, colorbar=dict(title="PSI")))
    c3.plotly_chart(layout(f3, "PSI vs reference (trigger at 0.20)", 420), width="stretch")
    f4 = go.Figure()
    f4.add_scatter(x=m.month, y=m.fraud_rate_pct, mode="lines+markers", line=dict(color=BLUE), name="Fraud rate %")
    c4.plotly_chart(layout(f4, "Fraud rate by month (%)", 420), width="stretch")

    st.subheader("Triggers fired")
    if len(d["fired"]):
        st.dataframe(d["fired"], hide_index=True, width="stretch")
    else:
        st.success("No monitoring triggers fired in the live period.")
    st.subheader("Stress tests (simulated failures)")
    st.dataframe(d["stress"].style.format("{:.3f}"), width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    b = d["bands"][d["bands"].split == split].set_index("band")
    st.subheader("Triage bands")
    st.caption(f"Cut-offs (chosen on validation): auto-hold ≥ {d['triage']['hold']:.3f} · priority ≥ {d['triage']['review']:.3f} · "
               f"standard ≥ {d['triage']['standard']:.4f} · monitor ≥ {d['triage']['monitor']:.5f}")
    st.dataframe(b.drop(columns=["split"]).style.format({"per_day": "{:.1f}", "fraud_rate_in_band": "{:.1%}",
                                                         "share_of_all_fraud": "{:.1%}", "fraud_usd": "${:,.0f}",
                                                         "transactions": "{:,.0f}", "frauds": "{:,.0f}"}),
                 width="stretch")
    st.subheader("Alert queue (test period)")
    q = d["queue"]
    fc = st.columns(3)
    band_sel = fc[0].multiselect("Band", sorted(q.band.unique()), default=sorted(q.band.unique()))
    dates = fc[1].date_input("Date range", (q.ts.min().date(), q.ts.max().date()))
    show_label = fc[2].checkbox("Show hindsight label", False, help="Not available to an analyst at alert time")
    if isinstance(dates, tuple) and len(dates) == 2:
        qq = q[q.band.isin(band_sel) & (q.ts.dt.date >= dates[0]) & (q.ts.dt.date <= dates[1])]
    else:
        qq = q[q.band.isin(band_sel)]
    cols = ["ts", "card", "category", "amt", "score", "band", "top_reasons"] + (["label_in_hindsight"] if show_label else [])
    st.dataframe(qq.sort_values("score", ascending=False)[cols], hide_index=True, width="stretch", height=420,
                 column_config={"score": st.column_config.NumberColumn(format="%.3f"),
                                "amt": st.column_config.NumberColumn("amount", format="$%.2f")})
    st.caption(f"{len(qq):,} alerts shown. Card numbers masked; reasons are the top-3 SHAP drivers in plain language.")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.subheader("Model comparison (test period, thresholds chosen on validation)")
    mc = d["models"].rename(columns={"Unnamed: 0": "model"})
    show = ["model", "PR-AUC", "PR-AUC 95% CI", "ROC-AUC", "precision", "recall", "TP", "FP", "alerts_per_day", "fraud_caught_usd"]
    st.dataframe(mc[show].style.format({"PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "precision": "{:.1%}", "recall": "{:.1%}",
                                         "alerts_per_day": "{:.1f}", "fraud_caught_usd": "${:,.0f}"}),
                 hide_index=True, width="stretch")
    img = REPORTS / "figures" / "pr_curves_test.png"
    if img.exists():
        st.image(str(img), caption="Precision-recall curves on the test period")
    st.caption("PR-AUC is the primary metric (0.5% fraud makes accuracy and ROC-AUC look flattering). "
               "Sparkov is synthetic; real-world performance would be lower.")
