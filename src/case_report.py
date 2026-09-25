"""Generate a neutral, evidence-based alert case report from data only.

Structure: the five sections required by the project brief (observed facts,
risk indicators, data gaps, escalation recommendation, next steps). The
following were added using public investigation frameworks from
ClaudeFinanceLab, adapted from AML to card fraud:
  * an alert-referral block
  * an "account baseline vs this activity" comparison
  * an investigation-timeline table that makes data gaps explicit
  * disposition checkboxes and sign-off fields

Every value is computed from the transaction data, model scores and SHAP.
Nothing is inferred beyond them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import explain

NIGHT_COMMON = 0.20   # share of a card's transactions at night above which night activity is called "common" (a stated convention)


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _hours(h) -> str:
    if pd.isna(h):
        return "n/a (no prior transaction)"
    return f"{h * 60:,.0f} minutes" if h < 1 else f"{h:,.1f} hours"


def baseline_table(c: pd.Series, history: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """Compare this transaction with the card's own behaviour in the prior `days` days (strictly earlier)."""
    h = history[(history.ts < c.ts) & (history.ts >= c.ts - pd.Timedelta(days=days))]
    n_days = max(days, 1)
    daily = h.groupby(h.ts.dt.normalize()).amt.agg(["size", "sum"]).reindex(
        pd.date_range(c.ts.normalize() - pd.Timedelta(days=days), c.ts.normalize() - pd.Timedelta(days=1)), fill_value=0)
    rows = [
        ("Transaction amount", f"median {_money(h.amt.median())}, 95th percentile {_money(h.amt.quantile(0.95))}" if len(h) else "no history",
         _money(c.amt), f"{c.amt / h.amt.median():,.1f}x the median" if len(h) and h.amt.median() else "n/a"),
        ("Share of transactions at night (22:00-04:00)", f"{h.is_night.mean():.0%}" if len(h) else "no history",
         "night" if c.is_night == 1 else "day",
         ((f"night activity is common for this card (>= {NIGHT_COMMON:.0%})" if h.is_night.mean() >= NIGHT_COMMON
           else f"night activity is uncommon for this card (< {NIGHT_COMMON:.0%})") if len(h) else "")),
        (f"Share of transactions in '{c.category}'", f"{(h.category == c.category).mean():.0%} ({int((h.category == c.category).sum())} transactions)" if len(h) else "no history",
         c.category, "category used before in this window" if len(h) and (h.category == c.category).any() else "category not used in this window"),
        ("Largest single transaction", _money(h.amt.max()) if len(h) else "no history", _money(c.amt),
         "exceeds prior maximum" if len(h) and c.amt > h.amt.max() else "within prior range"),
        ("Transactions per calendar day (incl. zero-activity days)", f"median {daily['size'].median():.0f}, max {daily['size'].max():.0f}",
         f"{int(c.n_txn_prior_24h)} in the rolling prior 24h", "within prior range" if c.n_txn_prior_24h <= daily['size'].max() else "above prior daily maximum"),
        ("Spend per calendar day (incl. zero-activity days)", f"median {_money(daily['sum'].median())}, max {_money(daily['sum'].max())}",
         f"{_money(c.amt_sum_prior_24h)} in the rolling prior 24h (+ this {_money(c.amt)})",
         "within prior range" if c.amt_sum_prior_24h + c.amt <= daily['sum'].max() else "above prior daily maximum"),
    ]
    return pd.DataFrame(rows, columns=["Measure", f"Card baseline (prior {n_days} days)", "This alert", "Comparison"])


BAND_ACTION = {
    "Auto-hold": ("The transaction is held at authorisation (auto-hold band). Confirm with the cardholder through a registered "
                  "channel; release if confirmed genuine, otherwise treat as confirmed/suspected fraud."),
    "Priority review": ("Route to the L1 priority-review queue (same-day SLA; workflow assumption). The transaction has already been "
                        "authorised, so the purpose of review is to confirm or clear it quickly and, if fraud is confirmed, block the card "
                        "before further transactions. Contact the cardholder through a registered channel before disposition."),
    "Standard review": ("Route to the L1 standard-review queue (24-hour SLA; workflow assumption). Contact the cardholder before "
                        "disposition; block the card if fraud is confirmed."),
}


def build(c: pd.Series, history: pd.DataFrame, scored: pd.DataFrame, cut: dict, drivers: pd.DataFrame,
          score_col: str = "score_LightGBM", band: str = "Priority review",
          selection_note: str = "", label_maturity_days: int = 60) -> tuple[str, str]:
    """Render the case report and, separately, the hindsight outcome note.

    c        : the alerted transaction (row with features, score, category, merchant, ts, amt, is_fraud)
    history  : all transactions of the dataset (only rows strictly before c.ts on this card are used,
               and only labels older than `label_maturity_days` are treated as known)
    scored   : scored transactions of the monitoring period (for prior alerts on the card)
    cut      : triage cut-offs
    drivers  : top SHAP drivers for c (from explain.top_drivers)
    """
    card = history[history.cc_num == c.cc_num]
    prior = card[card.ts < c.ts]
    first_seen = card.ts.min()
    last48 = prior[prior.ts >= c.ts - pd.Timedelta(hours=48)].sort_values("ts")
    h90 = prior[prior.ts >= c.ts - pd.Timedelta(days=90)]
    prior_alerts = scored[(scored.cc_num == c.cc_num) & (scored.ts < c.ts) & (scored.ts >= c.ts - pd.Timedelta(days=30))
                          & (scored[score_col] >= cut["standard"])]
    known_fraud = prior[(prior.is_fraud == 1) & (prior.ts < c.ts - pd.Timedelta(days=label_maturity_days))]
    same_merchant = prior[prior.merchant == c.merchant].sort_values("ts")
    same_cat = prior[prior.category == c.category]
    base = baseline_table(c, history[history.cc_num == c.cc_num])
    base_md = "\n".join(f"| {r.Measure} | {r[1]} | {r[2]} | {r[3]} |" for r in base.itertuples(index=False))
    hist_md = "\n".join(f"| {r.ts:%Y-%m-%d %H:%M} | {r.category} | {_money(r.amt)} |" for r in last48.itertuples()) \
        or "| none in the prior 48 hours | | |"
    up = drivers[drivers.shap_log_odds > 0]
    down = drivers[drivers.shap_log_odds < 0]
    up_md = "\n".join(f"- **{d.description}: {explain._fmt(d.value)}** (SHAP {d.shap_log_odds:+.2f})" for d in up.itertuples()) or "- none"
    down_md = "\n".join(f"- **{d.description}: {explain._fmt(d.value)}** (SHAP {d.shap_log_odds:+.2f})" for d in down.itertuples()) or "- none"
    avg_prior = c.amt / c.amt_to_cust_avg if pd.notna(c.amt_to_cust_avg) and c.amt_to_cust_avg else np.nan

    # facts for and against the alert pattern (only statements the data supports)
    consistent, inconsistent = [], []
    if pd.notna(c.amt_to_cust_avg):
        consistent.append(f"The amount is {c.amt_to_cust_avg:,.1f}x the card's average transaction since {first_seen:%Y-%m-%d}.")
    if len(h90) and c.amt > h90.amt.quantile(0.95):
        consistent.append(f"The amount is above the card's 95th-percentile transaction in the prior 90 days ({_money(h90.amt.quantile(0.95))}).")
    if c.is_night == 1 and len(h90) and h90.is_night.mean() < NIGHT_COMMON:
        consistent.append(f"The transaction is at night, which is uncommon for this card ({h90.is_night.mean():.0%} of prior-90-day transactions; "
                          f"this report treats {NIGHT_COMMON:.0%} or more as common).")
    if len(h90):
        daily_max = h90.groupby(h90.ts.dt.normalize()).amt.sum().max()
        if c.amt_sum_prior_24h + c.amt > daily_max:
            consistent.append(f"Spend in the rolling prior 24h plus this transaction ({_money(c.amt_sum_prior_24h + c.amt)}) exceeds the card's largest "
                              f"calendar-day spend in the prior 90 days ({_money(daily_max)}).")
    if len(h90) and c.amt <= h90.amt.max():
        inconsistent.append(f"The card's largest transaction in the prior 90 days ({_money(h90.amt.max())}) exceeds this amount.")
    larger = int((prior.amt > c.amt).sum())
    if larger:
        inconsistent.append(f"{larger} earlier transaction(s) on this card since {first_seen:%Y-%m-%d} were larger than this one.")
    if len(same_cat):
        inconsistent.append(f"The card has {len(same_cat)} earlier '{c.category}' transactions (largest {_money(same_cat.amt.max())}).")
    if len(same_merchant):
        lm = same_merchant.iloc[-1]
        inconsistent.append(f"The card has used this merchant before ({len(same_merchant)} time(s); most recently {lm.ts:%Y-%m-%d}, {_money(lm.amt)}).")
    if c.is_night == 1 and len(h90) and h90.is_night.mean() >= NIGHT_COMMON:
        inconsistent.append(f"Night-time activity is common for this card ({h90.is_night.mean():.0%} of prior-90-day transactions; this report treats "
                            f"{NIGHT_COMMON:.0%} or more as common), although the model scores night-time as higher risk.")
    other = []
    if len(known_fraud):
        other.append(f"The card has confirmed fraud in its history ({len(known_fraud)} transactions, {known_fraud.ts.min():%Y-%m-%d} to "
                     f"{known_fraud.ts.max():%Y-%m-%d}). This is relevant context, not evidence about this transaction.")
        used_after = int((prior.ts > known_fraud.ts.max()).sum())
        if used_after:
            other.append(f"The same card number was used {used_after:,} more times after that fraud. A compromised card would normally be reissued, "
                         "so this is likely a quirk of the synthetic data: the history may describe the customer rather than one physical card.")
    cons_md = "\n".join(f"- {x}" for x in consistent) or "- none identified"
    other_md = "\n".join(f"- {x}" for x in other) or "- none"
    incons_md = "\n".join(f"- {x}" for x in inconsistent) or "- none identified"

    if len(known_fraud):
        kf = f"{len(known_fraud)} transaction(s) labelled fraud, {known_fraud.ts.min():%Y-%m-%d} to {known_fraud.ts.max():%Y-%m-%d} (total {_money(known_fraud.amt.sum())})"
    else:
        kf = "none"
    merchant_note = (f"No. {len(same_merchant)} prior transaction(s) at this merchant, most recently {same_merchant.iloc[-1].ts:%Y-%m-%d} ({_money(same_merchant.iloc[-1].amt)})"
                     if len(same_merchant) else "Yes, first transaction at this merchant")
    alerts_note = (f"{len(prior_alerts)} prior alert(s); outcomes not yet known if within the {label_maturity_days}-day label-maturity window (workflow assumption)"
                   if len(prior_alerts) else "No prior model alerts in the 30 days before this alert")
    action = BAND_ACTION.get(band, BAND_ACTION["Standard review"])

    report = f"""# Sample alert case report

| | |
|---|---|
| **Case ID** | {c.trans_num[:12]}… |
| **Referral source** | Automated model alert: fraud score **{c[score_col]:.3f}**, band **{band}** (band range {cut['review']:.3f}–{cut['hold']:.3f}) |
| **Alert date/time** | {c.ts:%Y-%m-%d %H:%M:%S} |
| **How this case was selected** | {selection_note} |
| **Card** | **** {str(c.cc_num)[-4:]} (first seen in data {first_seen:%Y-%m-%d}) |
| **Assigned analyst (L1)** | ____________ |
| **L2 / senior analyst** | ____________ |

*Scope: facts are taken from the Sparkov synthetic dataset and the model output only. Service levels, band actions and the {label_maturity_days}-day label-maturity rule are **workflow assumptions** (see `fraud_triage_workflow.md`), not facts from the data. The section headings follow the project brief; the referral block, baseline comparison, investigation timeline and disposition options follow public financial-crime investigation frameworks (see README, "How ClaudeFinanceLab was used").*

---

## 1. Observed facts (as at alert time)

**Transaction under review**

| Field | Value |
|---|---|
| Date/time | {c.ts:%Y-%m-%d %H:%M:%S} |
| Merchant | {c.merchant} |
| Merchant category | {c.category} |
| Amount | {_money(c.amt)} |
| Card's average transaction before this one (all history since {first_seen:%Y-%m-%d}) | {_money(avg_prior) if pd.notna(avg_prior) else "n/a"} over {int(c.cust_n_prior_txn):,} transactions ({c.amt_to_cust_avg:,.1f}x) |
| Card transactions in the rolling prior 1h / 24h | {int(c.n_txn_prior_1h)} / {int(c.n_txn_prior_24h)} |
| Card spend in the rolling prior 24h | {_money(c.amt_sum_prior_24h)} |
| Time since card's previous transaction | {_hours(c.hours_since_last_txn)} |
| First transaction at this merchant for this card? | {merchant_note} |
| Distinct merchants used by the card (all history) | {prior.merchant.nunique():,} |
| Distance from cardholder's home location to merchant location | {c.dist_home_merchant_km:,.0f} km |
| Model alerts on this card in the prior 30 days | {len(prior_alerts)} |
| Confirmed fraud on this card with labels older than {label_maturity_days} days (known at alert time) | {kf} |

**Account baseline vs this activity**

| Measure | Card baseline (prior 90 days) | This alert | Comparison |
|---|---|---|---|
{base_md}

**Card activity in the 48 hours before the alert**

| Time | Category | Amount |
|---|---|---|
{hist_md}

## 2. Risk indicators (model explanation)

Factors that **raised** the score (SHAP contribution, log-odds):
{up_md}

Factors that **lowered** the score:
{down_md}

These show what the model reacted to. They are indicators, not evidence that fraud occurred, and not causes.

**Facts consistent with the alert pattern**
{cons_md}

**Facts that do not fit the alert pattern**
{incons_md}

**Other relevant history (neutral)**
{other_md}

## 3. Data gaps: investigation timeline

| Step | Source | Status | Finding |
|---|---|---|---|
| Transaction record reviewed | Transaction data | Completed | See section 1 |
| Card history (all history, 90-day baseline, prior 48h) reviewed | Transaction data | Completed | See section 1 |
| Model score and explanation reviewed | Fraud model (LightGBM, SHAP) | Completed | See section 2 |
| Prior model alerts on the card (30 days) | Model scores | Completed | {alerts_note} |
| Prior confirmed fraud on the card | Fraud labels older than {label_maturity_days} days | Completed | {kf} |
| Cardholder contact | Customer channel | **Not available** | Not in the dataset. Required before disposition |
| Device, IP address, channel | Digital/auth logs | **Not available** | Card-present vs card-not-present cannot be established from this record |
| Authorisation details (CVV / 3-D Secure / POS entry mode) | Authorisation system | **Not available** | |
| Merchant risk profile (chargeback rate, onboarding date) | Merchant data | **Not available** | |
| Disputes / fraud reported in the last {label_maturity_days} days | Case management | **Not available at alert time** | Labels not yet mature |

The data is synthetic (Sparkov simulator), so merchant names and locations are generated.

## 4. Escalation recommendation

**{action}**

The score is driven mainly by {drivers.description.iloc[0].lower()} and {drivers.description.iloc[1].lower()} (section 2). Section 2 also lists facts that do not fit the alert pattern. The information in this record neither confirms nor rules out fraud.

**Disposition (to be completed by the analyst after contact; categories as in the workflow):**
- [ ] Close: false positive (cardholder confirms the transaction)
- [ ] Close: explanation obtained and documented
- [ ] Refer to L2 / senior analyst (conflicting or insufficient information)
- [ ] Confirmed / suspected fraud (cardholder denies, or cannot be reached and evidence supports fraud): block card, reissue, start chargeback. L2 confirms the fraud; the supervisor signs off the account action; compliance decides on any regulatory reporting

Rationale (analyst): __________________________________________

## 5. Next steps

1. Contact the cardholder through a registered channel (not contact details supplied with the transaction) to confirm or deny the transaction.
2. Review the card's other transactions in the same period (section 1) for the same pattern.
3. Obtain the missing items in section 3 (authorisation details, device data, merchant profile) where source systems hold them.
4. Record the disposition and outcome so the label feeds monitoring and retraining.
5. If fraud is confirmed: block and reissue the card, and review other cards that transacted with {c.merchant} in the same window.

| Sign-off | Role | Name | Date |
|---|---|---|---|
| Analyst | L1: prepares case and disposition | | |
| Reviewer | L2 / senior analyst: required for referrals and fraud confirmations | | |
| Approver | Supervisor: required for account action and compliance referral | | |
"""
    outcome = f"""# Sample alert case: hindsight outcome (evaluation only)

This file is kept separate from the case report on purpose: an analyst would **not** have this information at alert time.

- Case ID: {c.trans_num[:12]}…
- Dataset label for the alerted transaction: **{"fraud" if c.is_fraud == 1 else "legitimate"}**
- Model score {c[score_col]:.3f} (band: {band})

Outcome: {"false positive (genuine transaction)" if c.is_fraud == 0 else "true positive (fraud)"}.
"""
    return report, outcome
