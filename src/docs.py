"""Generate README.md and the two report documents from saved notebook outputs.

    python -m src.docs

Every number in these documents is read from reports/headline_metrics.json and
the CSV outputs, so re-running the notebooks and then this script keeps the
documentation consistent with the code.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config, summary

R = config.REPORTS



CFL = {
    "tm": ("AML Transaction Monitoring Analyst", "https://claudefinancelab.com/skill/aml-transaction-monitoring-skill/"),
    "triage": ("AML Transaction Monitoring Alert Triage Tool", "https://claudefinancelab.com/skill/aml-transaction-monitoring-alert-triage/"),
    "narr": ("Financial Crime Investigation Narrative Writer", "https://claudefinancelab.com/skill/financial-crime-investigation-narrative-writer/"),
    "sar": ("SAR Narrative Writer", "https://claudefinancelab.com/skill/sar-narrative-writer-skill/"),
    "mrv": ("Model Risk Validator", "https://claudefinancelab.com/skill/model-risk-validator-skill/"),
    "mrg": ("Model Risk Governance Validator", "https://claudefinancelab.com/skill/model-risk-governance-validator/"),
}


def cfl(key):
    name, url = CFL[key]
    return f"[{name}]({url})"


DATA_FIELDS = [
    # field, expected by, in Sparkov?, note
    ("Transaction amount, date/time", "Transaction Monitoring Analyst skill; card-fraud practice", "Yes", "amt, trans_date_trans_time"),
    ("Transaction type / channel (card-present vs card-not-present)", "Transaction Monitoring Analyst skill; card-fraud practice", "Partial", "only implied by category suffix (_pos / _net)"),
    ("Merchant / counterparty identity", "Transaction Monitoring Analyst skill", "Yes (synthetic)", "merchant name; no merchant ID"),
    ("Counterparty jurisdiction / country risk", "Transaction Monitoring Analyst skill", "No", "all transactions are US"),
    ("Merchant category code (MCC)", "Card-fraud practice", "Partial", "14 broad categories, not MCCs"),
    ("Merchant location", "Card-fraud practice", "Yes (synthetic)", "merch_lat / merch_long"),
    ("Card / account identifier", "Transaction Monitoring Analyst skill", "Yes", "cc_num (masked in outputs)"),
    ("Transaction frequency in review period", "Transaction Monitoring Analyst skill", "Derived", "velocity windows in sql/02"),
    ("Customer baseline (historical behaviour)", "Transaction Monitoring Analyst skill", "Derived", "prior averages, 7-day and category baselines"),
    ("Customer type (individual / business)", "Transaction Monitoring Analyst skill", "No", "all individuals"),
    ("Customer demographics (age, gender, occupation)", "Card-fraud practice", "Yes", "excluded from the model (fairness)"),
    ("Customer home location", "Card-fraud practice", "Yes (synthetic)", "used for home-to-merchant distance"),
    ("Account open date / card tenure", "Card-fraud practice", "No", "only 'first seen in data'"),
    ("PEP / high-risk customer flag", "Transaction Monitoring Analyst skill", "No", "AML field; rarely used for card fraud"),
    ("Prior alert and dispute history", "Transaction Monitoring Analyst skill; Alert Triage Tool", "No", "only reconstructable from this project's own model"),
    ("Device ID, IP address, geolocation of device", "Card-fraud practice", "No", "a key gap for card-not-present fraud"),
    ("Authorisation result (CVV, 3-D Secure, AVS, POS entry mode)", "Card-fraud practice", "No", ""),
    ("Credit limit / available balance", "Card-fraud practice", "No", ""),
    ("Fraud label and label date (chargeback / confirmation date)", "Card-fraud practice", "Partial", "is_fraud only; no label date, so label delay is simulated"),
    ("Customer contact / analyst disposition outcomes", "Alert Triage Tool skill", "No", "needed for feedback and override tracking"),
]


GLOSSARY = [
    ("Alert", "A payment the system flags for action: held, or sent to an analyst."),
    ("False alert (false positive, FP)", "An alert on a genuine payment. It costs analyst time and can annoy the customer."),
    ("Missed fraud (false negative, FN)", "A fraudulent payment the system did not flag."),
    ("Fraud caught (true positive, TP)", "A fraudulent payment the system correctly flagged."),
    ("Precision", "Of all alerts, the share that were really fraud."),
    ("Recall", "Of all fraud, the share the system caught."),
    ("F1 score", "A single number that balances precision and recall (their harmonic mean)."),
    ("Precision-Recall Area Under the Curve (PR-AUC)", "How well the model ranks fraud above genuine payments across every possible cut-off, focused on the rare fraud cases. 1.0 is perfect; the fraud rate (about 0.005 here) is what random guessing would score."),
    ("Receiver Operating Characteristic Area Under the Curve (ROC-AUC)", "Another ranking measure. It looks flattering when fraud is rare, so it is reported only as a secondary figure."),
    ("Threshold (cut-off)", "The score above which a payment becomes an alert."),
    ("Auto-hold", "The highest-risk band: the payment is stopped before it goes through, and the customer is asked to confirm it."),
    ("Triage band", "A score range that decides the action: auto-hold, priority review, standard review, or monitor only."),
    ("Analyst levels (L1 / L2)", "First-line (L1) analysts review alerts; second-line (L2) senior analysts confirm fraud and handle unclear cases."),
    ("Train / validation / test split", "Past data used to build the model (train), a later period used to tune it (validation), and a final period used once to check it (test)."),
    ("Time-based split", "Splitting the data by date, so the model is always tested on a period after the one it learned from, as in real life."),
    ("Data leakage", "When a model accidentally sees information it would not have at decision time (such as future payments), making results look better than they really are."),
    ("Class imbalance", "When one outcome is very rare. Here only about 1 payment in 190 is fraud."),
    ("Class weights", "Telling the model to treat each fraud case as more important during training, to offset how rare fraud is."),
    ("Undersampling", "Training on only a random part of the genuine payments, so fraud is less rare in the training data."),
    ("Synthetic Minority Over-sampling Technique (SMOTE)", "Creating artificial fraud examples by blending real ones, so fraud is less rare in training."),
    ("Logistic regression", "A simple model that adds up weighted risk factors. Easy to explain, but it cannot capture combinations such as 'large amount AND night'."),
    ("Random forest", "A model that averages many decision trees."),
    ("Light Gradient Boosting Machine (LightGBM)", "A fast model that builds decision trees one after another, each correcting the previous ones' mistakes. The chosen model here."),
    ("Feature", "An input the model uses, such as the amount or the number of payments in the last 24 hours."),
    ("SHapley Additive exPlanations (SHAP)", "A method that shows how much each feature pushed one payment's score up or down."),
    ("Permutation importance", "How much the model gets worse when one feature is shuffled; a measure of how much it relies on that feature."),
    ("Partial dependence", "A chart of how the average score changes as one feature changes."),
    ("Population Stability Index (PSI)", "A number that measures how much a distribution (for example, the scores) has shifted since a reference period. Above 0.2 is usually a warning."),
    ("Drift", "A change over time in the data or in fraud behaviour that can make the model less accurate."),
    ("Label delay", "Confirmed fraud outcomes arrive weeks later (for example, through chargebacks), so recent performance cannot be measured straight away."),
    ("Chargeback", "When a customer disputes a card payment and the money is reversed. A common source of fraud labels."),
    ("Confidence interval (CI)", "A range that shows how uncertain a number is. A 95% interval is built so that it contains the true value 95% of the time."),
    ("Card-level bootstrap", "Estimating uncertainty by repeatedly resampling whole cards, because fraud on the same card is linked."),
    ("Structured Query Language (SQL)", "The standard language for querying data tables."),
    ("DuckDB", "A fast database engine that runs SQL on a laptop."),
    ("Window function", "An SQL tool that computes values over a range of earlier rows, for example a card's spending in the prior 24 hours."),
    ("Exploratory data analysis (EDA)", "Looking at the data with charts and tables before modelling."),
    ("Sparkov", "The simulator that generated this dataset's card payments."),
    ("Streamlit", "A Python tool for building simple interactive web dashboards."),
    ("Anti-money-laundering (AML)", "The field of detecting criminals moving illegal money through the financial system. Related to, but different from, card fraud."),
    ("Politically exposed person (PEP)", "A customer in a prominent public role, who carries a higher money-laundering risk."),
    ("Suspicious Activity Report (SAR)", "A report that financial institutions file with regulators about possible crime."),
    ("Merchant category code (MCC)", "The standard four-digit code for a merchant's type of business."),
    ("Card verification value (CVV), 3-D Secure, address verification service (AVS)", "Security checks made when a card is used online or by phone."),
    ("Point of sale (POS)", "A card payment made in person at a shop terminal."),
    ("SR 11-7", "United States banking-regulator guidance on managing the risk of using models."),
]


def _expand(text: str) -> str:
    """Spell out abbreviations on first use and replace informal short forms, outside code and links."""
    import re
    first_use = [
        (r"\bPR-AUC\b", "Precision-Recall Area Under the Curve"),
        (r"\bROC-AUC\b", "Receiver Operating Characteristic Area Under the Curve"),
        (r"\bSQL\b", "Structured Query Language"),
        (r"\bSHAP\b", "SHapley Additive exPlanations"),
        (r"\bPSI\b", "Population Stability Index"),
        (r"\bSMOTE\b", "Synthetic Minority Over-sampling Technique"),
        (r"\bEDA\b", "exploratory data analysis"),
        (r"\bAML\b", "anti-money-laundering"),
        (r"\bCI\b", "confidence interval"),
        (r"\bLightGBM\b", "Light Gradient Boosting Machine"),
        (r"\bIP\b", "Internet Protocol"),
        (r"\bID\b", "identifier"),
        (r"\bUS\b", "United States"),
        (r"\bSAR\b", "Suspicious Activity Report"),
        (r"\bPOS\b", "point of sale"),
        (r"\bPEP\b", "politically exposed person"),
        (r"\bPEPs\b", "politically exposed persons"),
        (r"\bMCC\b", "merchant category code"),
        (r"\bMCCs\b", "merchant category codes"),
        (r"\bCVV\b", "card verification value"),
        (r"\bAVS\b", "address verification service"),
        (r"\bCC0\b", "Creative Commons Zero"),
        (r"\bL1\b", "first-line analyst"),
        (r"\bL2\b", "second-line senior analyst"),
    ]
    everywhere = [
        (r"\bvs\.?(?=\s)", "versus"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
        (r"\bCSVs\b", "data files"), (r"(?<=\d)h\b", " hours"), (r"\bAlerts/day\b", "Alerts per day"),
        (r"\balerts/day\b", "alerts per day"), (r"(?<=\d)/day\b", " per day"),
    ]
    # protect fenced code, inline code, link targets and image/link URLs
    parts = re.split(r"(```.*?```|`[^`]*`|!?\[[^\]]*\]\([^)]*\)|<[^>]+>)", text, flags=re.S)
    seen = set()
    for i, part in enumerate(parts):
        if i % 2 == 1:
            continue
        for pat, rep in everywhere:
            part = re.sub(pat, rep, part)
        for pat, full in first_use:
            if pat in seen:
                continue
            m = re.search(pat, part)
            if m:
                before = part[max(0, m.start() - len(full) - 2):m.start()]
                if before.lower() != f"{full} (".lower():
                    part = part[:m.start()] + f"{full} ({m.group(0)})" + part[m.end():]
                seen.add(pat)
        parts[i] = part
    return "".join(parts)


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


def num(x, d=0):
    return f"{x:,.{d}f}"


def load():
    h = summary.build()
    ctx = {"h": h}
    ctx["opts"] = pd.read_csv(R / "threshold_options_val_vs_test.csv").set_index(["option", "split"])
    ctx["test"] = pd.read_csv(R / "model_comparison_test.csv", index_col=0)
    ctx["val"] = pd.read_csv(R / "model_comparison_validation.csv", index_col=0)
    ctx["split"] = pd.read_csv(R / "split_summary.csv", index_col=0)
    ctx["facts"] = pd.read_csv(R / "dataset_facts.csv", index_col=0)["value"]
    ctx["assump"] = pd.read_csv(R / "assumptions.csv")
    ctx["rules"] = pd.read_csv(R / "rules_performance.csv")
    ctx["imb"] = pd.read_csv(R / "imbalance_comparison.csv")
    bands = pd.read_csv(R / "triage_band_table.csv")
    if "band" not in bands.columns:
        bands = bands.rename(columns={bands.columns[0]: "band"})
    ctx["bands"] = bands
    ctx["seg"] = pd.read_csv(R / "segment_error_analysis.csv")
    ctx["abl"] = pd.read_csv(R / "demographic_ablation.csv")
    ctx["pattern"] = pd.read_csv(R / "realism_off_pattern_fraud.csv")
    ctx["noise"] = pd.read_csv(R / "realism_label_noise.csv")
    ctx["stress"] = pd.read_csv(R / "monitoring_stress_tests.csv", index_col=0)
    ctx["mon"] = pd.read_csv(R / "monitoring_monthly.csv")
    ctx["fired"] = pd.read_csv(R / "monitoring_triggers_fired.csv")
    ctx["imp"] = pd.read_csv(R / "feature_importance.csv")
    ctx["sens"] = pd.read_csv(R / "sensitivity_analysis.csv")
    ctx["triage"] = json.loads((R / "triage_bands.json").read_text())
    ctx["meta"] = json.loads((R / "model_metadata.json").read_text())
    ctx["wl"] = pd.read_csv(R / "fp_workload_table.csv").set_index(["option", "split"])
    ctx["eda"] = json.loads((R / "eda_key_facts.json").read_text())
    ctx["lr_grid"] = pd.read_csv(R / "tuning_logreg.csv")
    ctx["ops"] = pd.read_csv(R / "operational_comparison_test.csv", index_col=0)
    ctx["opsum"] = json.loads((R / "operational_summary.json").read_text())
    ctx["savings"] = pd.read_csv(R / "operational_savings.csv")
    ctx["opsens"] = pd.read_csv(R / "operational_sensitivity.csv")
    ctx["fair"] = pd.read_csv(R / "fairness_with_intervals.csv")
    ctx["staff"] = pd.read_csv(R / "operational_staffing.csv")
    ctx["sampling"] = pd.read_csv(R / "missed_fraud_sampling.csv")
    ctx["findings"] = pd.read_csv(R / "review_findings.csv")
    return ctx


REC = "LightGBM + auto-hold, triage bands (recommended)"
FAIR = "Rules + auto-hold (same policy)"
CUR = "Rules, review only (current process)"
OPT = "LightGBM + auto-hold, cost-optimal review line"


def findings_md(c) -> str:
    p = R / "review_findings.csv"
    if not p.exists():
        return "_Independent review pending._"
    f = pd.read_csv(p)
    rows = "\n".join(f"| {r.id} | {r.reviewer} | {r.severity} | {r.finding} | {r.status} | {r.resolution} |" for r in f.itertuples())
    op = (R / "review_opinion.md").read_text() if (R / "review_opinion.md").exists() else ""
    return f"""{op}

| ID | Reviewer | Severity | Finding | Status | Resolution / where documented |
|---|---|---|---|---|---|
{rows}"""


def _cap_name(opts):
    return [o for o in opts.index.get_level_values(0).unique() if o.startswith("Capacity")][0]


def _rec_name(opts):
    return [o for o in opts.index.get_level_values(0).unique() if o.startswith("Recall")][0]


# ---------------------------------------------------------------------------------------------- README
def readme(c) -> str:
    h, H = c["h"], c["h"]["headline"]
    t, v, opts, sp, f = c["test"], c["val"], c["opts"], c["split"], c["facts"]
    cm_te, cm_va = opts.loc[("Cost-minimising", "test")], opts.loc[("Cost-minimising", "validation")]
    cap, rec = _cap_name(opts), _rec_name(opts)
    rules_te = h["rules_test"]
    b = c["bands"][c["bands"].split == "test"].set_index("band")
    abl = c["abl"]; ab_gain = abl.diff_vs_production.iloc[1]
    pat = c["pattern"].set_index("fraud group")
    off = pat.iloc[1]
    noise = c["noise"]
    st = c["stress"]
    meta = c["meta"]
    ops, sv, osum = c["ops"], c["savings"], c["opsum"]

    model_rows = "\n".join(
        f"| {m} | {t.loc[m, 'PR-AUC']:.3f} ({t.loc[m, 'PR-AUC 95% CI']}) | {t.loc[m, 'ROC-AUC']:.3f} | {pct(t.loc[m, 'precision'])} | "
        f"{pct(t.loc[m, 'recall'])} | {num(t.loc[m, 'TP'])} | {num(t.loc[m, 'FP'])} | {t.loc[m, 'alerts_per_day']:.1f} |"
        for m in t.index)
    split_rows = "\n".join(
        f"| {s.capitalize()} | {sp.loc[s, 'start']} to {sp.loc[s, 'end']} | {num(sp.loc[s, 'transactions'])} | "
        f"{num(sp.loc[s, 'frauds'])} | {sp.loc[s, 'fraud_rate_pct']:.3f}% |" for s in sp.index)
    assump_rows = "\n".join(f"| {r.assumption} | {r.value} | {r.rationale} |" for r in c["assump"].itertuples())
    opt_rows = "\n".join(
        f"| {o} | {opts.loc[(o, 'validation'), 'threshold']:.4f} | {opts.loc[(o, 'test'), 'alerts_per_day']:.1f} | "
        f"{pct(opts.loc[(o, 'test'), 'precision'])} | {pct(opts.loc[(o, 'test'), 'recall'], 1)} | {usd(opts.loc[(o, 'test'), 'total_cost_usd'])} |"
        for o in ["Cost-minimising", cap, rec])

    return f"""# Fraud Detection and Risk-Threshold Optimisation

**How can a financial institution catch more fraudulent card transactions without generating an unmanageable number of false alerts?**

## In plain English

When criminals use stolen card details, banks lose money. A bank sees thousands of card payments a day and can't check them all by hand, so it has to decide which few payments a person should look at.

This project builds that decision system and tests it on {f['Transactions']} example card payments from a public, computer-generated dataset:
1. **It learns what fraud looked like in the past**, for example large purchases late at night, sudden bursts of spending, or amounts far above what the customer normally spends.
2. **It gives every new payment a risk score.**
3. **It turns the score into an action:** stop the payment and ask the customer, send it to an analyst to check, or let it through. The cut-off points are chosen by weighing the money lost to missed fraud against the cost of analysts' time.
4. **It explains every alert in plain terms**, checks whether different customer groups are treated fairly, and watches for signs that the system has stopped working.

**The result, on payments the system had never seen:** it stopped **{pct(ops.loc[REC, 'fraud_prevented_pct'])}** of fraud, against **{pct(ops.loc[FAIR, 'fraud_prevented_pct'])}** for a set of simple rules. It asked analysts to check about **{ops.loc[REC, 'reviews_per_day']:.0f}** payments a day instead of **{ops.loc[FAIR, 'reviews_per_day']:.0f}**.

Because the data is computer-generated, real-world results would be lower. This page explains why, and how much lower was measured.

**How to read this page:** this summary and the results table are for everyone. The later sections give the technical detail, and the [glossary](#glossary) at the end explains every technical term in one sentence.

## What the project covers

This project answers that question end to end, the way a fraud-risk team would:
1. **Features built in SQL with DuckDB** that only use information available at authorisation time.
2. **Time-based validation**, with the test set used once.
3. **A transparent rule baseline** compared against logistic regression, random forest and LightGBM.
4. **An alert threshold chosen in dollars and analyst workload**, not by accuracy.
5. **Alert explanations with SHAP**, error analysis by segment and a fairness review.
6. **Triage bands and a sample case report** for analysts.
7. **Production-style monitoring**: drift, label delay and stress tests.
8. **An operational simulation** of the alert workflow (what auto-hold and post-authorisation review actually prevent), and **three independent reviews** with a findings log.
9. **A Streamlit dashboard.**

> **Read this first.** The dataset is **synthetic** (the Sparkov simulator). Its fraud follows a few scripted patterns, so the headline metrics are much higher than you should expect on real card data. Notebooks 05 and 06 measure how much of the performance depends on those patterns and on workflow assumptions. The value of this project is the **workflow and the decisions**, which transfer directly to real data; the specific numbers do not.

## Headline results (test period: {sp.loc['test', 'start']} to {sp.loc['test', 'end']}, {H['test_days']} days, unseen during development)

These results come from an **operational simulation** (notebook 06), not the idealised cost model:
- **Auto-hold** stops a transaction at authorisation.
- **Review-queue alerts** are worked *after* authorisation, so they can only stop the rest of a fraud burst by blocking the card after the review delay (12 hours).
- **The rules** are a proxy for the current process, built in this project. They get the same auto-hold policy: hold only where validation precision is at least 95%.

| Test period | Rules, review only | Rules + auto-hold (same policy) | **LightGBM + auto-hold + triage bands** |
|---|---|---|---|
| Frauds prevented | {pct(ops.loc[CUR, 'fraud_prevented_pct'])} | {pct(ops.loc[FAIR, 'fraud_prevented_pct'])} | **{pct(ops.loc[REC, 'fraud_prevented_pct'])}** |
| Fraud dollars lost | {usd(ops.loc[CUR, 'fraud_usd_lost'])} | {usd(ops.loc[FAIR, 'fraud_usd_lost'])} | **{usd(ops.loc[REC, 'fraud_usd_lost'])}** |
| Analyst reviews per day | {ops.loc[CUR, 'reviews_per_day']:.1f} | {ops.loc[FAIR, 'reviews_per_day']:.1f} | **{ops.loc[REC, 'reviews_per_day']:.1f}** |
| Auto-holds per day | {ops.loc[CUR, 'holds_per_day']:.1f} | {ops.loc[FAIR, 'holds_per_day']:.1f} | **{ops.loc[REC, 'holds_per_day']:.1f}** |
| Genuine customers wrongly held (period) | {num(ops.loc[CUR, 'wrong_holds'])} | {num(ops.loc[FAIR, 'wrong_holds'])} | **{num(ops.loc[REC, 'wrong_holds'])}** |
| Simulated total cost | {usd(ops.loc[CUR, 'total_cost_usd'])} | {usd(ops.loc[FAIR, 'total_cost_usd'])} | **{usd(ops.loc[REC, 'total_cost_usd'])}** |

- **Cost saving:** against the rules under the same policy, the saving is **{usd(sv.saving_usd.iloc[0])}** over {H['test_days']} days, with a 95% card-bootstrap interval of {usd(sv.ci95_low.iloc[0])} to {usd(sv.ci95_high.iloc[0])}. The saving stays positive in every sensitivity scenario (review delay 4–24h, recovery 0–50%, friction and review costs).
- **Model-only view (at a fixed threshold):** the model ranks fraud far better. Test PR-AUC is **{t.loc['LightGBM', 'PR-AUC']:.3f}** vs {t.loc['Rules baseline', 'PR-AUC']:.3f} for the rules. With the rules' own alert volume, it catches {num(H['same_volume_extra_frauds'])} more frauds.
- **Most of the gain comes from auto-hold**, which is possible only because the model's top scores are 95% precise. Reviews after authorisation stop only the tail of a burst.

![Operational comparison](reports/figures/operational_comparison.png)

## Data

| Item | Value |
|---|---|
| Source | Kaggle, [*Credit Card Transactions Fraud Detection Dataset*](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (kartik2112), generated with the Sparkov simulator |
| Licence | CC0 1.0 (public domain), as listed in the [Amazon Science fraud-dataset benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) |
| Transactions | {f['Transactions']} ({f['Date range']}) |
| Fraud rate | {f['Fraud rate']} ({f['Fraudulent transactions']} frauds; about {f['Legit : fraud ratio']} legitimate to fraud) |
| Customers (cards) / merchants | {f['Cards (customers)']} / {f['Merchants']} across {f['Merchant categories']} categories |
| Anonymised? | Not anonymised with principal component analysis (PCA). It is **synthetic** with readable fields (amount, category, time, location, age, gender). Names and street addresses are fake and are dropped during cleaning |
| Time order preserved? | Yes: real timestamps. The two Kaggle files are combined and re-split by time |
| Duplicates / missing values | None found (audited in SQL, `sql/01_load_and_clean.sql`) |

**Features used ({len(pd.read_csv(R / 'feature_list.csv').query('~group.str.startswith("Demographic")', engine='python'))} features, including merchant category).** All of them are computed from **earlier transactions only**, in `sql/02_behavioural_features.sql`. They cover:
- **Transaction:** log amount, category, hour, weekday, night flag.
- **Velocity:** transaction count and spend over the prior 1 hour, 24 hours and 7 days.
- **Deviation from the customer's norm:** amount vs the customer's historical, 7-day and same-category average, and an amount z-score.
- **Recency and novelty:** hours since the last transaction, first use of a merchant or category, card history length.
- **Context:** city population, distance from home to merchant.

**Fields a real fraud team would expect, and what this dataset has.** The expected fields come from the {cfl('tm')} skill plus common card-fraud practice:

| Field | Expected by | In Sparkov? | Note |
|---|---|---|---|
{chr(10).join(f"| {f} | {e} | {h} | {n} |" for f, e, h, n in DATA_FIELDS)}

Age and gender are **deliberately excluded** (see Fairness). State is excluded because it nearly identifies individual customers. The full list is in `reports/feature_list.csv`.

## Approach

| Step | What | Why |
|---|---|---|
| 1. SQL prep | Type casting, de-duplication, missing-value handling, backward-only window features | Features must be computable at authorisation time. Tests prove no window looks ahead (`tests/test_leakage.py`) |
| 2. EDA | Fraud by hour, weekday, category, state, age, gender, amount | {pct(c['eda']['night_share_of_fraud'])} of fraud is between 22:00 and 04:00; $250+ transactions are {pct(c['eda']['amt250_share_of_volume'], 1)} of volume but {pct(c['eda']['amt250_share_of_fraud'])} of fraud |
| 3. Time split | 60 / 20 / 20 by date (below) | Production scores the future. A random split leaks transactions from the same fraud burst into test |
| 4. Models | 4 rules, logistic regression, random forest, LightGBM; class weights vs undersampling vs SMOTE | PR-AUC is primary; confidence intervals come from a card-level bootstrap |
| 5. Thresholds | Cost sweep, capacity and recall-target options, ±50% sensitivity | The threshold is a business decision, so it is made explicit in dollars |
| 6. Explainability | Gain and permutation importance, SHAP (global, dependence, local), partial dependence, segment errors, fairness ablation | Analysts need reasons; risk teams need to know where the model fails |
| 7. Triage | Auto-hold, priority, standard and monitor-only bands; sample case report | Turns scores into queues and actions |
| 8. Monitoring | Monthly and weekly PSI (including the high-score tail), precision/recall, alert volume, missing rate; triggers; stress tests; realism checks | Drift, label delay and adversarial adaptation |
| 9. Operational review | Workflow simulation, staffing by role on peak days, missed-fraud sampling, fairness with confidence intervals; independent review findings log | Tests whether the business case survives how alerts are actually worked |

| Split | Dates | Transactions | Frauds | Fraud rate |
|---|---|---|---|---|
{split_rows}

## Model comparison (test period; each model at the threshold that maximised its F1 score on validation)

| Model | PR-AUC [95% CI] | ROC-AUC | Precision | Recall | Frauds caught | False alerts | Alerts per day |
|---|---|---|---|---|---|---|---|
{model_rows}

- LightGBM uses SMOTE (1:10), `num_leaves={meta['lgbm_num_leaves']}` and {meta['lgbm_n_trees']:,} trees, all chosen on validation.
- The four imbalance treatments were within {c['imb'].val_pr_auc.max() - c['imb'].val_pr_auc.min():.3f} validation PR-AUC of each other. The features matter far more than the resampling.
- Validation PR-AUC was {v.loc['LightGBM', 'PR-AUC']:.3f}; test PR-AUC was {t.loc['LightGBM', 'PR-AUC']:.3f}.

![PR curves](reports/figures/pr_curves_test.png)

## Threshold choice (idealised cost model, notebook 03)

This first-pass model assumes that **every alerted fraud is stopped**. The independent review showed that this holds only for auto-hold (see the headline). The idealised model counts every alert:

`Total cost = missed-fraud amount + (TP + FP) x $5 review`, where TP is frauds caught and FP is false alerts.

It reviews every alert, not just the false ones. A variant that counts only false alerts is reported alongside. Thresholds are chosen on validation and applied unchanged to test:

| Option | Threshold | Alerts/day (test) | Precision | Recall | Total cost (test) |
|---|---|---|---|---|---|
{opt_rows}

- **Stable under different cost assumptions:** with fraud loss and review cost each moved ±50%, the optimal volume stays between {H['sensitivity_alerts_per_day_range'][0]:.0f} and {H['sensitivity_alerts_per_day_range'][1]:.0f} alerts/day, and recall between {pct(H['sensitivity_recall_range'][0])} and {pct(H['sensitivity_recall_range'][1])}.
- **Same workload as the rules:** at the rules' own volume of {rules_te['alerts_per_day']:.0f} alerts/day, LightGBM catches {num(H['same_volume_extra_frauds'])} more frauds (recall {pct(H['same_volume_lgbm_recall'])} vs {pct(rules_te['recall'])}).

![Cost vs alert volume](reports/figures/cost_vs_alert_volume.png)

In the operational simulation, the cost-optimal *review* line moves up to {osum['t_review_operational']:.3f}: once auto-hold has blocked the card, reviewing lower scores prevents little extra fraud. The recommendation keeps the triage bands' review line ({osum['t_review_design']:.4f}) anyway. The analysts are a fixed cost with spare capacity, and every review produces a label for monitoring. Notebook 06 reports both options.

## Assumptions

| Assumption | Value | Rationale |
|---|---|---|
{assump_rows}

Notebook 03 assumes a caught fraud is stopped before loss. Notebook 06 relaxes this. It adds its own assumptions: a 12h review delay, a card block that stops the rest of the fraud episode, a 3-day reissue gap, $1 per auto-hold, $10 friction per wrongly held customer, and correct dispositions.

## Triage (test period)

| Band | Rule (set on validation) | Per day | Hit rate | Share of all fraud |
|---|---|---|---|---|
| Auto-hold | score ≥ {c['triage']['hold']:.3f} (≥95% precision) | {b.loc['Auto-hold', 'per_day']:.1f} | {pct(b.loc['Auto-hold', 'fraud_rate_in_band'])} | {pct(b.loc['Auto-hold', 'share_of_all_fraud'])} |
| Priority review | score ≥ {c['triage']['review']:.3f} (≥80% cumulative precision) | {b.loc['Priority review', 'per_day']:.1f} | {pct(b.loc['Priority review', 'fraud_rate_in_band'])} | {pct(b.loc['Priority review', 'share_of_all_fraud'])} |
| Standard review | score ≥ {c['triage']['standard']:.4f} (cost-optimal) | {b.loc['Standard review', 'per_day']:.1f} | {pct(b.loc['Standard review', 'fraud_rate_in_band'], 1)} | {pct(b.loc['Standard review', 'share_of_all_fraud'], 1)} |
| Monitor only | score ≥ {c['triage']['monitor']:.5f} (capacity fill) | {b.loc['Monitor only', 'per_day']:.1f} | {pct(b.loc['Monitor only', 'fraud_rate_in_band'], 1)} | {pct(b.loc['Monitor only', 'share_of_all_fraud'], 1)} |

There is a sample analyst case report in [`reports/sample_alert_case.md`](reports/sample_alert_case.md).

## Monitoring (simulated production, Aug to Dec 2020)

![Monitoring](reports/figures/monitoring_dashboard.png)

- **Stable model, falling fraud rate:** recall stayed between {pct(h['monitoring_summary']['live_recall_range'][0])} and {pct(h['monitoring_summary']['live_recall_range'][1])} every live month. Precision fell to {pct(h['monitoring_summary']['live_precision_range'][0])} in December, because the fraud rate fell during holiday volume; that's a base-rate effect, not model decay.
- **Only one trigger fired:** December's 24-hour velocity features had PSI > 0.2, consistent with holiday volume. Score PSI peaked at {h['monitoring_summary']['max_live_score_psi']:.2f}, and a new score-tail PSI (top 10% of scores) stayed low. The "no retrain" conclusion is **provisional**, because December labels are not mature.
- **Stress test, fraudsters move to daytime:** recall fell from {pct(st.iloc[0].recall)} to {pct(st.iloc[1].recall)} while score PSI did not move. Population drift metrics cannot see new fraud patterns, which is why the plan adds fast partial labels and a random sample of below-threshold reviews.
- **Stress test, feature-store outage:** the missing-input rate rose to {st.iloc[3].missing_rate_pct:.0f}% and alerts rose from {st.iloc[2].alerts_per_day:.0f} to {st.iloc[3].alerts_per_day:.0f}/day. The daily missing-data trigger catches it immediately.

The full plan (daily, weekly, monthly and quarterly checks, triggers and recalibration protocol) is in [`reports/model_risk_and_monitoring.md`](reports/model_risk_and_monitoring.md).

## Recommendation (summary)

**Recommendation: approve for a 3-month shadow run, not for live decisions yet.**
- **Shadow set-up:** score live traffic with LightGBM, apply the triage bands, and use auto-hold only in the top band (at least 95% precision on validation).
- **Keep the rules running** during the shadow period as the live control.
- **Go-live conditions:** fairness sign-off, replacing SMOTE, and re-validating on the institution's own data. These are the open findings below.

The one-page recommendation for a head of fraud operations is [`reports/business_recommendation.md`](reports/business_recommendation.md). The independent review findings and their status are in [`reports/model_risk_and_monitoring.md`](reports/model_risk_and_monitoring.md#10-independent-review-findings-log).

## Limitations and open issues

- **Synthetic data inflates every metric.** {pct(pat.iloc[0].share_of_fraud)} of test fraud matches one scripted pattern (night-time and $250+). On the remaining "off-pattern" fraud, PR-AUC is {off.pr_auc_vs_all_legit:.3f} and recall {pct(off.recall_at_threshold)}. The amount-SHAP curve shows the model has learned the simulator's fraud-amount bands, which would not transfer.
- **Operational simulation assumptions.** Analysts and customers are assumed to disposition correctly, the review delay is an average, and a card block stops the rest of the episode. The saving interval covers sampling uncertainty, not these assumptions (see the sensitivity table in notebook 06).
- **Open review findings** (details in the findings log):
  - **Fairness.** False-alert rates for older age bands exceed a proposed 1.25x tolerance even though age is not a model input. A proxy analysis and compliance sign-off are needed.
  - **SMOTE creates impossible synthetic records.** Switching to class weights costs about 0.005 PR-AUC and should be done before production.
  - **The validation set is reused** for tuning, early stopping and thresholds, so validation results are optimistic (test is clean).
  - **City population acts as a customer identifier, and card history length acts as a time proxy.** Both should be dropped or capped.
- **Labels are assumed complete and immediate** in training. Hiding 40% of fraud episodes lowers test PR-AUC from {noise.test_pr_auc_true_labels.iloc[0]:.3f} to {noise.test_pr_auc_true_labels.iloc[-1]:.3f} (undersampled variant, for speed). Real under-reporting is not random.
- **No selection bias from existing rules:** Sparkov labels every transaction. In production, labels exist mainly for what was reviewed or disputed.
- **Missing data sources:** no device, IP, channel, authorisation or merchant-risk data (see the fields table).
- **Demographics:** adding age and gender raises validation PR-AUC by {ab_gain:+.4f}. They remain excluded.
- **Logistic regression's best `C` ({c['lr_grid'].loc[c['lr_grid'].val_pr_auc.idxmax(), 'C']}) is at the edge of the grid searched**, but its validation PR-AUC had flattened ({c['lr_grid'].val_pr_auc.min():.3f} to {c['lr_grid'].val_pr_auc.max():.3f}), so it is kept as a baseline only.
- **Scores are not calibrated probabilities** (SMOTE shifts them). Thresholds and bands use ranks set on validation.

## How ClaudeFinanceLab was used

[ClaudeFinanceLab](https://claudefinancelab.com) publishes finance and financial-crime skills as instruction blocks. They were used for **structure, frameworks and review only**: every number, threshold and finding in this repo comes from the project's own code and data.
- The skills are AML-oriented, so they were adapted to card fraud, and the places they don't fit are stated.
- Their text is linked here, not copied, because the site's licence doesn't clearly cover reuse.
- The skills were not installed as Claude skills. Their published instructions were read and applied as frameworks.

| Stage | Resource | How it was used | What changed |
|---|---|---|---|
| Business framing | {cfl('triage')}, {cfl('tm')} | Workflow stages, roles (L1 / L2 / supervisor / compliance) and the four disposition outcomes, adapted from AML alerts to card-fraud alerts | New [`reports/fraud_triage_workflow.md`](reports/fraud_triage_workflow.md) |
| Data design | {cfl('tm')} | Its list of transaction and customer fields, compared with the dataset | New "fields" table in *Data* above |
| Rule baseline | {cfl('tm')} | Red-flag categories checked against our rules | **No change.** Our rules already cover deviation from the customer's baseline and velocity. The AML typologies (structuring, layering, high-risk jurisdictions, PEPs) need fields this dataset lacks |
| Model review | {cfl('mrv')} | Conceptual-soundness checklist (e.g. linear models for non-linear relationships) used in the independent model-risk review | Supports keeping logistic regression as a baseline only |
| Threshold setting | Project's ops-manager review prompt | Independent reviewer assessed whether the data supports the threshold, and what data is missing | See *Limitations*; thresholds remain computed in notebook 03 |
| Case workflow | {cfl('triage')}, {cfl('narr')}, {cfl('sar')} | Referral block, account-baseline comparison, investigation-timeline table for data gaps, disposition checkboxes, sign-off fields, neutral wording | [`reports/sample_alert_case.md`](reports/sample_alert_case.md) restructured (generated by `src/case_report.py`) |
| Monitoring | {cfl('mrg')} | Model tiering, override-rate tracking, change management, validation cadence | Added to [`reports/model_risk_and_monitoring.md`](reports/model_risk_and_monitoring.md) |
| Model-risk review | {cfl('mrv')} + project reviewer prompt | Severity scale (Critical / High / Medium / Low) for the independent review findings | Findings log in the model-risk document |
| Documentation | {cfl('mrg')} | Findings-log and validation-opinion structure | Model-risk document restructured |

## How to reproduce

```bash
git clone https://github.com/pooja003-cloud/fraud-detection-risk-thresholds.git
cd fraud-detection-risk-thresholds
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download fraudTrain.csv and fraudTest.csv from Kaggle into data/raw/ (see data/README.md)
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb   # 01 to 06, in order (~50 min on 2 cores)
pytest -q                                                               # cost, leakage and drift tests
python -m src.docs                                                      # regenerate README and reports from outputs
streamlit run dashboard/app.py                                          # dashboard (works from committed CSVs)
```

Seeds are fixed (`src/config.py`), and versions are pinned in `requirements.txt` (Python 3.11).

## Repository structure

```
fraud-detection-risk-thresholds/
├── README.md                  # generated by `python -m src.docs`
├── requirements.txt           # pinned versions
├── data/                      # raw/ + processed/ (not committed; see data/README.md)
├── sql/                       # DuckDB: 01 clean, 02 backward-only features, 03 EDA segments
├── notebooks/                 # 01 prep+EDA · 02 models · 03 thresholds+cost · 04 explainability+triage · 05 monitoring · 06 operational review
├── src/                       # config, data, features, rules, models, metrics, costs, operations, explain, triage, case_report, drift, docs
├── dashboard/                 # Streamlit app + the small CSVs it reads
├── reports/
│   ├── figures/               # every chart, with titles that state the finding
│   ├── threshold_cost_table.csv
│   ├── fp_workload_table.csv
│   ├── business_recommendation.md
│   ├── model_risk_and_monitoring.md
│   ├── fraud_triage_workflow.md
│   ├── review_findings.csv        # independent review findings and status
│   ├── sample_alert_case.md       # generated by src/case_report.py
│   └── sample_alert_case_outcome.md   # hindsight label, kept separate on purpose
└── tests/                     # cost function, leakage and drift tests
```

## Glossary

| Term | Meaning |
|---|---|
{chr(10).join(f"| {t} | {d} |" for t, d in GLOSSARY)}
"""


# ---------------------------------------------------------------------------------------------- triage workflow
def workflow(c) -> str:
    b = c["bands"][c["bands"].split == "test"].set_index("band")
    tr = c["triage"]
    per = lambda k: b.loc[k, "per_day"]
    hit = lambda k: b.loc[k, "fraud_rate_in_band"]
    share = lambda k: b.loc[k, "share_of_all_fraud"]
    return f"""# Fraud alert triage workflow

How a model alert moves from generation to closure: who does what, when, and what gets recorded.
- **Structure:** adapted from ClaudeFinanceLab's {cfl('triage')} and {cfl('tm')} skills (AML frameworks, adapted here to card-transaction fraud).
- **Volumes and hit rates:** from this project's test period (notebook 04). Service levels and staffing are proposed design choices, not measured values.

## Flow

```mermaid
flowchart LR
    A[Transaction authorised<br/>and scored in real time] --> B{{Score band}}
    B -->|Auto-hold| C[Hold transaction<br/>+ customer confirmation request]
    B -->|Priority review| D[L1 queue<br/>same-day SLA]
    B -->|Standard review| E[L1 queue<br/>24h SLA]
    B -->|Monitor only| F[No alert<br/>logged for sampling and drift]
    C --> G{{Customer response}}
    G -->|Confirms genuine| H[Release<br/>Close: false positive]
    G -->|Denies / no response| I[Confirmed or suspected fraud]
    D --> J[L1 review:<br/>facts, baseline, SHAP reasons, data gaps]
    E --> J
    J --> K{{Disposition}}
    K -->|Close: false positive| L[Close + record label]
    K -->|Close: explanation obtained| L
    K -->|Refer| M[L2 / senior analyst]
    K -->|Suspected fraud| I
    M --> K
    I --> N[Block card, reissue,<br/>chargeback, link analysis]
    N --> O[L2 confirms; supervisor signs off<br/>action and compliance referral]
    O --> L
    L --> P[Labels feed monitoring<br/>and retraining]
    F --> P
```

## Stages and roles

| # | Stage | Who | What happens | Record |
|---|---|---|---|---|
| 1 | Alert generation | Fraud model (owned by fraud analytics) | Every transaction is scored at authorisation and assigned a band from the cut-offs below | Score, band, model version, threshold version |
| 2 | Auto-hold confirmation | Automated channel, 24/7 (SMS / app push); CX agents for call-backs | Held transaction; the cardholder is asked to confirm through a registered channel. **Time-out:** an unanswered hold stays held and goes to the L1 priority queue at 08:00 | Customer response and time |
| 3 | L1 review | L1 fraud analyst | Six steps: collect alert data → check customer and account baseline → analyse the transaction and recent card activity → match to known fraud patterns → decide disposition → document | Case report ([sample](sample_alert_case.md)) |
| 4 | Disposition | L1 fraud analyst | One of four outcomes (below) | Disposition and rationale |
| 5 | Escalation and confirmation | L2 / senior analyst | Takes unclear or conflicting cases (referrals) and confirms every fraud disposition (four-eyes) | L2 decision |
| 6 | Action and sign-off | Supervisor | Signs off block, reissue and chargeback, and decides whether to refer to compliance for regulatory reporting | Sign-off |
| 7 | Closure and feedback | Fraud analytics | Outcome becomes the training/monitoring label; QA re-reviews a sample of closed cases; a random sample of below-threshold transactions is reviewed each month | Label, QA result |

## Score bands (cut-offs set on validation; volumes from the test period)

| Band | Score cut-off | Alerts/day | Share that are fraud | Share of all fraud | Proposed SLA |
|---|---|---|---|---|---|
| Auto-hold | ≥ {tr['hold']:.3f} | {per('Auto-hold'):.1f} | {pct(hit('Auto-hold'))} | {pct(share('Auto-hold'))} | Automated confirmation, 24/7 |
| Priority review | ≥ {tr['review']:.3f} | {per('Priority review'):.1f} | {pct(hit('Priority review'))} | {pct(share('Priority review'))} | Same day |
| Standard review | ≥ {tr['standard']:.4f} | {per('Standard review'):.1f} | {pct(hit('Standard review'), 1)} | {pct(share('Standard review'), 1)} | 24 hours |
| Monitor only | ≥ {tr['monitor']:.5f} | {per('Monitor only'):.1f} | {pct(hit('Monitor only'), 1)} | {pct(share('Monitor only'), 1)} | Reviewed with spare capacity (labels / QA) |

The band volumes above count every transaction that scores into a band. In operation, a confirmed fraud blocks the card, so later transactions on it never reach a queue. The notebook-06 simulation of the recommended set-up gives {c['ops'].loc[REC, 'holds_per_day']:.1f} auto-holds and {c['ops'].loc[REC, 'reviews_per_day']:.1f} analyst reviews per day.

## Staffing on peak days (notebook 06; handling times are assumptions)

| Role | Tasks per day (p95 day) | Staff-hours per day (p95 day) |
|---|---|---|
{chr(10).join(f"| {r.role} | {r.p95_tasks_per_day:.1f} | {r.p95_hours_per_day:.2f} |" for r in c['staff'].itertuples())}

About **{pct(c['opsum']['night_share_of_holds'])} of auto-holds occur between 22:00 and 04:00**, which is why confirmation is automated.

## Disposition outcomes (adapted from the Alert Triage Tool's four categories)

| Outcome | When | Follow-up |
|---|---|---|
| Close: false positive | Cardholder confirms the transaction, or the evidence clearly shows it is genuine | Release; label = legitimate |
| Close: explanation obtained | Unusual but explained (travel, large planned purchase) | Document explanation; label = legitimate |
| Refer to L2 / senior analyst | Conflicting or insufficient information | L2 decides within SLA |
| Confirmed / suspected fraud | Cardholder denies, or cannot be reached and the evidence supports fraud | Block, reissue, chargeback; supervisor sign-off; compliance decides on regulatory reporting |

## Controls
- **Four-eyes:** L2 confirms every fraud disposition made by L1. The supervisor signs off account actions (block, reissue) and any compliance referral.
- **QA:** re-review a sample of closed cases each month, and track the *analyst override rate* (dispositions that contradict the model band) as a monitoring metric.
- **Missed-fraud check:** use spare capacity to review the whole monitor-only band (about {c['sampling'].expected_frauds_found_per_month.iloc[0]:.0f} near-miss frauds a month on test). Fraud far below the threshold can only be found through customer reports and chargebacks, because a random sample there finds almost nothing.
- **Customer-facing wording:** neutral and non-accusatory, e.g. "we noticed an unusual transaction".
"""


# ---------------------------------------------------------------------------------------------- business recommendation
def business(c) -> str:
    h, H = c["h"], c["h"]["headline"]
    ops, sv, osum, st = c["ops"], c["savings"], c["opsum"], c["staff"]
    samp = c["sampling"]
    fair = c["fair"]
    fails = fair[fair["fpr_tolerance_1.25x"] == "fail"]
    t = c["test"]
    return f"""# Recommendation: shadow-run a scored, triaged fraud alert queue

**To:** Head of Fraud Operations · **Basis:** back-test on {H['test_days']} days of unseen transactions (Aug 25 to Dec 31, 2020), synthetic Sparkov data, independently reviewed

## Recommendation
Run the LightGBM fraud score for **three months in shadow** alongside the current rules, using four triage bands. Auto-hold applies only to the top band, which was at least 95% precise on validation. Switch the rules off only after the shadow run and the go-live conditions below.

## Fraud prevented vs the current rules (operational simulation, test period)
| | Rules, review only | Rules + auto-hold (same policy) | **Recommended** |
|---|---|---|---|
| Frauds prevented | {pct(ops.loc[CUR, 'fraud_prevented_pct'])} | {pct(ops.loc[FAIR, 'fraud_prevented_pct'])} | **{pct(ops.loc[REC, 'fraud_prevented_pct'])}** |
| Fraud dollars lost | {usd(ops.loc[CUR, 'fraud_usd_lost'])} | {usd(ops.loc[FAIR, 'fraud_usd_lost'])} | **{usd(ops.loc[REC, 'fraud_usd_lost'])}** |
| Simulated total cost | {usd(ops.loc[CUR, 'total_cost_usd'])} | {usd(ops.loc[FAIR, 'total_cost_usd'])} | **{usd(ops.loc[REC, 'total_cost_usd'])}** |

- **Where the gain comes from:** mostly the auto-hold band. It stops a fraudulent transaction *before* authorisation and triggers a card block that stops the rest of the fraud burst. Alerts reviewed after authorisation can only stop what comes next.
- **Cost saving:** against the rules under the same policy, **{usd(sv.saving_usd.iloc[0])} over {H['test_days']} days** (95% interval {usd(sv.ci95_low.iloc[0])} to {usd(sv.ci95_high.iloc[0])}). It stays positive in every sensitivity case tested.
- **Assumptions:** fraud loss = full amount, $5 per review, $1 per hold, $10 friction per wrongly held customer, 12h review delay. **Treat the figure as illustrative**: the data is synthetic and the rules are a proxy built for this project.

## Alert volume and workload
- **Daily load:** {ops.loc[REC, 'reviews_per_day']:.1f} analyst reviews and {ops.loc[REC, 'holds_per_day']:.1f} auto-holds a day, against {ops.loc[CUR, 'reviews_per_day']:.1f} reviews for the rules. On the busiest days (p95), all roles together need about **{st.p95_hours_per_day.sum():.1f} staff-hours a day**: L1 reviews, L2 and supervisor sign-off of fraud confirmations, and customer contact.
- **Use the spare capacity** (about {osum['spare_reviews_per_day']:.0f} reviews a day) to review the whole **monitor-only band** ({samp.per_day.iloc[0]:.0f} a day). It should surface about {samp.expected_frauds_found_per_month.iloc[0]:.0f} near-miss frauds a month, which is the only practical way to measure what the model misses. A small random sample far below the threshold finds almost nothing.
- **Overnight holds:** **{pct(osum['night_share_of_holds'])} of auto-holds happen between 22:00 and 04:00**, so customer confirmation must be automated around the clock (SMS/app), with a time-out rule for unanswered holds.
- **Most reviews are genuine:** only about {pct(ops.loc[REC, 'review_precision'])} of review-queue alerts are fraud, because auto-hold has already caught most of it. Set analysts' expectations accordingly.

## Risks
- **Synthetic data:** the model partly learned the simulator's patterns. Test PR-AUC is {t.loc['LightGBM', 'PR-AUC']:.3f} here; expect much less on real traffic. That's why the shadow run comes first.
- **Fairness:** false-alert rates for {', '.join(fails.group) if len(fails) else 'no group'} age groups exceed the proposed 1.25x tolerance, even though age is not a model input. A go-live condition.
- **New fraud patterns:** in a simulation where fraud moved to daytime, recall fell from {pct(c['stress'].iloc[0].recall)} to {pct(c['stress'].iloc[1].recall)} without any drift alarm. We need fast labels from customer reports and analyst dispositions.
- **Customer friction:** {num(ops.loc[REC, 'wrong_holds'])} genuine transactions were held over the test period, and blocked cards decline genuine spending until they're reissued.
- **Label delay:** precision and recall for the latest two months are always provisional.

## Next steps
1. **Months 0–3:** shadow-score live traffic. Compare catches, holds and false alerts with the rules every week.
2. **Model fixes before go-live:** replace SMOTE with class weights, drop the customer-identifying features (city population, card history length), and re-validate on our own data.
3. **Fairness:** agree tolerances with compliance and run a proxy analysis.
4. **Operations:** build the 24/7 automated confirmation channel and the time-out rule, and agree cost assumptions with Finance.
5. **Model-risk sign-off** (Tier 1), then go live with monthly threshold review and quarterly retraining.
"""


# ---------------------------------------------------------------------------------------------- model risk
def model_risk(c) -> str:
    h, H = c["h"], c["h"]["headline"]
    opts = c["opts"]
    rec = _rec_name(opts)
    seg = c["seg"]
    age = seg[seg.segment_type == "age_band"].set_index("segment")
    gen = seg[seg.segment_type == "gender"].set_index("segment")
    abl = c["abl"]; pat = c["pattern"]; noise = c["noise"]; st = c["stress"]; mon = c["mon"]
    worst = seg[(seg.segment_type == "category") & (seg.frauds >= 20)].sort_values("recall").head(3)
    fired = c["fired"]
    fired_txt = "; ".join(f"{r.period}: {r.trigger} ({r.detail})" for r in fired.itertuples()) or "none"
    imp = c["imp"]
    return f"""# Model risk and monitoring

Scope: the LightGBM fraud score (SMOTE 1:10, `num_leaves={c['meta']['lgbm_num_leaves']}`), used with a cost-minimising alert threshold and four triage bands. Evidence comes from notebooks 02–05. Every number here is generated from their saved outputs.

## 1. Data limitations
- **Synthetic data (Sparkov).** Fraud is generated from a few scripted profiles, so the metrics are optimistic:
  - {pct(pat.iloc[0].share_of_fraud)} of test fraud is night-time *and* $250+. The model catches {pct(pat.iloc[0].recall_at_threshold)} of it.
  - On the remaining {num(pat.iloc[1].frauds)} "off-pattern" frauds, PR-AUC is {pat.iloc[1].pr_auc_vs_all_legit:.3f} and recall {pct(pat.iloc[1].recall_at_threshold)}.
- **Memorised amount bands.** SHAP dependence shows the model learned the simulator's specific fraud-amount ranges (a non-monotonic amount effect). That would not transfer to real fraud, and it's the main reason not to quote these metrics as expected production performance.
- **Few customers.** There are only {c['h']['dataset']['Cards (customers)']} cards, and almost all of them have a fraud episode. Real portfolios have millions of cards and far rarer compromise.
- **Missing data sources.** There is no device, IP, channel, authorisation-result, customer-contact or merchant-risk data. Several false positives (e.g. large night-time online purchases by genuine customers) can't be resolved without them.
- **Tenure-like feature.** Card history length is a top-5 SHAP feature and grows over calendar time. In a portfolio with new cards, it will drift and may act as a proxy for account age.

## 2. Label delay
- **Why labels are late.** Fraud labels come from chargebacks and disputes, which take weeks. Monitoring assumes 60 days to maturity, so precision and recall for the latest two months are always provisional (marked in the monthly table and dashboard).
- **What to watch until then.** Alert volume, score and feature PSI, the missing-data rate, and **analyst dispositions** on worked alerts (a fast, partial precision signal).
- **Effect of unreported fraud on training.** If 20% or 40% of fraud episodes are never labelled, test PR-AUC moves from {noise.test_pr_auc_true_labels.iloc[0]:.3f} to {noise.test_pr_auc_true_labels.iloc[1]:.3f} and {noise.test_pr_auc_true_labels.iloc[2]:.3f}. Real under-reporting is not random (small amounts go unnoticed), so the damage would be concentrated in exactly those cases.

## 3. Feedback loops and selection bias
- **Labels follow reviews.** In production, most labels come from alerts that were reviewed or transactions that were disputed. A model retrained only on reviewed cases learns the old rules' blind spots, and never sees the fraud it doesn't alert on.
- **Mitigations:**
  - Use spare analyst capacity to review the whole **monitor-only band** (about {c['sampling'].per_day.iloc[0]:.0f} a day, around {c['sampling'].expected_frauds_found_per_month.iloc[0]:.0f} near-miss frauds a month on test). A small random sample far below the threshold would find about {c['sampling'].expected_frauds_found_per_month.iloc[1]:.3f} frauds a month.
  - Track customer-reported fraud the model scored low.
  - Keep the old rules in shadow as an independent detector.
  - Log the model version and threshold on every decision, so labels can be de-biased later.
- **Auto-hold changes the label itself.** A held transaction that the customer confirms is labelled legitimate, and a blocked one never becomes a chargeback. Record the customer confirmation outcome as the label for held transactions.

## 4. Concept drift and adversarial adaptation
- **Live period.** Recall stayed between {pct(h['monitoring_summary']['live_recall_range'][0])} and {pct(h['monitoring_summary']['live_recall_range'][1])}. Precision fell to {pct(h['monitoring_summary']['live_precision_range'][0])} in December as the fraud rate fell to {mon.iloc[-1].fraud_rate_pct:.2f}% (holiday volume). Triggers fired: {fired_txt}. That was seasonal velocity drift, investigated and not requiring retraining.
- **Simulated adversary.** Fraud moved to daytime: recall fell from {pct(st.iloc[0].recall)} to {pct(st.iloc[1].recall)}, while score PSI stayed at {st.iloc[1].psi_score:.3f}. **Population-level drift metrics can't detect a new fraud pattern**, because fraud is too rare to move them. Only labels, or proxies for them, can.
- **Response:**
  1. Deploy a targeted rule within hours, as a stop-gap.
  2. Retrain within weeks.
  3. Consider features that are harder to game (device, merchant risk, graph links between cards and merchants).

## 5. Threshold risk
- **Precision moves with the fraud rate.** A fixed threshold's precision changes with the base rate ({pct(opts.loc[('Cost-minimising', 'validation'), 'precision'])} on validation vs {pct(opts.loc[('Cost-minimising', 'test'), 'precision'])} on test).
- **The recall target drifted.** The 80% recall threshold, chosen on validation, delivered only {pct(opts.loc[(rec, 'test'), 'recall'], 1)} on test.
- **The cost curve is flat near its optimum**, so small threshold changes are noise. Recalibrate only if the optimum on the latest 3 matured months moves by more than ±25% in alerts/day, and log every change.
- **Assumption risk.** The cost assumptions ($5 per review, 0% recovery) drive the threshold. The ±50% sensitivity shows the optimum stays between {H['sensitivity_alerts_per_day_range'][0]:.0f} and {H['sensitivity_alerts_per_day_range'][1]:.0f} alerts/day, but Finance must own these numbers.
- **Scores are not calibrated probabilities** (SMOTE shifts them). Don't communicate scores as "% chance of fraud" without calibration.
- **Pipeline failures raise alerts.** In the outage simulation, alerts rose from {st.iloc[2].alerts_per_day:.0f} to {st.iloc[3].alerts_per_day:.0f}/day because the model treats missing history as risky. The daily missing-data trigger must page on-call.

## 6. Fairness
- **Protected characteristics excluded.** Age and gender are not model inputs. The production model is behavioural (amount, time, velocity, deviation from the customer's own norm).
- **What they would add.** An ablation adding them raises validation PR-AUC by {abl.diff_vs_production.iloc[1]:+.4f}, a measurable but small gain. We recommend keeping them out; the final decision belongs to model risk and compliance.
- **Error rates with uncertainty** (test period, at the recommended alert line, 95% card-level bootstrap intervals). The tolerances are proposed policy: false-alert rate ≤ 1.25x the lowest group, and recall within 3 points of the best group.

| Attribute | Group | False alerts per 10k genuine | Ratio vs lowest (CI) | Tolerance 1.25x | Recall | Gap vs best, pts (CI) | Tolerance 3 pts |
|---|---|---|---|---|---|---|---|
{chr(10).join(f"| {r.attribute} | {r.group} | {r.false_alerts_per_10k_legit:.1f} | {r.fpr_ratio_vs_lowest:.2f} ({r.ratio_ci}) | {r._6} | {pct(r.recall, 1)} | {r.recall_gap_vs_best_pts:.1f} ({r.gap_ci_pts}) | {r.recall_tolerance_3pts} |" for r in c['fair'].itertuples())}

- **Result.** Groups whose whole interval lies outside a tolerance fail it; the rest are inconclusive or pass. The gaps must come through correlated behaviour (spending patterns, time of day), because the model never sees age or gender. A proxy analysis and compliance-agreed tolerances are **go-live conditions** (finding MR-3).
- **No causal claims.** SHAP and importance describe what the model relies on, not what causes fraud.

## 6a. Business-case risk (operational simulation)
- **Where the money comes from.** The notebook-03 cost model assumed every alerted fraud is stopped. Notebook 06 simulates the workflow instead:
  - Auto-hold stops a transaction.
  - Post-authorisation review stops only the rest of a fraud burst, by blocking the card after the review delay.
  - The recommended set-up prevents {pct(c['ops'].loc[REC, 'fraud_prevented_pct'])} of test frauds. The rules under the same policy prevent {pct(c['ops'].loc[FAIR, 'fraud_prevented_pct'])}.
- **Most of the value is auto-hold**, which depends on the top band's precision (95% on validation). If real-world precision there is lower, customer friction rises and the case weakens. The shadow run must measure this first.
- **Simulation assumptions** (correct dispositions, a 12h average review delay, the block effect) are not covered by the saving's confidence interval. The sensitivity table in notebook 06 varies them one at a time.

## 7. Weak spots
- **Lowest-recall categories** (≥20 test frauds): {', '.join(f'{r.segment} ({pct(r.recall)})' for r in worst.itertuples())}. These are low-amount frauds that look like normal spending.
- **Top permutation features:** {', '.join(imp.label.head(3))}. With this much concentration, a change in how amount or category is recorded would have a large effect.

## 8. Monitoring plan

| Cadence | Checks | Trigger | Action |
|---|---|---|---|
| Daily | Alert volume; missing-input rate | > 100 alerts/day; > 2% missing | Raise threshold temporarily / page data engineering; fall back to rules if needed |
| Weekly | Score PSI, **score-tail PSI** (top 10% of scores) and key-feature PSI; analyst disposition and override rates; customer-reported fraud scored below threshold; frauds found in the monitor-only band | PSI > 0.20 | Check pipeline first, then behaviour change; review threshold |
| Monthly | Precision / recall on matured labels (60+ days); auto-hold precision and wrongly held customers; segment and fairness error rates vs tolerances; cost-optimal threshold on latest 3 matured months | Precision < 40%; recall < 90%; auto-hold precision < 90%; fairness tolerance failed; optimum moves > ±25% | Re-tune threshold; investigate new patterns; add stop-gap rule; fairness review |
| Quarterly | Retrain on rolling 12–15 months; champion/challenger on the same period; model-risk review | Challenger better on PR-AUC and cost | Promote with sign-off; keep previous model for rollback |

**Ownership:** fraud analytics (monitoring, thresholds), data science (retraining), fraud operations (dispositions, capacity), model risk and compliance (sign-off, fairness).

## 9. Governance (structure adapted from ClaudeFinanceLab's {cfl('mrg')})

| Item | Proposal |
|---|---|
| Model inventory record | Name: card-fraud transaction score · Type: supervised classifier (LightGBM) · Owner: fraud analytics · Users: fraud operations · Independent validator: model risk |
| Tier | **Tier 1 (high materiality).** It drives real-time holds on customer transactions and fraud losses |
| Independent validation | Before go-live, then annually and on any material change |
| Change management | Version the model, the feature SQL and the threshold together. Any threshold change is logged with date, reason and expected alerts/day. Retrains go through champion/challenger on the same period |
| Override tracking | Monthly analyst override rate (dispositions that contradict the band), by band and analyst. A rising rate signals score degradation or unclear reasons |
| Performance thresholds | As in the monitoring plan (section 8) |

## 10. Independent review findings log

{findings_md(c)}
"""


def main():
    c = load()
    (config.ROOT / "README.md").write_text(_expand(readme(c)))
    (R / "business_recommendation.md").write_text(business(c))
    (R / "model_risk_and_monitoring.md").write_text(model_risk(c))
    (R / "fraud_triage_workflow.md").write_text(workflow(c))
    print("Wrote README.md and reports/{business_recommendation, model_risk_and_monitoring, fraud_triage_workflow}.md")


if __name__ == "__main__":
    main()
