# Sample alert case report

| | |
|---|---|
| **Case ID** | cc49b7e7ddc2… |
| **Referral source** | Automated model alert: fraud score **0.185**, band **Priority review** (band range 0.034–0.244) |
| **Alert date/time** | 2020-12-31 03:21:50 |
| **How this case was selected** | Fixed rule: the highest-scoring priority-review alert on the last test day that has one |
| **Card** | **** 2231 (first seen in data 2019-01-01) |
| **Assigned analyst (L1)** | ____________ |
| **L2 / senior analyst** | ____________ |

*Scope: facts are taken from the Sparkov synthetic dataset and the model output only. Service levels, band actions and the 60-day label-maturity rule are **workflow assumptions** (see `fraud_triage_workflow.md`), not facts from the data. The section headings follow the project brief; the referral block, baseline comparison, investigation timeline and disposition options follow public financial-crime investigation frameworks (see README, "How ClaudeFinanceLab was used").*

---

## 1. Observed facts (as at alert time)

**Transaction under review**

| Field | Value |
|---|---|
| Date/time | 2020-12-31 03:21:50 |
| Merchant | Langworth, Boehm and Gulgowski |
| Merchant category | shopping_net |
| Amount | $965.86 |
| Card's average transaction before this one (all history since 2019-01-01) | $64.67 over 1,458 transactions (14.9x) |
| Card transactions in the rolling prior 1h / 24h | 1 / 4 |
| Card spend in the rolling prior 24h | $333.63 |
| Time since card's previous transaction | 5 minutes |
| First transaction at this merchant for this card? | No. 1 prior transaction(s) at this merchant, most recently 2020-03-23 ($7.32) |
| Distinct merchants used by the card (all history) | 575 |
| Distance from cardholder's home location to merchant location | 98 km |
| Model alerts on this card in the prior 30 days | 0 |
| Confirmed fraud on this card with labels older than 60 days (known at alert time) | 8 transaction(s) labelled fraud, 2019-07-05 to 2019-07-06 (total $5,362.37) |

**Account baseline vs this activity**

| Measure | Card baseline (prior 90 days) | This alert | Comparison |
|---|---|---|---|
| Transaction amount | median $51.96, 95th percentile $179.96 | $965.86 | 18.6x the median |
| Share of transactions at night (22:00-04:00) | 28% | night | night activity is common for this card (>= 20%) |
| Share of transactions in 'shopping_net' | 3% (6 transactions) | shopping_net | category used before in this window |
| Largest single transaction | $1,158.51 | $965.86 | within prior range |
| Transactions per calendar day (incl. zero-activity days) | median 2, max 9 | 4 in the rolling prior 24h | within prior range |
| Spend per calendar day (incl. zero-activity days) | median $117.31, max $1,158.51 | $333.63 in the rolling prior 24h (+ this $965.86) | above prior daily maximum |

**Card activity in the 48 hours before the alert**

| Time | Category | Amount |
|---|---|---|
| 2020-12-29 04:53 | shopping_pos | $7.14 |
| 2020-12-29 07:40 | shopping_pos | $7.25 |
| 2020-12-29 09:32 | gas_transport | $65.59 |
| 2020-12-29 11:27 | gas_transport | $62.48 |
| 2020-12-29 13:25 | health_fitness | $51.54 |
| 2020-12-29 22:39 | home | $34.22 |
| 2020-12-30 03:56 | grocery_pos | $132.55 |
| 2020-12-30 04:52 | grocery_pos | $78.16 |
| 2020-12-31 01:45 | food_dining | $59.55 |
| 2020-12-31 03:16 | gas_transport | $63.37 |

## 2. Risk indicators (model explanation)

Factors that **raised** the score (SHAP contribution, log-odds):
- **Transaction amount: $965.86** (SHAP +6.07)
- **Amount / customer's historical average: 14.94** (SHAP +2.79)
- **Night-time (22:00-04:00): 1** (SHAP +1.99)
- **Amount / customer's average in this category: 8.46** (SHAP +1.04)

Factors that **lowered** the score:
- **Card spend in prior 1h ($): 63.37** (SHAP -1.13)
- **Merchant category: shopping_net** (SHAP -1.05)

These show what the model reacted to. They are indicators, not evidence that fraud occurred, and not causes.

**Facts consistent with the alert pattern**
- The amount is 14.9x the card's average transaction since 2019-01-01.
- The amount is above the card's 95th-percentile transaction in the prior 90 days ($179.96).
- Spend in the rolling prior 24h plus this transaction ($1,299.49) exceeds the card's largest calendar-day spend in the prior 90 days ($1,158.51).

**Facts that do not fit the alert pattern**
- The card's largest transaction in the prior 90 days ($1,158.51) exceeds this amount.
- 6 earlier transaction(s) on this card since 2019-01-01 were larger than this one.
- The card has 73 earlier 'shopping_net' transactions (largest $1,079.44).
- The card has used this merchant before (1 time(s); most recently 2020-03-23, $7.32).
- Night-time activity is common for this card (28% of prior-90-day transactions; this report treats 20% or more as common), although the model scores night-time as higher risk.

**Other relevant history (neutral)**
- The card has confirmed fraud in its history (8 transactions, 2019-07-05 to 2019-07-06). This is relevant context, not evidence about this transaction.
- The same card number was used 1,116 more times after that fraud. A compromised card would normally be reissued, so this is likely a quirk of the synthetic data: the history may describe the customer rather than one physical card.

## 3. Data gaps: investigation timeline

| Step | Source | Status | Finding |
|---|---|---|---|
| Transaction record reviewed | Transaction data | Completed | See section 1 |
| Card history (all history, 90-day baseline, prior 48h) reviewed | Transaction data | Completed | See section 1 |
| Model score and explanation reviewed | Fraud model (LightGBM, SHAP) | Completed | See section 2 |
| Prior model alerts on the card (30 days) | Model scores | Completed | No prior model alerts in the 30 days before this alert |
| Prior confirmed fraud on the card | Fraud labels older than 60 days | Completed | 8 transaction(s) labelled fraud, 2019-07-05 to 2019-07-06 (total $5,362.37) |
| Cardholder contact | Customer channel | **Not available** | Not in the dataset. Required before disposition |
| Device, IP address, channel | Digital/auth logs | **Not available** | Card-present vs card-not-present cannot be established from this record |
| Authorisation details (CVV / 3-D Secure / POS entry mode) | Authorisation system | **Not available** | |
| Merchant risk profile (chargeback rate, onboarding date) | Merchant data | **Not available** | |
| Disputes / fraud reported in the last 60 days | Case management | **Not available at alert time** | Labels not yet mature |

The data is synthetic (Sparkov simulator), so merchant names and locations are generated.

## 4. Escalation recommendation

**Route to the L1 priority-review queue (same-day SLA; workflow assumption). The transaction has already been authorised, so the purpose of review is to confirm or clear it quickly and, if fraud is confirmed, block the card before further transactions. Contact the cardholder through a registered channel before disposition.**

The score is driven mainly by transaction amount and amount / customer's historical average (section 2). Section 2 also lists facts that do not fit the alert pattern. The information in this record neither confirms nor rules out fraud.

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
5. If fraud is confirmed: block and reissue the card, and review other cards that transacted with Langworth, Boehm and Gulgowski in the same window.

| Sign-off | Role | Name | Date |
|---|---|---|---|
| Analyst | L1: prepares case and disposition | | |
| Reviewer | L2 / senior analyst: required for referrals and fraud confirmations | | |
| Approver | Supervisor: required for account action and compliance referral | | |
