# Model risk and monitoring

Scope: the LightGBM fraud score (SMOTE 1:10, `num_leaves=15`), used with a cost-minimising alert threshold and four triage bands. Evidence comes from notebooks 02–05. Every number here is generated from their saved outputs.

## 1. Data limitations
- **Synthetic data (Sparkov).** Fraud is generated from a few scripted profiles, so the metrics are optimistic:
  - 63% of test fraud is night-time *and* $250+. The model catches 99% of it.
  - On the remaining 503 "off-pattern" frauds, PR-AUC is 0.890 and recall 94%.
- **Memorised amount bands.** SHAP dependence shows the model learned the simulator's specific fraud-amount ranges (a non-monotonic amount effect). That would not transfer to real fraud, and it's the main reason not to quote these metrics as expected production performance.
- **Few customers.** There are only 999 cards, and almost all of them have a fraud episode. Real portfolios have millions of cards and far rarer compromise.
- **Missing data sources.** There is no device, IP, channel, authorisation-result, customer-contact or merchant-risk data. Several false positives (e.g. large night-time online purchases by genuine customers) can't be resolved without them.
- **Tenure-like feature.** Card history length is a top-5 SHAP feature and grows over calendar time. In a portfolio with new cards, it will drift and may act as a proxy for account age.

## 2. Label delay
- **Why labels are late.** Fraud labels come from chargebacks and disputes, which take weeks. Monitoring assumes 60 days to maturity, so precision and recall for the latest two months are always provisional (marked in the monthly table and dashboard).
- **What to watch until then.** Alert volume, score and feature PSI, the missing-data rate, and **analyst dispositions** on worked alerts (a fast, partial precision signal).
- **Effect of unreported fraud on training.** If 20% or 40% of fraud episodes are never labelled, test PR-AUC moves from 0.965 to 0.950 and 0.919. Real under-reporting is not random (small amounts go unnoticed), so the damage would be concentrated in exactly those cases.

## 3. Feedback loops and selection bias
- **Labels follow reviews.** In production, most labels come from alerts that were reviewed or transactions that were disputed. A model retrained only on reviewed cases learns the old rules' blind spots, and never sees the fraud it doesn't alert on.
- **Mitigations:**
  - Use spare analyst capacity to review the whole **monitor-only band** (about 71 a day, around 6 near-miss frauds a month on test). A small random sample far below the threshold would find about 0.001 frauds a month.
  - Track customer-reported fraud the model scored low.
  - Keep the old rules in shadow as an independent detector.
  - Log the model version and threshold on every decision, so labels can be de-biased later.
- **Auto-hold changes the label itself.** A held transaction that the customer confirms is labelled legitimate, and a blocked one never becomes a chargeback. Record the customer confirmation outcome as the label for held transactions.

## 4. Concept drift and adversarial adaptation
- **Live period.** Recall stayed between 97% and 99%. Precision fell to 45% in December as the fraud rate fell to 0.18% (holiday volume). Triggers fired: 2020-12: Any feature PSI > 0.2 (n_txn_prior_24h, amt_sum_prior_24h). That was seasonal velocity drift, investigated and not requiring retraining.
- **Simulated adversary.** Fraud moved to daytime: recall fell from 97% to 88%, while score PSI stayed at 0.054. **Population-level drift metrics can't detect a new fraud pattern**, because fraud is too rare to move them. Only labels, or proxies for them, can.
- **Response:**
  1. Deploy a targeted rule within hours, as a stop-gap.
  2. Retrain within weeks.
  3. Consider features that are harder to game (device, merchant risk, graph links between cards and merchants).

## 5. Threshold risk
- **Precision moves with the fraud rate.** A fixed threshold's precision changes with the base rate (63% on validation vs 59% on test).
- **The recall target drifted.** The 80% recall threshold, chosen on validation, delivered only 78.6% on test.
- **The cost curve is flat near its optimum**, so small threshold changes are noise. Recalibrate only if the optimum on the latest 3 matured months moves by more than ±25% in alerts/day, and log every change.
- **Assumption risk.** The cost assumptions ($5 per review, 0% recovery) drive the threshold. The ±50% sensitivity shows the optimum stays between 17 and 28 alerts/day, but Finance must own these numbers.
- **Scores are not calibrated probabilities** (SMOTE shifts them). Don't communicate scores as "% chance of fraud" without calibration.
- **Pipeline failures raise alerts.** In the outage simulation, alerts rose from 19 to 28/day because the model treats missing history as risky. The daily missing-data trigger must page on-call.

## 6. Fairness
- **Protected characteristics excluded.** Age and gender are not model inputs. The production model is behavioural (amount, time, velocity, deviation from the customer's own norm).
- **What they would add.** An ablation adding them raises validation PR-AUC by +0.0079, a measurable but small gain. We recommend keeping them out; the final decision belongs to model risk and compliance.
- **Error rates with uncertainty** (test period, at the recommended alert line, 95% card-level bootstrap intervals). The tolerances are proposed policy: false-alert rate ≤ 1.25x the lowest group, and recall within 3 points of the best group.

| Attribute | Group | False alerts per 10k genuine | Ratio vs lowest (CI) | Tolerance 1.25x | Recall | Gap vs best, pts (CI) | Tolerance 3 pts |
|---|---|---|---|---|---|---|---|
| gender | F | 26.7 | 1.21 (1.04-1.47) | inconclusive | 96.1% | 2.7 (0.8-4.5) | inconclusive |
| gender | M | 22.1 | 1.00 (1.00-1.00) | pass | 98.8% | 0.0 (0.0-0.0) | pass |
| age_band | 25-34 | 22.9 | 1.17 (1.00-1.56) | inconclusive | 95.8% | 4.2 (1.5-7.4) | inconclusive |
| age_band | 35-49 | 19.6 | 1.00 (1.00-1.28) | inconclusive | 96.4% | 3.6 (2.1-5.5) | inconclusive |
| age_band | 50-64 | 29.3 | 1.49 (1.26-2.02) | fail | 99.2% | 0.8 (0.0-1.7) | pass |
| age_band | 65+ | 32.7 | 1.66 (1.32-2.25) | fail | 96.7% | 3.3 (0.6-7.0) | inconclusive |
| age_band | <25 | 21.8 | 1.11 (1.00-1.64) | inconclusive | 100.0% | 0.0 (0.0-0.0) | pass |

- **Result.** Groups whose whole interval lies outside a tolerance fail it; the rest are inconclusive or pass. The gaps must come through correlated behaviour (spending patterns, time of day), because the model never sees age or gender. A proxy analysis and compliance-agreed tolerances are **go-live conditions** (finding MR-3).
- **No causal claims.** SHAP and importance describe what the model relies on, not what causes fraud.

## 6a. Business-case risk (operational simulation)
- **Where the money comes from.** The notebook-03 cost model assumed every alerted fraud is stopped. Notebook 06 simulates the workflow instead:
  - Auto-hold stops a transaction.
  - Post-authorisation review stops only the rest of a fraud burst, by blocking the card after the review delay.
  - The recommended set-up prevents 96% of test frauds. The rules under the same policy prevent 69%.
- **Most of the value is auto-hold**, which depends on the top band's precision (95% on validation). If real-world precision there is lower, customer friction rises and the case weakens. The shadow run must measure this first.
- **Simulation assumptions** (correct dispositions, a 12h average review delay, the block effect) are not covered by the saving's confidence interval. The sensitivity table in notebook 06 varies them one at a time.

## 7. Weak spots
- **Lowest-recall categories** (≥20 test frauds): personal_care (81%), travel (83%), kids_pets (89%). These are low-amount frauds that look like normal spending.
- **Top permutation features:** Transaction amount (log), Merchant category, Card spend in prior 24h ($). With this much concentration, a change in how amount or category is recorded would have a large effect.

## 8. Monitoring plan

| Cadence | Checks | Trigger | Action |
|---|---|---|---|
| Daily | Alert volume; missing-input rate | > 100 alerts/day; > 2% missing | Raise threshold temporarily / page data engineering; fall back to rules if needed |
| Weekly | Score PSI, **score-tail PSI** (top 10% of scores) and key-feature PSI; analyst disposition and override rates; customer-reported fraud scored below threshold; frauds found in the monitor-only band | PSI > 0.20 | Check pipeline first, then behaviour change; review threshold |
| Monthly | Precision / recall on matured labels (60+ days); auto-hold precision and wrongly held customers; segment and fairness error rates vs tolerances; cost-optimal threshold on latest 3 matured months | Precision < 40%; recall < 90%; auto-hold precision < 90%; fairness tolerance failed; optimum moves > ±25% | Re-tune threshold; investigate new patterns; add stop-gap rule; fairness review |
| Quarterly | Retrain on rolling 12–15 months; champion/challenger on the same period; model-risk review | Challenger better on PR-AUC and cost | Promote with sign-off; keep previous model for rollback |

**Ownership:** fraud analytics (monitoring, thresholds), data science (retraining), fraud operations (dispositions, capacity), model risk and compliance (sign-off, fairness).

## 9. Governance (structure adapted from ClaudeFinanceLab's [Model Risk Governance Validator](https://claudefinancelab.com/skill/model-risk-governance-validator/))

| Item | Proposal |
|---|---|
| Model inventory record | Name: card-fraud transaction score · Type: supervised classifier (LightGBM) · Owner: fraud analytics · Users: fraud operations · Independent validator: model risk |
| Tier | **Tier 1 (high materiality).** It drives real-time holds on customer transactions and fraud losses |
| Independent validation | Before go-live, then annually and on any material change |
| Change management | Version the model, the feature SQL and the threshold together. Any threshold change is logged with date, reason and expected alerts/day. Retrains go through champion/challenger on the same period |
| Override tracking | Monthly analyst override rate (dispositions that contradict the band), by band and analyst. A rising rate signals score degradation or unclear reasons |
| Performance thresholds | As in the monitoring plan (section 8) |

## 10. Independent review findings log

**Overall opinion (independent model-risk reviewer): Approved with conditions, for shadow / challenger use only.** Not approved for live auto-hold decisions until the High findings marked *Open (condition)* are closed and the model is re-validated on the institution's own data after a 3-month shadow run. **Model risk rating: High (Tier 1).** The reviewer found no temporal leakage.

| ID | Reviewer | Severity | Finding | Status | Resolution / where documented |
|---|---|---|---|---|---|
| MR-1 | Model risk + Ops | High | Cost model assumed every alerted fraud is stopped; only auto-hold acts before authorisation | Fixed | Operational simulation (notebook 06) is the headline: saving $189,544 vs rules under the same policy (95% CI $149,958-$231,379); notebook-03 figure kept as an idealised upper bound |
| MR-2 | Model risk + Ops | High | New fraud patterns are invisible to PSI; a 50-a-month below-threshold sample finds almost nothing | Fixed (design) | Review the whole monitor-only band with spare capacity (~6.3 near-miss frauds/month); add score-tail PSI; fast labels from customer reports and dispositions |
| MR-3 | Model risk | High | Error-rate gaps by age and gender, with no tolerance defined | Open (condition) | Card-bootstrap intervals and proposed tolerances in notebook 06; worst age band 65+: 1.66x the lowest false-alert rate (CI 1.32-2.25). Compliance sign-off and a proxy analysis are needed before go-live |
| MR-4 | Model risk | Medium | SMOTE creates impossible synthetic records (fractional categories and hours) and was compared under a different encoding | Open | Recommended: switch to class weights or no weighting (validation PR-AUC within 0.005, notebook 02) and re-validate before production |
| MR-5 | Model risk | Medium | Validation set reused for early stopping, tuning, imbalance choice and all thresholds | Open | Test set untouched until the end; validation results are optimistic. Use rolling time-series CV plus a separate threshold window in production |
| MR-6 | Model risk | Medium | City population works as a customer identifier; most test cards also appear in train | Open | Drop or coarsen the feature and validate on unseen cards before production |
| MR-7 | Model risk | Medium | Card history length grows with calendar time (tenure / time proxy) | Open | Documented in the model-risk file; cap or drop before production |
| MR-8 | Model risk | Medium | Standard PSI cannot see changes among alerting scores; December 'no retrain' used immature labels | Fixed | Score-tail PSI and trigger added (notebook 05); December conclusion marked provisional |
| MR-9 | Model risk + Ops | Medium | Benchmark rules were built for this project; annualised saving extrapolated from synthetic data | Fixed (wording) | Rules described as a proxy for the current process; saving reported for the test period with a confidence interval; the annualised figure is removed |
| OPS-1 | Ops | High | Capacity check ignored L2, supervisor and customer-contact work, and peak days | Fixed | p95-day staffing by role in notebook 06: about 2.4 staff-hours/day across roles |
| OPS-2 | Ops | High | Auto-hold requires immediate customer contact, but most holds are overnight | Fixed (design) | 90% of holds are at 22:00-04:00; workflow now specifies automated 24/7 confirmation with a time-out rule |
| OPS-3 | Ops | Medium | Headline precision is not what analysts see in the review queues | Fixed | Review-queue precision reported separately (4% in the simulation) |
| OPS-4 | Ops | Medium | No friction cost for wrongly held customers; auto-hold was charged as an analyst review | Fixed | Simulation charges $1 per hold + $10 per wrongly held customer (61 on test) |
| OPS-5 | Ops | Low | The recall-target option missed its 80% target on test | Documented | Thresholds need a safety margin and must be recalibrated on matured recent data |
| INV-1 | Investigator | High | Case report recommended a hold for a priority-review alert | Fixed | Recommended action now follows the band's workflow (route to L1 priority review; contact before disposition) |
| INV-2 | Investigator | Medium | Case selection described inaccurately; the card's earlier confirmed fraud was not reported; no contrary evidence | Fixed | Selection rule stated; confirmed fraud older than 60 days reported; 'facts that do not fit' section added |
| INV-3 | Investigator | Low | Time windows unlabelled; dispositions and roles did not match the workflow; hindsight label in the same file | Fixed | Windows labelled; dispositions and roles aligned; hindsight outcome moved to a separate file |
