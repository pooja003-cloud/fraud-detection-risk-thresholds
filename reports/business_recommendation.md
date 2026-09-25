# Recommendation: shadow-run a scored, triaged fraud alert queue

**To:** Head of Fraud Operations · **Basis:** back-test on 129 days of unseen transactions (Aug 25 to Dec 31, 2020), synthetic Sparkov data, independently reviewed

## Recommendation
Run the LightGBM fraud score for **three months in shadow** alongside the current rules, using four triage bands. Auto-hold applies only to the top band, which was at least 95% precise on validation. Switch the rules off only after the shadow run and the go-live conditions below.

## Fraud prevented vs the current rules (operational simulation, test period)
| | Rules, review only | Rules + auto-hold (same policy) | **Recommended** |
|---|---|---|---|
| Frauds prevented | 63% | 69% | **96%** |
| Fraud dollars lost | $239,089 | $188,775 | **$22,012** |
| Simulated total cost | $267,274 | $216,637 | **$27,093** |

- **Where the gain comes from:** mostly the auto-hold band. It stops a fraudulent transaction *before* authorisation and triggers a card block that stops the rest of the fraud burst. Alerts reviewed after authorisation can only stop what comes next.
- **Cost saving:** against the rules under the same policy, **$189,544 over 129 days** (95% interval $149,958 to $231,379). It stays positive in every sensitivity case tested.
- **Assumptions:** fraud loss = full amount, $5 per review, $1 per hold, $10 friction per wrongly held customer, 12h review delay. **Treat the figure as illustrative**: the data is synthetic and the rules are a proxy built for this project.

## Alert volume and workload
- **Daily load:** 6.6 analyst reviews and 1.4 auto-holds a day, against 43.7 reviews for the rules. On the busiest days (p95), all roles together need about **2.4 staff-hours a day**: L1 reviews, L2 and supervisor sign-off of fraud confirmations, and customer contact.
- **Use the spare capacity** (about 93 reviews a day) to review the whole **monitor-only band** (71 a day). It should surface about 6 near-miss frauds a month, which is the only practical way to measure what the model misses. A small random sample far below the threshold finds almost nothing.
- **Overnight holds:** **90% of auto-holds happen between 22:00 and 04:00**, so customer confirmation must be automated around the clock (SMS/app), with a time-out rule for unanswered holds.
- **Most reviews are genuine:** only about 4% of review-queue alerts are fraud, because auto-hold has already caught most of it. Set analysts' expectations accordingly.

## Risks
- **Synthetic data:** the model partly learned the simulator's patterns. Test PR-AUC is 0.970 here; expect much less on real traffic. That's why the shadow run comes first.
- **Fairness:** false-alert rates for 50-64, 65+ age groups exceed the proposed 1.25x tolerance, even though age is not a model input. A go-live condition.
- **New fraud patterns:** in a simulation where fraud moved to daytime, recall fell from 97% to 88% without any drift alarm. We need fast labels from customer reports and analyst dispositions.
- **Customer friction:** 61 genuine transactions were held over the test period, and blocked cards decline genuine spending until they're reissued.
- **Label delay:** precision and recall for the latest two months are always provisional.

## Next steps
1. **Months 0–3:** shadow-score live traffic. Compare catches, holds and false alerts with the rules every week.
2. **Model fixes before go-live:** replace SMOTE with class weights, drop the customer-identifying features (city population, card history length), and re-validate on our own data.
3. **Fairness:** agree tolerances with compliance and run a proxy analysis.
4. **Operations:** build the 24/7 automated confirmation channel and the time-out rule, and agree cost assumptions with Finance.
5. **Model-risk sign-off** (Tier 1), then go live with monthly threshold review and quarterly retraining.
