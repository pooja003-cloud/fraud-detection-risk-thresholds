-- =====================================================================
-- 03_eda_segments.sql
-- Segment-level fraud rates for exploratory analysis.
-- Each query is preceded by a name marker line and run by
-- src/data.run_sql_file(). All queries read the cleaned feature table.
-- =====================================================================

-- name: class_balance
SELECT is_fraud,
       COUNT(*)                                   AS transactions,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct,
       ROUND(SUM(amt), 0)                         AS amount_usd
FROM features
GROUP BY is_fraud
ORDER BY is_fraud;

-- name: by_hour
SELECT hour,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM features
GROUP BY hour
ORDER BY hour;

-- name: by_weekday
SELECT day_of_week,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM features
GROUP BY day_of_week
ORDER BY day_of_week;

-- name: by_category
SELECT category,
       COUNT(*)                                             AS transactions,
       SUM(is_fraud)                                        AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3)                      AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt END), 0)   AS fraud_amount_usd,
       ROUND(MEDIAN(CASE WHEN is_fraud = 1 THEN amt END), 2) AS median_fraud_amt
FROM features
GROUP BY category
ORDER BY fraud_rate_pct DESC;

-- name: by_state
SELECT state,
       COUNT(DISTINCT cc_num)          AS customers,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM features
GROUP BY state
ORDER BY transactions DESC;

-- name: by_age_band
SELECT CASE
         WHEN cust_age < 25 THEN '1: <25'
         WHEN cust_age < 35 THEN '2: 25-34'
         WHEN cust_age < 50 THEN '3: 35-49'
         WHEN cust_age < 65 THEN '4: 50-64'
         ELSE '5: 65+'
       END                             AS age_band,
       COUNT(DISTINCT cc_num)          AS customers,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM features
GROUP BY age_band
ORDER BY age_band;

-- name: by_gender
SELECT gender,
       COUNT(DISTINCT cc_num)          AS customers,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM features
GROUP BY gender;

-- name: by_amount_band
SELECT CASE
         WHEN amt < 10   THEN '1: <$10'
         WHEN amt < 50   THEN '2: $10-49'
         WHEN amt < 100  THEN '3: $50-99'
         WHEN amt < 250  THEN '4: $100-249'
         WHEN amt < 500  THEN '5: $250-499'
         WHEN amt < 1000 THEN '6: $500-999'
         ELSE '7: $1,000+'
       END                             AS amount_band,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM features
GROUP BY amount_band
ORDER BY amount_band;

-- name: monthly
SELECT DATE_TRUNC('month', ts)         AS month,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS frauds,
       ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM features
GROUP BY month
ORDER BY month;

-- name: frauds_per_card
-- How is fraud distributed across cards? (Sparkov simulates fraud as
-- bursts on a compromised card.)
SELECT COUNT(*)                                   AS cards_with_fraud,
       ROUND(AVG(n_fraud), 1)                     AS avg_frauds_per_compromised_card,
       MEDIAN(n_fraud)                            AS median_frauds_per_compromised_card,
       ROUND(AVG(span_hours), 1)                  AS avg_fraud_span_hours,
       MEDIAN(span_hours)                         AS median_fraud_span_hours
FROM (
    SELECT cc_num,
           SUM(is_fraud)                                               AS n_fraud,
           DATE_DIFF('minute', MIN(ts) FILTER (WHERE is_fraud = 1),
                               MAX(ts) FILTER (WHERE is_fraud = 1)) / 60.0 AS span_hours
    FROM features
    GROUP BY cc_num
    HAVING SUM(is_fraud) > 0
);
