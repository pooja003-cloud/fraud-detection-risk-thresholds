-- =====================================================================
-- 02_behavioural_features.sql
-- Behavioural features built with window functions that ONLY look
-- backward in time.
--
-- Leakage rules enforced here:
--   * Every window frame ends at "1 PRECEDING" on an epoch-second
--     ordering key, so the current transaction, any transaction at the
--     same second and anything later are all excluded.
--   * No feature uses the fraud label of earlier transactions. In
--     production, labels arrive weeks later (chargebacks), so they are
--     not available at scoring time.
--   * No customer-level statistic is computed over the whole dataset
--     (that would leak future behaviour into past rows).
-- =====================================================================

CREATE OR REPLACE TABLE features AS
WITH base AS (
    SELECT
        *,
        CAST(EPOCH(ts) AS BIGINT) AS t     -- ordering key in seconds
    FROM clean_transactions
),
windows AS (
    SELECT
        b.*,

        -- ---- Velocity: activity in the prior 1 hour / 24 hours / 7 days
        COUNT(*)  OVER w1h   AS n_txn_prior_1h,
        COALESCE(SUM(amt) OVER w1h, 0)  AS amt_sum_prior_1h,
        COUNT(*)  OVER w24h  AS n_txn_prior_24h,
        COALESCE(SUM(amt) OVER w24h, 0) AS amt_sum_prior_24h,
        COUNT(*)  OVER w7d   AS n_txn_prior_7d,
        AVG(amt)  OVER w7d   AS avg_amt_prior_7d,

        -- ---- Customer history (all time up to, not including, now)
        COUNT(*)            OVER wall AS cust_n_prior_txn,
        AVG(amt)            OVER wall AS cust_avg_amt_prior,
        STDDEV_SAMP(amt)    OVER wall AS cust_std_amt_prior,
        MAX(t)              OVER wall AS cust_last_t,

        -- ---- Customer x merchant and customer x category history
        COUNT(*) OVER (PARTITION BY cc_num, merchant ORDER BY t
                       RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cust_merchant_n_prior,
        COUNT(*) OVER (PARTITION BY cc_num, category ORDER BY t
                       RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cust_category_n_prior,
        AVG(amt) OVER (PARTITION BY cc_num, category ORDER BY t
                       RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cust_category_avg_amt_prior
    FROM base b
    WINDOW
        w1h  AS (PARTITION BY cc_num ORDER BY t RANGE BETWEEN 3600   PRECEDING AND 1 PRECEDING),
        w24h AS (PARTITION BY cc_num ORDER BY t RANGE BETWEEN 86400  PRECEDING AND 1 PRECEDING),
        w7d  AS (PARTITION BY cc_num ORDER BY t RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
        wall AS (PARTITION BY cc_num ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    -- identifiers and label (not model features)
    trans_num, ts, cc_num, merchant, is_fraud, source_file,

    -- raw transaction attributes
    amt,
    LN(1 + amt)                                   AS log_amt,
    category,
    gender,
    state,
    city_pop,
    LN(1 + city_pop)                              AS log_city_pop,

    -- time-of-day / calendar
    HOUR(ts)                                      AS hour,
    ISODOW(ts)                                    AS day_of_week,     -- 1 = Monday
    CASE WHEN HOUR(ts) >= 22 OR HOUR(ts) < 4 THEN 1 ELSE 0 END AS is_night,

    -- customer age at transaction time (years)
    DATE_DIFF('day', dob, CAST(ts AS DATE)) / 365.25 AS cust_age,

    -- customer home to merchant distance (haversine, km)
    2 * 6371 * ASIN(SQRT(
        POWER(SIN(RADIANS(merch_lat - cust_lat) / 2), 2) +
        COS(RADIANS(cust_lat)) * COS(RADIANS(merch_lat)) *
        POWER(SIN(RADIANS(merch_long - cust_long) / 2), 2)
    ))                                            AS dist_home_merchant_km,

    -- velocity
    n_txn_prior_1h,
    amt_sum_prior_1h,
    n_txn_prior_24h,
    amt_sum_prior_24h,
    n_txn_prior_7d,

    -- amount versus the customer's own norm
    cust_n_prior_txn,
    cust_avg_amt_prior,
    amt / NULLIF(cust_avg_amt_prior, 0)                           AS amt_to_cust_avg,
    (amt - cust_avg_amt_prior) / NULLIF(cust_std_amt_prior, 0)    AS amt_zscore_cust,
    amt / NULLIF(avg_amt_prior_7d, 0)                             AS amt_to_cust_avg_7d,
    amt / NULLIF(cust_category_avg_amt_prior, 0)                  AS amt_to_cust_category_avg,

    -- recency
    (t - cust_last_t) / 3600.0                                    AS hours_since_last_txn,

    -- novelty
    CASE WHEN cust_merchant_n_prior = 0 THEN 1 ELSE 0 END         AS is_first_merchant_for_cust,
    CASE WHEN cust_category_n_prior = 0 THEN 1 ELSE 0 END         AS is_first_category_for_cust

    -- NULLs remain where there is no history (first transaction of a card,
    -- first transaction in a category, zero variance). They are handled in
    -- src/features.py with training-fitted imputation + missing flags,
    -- because "no history" is itself informative and we do not want to
    -- invent a value in SQL.
FROM windows
ORDER BY ts, trans_num;
