-- =====================================================================
-- 01_load_and_clean.sql
-- Load the two Kaggle Sparkov files into one table, fix types, remove
-- duplicates and handle missing values.
--
-- Why combine the files? Kaggle's train/test files are split by date
-- but provide no validation period. We rebuild our own
-- train / validation / test split by time (see src/split.py).
--
-- Parameters (substituted by src/data.py):  {raw_train}, {raw_test}
-- =====================================================================

CREATE OR REPLACE TABLE raw_transactions AS
SELECT *, 'kaggle_train' AS source_file
FROM read_csv('{raw_train}', header = true, all_varchar = true)
UNION ALL BY NAME
SELECT *, 'kaggle_test' AS source_file
FROM read_csv('{raw_test}', header = true, all_varchar = true);

-- ---------------------------------------------------------------------
-- Explicit type casting. Every column is read as VARCHAR first so that
-- nothing is silently mis-typed (e.g. card numbers as floats, zip codes
-- losing leading zeros). TRY_CAST returns NULL instead of failing, so
-- bad values surface in the missing-value audit below.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE typed_transactions AS
SELECT
    TRIM(trans_num)                                         AS trans_num,
    TRY_CAST(trans_date_trans_time AS TIMESTAMP)            AS ts,
    TRIM(cc_num)                                            AS cc_num,       -- ID, kept as text
    REGEXP_REPLACE(TRIM(merchant), '^fraud_', '')           AS merchant,     -- every merchant carries a 'fraud_' prefix in the simulator; it is NOT a label
    LOWER(TRIM(category))                                   AS category,
    TRY_CAST(amt AS DOUBLE)                                 AS amt,
    UPPER(TRIM(gender))                                     AS gender,
    TRIM(city)                                              AS city,
    UPPER(TRIM(state))                                      AS state,
    LPAD(TRIM(zip), 5, '0')                                 AS zip,
    TRY_CAST(lat AS DOUBLE)                                 AS cust_lat,
    TRY_CAST(long AS DOUBLE)                                AS cust_long,
    TRY_CAST(city_pop AS BIGINT)                            AS city_pop,
    TRIM(job)                                               AS job,
    TRY_CAST(dob AS DATE)                                   AS dob,
    TRY_CAST(merch_lat AS DOUBLE)                           AS merch_lat,
    TRY_CAST(merch_long AS DOUBLE)                          AS merch_long,
    TRY_CAST(is_fraud AS INTEGER)                           AS is_fraud,
    source_file
    -- Dropped on purpose: the unnamed row index, first/last name and street
    -- (synthetic PII with no legitimate predictive role), and unix_time
    -- (in this dataset it is offset from trans_date_trans_time by years and
    -- would be a confusing duplicate of the timestamp).
FROM raw_transactions;

-- ---------------------------------------------------------------------
-- Duplicate removal.
-- (a) exact duplicate transaction IDs  -> keep one
-- (b) same card, same timestamp, same merchant, same amount -> treat as
--     a double-posted record and keep the first by trans_num.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE dedup_transactions AS
SELECT * EXCLUDE (rn)
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY cc_num, ts, merchant, amt
               ORDER BY trans_num
           ) AS rn
    FROM (
        SELECT * EXCLUDE (rn_id)
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY trans_num ORDER BY ts) AS rn_id
            FROM typed_transactions
        )
        WHERE rn_id = 1
    )
)
WHERE rn = 1;

-- ---------------------------------------------------------------------
-- Missing-value handling.
-- Rows without a timestamp, card, amount or label cannot be scored or
-- evaluated and are dropped (and counted in the audit). Descriptive
-- fields are filled with explicit 'UNKNOWN' categories so the model can
-- learn from missingness instead of us silently imputing a real value.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE clean_transactions AS
SELECT
    trans_num, ts, cc_num, merchant,
    COALESCE(category, 'unknown')  AS category,
    amt,
    COALESCE(gender, 'U')          AS gender,
    COALESCE(city, 'UNKNOWN')      AS city,
    COALESCE(state, 'UNKNOWN')     AS state,
    zip, cust_lat, cust_long, city_pop, job, dob,
    merch_lat, merch_long, is_fraud, source_file
FROM dedup_transactions
WHERE ts IS NOT NULL
  AND cc_num IS NOT NULL
  AND amt IS NOT NULL AND amt >= 0
  AND is_fraud IN (0, 1);

-- ---------------------------------------------------------------------
-- Audit table: row counts at each cleaning step (read by notebook 01).
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE cleaning_audit AS
SELECT 'raw rows'                         AS step, COUNT(*) AS n_rows FROM raw_transactions
UNION ALL SELECT 'after type casting',                COUNT(*) FROM typed_transactions
UNION ALL SELECT 'after de-duplication',              COUNT(*) FROM dedup_transactions
UNION ALL SELECT 'after dropping unusable rows',      COUNT(*) FROM clean_transactions;

CREATE OR REPLACE TABLE missing_audit AS
UNPIVOT (
    SELECT
        COUNT(*) FILTER (WHERE ts IS NULL)         AS ts,
        COUNT(*) FILTER (WHERE cc_num IS NULL)     AS cc_num,
        COUNT(*) FILTER (WHERE merchant IS NULL)   AS merchant,
        COUNT(*) FILTER (WHERE category IS NULL)   AS category,
        COUNT(*) FILTER (WHERE amt IS NULL)        AS amt,
        COUNT(*) FILTER (WHERE gender IS NULL)     AS gender,
        COUNT(*) FILTER (WHERE state IS NULL)      AS state,
        COUNT(*) FILTER (WHERE cust_lat IS NULL)   AS cust_lat,
        COUNT(*) FILTER (WHERE city_pop IS NULL)   AS city_pop,
        COUNT(*) FILTER (WHERE dob IS NULL)        AS dob,
        COUNT(*) FILTER (WHERE merch_lat IS NULL)  AS merch_lat,
        COUNT(*) FILTER (WHERE is_fraud IS NULL)   AS is_fraud
    FROM typed_transactions
) ON COLUMNS(*) INTO NAME column_name VALUE n_missing;
