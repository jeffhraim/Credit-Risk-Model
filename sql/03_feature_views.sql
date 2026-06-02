-- 03_feature_views.sql
-- Feature engineering pushed down to SQL so the model always trains on the
-- same definitions the application layer serves at inference time.
USE CreditRisk;
GO

IF OBJECT_ID('curated.vw_model_features', 'V') IS NOT NULL
    DROP VIEW curated.vw_model_features;
GO

CREATE VIEW curated.vw_model_features AS
SELECT
    member_id,
    -- raw numeric features
    loan_amount,
    term_months,
    int_rate,
    annual_income,
    age,
    employment_length_yrs,
    dti,
    credit_score,
    revol_util,
    open_accounts,
    total_accounts,
    delinq_2yrs,
    inquiries_6m,
    pub_rec,
    -- categorical features
    grade,
    loan_purpose,
    home_ownership,
    verification_status,
    -- engineered ratios (computed in-database for parity with serving)
    CAST(loan_amount AS FLOAT) / NULLIF(annual_income, 0)        AS loan_to_income,
    CAST(loan_amount AS FLOAT) / NULLIF(term_months, 0)          AS installment_proxy,
    CASE WHEN credit_score >= 740 THEN 'prime'
         WHEN credit_score >= 670 THEN 'near_prime'
         WHEN credit_score >= 580 THEN 'subprime'
         ELSE 'deep_subprime' END                                AS score_band,
    open_accounts * 1.0 / NULLIF(total_accounts, 0)              AS open_ratio,
    -- label
    [default] AS target
FROM raw.loan_applications;
GO

-- Deterministic 80/20 split by hashing the key, so train/test are reproducible
-- and stable across runs (no leakage from random re-splits).
IF OBJECT_ID('curated.vw_train', 'V') IS NOT NULL DROP VIEW curated.vw_train;
GO
CREATE VIEW curated.vw_train AS
SELECT * FROM curated.vw_model_features
WHERE ABS(CAST(HASHBYTES('MD5', CAST(member_id AS VARCHAR(20))) AS BIGINT) % 100) >= 20;
GO

IF OBJECT_ID('curated.vw_test', 'V') IS NOT NULL DROP VIEW curated.vw_test;
GO
CREATE VIEW curated.vw_test AS
SELECT * FROM curated.vw_model_features
WHERE ABS(CAST(HASHBYTES('MD5', CAST(member_id AS VARCHAR(20))) AS BIGINT) % 100) < 20;
GO
