-- 02_create_tables.sql
USE CreditRisk;
GO

-- Raw landing table: mirrors the CSV 1:1, nullable where the source has gaps.
IF OBJECT_ID('raw.loan_applications', 'U') IS NOT NULL
    DROP TABLE raw.loan_applications;
GO

CREATE TABLE raw.loan_applications (
    member_id              BIGINT          NOT NULL PRIMARY KEY,
    issue_date             DATE            NULL,
    loan_amount            DECIMAL(12,2)   NULL,
    term_months            INT             NULL,
    int_rate               DECIMAL(6,2)    NULL,
    grade                  CHAR(1)         NULL,
    loan_purpose           VARCHAR(40)     NULL,
    home_ownership         VARCHAR(20)     NULL,
    annual_income          DECIMAL(14,2)   NULL,
    verification_status    VARCHAR(20)     NULL,
    age                    INT             NULL,
    employment_length_yrs  DECIMAL(5,1)    NULL,
    dti                    DECIMAL(6,2)    NULL,
    credit_score           INT             NULL,
    revol_util             DECIMAL(6,1)    NULL,
    open_accounts          INT             NULL,
    total_accounts         INT             NULL,
    delinq_2yrs            INT             NULL,
    inquiries_6m           INT             NULL,
    pub_rec                INT             NULL,
    [default]              INT             NULL,
    loaded_at              DATETIME2       NOT NULL DEFAULT SYSUTCDATETIME()
);
GO

CREATE INDEX ix_loan_grade  ON raw.loan_applications (grade);
CREATE INDEX ix_loan_default ON raw.loan_applications ([default]);
GO

-- Audit table for model predictions written back from the API.
IF OBJECT_ID('curated.prediction_log', 'U') IS NOT NULL
    DROP TABLE curated.prediction_log;
GO

CREATE TABLE curated.prediction_log (
    prediction_id   BIGINT IDENTITY(1,1) PRIMARY KEY,
    member_id       BIGINT          NULL,
    model_version   VARCHAR(50)     NULL,
    pd_score        DECIMAL(9,6)    NULL,   -- probability of default
    decision        VARCHAR(10)     NULL,   -- APPROVE / DECLINE / REVIEW
    request_json    NVARCHAR(MAX)   NULL,
    scored_at       DATETIME2       NOT NULL DEFAULT SYSUTCDATETIME()
);
GO
