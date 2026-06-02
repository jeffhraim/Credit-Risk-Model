-- 01_create_database.sql
-- Run once against the SQL Server instance (master DB).
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'CreditRisk')
BEGIN
    CREATE DATABASE CreditRisk;
END
GO

USE CreditRisk;
GO

-- A dedicated schema keeps raw/curated objects organized.
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'raw')
    EXEC('CREATE SCHEMA raw');
GO
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'curated')
    EXEC('CREATE SCHEMA curated');
GO
