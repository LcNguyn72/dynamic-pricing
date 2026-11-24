USE master;
GO

IF DB_ID(N'DynamicPricingM5') IS NULL
    CREATE DATABASE DynamicPricingM5;
GO

USE DynamicPricingM5;
GO

SET XACT_ABORT ON;
BEGIN TRANSACTION;

IF OBJECT_ID(N'dbo.M5Stores', N'U') IS NULL
    CREATE TABLE dbo.M5Stores (
        store_id varchar(16) NOT NULL PRIMARY KEY,
        state_id varchar(8) NOT NULL
    );

IF OBJECT_ID(N'dbo.M5Products', N'U') IS NULL
    CREATE TABLE dbo.M5Products (
        item_id varchar(32) NOT NULL PRIMARY KEY,
        department_id varchar(20) NOT NULL,
        category_id varchar(20) NOT NULL
    );

IF OBJECT_ID(N'dbo.M5StoreProducts', N'U') IS NULL
    CREATE TABLE dbo.M5StoreProducts (
        store_id varchar(16) NOT NULL REFERENCES dbo.M5Stores (store_id),
        item_id varchar(32) NOT NULL REFERENCES dbo.M5Products (item_id),
        PRIMARY KEY (store_id, item_id)
    );

IF OBJECT_ID(N'dbo.M5WeeklySales', N'U') IS NULL
    CREATE TABLE dbo.M5WeeklySales (
        store_id varchar(16) NOT NULL,
        item_id varchar(32) NOT NULL,
        week_start date NOT NULL CHECK (DATEDIFF(DAY, '19000101', week_start) % 7 = 0),
        quantity int NOT NULL CHECK (quantity >= 0),
        price float(53) NULL CHECK (price > 0),
        revenue decimal(19, 4) NOT NULL CHECK (revenue >= 0),
        observed_sales bit NOT NULL,
        PRIMARY KEY (store_id, item_id, week_start),
        FOREIGN KEY (store_id, item_id) REFERENCES dbo.M5StoreProducts (store_id, item_id),
        CHECK (observed_sales = 0 OR price IS NOT NULL)
    );

IF OBJECT_ID(N'dbo.M5Feedback', N'U') IS NULL
    CREATE TABLE dbo.M5Feedback (
        store_id varchar(16) NOT NULL,
        item_id varchar(32) NOT NULL,
        week_start date NOT NULL CHECK (DATEDIFF(DAY, '19000101', week_start) % 7 = 0),
        applied_price float(53) NOT NULL CHECK (applied_price > 0),
        observed_qty int NOT NULL CHECK (observed_qty >= 0),
        source varchar(12) NOT NULL CHECK (source IN ('m5_history', 'actual')),
        recorded_at datetime2(0) NOT NULL DEFAULT SYSUTCDATETIME(),
        PRIMARY KEY (store_id, item_id, week_start),
        FOREIGN KEY (store_id, item_id) REFERENCES dbo.M5StoreProducts (store_id, item_id)
    );

IF OBJECT_ID(N'dbo.M5PanelImports', N'U') IS NULL
    CREATE TABLE dbo.M5PanelImports (
        store_id varchar(16) NOT NULL PRIMARY KEY REFERENCES dbo.M5Stores (store_id),
        source_sha256 char(64) NOT NULL,
        product_count int NOT NULL CHECK (product_count > 0),
        row_count int NOT NULL CHECK (row_count > 0),
        first_week date NOT NULL,
        last_week date NOT NULL,
        imported_at datetime2(0) NOT NULL DEFAULT SYSUTCDATETIME(),
        CHECK (last_week >= first_week)
    );

COMMIT TRANSACTION;
GO
