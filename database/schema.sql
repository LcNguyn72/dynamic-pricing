IF DB_ID(N'DynamicPricingDemo') IS NULL
    CREATE DATABASE [DynamicPricingDemo];
GO
USE [DynamicPricingDemo];
GO

IF OBJECT_ID(N'dbo.Products', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Products (
        sku NVARCHAR(20) NOT NULL CONSTRAINT PK_Products PRIMARY KEY,
        description NVARCHAR(255) NOT NULL,
        tier NVARCHAR(20) NOT NULL,
        weeks SMALLINT NOT NULL,
        n_levels INT NOT NULL,
        price_median DECIMAL(18, 4) NOT NULL,
        price_min DECIMAL(18, 4) NOT NULL,
        price_max DECIMAL(18, 4) NOT NULL,
        qty_mean DECIMAL(18, 4) NOT NULL,
        selection_score DECIMAL(18, 2) NOT NULL,
        is_selected BIT NOT NULL CONSTRAINT DF_Products_Selected DEFAULT (1),
        CONSTRAINT CK_Products_Prices CHECK (price_min > 0 AND price_min <= price_median AND price_median <= price_max),
        CONSTRAINT CK_Products_Weeks CHECK (weeks > 0)
    );
END;
GO

IF OBJECT_ID(N'dbo.WeeklySales', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.WeeklySales (
        sku NVARCHAR(20) NOT NULL,
        week_start DATE NOT NULL,
        qty BIGINT NOT NULL,
        price DECIMAL(18, 4) NOT NULL,
        qty_all BIGINT NOT NULL,
        CONSTRAINT PK_WeeklySales PRIMARY KEY CLUSTERED (sku, week_start),
        CONSTRAINT FK_WeeklySales_Product FOREIGN KEY (sku) REFERENCES dbo.Products(sku),
        CONSTRAINT CK_WeeklySales_Values CHECK (qty > 0 AND qty_all >= qty AND price > 0)
    );
    CREATE INDEX IX_WeeklySales_Week ON dbo.WeeklySales(week_start) INCLUDE (sku, qty, price);
END;
GO

IF OBJECT_ID(N'dbo.ModelRuns', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ModelRuns (
        model_version VARCHAR(40) NOT NULL CONSTRAINT PK_ModelRuns PRIMARY KEY,
        trained_at DATETIME2(0) NULL,
        elasticity_method NVARCHAR(30) NOT NULL,
        global_elasticity FLOAT NOT NULL,
        tau2 FLOAT NOT NULL,
        sigma_extra FLOAT NOT NULL,
        test_r2 FLOAT NULL,
        test_mae_qty FLOAT NULL,
        baseline_r2 FLOAT NULL,
        baseline_mae_qty FLOAT NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.ProductModelStats', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ProductModelStats (
        model_version VARCHAR(40) NOT NULL,
        sku NVARCHAR(20) NOT NULL,
        p_ref DECIMAL(18, 4) NOT NULL,
        n_train_weeks INT NOT NULL,
        sku_base FLOAT NOT NULL,
        sku_lp_med FLOAT NOT NULL,
        sku_lq_std FLOAT NOT NULL,
        sku_lp_std FLOAT NOT NULL,
        sku_q4_effect FLOAT NOT NULL,
        eps_raw FLOAT NOT NULL,
        se_raw FLOAT NOT NULL,
        prior_mean FLOAT NOT NULL,
        weight_own FLOAT NOT NULL,
        eps_mean FLOAT NOT NULL,
        eps_sd FLOAT NOT NULL,
        CONSTRAINT PK_ProductModelStats PRIMARY KEY (model_version, sku),
        CONSTRAINT FK_ProductModelStats_Model FOREIGN KEY (model_version) REFERENCES dbo.ModelRuns(model_version),
        CONSTRAINT FK_ProductModelStats_Product FOREIGN KEY (sku) REFERENCES dbo.Products(sku),
        CONSTRAINT CK_ProductModelStats_Values CHECK (p_ref > 0 AND eps_sd > 0 AND n_train_weeks > 0)
    );
END;
GO

IF OBJECT_ID(N'dbo.RecommendationHistory', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.RecommendationHistory (
        recommendation_id BIGINT IDENTITY(1, 1) NOT NULL CONSTRAINT PK_RecommendationHistory PRIMARY KEY,
        created_at DATETIME2(0) NOT NULL CONSTRAINT DF_RecommendationHistory_Created DEFAULT (SYSUTCDATETIME()),
        model_version VARCHAR(40) NOT NULL,
        sku NVARCHAR(20) NOT NULL,
        scenario_date DATE NOT NULL,
        current_price DECIMAL(18, 4) NOT NULL,
        price_source NVARCHAR(30) NOT NULL,
        max_change DECIMAL(5, 4) NOT NULL,
        recommended_price DECIMAL(18, 4) NOT NULL,
        target_price DECIMAL(18, 4) NOT NULL,
        action NVARCHAR(10) NOT NULL,
        prob_gain FLOAT NULL,
        expected_gain_pct FLOAT NOT NULL,
        units_current FLOAT NOT NULL,
        units_recommended FLOAT NOT NULL,
        revenue_current DECIMAL(18, 2) NOT NULL,
        revenue_recommended DECIMAL(18, 2) NOT NULL,
        result_json NVARCHAR(MAX) NOT NULL,
        CONSTRAINT FK_RecommendationHistory_Model FOREIGN KEY (model_version) REFERENCES dbo.ModelRuns(model_version),
        CONSTRAINT FK_RecommendationHistory_Product FOREIGN KEY (sku) REFERENCES dbo.Products(sku),
        CONSTRAINT CK_RecommendationHistory_Prices CHECK (current_price > 0 AND recommended_price > 0 AND target_price > 0),
        CONSTRAINT CK_RecommendationHistory_ResultJson CHECK (ISJSON(result_json) = 1)
    );
    CREATE INDEX IX_RecommendationHistory_Created ON dbo.RecommendationHistory(created_at DESC);
    CREATE INDEX IX_RecommendationHistory_Product ON dbo.RecommendationHistory(sku, created_at DESC);
END;
GO
