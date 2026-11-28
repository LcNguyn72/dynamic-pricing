SET XACT_ABORT ON;
BEGIN TRANSACTION;

IF OBJECT_ID(N'dbo.M5Catalog', N'U') IS NULL
    CREATE TABLE dbo.M5Catalog (
        item_id varchar(32) NOT NULL PRIMARY KEY REFERENCES dbo.M5Products(item_id),
        display_name nvarchar(100) NOT NULL,
        position tinyint NOT NULL UNIQUE CHECK (position BETWEEN 1 AND 20),
        reference_week date NOT NULL
    );

IF OBJECT_ID(N'dbo.M5PriceChanges', N'U') IS NULL
    CREATE TABLE dbo.M5PriceChanges (
        change_id bigint IDENTITY PRIMARY KEY,
        store_id varchar(16) NOT NULL,
        item_id varchar(32) NOT NULL,
        old_price decimal(10,2) NOT NULL CHECK (old_price > 0),
        new_price decimal(10,2) NOT NULL CHECK (new_price > 0),
        changed_at datetime2(0) NOT NULL DEFAULT SYSUTCDATETIME(),
        FOREIGN KEY (store_id, item_id) REFERENCES dbo.M5StoreProducts(store_id, item_id)
    );

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE object_id = OBJECT_ID('dbo.M5PriceChanges') AND name = 'IX_M5PriceChanges_Offer')
    CREATE INDEX IX_M5PriceChanges_Offer ON dbo.M5PriceChanges(item_id, store_id, change_id DESC)
        INCLUDE (old_price, new_price, changed_at);

DECLARE @items TABLE (position tinyint, item_id varchar(32), display_name nvarchar(100));
INSERT @items VALUES
    (1, 'FOODS_1_096', N'Mứt quả mọng'),
    (2, 'FOODS_3_782', N'Yến mạch cán'),
    (3, 'FOODS_2_152', N'Mì ống khô'),
    (4, 'FOODS_1_012', N'Hạnh nhân rang'),
    (5, 'FOODS_2_241', N'Gạo trắng'),
    (6, 'FOODS_3_219', N'Sô-cô-la đen'),
    (7, 'FOODS_3_030', N'Ngũ cốc ăn sáng'),
    (8, 'FOODS_1_170', N'Dầu ô liu'),
    (9, 'HOUSEHOLD_1_016', N'Khăn tay cotton'),
    (10, 'HOUSEHOLD_1_365', N'Bộ khăn lau'),
    (11, 'HOUSEHOLD_1_318', N'Bàn chải rửa bát'),
    (12, 'HOUSEHOLD_1_215', N'Miếng rửa chén'),
    (13, 'HOUSEHOLD_2_219', N'Bình xịt thủy tinh'),
    (14, 'HOUSEHOLD_1_323', N'Hộp đựng thực phẩm'),
    (15, 'HOUSEHOLD_1_364', N'Bộ ly thủy tinh'),
    (16, 'HOBBIES_1_323', N'Bộ bút chì màu'),
    (17, 'HOBBIES_1_194', N'Sổ tay lò xo'),
    (18, 'HOBBIES_1_201', N'Bộ màu nước'),
    (19, 'HOBBIES_1_240', N'Băng keo giấy'),
    (20, 'HOBBIES_1_367', N'Bộ bài tây');

DECLARE @week date = '20160516';
IF (SELECT COUNT(*) FROM dbo.M5Stores) <> 10
    THROW 50100, 'Import all 10 M5 stores before seeding the catalog.', 1;

IF EXISTS (
    SELECT i.item_id
    FROM @items i
    CROSS JOIN dbo.M5Stores s
    LEFT JOIN dbo.M5WeeklySales w
        ON w.store_id = s.store_id AND w.item_id = i.item_id
        AND w.week_start = @week AND w.observed_sales = 1
    GROUP BY i.item_id HAVING COUNT(w.store_id) <> 10
)
    THROW 50100, 'Import all 10 M5 stores before seeding the catalog.', 1;

INSERT dbo.M5Catalog (item_id, display_name, position, reference_week)
SELECT i.item_id, i.display_name, i.position, @week FROM @items i
WHERE NOT EXISTS (SELECT 1 FROM dbo.M5Catalog c WHERE c.item_id = i.item_id);

EXEC(N'CREATE OR ALTER VIEW dbo.M5CatalogOffers AS
    SELECT c.item_id, w.store_id, s.state_id, c.reference_week,
           CAST(w.price AS decimal(10,2)) AS reference_price,
           COALESCE(edit.new_price, CAST(w.price AS decimal(10,2))) AS price,
           edit.changed_at
    FROM dbo.M5Catalog c
    CROSS JOIN dbo.M5Stores s
    JOIN dbo.M5WeeklySales w ON w.store_id = s.store_id AND w.item_id = c.item_id
                            AND w.week_start = c.reference_week AND w.observed_sales = 1
    OUTER APPLY (
        SELECT TOP (1) new_price, changed_at
        FROM dbo.M5PriceChanges p WHERE p.item_id = c.item_id AND p.store_id = w.store_id
        ORDER BY p.change_id DESC
    ) AS edit;');
COMMIT TRANSACTION;
