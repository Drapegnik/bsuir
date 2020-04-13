USE [accidents]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

-- adress dimension
CREATE TABLE [dbo].[DimAddress](
 [PK_id] [int] NOT NULL IDENTITY,
 [district_name] [nvarchar](100) NOT NULL,
 [neighborhood_name] [nvarchar](100) NOT NULL,
 [street] [nvarchar](100) NOT NULL,
 CONSTRAINT [PK_DimAddress_id] PRIMARY KEY CLUSTERED ([PK_id] ASC) WITH (
  PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF,
  IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON,
  ALLOW_PAGE_LOCKS = ON
 ) ON [PRIMARY]
) ON [PRIMARY]
GO

-- date dimension
CREATE TABLE [dbo].[DimDate](
 [PK_id] [int] NOT NULL IDENTITY,
 [weekday]  NOT NULL,
 [month] [nvarchar](100) NOT NULL,
 [day] [int] NOT NULL,
 [hour] [int] NOT NULL,
 [day_part] [nvarchar](100) NOT NULL,
 CONSTRAINT [PK_DimDate_id] PRIMARY KEY CLUSTERED ([PK_id] ASC) WITH (
  PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF,
  IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON,
  ALLOW_PAGE_LOCKS = ON
 ) ON [PRIMARY]
) ON [PRIMARY]
GO

-- accident facts
CREATE TABLE [dbo].[FactAccident](
 [PK_id] [bigint] NOT NULL,
 [FK_date_id] [int] NOT NULL,
 [FK_address_id] [int] NOT NULL,
 [mild_injuries] [int] NOT NULL,
 [serious_injuries] [int] NOT NULL,
 [victims] [int] NOT NULL,
 [vehicles_involved] [int] NOT NULL,
 [longitude] [float] NOT NULL,
 [latitude] [float] NOT NULL,
 CONSTRAINT [PK_FactAccident_id] PRIMARY KEY CLUSTERED ([PK_id] ASC) WITH (
  PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF,
  IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON,
  ALLOW_PAGE_LOCKS = ON
 ) ON [PRIMARY]
) ON [PRIMARY]

GO
ALTER TABLE [dbo].[FactAccident] WITH CHECK ADD CONSTRAINT
 [FK_FactAccident_DimDate_id] FOREIGN KEY([FK_date_id])
REFERENCES [dbo].[DimDate] ([PK_id])
ON UPDATE CASCADE
ON DELETE CASCADE

GO
ALTER TABLE
  [dbo].[FactAccident] CHECK CONSTRAINT [FK_FactAccident_DimDate_id]

GO
ALTER TABLE [dbo].[FactAccident] WITH CHECK ADD CONSTRAINT
 [FK_FactAccident_DimAddress_id] FOREIGN KEY([FK_address_id])
REFERENCES [dbo].[DimAddress] ([PK_id])
ON UPDATE CASCADE
ON DELETE CASCADE

GO
ALTER TABLE
  [dbo].[FactAccident] CHECK CONSTRAINT [FK_FactAccident_DimAddress_id]
GO
