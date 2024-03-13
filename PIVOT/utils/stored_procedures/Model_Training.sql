/*
Name: AL_TRAIN_SET
Description: This stored procedure generates a training set for active learning based on
             specified model and metric IDs, with options to exclude certain images from sampling.

Parameters:
- @MODEL_ID: Integer denoting Model ID for filtering predictions.
- @D_METRIC_ID: Integer denoting AL metric ID for filtering predictions.
- @TRAIN_SIZE: Integer denoting size of training set to generate.
- @IMAGE_IDS: Comma-separated string containing image IDs to be excluded from sampling.
*/

CREATE OR ALTER PROCEDURE AL_TRAIN_SET
    @MODEL_ID INT,
    @D_METRIC_ID INT,
    @TRAIN_SIZE INT,
    @IMAGE_IDS VARCHAR(MAX) -- other image_ids to be excluded from sampling
AS
BEGIN
    DECLARE @EXCLUDE_IDS TABLE (I_ID INT);

    -- Convert comma-separated string to a table variable
    INSERT INTO @EXCLUDE_IDS (I_ID)
    SELECT CAST(value AS INT)
    FROM STRING_SPLIT(@IMAGE_IDS, ',');

    -- CTE to calculate label counts and percentages for each image.
    WITH LABEL_COUNTS AS (
        SELECT I_ID,
               LABEL,
               SUM(WEIGHT) AS W_COUNT,
--                SUM(SUM(WEIGHT)) OVER (PARTITION BY I_ID) AS TOTAL_SUM,
               CAST(SUM(WEIGHT) AS FLOAT) / (SUM(SUM(WEIGHT)) OVER (PARTITION BY I_ID)) as PERCENT_CONSENSUS
        FROM LABELS
        GROUP BY I_ID, LABEL
    ),
    -- CTE to aggregate all labels and their percentages for each image.
    LABEL_STATS AS (
        SELECT I_ID,
               STRING_AGG(LABEL, ', ') WITHIN GROUP (ORDER BY W_COUNT DESC) AS ALL_LABELS,
               STRING_AGG(CAST(PERCENT_CONSENSUS AS NVARCHAR(255)), ', ') WITHIN GROUP (ORDER BY W_COUNT DESC) AS LABEL_PERCENTS
        FROM LABEL_COUNTS
        GROUP BY I_ID
    ),
    --  CTE selecting distinct image IDs from METRICS with D_ID = 0 indicating test images.
    TEST_IMAGES AS (
        SELECT DISTINCT I_ID
        FROM METRICS
        WHERE D_ID = 0
    )
    -- Selects top-ranked images for the training set based on the specified model and metric IDs,
    -- excluding specified image IDs and test images.
    SELECT TOP (@TRAIN_SIZE)
           I.I_ID AS IMAGE_ID,
           I.FILEPATH AS BLOB_FILEPATH,
           L.ALL_LABELS AS ALL_LABELS,
           L.LABEL_PERCENTS AS LABEL_PERCENTS,
           M.D_VALUE AS UNCERTAINTY
    FROM METRICS AS M
    INNER JOIN IMAGES AS I
        ON M.I_ID = I.I_ID
    INNER JOIN PREDICTIONS AS P
        ON M.I_ID = P.I_ID AND M.M_ID = P.M_ID
    INNER JOIN LABEL_STATS AS L
        ON M.I_ID = L.I_ID
    LEFT JOIN TEST_IMAGES AS TI
        ON I.I_ID = TI.I_ID
    LEFT JOIN @EXCLUDE_IDS O_EI
        ON I.I_ID = O_EI.I_ID
    WHERE TI.I_ID IS NULL
      AND O_EI.I_ID IS NULL
      AND M.M_ID = @MODEL_ID
      AND M.D_ID = @D_METRIC_ID
    ORDER BY UNCERTAINTY DESC;
END;