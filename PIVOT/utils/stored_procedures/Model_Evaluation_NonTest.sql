/*
Name: MODEL_EVALUATION_NON_TEST
Description: This stored procedure evaluates model predictions on non-test images
             based on label consensus and a minimum percentage threshold.
             Take label with a weighted sum > some fraction of the total sum for that image.
Parameters:
- @MODEL_ID: Integer denoting Model ID for filtering predictions.
- @MINIMUM_PERCENT: Float denoting minimum percentage to achieve label consensus.
*/

CREATE OR ALTER PROCEDURE MODEL_EVALUATION_NON_TEST
    @MODEL_ID INT,
    @MINIMUM_PERCENT FLOAT
AS
BEGIN
    -- (CTE) to calculate label counts, percentages, and ranks for each image.
    WITH LABEL_COUNTS AS (
        SELECT I_ID,
               LABEL,
               SUM(WEIGHT) AS W_COUNT,
               SUM(SUM(WEIGHT)) OVER (PARTITION BY I_ID) AS TOTAL_SUM,
               CAST(SUM(WEIGHT) AS FLOAT) / (SUM(SUM(WEIGHT)) OVER (PARTITION BY I_ID)) as PERCENT_CONSENSUS,
               ROW_NUMBER() OVER (PARTITION BY I_ID ORDER BY SUM(WEIGHT) DESC) AS LABEL_RANK -- MIGHT BE TIES!
        FROM LABELS
        GROUP BY I_ID, LABEL
    ),
    -- CTE to select distinct image IDs from the METRICS table where D_ID = 0 (test data).
    EXISTING_IMAGES AS (
        SELECT DISTINCT I_ID
        FROM METRICS
        WHERE D_ID = 0
    )
    -- Selects model predictions for non-test images based on label consensus and minimum percentage threshold.
    SELECT I.I_ID AS IMAGE_ID,
           P.PRED_LABEL AS PRED_LABEL,
           P.CLASS_PROB AS PROBS,
           L.LABEL AS CONSENSUS -- CURRENTLY TAKING THE LABEL WITH MAX WEIGHT, TIES BROKEN AT RANDOM
    FROM PREDICTIONS AS P
    INNER JOIN IMAGES AS I
        ON P.I_ID = I.I_ID
    INNER JOIN LABEL_COUNTS AS L
        ON P.I_ID = L.I_ID
    LEFT JOIN EXISTING_IMAGES EI
        ON I.I_ID = EI.I_ID
    WHERE EI.I_ID IS NULL
      AND P.M_ID = @MODEL_ID
      AND L.LABEL_RANK = 1
      AND L.PERCENT_CONSENSUS > @MINIMUM_PERCENT;
END;
