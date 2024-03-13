/*
Name: GENERATE_IMAGES_TO_METRIZE
Description: This stored procedure selects images from the PREDICTIONS table to be inserted
             into the METRICS table for a specific model and metric combination.
Parameters:
- @MODEL_ID: Integer denoting Model ID for filtering predictions
- @D_ID: Integer denoting AL metric ID for filtering predictions.
*/

CREATE OR ALTER PROCEDURE GENERATE_IMAGES_TO_METRIZE
    @MODEL_ID INT,
    @D_ID INT
AS
BEGIN
    WITH EXISTING_METRICS AS (
        SELECT DISTINCT I_ID
        FROM METRICS
        WHERE M_ID = @MODEL_ID
          AND D_ID = @D_ID
    )
    SELECT P.I_ID AS IMAGE_ID,
           P.CLASS_PROB AS PROBS
    FROM PREDICTIONS AS P
    LEFT JOIN EXISTING_METRICS EM ON P.I_ID = EM.I_ID
    WHERE EM.I_ID IS NULL;
END;
