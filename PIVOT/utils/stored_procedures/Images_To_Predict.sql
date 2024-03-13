/*
Name: GENERATE_IMAGES_TO_PREDICT
Description: This stored procedure selects images from the IMAGES table
             that have not been previously predicted by the specified model.
Parameters:
- @MODEL_ID: Integer denoting Model ID for subsequent predictions
*/

CREATE OR ALTER PROCEDURE GENERATE_IMAGES_TO_PREDICT
    @MODEL_ID INT
AS
BEGIN
    WITH EXISTING_IMAGES AS (
        SELECT DISTINCT I_ID
        FROM PREDICTIONS
        WHERE M_ID = @MODEL_ID
    )
    -- Selects image IDs and their associated file paths from the IMAGES table,
    -- excluding images that have been previously predicted by the specified model.
    SELECT I.I_ID AS IMAGE_ID,
           I.FILEPATH AS BLOB_FILEPATH
    FROM IMAGES AS I
    LEFT JOIN EXISTING_IMAGES EI ON I.I_ID = EI.I_ID
    WHERE EI.I_ID IS NULL;
END;
