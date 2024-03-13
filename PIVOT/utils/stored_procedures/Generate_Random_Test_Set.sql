/*
Name: GENERATE_RANDOM_TEST_SET
Description: This stored procedure generates a random test set of specified size from a
             pool of available images, excluding any image IDs provided in @IMAGE_IDS.
Parameters:
- @TEST_SIZE: Integer value specifying the size of the test set to be generated.
- @IMAGE_IDS: Comma-separated string containing image IDs to be excluded from sampling.
*/

CREATE OR ALTER PROCEDURE GENERATE_RANDOM_TEST_SET
    @TEST_SIZE INT,
    @IMAGE_IDS VARCHAR(MAX) -- other image_ids to be excluded from sampling
AS
BEGIN
    DECLARE @EXCLUDE_IDS TABLE (I_ID INT);

    -- Convert comma-separated string to a table variable
    INSERT INTO @EXCLUDE_IDS (I_ID)
    SELECT CAST(value AS INT)
    FROM STRING_SPLIT(@IMAGE_IDS, ',');

    -- Common Table Expression (CTE) to retrieve existing images
    WITH EXISTING_IMAGES AS (
        SELECT DISTINCT I_ID
        FROM METRICS
        WHERE D_ID = 0
    )
    -- Insert into METRICS a random sample of images not in existing images or excluded IDs
    INSERT INTO METRICS (I_ID, M_ID, D_ID, D_VALUE)
    SELECT TOP (@TEST_SIZE)
           I.I_ID AS I_ID,
           0 as M_ID,
           0 as D_ID,
           1 as D_VALUE
    FROM IMAGES AS I
    LEFT JOIN EXISTING_IMAGES EI ON I.I_ID = EI.I_ID
    LEFT JOIN @EXCLUDE_IDS O_EI ON I.I_ID = O_EI.I_ID
    WHERE EI.I_ID IS NULL
      AND O_EI.I_ID IS NULL
    ORDER BY NEWID();
END;
