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

    WITH EXISTING_IMAGES AS (
        SELECT DISTINCT I_ID
        FROM METRICS
        WHERE D_ID = 0
    )
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
