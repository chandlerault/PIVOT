import os
from typing import Optional, Dict, Any, Tuple, Union
import warnings
from collections import OrderedDict
from collections.abc import Sequence
import pandas as pd

import yaml
import pymssql

# import constants
# from data_utils.sql_constants import SP_ARGS_TYPE_MAPPING, SP_FILE_NAMES
from sql_constants import SP_ARGS_TYPE_MAPPING, SP_FILE_NAMES

# CONFIG_FILE_PATH = 'config/config.yaml'
# config = yaml.load(open(CONFIG_FILE_PATH, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

config = {
    "server": "capstoneservercjault.database.windows.net",
    "database": "capstoneazure",
    "db_user": "CloudSA85da88d3",
    "db_password": "Passwordtest123"
}


def generate_random_evaluation_set(test_size: int = 100000,
                                   train_ids: Optional[Sequence[int]] = None,
                                   server_args: Optional[Dict[str, str]] = {}) -> None:
    """
    Uses a stored procedure to create new test data and add it to the Metrics Table.
    Note: This new data doesn't have any predictions.

    Parameters:
        test_size (int): The number of new test images to gather. Default: 100,000
        train_ids (list): A set of image IDs for images that have already been used for training.
            Default is [-1]
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `config` dictionary.
    Returns:
        None
    """
    # Check types of train_ids
    if train_ids is None:
        train_ids = [-1]
    if not(isinstance(train_ids, Sequence)):
        raise ValueError("The train_ids must be a list or other iterable.")
    for i in train_ids:
        if not(isinstance(i, int)):
            raise ValueError("All elements in train_ids must be integers.")
    # Convert list into string
    train_ids = ','.join(str(i) for i in train_ids)
    args = OrderedDict([
        ("TEST_SIZE", test_size),
        ("IMAGE_IDS", train_ids),
    ])
    # validate argument types
    validate_args(sp_name='GENERATE_RANDOM_TEST_SET', args=args)
    # ensure test_size is a valid range.
    if test_size <= 0:
        raise ValueError("The test_size must be a positive integer.")
    # Call stored procedure for getting metric data
    # expect to raise an empty return warning that we'll supress.
    with warnings.catch_warnings(record=True) as w:
        df = execute_stored_procedure(sp="GENERATE_RANDOM_TEST_SET", args=args, server_args=server_args)
        if w:
            for warning in w:
                if "arguments returned empty" in str(warning.message).lower():
                    continue
                # Re-emit other warnings
                warnings.warn(warning.message, category=warning.category, stacklevel=1)
    if df:
        warnings.warn(f"Here are the results (expected none):\n{df}", stacklevel=2)

    return


def get_test_set_df(model_id: int,
                    minimum_percent: Optional[float] = 0.0,
                    sp_name: Optional[str] = 'MODEL_EVALUATION_MAX_CONSENSUS_FILTERING',
                    server_args: Optional[Dict[str, str]] = {}) -> pd.DataFrame:
    """
    Get all labeled test data along with model predictions for a given model_id.

    Parameters:
        model_id (int): The identifier of a specific model to evaluate.
        minimum_percent (float): A minimum threshold of % agreement among labels for a given image.
            Default is 0.0, signifying no filtering
        sp_name (str): The name of the stored procedure to get "test" set for evaluation.
            Default is "MODEL_EVALUATION_MAX_CONSENSUS_FILTERING", which actually generates a test set.
            Other option is "MODEL_EVALUATION_NON_TEST", which gathers all non-test labels.
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `config` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata and model predictions for a given model_id.
            Columns: IMAGE_ID, PRED_LABEL, CONSENSUS
    """
    # Check that the sp_name is valid
    valid_sp_names = {'MODEL_EVALUATION_MAX_CONSENSUS_FILTERING', 'MODEL_EVALUATION_NON_TEST'}
    if sp_name not in valid_sp_names:
        raise ValueError(f"Invalid sp_name {sp_name}, expected one of these two: {valid_sp_names}.")
    # Call stored procedure for getting metric data
    args = OrderedDict([
        ("MODEL_ID", model_id),
        ("MINIMUM_PERCENT", minimum_percent),
    ])
    if (minimum_percent < 0.0) or (minimum_percent > 1.0):
        raise ValueError("The minimum_percent must be a positive float between 0.0 and 1.0")

    df = execute_stored_procedure(sp=sp_name, args=args, server_args=server_args)

    return df


def get_label_rank_df(model_id: int,
                      dissimilarity_id: int,
                      batch_size: int = 100,
                      relabel_lambda: float = 0.069,
                      random_ratio: float = 0.5,
                      server_args: Optional[Dict[str, str]] = {}) -> pd.DataFrame:
    """
    Get a DataFrame containing label rankings based on the specified parameters.
    Calls the AL_RANKINGS stored procedure

    Parameters:
        model_id (int): The identifier of the model.
        dissimilarity_id (int): The identifier for dissimilarity images.
        batch_size (int, optional): The total batch size for label ranking (default is 100).
        relabel_lambda (float, optional): The relabeling lambda parameter (default is 0.069).
        random_ratio (float, optional): The ratio of random images in the batch (default is 0.5).
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `config` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata ranked by dissimilarity and label count.
            Columns: IMAGE_ID, BLOB_FILEPATH, UNCERTAINTY, PRED_LABEL, PROBS, RANK_SCORE
    """

    # Check basic arguments:
    args = OrderedDict([
        ("MODEL_ID", model_id),
        ("D_METRIC_ID", dissimilarity_id),
        ("RELABEL_LAMBDA", relabel_lambda),
        ("BATCH_SIZE", batch_size)
    ])
    # check types
    validate_args("AL_RANKINGS", args)
    # check fixed ranges
    if batch_size <= 0:
        raise ValueError("The batch_size must be a positive integer.")
    if relabel_lambda < 0:
        raise ValueError("The relabel_lambda must be a positive float.")
    if (random_ratio < 0) or (random_ratio > 1):
        raise ValueError("The random_ratio must be a positive float between 0 & 1.")

    # Calculate the number of random and dissimilarity images based on the ratio
    batch_size_r = int(random_ratio * batch_size)
    batch_size_d = batch_size - batch_size_r

    # Call stored procedure for label ranking with dissimilarity scores
    if batch_size_d > 0:
        args["BATCH_SIZE"] = batch_size_d
        d_df = execute_stored_procedure(sp="AL_RANKINGS", args=args, server_args=server_args)
    else:
        d_df = None
    # Call stored procedure for label ranking with D_ID = 0 (represents random images)
    if batch_size_r > 0:
        args['D_ID'] = 0
        args['BATCH_SIZE'] = batch_size_r
        r_df = execute_stored_procedure(sp="AL_RANKINGS", args=args, server_args=server_args)
    else:
        r_df = None
    # Check that both d_df and r_df are not None:
    if d_df is None:
        if r_df is None:
            warnings.warn("There are no label ranking results available!", stacklevel=2)
        # if batch_size is 0, then we expect to return 1 of the dfs
        if batch_size_d != 0:
            warnings.warn("Unexpectedly, there are no results for the uncertainty samples.", stacklevel=2)
        return r_df
    if r_df is None:
        if batch_size_r != 0:
            warnings.warn("Unexpectedly, there are no results for the random samples.", stacklevel=2)

    # Concatenate the results into a single DataFrame (may have duplicates)
    full_df = pd.concat([d_df, r_df])

    return full_df


def get_train_df(model_id: int,
                 dissimilarity_id: int,
                 all_classes: list[str],
                 train_size: int = 100,
                 train_ids: Optional[Sequence[int]] = None,
                 server_args: Optional[Dict[str, str]] = {}) -> pd.DataFrame:
    """
    Get a DataFrame for training containing labels based on the specified parameters.
    Calls the AL_TRAIN_SET stored procedure.

    Parameters:
        model_id (int): The identifier of the model.
        dissimilarity_id (int): The identifier for dissimilarity images.
        all_classes (list): A sorted set of all classes for the model.
        train_size (int, optional): The total train size for finetuning(default is 100).
        train_ids (list): A set of image IDs for images that have already been used for training.
            Default is [-1]
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `config` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata ranked by dissimilarity and label count.
            Columns: IMAGE_ID, BLOB_FILEPATH, ALL_LABELS, LABEL_PERCENTS, UNCERTAINTY
    """


    def generate_class_vectors(row, all_classes):
        # Apply the function to create the 'ClassVectors' column
        #

        labels = row['Labels'].split(', ')
        percent_consensus = [float(val) for val in row['PercentConsensus'].split(', ')]
        class_vectors = [percent_consensus[all_classes.index(label)] if label in labels else 0.0 for label in
                         all_classes]
        return class_vectors

    # Check types of train_ids
    if train_ids is None:
        train_ids = [-1]
    if not (isinstance(train_ids, Sequence)):
        raise ValueError("The train_ids must be a list or other iterable.")
    for i in train_ids:
        if not (isinstance(i, int)):
            raise ValueError("All elements in train_ids must be integers.")
    # Convert list into string
    train_ids = ','.join(str(i) for i in train_ids)
    # Check basic arguments:
    args = OrderedDict([
        ("MODEL_ID", model_id),
        ("D_METRIC_ID", dissimilarity_id),
        ("TRAIN_SIZE", train_size),
        ("TRAIN_IDS", train_ids)
    ])
    # check types
    validate_args("AL_TRAIN_SET", args)
    # check fixed ranges
    if train_size <= 0:
        raise ValueError("The batch_size must be a positive integer.")
    # Execute stored procedure
    df = execute_stored_procedure(sp='AL_TRAIN_SET', args=args, server_args=server_args)
    # Generate single class label
    df['OneLabel'] = df['ALL_LABELS'].str.split(',', expand=True)[0]
    class_vectors = df.apply(lambda row: generate_class_vectors(row, all_classes), axis=1)
    df['class_vectors'] = class_vectors
    return df


def validate_args(sp_name: str, args: Optional[OrderedDict[str, Any]]) -> None:
    """
    Validate the arguments for a stored procedure against the specified types.

    Parameters:
        sp_name (str): The name of the stored procedure.
        args (dict, optional): A dictionary containing the arguments for the stored procedure.

    Raises:
        ValueError: If any argument has an unexpected type based on the specified types in SP_ARGS_TYPE_MAPPING.
                    Or if an argument is expected and isn't found.
    """
    if sp_name not in SP_ARGS_TYPE_MAPPING:
        warnings.warn(f"The stored procedure, {sp_name}, hasn't been strongly typed. Proceed with caution!")
        return
    expected_types = SP_ARGS_TYPE_MAPPING.get(sp_name, {})
    if expected_types and args is not None:
        for key, expected_type in expected_types.items():
            value = args.get(key)
            if (value is not None) and not isinstance(value, expected_type):
                raise ValueError(
                    f"Invalid value type for '{key}' in stored procedure '{sp_name}'. Expected {expected_type}.")
            if value is None and key not in args:
                raise ValueError(f"Missing required argument '{key}' in stored procedure '{sp_name}'.")
    return


def get_server_arguments(server_args: Optional[Dict[str, str]] = {}) -> Tuple[str, str, str, str]:
    """
    Returns the server arguments for a secure connection to Azure SQL via Pymssql.

    Parameters
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `config` dictionary.
    Returns:
        Tuple: a tuple containing the strings for server, database, username, and password
    """
    # if new parameters are passed, load from dict or use config file
    server = server_args.get('server', config['server'])
    database = server_args.get('database', config['database'])
    user = server_args.get('username', config['db_user'])
    password = server_args.get('password', config['db_password'])

    return server, database, user, password


def execute_stored_procedure(sp: str,
                             args: Optional[OrderedDict[str, Any]] = {},
                             server_args: Optional[Dict[str, str]] = {}) -> Union[pd.DataFrame, None]:
    """
    Execute a stored procedure and return the result as a Pandas DataFrame if there is any.

    Parameters:
        sp (str): The name of the stored procedure to execute.
        args (dict, optional): A dictionary containing parameters for the stored procedure.
            Expected keys: Must match those defined in SP_ARGS_TYPE_MAPPING.
            Values should be either int, float, or str, depending on the stored procedure.
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `config` dictionary.
    Returns:
        pd.DataFrame: The result of the stored procedure in a Pandas DataFrame format
                      or None if the stored procedure didn't return any results.
    """
    # Validate args dictionary
    if args is not None:
        validate_args(sp_name=sp, args=args)
    # Get authentication strings
    server, database, user, password = get_server_arguments(server_args=server_args)
    results = None
    # set up a connection to Azure
    with pymssql.connect(server, user, password, database) as conn:
        # set up a cursor object
        with conn.cursor() as cursor:
            # gather variables for the stored procedure
            arg_tuples = tuple([args[k] for k in list(args.keys())])
            # execute stored procedure
            cursor.callproc(sp, arg_tuples)
            # Fetch the results
            try:
                results = cursor.fetchall()
                # get column names
                columns = [column[0] for column in cursor.description]
            except pymssql.OperationalError as e:
                if ("executed statement has no resultset" in str(e)) and (cursor.rowcount == -1):
                    results = None
                    columns = None
            # Commit the changes (if needed)
            conn.commit()
    # Check that results isn't empty.
    if not results:
        warnings.warn(f"The stored procedure {sp} with the following arguments returned empty:\n{args}",
                      stacklevel=2)
        return None
    # Fetch the results into a Pandas DataFrame
    df = pd.DataFrame(results, columns=columns)

    return df


def load_file_from_sql(file_path: str) -> str:
    """Loads SQL file from file path and returns string."""
    with open(file_path, 'r') as file:
        sql_script = file.read()
    assert sql_script is not None, f"{sql_script} is empty"

    return sql_script


def create_alter_stored_procedure(sp_name: str, file_path: Optional[str] = None,
                                  server_args: Optional[Dict[str, str]] = {}) -> None:
    """
    Create or alter a stored procedure by its name. It defaults to a preset file location
    based on the string using SP_FILE_NAMES but this is overriden by file_path.

    Parameters:
        sp_name (str): The name of the stored procedure to execute.
        file_path (str): The path to the SQL script file containing the stored procedure.
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `config` dictionary.

    Returns:
        None: The function does not return any value.

    Raises:
        pymssql.Error: If there is an error during the execution of the stored procedure.
        FileNotFoundError: If the specified SQL script file is not found.
    """

    # Get authentication strings
    server, database, user, password = get_server_arguments(server_args=server_args)

    # Validate the stored_procedure
    validate_args(sp_name=sp_name, args=None)
    # get file_path
    if sp_name in SP_FILE_NAMES:
        if file_path:
            print(f"Using custom file to create procedure {sp_name}: {file_path}")
        else:
            file_path = SP_FILE_NAMES[sp_name]
            file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), file_path))
            print(f"Using preset file to create procedure {sp_name}: {file_path}")
    else:
        warnings.warn(f"No record of procedure called {sp_name}. Running wild, buckoo!")

    try:
        # Read the SQL script from the file
        sql_script = load_file_from_sql(file_path)
        # Establish a connection to the database
        with pymssql.connect(server, user, password, database) as conn:
            # Create a cursor
            with conn.cursor() as cursor:
                # Execute the stored procedure using the content of the SQL script
                cursor.execute(sql_script)
                # Commit the changes (if needed)
                conn.commit()
    except FileNotFoundError as file_error:
        raise FileNotFoundError(f"SQL script file not found: {file_path}") from file_error
    except pymssql.Error as sql_error:
        # Handle specific exceptions related to pymssql, if needed
        raise sql_error

    return


def run_sql_query(query: str, server_args: Optional[Dict[str, str]] = {}) -> Union[pd.DataFrame, None]:
    """
    Execute a SQL query on a database and return the results as a Pandas DataFrame.

    Parameters:
        query (str): The SQL query to execute.
        server_args (dict, optional): A dictionary containing server connection parameters.
                                   Defaults to an empty dictionary.

    Returns:
        Union[pd.DataFrame, None]: A Pandas DataFrame containing the query results,
                                 or None if the query didn't return any results.

    Raises:
        pymssql.Error: If there is an error during the execution of the SQL query.
    """
    # Get authentication strings
    server, database, user, password = get_server_arguments(server_args=server_args)
    results = None
    # Establish a connection to the database
    with pymssql.connect(server, user, password, database) as conn:
        # Create a cursor
        with conn.cursor() as cursor:
            # Execute the stored procedure using the content of the SQL script
            cursor.execute(query)
            # Fetch the results
            try:
                results = cursor.fetchall()
                # get column names
                columns = [column[0] for column in cursor.description]
            except pymssql.OperationalError as e:
                if ("executed statement has no resultset" in str(e)) and (cursor.rowcount == -1):
                    results = None
                    columns = None
            # Commit the changes (if needed)
            conn.commit()

    # Check that results isn't empty.
    if not results:
        warnings.warn(f"The query returned empty.",
                      stacklevel=2)
        return None
    # Fetch the results into a Pandas DataFrame
    df = pd.DataFrame(results, columns=columns)

    return df
