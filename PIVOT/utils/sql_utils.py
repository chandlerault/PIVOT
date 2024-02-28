"""
This module provides utility functions for interacting with stored procedures in a SQL database.
It includes functions to execute stored procedures, run SQL queries, create or alter stored procedures,
and load SQL scripts from files.
It also provides functions to validate arguments for stored procedures, generate argument strings for SQL queries,
and map probability columns in DataFrames from string representations to dictionaries with full-form class labels.

Functions:
    - get_images_to_metrize: Get new images for metric calculation based on model and dissimilarity ID.
    - get_images_to_predict: Get new images for model predictions based on model ID.
    - generate_random_evaluation_set: Generate a random evaluation set for model testing.
    - get_test_set_df: Get labeled test data and predictions for model evaluation.
    - get_label_rank_df: Get label rankings based on model and dissimilarity ID.
    - get_train_df: Get a DataFrame for model training based on model, dissimilarity ID, and class labels.
    - validate_args: Validate arguments for a stored procedure against specified types.
    - get_server_arguments: Get server connection parameters for secure connection.
    - execute_stored_procedure: Execute a stored procedure and return the result as a Pandas DataFrame.
    - load_file_from_sql: Load SQL script from file path and return as string.
    - create_alter_stored_procedure: Create or alter a stored procedure in the database.
    - run_sql_query: Execute a SQL query on a database and return results as a Pandas DataFrame.
    - generate_arg_strings: Generate a string representation of arguments suitable for SQL queries.
    - get_class_map: Get class label mappings from the Models table based on model ID.
    - map_probs_column: Map probability column from SQL string representation to 
        dictionaries with full-form class labels.
"""
import os
import ast
from typing import Optional, Dict, Any, Tuple, Union, AsyncGenerator
import warnings
from collections import OrderedDict
from collections.abc import Sequence
import numpy as np
import pandas as pd
import streamlit as st

import pymssql
from tqdm.auto import tqdm, trange
# import constants
from utils.sql_constants import SP_ARGS_TYPE_MAPPING, SP_FILE_NAMES, RELABEL_LAMBDA
from utils import load_config


def get_images_to_metrize(model_id: int, dissimilarity_id: int,
                          server_args: Optional[Dict[str, str]] = {}) -> pd.DataFrame:
    """
    Get all new images that need scores calculated for a given model and dissimilarity/uncertainty ID.

    Parameters:
        model_id (int): The identifier of a specific model used for predictions
        dissimilarity_id (int): The identifier for the various dissimilarity/uncertainty measures of images.
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata and model predictions for a given model_id.
            Columns: IMAGE_ID, PROBS
    """
    # Call stored procedure for getting metric data
    args = OrderedDict([
        ("MODEL_ID", model_id),
        ("D_METRIC_ID", dissimilarity_id)
    ])
    df = execute_stored_procedure(sp="GENERATE_IMAGES_TO_METRIZE", args=args, server_args=server_args)

    # Convert PROBS from strings to dict with full-form class_labels
    df['PROBS'] = map_probs_column(model_id=model_id, prob_col=df['PROBS'])

    return df


def get_images_to_predict(model_id: int,
                          server_args: Optional[Dict[str, str]] = {}) -> pd.DataFrame:
    """
    Get all new images that need model predictions for a given model_id.

    Parameters:
        model_id (int): The identifier of a specific model to predict with.
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata.
            Columns: IMAGE_ID, BLOB_FILEPATH
    """
    # Call stored procedure for getting metric data
    args = OrderedDict([
        ("MODEL_ID", model_id)
    ])
    df = execute_stored_procedure(sp="GENERATE_IMAGES_TO_PREDICT", args=args, server_args=server_args)
    return df


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
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        None
    """
    # Check types of train_ids
    if train_ids is None:
        train_ids = [-1]
    if not isinstance(train_ids, Sequence):
        raise ValueError("The train_ids must be a list or other iterable.")
    for i in train_ids:
        if not isinstance(i, int):
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
    if df is not None:
        warnings.warn(f"Here are the results (expected none):\n{df.head()}", stacklevel=2)
        return df
    return None


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
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata and model predictions for a given model_id.
            Columns: IMAGE_ID, PRED_LABEL, PROBS, CONSENSUS
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

    # Convert PROBS from strings to dict with full-form class_labels
    df['PROBS'] = map_probs_column(model_id=model_id, prob_col=df['PROBS'])

    return df

@st.cache_data(show_spinner="Retrieving new batch")
def get_label_rank_df(model_id: int, #pylint: disable=too-many-arguments
                      dissimilarity_id: int,
                      batch_size: int = 100,
                      random_ratio: float = 0.5,
                      server_args: Optional[Dict[str, str]] = {}) -> pd.DataFrame:
    """
    Get a DataFrame containing label rankings based on the specified parameters.
    Calls the AL_RANKINGS stored procedure

    Parameters:
        model_id (int): The identifier of the model.
        dissimilarity_id (int): The identifier for the various dissimilarity/uncertainty measures of images.
        batch_size (int, optional): The total batch size for label ranking (default is 100).
        random_ratio (float, optional): The ratio of random images in the batch (default is 0.5).
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata ranked by dissimilarity and label count.
            Columns: IMAGE_ID, BLOB_FILEPATH, UNCERTAINTY, PRED_LABEL, PROBS, RANK_SCORE
    """
    # Check basic arguments:
    args = OrderedDict([
        ("MODEL_ID", model_id),
        ("D_METRIC_ID", dissimilarity_id),
        ("BATCH_SIZE", batch_size)
    ])
    # check types
    validate_args("AL_RANKINGS", args)
    # check fixed ranges
    if batch_size <= 0:
        raise ValueError("The batch_size must be a positive integer.")
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
        args['D_METRIC_ID'] = 0
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
    if r_df is None:
        if batch_size_r != 0:
            warnings.warn("Unexpectedly, there are no results for the random samples.", stacklevel=2)

    # Concatenate the results into a single DataFrame (may have duplicates)
    full_df = pd.concat([d_df, r_df])

    # Convert PROBS from strings to dict with full-form class_labels
    full_df['PROBS'] = map_probs_column(model_id=model_id, prob_col=full_df['PROBS'])

    return full_df


def get_train_df(model_id: int, #pylint: disable=too-many-arguments
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
        dissimilarity_id (int): The identifier for the various dissimilarity/uncertainty measures of images.
        all_classes (list): A sorted set of all classes for the model.
        train_size (int, optional): The total train size for finetuning(default is 100).
        train_ids (list): A set of image IDs for images that have already been used for training.
            Default is [-1]
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        pd.DataFrame: A DataFrame containing image metadata ranked by dissimilarity and label count.
            Columns: IMAGE_ID, BLOB_FILEPATH, ALL_LABELS, LABEL_PERCENTS, UNCERTAINTY
    """

    def generate_class_vectors(row: pd.Series, all_classes: list) -> list:
        """
        Apply the function to create the 'ClassVectors' column:
            a vector of label probabilities for each class based on the relative frequency
            of user labels.

        Note that this function isn't currently used.
        Parameters:
            row: a row of a pd.DataFrame that contains 'PercentConsensus' and 'Labels'.
            all_classes: a list of the classes in the order for training.

        Return:
            list of relative frequencies for each class in all_classes.
        """

        labels = row['Labels'].split(', ')
        percent_consensus = [float(val) for val in row['PercentConsensus'].split(', ')]
        class_vectors = [percent_consensus[all_classes.index(label)] if label in labels else 0.0 for label in
                         all_classes]
        return class_vectors

    # Check types of train_ids
    if train_ids is None:
        train_ids = [-1]
    if not isinstance(train_ids, Sequence):
        raise ValueError("The train_ids must be a list or other iterable.")
    for i in train_ids:
        if not isinstance(i, int):
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
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        Tuple: a tuple containing the strings for server, database, username, and password
    """
    # if new parameters are passed, load from dict or use config file
    CONFIG = load_config()
    server = server_args.get('server', CONFIG['server'])
    database = server_args.get('database', CONFIG['database'])
    user = server_args.get('username', CONFIG['db_user'])
    password = server_args.get('password', CONFIG['db_password'])

    return server, database, user, password


def execute_stored_procedure(sp: str,
                             args: Optional[OrderedDict[str, Any]] = {},
                             server_args: Optional[Dict[str, str]] = {}) -> Union[pd.DataFrame, None]:
    """
    Execute a stored procedure and return the result as a Pandas DataFrame if there is any.
    NOTE: Callproc() truncates arguments greater than 8000 bytes!
    Parameters:
        sp (str): The name of the stored procedure to execute.
        args (dict, optional): A dictionary containing parameters for the stored procedure.
            Expected keys: Must match those defined in SP_ARGS_TYPE_MAPPING.
            Values should be either int, float, or str, depending on the stored procedure.
        server_args (dict, optional): A dictionary containing connection parameters for the server.
            Expected keys: 'server', 'database', 'username', 'password'.
            Default values are taken from the `CONFIG` dictionary.
    Returns:
        pd.DataFrame: The result of the stored procedure in a Pandas DataFrame format
                      or None if the stored procedure didn't return any results.
    """
    use_argument_workaround = False
    # Validate args dictionary
    if args is not None:
        validate_args(sp_name=sp, args=args)
        # Check if any argument is >=8000 (max length of callproc arguments).
        for arg_k in args.keys():
            arg_v = args[arg_k]
            if isinstance(arg_v, str):
                if len(arg_v) >= 8000:
                    use_argument_workaround = True
                    warnings.warn(f"Using argument workaround with execute because {arg_k} has length {len(arg_v)}.") #pylint: disable=line-too-long
                    break
    # Get authentication strings
    server, database, user, password = get_server_arguments(server_args=server_args)
    results = None
    # set up a connection to Azure
    with pymssql.connect(server, user, password, database) as conn:
        # set up a cursor object
        with conn.cursor() as cursor:
            if use_argument_workaround:
                # construct dynamic query locally
                query = f"EXECUTE {sp} " + generate_arg_strings(args)
                cursor.execute(query)
            else:
                # gather variables for the stored procedure
                arg_tuples = tuple([args[k] for k in list(args.keys())]) #pylint: disable=consider-using-generator
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
    with open(file_path, 'r') as file: #pylint: disable=unspecified-encoding
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
            Default values are taken from the `CONFIG` dictionary.

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
        print(sql_script)
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
        warnings.warn("The query returned empty.",
                      stacklevel=2)
        return None
    # Fetch the results into a Pandas DataFrame
    df = pd.DataFrame(results, columns=columns)

    return df


def generate_arg_strings(arg_dict: OrderedDict[str, Any]) -> str:
    """
    Generates a string representation of arguments suitable for SQL queries.

    This function takes a dictionary of arguments (`arg_dict`) and constructs a string
    containing formatted key-value pairs, where keys are treated as argument names and
    values are appropriately formatted for SQL queries.

    Parameters:
        arg_dict (OrderedDict[str, Any]): A dictionary containing argument names and values.
    Returns:
        str: A string representation of formatted arguments for SQL queries.
    """
    string = ""
    i = 0
    for arg in arg_dict:
        if i != 0:
            string += ", "
        arg_val = arg_dict[arg]
        if isinstance(arg_val, str):
            string += f"@{arg}='{arg_val}'"
        else:
            string += f"@{arg}={arg_val}"
        i += 1
    return string


def get_class_map(model_id: int) -> dict:
    """
    Access `class_map` from Models table to map class labels from integers to full form.

    Parameters:
        model_id: the ID of the model for which to get class labels.
    Return:
        dict: a dictionary mapping label ints to their full name.
    """
    if not isinstance(model_id, int):
        raise ValueError(f"Expected int for model_id, received Type {type(model_id)}: {model_id}")

    class_map = run_sql_query(f"SELECT class_map FROM models WHERE m_id = {model_id};")
    class_map = ast.literal_eval(class_map.class_map[0])

    return class_map


def map_probs_column(model_id: int, prob_col: pd.Series) -> pd.Series:
    """
    Maps the probability column in a dataframe from the string output of SQL to dicts
    with keys as the true class labels instead of the numeric placeholders.

    Parameters:
        model_id (int): model ID for which to get class labels.
        prob_col (pd.Series): the series to modify
    Returns:
        pd.Series
    """
    if not isinstance(prob_col, pd.Series):
        raise ValueError(f"Expected a pandas Series, received Type {type(prob_col)}.")
    if not isinstance(prob_col.values[0], str):
        raise ValueError(f"Expected values to be str, received Type {type(prob_col.values[0])}.")

    # get mapper:
    class_map = get_class_map(model_id=model_id)

    # Define a function to map keys in each dictionary
    def map_keys(dictionary: dict) -> dict:
        return {class_map[key]: value for key, value in dictionary.items()}

    # Convert from strings to dict
    prob_col = prob_col.apply(ast.literal_eval)
    # Convert to dict with full-form class_labels
    prob_col = prob_col.apply(map_keys)
    return prob_col


def update_scores(i_ids: list[int], label_weight: int = 1, mode: str = 'insert') -> None:
    """
    Update the Dissimilarity metrics of a set of image_ids based on
    the decay function with parameter RELABEL_LAMBDA.

    Note that this only works for image_ids <1000
    Parameters:
        i_ids (list, int): list of Image_Ids that have been labeled.
        label_weight (int): the experience of the user that is labeling.
        mode (str): when this is being called;
            'insert': inserting new labels
            'delete': deleting existing labels from the table.
    Returns:
        None
    """
    # Check label_weight validity
    if not isinstance(label_weight, (int, np.int32, np.int64)):
        raise ValueError(f"Expected positive int for label_weight. Received type{type(label_weight)}, value {label_weight}")
    # Check i_id validity
    for i_id in i_ids:
        if not (isinstance(i_id, (int, np.int32, np.int64)) and i_id > 0):
            raise ValueError(f"Provided i_id {i_id} is not a positive integer.")

    # Generate Query Arguments
    direction = 1 if (mode == 'insert') else -1
    MULTIPLIER = np.exp(-RELABEL_LAMBDA * label_weight * direction)
    I_ID_TUPLE = ','.join(str(i_id) for i_id in i_ids)
    # Generate Query
    query_string = f"""
        UPDATE METRICS
        SET D_VALUE = D_VALUE * {MULTIPLIER}
        WHERE I_ID IN ({I_ID_TUPLE})
    """
    # Run Query that is expected to return nothing.
    with warnings.catch_warnings(record=True) as w:
        df = run_sql_query(query=query_string, server_args={})
        if w:
            for warning in w:
                if "arguments returned empty" in str(warning.message).lower():
                    continue
                # Re-emit other warnings
                warnings.warn(warning.message, category=warning.category, stacklevel=1)
    if df is not None:
        warnings.warn(f"Here are the results (expected none):\n{df.head()}", stacklevel=2)
        print(df)

    return


def delete_labels_cleanup(remove_labels_query: str):
    """
    Special function to remove bad labels while also accounting for updating
    the scores of the metric table.

    Parameters:
        remove_labels_query (str): the preformatted query to be executed.
    Return:
        None
    """
    # Get a DF of the grouped Label counts per I_IDs
    label_counts_query = """
        SELECT
            I_ID,
            SUM(WEIGHT) AS W_COUNT
        FROM LABELS
        GROUP BY I_ID
    """
    pre_del_counts = run_sql_query(query=label_counts_query, server_args={})
    pre_del_counts = pre_del_counts.rename(columns={'W_COUNT': 'BEFORE'})

    # Run delete query that is expected to return nothing.
    with warnings.catch_warnings(record=True) as w:
        df = run_sql_query(query=remove_labels_query, server_args={})
        if w:
            for warning in w:
                if "arguments returned empty" in str(warning.message).lower():
                    continue
                # Re-emit other warnings
                warnings.warn(warning.message, category=warning.category, stacklevel=1)
    if df is not None:
        warnings.warn(f"Here are the results (expected none):\n{df.head()}", stacklevel=2)
        print(df)

    # Post deletion query
    post_del_counts = run_sql_query(query=label_counts_query, server_args={})
    post_del_counts = post_del_counts.rename(columns={'W_COUNT': 'AFTER'})

    # Label differences
    diff_df = pre_del_counts.merge(post_del_counts, on='I_ID', how='left').fillna(0)
    diff_df['WEIGHT'] = diff_df['BEFORE'].astype(int) - diff_df['AFTER'].astype(int)

    # check to make sure that no weights increased during deletion
    assert diff_df['WEIGHT'].min() > 0, "Unexpectedly increased total label count during deletion!"
    label_difference = diff_df.query("WEIGHT > 0")

    # Extract image IDs that need to be modified per label_weights & modify them
    for label_weight in label_difference['WEIGHT'].unique():
        subset = diff_df.query(f'WEIGHT == {label_weight}')
        i_id_list = list(subset['I_ID'].values)
        for i_id_chunk in tqdm(chunky(i_id_list, 500), total=len(i_id_list)//500, desc="Chunked I_IDs", leave=False):
            update_scores(i_ids=i_id_chunk, label_weight=label_weight, mode='delete')

    return


def chunky(lst: list, n: int = 500) -> AsyncGenerator[int, None]:
    """
    Break up a list in into chunks to make it easier to push.
    Parameters:
        lst (list): the list to chunkify
        n (int): the size of each chunk. Size is 500.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
