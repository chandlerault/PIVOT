from typing import Optional, Dict, Any, Tuple, Union
import warnings
from collections import OrderedDict
import pandas as pd

import yaml
import pymssql

# import constants
from data_utils.sql_constants import SP_ARGS_TYPE_MAPPING, SP_FILE_NAMES

CONFIG_FILE_PATH = 'config/config.yaml'
config = yaml.load(open(CONFIG_FILE_PATH, 'r', encoding='utf-8'), Loader=yaml.FullLoader)


def get_label_rank_df(model_id: int,
                      dissimilarity_id: int,
                      batch_size: int = 100,
                      relabel_lambda: float = 0.069,
                      random_ratio: float = 0.5) -> pd.DataFrame:
    """
    Get a DataFrame containing label rankings based on the specified parameters.
    Calls the AL_RANKINGS stored procedure

    Parameters:
    - model_id (int): The identifier of the model.
    - dissimilarity_id (int): The identifier for dissimilarity images.
    - batch_size (int, optional): The total batch size for label ranking (default is 100).
    - relabel_lambda (float, optional): The relabeling lambda parameter (default is 0.069).
    - random_ratio (float, optional): The ratio of random images in the batch (default is 0.5).

    Returns:
    - pd.DataFrame: A DataFrame containing label rankings.
    """
    # Calculate the number of random and dissimilarity images based on the ratio
    batch_size_r = int(random_ratio * batch_size)
    batch_size_d = batch_size - batch_size_r

    # Call stored procedure for label ranking with dissimilarity scores
    args = OrderedDict([
        ("M_ID", model_id),
        ("D_ID", dissimilarity_id),
        ("BATCH_SIZE", batch_size_d),
        ("RELABEL_LAMBDA", relabel_lambda),
    ])
    d_df = get_data_stored_procedure(sp="AL_RANKINGS", args=args)

    # Call stored procedure for label ranking with D_ID = 0 (represents random images)
    args['D_ID'] = 0
    args['BATCH_SIZE'] = batch_size_r
    r_df = get_data_stored_procedure(sp="AL_RANKINGS", args=args)

    # Concatenate the results into a single DataFrame (may have duplicates)
    full_df = pd.concat([d_df, r_df])

    return full_df


def validate_args(sp_name: str, args: Optional[OrderedDict[str, Any]]) -> None:
    """
    Validate the arguments for a stored procedure against the specified types.

    Parameters:
    - sp_name (str): The name of the stored procedure.
    - args (dict, optional): A dictionary containing the arguments for the stored procedure.

    Raises:
    - ValueError: If any argument has an unexpected type based on the specified types in SP_ARGS_TYPE_MAPPING.
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
    - server_args (dict, optional): A dictionary containing connection parameters for the server.
        Expected keys: 'server', 'database', 'username', 'password'.
        Default values are taken from the `config` dictionary.
    Returns:
    - Tuple: a tuple containing the strings for server, database, username, and password
    """
    # if new parameters are passed, load from dict or use config file
    server = server_args.get('server', config['server'])
    database = server_args.get('database', config['database'])
    user = server_args.get('username', config['username'])
    password = server_args.get('password', config['password'])

    return server, database, user, password


def get_data_stored_procedure(sp: str = None,
                args: Optional[OrderedDict[str, Any]] = None,
                server_args: Optional[Dict[str, str]] = {}) -> pd.DataFrame:
    """
    Execute a stored procedure and return the result as a Pandas DataFrame.

    Parameters:
    - sp (str): The name of the stored procedure to execute.
    - args (dict, optional): A dictionary containing parameters for the stored procedure.
        Expected keys: Must match those defined in SP_ARGS_TYPE_MAPPING.
        Values should be either int, float, or str, depending on the stored procedure.
    - server_args (dict, optional): A dictionary containing connection parameters for the server.
        Expected keys: 'server', 'database', 'username', 'password'.
        Default values are taken from the `config` dictionary.
    Returns:
    - DataFrame: The result of the stored procedure in a Pandas DataFrame format.
    """
    # Validate args dictionary
    if args is not None:
        validate_args(sp_name=sp, args=args)

    # Get authentication strings
    server, database, user, password = get_server_arguments(server_args=server_args)

    # set up a connection to Azure
    with pymssql.connect(server, user, password, database) as conn:
        # set up a cursor object
        with conn.cursor(as_dict=True) as cursor:
            # gather variables for the stored procedure
            arg_tuples = tuple([args[k] for k in args.keys()])
            # execute stored procedure
            cursor.callproc(sp, arg_tuples)
            # Fetch the results into a Pandas DataFrame
            df = pd.DataFrame(cursor.fetchall())

    return df


def execute_stored_procedure(sp_name: str, file_path: Optional[str],
                             server_args: Optional[Dict[str, str]] = {}) -> None:
    """
    Execute a stored procedure by its name. It defaults to a preset file location
    based on the string using SP_FILE_NAMES but this is overriden by file_path.

    Parameters:
    - sp_name (str): The name of the stored procedure to execute.
    - file_path (str): The path to the SQL script file containing the stored procedure.
    - server_args (dict, optional): A dictionary containing connection parameters for the server.
        Expected keys: 'server', 'database', 'username', 'password'.
        Default values are taken from the `config` dictionary.

    Returns:
    - None: The function does not return any value.

    Raises:
    - pymssql.Error: If there is an error during the execution of the stored procedure.
    - FileNotFoundError: If the specified SQL script file is not found.
    """

    # Get authentication strings
    server, database, user, password = get_server_arguments(server_args=server_args)

    # Validate the stored_procedure
    validate_args(sp_name=sp, args=None)
    # get file_path
    if sp_name in SP_FILE_NAMES:
        if file_path:
            print(f"Using custom file to create procedure {sp_name}: {file_path}")
        else:
            print(f"Using preset file to create procedure {sp_name}: {SP_FILE_NAMES[sp_name]}")
    else:
        warnings.warn(f"No record of procedure called {sp_name}. Running wild, buckoo!")

    try:
        # Read the SQL script from the file
        with open(file_path, 'r') as file:
            sql_script = file.read()

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


def run_sql_query(query: str,
                  server_args: Optional[Dict[str, str]] = {}) -> Union[pd.DataFrame, None]:
    """
    Execute a SQL query on a database and return the results as a Pandas DataFrame.

    Parameters:
    - query (str): The SQL query to execute.
    - server_args (dict, optional): A dictionary containing server connection parameters.
                                   Defaults to an empty dictionary.

    Returns:
    - Union[pd.DataFrame, None]: A Pandas DataFrame containing the query results,
                                 or None if the query didn't return any results.

    Raises:
    - pymssql.Error: If there is an error during the execution of the SQL query.
    """
    # Get authentication strings
    server, database, user, password = get_server_arguments(server_args=server_args)

    # Establish a connection to the database
    with pymssql.connect(server, user, password, database) as conn:
        # Create a cursor
        with conn.cursor() as cursor:
            # Execute the stored procedure using the content of the SQL script
            cursor.execute(query)
            # Fetch the results into a Pandas DataFrame
            results = cursor.fetchall()
            # Commit the changes (if needed)
            conn.commit()
    if results:
        df = pd.DataFrame(results)
        return df

    return
