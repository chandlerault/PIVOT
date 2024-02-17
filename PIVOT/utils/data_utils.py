"""
This module provides functions for interacting with databases, Azure Blob Storage, and image preprocessing.

Configuration:
    The module reads configuration settings from 'config/config.yaml' for Azure Blob Storage and Database access.

Functions:
    - get_blob_bytes(blob_path)
    - insert_data(table_name, data) 
    - select(table_name, conditions, columns=['*'])
    - select_distinct(table_name, columns)
    - get_status()
    - preprocess_input(image, fixed_size=128)

Usage:
    - Import this module and use its functions to retrieve data from Azure Blob Storage or query database.
    - Ensure that the 'config/config.yaml' file contains the necessary configuration settings.
"""
import pymssql
from azure.storage.blob import BlobServiceClient
import numpy as np
import cv2
from utils import load_config


def get_blob_bytes(blob_path):
    """
    Retrieve images from Azure Blob Storage based on a filepath. Takes ~0.25s per image.

    Args:
        filepath (pandas.DataFrame): DataFrame containing filtered blob names.

    Returns:
        PIL.Image.Image: A  PIL.Image.Image object  with image retrieved from a filepath.
    """
    try:

        if not isinstance(blob_path, str):
            raise TypeError("filepath must be a string")

        CONFIG = load_config()
        connection_string = CONFIG['connection_string']
        container_name = CONFIG['image_container']

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
        blob_exists = blob_client.exists()

        if blob_exists:
            pass
        else:
            print(f"Blob '{blob_path}' does not exist in the container.")
            return None

        blob_data = blob_client.download_blob()
        content = blob_data.readall()
        blob_service_client.close()
        return content

    except TypeError as e:
        print("Error:", str(e))
        return None

def insert_data(table_name, data):
    """
    Inserts data into the table with the corresponding table_name.

    Args:
        table_name (str): The name of the table in the database 
            that the data should be inserted into.
        data (dict/list<dict>): A single dict or list of dicts. Each key represents a 
            column in the table and each value is a value to be inserted. 
    
    Returns:
        id (int, list<int>): id's of inserted data.
    """
    CONFIG = load_config()
    try:
        # Define your database connection parameters
        with pymssql.connect(CONFIG['server'], CONFIG['db_user'],
                             CONFIG['db_password'], CONFIG['database']) as conn:
            with conn.cursor() as cursor:
                if isinstance(data, list) and len(data) > 0:
                    # Generate the INSERT statement dynamically based on the dictionary keys
                    columns = ', '.join(data[0].keys())  # Assuming all dictionaries have the same keys
                    placeholders = ', '.join('%(' + key + ')s' for key in data[0].keys())
                    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    # Execute the INSERT statement using executemany and the list of dictionaries
                    cursor.executemany(insert_query, data)
                    conn.commit()
                if isinstance(data, dict):
                    # Generate the INSERT statement dynamically based on the dictionary keys
                    columns = ', '.join(data.keys())  # Assuming all dictionaries have the same keys
                    placeholders = ', '.join('%(' + key + ')s' for key in data.keys())
                    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    # Execute the INSERT statement using executemany and the list of dictionaries
                    cursor.execute(insert_query, data)
                    conn.commit()
                    # Optionally, retrieve the identity of the inserted row
                    cursor.execute("SELECT SCOPE_IDENTITY()")
                    ids = cursor.fetchone()[0]
                    return ids
    except pymssql.InterfaceError as ie: # pylint: disable=no-member
        print("InterfaceError:", str(ie))
    except pymssql.DatabaseError as de: # pylint: disable=no-member
        print("DatabaseError:", str(de))
    return None
def select(table_name, conditions, columns=['*']): # pylint: disable=dangerous-default-value
    """
    Executes a SELECT query on a specified table in the database.

    Args:
        table_name (str): The name of the table to select from.
        conditions (dict): A dictionary representing the conditions to apply to the query.
            Keys are column names and values are the corresponding values to match.
        columns (list, optional): A list of column names to retrieve. Defaults to ['*'].

    Returns:
        list: A list of dictionaries, each representing a row of the result set.
            Keys of each dictionary are column names and values are the corresponding values in the row.
            An empty list is returned if there are no results or if an error occurs.
    """
    CONFIG = load_config()
    try:
        if CONFIG is not None:
            server = CONFIG['server']
            database = CONFIG['database']
            user = CONFIG['db_user']
            password = CONFIG['db_password']
            with pymssql.connect(server=server, database=database, user=user, password=password) as conn: # pylint: disable=no-member
                with conn.cursor() as cursor:
                    query = f"SELECT {', '.join(columns)} FROM {table_name}"

                    if conditions:
                        condition_strings = []
                        for column, value in conditions.items():
                            condition_strings.append(f"{column} = '{value}'")
                        where_clause = " WHERE " + " AND ".join(condition_strings)
                        query += where_clause

                    cursor.execute(query)
                    rows = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    result = [dict(zip(column_names, row)) for row in rows]
                    return result
    except pymssql.InterfaceError as ie: # pylint: disable=no-member
        print("InterfaceError:", str(ie))
    except pymssql.DatabaseError as de: # pylint: disable=no-member
        print("DatabaseError:", str(de))
    return []

def select_distinct(table_name, columns):
    """
    Executes a SELECT DISTINCT query on a specified table in the database.

    Args:
        table_name (str): The name of the table to select from.
        columns (list): A list of column names to retrieve distinct values from.

    Returns:
        list: A list of dictionaries, each representing a row of the result set.
            Keys of each dictionary are column names and values are the corresponding values in the row.
            An empty list is returned if there are no results or if an error occurs.
    """
    CONFIG = load_config()
    try:
        if CONFIG is not None:
            server = CONFIG['server']
            database = CONFIG['database']
            user = CONFIG['db_user']
            password = CONFIG['db_password']
            with pymssql.connect(server=server, database=database, user=user, password=password) as conn: # pylint: disable=no-member
                with conn.cursor() as cursor:
                    query = f"SELECT DISTINCT {', '.join(columns)} FROM {table_name}"
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    result = [dict(zip(column_names, row)) for row in rows]
                    return result
        else:
            pass
    except pymssql.InterfaceError as ie: # pylint: disable=no-member
        print("InterfaceError:", str(ie))
    except pymssql.DatabaseError as de: # pylint: disable=no-member
        print("DatabaseError:", str(de))
    return []

def get_status():
    """
    Checks if the specified database is online.

    Returns:
        bool: True if the database is online, False otherwise.
    """
    CONFIG = load_config()
    try:
        server = CONFIG['server']
        database = CONFIG['database']
        user = CONFIG['db_user']
        password = CONFIG['db_password']
        with pymssql.connect(server=server, database=database, user=user, password=password) as conn: # pylint: disable=no-member
            with conn.cursor() as cursor:
                cursor.execute("SELECT state_desc FROM sys.databases WHERE name = %s", (database,))
                row = cursor.fetchone()
                if row and row[0] == 'ONLINE':
                    return True
                return False
    except pymssql.InterfaceError as ie: # pylint: disable=no-member
        print("InterfaceError:", str(ie))
    except pymssql.DatabaseError as de: # pylint: disable=no-member
        print("DatabaseError:", str(de))
    return False

def preprocess_input(image, fixed_size=128):
    """
    Preprocesses an input image by resizing it to a fixed size and normalizing pixel values.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        fixed_size (int): Target size for the image after resizing. Default is 128.

    Returns:
        numpy.ndarray: Preprocessed image with the specified fixed size.
    """
    image_size = image.shape[:2]
    ratio = float(fixed_size)/max(image_size)
    new_size = tuple([int(x*ratio) for x in image_size]) # pylint: disable=consider-using-generator
    img = cv2.resize(image, (new_size[1], new_size[0])) # pylint: disable=no-member
    delta_w = fixed_size - new_size[1]
    delta_h = fixed_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    ri = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) # pylint: disable=no-member
    gimg = np.array(ri).reshape(fixed_size,fixed_size,1) # pylint: disable=too-many-function-args
    img_n = cv2.normalize(gimg, gimg, 0, 255, cv2.NORM_MINMAX) # pylint: disable=no-member
    return img_n
