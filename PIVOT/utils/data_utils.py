"""
Data Utilities Module for Blob Storage Data Retrieval

This module provides functions for retrieving data from Azure Blob Storage, including metrics data in Parquet format
and images in various formats. It uses the Azure SDK, PyArrow for Parquet handling, and PIL (Pillow) for image processing.

Configuration:
    The module reads configuration settings from 'config/config.yaml' for Azure Blob Storage access.

Functions:
    - save_df(df, blob_name, container_name=None, connection_string=None): Save a df to blob_name in container_name. container_name defaults to metrics_container
    - get_df(blob_name, container_name, connection_string=None): Retrieves a dataframe from blob storage that has blob_name in container container_name
    - get_metrics(): Retrieve metrics data from Azure Blob Storage and convert it to a Pandas DataFrame.
    - get_images(filtered_df, batch_size=1): Retrieve images from Azure Blob Storage based on a filtered DataFrame returns an iterator.

Usage:
    - Import this module and use its functions to retrieve data from Azure Blob Storage.
    - Ensure that the 'config/config.yaml' file contains the necessary configuration settings.

Example:
    from data_utils import get_metrics, get_images

    metrics_df = get_metrics()
    filtered_df = metrics_df[metrics_df['confidence_interval']<.1]
    image_iterator = get_images(filtered_df, 10)
    for image_set in image_iterator:
        for image in image_set:
            # show(image)
"""
import yaml
import pymssql
from azure.storage.blob import BlobServiceClient
from PIL import Image
import pandas as pd
import numpy as np
import cv2


from utils import CONFIG

def get_blob_bytes(blob_path):
    """
    Retrieve images from Azure Blob Storage based on a filepath. Takes ~0.25s per image.

    Args:
        filepath (pandas.DataFrame): DataFrame containing filtered blob names.

    Returns:
        PIL.Image.Image: A  PIL.Image.Image object  with image retrieved from a filepath.
    """

    if not isinstance(blob_path, str):
        raise TypeError("filepath must be a string")

    connection_string = CONFIG['connection_string'] # TODO: eventually make this one connection string.
    container_name = CONFIG['image_container']

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
    blob_exists = blob_client.exists()

    if blob_exists:
        pass
    else:
        print(f"Blob '{blob_path}' does not exist in the container.")

    blob_data = blob_client.download_blob()
    content = blob_data.readall()
    blob_service_client.close()
    return content        

def insert_data(table_name, data):
    """
    Inserts data into the table with the corresponding table_name.

    Args:
        table_name (str): The name of the table in the database that the data should be inserted into.
        data (dict/list<dict>): A single dict or list of dicts. Each key represents a column in the table and each value is a value to be inserted.
    
    Returns:
        id (int, list<int>): id's of inserted data.
    """
    try:
        # Define your database connection parameters
        server = CONFIG['server']
        database = CONFIG['database']
        user = CONFIG['db_user']
        password = CONFIG['db_password']
        

        with pymssql.connect(server, user, password, database) as conn:
            with conn.cursor() as cursor:
                if isinstance(data, list) and len(data) > 0:
                    # Generate the INSERT statement dynamically based on the dictionary keys
                    columns = ', '.join(data[0].keys())  # Assuming all dictionaries have the same keys
                    placeholders = ', '.join('%(' + key + ')s' for key in data[0].keys())
                    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    # Execute the INSERT statement using executemany and the list of dictionaries
                    cursor.executemany(insert_query, data)
                    conn.commit()

                    # TODO: dump id into temporary tables in batch call and return id's

                elif isinstance(data, dict):
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
    except Exception as e:
        print("Error:", str(e))

def select(table_name, conditions, columns=['*']):
    try:
        server = CONFIG['server']
        database = CONFIG['database']
        user = CONFIG['db_user']
        password = CONFIG['db_password']
        with pymssql.connect(server=server, database=database, user=user, password=password) as conn:
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
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def select_distinct(table_name, columns):
    try:
        server = CONFIG['server']
        database = CONFIG['database']
        user = CONFIG['db_user']
        password = CONFIG['db_password']
        with pymssql.connect(server=server, database=database, user=user, password=password) as conn:
            with conn.cursor() as cursor:
                query = f"SELECT DISTINCT {', '.join(columns)} FROM {table_name}"
                cursor.execute(query)
                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                result = [dict(zip(column_names, row)) for row in rows]
                return result
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def get_status():
    try:
        server = CONFIG['server']
        database = CONFIG['database']
        user = CONFIG['db_user']
        password = CONFIG['db_password']
        with pymssql.connect(server=server, database=database, user=user, password=password) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT state_desc FROM sys.databases WHERE name = %s", (database,))
                row = cursor.fetchone()
                if row and row[0] == 'ONLINE':
                    return True
                else:
                    return False
    except Exception as e:
        print(f"Error: {str(e)}")
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
    new_size = tuple([int(x*ratio) for x in image_size])
    img = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = fixed_size - new_size[1]
    delta_h = fixed_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    ri = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #print(ri.shape)
    # gray_image = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
    gimg = np.array(ri).reshape(fixed_size,fixed_size,1)
    #gimg = np.array(ri).reshape(fixed_size,fixed_size,1)
    img_n = cv2.normalize(gimg, gimg, 0, 255, cv2.NORM_MINMAX)
    return img_n