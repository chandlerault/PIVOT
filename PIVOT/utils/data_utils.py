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
from io import BytesIO
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
import pymssql
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

from PIL import Image
import pandas as pd
import numpy as np


CONFIG_FILE_PATH = 'config/config.yaml'
config = yaml.load(open(CONFIG_FILE_PATH, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

def save_df(df, blob_name, container_name=None, connection_string=None):
    """
    Saves a dataframe to blob storage.

    Args:
        df (pandas.DataFrame): DataFrame containing blob_names, labels, and metrics.
        blob_name (str): Name to save the blob to. Must end in pd, parquet, or csv.
        container_name (str, optional): Name of container for the blob. Defaults to metrics_container in config.
        connection_string (str, optional): Connection string for azure. Defaults to the one saved in config file.
    """
    try:
        if container_name is None:
            container_name =  config['metrics_container']
        if connection_string is None:
            connection_string = config['connection_string']

        # Check extension
        valid_extensions = ['parquet', 'pq', 'csv']
        if blob_name.lower().split('.')[-1] not in valid_extensions:
            print('s',blob_name.lower().split('.')[-1])
            raise ValueError(f'blob_name: {blob_name} does not have a valid extension. Must be in: {valid_extensions}')
         # Azure clients initialization
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create the container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # table = pa.Table.from_pandas(df)
        if blob_name.lower().split('.')[-1] == 'csv':
            # Save the DataFrame as CSV format
            with BytesIO() as buf:
                df.to_csv(buf, index=False)
                buf.seek(0)
                blob_client.upload_blob(buf.read(), overwrite=True)
        else:
            table = pa.Table.from_pandas(df)
            with BytesIO() as buf:
                pq.write_table(table, buf)
                buf.seek(0)
                blob_client.upload_blob(buf.read(), overwrite=True)

    except ValueError as e:
        print(str(e))
        return None

def get_df(blob_name, container_name, connection_string=None):
    """
    Retrieve metrics data from Azure Blob Storage and convert it to a Pandas DataFrame.
    Args:
        df (pandas.DataFrame): DataFrame containing blob_names, labels, and metrics.
        blob_name (str): Name to save the blob to. Must end in pd, parquet, or csv.
        container_name (str): Name of container for the blob.
        connection_string (str, optional): Connection string for azure. Defaults to the one saved in config file.
    Returns:
        pandas.DataFrame:   A DataFrame containing the data in the blob.
    """
    try:
        # Make sure the extension is correct
        valid_extensions = ['parquet', 'pq', 'csv']
        if blob_name.lower().split('.')[-1] not in valid_extensions:
            raise ValueError(f'metric_blob: {blob_name} does not have a valid extension.')
        
        if connection_string is None:
            connection_string = config['connection_string']
        file_extension = blob_name.split(".")[-1].lower()  # Get the file extension

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        data = blob_client.download_blob()
        file_data = data.readall()

        if file_extension == "parquet":
            table = pq.read_table(BytesIO(file_data))
            return table.to_pandas()
            
        elif file_extension == "csv":
            return pd.read_csv(BytesIO(file_data))
            
        else:
            print(f"Unsupported file format: {file_extension}")
    except ResourceNotFoundError as _:
        print(f"Blob '{blob_name}' not found.")
        return None
    except ValueError as e:
        print(str(e))
        return None

# TODO: add catches for non-existent containers
def get_metrics():
    """
    Retrieve metrics data from Azure Blob Storage and convert it to a Pandas DataFrame.

    Returns:
        pandas.DataFrame:   A DataFrame containing the metrics data.
                            Columns: blob_name, class, metrics (can vary)
    """
    return get_df(blob_name=config['metrics_blob'], container_name=config['metrics_container'])
        # # Retrieve config information for accessing Azure
        # connection_string = config['connection_string']
        # container_name =  config['metrics_container']
        # # TODO: make this handle all data types (e.g. csv, tsv, etc)
        # blob_name = config['metrics_blob'] # could make this a list of metric files and combine them.

        # # Make sure the extension is correct
        # valid_extensions = ['parquet', 'pq']
        # if blob_name.lower().split('.')[-1] not in valid_extensions:
        #     print(blob_name.lower().split('.')[-1])
        #     raise ValueError(f'metric_blob: {blob_name} does not have a valid extension.')
        
        # # Set up Azure clients
        # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # container_client = blob_service_client.get_container_client(container_name)

        # # Load blob data
        # blob_data = container_client.get_blob_client(blob_name).download_blob()

        # parquet_bytes = blob_data.readall()

        # table = pq.read_table(BytesIO(parquet_bytes))

        # return table.to_pandas()

def get_images(filepath):
    """
    Retrieve images from Azure Blob Storage based on a filepath. Takes ~0.25s per image.

    Args:
        filepath (pandas.DataFrame): DataFrame containing filtered blob names.

    Returns:
        PIL.Image.Image: A  PIL.Image.Image object  with image retrieved from a filepath.
    """

    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")


    connection_string = config['connection_string2'] # TODO: eventually make this one connection string.
    container_name =  config['image_container']

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filepath)
    blob_exists = blob_client.exists()

    if blob_exists:
        pass
    else:
        print(f"Blob '{filepath}' does not exist in the container.")

    blob_data = blob_client.download_blob()
    image_content = blob_data.readall()

    image = Image.open(BytesIO(image_content))

    return image        


# def get_images(filtered_df, batch_size=1):
#     """
#     Retrieve images from Azure Blob Storage based on a filtered DataFrame. Takes ~0.25s per image.

#     Args:
#         filtered_df (pandas.DataFrame): DataFrame containing filtered blob names.
#         batch_size (int, optional): Batch size for processing blobs. Defaults to 1.

#     Yields:
#         List<PIL.Image.Image>: A list of PIL.Image.Image object of size batch_size 
#             representing images retrieved from a blob.
#     """

#     if not isinstance(filtered_df, pd.DataFrame):
#         raise TypeError("filtered_df must be panda dataframe")
#     if not isinstance(batch_size, int):
#         raise TypeError("batch_size must be an integer")

#     blobs = filtered_df['blob_name'].tolist()
#     for i in range(0,len(blobs),batch_size):
#         images = []
#         batch = blobs[i:i + batch_size]

#         # Process one batch at a time
#         for blob_name in batch:
#             # Get a BlobClient for the current blob
#             connection_string = config['connection_string2'] # TODO: eventually make this one connection string.
#             container_name =  config['image_container']
#             print(connection_string)
#             # Set up Azure clients
#             blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#             blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
#             blob_exists = blob_client.exists()

#             if blob_exists:
#                 pass
#             else:
#                 print(f"Blob '{blob_name}' does not exist in the container.")

#             blob_data = blob_client.download_blob()
#             image_content = blob_data.readall()

#             if image_content.startswith(b'\x89PNG\r\n\x1a\n'):
#                 print("The blob contains a valid PNG image.")
#             else:
#                 print("The blob does not appear to be a valid PNG image.")

#             image = Image.open(BytesIO(image_content))
#             # yield image # for single image
#             images.append([image, filtered_df.iloc[i,:]])
#             print(images)
#             # images.append([np.random.uniform(size=(124,124,1)), filtered_df.iloc[i,:]])
        
#         yield images

def create_user(user_info):
    id = insert_data('users', user_info)
    return id
    
def get_user(email):
    user = select('users', {'email':email}, ['u_id']) 
    if len(user) > 0:
        return user[0]['u_id']
    else:
        return None

    
    pass # None, uid

def get_models():
    return select_distinct('models', ['model_name','m_id'])

def get_dissimilarities():
    return select_distinct('dissimilarity', ['name','d_id'])

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
        server = config['server']
        database = config['database']
        user = config['db_user']
        password = config['db_password']
        

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
                    id = cursor.fetchone()[0]

                    return id
    except Exception as e:
        print("Error:", str(e))

def select(table_name, conditions, columns=['*']):
    try:
        server = config['server']
        database = config['database']
        user = config['db_user']
        password = config['db_password']
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

def select_distinct(table_name, columns):
    try:
        server = config['server']
        database = config['database']
        user = config['db_user']
        password = config['db_password']
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
