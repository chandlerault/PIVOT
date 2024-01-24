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


def get_images(filtered_df, batch_size=1):
    """
    Retrieve images from Azure Blob Storage based on a filtered DataFrame. Takes ~0.25s per image.

    Args:
        filtered_df (pandas.DataFrame): DataFrame containing filtered blob names.
        batch_size (int, optional): Batch size for processing blobs. Defaults to 1.

    Yields:
        List<PIL.Image.Image>: A list of PIL.Image.Image object of size batch_size 
            representing images retrieved from a blob.
    """

    if not isinstance(filtered_df, pd.DataFrame):
        raise TypeError("filtered_df must be panda dataframe")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")

    blobs = filtered_df['blob_name'].tolist()
    for i in range(0,len(blobs),batch_size):
        images = []
        batch = blobs[i:i + batch_size]

        # Process one batch at a time
        for blob_name in batch:
            # Get a BlobClient for the current blob
            connection_string = config['connection_string2'] # TODO: eventually make this one connection string.
            container_name =  config['image_container']
            print(connection_string)
            # Set up Azure clients
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            blob_exists = blob_client.exists()

            if blob_exists:
                pass
            else:
                print(f"Blob '{blob_name}' does not exist in the container.")

            blob_data = blob_client.download_blob()
            image_content = blob_data.readall()

            if image_content.startswith(b'\x89PNG\r\n\x1a\n'):
                print("The blob contains a valid PNG image.")
            else:
                print("The blob does not appear to be a valid PNG image.")

            image = Image.open(BytesIO(image_content))
            # yield image # for single image
            images.append([image, filtered_df.iloc[i,:]])
            print(images)
            # images.append([np.random.uniform(size=(124,124,1)), filtered_df.iloc[i,:]])
        
        yield images