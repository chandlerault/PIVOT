"""
This module provides functions for ingesting initial data from blob filepaths into a database.
It includes a function to ingest data from blob filepaths, check if the paths exist, and insert them into the IMAGES table in the database.

Functions:
    - initial_ingestion: Ingest data from blob filepaths into the IMAGES table, checking for existence and inserting into the database.
    - bulk_insert_data: Insert data into a specified table in the database, in batches if necessary.
"""
from tqdm.auto import tqdm, trange
from typing import Any, Tuple, Union
import pandas as pd
from utils import CONFIG

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import concurrent.futures
import pymssql


def initial_ingestion(image_filepaths: list = None, parallelize: bool = True, batch_size: int = 1000,
                      return_val_paths: bool = False, **kwargs) -> Union[None, list[str]]:
    """
    This function takes in data from an initial set of blob filepaths.
    It will check whether the filepath exists and then inserts it into the
    IMAGES table.

    NOTE: the dataframe must have column labeled "image_path".
    These paths shouldn't include the container name or "Https:..."

    Parameters:
        image_filepaths (list): a list of blob filepaths for images.
            Default is extracted from the 5.5 M images in NAAMES-predicted-labels-model-cnn-v1-b3.csv
        parallelize (bool): Whether or not to use concurrent.futures to speed up
        batch_size (int): number of images to insert at a time.
            Default is 320.
        return_val_paths (bool): Whether or not to return a list of all valid images.
    Returns:
        None
    """

    # Gather database connection parameters:
    server = CONFIG['server']
    database = CONFIG['database']
    user = CONFIG['db_user']
    password = CONFIG['db_password']

    # get database connection:
    conn = pymssql.connect(server, user, password, database)

    if image_filepaths is None:
        container_name = 'naames'
        account_name = 'ifcb'
        url_prefix = f"https://{account_name}.blob.core.windows.net/{container_name}/"
        csv_path = url_prefix + 'NAAMES-predicted-labels-model-cnn-v1-b3.csv'

        df = pd.read_csv(csv_path)
        image_filepaths = [
            f"{df['image_path'][i].split('NAAMES/')[1]}"
            for i in trange(df.shape[0]//500)
        ]
        print(f"Got all image paths from {csv_path}.")
    # check whether path exists
    verified_paths = []
    connection_string = CONFIG['connection_string']
    container_name = CONFIG['image_container']

    # get the blob
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    if parallelize:
        # Specify the number of workers for multiprocessing
        num_workers = kwargs.get("num_workers", 24) # Adjust based on your system's capabilities

        # Split URLs into batches
        start_index = 0
        url_batches = [image_filepaths[i:i + batch_size] for i in range(start_index, len(image_filepaths), batch_size)]

        # Create an empty list to store whether or not it exists
        # is_exists = []
        # Create a list to store the order of URLs
        # url_order = []

        # Define a function for parallel preprocessing
        def preprocess_batch(order_index, batch_urls):
            idxs = []
            data = []
            for i, c_url in enumerate(tqdm(batch_urls, leave=False)):
                data.append(blob_service_client.get_blob_client(container=container_name, blob=c_url).exists())
                idxs.append(order_index * batch_size + i + start_index)

            result = (idxs, data)
            return result
        try:
            # Use concurrent.futures for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_order_index = {
                    executor.submit(preprocess_batch, i, batch_urls): i
                    for i, batch_urls in enumerate(url_batches)
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_order_index),
                                   total=len(url_batches)):
                    order_index = future_to_order_index[future]
                    try:
                        # Check each batch to see if it exists
                        url_idxs, exist_data = future.result()
                        # is_exists.extend(exist_data)
                        # url_order.extend(url_idxs)

                        local_exists = []
                        for i,e in enumerate(exist_data):
                            if not(e):
                                print(f"Following image path doesn't exist:{image_filepaths[url_idxs[i]]}")
                                continue
                            if return_val_paths:
                                verified_paths.append({"filepath": image_filepaths[url_idxs[i]]})
                            # keep track of local batch_size
                            local_exists.append({"filepath": image_filepaths[url_idxs[i]]})

                        # Insert batch to IMAGES table
                        print(f"Inserting Batch {order_index}:")
                        bulk_insert_data(table_name="IMAGES", data=local_exists, conn=conn)
                        print(f"\tInserted {len(local_exists)} images")
                    except Exception as e:
                        print(f"Error processing batch {order_index}: {e}")
                    except KeyboardInterrupt:
                        print("Process interrupted by user.")
        except KeyboardInterrupt:
            print("Process interrupted by user.")

    else:
        local_exists = []
        order_index = 0
        for c_url in tqdm(image_filepaths):
            bb = blob_service_client.get_blob_client(container=container_name, blob=c_url)
            if bb.exists():
                if return_val_paths:
                    verified_paths.append({"filepath": c_url})
                local_exists.append({"filepath": c_url})
            else:
                print(f"Following image path doesn't exist:{c_url}")
            # insert images if we're at the batch size
            if len(local_exists) == batch_size:
                print(f"Inserting Batch {order_index}:")
                bulk_insert_data(table_name="IMAGES", data=local_exists, conn=conn)
                print(f"\tInserted {len(local_exists)} images")
                # reset local_exists and increment order_index
                order_index += 1
                local_exists = []

    # Insert last set of images
    if local_exists and len(local_exists) > 0:
        print(f"Inserting Last Batch ({order_index}):")
        bulk_insert_data(table_name="IMAGES", data=local_exists,conn=conn)
        print(f"\tInserted {len(local_exists)} images")

    # close the connection
    conn.close()

    if return_val_paths:
        return verified_paths

    return None


def bulk_insert_data(table_name: str, data: list[dict[str, Any]],
                     conn=None, max_batch: int = 1000) -> None:
    """
    Inserts data into the table with the corresponding table_name.

    Args:
        table_name (str): The name of the table in the database that the data should be inserted into.
        data (dict/list<dict>): A single dict or list of dicts.
            Each key represents a column in the table and each value is a value to be inserted.
        conn (object): the pymssql connection object.
        max_batch (int): the max number of rows that can be inserted at once.
            Default is 1000, the max possible value for Pymssql execute
    Returns:
        id (int, list<int>): id's of inserted data.
    """

    def generate_query_args(data: list) -> Tuple[str, Tuple[str]]:
        """
        Get the complete query and the arguments for a batch of data.
        """
        columns = ', '.join(data[0].keys())  # Assuming all dictionaries have the same keys
        values = [[row[col] for col in row] for row in data]
        values_placeholder = ', '.join([f'({", ".join(["%s" for _ in row])})' for row in values])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES "
        insert_query += values_placeholder
        flattened_values = tuple([value for row in values for value in row])

        return insert_query, flattened_values

    def chunks(lst:list, n=max_batch):
        """
        Break up full data into chunks to make it easier to push.
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    if conn is None:
        # Define your database connection parameters
        server = CONFIG['server']
        database = CONFIG['database']
        user = CONFIG['db_user']
        password = CONFIG['db_password']

        with pymssql.connect(server, user, password, database) as conn:
            with conn.cursor() as cursor:
                if isinstance(data, list) and len(data) > 0:
                    for data_chunk in tqdm(chunks(data)):
                        insert_query, insert_args = generate_query_args(data_chunk)
                        # Execute the INSERT statement using execute and the flattened tuples
                        cursor.execute(insert_query, insert_args)
                        conn.commit()
    else:
        with conn.cursor() as cursor:
            if isinstance(data, list) and len(data) > 0:
                for data_chunk in tqdm(chunks(data)):
                    insert_query, insert_args = generate_query_args(data_chunk)
                    # Execute the INSERT statement using execute and the flattened tuples
                    cursor.execute(insert_query, insert_args)
                    conn.commit()

    return
