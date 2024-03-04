"""
This file contains utility functions for making model predictions via requests
to the REST endpoint of deployed (stream endpoint) model on Azure ML.

Classes:
    - NumpyArrayEncoder: Helper function to serialize a np.ndarray into a JSON String.
        Sourced from https://pynative.com/python-serialize-numpy-ndarray-into-json/

Functions:
    - get_model_info: Retrieves the name of the deployed Azure ML model endpoint given a model ID.
    - preprocess_input: Preprocesses an input image by resizing it to a fixed size and normalizing pixel values.
    - predict: Calls the REST endpoint of the specified model to make predictions for all specified images.
    - get_predictions: Retrieves and formats model predictions.
    - load_local_model: Loads locally saved model as keras model.
"""
import json
from json import JSONEncoder
import requests

import numpy as np
import pandas as pd
import imageio
from tensorflow.keras.models import model_from_json
import data_utils as du
from azureml.core import Workspace, Experiment

from utils import load_config

def get_model_info(m_id):
    """
    Retrieves the name of the deployed Azure ML model endpoint given a model ID.

    Parameters:
        m_id (int): The identifier of a specific model to predict with.

    Returns:
        endpoint_name (str): The name of the model endpoint to be called to make predictions.
    """
    CONFIG = load_config() # pylint: disable=invalid-name

    workspace = Workspace.create(name=CONFIG['workspace_name'],
                      subscription_id=CONFIG['subscription_id'],
                      resource_group=CONFIG["resource_group"],
                      location='westus2'
                     )

    experiment_name = CONFIG['experiment_name']
    experiment = Experiment(workspace=workspace, name=experiment_name)

    for i in experiment.get_runs():
        if i==m_id:
            endpoint_name = CONFIG['endpoint_name']

    return endpoint_name

class NumpyArrayEncoder(JSONEncoder):
    """
    Helper function to serialize a np.ndarray into a JSON String.
    Sourced from: https://pynative.com/python-serialize-numpy-ndarray-into-json/

    Modifies the JSONEncoder scope by overriding the default() method to accept np.ndarray.
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)

def predict(df, m_id):
    """
    Calls the REST endpoint of the specified model to make predictions for all specified images.
    Input to be formatted as per the output of sql_utils.get_images_to_predict().

    Parameters:
        df (pd.DataFrame): DataFrame containing image metadata.
            Columns: IMAGE_ID (int): The image ID.
                     BLOB_FILEPATH (str): The filepath of the image.
                     cloud_urls (str): The full URL of the image's blob storage endpoint.
        m_id (int): The model ID.

    Returns:
        df (pd.DataFrame): DataFrame containing model predictions for each image.
            Columns: i_id (int): The image ID.
                     probs: A list of 10 class probabilities.
    """
    CONFIG = load_config() # pylint: disable=invalid-name

    endpoint_name = get_model_info(m_id)

    scoring_uri = f'https://{endpoint_name}.westus2.inference.ml.azure.com/score'.format(
        endpoint_name=endpoint_name)
    api_key = CONFIG['api_key']

    if not api_key:
        raise KeyError("A key should be provided to invoke the endpoint")
    cloud_urls = df.cloud_urls.values
    data = []
    for c_url in cloud_urls:
        data.append(du.preprocess_input(np.expand_dims(imageio.v2.imread(c_url), axis=-1)))

    # This takes longer for some reason
    # data = df.cloud_urls.apply(lambda x: preprocess_input(np.expand_dims(imageio.v2.imread(x), axis=-1))).values

    data_dic = {"input_data": [i.reshape((128, 128)).tolist() for i in data]}
    json_payload = json.dumps(data_dic, cls=NumpyArrayEncoder)

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    headers = {'Content-Type':'application/json',
               'Authorization':('Bearer '+ api_key),
               'azureml-model-deployment': CONFIG['deployment_name']}

    # Make the prediction request
    response = requests.post(scoring_uri,
                             data=json_payload,
                             headers=headers,
                             timeout=10)

    # Check the response status code
    if response.status_code == 200:
        result = response.json()
    else:
        print("Prediction request failed with status code:", response.status_code)
        print(response.text)

    df = pd.DataFrame({'i_id': df.I_ID.values,
                       'probs': result})
    return df

def get_predictions(df, m_id):
    """
    Retrieves and formats model predictions.

    Parameters:
        df (pd.DataFrame): DataFrame containing image metadata.
            Columns: IMAGE_ID (int): The image ID.
                     BLOB_FILEPATH (str): The filepath of the image.
        m_id (int): The model ID.

    Returns:
        out (list): A list of dictionaries to be inputted into the predictions table.
    """
    cloud_url = 'https://ifcb.blob.core.windows.net/naames/{filepath}'
    df['cloud_urls'] = df.filepath.apply(lambda x: cloud_url.format(filepath=x))
    preds = predict(df, m_id)

    classes = ['Chloro',
          'Cilliate',
          'Crypto',
          'Diatom',
          'Dictyo',
          'Dinoflagellate',
          'Eugleno',
          'Other',
          'Prymnesio',
          'Unidentifiable']

    preds['class_prob'] = preds.probs.apply(lambda x: x[pd.Series(x).idxmax()])
    preds['predlabel'] = preds.probs.apply(lambda x: classes[pd.Series(x).idxmax()])
    preds['m_id'] = [m_id] * len(preds)
    preds = preds.drop(['probs'], axis=1)
    out = preds.to_dict(orient='records')

    return out

def load_local_model(json_file_path, h5_file_path):
    """
    Loads locally saved model as Tensorflow keras model.

    Parameters:
        json_file_path (str): Path to local .json model file.
        h5_file_path (str): Path to local .h5 model weights file.

    Returns:
        loaded_model (tf.keras.Model): Loaded keras model.
    """
    # Load the model architecture from JSON file
    with open(json_file_path, 'r', encoding="utf-8") as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights from H5 file
    loaded_model.load_weights(h5_file_path)

    return loaded_model
