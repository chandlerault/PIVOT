"""
This file contains utility functions for making model predictions via requests
to the REST endpoint of deployed (stream endpoint) model on Azure ML.

Classes:
    - NumpyArrayEncoder:

Functions:
    - get_model_info: Retrieves the name of the deployed Azure ML model endpoint given a model ID.
    - preprocess_input: Preprocesses an input image by resizing it to a fixed size and normalizing pixel values.
    - predict: Calls the REST endpoint of the specified model to make predictions for all specified images.
    - get_predictions: Retrieves and formats model predictions.
"""
import json
from json import JSONEncoder
import yaml
import requests

import numpy as np
import pandas as pd
import cv2
import imageio

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azureml.core import Workspace, Experiment

def get_model_info(m_id):
    """
    Retrieves the name of the deployed Azure ML model endpoint given a model ID.

    Parameters:
        m_id (int): The identifier of a specific model to predict with.

    Returns:
        endpoint_name (str): The name of the model endpoint to be called to make predictions.
    """
    # Remove this once I can actually access config through import
    with open("../model_serving/config.yaml", encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    ml_client = MLClient(subscription_id=config['subscription_id'],
                         resource_group=config["resource_group"],
                         workspace_name=config['workspace_name'],
                         credential=DefaultAzureCredential())

    workspace = Workspace.from_config('../model_serving/config.json')
    experiment_name = config['experiment_name']
    experiment = Experiment(workspace=workspace, name=experiment_name)
    # run_name = 'basemodel'
    # for i in experiment.get_runs():
        # how get the endpoint name from the model ID? idk yet
    endpoint_name = 'basemodel-endpoint'
    return endpoint_name

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
    new_size = tuple(int(x * ratio) for x in image_size)
    img = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = fixed_size - new_size[1]
    delta_h = fixed_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    resized = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # gray_image = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
    gray_image = resized
    gimg = np.array(gray_image).reshape((fixed_size, fixed_size, 1))
    img_n = cv2.normalize(gimg, gimg, 0, 255, cv2.NORM_MINMAX)
    return img_n

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
    # Remove this once I can actually access config through import
    with open("../model_serving/config.yaml", encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint_name = get_model_info(m_id)

    scoring_uri = config['scoring_uri'].format(endpoint_name=endpoint_name)
    api_key = config['api_key']

    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")
    cloud_urls = df.cloud_urls.values
    data = []
    for c_url in cloud_urls:
        data.append(preprocess_input(np.expand_dims(imageio.v2.imread(c_url), axis=-1)))

    # This takes longer for some reason
    # data = df.cloud_urls.apply(lambda x: preprocess_input(np.expand_dims(imageio.v2.imread(x), axis=-1))).values

    data_dic = {"input_data": [i.reshape((128, 128)).tolist() for i in data]}
    json_payload = json.dumps(data_dic, cls=NumpyArrayEncoder)

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # TODO: figure out how to get 'azureml-model-deployment'
    headers = {'Content-Type':'application/json',
               'Authorization':('Bearer '+ api_key),
               'azureml-model-deployment': 'pivot-basemodel'}
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
    with open("../model_serving/config.yaml", encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    df['cloud_urls'] = df.filepath.apply(lambda x: config['cloud_url'].format(filepath=x))
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
