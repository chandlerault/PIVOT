"""
"""
import json
import sys
from tqdm.auto import trange, tqdm
import concurrent.futures
import yaml
import os

import numpy as np
import pandas as pd
import cv2
import imageio

from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model as AzureMLModel,
    Environment,
    CodeConfiguration,
)
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core import Workspace, Model, Experiment, Run
from azureml.core.model import InferenceConfig

import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel, model_from_json
from tensorflow.keras.optimizers import Adam
from keras.callbacks import Callback

import mlflow
import mlflow.keras

from model_serving import prediction

with open("../model_serving/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

scoring_uri = config['scoring_uri'].format(endpoint_name='basemodel-endpoint')
api_key = config['api_key']

def get_predictions(df, scoring_uri=scoring_uri, api_key=api_key):
    """
    df: columns [IMAGE_ID, BLOB_FILEPATH]
    scoring_uri:
    api_key:
    """
    with open("../model_serving/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    df['cloud_urls'] = df.BLOB_FILEPATH.apply(lambda x: config['cloud_url'].format(filepath=x))
    
    preds = prediction.predict(df, scoring_uri, api_key)
    
    classes = pd.DataFrame({'label': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                    'class': ['Chloro',
                                'Cilliate',
                                'Crypto',
                                'Diatom',
                                'Dictyo',
                                'Dinoflagellate',
                                'Eugleno',
                                'Other',
                                'Prymnesio',
                                np.nan]})
    
    preds['probs'] = preds.probs.apply(lambda x: {classes['class'].values[i]: x[i] for i in range(9)})
    
    preds['predlabel'] = preds.probs.apply(lambda x: list(x.keys())[pd.Series(x.values()).idxmax()])
    
    return preds
