import json
import sys
from tqdm.auto import trange, tqdm
import concurrent.futures

import numpy as np
import pandas as pd
import cv2
import imageio
import yaml

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

import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel, model_from_json
from tensorflow.keras.optimizers import Adam
from keras.callbacks import Callback

import mlflow
import mlflow.keras

from azureml.core import Workspace, Model as AzureMLWorkspaceModel


def load_local_model():
    # Load the model architecture from JSON file
    json_file_path = './ml-workflow/model_ckpt/model-cnn-v1-b3.json'
    with open(json_file_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights from H5 file
    h5_file_path = './ml-workflow/model_ckpt/model-cnn-v1-b3.h5'
    loaded_model.load_weights(h5_file_path)