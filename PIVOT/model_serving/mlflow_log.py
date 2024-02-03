"""
This is for logging a new model on MLflow, 
an integrating with Azure ML for model tracking and versioning.

Logs model using MLFlow, sets path of all artifacts to be saved,
connects to deployed model on Azure ML.
"""
import json
import sys
from tqdm.auto import trange, tqdm
import concurrent.futures
import yaml

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

import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel, model_from_json
from tensorflow.keras.optimizers import Adam
from keras.callbacks import Callback

import mlflow
import mlflow.keras

from azureml.core import Workspace, Model as AzureMLWorkspaceModel

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def log_model(model, ml_client, run_name):
    """
    Blah dhwkedhewdhewiudhw
    """
    # Change based on model to log
    # loaded_model = make any optional changes
    
    # Connect to azure ML
    # ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

    # For local testing
    # remote_server_uri = "http://localhost:5000" 

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(config['experiment_name'])

    with mlflow.start_run(run_name=run_name): 

        # # Initialize Azure ML workspace
        # ws = Workspace.from_config()

        # Log entire model with MLflow
        # Input example if needed
        mlflow.keras.log_model(loaded_model,
                               artifact_path="model-artifacts",
                               registered_model_name='ifcb-image-class')
                               # input_example=input_example)
            
        # Log model weights
        model_weights_path = "model_weights.h5"
        mlflow.log_artifact(model_weights_path,
                            artifact_path='model-artifacts')
    