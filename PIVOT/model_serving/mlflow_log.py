"""
This file logs a new model on MLflow,
and integrates with Azure ML for model tracking and versioning.

Logs model using MLFlow, sets path of all artifacts to be saved,
connects to deployed model on Azure ML.
"""
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import mlflow
import mlflow.keras

from utils import load_config
from utils import model_utils as mu

CONFIG = load_config()

# Initialize Azure ML Client
ml_client = MLClient(subscription_id=CONFIG['subscription_id'],
                         resource_group=CONFIG['resource_group'],
                         workspace_name=CONFIG['workspace_name'],
                         credential=DefaultAzureCredential())

# Load the model architecture from JSON file
# Change based on model to log
loaded_model = mu.load_local_model('../model_serving/model-cnn-v1-b3.json',
                                   '../model_serving/model-cnn-v1-b3.h5')

# Connect to azure ML
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

# For local testing
# remote_server_uri = "http://localhost:5000"

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(CONFIG['experiment_name'])

RUN_NAME = 'run10'

with mlflow.start_run(run_name=RUN_NAME):
    # Log entire model with MLflow
    # Input example if needed
    mlflow.keras.log_model(loaded_model,
                           artifact_path="model-artifacts",
                           registered_model_name='ifcb-image-class')
                           # input_example=input_example)

    # Log model weights
    mlflow.log_artifact("model_weights.h5",
                        artifact_path='model-artifacts')
    