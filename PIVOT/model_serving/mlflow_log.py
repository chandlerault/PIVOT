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
    