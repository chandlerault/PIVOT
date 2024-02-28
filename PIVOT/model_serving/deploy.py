"""
Deploying model in Azure Machine Learning (Azure ML) as real-time endpoint.

Initializing the Azure ML workspace,
loading pre-trained model, registering model in Azure ML workspace,
defining the inference configuration.
"""
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

# Initialize Azure ML workspace
workspace = Workspace.from_config()

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Change!
MODEL_NAME = 'basemodel'

# Register model
registered_model = Model.register(workspace=workspace,
                                  model_name=MODEL_NAME,
                                  model_path='./ml-workflow/model_ckpt/')

# Get scoring file
SCORING_FILE = 'score.py'

# Define inference configuration
inference_config = InferenceConfig(entry_script=SCORING_FILE,
                                   runtime='python',
                                   conda_file='environment.yml',
                                   source_directory='./')

# Change!
RUN_ID = '3ce8d4e5-6343-4d44-be1e-e9c99b3df968'
MODEL_PATH = 'azureml://locations/westus2/workspaces/83932448-b7da-4483-8ae6-293b4223feb9/models/basemodel/versions/1'


# Online endpoint without scoring file?
ml_client.models.create_or_update(
    Model(
        path=f"azureml://jobs/{RUN_ID}/outputs/artifacts/{MODEL_PATH}",
        name=MODEL_NAME,
        type=AssetTypes.MLFLOW_MODEL
    )
)

# Define an endpoint name
ENDPOINT_NAME = "basemodel-endpoint"

# Define online deployment configuration
endpoint = ManagedOnlineEndpoint(
    model = registered_model,
    name = ENDPOINT_NAME,
    description="Endpoint for base model",
    auth_mode="key"
)

# Creating endpoint
ml_client.begin_create_or_update(endpoint)
