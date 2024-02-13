"""
Deploying model in Azure Machine Learning (Azure ML) as real-time endpoint.

Initializing the Azure ML workspace,
loading pre-trained model, registering model in Azure ML workspace,
defining the inference configuration.
"""
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azure.ai.ml.entities import ManagedOnlineEndpoint

# Initialize Azure ML workspace
workspace = Workspace.from_config()

# Register model
registered_model = Model.register(workspace=workspace,
                                  model_name='basemodel',
                                  model_path='./ml-workflow/model_ckpt/')

# Get scoring file
SCORING_FILE = 'score.py'

# Define inference configuration
inference_config = InferenceConfig(entry_script=SCORING_FILE,
                                   runtime='python',
                                   conda_file='environment.yml',
                                   source_directory='./')

# Define an endpoint name
ENDPOINT_NAME = "basemodel-endpoint"

# Define online deployment configuration
endpoint = ManagedOnlineEndpoint(
    model = registered_model,
    name = ENDPOINT_NAME,
    description="Endpoint for base model",
    auth_mode="key"
)
