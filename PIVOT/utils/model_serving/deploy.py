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

from utils import load_config

CONFIG = load_config()

# Initialize Azure ML workspace
ws = Workspace.create(name=CONFIG['workspace_name'],
                      subscription_id=CONFIG['subscription_id'],
                      resource_group=CONFIG["resource_group"],
                      location='westus2'
                     )

ml_client = MLClient(subscription_id=CONFIG['subscription_id'],
                         resource_group=CONFIG['resource_group'],
                         workspace_name=CONFIG['workspace_name'],
                         credential=DefaultAzureCredential())

# Register model
registered_model = Model.register(workspace=ws,
                                  model_name=CONFIG['model_name'],
                                  model_path='./ml-workflow/model_ckpt/')

# Get path to scoring file
SCORING_FILE = 'score.py'

# Define inference configuration
inference_config = InferenceConfig(entry_script=SCORING_FILE,
                                   runtime='python',
                                   conda_file='environment.yml',
                                   source_directory='./')

# # Online endpoint without scoring file
# # Delete i think
# ml_client.models.create_or_update(
#     Model(
#         name=CONFIG['model_name'],
#         run_id=CONFIG['run_id'],
#         type=AssetTypes.MLFLOW_MODEL
#     )
# )

# Define online deployment configuration
endpoint = ManagedOnlineEndpoint(
    model = registered_model,
    name = CONFIG['endpoint_name'],
    description="Endpoint for base model",
    auth_mode="key"
)

# Creating endpoint
ml_client.begin_create_or_update(endpoint, deployment_name=CONFIG['deployment_name'])
