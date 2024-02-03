"""
(Still WIP)

Deploying model in Azure Machine Learning (Azure ML). 

Initializing the Azure ML workspace,
loading pre-trained model,
registering the model in Azure ML workspace,
defining the inference configuration (scoring file is score.py) and ACI deployment configuration,
deploying the model as a web service to ACI.
"""

from azureml.core.webservice import AciWebservice, Webservice
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from keras.models import model_from_json

def deploy(model_name, scoring_file):
    # Initialize Azure ML workspace
    ws = Workspace.from_config()

    # Register model
    registered_model = Model.register(workspace=ws,
                                      model_name='basemodel-endpoint',
                                      model_path='./ml-workflow/model_ckpt/')

    # Access registered model
    registered_model = Model(ws, 'basemodel_endpoint')

    # Define inference configuration
    inference_config = InferenceConfig(entry_script=scoring_file,
                                       runtime='python',
                                       conda_file='environment.yml',  
                                       source_directory='./')

    # Define ACI deployment configuration
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

    # Deploy the model to ACI
    service_name = "myaciservice"
    service = Model.deploy(workspace=ws,
                          name=service_name,
                          models=[registered_model],
                          inference_config=inference_config,
                          deployment_config=aci_config,
                          deployment_target=None)

    service.wait_for_deployment(show_output=True)
