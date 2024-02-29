"""
Contains the logic about how to run the model and
read the input data submitted by the deployment executor.
Each model deployment has a scoring script (and any other required dependencies).

Intended input for run() to be './scoring_data.json'.
"""
import os
import logging
import joblib
import requests
from utils import load_config

def init():
    """
    This function is called when the container is initialized/started,
    typically after create/update of the deployment.
    """
    global MODEL
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # Path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/"
    )

    # deserialize the model file back into a sklearn model
    MODEL = joblib.load(model_path)
    logging.info("Init complete")


def run(json_payload):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.

    Parameters:
        json_payload (json): JSON file containing three sample data to be used for testing.
    """
    CONFIG = load_config()
    
    scoring_uri = f'https://{endpoint_name}.westus2.inference.ml.azure.com/score'.format(
        endpoint_name=CONFIG['endpoint_name'])

    api_key = CONFIG['api_key']

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    headers = {'Content-Type':'application/json',
               'Authorization':('Bearer '+ api_key),
               'azureml-model-deployment': CONFIG['deployment_name']}

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

    return result
