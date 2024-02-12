"""
Make predictions via requests to REST endpoint of deployed (stream endpoint) model.
"""
import json
import yaml
import requests

import numpy as np
import pandas as pd
import cv2
import imageio

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azureml.core import Workspace, Model, Experiment

from tensorflow.keras.models import Model as KerasModel

def get_model_info(m_id):
    """
    """
    # Remove this once I can actually access config through import
    with open("../model_serving/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ml_client = MLClient(subscription_id=config['subscription_id'],
                         resource_group=config["resource_group"],
                         workspace_name=config['workspace_name'],
                         credential=DefaultAzureCredential())

    ws = Workspace.from_config('../model_serving/config.json')
    experiment_name = config['experiment_name']
    experiment = Experiment(workspace=ws, name=experiment_name)
    # run_name = 'basemodel'
    # for i in experiment.get_runs():
        # how get the endpoint name from the model ID? idk yet
    endpoint_name = 'basemodel_endpoint'
    return endpoint_name

def preprocess_input(image, fixed_size=128):
    """
    """
    image_size = image.shape[:2] 
    ratio = float(fixed_size)/max(image_size)
    new_size = tuple([int(x*ratio) for x in image_size])
    img = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = fixed_size - new_size[1]
    delta_h = fixed_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    ri = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # gray_image = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
    gray_image = ri
    gimg = np.array(gray_image).reshape(fixed_size, fixed_size, 1)
    img_n = cv2.normalize(gimg, gimg, 0, 255, cv2.NORM_MINMAX)
    return img_n

class NumpyArrayEncoder(JSONEncoder):
    """
    Add link!
    """
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def predict(df, m_id):
    """
    Afhwefuhr ef

    Args:
        df: A pd.DataFrame containing image metadata.
            Columns: IMAGE_ID, BLOB_FILEPATH, cloud_urls
    """
    # Remove this once I can actually access config through import 
    with open("../model_serving/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    endpoint_name = get_model_info(m_id)

    scoring_uri = config['scoring_uri'].format(endpoint_name='basemodel-endpoint')
    api_key = config['api_key']

    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")
    cloud_urls = df.cloud_urls.values
    data = []
    for c_url in cloud_urls:
        data.append(preprocess_input(np.expand_dims(imageio.v2.imread(c_url), axis=-1)))

    # This takes longer for some reason
    # data = df.cloud_urls.apply(lambda x: preprocess_input(np.expand_dims(imageio.v2.imread(x), axis=-1))).values

    data_dic = {"input_data": [i.reshape((128, 128)).tolist() for i in data]}
    json_payload = json.dumps(data_dic, cls=NumpyArrayEncoder)

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # TODO: figure out how to get 'azureml-model-deployment'
    headers = {'Content-Type':'application/json',
               'Authorization':('Bearer '+ api_key),
               'azureml-model-deployment': 'pivot-basemodel' }
    # Make the prediction request
    response = requests.post(scoring_uri, data=json_payload, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        result = response.json()
    else:
        print("Prediction request failed with status code:", response.status_code)
        print(response.text)
    df = pd.DataFrame({'i_id': df.IMAGE_ID.values, 
                         'probs': result})
    return df

def get_predictions(df, m_id):
    """
    Gets and formats predictions. 
    df: columns [IMAGE_ID, BLOB_FILEPATH]
    """
    with open("../model_serving/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    df['cloud_urls'] = df.BLOB_FILEPATH.apply(lambda x: config['cloud_url'].format(filepath=x))
    preds = predict(df, m_id)

    classes = ['Chloro',
          'Cilliate',
          'Crypto',
          'Diatom',
          'Dictyo',
          'Dinoflagellate',
          'Eugleno',
          'Other',
          'Prymnesio',
          'Unidentifiable']

    preds['class_prob'] = preds.probs.apply(lambda x: x[pd.Series(x).idxmax()])
    preds['predlabel'] = preds.probs.apply(lambda x: classes[pd.Series(x).idxmax()])
    preds['m_id'] = [m_id] * len(preds)
    preds = preds.drop(['probs'], axis=1)
    ls = preds.to_dict(orient='records')

    return ls
