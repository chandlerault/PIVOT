"""
Make predictions via requests to REST endpoint of deployed (stream endpoint) model.
"""
import pandas as pd
from azureml.core import Workspace, Dataset, Experiment
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig
from azureml.data import OutputFileDatasetConfig
from azureml.core import Workspace
from azureml.core.webservice import Webservice
from azureml.core.authentication import InteractiveLoginAuthentication
import requests
import json
import numpy as np
from json import JSONEncoder
import imageio
import cv2

def preprocess_input(image, fixed_size=128):
    '''
    '''
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

# with open('cloud_urls_10000.json', 'r') as file:
#     cloud_urls = json.load(file)

# cloud_urls = cloud_urls[:100]

def predict(df, scoring_uri, api_key):
    """
    Afhwefuhr ef
    
    Args:
        pd.DataFrame: A DataFrame containing image metadata.
            Columns: IMAGE_ID, BLOB_FILEPATH
    """
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")
    
    cloud_urls = df.cloud_urls.values
    data = []
    for c_url in cloud_urls:
        data.append(preprocess_input(np.expand_dims(imageio.v2.imread(c_url), axis=-1)))
    
    # This takes longer for some reason
    # data = df.cloud_urls.apply(lambda x: preprocess_input(np.expand_dims(imageio.v2.imread(x), axis=-1))).values

    numpyArrayOne = [i.reshape((128, 128)).tolist() for i in data]
    numpyData = {"input_data": numpyArrayOne}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder) 

    # Convert the payload to JSON
    json_payload = encodedNumpyData

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # TODO: figure out how to get 'azureml-model-deployment'
    headers = {'Content-Type':'application/json',
               'Authorization':('Bearer '+ api_key),
               'azureml-model-deployment': 'pivot-basemodel' }

    # Make the prediction request
    response = requests.post(scoring_uri, data=json_payload, headers=headers)
    print(response)

    # Check the response status code
    if response.status_code == 200:
        result = response.json()
        print(result)
    else:
        print("Prediction request failed with status code:", response.status_code)
        print(response.text)
    
    df = pd.DataFrame({'i_id': df.IMAGE_ID.values, 
                         'probs': result})
    
    return df
