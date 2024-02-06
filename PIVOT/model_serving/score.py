"""
This isn't being used at all and I will probably delete. 

Only needed in the case of batch deployment.
Contains the logic about how to run the model and 
read the input data submitted by the batch deployment executor.
Each model deployment has a scoring script
(and any other required dependencies).

Used as the inference file in Dockerfile and as scoring file during deployment. 
"""
from azureml.core.model import Model
import mlflow
import mlflow.keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import tensorflow as tf
from keras.callbacks import Callback
import imageio
import numpy as np
import cv2
import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from tqdm.auto import trange
from tqdm.auto import tqdm
import yaml
import concurrent.futures
import json

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

def preprocess_batch(order_index, batch_urls):
    idxs = []
    start_index = 0
    data = []
    for i, c_url in enumerate(tqdm(batch_urls, leave=False)):
        data.append(preprocess_input(np.expand_dims(imageio.v2.imread(c_url), axis=-1)))
        idxs.append(order_index*batch_size + i + start_index)

    result = (idxs, data)
    return result

def init():
    global loaded_model
    model_path = Model.get_model_path('ifcb-image-class')
    with open(model_path + '/model-cnn-v1-b3.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_path + '/model-cnn-v1-b3.h5')

def run(cloud_urls):   
    # Specify the batch size and number of workers
    batch_size = 320
    num_workers = 24  

    # Split URLs into batches
    start_index = 0
    url_batches = [cloud_urls[i:i + batch_size] for i in range(start_index, len(cloud_urls), batch_size)]

    # Create an empty list to store probabilities
    all_probabilities = []

    # Create a list to store the order of URLs
    url_order = []

    try:
        # Use concurrent.futures for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_order_index = {
                executor.submit(preprocess_batch, i, batch_urls): i 
                for i, batch_urls in enumerate(url_batches)
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_order_index), total=len(url_batches)):
                order_index = future_to_order_index[future]
                try:
                    # Get batch of images
                    url_idxs, img_data = future.result()
                    url_order.extend(url_idxs)

                    # predict images
                    batch_probabilities = loaded_model.predict(np.array(img_data), verbose=0)
                    all_probabilities.extend(batch_probabilities)
                except Exception as e:
                    print(f"Error processing batch {order_index}: {e}")

    except Exception as e:
        print(f"Error: {e}")

    except KeyboardInterrupt:
        print("Process interrupted by user.")

    sorted_indices = np.argsort(url_order)
    ordered_all_probs = np.vstack(all_probabilities)[sorted_indices]
    all_probs = np.vstack(all_probabilities)
    all_probs_df = pd.DataFrame(all_probs)
    all_probs_df.index = url_order

    return all_probs_df
    