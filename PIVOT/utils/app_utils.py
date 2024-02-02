from utils import data_utils
from PIL import Image
import numpy as np
import cv2
import time


def create_user(user_info):
    """
    Create a user from a dictionary of user_info and returns new user id.

    Args: 
        user_info (dict): Dict with keys associated to columns in user table.

    Returns:
        u_id (int): The u_id for the new user
    """

    u_id = data_utils.insert_data('users', user_info)
    return u_id
    
def get_user(email):
    """
    Gets the u_id associated with the email if it exists. Otherwise returns None.

    Args: 
        email (str): String of a user's email.

    Returns:
        u_id (int/None): The u_id for the user with the email. Otherwise None.
    """
    user = data_utils.select('users', {'email':email}, ['u_id']) 
    if user and len(user) > 0:
        return user[0]['u_id']
    else:
        return None
    
def get_models():
    """
    Gets all the unique models and their m_id's.

    Returns:
        model_list (list<Dict>): A list of dictionaries with keys model_name and m_id.
    """
    return data_utils.select_distinct('models', ['model_name','m_id'])

def get_dissimilarities():
    """
    Gets all the unique dissimilarity metrics and their d_id's.

    Returns:
        model_list (list<Dict>): A list of dictionaries with keys name and d_id.
    """
    return data_utils.select_distinct('dissimilarity', ['name','d_id'])

def get_image(file_path):
    """
    Gets the image associated with the file_path. Otherwise returns None.

    Args: 
        email (str): String of a user's email.

    Returns:
        u_id (int/None): The u_id for the user with the email. Otherwise None.
    """
     
    image_contents = data_utils.get_blob_bytes(file_path)
    if image_contents.startswith(b'\x89PNG\r\n\x1a\n'):
                print("The blob contains a valid PNG image.")
    else:
        print("The blob does not appear to be a valid PNG image.")
    im = np.frombuffer(image_contents, dtype=np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_UNCHANGED)
    im = data_utils.preprocess_input(im)
    # image = Image.open(BytesIO(image_contents))
    return Image.fromarray(im.reshape(im.shape[:2]))

def await_connection(max_time=60, step=5):
    for _ in range(max_time//step):
          if data_utils.get_status():
               return True
          time.sleep(step)
    return False

def insert_label(labels):
     data_utils.insert_data('labels', labels) 