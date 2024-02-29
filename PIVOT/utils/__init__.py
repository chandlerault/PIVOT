"""
This module loads configuration settings from a YAML file located at 'config/config.yaml'.

Global Variables:
    - CONFIG_FILE_PATH (str): Absolute path to the configuration file.
    - CONFIG (dict): Dictionary containing configuration settings loaded from the configuration file.
"""
import os
import yaml
import time

CONFIG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../config/config.yaml'))
CONFIG = yaml.load(open(CONFIG_FILE_PATH, 'r', encoding='utf-8'), Loader=yaml.FullLoader) # pylint: disable=consider-using-with
CONFIG_LAST_CHANGED = os.path.getmtime(CONFIG_FILE_PATH)

CONFIG_ARGS = [
    "connection_string",
    "image_container",
    "server",
    "database",
    "db_user",
    "db_password",
    'subscription_id',
    'resource_group',
    'workspace_name',
    'experiment_name',
    'api_key',
    'model_name',
    'endpoint_name',
    'deployment_name'
]

def load_config(file_path=None, interval=10):
    """
    Load config file from the file path. This allows us to change the
    file within the app without restarting it.

    Parameters:
        file_path: the file path of the file. Default is that stored in __init__
        interval: the amount of time to wait to load file if config is empty.
    Return:
        dict: a dictionary of all config arguments.
    """
    global CONFIG_LAST_CHANGED
    global CONFIG_FILE_PATH
    global CONFIG
    # update file path if needed
    if file_path:
        CONFIG_FILE_PATH = file_path
    else:
        file_path = CONFIG_FILE_PATH
    current_modified = os.path.getmtime(file_path)

    while current_modified != CONFIG_LAST_CHANGED:
        current_modified = os.path.getmtime(file_path)
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as file:
            CONFIG = yaml.load(file, Loader=yaml.FullLoader) # pylint: disable=consider-using-with
        print("File changed. Constants updated.")
        if CONFIG is not None:
            CONFIG_LAST_CHANGED = current_modified
            return CONFIG
        else:
            print("Config file is empty. Waiting to get new data.")
            if interval is None:
                raise ValueError("Config file is empty! Please modify!")
            else:
                time.sleep(interval)
    return CONFIG
