"""
This module loads configuration settings from a YAML file located at 'config/config.yaml'.

Global Variables:
    - CONFIG_FILE_PATH (str): Absolute path to the configuration file.
    - CONFIG (dict): Dictionary containing configuration settings loaded from the configuration file.
"""
import os
import yaml

CONFIG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../config/config.yaml'))
CONFIG = yaml.load(open(CONFIG_FILE_PATH, 'r', encoding='utf-8'), Loader=yaml.FullLoader) # pylint: disable=consider-using-with
