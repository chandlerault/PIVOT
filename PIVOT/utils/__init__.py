import yaml
import os

CONFIG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../config/config.yaml'))
CONFIG = yaml.load(open(CONFIG_FILE_PATH, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
