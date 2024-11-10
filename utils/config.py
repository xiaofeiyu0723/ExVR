from copy import deepcopy

from utils.json_manager import load_json, save_json

def setup_config():
    config = load_json("./settings/config.json")
    return config

def save_config(config):
    config = deepcopy(config)
    save_json(config, "./settings/config.json")
