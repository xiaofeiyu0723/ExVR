from copy import deepcopy
from utils.json_manager import load_json, save_json

def setup_data():
    data = load_json("./settings/data.json")
    default_data = deepcopy(data)
    return data, default_data

def save_data(data):
    data=deepcopy(data)
    save_json(data, "./settings/data.json")