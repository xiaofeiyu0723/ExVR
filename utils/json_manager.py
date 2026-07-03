import json

from utils.paths import app_path

def load_json(file_path):
    # Load a JSON file and return the data.
    path = app_path(file_path)
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return {}


def get_data(data, key):
    # Retrieve a specific value from loaded JSON data using a key.
    return data.get(key, None)


def update_data(data, key, value):
    # Update the loaded JSON data with a new key-value pair.
    data[key] = value


def save_json(data, file_path):
    # Save the modified data back to the JSON file.
    path = app_path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving data: {e}")
