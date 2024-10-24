import json

def load_json(file_path):
    # Load a JSON file and return the data.
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}


def get_data(data, key):
    # Retrieve a specific value from loaded JSON data using a key.
    return data.get(key, None)


def update_data(data, key, value):
    # Update the loaded JSON data with a new key-value pair.
    data[key] = value


def save_json(data, file_path):
    # Save the modified data back to the JSON file.
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error saving data: {e}")
