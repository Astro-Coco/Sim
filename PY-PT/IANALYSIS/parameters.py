import json
import os
from interactive_analysis import path_to_PYPT


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_values_and_paths(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    flat_dict = flatten_dict(data)
    values_and_paths = [(value, path) for path, value in flat_dict.items() if (type(value) != list)]
    
    values = []
    for value in values_and_paths:
        value += (value[1].split('_'),)
        values.append(value)

    return values


if __name__ == '__main__':
    path = os.path.join(path_to_PYPT, 'MOT', 'config_vol_2024.json')
    # Example usage:
    json_file_path = 'path/to/your/file.json'  # Replace with the actual path to your JSON file
    result = get_values_and_paths(path)
    print(result)
    # Displaying the result
    for value, name, path in result:
        print(f"Value: {value}, Path: {' -> '.join(path)}, name: {name}")
