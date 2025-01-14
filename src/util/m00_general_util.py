import os
import numpy as np
import json

def get_files(path, endswith = ".csv"):
    files_in_path = os.listdir(path)
    csv_files = filter(lambda f: f.endswith(endswith), files_in_path)
    return list(csv_files)

def merge_json_files(json_files, output_file = "combined.json"):
    result = list()
    for file in json_files:
        with open(file, 'r') as infile:
            result.extend(json.load(infile))

    with open(output_file, 'w') as output_file:
        json.dump(result, output_file, indent=4)

# TODO unused (consider deleting)
def calculate_averages(d):
    if isinstance(d, dict):
        # Recursively calculate for nested dictionaries
        return {k: calculate_averages(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        # Calculate the mean for NumPy arrays
        return np.mean(d).tolist()  # Convert the mean to list (in this case, a scalar)
    else:
        return d