import os
import numpy as np

def get_csv_files(path):
    files_in_path = os.listdir(path)
    csv_files = filter(lambda f: f.endswith(".csv"), files_in_path)
    return list(csv_files)

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