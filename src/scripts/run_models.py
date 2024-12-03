import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "..")

import os
import json
import util.m00_general_util as util

def test_model():
    print("Test model!")
    rmse = 0.3
    r_squared = 0.6
    return (rmse, r_squared)

def run_model(csv_file, model):
    if(model == "test"):
        return test_model()
    elif(model == "random_forest"):
        print("Call random forest here")
        return (0.6, 0.8)
    else:
        raise ValueError(f"{model} not supported!")

def get_baseline_corr_type(csv_file):
    name_without_ending = os.path.splitext(csv_file)[0]
    parts = name_without_ending.split("_")
    return parts[-1]

print(os.getcwd())
input_dir = "temp/nir"
output_file = "temp/model_output.json"
models = ["test",  "random_forest"]
csv_files = util.get_csv_files(input_dir)

output = []
for csv_file in csv_files:
    for model in models:
        result = {}
        baseline_corr_type = get_baseline_corr_type(csv_file)
        (rmse, r_squared) = run_model(csv_file, model)
        result.update(
            model = model,
            baseline_corr = baseline_corr_type,
            rmse = rmse,
            r_squared = r_squared,
        )
        output.append(result)

print(output)
with open(output_file, "w") as outfile:
    json.dump(output, outfile)