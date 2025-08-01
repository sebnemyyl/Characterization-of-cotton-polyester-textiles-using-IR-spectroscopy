import os
import sys
# Needed for Python to find the util modules
sys.path.insert(0, "src")
sys.path.insert(0, "../..")

import util.m05_resampling as resampling
import util.m00_general_util as util



os.chdir("../..")
print(os.getcwd())

input_dir = "../input/clean_csv/resampling"
output_dir = "../input"
csv_files = util.get_files(input_dir)
for file in csv_files:
    csv_file = os.path.join(input_dir, file)
    print(csv_file)
    resampling.resample_csv_file(csv_file, cotton=50, type="bootstrap", output_dir=output_dir)
