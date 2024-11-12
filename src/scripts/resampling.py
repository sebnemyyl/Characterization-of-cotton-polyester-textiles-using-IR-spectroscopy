import os
import util.m05_resampling as resampling
import util.m00_general_util as util

print(os.getcwd())
input_dir = "temp/resampling"
output_dir = "temp/bootstrap50"
csv_files = util.get_csv_files(input_dir)
for file in csv_files:
    csv_file = os.path.join(input_dir, file)
    print(csv_file)
    resampling.resample_csv_file(csv_file, cotton=50, type="bootstrap", output_dir=output_dir)
