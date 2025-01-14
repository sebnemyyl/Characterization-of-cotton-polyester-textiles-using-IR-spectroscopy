import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import importlib.util
import pickle
import util.m00_general_util as util
from scipy import stats
# resample = importlib.util.spec_from_file_location("resampling_methods", "util\\05_jackknife_bootstrap_resampling.py")
# resampling_methods = importlib.util.module_from_spec(resample)
# resample.loader.exec_module(resampling_methods)

#my_path = "../temp/all"

pkl_path = f'../temp/bootstrap/bootstrap_spectra_treated_snv_50_cotton_nir.csv.pkl'
loaded_dict = pickle.load(open(pkl_path,"rb"))

# Dictionary to store the minimum sample size where p-value crosses 0.05 for each file and wavenumber
crossing_sample_size_by_file_and_wavenumber = defaultdict(list)


def detect_pvalue_crossing(final_dict):
    p_values_by_spectrum_and_sample_size = defaultdict(lambda: defaultdict(list))
    averaged_p_values_by_spectrum_and_sample_size = defaultdict(dict)

    for spectra, specimen_sample_size in final_dict.items():
        for specimen, bootstrap in specimen_sample_size.items():
            baseline_data = bootstrap[19]

            for sample_size in range(2, 19):
                current_data = bootstrap[sample_size]
                F_statistic, p_value = stats.f_oneway(current_data, baseline_data)
                #statistic = alexandergovern(current_data, baseline_data)
                p_values_by_spectrum_and_sample_size[spectra][sample_size].append(p_value)
                #p_values_by_spectrum_and_sample_size[spectra][sample_size].append(statistic.pvalue)

    for spectra, sample_size_dict in p_values_by_spectrum_and_sample_size.items():
        for sample_size, pval_list in sample_size_dict.items():
            avg_p_value = np.mean(pval_list)
            averaged_p_values_by_spectrum_and_sample_size[spectra][sample_size] = avg_p_value

    return averaged_p_values_by_spectrum_and_sample_size



input_dir = "../temp/bootstrap"
output_dir = "../temp/bootstrap"
pkl_files = util.get_files(input_dir, endswith=".pkl")
for file in pkl_files:
    pkl_file_path = os.path.join(input_dir, file)
    print(file)
    loaded_dict = pickle.load(open(pkl_file_path, "rb"))
    # Detect the **minimum** sample size where p-value crosses 0.05
    averaged_p_values = detect_pvalue_crossing(loaded_dict)
    threshold = 0.05
    for spectra, specimen_p_value in averaged_p_values.items():
        for sample_size, avg_p_value in specimen_p_value.items():
            if avg_p_value > threshold:
                # Collect the **minimum** sample size and stop for this wavenumber
                crossing_sample_size_by_file_and_wavenumber["file"].append((sample_size, spectra))
                break  # Stop after finding the first sample size that exceeds 0.05


# Prepare data for scatter plot
file_names = []
sample_sizes = []
wavenumber_counts = []  # Store the number of wavenumbers for each file


# Scatter plot with multiple files
plt.figure(figsize=(12, 8))

sample_size_count_by_file = defaultdict(lambda: defaultdict(int))

# Loop over the files and wavenumber data to count occurrences of each sample size
for file, spectra_data in crossing_sample_size_by_file_and_wavenumber.items():
    for sample_size, spectra in spectra_data:
        # Increment the count of this sample size for the given file
        sample_size_count_by_file[file][sample_size] += 1

# Prepare data for scatter plot
file_names = []
sample_sizes = []
occurrences = []

# Flatten the data for plotting
for file, sample_size_counts in sample_size_count_by_file.items():
    for sample_size, count in sample_size_counts.items():
        parts = file.split('_')
        extracted_part = '_'.join(parts[2:5])
        file_names.append(extracted_part)  # Store blend ratios and baseline correction
        sample_sizes.append(sample_size)  # Store the sample size
        occurrences.append(count)  # Store the count of occurrences of this sample size

# Scatter plot with file names on x-axis and sample sizes on y-axis
plt.figure(figsize=(12, 8))

# Plot each point, with the size representing the count of occurrences
scatter = plt.scatter(file_names, sample_sizes, s=[occ * 100 for occ in occurrences], c=occurrences, cmap='viridis', edgecolor='k')

# Add color bar to indicate the count of occurrences
cbar = plt.colorbar(scatter)
cbar.set_label('Count of Sample Size Occurrences')


# Set plot labels and title
plt.xlabel('Selected Blends and Baseline Correction Models ')
plt.ylabel('Sample Sizes')
plt.title('Sample Sizes Where p_value Exceeds Critical Threshold')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)

plt.tight_layout()
plt.show()