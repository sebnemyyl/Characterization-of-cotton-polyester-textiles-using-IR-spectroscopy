import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import importlib.util

resample = importlib.util.spec_from_file_location("resampling_methods", "util\\05_jackknife_bootstrap_resampling.py")
resampling_methods = importlib.util.module_from_spec(resample)
resample.loader.exec_module(resampling_methods)

my_path = "../temp/all"

# Dictionary to store the minimum sample size where p-value crosses 0.05 for each file and wavenumber
crossing_sample_size_by_file_and_wavenumber = defaultdict(list)


# Function to detect minimum sample size where p-value crosses 0.05
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


for r, d, f in os.walk(my_path):
    for file in f:
        if file.endswith(".csv"):
            print(f"Processing file: {file}")

            # Load the data (adjust this path if needed)
            data = pd.read_csv(os.path.join(r, file), sep=',', header=0)
            related_data = data[data['reference.specimen'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
            related_data = related_data.drop(['reference.pet', 'reference.cotton', 'reference.area',
                                              'reference.spot', 'reference.measuring_date', 'Unnamed: 0'], axis=1)
            specimens = related_data[related_data.columns[0]]

            (sorted_peak_indices, key_wavenumbers) = resampling_methods.find_key_wavenumbers(related_data)
            result_series = resampling_methods.get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)
            final_dict = {}

            for wn_index, wn in enumerate(key_wavenumbers):
                bootstrap_by_specimen_agg = {}
                for specimen in result_series.keys():
                    absorb_val_by_peaks = result_series.get(specimen)[wn_index][:]
                    bootstrap_by_specimen = resampling_methods.run_bootstrap(absorb_val_by_peaks)
                    bootstrap_by_specimen_agg[specimen] = bootstrap_by_specimen
                final_dict[wn] = bootstrap_by_specimen_agg

            # Detect the **minimum** sample size where p-value crosses 0.05
            averaged_p_values = detect_pvalue_crossing(final_dict)
            threshold = 0.05
            for spectra, specimen_p_value in averaged_p_values.items():
                for sample_size, avg_p_value in specimen_p_value.items():
                    if avg_p_value > threshold:
                        # Collect the **minimum** sample size and stop for this wavenumber
                        crossing_sample_size_by_file_and_wavenumber[file].append((sample_size, spectra))
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