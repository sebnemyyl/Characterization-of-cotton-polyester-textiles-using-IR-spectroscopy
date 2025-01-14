import pandas as pd
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import matplotlib.cm as cm
import importlib.util
# spec = importlib.util.spec_from_file_location("plot_p_values", "util\\05_data_analysis_plot_jackknife.py")
# plot_p_val = importlib.util.module_from_spec(spec)
from scipy.signal import find_peaks


print(os.getcwd())
my_path = "../temp/reproducibility/bsn_corr"

variance_dic = {}
rsd_dic = {}

def find_key_wavenumbers(related_data, top_n_peaks=20):
    # Calculate the mean spectrum to identify peaks
    mean_spectrum = related_data.mean(axis=0)

    # Find peaks in the mean spectrum
    peaks, properties = find_peaks(mean_spectrum, height=0)

    # Filter peaks by topN heights
    peak_heights = properties['peak_heights']
    # Sort the peak indices based on the heights in descending order and get the top n peaks
    sorted_peak_indices = np.argsort(peak_heights)[-top_n_peaks:]
    top_n_peaks = peaks[sorted_peak_indices]

    # The peaks sorted by height
    top_n_peaks_sorted_by_height = top_n_peaks[np.argsort(peak_heights[sorted_peak_indices])[::-1]]
    key_wavenumbers = related_data.columns[top_n_peaks_sorted_by_height].astype(str).tolist()
    return sorted_peak_indices, key_wavenumbers

def get_spectra_from_key_wavenumbers(data, key_wavenumbers):
    specimens = data[data.columns[0]]
    # Extract relevant columns
    selected_data = data[key_wavenumbers]
    selected_data_spcm = pd.concat([selected_data, specimens], axis=1)
    # print(selected_data.describe())

    grouped_data = selected_data_spcm.groupby('reference.specimen').apply(lambda x: x.values.tolist(),
                                                                          include_groups=False)

    original_keys = grouped_data.index
    stacked_array = np.stack(grouped_data.values)
    transposed_array = stacked_array.transpose(0, 2, 1)

    result_series = pd.Series([transposed_array[i] for i in range(transposed_array.shape[0])], index=original_keys)
    return result_series

def bootstrap_resampling(data, n_iterations=100, sample_size = 20):
    bootstrap_samples = np.random.choice(data, (n_iterations, sample_size), replace=True)
    means = np.mean(bootstrap_samples, axis=1)
    return means


def run_bootstrap(absorb_val_by_peaks):
    full_data_mean = np.mean(absorb_val_by_peaks)
    max_resample_size = 20
    bootstrap_by_specimen = {}  # to be refreshed each specimen and collect avg variance
    for L in range(2, max_resample_size):
        # len(absorb_val_by_peaks) is the number of spots measured
        bootstrap_est_var = []
        # Generate bootstrap samples
        bootstrap_means = bootstrap_resampling(absorb_val_by_peaks, 100, L)

        lower_percentile = np.percentile(bootstrap_means, 2.5)
        upper_percentile = np.percentile(bootstrap_means, 97.5)
        margin_of_error = (upper_percentile - lower_percentile) / 2

        threshold = (full_data_mean * 0.05) / 2
        if margin_of_error <= threshold:
            print(f"Minimum sample size: {L}")
            return L  # This sample size is statistically representative

#acceptable_margin_of_error = 0.05

final_dict = {}
# r=root, d=directories, f = files
for r, d, f in os.walk(my_path):
    for file in f:
        if file.endswith(".csv"):
            print(file)
            # Load the data
            data = pd.read_csv(f'../temp/reproducibility/bsn_corr/{file}', sep=',', header=0)
            related_data = data[data['reference.specimen'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
            related_data = related_data.drop(['reference.pet', 'reference.cotton','reference.area','reference.spot','reference.measuring_date','Unnamed: 0'], axis=1)

            specimens = related_data[related_data.columns[0]]
            (sorted_peak_indices, key_wavenumbers) = find_key_wavenumbers(related_data)
            result_series = get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)


            for wn_index, wn in enumerate(key_wavenumbers):
                print(f"wn: {wn}")
                #jackknife_by_specimen_agg = {}
                bootstrap_by_specimen_agg = {}

                for specimen in result_series.keys():
                    absorb_val_by_peaks = result_series.get(specimen)[wn_index][:]

                    Z = 1.96  # Z-score for 95% confidence level
                    E = 0.005  # Margin of error
                    absorb_val_by_peaks_stdev = np.std(absorb_val_by_peaks, axis=0)
                    # Function to calculate required spots
                    approximate_required_spots = int(np.square((Z * absorb_val_by_peaks_stdev) / E))
                    print(f' approximate sample size: {approximate_required_spots}')

                    bootstrap_by_specimen = run_bootstrap(absorb_val_by_peaks)
                    bootstrap_by_specimen_agg[specimen] = bootstrap_by_specimen
                final_dict[(file, wn)] = {'bootstrap_by_specimen': bootstrap_by_specimen_agg}



# Initialize defaultdict for averaged sample size
averaged_sample_size = defaultdict(dict)

# Iterate over the final_dict
for (file, spectra), specimen_data in final_dict.items():
    bootstrap_by_specimen_agg = specimen_data['bootstrap_by_specimen']

    # Initialize a list to store all bootstrap values for (file, spectra) combination
    all_bootstrap_values = []

    # Iterate through the specimens and collect bootstrap values
    for specimen, bootstrap_values in bootstrap_by_specimen_agg.items():
        # Ensure that bootstrap_values is treated as a list, even if it's a single int
        if not isinstance(bootstrap_values, list):
            bootstrap_values = [bootstrap_values]

        # Filter out None values from bootstrap_values
        cleaned_bootstrap_values = [val for val in bootstrap_values if val is not None]

        # Add the cleaned bootstrap values to the list
        all_bootstrap_values.extend(cleaned_bootstrap_values)

    # Once all specimen values for this (file, spectra) are collected, compute the overall average
    if all_bootstrap_values:
        avg_bootstrap_value = np.nanmean(all_bootstrap_values)

        # Store the averaged bootstrap value with file and spectra as keys (without specimen)
        averaged_sample_size[(file, spectra)] = avg_bootstrap_value



##Plotting
# Extract spectra, averaged bootstrap values, and file names from averaged_sample_size
spectra_values = []
avg_bootstrap_values = []
file_names = []

for (file, spectra), avg_value in averaged_sample_size.items():
    # Extract the numeric part of the spectra (after "spectra.")
    spectra_num = float(spectra.split('spectra.')[1])  # Extract the number after 'spectra.'
    spectra_values.append(spectra_num)
    avg_bootstrap_values.append(avg_value)
    file_names.append(file)

# Create a unique color for each file name
unique_file_names = list(set(file_names))  # Get a list of unique file names
colors = cm.rainbow(np.linspace(0, 1, len(unique_file_names)))  # Generate a color map
color_map = dict(zip(unique_file_names, colors))  # Map each file name to a unique color

# Assign a color to each point based on the file name
point_colors = [color_map[file] for file in file_names]

# Create scatter plot with colors
plt.figure(figsize=(10, 6))
scatter = plt.scatter(spectra_values, avg_bootstrap_values, c=point_colors, marker='o')

# Add labels and title
plt.xlabel('Wavenumbers')
plt.ylabel('Min Sample Size')
plt.title('Minimum Sample Size Required for Different Blend and Baseline Correction Combinations')

# Create a legend based on file names
handles = [plt.Line2D([0], [0], marker='o', color=color_map[file], linestyle='', markersize=10) for file in unique_file_names]
plt.legend(handles, unique_file_names, title="File Names", loc='upper right')

# Show plot
plt.show()