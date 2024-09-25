import json
import pandas as pd
import itertools
import numpy as np
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from scipy import stats
# import importlib.util
# spec = importlib.util.spec_from_file_location("plot_p_values", "util\\05_data_analysis_plot_jackknife.py")
# plot_p_val = importlib.util.module_from_spec(spec)
import matplotlib.pyplot as plt

print(os.getcwd())
my_path = "../temp/50"

variance_dic = {}
rsd_dic = {}

def plot_p_values(averaged_pvalues, filename):
    # Line plot of RSDs
    # Iterate over each specimen in the nested dictionary
    for spectra, specimen_p_value in averaged_pvalues.items():
        sample_size = list(specimen_p_value.keys())
        avg_p_values = list(specimen_p_value.values())
        plt.plot(sample_size, avg_p_values, marker='o', label=spectra)

    plt.title(f'{filename} Averaged p-value for top 10 wavenumber')
    plt.xlabel('Sample size')
    plt.ylabel('Avg p-value')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()


def get_csv_files(path):
    files_in_path = os.listdir(path)
    csv_files = filter(lambda f: f.endswith(".csv"), files_in_path)
    return list(csv_files)

def calc_relative_std(X):
    specimen_mean = np.mean(X)
    specimen_std = np.std(X)
    if specimen_mean != 0:  # To avoid division by zero
        specimen_rsd = (specimen_std / specimen_mean) * 100  # RSD as percentage
    else:
        specimen_rsd = np.nan  # Handle case where mean is zero
    return specimen_rsd

def find_key_wavenumbers(related_data):
    # Calculate the mean spectrum to identify peaks
    mean_spectrum = related_data.mean(axis=0)

    # Find peaks in the mean spectrum
    peaks, properties = find_peaks(mean_spectrum, height=0)

    # Filter peaks by topN heights
    peak_heights = properties['peak_heights']
    # Sort the peak indices based on the heights in descending order and get the top 20
    sorted_peak_indices = np.argsort(peak_heights)[-10:]
    top_n_peaks = peaks[sorted_peak_indices]

    # The peaks sorted by height
    top_n_peaks_sorted_by_height = top_n_peaks[np.argsort(peak_heights[sorted_peak_indices])[::-1]]
    key_wavenumbers = related_data.columns[top_n_peaks_sorted_by_height].astype(str).tolist()
    return sorted_peak_indices, key_wavenumbers


def get_spectra_from_key_wavenumbers(related_data, key_wavenumbers):
    # Extract relevant columns
    selected_data = related_data[key_wavenumbers]
    selected_data_spcm = pd.concat([selected_data, specimens], axis=1)
    # print(selected_data.describe())

    grouped_data = selected_data_spcm.groupby('reference.specimen').apply(lambda x: x.values.tolist(),
                                                                          include_groups=False)

    ## Variances by specimen
    original_keys = grouped_data.index
    stacked_array = np.stack(grouped_data.values)
    transposed_array = stacked_array.transpose(0, 2, 1)

    result_series = pd.Series([transposed_array[i] for i in range(transposed_array.shape[0])], index=original_keys)
    return result_series

def bootstrap_resampling(data, n_iterations):
    n = 1000
    bootstrap_samples = np.random.choice(data, (n_iterations, n), replace=True)
    return bootstrap_samples

def run_jackknife(absorb_val_by_peaks):
    group_size_leave_out = 10
    jackknife_by_specimen = {}  # to be refreshed each specimen and collect avg variance
    for L in range(0, len(absorb_val_by_peaks) - group_size_leave_out):
        # len(absorb_val_by_peaks) is the number of spots measured
        jackknife_est_var = []
        # Generate all the combinations of given L
        index_combinations = list(itertools.combinations(np.where(absorb_val_by_peaks)[0], L))
        print(f"{L} has number of index combinations: {len(index_combinations)}")
        for indices_to_leave_out in index_combinations:
            # print(indices_to_leave_out)
            X_reduced = np.delete(absorb_val_by_peaks, indices_to_leave_out, axis=0)

            specimen_rsd = calc_relative_std(X_reduced)

            jackknife_est_var = np.append(jackknife_est_var, specimen_rsd)
        jackknife_by_specimen[L] = np.nanmean(jackknife_est_var)
    return jackknife_by_specimen

def run_bootstrap(absorb_val_by_peaks):
    max_resample_size = 20
    bootstrap_by_specimen = {}  # to be refreshed each specimen and collect avg variance
    for L in range(2, max_resample_size):
        # len(absorb_val_by_peaks) is the number of spots measured
        bootstrap_est_var = []
        # Generate bootstrap samples
        bootstrap_samples = bootstrap_resampling(absorb_val_by_peaks, L)

        # Calculate the mean and stdev of each bootstrap sample
        bootstrap_std_error = np.std(bootstrap_samples, axis=0)
        bootstrap_std_means = np.mean(bootstrap_samples, axis=0)
        specimen_rsd = (bootstrap_std_error / bootstrap_std_means) * 100

        bootstrap_est_var = np.append(bootstrap_est_var, specimen_rsd)
        bootstrap_by_specimen[L] = bootstrap_est_var
    return bootstrap_by_specimen


start = time.perf_counter()

# r=root, d=directories, f = files
for r, d, f in os.walk(my_path):
    for file in f:
        if file.endswith(".csv"):
            print(file)
            # Load the data
            data = pd.read_csv(f'../temp/50/{file}', sep=',', header=0)
            related_data = data[data['reference.specimen'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
            related_data = related_data.drop(['reference.pet', 'reference.cotton','reference.area','reference.spot','reference.measuring_date','Unnamed: 0'], axis=1)

            specimens = related_data[related_data.columns[0]]

            (sorted_peak_indices, key_wavenumbers) = find_key_wavenumbers(related_data)

            result_series = get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)

            final_dict = {}
            for wn_index, wn in enumerate(key_wavenumbers):
                print(f"wn: {wn}")
                #jackknife_by_specimen_agg = {}
                bootstrap_by_specimen_agg = {}

                for specimen in result_series.keys():
                    absorb_val_by_peaks = result_series.get(specimen)[wn_index][:]
                    #jackknife_by_specimen = run_jackknife(absorb_val_by_peaks)
                    #jackknife_by_specimen_agg[specimen] = jackknife_by_specimen
                    bootstrap_by_specimen = run_bootstrap(absorb_val_by_peaks)
                    bootstrap_by_specimen_agg[specimen] = bootstrap_by_specimen
                final_dict[wn] = bootstrap_by_specimen_agg
                #final_dict[wn] = jackknife_by_specimen_agg


        #print(final_dict)
        #json_path = f'../temp/50/bootstrap_{file}.json'
        #json.dump(final_dict, open(json_path, 'w'))

#end = time.perf_counter()
#print(f" {end - start:0.4f} seconds")

# loaded_dict = json.load(open(json_path,"r"))
# for wn in loaded_dict.keys():
#     jackknife_agg = final_dict[wn]
#     plot_jackknife(jackknife_agg)



        # Dictionary to store p-values for each specimen, grouped by spectrum
        p_values_by_spectrum_and_sample_size = defaultdict(lambda: defaultdict(list))

        # Initialize dictionary to store p-values for each specimen within a spectrum
        p_values = {}

        for spectra, specimen_sample_size in final_dict.items():
            for specimen, bootstrap in specimen_sample_size.items():

                baseline_data = bootstrap[19]

                # Loop through sample sizes from 2 to 9 for comparison
                for sample_size in range(4, 19):
                    # Extract the bootstrap data for the current sample_size
                    current_data = bootstrap[sample_size]

                    # Perform an F-test to compare the variances (or use a different test if needed)
                    F_statistic, p_value = stats.f_oneway(current_data, baseline_data)

                    # Store the p-value for this sample size comparison
                    p_values_by_spectrum_and_sample_size[spectra][sample_size].append(p_value)


        averaged_pvalues = defaultdict(dict)

        # Dictionary to store the averaged p-values for each spectrum and sample size
        averaged_p_values_by_spectrum_and_sample_size = defaultdict(dict)

        # Compute the average p-value for each spectrum and sample size across specimens
        for spectra, sample_size_dict in p_values_by_spectrum_and_sample_size.items():
            for sample_size, pval_list in sample_size_dict.items():
                # Average the p-values for each spectrum and sample size across all specimens
                avg_p_value = np.mean(pval_list)
                averaged_p_values_by_spectrum_and_sample_size[spectra][sample_size] = avg_p_value
        print(averaged_p_values_by_spectrum_and_sample_size)


        plot_p_values(averaged_p_values_by_spectrum_and_sample_size, file)


### test
print("test")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

my_path = "../temp/50"

# Dictionary to store the minimum sample size where p-value crosses 0.05 for each file and wavenumber
crossing_sample_size_by_file_and_wavenumber = defaultdict(list)


# Function to detect minimum sample size where p-value crosses 0.05
def detect_pvalue_crossing(final_dict):
    p_values_by_spectrum_and_sample_size = defaultdict(lambda: defaultdict(list))
    averaged_p_values_by_spectrum_and_sample_size = defaultdict(dict)

    for spectra, specimen_sample_size in final_dict.items():
        for specimen, bootstrap in specimen_sample_size.items():
            baseline_data = bootstrap[19]
            for sample_size in range(4, 19):
                current_data = bootstrap[sample_size]
                F_statistic, p_value = stats.f_oneway(current_data, baseline_data)
                p_values_by_spectrum_and_sample_size[spectra][sample_size].append(p_value)

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

            (sorted_peak_indices, key_wavenumbers) = find_key_wavenumbers(related_data)
            result_series = get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)
            final_dict = {}

            for wn_index, wn in enumerate(key_wavenumbers):
                bootstrap_by_specimen_agg = {}
                for specimen in result_series.keys():
                    absorb_val_by_peaks = result_series.get(specimen)[wn_index][:]
                    bootstrap_by_specimen = run_bootstrap(absorb_val_by_peaks)
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

# Plot each point, color-coded by the number of wavenumbers
scatter = plt.scatter(crossing_sample_size_by_file_and_wavenumber, c=wavenumber_counts, cmap='viridis', s=100, edgecolor='k')

# Add labels near each point for the count of wavenumbers
for i, txt in enumerate(wavenumber_counts):
    plt.text(file_names[i], sample_sizes[i], str(txt), fontsize=9, ha='right', va='bottom', color='black')

# Add color bar to indicate the count of wavenumbers
cbar = plt.colorbar(scatter)
cbar.set_label('Count of Wavenumbers')

# Set plot labels and title
plt.xlabel('File Names')
plt.ylabel('Minimum Sample Size (p-value > 0.05)')
plt.title('Minimum Sample Size where p-value crosses 0.05 for different files, colored by wavenumber count')
plt.xticks(rotation=90)
plt.grid(True)

plt.tight_layout()
plt.show()
