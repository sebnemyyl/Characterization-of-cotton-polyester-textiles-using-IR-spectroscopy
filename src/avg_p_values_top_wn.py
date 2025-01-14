# First Bootstrap will be migrated here and then the Jackknife is to be implemented
import pickle
from collections import defaultdict
from scipy import stats
# import importlib.util
# spec = importlib.util.spec_from_file_location("plot_p_values", "util\\05_data_analysis_plot_jackknife.py")
# plot_p_val = importlib.util.module_from_spec(spec)
import os
import json
import numpy as np
import matplotlib.pyplot as plt

print(os.getcwd())
my_path = "../temp/jackknife"
#os.chdir("")

def get_csv_files(path):
    files_in_path = os.listdir(path)
    csv_files = filter(lambda f: f.endswith(".csv"), files_in_path)
    return list(csv_files)

#loaded_dict = json.load(open("../temp/bootstrap/bootstrap_spectra_treated_als_50_cotton_nir.csv.json","r"))


# Load the dictionary from a file
with open('../temp/bootstrap/bootstrap_spectra_treated_als_50_cotton_nir.csv.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# Dictionary to store p-values for each specimen, grouped by spectrum
p_values_by_spectrum_and_sample_size = defaultdict(lambda: defaultdict(list))

# Initialize dictionary to store p-values for each specimen within a spectrum
p_values = {}

for spectra, specimen_sample_size in loaded_dict.items():
    for specimen, jackknife in specimen_sample_size.items():
        baseline_data = jackknife[19]

        # Loop through sample sizes from 2 to 9 for comparison
        for sample_size in range(4, 16):
            # Extract the bootstrap data for the current sample_size
            current_data = jackknife[sample_size]

            # Perform an F-test to compare the variances (or use a different test if needed)
            F_statistic, p_value = stats.f_oneway(current_data, baseline_data)
            print(F_statistic)
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
#print(averaged_p_values_by_spectrum_and_sample_size)

def plot_p_values(averaged_pvalues):
    # Line plot of RSDs
    # Iterate over each specimen in the nested dictionary
    for spectra, specimen_p_value in averaged_pvalues.items():
        sample_size = list(specimen_p_value.keys())
        avg_p_values = list(specimen_p_value.values())
        plt.plot(sample_size, avg_p_values, marker='o', label=spectra)

    plt.title(f'Averaged p-value for top 10 wavenumber')
    plt.xlabel('Sample size')
    plt.ylabel('Avg p-value')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()

plot_p_values(averaged_p_values_by_spectrum_and_sample_size)