import json
import importlib.util
spec = importlib.util.spec_from_file_location("plot_jackknife", "util\\05_data_analysis_plot_jackknife.py")
jackknife = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jackknife)
import os
import matplotlib.pyplot as plt
import numpy as np

print(os.getcwd())
#os.chdir("..")


json_path = f'../temp/jackknife_spectra_treated_snv_30_cotton_nir.json'

loaded_dict = json.load(open(json_path,"r"))
for wn in loaded_dict.keys():
   jackknife_agg = loaded_dict[wn]
   jackknife.plot_jackknife(jackknife_agg, wn, type="jackknife")


# Average the Jackknife results
# First it collects the Jk results from each wavenumber and dumps into
# an array associated to a specimen number
# Then averages the values in each array

averages = {}

for spectra_key, spectra_value in loaded_dict.items():
    for sub_key, sub_value in spectra_value.items():
        if sub_key not in averages:
            averages[sub_key] = {}

        for inner_key, inner_value in sub_value.items():
            if inner_key not in averages[sub_key]:
                averages[sub_key][inner_key] = []

            # Append the float values to a list to compute the average later
            averages[sub_key][inner_key].append(inner_value)

# Compute the average for each key
averaged_data = {key: {inner_key: np.mean(values) for inner_key, values in inner_dict.items()}
                 for key, inner_dict in averages.items()}

# Convert to the required format
spectra_avg = {"spectra_avg": averaged_data}


jackknife.plot_jackknife(averaged_data, wn="Top 10 wavenumbers", type="jackknife")


### Hypothesis testing

from scipy.stats import chi2


def chi_squared_test(sample_variance, full_variance, sample_size, confidence_level=0.95, alpha=0.05):
    # Degrees of freedom
    df = sample_size - 1

    # Calculate chi-squared statistic
    chi_stat = (df * sample_variance) / full_variance

    # Calculate critical value for the confidence level
    chi_critical = chi2.ppf(confidence_level, df)

    # Calculate p-value
    p_value = chi2.sf(chi_stat, df)

    # Compare chi-squared statistic to critical value
    #return chi_stat < chi_critical

    # Check if p-value is less than or equal to the significance level
    return p_value <= alpha, p_value, chi_stat

averages_for_chi_sq = {}
for specimen_key, jk_value in spectra_avg.items():
    for sub_key, sub_value in jk_value.items():
        #print(sub_key, sub_value)
        if sub_key not in averages_for_chi_sq:
            averages_for_chi_sq[sub_key] = {}

        for inner_key, inner_value in sub_value.items():
            if inner_key not in averages_for_chi_sq[sub_key]:
                averages_for_chi_sq[sub_key][inner_key] = []
            averages_for_chi_sq[sub_key][inner_key].append(inner_value)
print(averages_for_chi_sq)


if chi_squared_test(sample_variance=0.012, full_variance=0.010, sample_size=100, confidence_level=0.95, alpha=0.05):
    print(f"You can remove  items and still maintain true variance with 95% confidence.")
else:
    print(f"Removing  items breaks the 95% confidence interval.")
