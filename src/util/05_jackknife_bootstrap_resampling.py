import pandas as pd
import itertools
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict
from scipy import stats



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

def bootstrap_resampling(data, n_iterations):
    n = 10000
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
        #print(f"{L} has number of index combinations: {len(index_combinations)}")
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
        bootstrap_stdev = np.std(bootstrap_samples, axis=0)
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        specimen_rsd = (bootstrap_stdev / bootstrap_means) * 100
        bootstrap_var = np.var(bootstrap_samples, axis=0)


        bootstrap_est_var = np.append(bootstrap_est_var, bootstrap_var)
        bootstrap_by_specimen[L] = bootstrap_est_var
    return bootstrap_by_specimen