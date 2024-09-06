import json
import pandas as pd
import itertools
import numpy as np
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import time

print(os.getcwd())
my_path = "../temp/50"

variance_dic = {}
rsd_dic = {}

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
    sorted_peak_indices = np.argsort(peak_heights)[-8:]
    top_20_peaks = peaks[sorted_peak_indices]

    # The peaks sorted by height
    top_20_peaks_sorted_by_height = top_20_peaks[np.argsort(peak_heights[sorted_peak_indices])[::-1]]
    key_wavenumbers = related_data.columns[top_20_peaks_sorted_by_height].astype(str).tolist()
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



start = time.perf_counter()

# r=root, d=directories, f = files
for r, d, f in os.walk(my_path):
    for file in f:
        if file.endswith(".csv"):
            print(file)
            # Load the data
            data = pd.read_csv(f'../temp/50/{file}', sep=',', header=0)
            related_data = data[data['reference.specimen'].isin([1, 2, 3])]
            related_data = related_data.drop(['reference.pet', 'reference.cotton','reference.area','reference.spot','reference.measuring_date','Unnamed: 0'], axis=1)

            specimens = related_data[related_data.columns[0]]

            (sorted_peak_indices, key_wavenumbers) = find_key_wavenumbers(related_data)

            result_series = get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)

            final_dict = {}
            for wn_index, wn in enumerate(key_wavenumbers):
                print(f"wn: {wn}")
                jackknife_by_specimen_agg = {}
                for specimen in result_series.keys():
                    print(specimen)
                    absorb_val_by_peaks = result_series.get(specimen)[wn_index][:]
                    jackknife_by_specimen = run_jackknife(absorb_val_by_peaks)
                    jackknife_by_specimen_agg[specimen] = jackknife_by_specimen
                final_dict[wn] = jackknife_by_specimen_agg
                end = time.perf_counter()
                print(f" {end - start:0.4f} seconds")

        #print(final_dict)
        json_path = f'../temp/50/jackknife_{file}.json'
        json.dump(final_dict, open(json_path, 'w'))

#loaded_dict = json.load(open(json_path,"r"))
#for wn in loaded_dict.keys():
#    jackknife_agg = final_dict[wn]
#    plot_jackknife(jackknife_agg)
