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
import util.m05_jackknife_bootstrap_resampling as resampling_methods
# spec = importlib.util.spec_from_file_location("plot_p_values", "util\\05_data_analysis_plot_jackknife.py")
# plot_p_val = importlib.util.module_from_spec(spec)
import pickle

print(os.getcwd())

my_path = "output/example50"
#my_path = "temp/50"

variance_dic = {}
rsd_dic = {}

def calculate_averages(d):
    if isinstance(d, dict):
        # Recursively calculate for nested dictionaries
        return {k: calculate_averages(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        # Calculate the mean for NumPy arrays
        return np.mean(d).tolist()  # Convert the mean to list (in this case, a scalar)
    else:
        return d



def get_csv_files(path):
    files_in_path = os.listdir(path)
    csv_files = filter(lambda f: f.endswith(".csv"), files_in_path)
    return list(csv_files)

# Needs to be true otherwise JackKnife doesn't work correctly!!
# TODO move it somewhere better!
def has_equal_specimen_count(df):
    specimen_counts_df = df.groupby('reference.specimen').size()
    specimen_counts = specimen_counts_df.to_numpy()
    all_counts_equal = (specimen_counts[0] == specimen_counts).all()
    if not all_counts_equal:
        print("Actual Specimen Counts:")
        print(specimen_counts_df)
    return all_counts_equal

def resample_csv_file(csv_file):
    data = pd.read_csv(csv_file, sep=',', header=0)
    related_data = data[data['reference.specimen'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    #related_data = related_data.query("`reference.cotton` == 50")
    related_data = related_data.drop(['reference.pet', 'reference.cotton', 'reference.area', 'reference.spot',
                                              'reference.measuring_date', 'Unnamed: 0'], axis=1)

    assert has_equal_specimen_count(related_data), "Specimen counts are not equal!"

    (sorted_peak_indices, key_wavenumbers) = resampling_methods.find_key_wavenumbers(related_data)

    result_series = resampling_methods.get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)

    final_dict = {}
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

            #jackknife_by_specimen = resampling_methods.run_jackknife(absorb_val_by_peaks)
            #jackknife_by_specimen_agg[specimen] = jackknife_by_specimen
            bootstrap_by_specimen = resampling_methods.run_bootstrap(absorb_val_by_peaks)
            bootstrap_by_specimen_agg[specimen] = bootstrap_by_specimen
        final_dict[wn] = bootstrap_by_specimen_agg
        #final_dict[wn] = jackknife_by_specimen_agg


        #final_dict_averaged = calculate_averages(final_dict)
        #json_path = f'../temp/bootstrap/bootstrap_{file}.json'
        #json.dump(final_dict, open(json_path, 'w'))
        with open(f'bootstrap_{file}.pkl', 'wb') as f:
            pickle.dump(final_dict, f)
        print("Pickle dumped!")

csv_files = get_csv_files(my_path)
for file in csv_files:
    csv_file = os.path.join(my_path, file)
    print(csv_file)
    resample_csv_file(csv_file)

## TODO make configurable for bootstrap and Jackknife

'''
## TODO remove this original code, is extracted as method above
# r=root, d=directories, f = files
for r, d, f in os.walk(my_path):
    for file in f:
        if file.endswith(".csv"):
            print(file)
            # Load the data
            data = pd.read_csv(f'temp/{file}', sep=',', header=0)
            related_data = data[data['reference.specimen'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
            related_data = related_data.drop(['reference.pet', 'reference.cotton', 'reference.area', 'reference.spot',
                                              'reference.measuring_date', 'Unnamed: 0'], axis=1)

            (sorted_peak_indices, key_wavenumbers) = resampling_methods.find_key_wavenumbers(related_data)

            result_series = resampling_methods.get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)

            final_dict = {}
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

                    #jackknife_by_specimen = resampling_methods.run_jackknife(absorb_val_by_peaks)
                    #jackknife_by_specimen_agg[specimen] = jackknife_by_specimen
                    bootstrap_by_specimen = resampling_methods.run_bootstrap(absorb_val_by_peaks)
                    bootstrap_by_specimen_agg[specimen] = bootstrap_by_specimen
                final_dict[wn] = bootstrap_by_specimen_agg
                #final_dict[wn] = jackknife_by_specimen_agg


        #final_dict_averaged = calculate_averages(final_dict)
        #json_path = f'../temp/bootstrap/bootstrap_{file}.json'
        #json.dump(final_dict, open(json_path, 'w'))
        with open(f'bootstrap_{file}.pkl', 'wb') as f:
            pickle.dump(final_dict, f)

#end = time.perf_counter()
#print(f" {end - start:0.4f} seconds")

# loaded_dict = json.load(open(json_path,"r"))
# for wn in loaded_dict.keys():
#     jackknife_agg = final_dict[wn]
#     plot_jackknife(jackknife_agg)
'''