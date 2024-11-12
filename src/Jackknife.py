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

my_path = "temp/resampling"

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
# TODO rename to something like every specimen should have same number of spots!
def are_spots_per_specimen_equal(df):
    spots_per_specimen_df = df.groupby('reference.specimen').size()
    spots_per_specimen = spots_per_specimen_df.to_numpy()
    all_counts_equal = (spots_per_specimen[0] == spots_per_specimen).all()
    if not all_counts_equal:
        print("Number of spots per specimen:")
        print(spots_per_specimen_df)
    return all_counts_equal

def approximate_required_spots(absorb_val_by_peaks):
    Z = 1.96  # Z-score for 95% confidence level
    E = 0.005  # Margin of error
    absorb_val_by_peaks_stdev = np.std(absorb_val_by_peaks, axis=0)
    # Function to calculate required spots
    approximate_required_spots = int(np.square((Z * absorb_val_by_peaks_stdev) / E))
    print(f' approximate sample size: {approximate_required_spots}')

def resample_csv_file(csv_file, cotton = -1, type = "bootstrap", output_dir="."):
    data = pd.read_csv(csv_file, sep=',', header=0)
    #related_data = data[data['reference.specimen'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    #cotton_contents = related_data['reference.cotton'].unique() # contents: [50 23 35 30 25]
    related_data = data
    if cotton > 0:
        related_data = related_data.query(f"`reference.cotton` == {cotton}") 
    related_data = related_data.drop(['reference.pet', 'reference.cotton', 'reference.area', 'reference.spot',
                                              'reference.measuring_date', 'Unnamed: 0'], axis=1)
    assert are_spots_per_specimen_equal(related_data), "Number of spots per specimen are not equal!"

    # We find the 10 highest peaks, which are our key wavenumbers
    (_, key_wavenumbers) = resampling_methods.find_key_wavenumbers(related_data, 10)
    key_spectra_grouped_by_specimen = resampling_methods.get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)

    final_dict = {}
    for wn_index, wn in enumerate(key_wavenumbers):
        print(f"Wave number: {wn}")
        resample_by_specimen_agg = {}

        for specimen in key_spectra_grouped_by_specimen.keys():
            absorb_val_by_peaks = key_spectra_grouped_by_specimen.get(specimen)[wn_index][:]
            #approximate_required_spots(absorb_val_by_peaks)

            if type == "jackknife":
                jackknife_by_specimen = resampling_methods.run_jackknife(absorb_val_by_peaks)
                resample_by_specimen_agg[specimen] = jackknife_by_specimen
            elif type == "bootstrap":
                bootstrap_by_specimen = resampling_methods.run_bootstrap(absorb_val_by_peaks)
                resample_by_specimen_agg[specimen] = bootstrap_by_specimen
            else:
                raise Exception(f"{type} not supported!")
        final_dict[wn] = resample_by_specimen_agg

        #final_dict_averaged = calculate_averages(final_dict)
        #json_path = f'../temp/bootstrap/bootstrap_{file}.json'
        #json.dump(final_dict, open(json_path, 'w'))
    file_name_without_ending = os.path.splitext(file)[0]
    pickle_file_name = f'{type}_{file_name_without_ending}.pkl'
    output_path = os.path.join(output_dir, pickle_file_name)
    with open(output_path, 'wb') as f:
        pickle.dump(final_dict, f)
    print(f"Pickle dumped to {output_path}")

output_dir = "temp/bootstrap50"
csv_files = get_csv_files(my_path)
for file in csv_files:
    csv_file = os.path.join(my_path, file)
    print(csv_file)
    resample_csv_file(csv_file, cotton=50, type="bootstrap", output_dir=output_dir)

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