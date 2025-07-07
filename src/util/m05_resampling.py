import pandas as pd
import itertools
import numpy as np
from scipy.signal import find_peaks
import os
import pickle


def calc_relative_std(X):
    specimen_mean = np.mean(X)
    specimen_std = np.std(X)
    if specimen_mean != 0:  # To avoid division by zero
        specimen_rsd = (specimen_std / specimen_mean) * 100  # RSD as percentage
    else:
        specimen_rsd = np.nan  # Handle case where mean is zero
    return specimen_rsd

def find_key_wavenumbers(related_data, top_n_peaks=10):
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

def bootstrap_resampling(data, n_iterations):
    n = 1000
    bootstrap_samples = np.random.choice(data, (n_iterations, n), replace=True)
    return bootstrap_samples

def run_jackknife(absorb_val_by_peaks):
    group_size_leave_out = 0
    jackknife_by_specimen = {}  # to be refreshed each specimen and collect avg variance
    for L in range(0, len(absorb_val_by_peaks) - group_size_leave_out):
        print(f"the L {L}")
        # len(absorb_val_by_peaks) is the number of spots measured
        jackknife_est_var = []
        # Generate all the combinations of given L
        index_combinations = list(itertools.combinations(np.where(absorb_val_by_peaks)[0], L)) #np.arrange
        print(f"the combinations  {np.where(absorb_val_by_peaks)[0]}")
        print(f"absorb_val_by_peaks {absorb_val_by_peaks}")
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
    for L in range(3, max_resample_size+1):
        # len(absorb_val_by_peaks) is the number of spots measured
        bootstrap_est_var = []
        # Generate bootstrap samples
        bootstrap_samples = bootstrap_resampling(absorb_val_by_peaks, L)

        # Calculate the mean and stdev of each bootstrap sample
        bootstrap_stdev = np.std(bootstrap_samples, axis=0)
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        specimen_rsd = (bootstrap_stdev / bootstrap_means) * 100
        bootstrap_var = np.var(bootstrap_samples, axis=0)
        #specimen_rsd = calc_relative_std(bootstrap_samples)

        bootstrap_est_var = np.append(bootstrap_est_var, specimen_rsd)
        bootstrap_by_specimen[L] = bootstrap_est_var
    return bootstrap_by_specimen

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
    related_data = related_data[related_data['reference.spot'] <= 20]
    related_data = related_data.drop(['reference.pet', 'reference.cotton', 'reference.area', 'reference.spot',
                                              'reference.measuring_date', 'Unnamed: 0'], axis=1)

    # Needs to be true otherwise resampling doesn't work correctly!!
    assert are_spots_per_specimen_equal(related_data), "Number of spots per specimen are not equal!"

    # We find the 10 highest peaks, which are our key wavenumbers
    (_, key_wavenumbers) = find_key_wavenumbers(related_data, 10)
    key_spectra_grouped_by_specimen = get_spectra_from_key_wavenumbers(related_data, key_wavenumbers)

    final_dict = {}
    for wn_index, wn in enumerate(key_wavenumbers):
        print(f"Wave number: {wn}")
        resample_by_specimen_agg = {}

        for specimen in key_spectra_grouped_by_specimen.keys():
            absorb_val_by_peaks = key_spectra_grouped_by_specimen.get(specimen)[wn_index][:]
            #approximate_required_spots(absorb_val_by_peaks)

            if type == "jackknife":
                jackknife_by_specimen = run_jackknife(absorb_val_by_peaks)
                resample_by_specimen_agg[specimen] = jackknife_by_specimen
            elif type == "bootstrap":
                bootstrap_by_specimen = run_bootstrap(absorb_val_by_peaks)
                resample_by_specimen_agg[specimen] = bootstrap_by_specimen
            else:
                raise Exception(f"{type} not supported!")
        final_dict[wn] = resample_by_specimen_agg

    file_name_with_ending = os.path.basename(csv_file)
    file_name = os.path.splitext(file_name_with_ending)[0]
    pickle_file_name = f'{type}_{file_name}.pkl'
    output_path = os.path.join(output_dir, pickle_file_name)
    with open(output_path, 'wb') as f:
        pickle.dump(final_dict, f)
    print(f"Pickle dumped to {output_path}")