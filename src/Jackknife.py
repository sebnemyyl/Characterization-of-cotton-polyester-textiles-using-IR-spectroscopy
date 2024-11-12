import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
import util.m05_jackknife_bootstrap_resampling as resampling_methods
import util.m00_general_util as util
import pickle

print(os.getcwd())

my_path = "temp/resampling"


# Needs to be true otherwise resampling doesn't work correctly!!
def are_spots_per_specimen_equal(df):
    spots_per_specimen_df = df.groupby('reference.specimen').size()
    spots_per_specimen = spots_per_specimen_df.to_numpy()
    all_counts_equal = (spots_per_specimen[0] == spots_per_specimen).all()
    if not all_counts_equal:
        print("Number of spots per specimen:")
        print(spots_per_specimen_df)
    return all_counts_equal

# TODO consider deleting, because not really useful 
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

    file_name_without_ending = os.path.splitext(file)[0]
    pickle_file_name = f'{type}_{file_name_without_ending}.pkl'
    output_path = os.path.join(output_dir, pickle_file_name)
    with open(output_path, 'wb') as f:
        pickle.dump(final_dict, f)
    print(f"Pickle dumped to {output_path}")

output_dir = "temp/bootstrap50"
csv_files = util.get_csv_files(my_path)
for file in csv_files:
    csv_file = os.path.join(my_path, file)
    print(csv_file)
    resample_csv_file(csv_file, cotton=50, type="bootstrap", output_dir=output_dir)
