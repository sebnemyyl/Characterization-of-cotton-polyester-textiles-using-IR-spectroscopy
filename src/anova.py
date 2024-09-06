import pandas as pd
from scipy.stats import f_oneway
import numpy as np
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt

print(os.getcwd())
my_path = "../temp/40"

variance_dic = {}
rsd_dic = {}

# r=root, d=directories, f = files
for r, d, f in os.walk(my_path):
    for file in f:
        if file.endswith(".csv"):
            # Load the data
            data = pd.read_csv(f'../temp/50/{file}', sep=',', header=0)

            #related_data = data[(data['pet'] == 70) & (data['measuring_date'] >= 240626) & (data['specimen'].isin([1,2,3,4,5,6,7,8,9,10]))]
            related_data = data.drop(['reference.pet', 'reference.cotton','reference.area','reference.spot','reference.measuring_date','Unnamed: 0'], axis=1)

            specimens = related_data[related_data.columns[0]]

            # Calculate the mean spectrum to identify peaks
            mean_spectrum = related_data.mean(axis=0)

            # Find peaks in the mean spectrum
            peaks, properties = find_peaks(mean_spectrum, height=0)
            #key_wavenumbers = related_data.columns[peaks].astype(str).tolist()
            #print("Key wavenumbers for ANOVA analysis: ", key_wavenumbers)

            # Filter peaks by topN heights
            peak_heights = properties['peak_heights']
            # Sort the peak indices based on the heights in descending order and get the top 20
            sorted_peak_indices = np.argsort(peak_heights)[-20:]
            top_20_peaks = peaks[sorted_peak_indices]

            # The peaks sorted by height
            top_20_peaks_sorted_by_height = top_20_peaks[np.argsort(peak_heights[sorted_peak_indices])[::-1]]
            key_wavenumbers = related_data.columns[top_20_peaks_sorted_by_height].astype(str).tolist()

            # Print the top 20 peak indices
            print("Indices of the 20 highest peaks:", key_wavenumbers)

            # Extract relevant columns
            selected_data = related_data[key_wavenumbers]
            selected_data_spcm = pd.concat([selected_data,specimens], axis=1)
            print(selected_data.describe())

            grouped_data = selected_data_spcm.groupby('reference.specimen').apply(lambda x: x.values.tolist(), include_groups=False)


            ## Variances by specimen
            original_keys = grouped_data.index
            stacked_array = np.stack(grouped_data.values)
            transposed_array = stacked_array.transpose(0, 2, 1)

            result_series = pd.Series([transposed_array[i] for i in range(transposed_array.shape[0])], index=original_keys)

            variances_by_specimen = {}
            rsd_by_specimen = {}

            for specimen in result_series.keys():
                temp_dict = result_series[specimen]
                variances = []
                rsds = []
                for group in temp_dict:
                    # Calculate the variance across all wavenumbers for this specimen
                    #specimen_variances = np.var(group)
                    #variances = np.append(variances, specimen_variances)
                #variances_by_specimen[specimen] = np.average(variances) # mean variance over wavenumbers

                    # Calculate the relative standard deviation across all wavenumbers for this specimen
                    specimen_mean = np.mean(group)
                    specimen_std = np.std(group)

                    if specimen_mean != 0:  # To avoid division by zero
                        specimen_rsd = (specimen_std / specimen_mean) * 100  # RSD as percentage
                    else:
                        specimen_rsd = np.nan  # Handle case where mean is zero

                    rsds = np.append(rsds, specimen_rsd)
                rsd_by_specimen[specimen] = np.nanmean(rsds)

            print(f'Variance by specimen: {variances_by_specimen}')
            key = file.split('_')[2]
            variance_dic[key] = {}
            rsd_dic[key] = {}
            variance_dic[key].update(variances_by_specimen)
            rsd_dic[key].update(rsd_by_specimen)

            ## Plotting
            # plt.bar(range(len(variances_by_specimen)), list(variances_by_specimen.values()), align='center')
            # plt.xticks(range(len(variances_by_specimen)), list(variances_by_specimen.keys()))
            # plt.title('50% Cotton- 50% Polyester ~ NIR ~ Variances by Specimens')
            # plt.show()

            ## Perform ANOVA on each wavenumber
            results = {}
            for i, wavenumber in enumerate(key_wavenumbers):
                #print(i, wavenumber)
                try:
                    input = [np.array(group)[:, i].tolist() for group in grouped_data]
                    f_statistic, p_value = f_oneway(*input)
                    results[wavenumber] = {'F-statistic': f_statistic, 'p-value': p_value}
                except IndexError as e:
                    print("IndexError")
                    break
            # Output the results
            for wavenumber, stats in results.items():
                print(f'Wavenumber: {wavenumber}, F-statistic: {stats["F-statistic"]}, p-value: {stats["p-value"]}')

print(variance_dic)


# Line Plot -- Variance/RSD for Different Baseline Correction Methods
#plt.figure(figsize=(10, 6))

# Plot each method's variances
for method, specimen_variances in variance_dic.items():
    specimens = list(specimen_variances.keys())
    variance_values = list(specimen_variances.values())
    plt.plot(specimens, variance_values, marker='o', label=method)

plt.title('Variance for Different Baseline Correction Methods - 23% Cotton')
plt.xlabel('Specimen')
plt.ylabel('Variance')
plt.legend(title='Bl Correction Method')
plt.grid(True)

plt.show()


# Line plot of RSDs
for method, specimen_rsds in rsd_dic.items():
    specimens = list(specimen_rsds.keys())
    rsd_values = list(specimen_rsds.values())
    plt.plot(specimens, rsd_values, marker='o', label=method)

plt.title('RSD for Different Baseline Correction Methods - 50% Cotton')
plt.xlabel('Specimen')
plt.ylabel('RSD (%)')
plt.ylim(0, 5)
plt.legend(title='Bl Correction Method')
plt.grid(True)

plt.show()