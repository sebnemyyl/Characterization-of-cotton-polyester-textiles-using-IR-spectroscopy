import pandas as pd
from scipy.stats import f_oneway
import numpy as np
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import pybaselines

print(os.getcwd())

# Load the data
data = pd.read_csv('input/spectra_mir_240814.csv', sep=';', header=0)

related_data = data[(data['pet'] == 70) & (data['measuring_date'] >= 240626) & (data['specimen'].isin([1,2,3,4,5,6,7,8,9,10]))]
related_data = related_data.drop(['pet', 'cotton','area','spot','measuring_date','Unnamed: 0'], axis=1)

data_anova = related_data.replace(',', '.', regex=True)
data_anova = data_anova.astype(float)

##Baseline treatment
## Function to apply SNV
def apply_snv(spectrum):
    mean = np.mean(spectrum)
    std_dev = np.std(spectrum)
    return (spectrum - mean) / std_dev

# Apply SNV to each row (spectrum) in the DataFrame
snv_df = data_anova.iloc[:, :-1]
specimens = data_anova[data_anova.columns[-1]]

snv_transformed_df = snv_df.apply(apply_snv, axis=1)

#print(snv_transformed_df.head())

# Calculate the mean spectrum to identify peaks
mean_spectrum = snv_transformed_df.mean(axis=0)


# Find peaks in the mean spectrum
peaks, _ = find_peaks(mean_spectrum, height=0.9)
key_wavenumbers = snv_transformed_df.columns[peaks].astype(str).tolist()

print("Key Wavenumbers for ANOVA analysis: ", key_wavenumbers)
##
for specimen in data_anova.iterrows():
    a = np.array(specimen)
    asls_data = pybaselines.whittaker.asls(specimen)


##

# Extract relevant columns
selected_data = snv_transformed_df[key_wavenumbers]
selected_data_spcm = pd.concat([selected_data,specimens], axis=1)

grouped_data = selected_data_spcm.groupby('specimen').apply(lambda x: x.values.tolist(), include_groups=False)


## Variances by specimen
original_keys = grouped_data.index
stacked_array = np.stack(grouped_data.values)
transposed_array = stacked_array.transpose(0, 2, 1)

result_series = pd.Series([transposed_array[i] for i in range(transposed_array.shape[0])], index=original_keys)

variances_by_specimen = {}

for specimen in result_series.keys():
    temp_dict = result_series[specimen]
    variances = []
    for group in temp_dict:
        # Calculate the variance across all wavenumbers for this specimen
        specimen_variances = np.var(group)
        variances = np.append(variances, specimen_variances)
    variances_by_specimen[specimen] = np.average(variances)

print(f'Variance by specimen: {variances_by_specimen}')

## Plotting
plt.bar(range(len(variances_by_specimen)), list(variances_by_specimen.values()), align='center')
plt.xticks(range(len(variances_by_specimen)), list(variances_by_specimen.keys()))
plt.title('30% Cotton- 70% Polyester ~ MIR ~ Variances by Specimens')
plt.show()

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