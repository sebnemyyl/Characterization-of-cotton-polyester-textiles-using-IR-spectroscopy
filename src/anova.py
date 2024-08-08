import pandas as pd
from scipy.stats import f_oneway
import numpy as np
from scipy.signal import find_peaks

# Load the data
data = pd.read_csv('input/spectra_nir_240807.csv', sep=';', header=0)

related_data = data[(data['pet'] == 50) & (data['measuring_date'] == 240807)]
related_data = related_data.drop(['pet', 'cotton','area','spot','measuring_date','Unnamed: 0'], axis=1)

data_anova = related_data.replace(',', '.', regex=True)
data_anova = data_anova.astype(float)

# Function to apply SNV
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
peaks, _ = find_peaks(mean_spectrum, height=-0.695)  # Adjust height based on your data
key_wavenumbers = snv_transformed_df.columns[peaks].astype(str).tolist()

print("Key Wavenumbers for ANOVA analysis: ", key_wavenumbers)


# Extract relevant columns
selected_data = snv_transformed_df[key_wavenumbers]
selected_data_spcm = pd.concat([selected_data,specimens], axis=1)

grouped_data = selected_data_spcm.groupby('specimen').apply(lambda x: x.values.tolist())

# Debugging: Check the length of each group and compare with key_wavenumbers
for i, group in enumerate(grouped_data):
    print(f"Group {i} length: {len(group)} (Expected: {len(key_wavenumbers)})")

# Perform ANOVA on each wavenumber
results = {}
for i, wavenumber in enumerate(key_wavenumbers):
    print(i, wavenumber)
    try:
        # This line assumes each group has data for every key_wavenumber
        f_statistic, p_value = f_oneway(*[group[i] for group in grouped_data])
        results[wavenumber] = {'F-statistic': f_statistic, 'p-value': p_value}
    except IndexError as e:
        print(f"IndexError at wavenumber index {i}: {e}")
        break


# Output the results
for wavenumber, stats in results.items():
    print(f'Wavenumber: {wavenumber}, F-statistic: {stats["F-statistic"]}, p-value: {stats["p-value"]}')