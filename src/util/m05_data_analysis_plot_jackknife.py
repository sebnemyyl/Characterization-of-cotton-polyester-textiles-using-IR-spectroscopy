import matplotlib.pyplot as plt
import numpy as np

def plot_resampling(jackknife_bootstrap_agg, wn, type='jackknife'):
    # Line plot of RSDs
    # Iterate over each specimen in the nested dictionary
    if type == 'jackknife':
        for specimen, data in jackknife_bootstrap_agg.items():
            left_spots = list(data.keys())
            rsd_values = list(data.values())
            plt.plot(left_spots, rsd_values, marker='o', label=specimen)

        plt.title(f'Jackknife Sampling by Specimen for {wn}')
        plt.xlabel('Left-out spot')
        plt.ylabel('Relative Std Dev')
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.show()

    elif type == 'bootstrap':
        for specimen, data in jackknife_bootstrap_agg.items():
            left_spots = list(data.keys())
            rsd_values = list(data.values())
            avg_rsd_values = np.mean(rsd_values, axis=1)
            plt.plot(left_spots, avg_rsd_values, marker='o', label=specimen)

        plt.title(f'Bootstrap Sampling by Specimen for {wn}')
        plt.xlabel('Resample size')
        plt.ylabel('Relative Std Dev')
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.gca().invert_xaxis()
        plt.show()

def plot_resampling_top_n(loaded_dict , type='jackknife'):
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

    for specimen, data in averaged_data.items():
        left_spots = list(data.keys())
        rsd_values = list(data.values())
        plt.plot(left_spots, rsd_values, marker='o', label=specimen)

    if type == 'jackknife':
        plt.title(f'Jackknife Sampling by top 10 wavenumbers')
        plt.xlabel('Left-out spot')
        plt.ylabel('Relative Std Dev')
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.show()
    elif type == 'bootstrap':
        plt.title(f'Bootstrap Sampling by top 10 wavenumbers')
        plt.xlabel('Resample size')
        plt.ylabel('Relative Std Dev')
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.gca().invert_xaxis()
        plt.show()

def plot_p_values(averaged_pvalues):
    # Line plot of RSDs
    # Iterate over each specimen in the nested dictionary
    for spectra, specimen_p_value in averaged_pvalues.items():
        sample_size = list(specimen_p_value.keys())
        avg_p_values = list(specimen_p_value.values())
        plt.plot(sample_size, avg_p_values, marker='o', label=spectra)

    plt.title(f'Averaged p-value for top 10 wavenumber')
    plt.xlabel('Sample size')
    plt.ylabel('Avg p-value')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()