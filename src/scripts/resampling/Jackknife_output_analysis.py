import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Load jackknife and bootstrap results
jackknife_path = '../../../temp/jackknife/060725/jackknife_spectra_reproducibility_50_10specimen.pkl'
bootstrap_path = '../../../temp/bootstrap/bootstrap_spectra_reproducibility_50_10specimen.pkl'

jackknife_data = pickle.load(open(jackknife_path, "rb"))
bootstrap_data = pickle.load(open(bootstrap_path, "rb"))

def compute_final_averaged_data(loaded_dict, type="jackknife"):
    final_averaged = {}
    for spectra_key, spectra_data in loaded_dict.items():
        #wavenumber_values = {i: [] for i in range(0, 17)}  # keys 3 to 20
        if type == "jackknife":
            wavenumber_values = {i: [] for i in range(0, 18)}
            for group_data in spectra_data.values():  # e.g. for keys 1 to 5
                for wn_key in range(0, 18):
                    if wn_key in group_data:
                        wavenumber_values[wn_key].append(group_data[wn_key])

        elif type == "bootstrap":
            wavenumber_values = {i: [] for i in range(3, 21)}
            for group_data in spectra_data.values():  # e.g. for keys 1 to 5
                for wn_key in range(3, 21):
                    if wn_key in group_data:
                        wavenumber_values[wn_key].append(group_data[wn_key])

        averaged_per_wavenumber = {
            wn_key: np.mean(values) for wn_key, values in wavenumber_values.items() if values
        }
        final_averaged[spectra_key] = averaged_per_wavenumber
    return final_averaged

jackknife_avg = compute_final_averaged_data(jackknife_data, type="jackknife")
bootstrap_avg = compute_final_averaged_data(bootstrap_data, type="bootstrap")

# Use union of all spectra keys
all_keys = sorted(set(jackknife_avg.keys()) | set(bootstrap_avg.keys()))
colors = cm.rainbow(np.linspace(0, 1, len(all_keys)))
color_map = {specimen: color for specimen, color in zip(all_keys, colors)}

# Plotting both side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Jackknife subplot
for specimen, data in jackknife_avg.items():
    left_spots = sorted(data.keys())
    rsd_values = [data[k] for k in left_spots]
    axes[0].plot(left_spots, rsd_values, marker='o', label=specimen, color=color_map[specimen])

axes[0].set_title("RSD Change With Jackknife Resampling")
axes[0].set_xlabel("Left-out spot")
axes[0].set_ylabel("Relative Std Dev")
axes[0].grid(True)
axes[0].set_xticks(range(min(left_spots), max(left_spots) + 1))

# Bootstrap subplot
for specimen, data in bootstrap_avg.items():
    left_spots = sorted(data.keys())
    rsd_values = [data[k] for k in left_spots]
    axes[1].plot(left_spots, rsd_values, marker='o', label=specimen, color=color_map[specimen])

axes[1].set_title("RSD Change With Bootstrap Resampling")
axes[1].set_xlabel("Resample Size")
axes[1].grid(True)
axes[1].set_xticks(range(min(left_spots), max(left_spots) + 1))
axes[1].invert_xaxis()

# Create one shared legend
fig.legend(
    handles=[plt.Line2D([0], [0], marker='o', color=color_map[key], label=key.replace("spectra.", ""), linestyle='') for key in reversed(all_keys)],
    loc='center left', bbox_to_anchor=(0.80, 0.5), title="Spectra Keys"
)

plt.tight_layout()
plt.subplots_adjust(right=0.79)  # Make space for legend
plt.show()