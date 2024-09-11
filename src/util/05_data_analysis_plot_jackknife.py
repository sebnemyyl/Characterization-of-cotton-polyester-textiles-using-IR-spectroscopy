import matplotlib.pyplot as plt

def plot_jackknife(jackknife_bootstrap_agg, wn, type='jackknife'):
    # Line plot of RSDs
    # Iterate over each specimen in the nested dictionary
    for specimen, data in jackknife_bootstrap_agg.items():
        left_spots = list(data.keys())
        rsd_values = list(data.values())
        plt.plot(left_spots, rsd_values, marker='o', label=specimen)

    if type == 'jackknife':
        plt.title(f'Jackknife Sampling by Specimen for {wn}')
        plt.xlabel('Left-out spot')
        plt.ylabel('Relative Std Dev')
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.show()
    elif type == 'bootstrap':
        plt.title(f'Bootstrap Sampling by Specimen for {wn}')
        plt.xlabel('Resample size')
        plt.ylabel('Relative Std Dev')
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.show()