import matplotlib.pyplot as plt

def plot_jackknife(jackknife_agg, wn):
    # Line plot of RSDs
    # Iterate over each specimen in the nested dictionary
    for specimen, data in jackknife_agg.items():
        left_spots = list(data.keys())
        rsd_values = list(data.values())
        plt.plot(left_spots, rsd_values, marker='o', label=specimen)

    plt.title(f'Jackknife Sampling by Specimen for {wn}')
    plt.xlabel('Left-out spot')
    plt.ylabel('Relative Std Dev')
    plt.grid(True)

    plt.show()
