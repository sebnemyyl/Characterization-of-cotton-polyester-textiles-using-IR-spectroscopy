import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def create_model_performance_pivot(regression_results, metric = "R2", title = "Heatmap"): 
    results_heatmap = regression_results[["model", "baseline_corr", metric]]
    pivot = results_heatmap.pivot_table(index = 'model', columns = 'baseline_corr', values = metric)
    return pivot

def show_pivot(pivot, title = "Heatmap"):
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".3f")
    plt.title(f'{title}')
    plt.show()
