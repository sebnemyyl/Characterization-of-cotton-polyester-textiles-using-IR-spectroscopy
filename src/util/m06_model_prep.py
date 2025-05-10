import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, PredefinedSplit 
from sklearn.decomposition import PCA


def load_feature_set_from_csv(csv_file):
    baseline_corrected_data = pd.read_csv(csv_file, sep=',', header=0)
    # Convert the cotton column to numeric and handle errors
    baseline_corrected_data['reference.cotton'] = pd.to_numeric(baseline_corrected_data['reference.cotton'], errors='coerce')
    # Drop rows with missing cotton values
    data_clean = baseline_corrected_data.dropna(subset=['reference.cotton'])
    return data_clean

def get_baseline_corr_type(csv_file):
    name_without_ending = os.path.splitext(csv_file)[0]
    parts = name_without_ending.split("_")
    return parts[-1]

def run_pca(X_train, X_test, n_comps=50):
    pca = PCA(n_components=n_comps)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return (X_train_pca, X_test_pca)

# Prepare the feature set (exclude non-spectral columns) and
# remove spectra from column name
def get_X(data):
    X = data.loc[:,~data.columns.str.startswith('reference')]
    X = X.drop(columns=['Unnamed: 0'])
    X.columns = X.columns.str.replace('spectra.', '')
    return X

# Creates group list (combines cotton and specimen for unique specimen id)
def get_groups(data):
    groups = data['reference.cotton'] * 100000 + data['reference.specimen']
    unique_groups = np.unique(groups)
    print(f"Data set has {len(unique_groups)} number of unique specimen: {unique_groups}")
    return groups

def split_feature_set_randomly(data_clean, test_size = 0.2):
    # Prepare the feature set (exclude non-spectral columns)
    # Prepare the target column (cotton content)
    y = data_clean['reference.cotton']
    X_train, X_test, y_train, y_test = train_test_split(data_clean, y, test_size=test_size)
    groups_train = get_groups(X_train)
    X_train = get_X(X_train)
    X_test = get_X(X_test)
    return (X_train, X_test, y_train, y_test, groups_train)

def split_feature_set_with_attribute(data_clean, selected_attribute = 'reference.specimen', test_value = 1):
    # Put all measurements of certain column into test data set
    test_data = data_clean.loc[data_clean[selected_attribute] == test_value]
    print(f"number of samples in test: {len(np.unique(test_data["reference.cotton"]))}")
    X_test = get_X(test_data)
    training_data = data_clean[~data_clean.isin(test_data)].dropna()  
    groups_train = get_groups(training_data)
    X_train = get_X(training_data)

    # Prepare the target column (cotton content)
    y_test = test_data['reference.cotton']
    y_train = training_data['reference.cotton']
    return (X_train, X_test, y_train, y_test, groups_train)


def predefined_cv_split(groups):
    unique_groups = np.unique(groups)
    fold_to_groups = {}
    # The number of folds to be created
    n_folds=4
    for start in range(n_folds):
        # Groups are evenly split to folds according to their index
        fold_to_groups[start] = unique_groups[start::n_folds]
    print(f"Selected Groups to fold: {fold_to_groups}")

    # Automatically map groups to fold numbers
    group_to_fold = {group: fold for fold, group_list in fold_to_groups.items() for group in group_list}

    test_fold = np.array([group_to_fold[g] for g in groups])

    return PredefinedSplit(test_fold=test_fold)