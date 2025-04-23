from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import random
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
from pathlib import Path
from global_files import dataframe_processing
# Project-specific imports
from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

import pandas as pd
import math
import re
import os

def standardize_dataframe(df):
    """Preprocess the dataframe by handling NaNs and standardizing."""
    # Handle NaNs: drop rows with NaNs or fill them
    df_cleaned = df.dropna()  # or df.fillna(df.mean())

    # Identify which non-feature columns to keep
    non_feature_columns = ['mol_id','PKI','conformations (ns)']
    existing_non_features = [col for col in non_feature_columns if col in df_cleaned.columns]

    # Drop non-numeric target columns if necessary
    features_df = df_cleaned.drop(columns=existing_non_features, axis=1, errors='ignore')
    # Standardize the dataframe
    scaler = StandardScaler()
    features_scaled_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)

    # Concatenate the non-feature columns back into the standardized dataframe
    standardized_df = pd.concat([df_cleaned[existing_non_features].reset_index(drop=True), features_scaled_df.reset_index(drop=True)], axis=1)

    return standardized_df

def calculate_correlation_matrix(df):
    """Calculate the correlation matrix of a standardized dataframe."""
    df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
    return df.corr()


def correlation_matrix_single_csv(df):
    # Preprocess the dataframe: handle NaNs and standardize
    st_df = standardize_dataframe(df)
    
    # Calculate and visualize correlation matrix for the standardized dataframe
    correlation_matrix = calculate_correlation_matrix(st_df)
    return st_df, correlation_matrix

def compute_correlation_matrices_of_dictionary(dfs_dictionary, exclude_files: list=None):
    """
    Compute and visualize the correlation matrices for a list of dataframes.
    
    Args:
        dfs (list): List of dataframes to process.
        save_plot_folder (Path): Folder path where to save the plots.
    
    Returns:
        list of tuples: Each tuple contains (original_df, df_notargets, st_df, correlation_matrix).
    """
    print(f'correlation matrix of {dfs_dictionary.keys()}')

    standardized_dfs_dic = {}
    correlation_matrices_dic = {}
    
    for name, df in dfs_dictionary.items():
        print(f'correlation matrix of: {name}')
        
        st_df, correlation_matrix = correlation_matrix_single_csv(df)
        
        # visualize_matrix(correlation_matrix, dfs_path, name, title_suffix="Original")
        
        # Store the results for potential further use
        standardized_dfs_dic[name] = st_df
        correlation_matrices_dic[name] = correlation_matrix
    return standardized_dfs_dic, correlation_matrices_dic

def identify_columns_to_drop(correlation_matrix, st_df, variances, threshold):
    """Identify columns to drop based on correlation threshold and variance."""
    corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    columns_to_drop = set()
    processed_pairs = set()

    for i, j in zip(*corr_pairs):
        if i != j:  # Skip self-correlation
            pair = tuple(sorted((i, j)))  # Ensure consistent ordering
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                # Choose column to drop based on variance
                if variances[i] > variances[j]:
                    columns_to_drop.add(st_df.columns[j])
                else:
                    columns_to_drop.add(st_df.columns[i])
    
    return columns_to_drop

def identify_columns_to_drop_2_keep_lowest(correlation_matrix, df, variances, threshold):
    """Identify columns to drop based on correlation threshold and keeping the lowest indexed feature."""
    corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    columns_to_drop = set()
    processed_pairs = set()
    
    for i, j in zip(*corr_pairs):
        if i != j:  # Skip self-correlation
            pair = tuple(sorted((i, j)))  # Ensure consistent ordering
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                # Drop the column with the higher index
                if i < j:
                    columns_to_drop.add(df.columns[j])  # Drop column j (higher index)
                else:
                    columns_to_drop.add(df.columns[i])  # Drop column i (higher index)
    
    return columns_to_drop

def plot_scree_for_dfs_in_dict(dfs_in_dict, dfs_dPCA_path):
    for name, df in dfs_in_dict.items():
        plot_scree(name, df, threshold=0.98, title='Scree Plot', save_path=dfs_dPCA_path, max_components_to_plot=30)
    return

def plot_scree(name, df, threshold=0.98, title='Scree Plot', save_path=None, max_components_to_plot=30):
    """
    Creates a scree plot with an optional threshold line for cumulative explained variance.

    Parameters:
    - data: pandas DataFrame or NumPy array
    - threshold: float, cumulative variance cutoff (e.g., 0.98 for 98%)
    - title: str, plot title
    - save_path: str or Path, where to save the plot (optional)
    """
    # Standardize the dataframe
    standardized_df = standardize_dataframe(df)
    features_df = standardized_df.drop(columns=['mol_id', 'PKI', 'conformations (ns)'], errors='ignore')

    # Apply PCA
    pca = PCA(n_components=None)
    pca.fit(features_df)

    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    # Determine number of components to retain based on threshold
    n_keep = np.argmax(cum_var >= threshold) + 1

    # Limit the number of components to plot
    num_to_plot = min(max_components_to_plot, len(explained_var))
    x_range = range(1, num_to_plot + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x_range, cum_var[:num_to_plot] * 100, marker='o', color='royalblue', label='Cumulative Variance')

    if n_keep <= num_to_plot:
        plt.axvline(x=n_keep, color='green', linestyle='--', label=f'{n_keep} Components')
    plt.axhline(y=threshold * 100, color='red', linestyle='--', label=f'{int(threshold * 100)}% Threshold')

    plt.title(f'{title} - {name}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.xticks(x_range)
    plt.grid(True)
    plt.legend(loc='best')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path / f'{name}_screeplot.png')
    else:
        plt.show()

    plt.close()
    return n_keep

def get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dfs_dictionary, threshold):
    """
    Reduce dataframes based on correlation and visualize the reduced matrices.
    
    Args:
        correlation_matrices_dic (dict): Dictionary of correlation matrices.
        dfs_dictionary (dict): Dictionary of dataframes corresponding to the correlation matrices.
        threshold (float): Correlation threshold for dropping features.
        save_plot_folder_reduced (Path): Folder path where to save the reduced matrix plots.
    
    Returns:
        dict: Dictionary of reduced dataframes.
    """
    reduced_dfs_dictionary = {}
    
    for key in correlation_matrices_dic.keys():
        # Calculate variances for the non-standardized dataframe
        
        
        # Identify non-feature columns to retain
        non_feature_columns = ['mol_id','PKI','conformations (ns)']
        existing_non_features = [col for col in non_feature_columns if col in dfs_dictionary[key].columns]
        
        # Drop only the features for correlation analysis
        features_df = dfs_dictionary[key].drop(columns=existing_non_features, axis=1)
        variances = features_df.var()

        # Identify columns to drop based on high correlation and variance
        columns_to_drop = identify_columns_to_drop_2_keep_lowest(correlation_matrices_dic[key], features_df, variances, threshold)
        
        # Create the reduced dataframe by including the retained non-feature columns
        reduced_df = pd.concat([dfs_dictionary[key][existing_non_features], features_df], axis=1)
        reduced_df = reduced_df.drop(columns=columns_to_drop, axis=1)

        reduced_dfs_dictionary[key] = reduced_df
    
    return reduced_dfs_dictionary

def save_dataframes_to_csv(dic_with_dfs,save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dic_with_dfs.items():
        print(f"save dataframe: {name}")
        df.to_csv(save_path / f'{name}.csv', index=False)

def save_reduced_dataframes(dfs, base_path):
    dir = pv.dfs_reduced_path_
    final_path = base_path / dir
    final_path.mkdir(parents=True, exist_ok=True)
    timeinterval = pv.timeinterval_snapshots

    for i, x in enumerate(np.arange(0, len(dfs) * timeinterval, timeinterval)):
        if x.is_integer():
            x = int(x)
        print(f"x: {x}, i: {i}")
        dfs[i].to_csv(final_path / f'{x}ns.csv', index=False)

def load_results(csv_path):
    df = pd.DataFrame()
    return df

def remove_constant_columns_from_dfs(dfs_dictionary):
    cleaned_dfs = {}
    
    for key, df in dfs_dictionary.items():
        # Identify constant columns, excluding 'picoseconds' and 'conformations (ns)'
        constant_columns = df.columns[(df.nunique() <= 1) & ~df.columns.isin(['picoseconds', 'conformations (ns)'])]
        print(key)
        print(dfs_dictionary[key])
        print(constant_columns)
        if len(constant_columns) > 0:
            print(f"In '{key}', the following constant columns were removed: {', '.join(constant_columns)}")
        
        # Remove constant columns and keep only non-constant columns, excluding 'picoseconds' and 'conformations (ns)'
        non_constant_columns = df.loc[:, (df.nunique() > 1) | df.columns.isin(['picoseconds', 'conformations (ns)'])]
        cleaned_dfs[key] = non_constant_columns
    return cleaned_dfs


def PCA_for_dfs(dfs_dictionary, variance=0.90):
    dfs_dictionary_pca = {}
    for key, df in dfs_dictionary.items():
        
        # Standardize the dataframe
        standardized_df = standardize_dataframe(df)  # Assuming this function returns the standardized df

        # Drop non-feature columns for PCA
        features_df = standardized_df.drop(columns=['mol_id', 'PKI', 'conformations (ns)'], errors='ignore')

        # Apply PCA with enough components to explain the desired variance
        pca = PCA(n_components=None)  # None will allow PCA to calculate all components
        pca_result = pca.fit_transform(features_df)

        # Find the number of components that explain the desired variance
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        n_components = (cumulative_variance >= variance).argmax() + 1  # Find first component where cumulative variance exceeds the threshold

        # Apply PCA again with the determined number of components
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(features_df)

        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_{i+1}' for i in range(pca_result.shape[1])])

        # Re-add the non-feature columns to the PCA dataframe at the start
        non_feature_columns = ['mol_id', 'PKI', 'conformations (ns)']
        existing_non_feature_df = standardized_df[standardized_df.columns.intersection(non_feature_columns)].reset_index(drop=True)

        # Insert non-feature columns at the start of the PCA DataFrame
        pca_df = pd.concat([existing_non_feature_df.reset_index(drop=True), pca_df], axis=1)

        # Store the PCA results in the new dictionary
        dfs_dictionary_pca[key] = pca_df

        # Optionally, save PCA results to CSV
        # pca_df.to_csv(dfs_dictionary_pca / f'{key}', index=False)

    return dfs_dictionary_pca

def main(savefolder_name = pv.dfs_dPCA_var_path_, variance = 0.90, include = [0,1,'c10','c20'], write_out = True):
    if write_out:
        dfs_dPCA_path = savefolder_name
        savefolder_name.mkdir(parents=True, exist_ok=True)

    # dfs_dictionary = csv_to_dictionary.csvfiles_to_dic_include(pv.dfs_descriptors_only_path_,include_files=['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_10.csv'])#,'conformations_1000.csv','conformations_1000_molid.csv'])
    dfs_in_dict = csv_to_dictionary.create_dfs_dic(total_df=pv.initial_dataframe_,include=include)#,'conformations_1000.csv','conformations_1000_molid.csv'])
    dfs_in_dict = dataframe_processing.remove_constant_columns_from_dict_of_dfs(dfs_in_dict)
    
    dfs_in_dict_pca = PCA_for_dfs(dfs_in_dict, variance)

    # Reduce the dataframes based on correlation
    # reduced_dfs_in_dic = get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dfs_dictionary, threshold)
    #reduced dataframes including mol_ID and PKI. so for 0ns 1ns etc.
    if write_out:
        plot_scree_for_dfs_in_dict(dfs_in_dict, dfs_dPCA_path)
        save_dataframes_to_csv(dfs_in_dict_pca, save_path=dfs_dPCA_path)
    return dfs_in_dict_pca

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4)

    main(save_foldername = 'dPCA', variance = 0.98)
    
    # bigdf = pd.read_csv(public_variables.initial_dataframe)
    # dic = {}
    # dic['conformations_1000.csv'] = bigdf
    # standardized_dfs_dic, correlation_matrices_dic = compute_correlation_matrices_of_dictionary(dic)
    # print(correlation_matrices_dic)
    # reduced_dfs_in_dic = get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dic, threshold=0.65)
    # save_dataframes_to_csv(reduced_dfs_in_dic, save_path=public_variables.dfs_reduced_path_)

