from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import random
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
from pathlib import Path

# Project-specific imports
from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

import pandas as pd
import math
import re
import os

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

def reduce_features_of_dfs_in_dict(correlation_matrices_dict, dfs_in_dict, threshold):
    """
    Reduce dataframes based on correlation and visualize the reduced matrices.
    
    Args:
        correlation_matrices_dict (dict): Dictionary of correlation matrices.
        dfs_dictionary (dict): Dictionary of dataframes corresponding to the correlation matrices.
        threshold (float): Correlation threshold for dropping features.
        save_plot_folder_reduced (Path): Folder path where to save the reduced matrix plots.
    
    Returns:
        dict: Dictionary of reduced dataframes.
    """
    reduced_dfs_in_dict = {}
    
    for key in correlation_matrices_dict.keys():
        # Calculate variances for the non-standardized dataframe
        
        
        # Identify non-feature columns to retain
        non_feature_columns = ['mol_id','PKI','conformations (ns)']
        existing_non_features = [col for col in non_feature_columns if col in dfs_in_dict[key].columns]
        
        # Drop only the features for correlation analysis
        features_df = dfs_in_dict[key].drop(columns=existing_non_features, axis=1)
        variances = features_df.var()

        # Identify columns to drop based on high correlation and variance
        columns_to_drop = identify_columns_to_drop_2_keep_lowest(correlation_matrices_dict[key], features_df, variances, threshold)
        
        # Create the reduced dataframe by including the retained non-feature columns
        reduced_df = pd.concat([dfs_in_dict[key][existing_non_features], features_df], axis=1)
        reduced_df = reduced_df.drop(columns=columns_to_drop, axis=1)

        reduced_dfs_in_dict[key] = reduced_df
    
    return reduced_dfs_in_dict

###############################################################################

def main(threshold = pv.correlation_threshold_, include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20'], write_out = True):
    '''xxx what is write out?
    '''
    if write_out:
        dfs_reduced_path = pv.dataframes_master_ / f'reduced_t{threshold}'  # e.g., 'dataframes_JAK1_WHIM
        dfs_reduced_path.mkdir(parents=True, exist_ok=True)

    dfs_in_dict = dataframe_processing.create_dfs_dict(pv.initial_dataframe_, include = include)
    dfs_in_dict = dataframe_processing.remove_constant_columns_from_dict_of_dfs(dfs_in_dict)
    std_dfs_dict, correlation_matrices_dic = dataframe_processing.correlation_matrices_of_dfs_in_dict(dfs_in_dict)

    # Reduce the dataframes based on correlation
    reduced_dfs_in_dict = reduce_features_of_dfs_in_dict(correlation_matrices_dic, dfs_in_dict, threshold)

    #reduced dataframes including mol_ID and PKI. so for 0ns 1ns etc.
    if write_out:
        dataframe_processing.save_dict_with_dfs(reduced_dfs_in_dict, save_path=dfs_reduced_path)
    return reduced_dfs_in_dict

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    main()

