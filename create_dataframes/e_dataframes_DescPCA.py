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

# Project-specific imports
from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

import pandas as pd
import math
import re
import os

def PCA_for_dfs(dfs_dictionary, components):
    dfs_in_dict_pca = {}
    for key, df in dfs_dictionary.items():
        # Standardize the dataframe
        standardized_df = dataframe_processing.standardize_dataframe(df)  # Assuming this function returns the standardized df

        # Drop non-feature columns for PCA
        features_df = standardized_df.drop(columns=['mol_id', 'PKI', 'conformations (ns)'], errors='ignore')

        # Apply PCA
        pca = PCA(n_components=components)  # Use the specified number of components
        pca_result = pca.fit_transform(features_df)

        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_{i+1}' for i in range(pca_result.shape[1])])

        # Re-add the non-feature columns to the PCA dataframe at the start
        non_feature_columns = ['mol_id', 'PKI', 'conformations (ns)']
        existing_non_feature_df = standardized_df[standardized_df.columns.intersection(non_feature_columns)].reset_index(drop=True)

        # Insert non-feature columns at the start of the PCA DataFrame
        pca_df = pd.concat([existing_non_feature_df.reset_index(drop=True), pca_df], axis=1)

        # Store the PCA results in the new dictionary
        dfs_in_dict_pca[key] = pca_df

    return dfs_in_dict_pca

def main(components = 10, include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20'], write_out = True):
    if write_out:
        new_name = f"desc_PCA{components}"
        dfs_dPCA_path = pv.dataframes_master_ / new_name
        dfs_dPCA_path.mkdir(parents=True, exist_ok=True)

    dfs_in_dict = csv_to_dictionary.create_dfs_dic(pv.initial_dataframe_, include = include)
    dfs_in_dict = csv_to_dictionary.remove_constant_columns_from_dfs(dfs_in_dict)
    dfs_in_dic_pca = PCA_for_dfs(dfs_in_dict, components)
    if write_out:
        dataframe_processing.save_dict_with_dfs(dfs_in_dic_pca, save_path=dfs_dPCA_path)
    return dfs_in_dic_pca

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main(components = 20)

