from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import random
from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
# import csv_to_dataframes
# import csv_to_dictionary
# from csv_to_dataframes import csvfiles_to_dfs
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os

def preprocess_dataframe(df):
    """Preprocess the dataframe by handling NaNs and standardizing."""
    # Handle NaNs: drop rows with NaNs or fill them
    df_cleaned = df.dropna()  # or df.fillna(df.mean())
    
    # Drop non-numeric target columns if necessary
    df_notargets = df_cleaned.drop(['mol_id', 'PKI', 'conformations (ns)'], axis=1, errors='ignore')
    
    # Standardize the dataframe
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df_notargets), columns=df_notargets.columns)
    
    return standardized_df

def calculate_correlation_matrix(df):
    """Calculate the correlation matrix of a standardized dataframe."""
    return df.corr()

def visualize_matrix(matrix, save_plot_path, idx='test', title_suffix="test"):
    """Visualize a matrix (e.g., correlation matrix) with values inside squares if they exceed 0.85."""
    print(save_plot_path)
    save_plot_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('coolwarm')
    
    # Plot the matrix
    plt.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f'Matrix Visualization {title_suffix}')
    
    # Add value annotations only if |value| > 0.85
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if abs(value) > 0.99:  # Only display if above 0.85
                plt.text(j, i, f"{value:.2f}", ha='center', va='center', 
                         color='black' if abs(value) < 0.5 else 'white')

    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(range(len(matrix.columns)), matrix.columns)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_plot_path / f'matrix_{idx}.png')
    plt.close()
    # plt.show()

def compute_and_visualize_correlation_matrices_df(df, path, idx='test', title='test'):
    standardized_df = preprocess_dataframe(df)
    corr_matrix = calculate_correlation_matrix(standardized_df)
    visualize_matrix(corr_matrix, path, idx, title)
    return

def main(path):

    return


