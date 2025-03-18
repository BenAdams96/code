import math
import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
import rdkit
from global_files import public_variables as pv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler

import pandas as pd
from pathlib import Path
import re
import pathlib
import numpy as np
from pathlib import Path

import glob
from global_files import public_variables
import pandas as pd
import math
import re

def standardize_dataframe(df):
    """Preprocess the dataframe by handling NaNs and standardizing."""
    # Handle NaNs: drop rows with NaNs or fill them
    df_cleaned = df.dropna().reset_index(drop=True)  # or df.fillna(df.mean())
    
    # Identify which non-feature columns to keep
    non_feature_columns = ['mol_id','PKI','conformations (ns)','picoseconds']
    existing_non_features = [col for col in non_feature_columns if col in df_cleaned.columns]
    print(existing_non_features)
    # Drop non-numeric target columns if necessary
    features_df = df_cleaned.drop(columns=existing_non_features, axis=1, errors='ignore')
    
    # Standardize the dataframe
    scaler = StandardScaler()
    features_scaled_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)

    # Create standardized DataFrame
    standardized_df = df_cleaned.copy()  # Copy to preserve original structure
    standardized_df.loc[:, features_df.columns] = features_scaled_df  # Replace feature columns
    # Concatenate the non-feature columns back into the standardized dataframe
    # standardized_df = pd.concat([df_cleaned[existing_non_features], features_scaled_df], axis=1)

    return standardized_df

def calculate_correlation_matrix(df):
    """Calculate the correlation matrix of a standardized dataframe."""
    df = df.drop(columns=['mol_id','PKI','conformations (ns)','picoseconds'], axis=1, errors='ignore')
    return df.corr()


def correlation_matrix_single_df(df):
    # Preprocess the dataframe: handle NaNs and standardize
    st_df = standardize_dataframe(df)
    
    # Calculate and visualize correlation matrix for the standardized dataframe
    correlation_matrix = calculate_correlation_matrix(st_df)
    return st_df, correlation_matrix

def correlation_matrices_of_dfs_in_dict(dfs_dictionary, exclude_files: list=None):
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
        
        st_df, correlation_matrix = correlation_matrix_single_df(df)
        
        # visualize_matrix(correlation_matrix, dfs_path, name, title_suffix="Original")
        
        # Store the results for potential further use
        standardized_dfs_dic[name] = st_df
        correlation_matrices_dic[name] = correlation_matrix
    return standardized_dfs_dic, correlation_matrices_dic

###############################################################################

def change_MD_column_names(MD_output):
    """Rename specific columns in the given DataFrame."""
    
    column_mapping = {
        'Total': 'SASA',
        'num_of_hbonds': 'num of H-bonds',
        'within_distance': 'H-bonds within 0.35A',
        'Mtot': 'Total dipole moment',
        'Bond': 'Ligand Bond energy',
        'U-B': 'Urey-Bradley energy',
        'Proper Dih.': 'Torsional energy',
        'Coul-SR:Other-Other': 'Coul-SR: Lig-Lig',
        'LJ-SR:Other-Other': 'LJ-SR: Lig-Lig',
        'Coul-14:Other-Other': 'Coul-14: Lig-Lig',
        'LJ-14:Other-Other': 'LJ-14: Lig-Lig',
        'Coul-SR:Other-SOL': 'Coul-SR: Lig-Sol',
        'LJ-SR:Other-SOL': 'LJ-SR: Lig-Sol',
        # Add more mappings as needed
    }

    MD_output = MD_output.rename(columns=column_mapping)
    return MD_output

###############################################################################
def remove_constant_columns_from_dict_of_dfs(dfs_dictionary):
    '''xxx
    '''
    cleaned_dfs = {}
    
    for name, df in dfs_dictionary.items():
        non_constant_columns = remove_constant_columns_from_df(df,name)
        cleaned_dfs[name] = non_constant_columns

    return  cleaned_dfs

def remove_constant_columns_from_df(df, name):
    '''xxx
    '''
    # Identify constant columns, excluding 'picoseconds' and 'conformations (ns)'
    constant_columns = df.columns[(df.nunique() <= 1) & ~df.columns.isin(['picoseconds', 'conformations (ns)'])]
    
    if len(constant_columns) > 0:
        print(f"In '{name}', the following constant columns were removed: {', '.join(constant_columns)}")
    
    # Remove constant columns and keep only non-constant columns, excluding 'picoseconds' and 'conformations (ns)'
    non_constant_columns = df.loc[:, (df.nunique() > 1) | df.columns.isin(['picoseconds', 'conformations (ns)'])]
    return non_constant_columns
###############################################################################

def get_targets(dataset):
    """read out the original dataset csv file and get the targets + convert exp_mean to PKI
    out: pandas dataframe with two columns: ['mol_id','PKI']
    xxx
    """
    df = pd.read_csv(dataset)
    df['PKI'] = -np.log10(df['exp_mean [nM]'] * 1e-9)
    return df[['mol_id','PKI']]

def create_dfs_dict(df_path, to_keep=None, include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20']):
    print(f'create dfs in dict for {df_path}')
    totaldf = pd.read_csv(df_path)
    target_df = get_targets(pv.dataset_path_)
    # Check if conformations or picoseconds
    dfs_in_dict = {}
    # time_interval, time_col = (1, "nanoseconds (ns)") if "nanoseconds (ns)" in totaldf.columns else (1000, "picoseconds") if "picoseconds" in totaldf.columns else (None, None)
    always_keep = ['mol_id', 'PKI', 'conformations (ns)']

    # Drop PKI from totaldf if it exists (so only the PKI from target_df will be kept)
    if 'PKI' in totaldf.columns:
        totaldf.drop(columns=['PKI'], inplace=True)
        
    # Merge totaldf with target_df on 'mol_id'
    totaldf = pd.merge(totaldf, target_df, on='mol_id', how='left')
    
    # If the 'picoseconds' column exists, convert it to 'nanoseconds (ns)'
    if 'picoseconds' in totaldf.columns:
        totaldf['conformations (ns)'] = totaldf['picoseconds'] / 1000
        totaldf.drop(columns=['picoseconds'], inplace=True)
    
    # If to_keep is empty, keep all columns except 'mol_id', 'PKI', and 'conformations (ns)'
    if not to_keep:
        to_keep = [col for col in totaldf.columns if col not in always_keep]

    columns_to_keep = always_keep + to_keep

    # Reorganize columns: 'mol_id', 'pKi', and 'conformations (ns)' are first, then to_keep columns
    totaldf = totaldf[columns_to_keep]
    for x in include:
        if isinstance(x, int):
            filtered_df = totaldf[totaldf['conformations (ns)'] == x].copy()

            # Store the DataFrame in the dictionary
            dfs_in_dict[str(x) + 'ns'] = filtered_df
        elif isinstance(x, str) and x.startswith("c"):
            num_conformations = int(x[1:])
            total_time = totaldf['conformations (ns)'].max()
            
            target_conformations = [round(i * (total_time / num_conformations),1) for i in range(1, num_conformations + 1)]
            filtered_df = totaldf[totaldf['conformations (ns)'].isin(target_conformations)].copy()
            # Store the DataFrame in the dictionary
            dfs_in_dict[x] = filtered_df
        elif isinstance(x, str) and x.startswith("ti"):
            time_part, conformations_part = x.split('c')
            start_time, end_time = map(int, time_part[2:].split('_'))
            num_conformations = int(conformations_part)
            print(start_time)
            print(end_time)
            print(num_conformations)
            stepsize = (end_time - start_time) / num_conformations
            target_conformations = np.arange(start_time + stepsize, end_time+stepsize, stepsize).round(1)
            filtered_df = totaldf[totaldf['conformations (ns)'].isin(target_conformations)].copy()
            # Store the DataFrame in the dictionary
            dfs_in_dict[x] = filtered_df
        elif isinstance(x, str) and x.startswith("ta"):
            bins, conformations_part = x.split('c')
            num_conformations = int(conformations_part)
            total_time = totaldf['conformations (ns)'].max()
            bins = int(bins[2:])
            stepsize_outer = total_time/bins
            for i in np.arange(0,total_time,stepsize_outer):
                start_time = i
                end_time = i+stepsize_outer
                stepsize_inner = (end_time - start_time) / num_conformations
                target_conformations = np.arange(start_time+stepsize_inner, end_time+(stepsize_inner*0.1), stepsize_inner).round(2)
                filtered_df = totaldf[totaldf['conformations (ns)'].isin(target_conformations)].copy()
                # Round the start_time and end_time and check if they are effectively integers
                if start_time.is_integer():
                    start_time = int(start_time)
                if end_time.is_integer():  
                    end_time = int(end_time)
                dfs_in_dict[f't{start_time}_{end_time}c{num_conformations}'] = filtered_df
    sorted_keys_list = sort_columns(list(dfs_in_dict.keys()))
    dfs_in_dict = {key: dfs_in_dict[key] for key in sorted_keys_list if key in dfs_in_dict}
    return dfs_in_dict
###############################################################################

def csvfiles_to_dict_include(dfs_path, include_files: list = []):
    '''xxx
    '''
    if include_files is None:
        include_files = []
    dict = {}
    for csv_file in dfs_path.glob('*.csv'):
        if csv_file.name in include_files:
            dict[csv_file.stem] = pd.read_csv(csv_file)
        else:
            continue
    return dict

###############################################################################
def categorize_columns(column_list):
    """Categorizes columns into predefined groups based on regex patterns."""
    patterns = {
        "ns": re.compile(r'^(\d+(\.\d+)?)ns$'),
        "conformations": re.compile(r'^c(\d+)$'),
        "minimized_conformations": re.compile(r'^minimized_conformations_(\d+)$'),
        "clustering": re.compile(r'^clustering_target(\d+)_cluster(\d+)$')
    }

    categorized = {key: [] for key in patterns}
    other_columns = []

    for col in column_list:
        matched = False
        for key, pattern in patterns.items():
            if pattern.match(col):
                categorized[key].append(col)
                matched = True
                break
        if not matched:
            other_columns.append(col)

    return categorized, other_columns

def sort_columns(column_list):
    """
    Sorts columns in the following order:
    1. 'ns' columns (numerically)
    2. 'c' (conformations) columns (numerically)
    3. 'minimized_conformations' columns (numerically)
    4. Other columns (alphabetically)
    5. 'clustering' columns (target descending, cluster ascending)
    """
    categorized, other_columns = categorize_columns(column_list)

    clustering_pattern = re.compile(r'^clustering_target(\d+)_cluster(\d+)$')

    sorted_columns = (
        sorted(categorized["ns"], key=lambda x: float(x[:-2])) +
        sorted(categorized["conformations"], key=lambda x: int(x[1:])) +
        sorted(categorized["minimized_conformations"], key=lambda x: int(x.split('_')[1])) +
        sorted(other_columns) +
        sorted(categorized["clustering"], key=lambda x: (
            -int(clustering_pattern.match(x).group(1)), 
            int(clustering_pattern.match(x).group(2))
        ))
    )

    return sorted_columns

###############################################################################
def save_dict_with_dfs(dict_with_dfs, save_path):
    '''save the dataframes that are in the dictionary to the path, using the key as filename'''
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dict_with_dfs.items():
        save_df(save_path=save_path, df=df, name=name)

def save_df(save_path, df, name):
    df.to_csv(save_path / f'{name}.csv', index=False)
###############################################################################

def main():
    return

if __name__ == "__main__":
    main()
