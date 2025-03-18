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

#NOTE: takes in a folder with csv files 'dataframes_WHIMJAK1' (so 0ns.csv, 1ns.csv) and will convert it to a dictionary with keys the foldername and value a panda dataframe
# so, takes in 'dataframes_WHIMJAK1' or 'dataframes_WHIMJAK1_with0.85', and creates 
def main(folder_name = public_variables.dfs_descriptors_only_path_, exclude_files=None):
    base_path = public_variables.base_path_
    dfs_path = Path(base_path) / folder_name
    
    dic = csvfiles_to_dic_exclude(dfs_path, exclude_files)
    return dic

def remove_constant_columns_from_dfs(dfs_dictionary):
    cleaned_dfs = {}
    
    for key, df in dfs_dictionary.items():
        # Identify constant columns, excluding 'picoseconds' and 'conformations (ns)'
        constant_columns = df.columns[(df.nunique() <= 1) & ~df.columns.isin(['picoseconds', 'conformations (ns)'])]
        
        if len(constant_columns) > 0:
            print(f"In '{key}', the following constant columns were removed: {', '.join(constant_columns)}")
        
        # Remove constant columns and keep only non-constant columns, excluding 'picoseconds' and 'conformations (ns)'
        non_constant_columns = df.loc[:, (df.nunique() > 1) | df.columns.isin(['picoseconds', 'conformations (ns)'])]
        cleaned_dfs[key] = non_constant_columns
    return cleaned_dfs

def get_targets(dataset):
    """read out the original dataset csv file and get the targets + convert exp_mean to PKI
    out: pandas dataframe with two columns: ['mol_id','PKI']
    """
    df = pd.read_csv(dataset)
    df['PKI'] = -np.log10(df['exp_mean [nM]'] * 1e-9)
    return df[['mol_id','PKI']]

def create_dfs_dic(total_df, to_keep=None, include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20']):
    totaldf = pd.read_csv(total_df)
    target_df = get_targets(pv.dataset_path_)
    # Check if conformations or picoseconds
    df_dict = {}
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
        print(totaldf.columns)
        to_keep = [col for col in totaldf.columns if col not in always_keep]
        print(to_keep)

    columns_to_keep = always_keep + to_keep

    # Reorganize columns: 'mol_id', 'pKi', and 'conformations (ns)' are first, then to_keep columns
    totaldf = totaldf[columns_to_keep]
    print(totaldf)
    for x in include:
        if isinstance(x, int):
            filtered_df = totaldf[totaldf['conformations (ns)'] == x].copy()

            # Store the DataFrame in the dictionary
            df_dict[str(x) + 'ns'] = filtered_df
        elif isinstance(x, str) and x.startswith("c"):
            num_conformations = int(x[1:])
            total_time = totaldf['conformations (ns)'].max()
            
            target_conformations = [round(i * (total_time / num_conformations),1) for i in range(1, num_conformations + 1)]
            filtered_df = totaldf[totaldf['conformations (ns)'].isin(target_conformations)].copy()
            # Store the DataFrame in the dictionary
            df_dict[x] = filtered_df

    return df_dict

def csvfiles_to_dic_exclude(dfs_path, exclude_files: list = []):
    '''The folder with CSV files 'dataframes_JAK1_WHIM' is the input and these CSV files will be
       put into a list of pandas DataFrames, also works with 0.5 1 1.5 formats.
    '''
    if exclude_files is None:
        exclude_files = []
    dic = {}
    for csv_file in dfs_path.glob('*.csv'):
        if csv_file.name not in exclude_files:
            dic[csv_file.stem] = pd.read_csv(csv_file)
        else:
            continue
            # print(f'name {csv_file} is in exclude')
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    # pattern = re.compile(r'^\d+(\.\d+)?ns\.csv$')

    # for csv_file in sorted(dfs_path.glob('*.csv'), key=lambda x: extract_number(x.name)):
    #     print(csv_file)
    #     if pattern.match(csv_file.name):  # Check if the file name matches the pattern
    #         print(f"Reading {csv_file}")
    #         # Read CSV file into a DataFrame and append to the list
    #         dfs.append(pd.read_csv(csv_file))
    return dic


def csvfiles_to_dic_include(dfs_path, include_files: list = []):
    '''The folder with CSV files 'dataframes_JAK1_WHIM' is the input and these CSV files will be
       put into a list of pandas DataFrames, also works with 0.5 1 1.5 formats.
    '''
    if include_files is None:
        include_files = []
    dic = {}
    for csv_file in dfs_path.glob('*.csv'):
        if csv_file.name in include_files:
            dic[csv_file.stem] = pd.read_csv(csv_file)
        else:
            continue
            # print(f'name {csv_file} is in exclude')
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    # pattern = re.compile(r'^\d+(\.\d+)?ns\.csv$')

    # for csv_file in sorted(dfs_path.glob('*.csv'), key=lambda x: extract_number(x.name)):
    #     print(csv_file)
    #     if pattern.match(csv_file.name):  # Check if the file name matches the pattern
    #         print(f"Reading {csv_file}")
    #         # Read CSV file into a DataFrame and append to the list
    #         dfs.append(pd.read_csv(csv_file))
    return dic

# def csvfile_to_df(csvfile):
#     return df

def get_sorted_folders(base_path):
    '''The folder with CSV files 'dataframes_JAK1_WHIM' is the input and these CSV files will be
       put into a list of pandas DataFrames, also works with 0.5 1 1.5 formats.
    '''
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    sorted_folders = []
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    pattern = re.compile(r'^\d+(\.\d+)?ns$')

    for csv_file in sorted(base_path.glob('*'), key=lambda x: extract_number(x.name)):
        if pattern.match(csv_file.name):  # Check if the file name matches the pattern
            sorted_folders.append(csv_file)
        else:
            sorted_folders.insert(0,csv_file)
    return sorted_folders

# # def csvfile_to_df(csvfile):
# #     return df

def extract_number(filename):
    # Use regular expression to extract numeric part (integer or float) before 'ns.csv'
    match = re.search(r'(\d+(\.\d+)?)ns$', filename)
    if match:
        number_str = match.group(1)
        # Convert to float first
        number = float(number_str)
        # If it's an integer, convert to int
        if number.is_integer():
            return int(number)
        return number
    else:
        return float('inf')


def get_sorted_folders_namelist(file_list):
    '''
    This function takes a list of CSV filenames and returns a sorted list of filenames.
    Files with numeric values before 'ns.csv' are sorted based on these values.
    Files without 'ns.csv' are placed at the beginning of the list.
    '''
    # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
    pattern = re.compile(r'^(\d+(\.\d+)?)ns$')

    # Sort the files based on the extracted number or position them at the beginning
    sorted_files = sorted(file_list, key=lambda x: extract_number2(x, pattern))

    return sorted_files

def extract_number2(filename, pattern):
    '''
    Extracts the numeric value from filenames matching the pattern before 'ns.csv'.
    Returns float('inf') for filenames that do not match the pattern.
    '''
    match = pattern.search(filename)
    if match:
        number_str = match.group(1)
        number = float(number_str)
        return number
    else:
        return float('-inf')
    

def get_sorted_columns_small(column_list):
    """
    Sorts columns based on the following order:
    1. Columns with 'ns', sorted numerically.
    2. Columns with 'c' (e.g., 'c10', 'c20'), sorted numerically.
    3. Columns with 'minimized_conformations', sorted numerically.
    4. Columns with 'clustering', sorted by target descending and cluster ascending.
    5. Columns that don't match any pattern, sorted alphabetically.
    """
    # Define the regex patterns
    ns_pattern = re.compile(r'^(\d+(\.\d+)?)ns$')
    conformations_pattern = re.compile(r'^c(\d+)$')  # Updated to match 'c10', 'c20', etc.
    minimized_conformations_pattern = re.compile(r'^minimized_conformations_(\d+)$')
    clustering_pattern = re.compile(r'^clustering_target(\d+)_cluster(\d+)$')

    # Separate the columns into categories
    ns_columns = [col for col in column_list if ns_pattern.match(col)]
    conformations_columns = [col for col in column_list if conformations_pattern.match(col)]
    minimized_conformations_columns = [col for col in column_list if minimized_conformations_pattern.match(col)]
    clustering_columns = [col for col in column_list if clustering_pattern.match(col)]
    other_columns = [col for col in column_list if not any(
        pattern.match(col) for pattern in [ns_pattern, conformations_pattern, minimized_conformations_pattern, clustering_pattern]
    )]

    # Sort 'ns' columns by the numeric value before 'ns'
    sorted_ns = sorted(ns_columns, key=lambda x: float(x[:-2]))

    # Sort 'conformations' columns by the numeric value after 'c'
    sorted_conformations = sorted(conformations_columns, key=lambda x: int(x[1:]))

    # Sort 'minimized_conformations' columns by the numeric value after 'minimized_conformations_'
    sorted_minimized_conformations = sorted(minimized_conformations_columns, key=lambda x: int(x.split('_')[1]))

    # Sort 'clustering' columns by target descending and cluster ascending
    sorted_clustering = sorted(clustering_columns, key=lambda x: (
        -int(clustering_pattern.match(x).group(1)),  # Descending target
        int(clustering_pattern.match(x).group(2))   # Ascending cluster
    ))

    # Sort 'other' columns alphabetically
    sorted_other = sorted(other_columns)

    # Combine sorted lists
    sorted_columns = sorted_ns + sorted_conformations + sorted_minimized_conformations + sorted_other + sorted_clustering

    return sorted_columns


def get_sorted_columns(column_list):
    """
    Sorts columns based on the following order:
    1. Columns with 'ns', sorted numerically.
    2. Columns with 'conformations', sorted numerically.
    3. Columns with 'minimized_conformations', sorted numerically.
    4. Columns with 'clustering', sorted by target descending and cluster ascending.
    5. Columns that don't match any pattern, sorted alphabetically.
    """
    import re
    # Define the regex patterns
    ns_pattern = re.compile(r'^(\d+(\.\d+)?)ns$')
    conformations_pattern = re.compile(r'^conformations_(\d+)$')
    minimized_conformations_pattern = re.compile(r'^minimized_conformations_(\d+)$')
    clustering_pattern = re.compile(r'^clustering_target(\d+)_cluster(\d+)$')

    # Separate the columns into categories
    ns_columns = [col for col in column_list if ns_pattern.match(col)]
    conformations_columns = [col for col in column_list if conformations_pattern.match(col)]
    minimized_conformations_columns = [col for col in column_list if minimized_conformations_pattern.match(col)]
    clustering_columns = [col for col in column_list if clustering_pattern.match(col)]
    other_columns = [col for col in column_list if not any(
        pattern.match(col) for pattern in [ns_pattern, conformations_pattern, minimized_conformations_pattern, clustering_pattern]
    )]

    # Sort 'ns' columns by the numeric value before 'ns'
    sorted_ns = sorted(ns_columns, key=lambda x: float(x[:-2]))

    # Sort 'conformations' columns by the numeric value after 'conformations_'
    sorted_conformations = sorted(conformations_columns, key=lambda x: int(x.split('_')[1]))

    # Sort 'minimized_conformations' columns by the numeric value after 'minimized_conformations_'
    sorted_minimized_conformations = sorted(minimized_conformations_columns, key=lambda x: int(x.split('_')[2]))

    # Sort 'clustering' columns by target descending and cluster ascending
    sorted_clustering = sorted(clustering_columns, key=lambda x: (
        -int(clustering_pattern.match(x).group(1)),  # Descending target
        int(clustering_pattern.match(x).group(2))   # Ascending cluster
    ))

    # Sort 'other' columns alphabetically
    sorted_other = sorted(other_columns)

    # Combine sorted lists
    sorted_columns = sorted_ns + sorted_conformations + sorted_minimized_conformations + sorted_other + sorted_clustering

    return sorted_columns

    

if __name__ == "__main__":
    
    main()
