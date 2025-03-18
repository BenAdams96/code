# import randomForest_read_in_models

from sklearn.preprocessing import StandardScaler

# Project-specific imports
from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
import numpy as np
from pathlib import Path
import shutil
from itertools import product

import pandas as pd
import math
import re
import os

def create_dfs_dic(totaldf, time_interval = 1):

    df_dict = {}

    # Loop over the range from range_start to range_end (inclusive)
    for i in np.arange(0,10+time_interval,time_interval):
        i = round(i, 2)
        if i.is_integer():
            i = int(i)
        # Create a new dataframe with rows where 'conformations (ns)' == i
        filtered_df = totaldf[totaldf['conformations (ns)'] == i].copy()

        # Drop the 'conformations (ns)' column
        filtered_df.drop(columns=['conformations (ns)'], inplace=True)
        
        # # Store the dataframe in the dictionary with a key like '0ns', '1ns', etc.
        df_dict[f'{i}ns'] = filtered_df.reset_index(drop=True)

    return df_dict

def extract_k_and_scoring(filename):
    # Use regex to find the pattern 'k' followed by digits and the scoring method
    match = re.search(r'k(\d+)_(.+)\.pkl', filename)
    if match:
        k_value = int(match.group(1))
        scoring_metric = match.group(2)
        return k_value, scoring_metric
    else:
        return None, None
    
def get_molecules_lists_temp(parent_path):
    folder = pv.dfs_descriptors_only_path_
    csv_file = '0ns.csv'
    final_path = parent_path / folder / csv_file
    molecules_list = []
    invalid_mols = []
    df = pd.read_csv(final_path)
    mol_id_column = df['mol_id']

    valid_mol_list_str = list(map(str, mol_id_column))
    print(valid_mol_list_str)
    print(len(valid_mol_list_str))
    return  valid_mol_list_str

def extract_number(filename):
    return int(filename.split('ns.csv')[0])

def copy_redfolder_only_csv_files(source_folder, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            # Copy each CSV file to the destination folder
            shutil.copy2(source_file, destination_file)

def concatenate_group(group):
    # Separate out the PKI and mol_id
    mol_id = group['mol_id'].iloc[0]
    pki_value = group['PKI'].iloc[0]
    
    # Drop 'PKI' and 'mol_id' from the group to avoid duplication in concatenation
    group = group.drop(columns=['mol_id', 'PKI'])
    
    # Concatenate the remaining columns horizontally (axis=1)
    concatenated = pd.concat([group.reset_index(drop=True)], axis=1).T.reset_index(drop=True)
    
    # Flatten the column names
    concatenated.columns = [f"{col}_{i}" for i, col in enumerate(concatenated.columns)]
    
    # Add the 'mol_id' and 'PKI' back to the start
    concatenated.insert(0, 'mol_id', mol_id)
    concatenated.insert(1, 'PKI', pki_value)
    
    return concatenated

def create_dataframes_MD_only(savefolder_name, to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol']):
    MD_output_path = pv.MD_outputfile_
    dfs_DescPCA_path = pv.dfs_descriptors_only_path_.parent / savefolder_name

    # if dfs_DescPCA_path.exists():
    #     if dfs_DescPCA_path.is_dir():
    #         print('already MD folder')
    #         # Remove the existing destination folder
    #         shutil.rmtree(dfs_DescPCA_path)

    dfs_DescPCA_path.mkdir(parents=True, exist_ok=True)

    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_exclude(MD_output_path, exclude_files=['concat_ver.csv', 'concat_hor.csv','rdkit_min.csv','MD_output.csv', 'conformations_1000.csv']) # , '0ns.csv', '1ns.csv', '2ns.csv', '3ns.csv', '4ns.csv', '5ns.csv', '6ns.csv', '7ns.csv', '8ns.csv', '9ns.csv', '10ns.csv'
    # dfs_in_dic_concat = csv_to_dictionary.csvfiles_to_dic_exclude(reduced_MD_dataframes_folder, exclude_files=['rdkit_min', '0ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns'])
    print(dfs_in_dic)
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys()))
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic}

    #csv_files = sorted([f for f in os.listdir(reduced_dataframes_folder) if f.endswith('.csv')], key=extract_number)
    column_mapping = {
        'Total': 'SASA',
        'num_of_hbonds': 'num of H-bonds',
        'within_distance': 'H-bonds within 0.35A',
        'Mtot': 'Total dipole moment',
        'Bond': 'Ligand Bond energy',
        'U-B': 'Urey-Bradley energy',
        'Proper dih.': 'Torsional energy',
        'Coul-SR:Other-Other': 'Coul-SR: Lig-Lig',
        'LJ-SR:Other-Other': 'LJ-SR: Lig-Lig',
        'Coul-14:Other-Other': 'Coul-14: Lig-Lig',
        'LJ-14:Other-Other': 'LJ-14: Lig-Lig',
        'Coul-SR:Other-SOL': 'Coul-SR: Lig-Sol',
        'Coul-SR:Other-SOL': 'Coul-SR: Lig-Sol',
        # Add more mappings as needed
    }

    always_keep = ['mol_id', 'PKI']

    for name, df in list(dfs_in_dic.items()):
        if name.startswith('conformations') or name.endswith('ns'):
            # Keep only the columns that are not numeric
            df_cleaned = df.loc[:, ~df.columns.str.isnumeric()]
            
            # Combine always-keep columns with the ones in the to_keep list
            columns_to_keep = always_keep + [col for col in to_keep if col in df_cleaned.columns]
            
            # Also keep 'conformations' if it exists
            if 'conformations (ns)' in df_cleaned.columns:
                columns_to_keep.append('conformations (ns)')
            
            # Filter the DataFrame to only keep the desired columns
            df_cleaned = df_cleaned[columns_to_keep]
            
            # Save the cleaned DataFrame to a CSV
            df_cleaned.to_csv(dfs_DescPCA_path / Path(name + '.csv'), index=False)
            print(f'done with {name}')
        else:
            # If no cleaning is needed, just save the original DataFrame
            df.to_csv(dfs_DescPCA_path / Path(name + '.csv'), index=False)
            continue
    return



def main(savefolder_name, to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol']):
    #Works by copying the reduced+MD csv files and removes all features that are numeric.
    create_dataframes_MD_only(savefolder_name, to_keep)

    return

if __name__ == "__main__":
    main()
