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

def get_targets(dataset):
    """read out the original dataset csv file and get the targets + convert exp_mean to PKI
    out: pandas dataframe with two columns: ['mol_id','PKI']
    """
    df = pd.read_csv(dataset)
    df['PKI'] = -np.log10(df['exp_mean [nM]'] * 1e-9)
    return df[['mol_id','PKI']]

def create_dfs_dic(totaldf,target_df, to_keep, include = [0,1,2,3,4,5,6,7,8,9,10,'10c','20c']):
    # Check if conformations or picoseconds
    df_dict = {}
    # time_interval, time_col = (1, "nanoseconds (ns)") if "nanoseconds (ns)" in totaldf.columns else (1000, "picoseconds") if "picoseconds" in totaldf.columns else (None, None)
    always_keep = ['mol_id', 'PKI', 'conformations (ns)']

    # Merge totaldf with target_df on 'mol_id'
    totaldf = pd.merge(totaldf, target_df, on='mol_id', how='left')
    
    # If the 'picoseconds' column exists, convert it to 'nanoseconds (ns)'
    if 'picoseconds' in totaldf.columns:
        totaldf['conformations (ns)'] = totaldf['picoseconds'] / 1000
        totaldf.drop(columns=['picoseconds'], inplace=True)
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
# def create_dfs_dic(totaldf, time_interval = 1):

#     #check if conformations or picoseconds
#     df_dict = {}

#     # Loop over the range from range_start to range_end (inclusive)
#     for i in np.arange(0,10+time_interval,time_interval):
#         i = round(i, 2)
#         if i.is_integer():
#             i = int(i)
#         # Create a new dataframe with rows where 'conformations (ns)' == i
#         filtered_df = totaldf[totaldf['conformations (ns)'] == i].copy()

#         # Drop the 'conformations (ns)' column
#         filtered_df.drop(columns=['conformations (ns)'], inplace=True)
        
#         # # Store the dataframe in the dictionary with a key like '0ns', '1ns', etc.
#         df_dict[f'{i}ns'] = filtered_df.reset_index(drop=True)

#     return df_dict

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
    MD_output_df = pd.read_csv(MD_output_path)

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
    MD_output_df.rename(columns=column_mapping, inplace=True)

    savefolder_path = pv.dfs_descriptors_only_path_.parent / savefolder_name
    savefolder_path.mkdir(parents=True, exist_ok=True)
    target_df = get_targets(pv.dataset_path_)
    dfs_in_dic = create_dfs_dic(MD_output_df,target_df,to_keep, include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20'])
    print(dfs_in_dic.keys())
 
    # if dfs_DescPCA_path.exists():
    #     if dfs_DescPCA_path.is_dir():
    #         print('already MD folder')
    #         # Remove the existing destination folder
    #         shutil.rmtree(dfs_DescPCA_path)

    # dfs_DescPCA_path.mkdir(parents=True, exist_ok=True)
    # dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_exclude(MD_output_path, exclude_files=['concat_ver.csv', 'concat_hor.csv','rdkit_min.csv','MD_output.csv', 'conformations_1000.csv']) # , '0ns.csv', '1ns.csv', '2ns.csv', '3ns.csv', '4ns.csv', '5ns.csv', '6ns.csv', '7ns.csv', '8ns.csv', '9ns.csv', '10ns.csv'
    # dfs_in_dic_concat = csv_to_dictionary.csvfiles_to_dic_exclude(reduced_MD_dataframes_folder, exclude_files=['rdkit_min', '0ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns'])
    # print(dfs_in_dic)
    sorted_keys_list = csv_to_dictionary.get_sorted_columns_small(list(dfs_in_dic.keys()))
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic}

    #csv_files = sorted([f for f in os.listdir(reduced_dataframes_folder) if f.endswith('.csv')], key=extract_number)

    always_keep = ['mol_id', 'PKI']

    for name, df in list(dfs_in_dic.items()):
        if name.endswith('c') or name.endswith('ns'):
            # Save the cleaned DataFrame to a CSV
            df.to_csv(savefolder_path / Path(name + '.csv'), index=False)
            print(f'done with {name}')
        else:
            # If no cleaning is needed, just save the original DataFrame
            df.to_csv(savefolder_path / Path(name + '.csv'), index=False)
            continue
    return



def main(savefolder_name, to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol']):
    #Works by copying the reduced+MD csv files and removes all features that are numeric.
    create_dataframes_MD_only(savefolder_name, to_keep)

    return

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    main(savefolder_name='MD_new only2', to_keep=['rmsd','Gyration','Schlitter Entropy','Quasiharmonic Entropy','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    main(savefolder_name='MD_new only2', to_keep=['rmsd','Gyration','Schlitter Entropy','Quasiharmonic Entropy','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
