# Project-specific imports
from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from create_dataframes import b_dataframes_reduced, d_dataframes_MD_only, b_dataframes_reduced_redbefore

from pathlib import Path
import shutil
import pandas as pd
import re
import os

def MD_features_implementation(savefolder_name, include, threshold, to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol']):
    destination_folder = pv.dataframes_master_ / savefolder_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    reduced_dfs_in_dict = b_dataframes_reduced_redbefore.main(threshold, include = include, write_out = False)
    MD_dfs_in_dict = d_dataframes_MD_only.main(to_keep=to_keep, include = include, write_out = False)

    print(reduced_dfs_in_dict['0ns'])
    print(MD_dfs_in_dict['0ns'])

    merged_dfs_dict = {}

    for key in reduced_dfs_in_dict.keys():
        if key in MD_dfs_in_dict:
            merged_dfs_dict[key] = pd.merge(
                reduced_dfs_in_dict[key], 
                MD_dfs_in_dict[key].drop(columns=['PKI'], errors='ignore'),
                on=['mol_id', 'conformations (ns)'],  # Adjust these keys based on your data
                how='inner'  # Use 'inner' if you only want matching rows
            )
        else:
            merged_dfs_dict[key] = reduced_dfs_in_dict[key]  # Keep the original if no match
        
    #write out the dataframe to csv files
    dataframe_processing.save_dict_with_dfs(merged_dfs_dict, save_path=destination_folder)
    return

def main(savefolder_name = pv.dfs_reduced_and_MD_path_, include=[], threshold=0.85, to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol'], write_out = True):
    destination_folder = pv.dataframes_master_ / savefolder_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    reduced_dfs_in_dict = b_dataframes_reduced_redbefore.main(threshold, include = include, write_out = False)
    MD_dfs_in_dict = d_dataframes_MD_only.main(to_keep=to_keep, include = include, write_out = False)

    # print(reduced_dfs_in_dict['0ns'])
    # print(MD_dfs_in_dict['0ns'])

    merged_dfs_dict = {}

    for key in reduced_dfs_in_dict.keys():
        if key in MD_dfs_in_dict:
            merged_dfs_dict[key] = pd.merge(
                reduced_dfs_in_dict[key], 
                MD_dfs_in_dict[key].drop(columns=['PKI'], errors='ignore'),
                on=['mol_id', 'conformations (ns)'],  # Adjust these keys based on your data
                how='inner'  # Use 'inner' if you only want matching rows
            )
        else:
            merged_dfs_dict[key] = reduced_dfs_in_dict[key]  # Keep the original if no match
        
    #write out the dataframe to csv files
    if write_out:
        dataframe_processing.save_dict_with_dfs(merged_dfs_dict, save_path=destination_folder)
    return merged_dfs_dict

if __name__ == "__main__":
    main()
