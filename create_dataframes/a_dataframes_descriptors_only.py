import math
import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
import rdkit
# Project-specific imports
from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors

import pandas as pd
from pathlib import Path
import re
import pathlib

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

def create_df_multiple_conformations(df, interval=1):
    """
    Reduces the number of conformations per molecule in the dataframe
    by selecting only specific conformations at given intervals, excluding 0.
    
    Parameters:
        df (pd.DataFrame): The large dataframe containing all conformations.
        interval (float): The desired interval for selection, default is 1ns.
    
    Returns:
        pd.DataFrame: A reduced dataframe with only the specified conformations per molecule.
    """
    # Define the target conformations, starting from the first interval, excluding 0
    target_conformations = [round(i * interval, 2) for i in range(1, int(10 / interval) + 1)]
    
    # Filter the dataframe to only include rows with conformations in target_conformations
    reduced_df = df[df['conformations (ns)'].isin(target_conformations)].copy(False)
    
    return reduced_df

def main(time_interval = 1, include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20'], write_out = True):
    initial_df = pd.read_csv(pv.initial_dataframe_)

    # dfs_in_dict = create_dfs_dic(initial_df, time_interval) #only single conformations
    dfs_in_dict = dataframe_processing.create_dfs_dict(pv.initial_dataframe_, include = include)
    if write_out:
        dataframe_processing.save_dict_with_dfs(dfs_in_dict, pv.dfs_descriptors_only_path_) #automatically save in descriptors_only_folder
    
    # for t in timeinterval_list:
    #     print(t)
    #     reduced_dataframe = create_df_multiple_conformations(initial_df, interval=t)
    #     reduced_dataframe.to_csv(pv.dfs_descriptors_only_path_ / f'c{int(10/t)}.csv', index=False)
    return dfs_in_dict

if __name__ == "__main__":

    main(time_interval = 1)

