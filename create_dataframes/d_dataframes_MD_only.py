# import randomForest_read_in_models

from sklearn.preprocessing import StandardScaler

# Project-specific imports
from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
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

def create_dataframes_MD_only(savefolder_name = 'MD only',include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20'], to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol'], write_out = True):
    if write_out:
        savefolder_path = pv.dataframes_master_ / savefolder_name
        savefolder_path.mkdir(parents=True, exist_ok=True)
    
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
    }
    MD_output_df.rename(columns=column_mapping, inplace=True)
    
    dfs_in_dict = dataframe_processing.create_dfs_dict(MD_output_path,to_keep, include = include)
    print(dfs_in_dict.keys())

    always_keep = ['mol_id', 'PKI']
    if write_out:
        dataframe_processing.save_dict_with_dfs(dict_with_dfs=dfs_in_dict, save_path=savefolder_path)
        # for name, df in list(dfs_in_dict.items()):
        #     if name.endswith('c') or name.endswith('ns'):
        #         # Save the cleaned DataFrame to a CSV
        #         df.to_csv(savefolder_path / Path(name + '.csv'), index=False)
        #         print(f'done with {name}')
        #     else:
        #         # If no cleaning is needed, just save the original DataFrame
        #         df.to_csv(savefolder_path / Path(name + '.csv'), index=False)
        #         continue
    return dfs_in_dict

def main(savefolder_name='MD only', include=[], to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol'], write_out = True):
    # include = ['CLt100_cl10_c10']
    dfs_in_dic = create_dataframes_MD_only(savefolder_name, include, to_keep, write_out)
    return dfs_in_dic

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main(savefolder_name='MD only', to_keep=['rmsd','Gyration','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    
