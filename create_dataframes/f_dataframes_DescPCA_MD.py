# Project-specific imports
from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from create_dataframes import d_dataframes_MD_only, e_dataframes_DescPCA

from pathlib import Path
import shutil
import pandas as pd
import re
import os

def main(savefolder_name = 'desc_PCA MD', include=[], components=20, to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol'], write_out = True):
    destination_folder = pv.dataframes_master_ / savefolder_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    dfs_in_dict_dPCA = e_dataframes_DescPCA.main(components=components, include = include, write_out = False)
    dfs_in_dict_MD = d_dataframes_MD_only.main(savefolder_name=savefolder_name, to_keep=to_keep,include = include, write_out = False)

    dfs_in_dict_merged = {}

    for key in dfs_in_dict_dPCA.keys():
        if key in dfs_in_dict_MD:
            dfs_in_dict_merged[key] = pd.merge(
                dfs_in_dict_dPCA[key], 
                dfs_in_dict_MD[key].drop(columns=['PKI'], errors='ignore'),
                on=['mol_id', 'conformations (ns)'],  # Adjust these keys based on your data
                how='inner'  # Use 'inner' if you only want matching rows
            )
        else:
            dfs_in_dict_merged[key] = dfs_in_dict_dPCA[key]  # Keep the original if no match
    if write_out:
        dataframe_processing.save_dict_with_dfs(dfs_in_dict_merged, save_path=destination_folder)
    return dfs_in_dict_merged

if __name__ == "__main__":
    main()
