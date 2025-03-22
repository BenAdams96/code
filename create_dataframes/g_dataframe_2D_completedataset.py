from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

from global_files import dataframe_processing, global_functions, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def generate_fingerprints_dataframe(smiles_tuple_list, radius=2, n_bits=2048):
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    data = []
    for str_idx, smiles in smiles_tuple_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(smiles))
            fp_list = [int(bit) for bit in fp.ToBitString()]
            data.append((int(str_idx), *fp_list))  # Store SMILES and fingerprint as a list

    # Create DataFrame
    df = pd.DataFrame(data, columns=["mol_id"] + [f"bit_{i}" for i in range(n_bits)])
    return df

def main(dataset_path):

    # Ensure the output directory exists
    dataset_df = pd.read_csv(dataset_path)
    # Check if the 'mol_id' column exists
    if 'mol_id' not in dataset_df.columns:
        # If 'mol_id' doesn't exist, create it with zero-padded ascending values
        dataset_df['mol_id'] = [f"{i+1:03d}" for i in range(len(dataset_df))]
        dataset_df.to_csv(dataset_path, index=False)
    else:
        dataset_df['mol_id'] = dataset_df['mol_id'].apply(lambda x: f"{x:03d}") #3forward padding for the column mol_id

    # # Ensure the output directory exists
    dfs_2D_path = pv.dfs_2D_path
    dfs_2D_path.mkdir(parents=True, exist_ok=True)

    # all_molecules_list, valid_mols, invalid_mols = global_functions.get_molecules_lists(pv.MDsimulations_path_)
    smiles_list = global_functions.get_smiles_list(dataset_path=dataset_path)
    # valid_smiles_list = [tuple for tuple in smiles_list if tuple[0] in valid_mols]

    targets_df = dataframe_processing.get_targets(dataset_path) #
    # targets_df_valid = targets_df[dataset_df['mol_id'].isin(valid_mols)].reset_index(drop=True)
    # print(targets_df)
    fingerprint_df = generate_fingerprints_dataframe(smiles_list, radius=2, n_bits=2048)
    # print(fingerprint_df)

    fingerprint_df.insert(1, 'PKI', targets_df['PKI'])
    # print(fingerprint_df)

    fingerprint_df.to_csv(dfs_2D_path / f'2D_ECFP_{pv.PROTEIN}.csv', index=False)

if __name__ == "__main__":
    for protein in DatasetProtein:
        print(protein)
        pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=protein)
        main(pv.dataset_path_)
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dataset_path_)
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(pv.dataset_path_)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dataset_path_)


    # dataset_df = pd.read_csv(dataset_path)
    # dataset_df['mol_id'] = dataset_df['mol_id'].apply(lambda x: f"{x:03d}") #3forward padding for the column mol_id

    # # # Ensure the output directory exists
    # dfs_2D_path = pv.dfs_2D_path
    # dfs_2D_path.mkdir(parents=True, exist_ok=True)

    # all_molecules_list, valid_mols, invalid_mols = global_functions.get_molecules_lists(pv.MDsimulations_path_)
    # smiles_list = global_functions.get_smiles_list(dataset_path=dataset_path)
    # valid_smiles_list = [tuple for tuple in smiles_list if tuple[0] in valid_mols]

    # targets_df = dataframe_processing.get_targets(dataset_path) #
    # targets_df_valid = targets_df[dataset_df['mol_id'].isin(valid_mols)]
    # print(targets_df_valid.values)