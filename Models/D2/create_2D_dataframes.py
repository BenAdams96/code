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

    df_2d = pd.read_csv(dataset_path)

    # # Ensure the output directory exists
    dfs_2D_path = pv.dfs_2D_path
    dfs_2D_path.mkdir(parents=True, exist_ok=True)

    all_molecules_list, valid_mols, invalid_mols = global_functions.get_molecules_lists(pv.MDsimulations_path_)
    
    smiles_list = [(f"{mol_id:03d}", smiles) for mol_id, smiles in zip(df_2d['mol_id'], df_2d['smiles'])]
    targets_df = -np.log10(df_2d['exp_mean [nM]'] * 1e-9)
    df_2d['mol_id'] = df_2d['mol_id'].apply(lambda x: f"{x:03d}")
    targets_df_valid = targets_df[df_2d['mol_id'].isin(valid_mols)]
    
    valid_smiles_list = [tuple for tuple in smiles_list if tuple[0] in valid_mols]

    fp_df = generate_fingerprints_dataframe(valid_smiles_list, radius=2, n_bits=2048)
    fp_df.insert(1, 'PKI', targets_df_valid.values)
    fp_df.to_csv(dfs_2D_path / f'2D_ECFP_{pv.PROTEIN}.csv', index=False)

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    main(pv.dataset_path_)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(pv.dataset_path_)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dataset_path_)

