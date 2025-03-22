from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

from global_files import public_variables, csv_to_dictionary
from extract_ligand_conformations import trj_to_pdbfiles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import itertools
import pickle
import seaborn as sns

from typing import List
from typing import Dict
import numpy as np
from pathlib import Path

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolfiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import pandas as pd
import math
import re
import os

def smiles_to_fingerprint_list(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string to an RDKit Morgan fingerprint."""
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    data = []
    for str_idx, smiles in smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            # fp_list = [int(bit) for bit in fp.ToBitString()]
            data.append((int(str_idx), fp))
    return data

def tanimoto_similarity_matrix(smiles_list):
    """Compute the Tanimoto similarity matrix for a list of SMILES strings."""
    fingerprints = smiles_to_fingerprint_list(smiles_list)

    # Handle None values (invalid SMILES)
    valid_mols = [i for i, fp in fingerprints if fp is not None]
    valid_fps = [fp for i, fp in fingerprints if fp is not None]

    n = len(valid_fps)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # Compute upper triangle only (matrix is symmetric)
            sim = DataStructs.TanimotoSimilarity(valid_fps[i], valid_fps[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Mirror it

    # Convert to a DataFrame for better visualization
    valid_smiles = [smiles_list[i][1] for i,smiles in enumerate(valid_mols)]
    df = pd.DataFrame(similarity_matrix, index=valid_smiles, columns=valid_smiles)
    return df

def plot_tanimoto_density(sim_matrix,valid=True):
    """Plot the density distribution of Tanimoto similarity scores."""
    # Extract the upper triangle of the matrix (excluding diagonal)
    similarity_values = sim_matrix.values[np.triu_indices(len(sim_matrix), k=1)]

    # Plot density
    sns.kdeplot(similarity_values, fill=True)
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Density")
    plt.title("Tanimoto Similarity Distribution")
    if valid:
        plt.savefig(pv.dataframes_master_ / f'tanimoto_similarity_{pv.PROTEIN}_validmols.png')
    else:
        plt.savefig(pv.dataframes_master_ / f'tanimoto_similarity_{pv.PROTEIN}_allmols.png')
    plt.close()
    
def main():
    # csv_file_path = pv.dataset_path_
    # df = pd.read_csv(csv_file_path)

    # smiles_list = df["smiles"].tolist()
    # print(smiles_list)
    # 
    for protein in DatasetProtein:
        pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=protein)
        df = pd.read_csv(pv.dataset_path_)
        smiles_list = [(f"{mol_id:03d}", smiles) for mol_id, smiles in zip(df['mol_id'], df['smiles'])]
        sim_matrix = tanimoto_similarity_matrix(smiles_list)
        plot_tanimoto_density(sim_matrix, valid=False)
    # csv_file_path = pv.dataset_path_
    # df = pd.read_csv(csv_file_path)
    # all_molecules_list, valid_mols, invalid_mols = trj_to_pdbfiles.get_molecules_lists(pv.MDsimulations_path_)
    
    # smiles_list = [(f"{mol_id:03d}", smiles) for mol_id, smiles in zip(df['mol_id'], df['smiles'])]
    # valid_smiles_list = [tuple for tuple in smiles_list if tuple[0] in valid_mols]
    # sim_matrix = tanimoto_similarity_matrix(smiles_list=valid_smiles_list)
    # plot_tanimoto_density(sim_matrix, valid=True)

    # sim_matrix = tanimoto_similarity_matrix(smiles_list=smiles_list)
    # plot_tanimoto_density(sim_matrix, valid=False)

    # plot_path = final_path / 'plots'
    # plot_path.mkdir(parents=True, exist_ok=True)
    # print(plot_path)
    # dfs_in_dic_t_vs_p = csv_to_dictionary.csvfiles_to_dic_include(final_path , include_files=['0ns.csv', '1ns.csv', 'conformations_10.csv', 'conformations_20.csv'])
    # sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic_t_vs_p.keys())) #RDKIT first
    # dfs_in_dic_t_vs_p = {key: dfs_in_dic_t_vs_p[key] for key in sorted_keys_list if key in dfs_in_dic_t_vs_p} #order
    # print(dfs_in_dic_t_vs_p.keys())

    # for name, df in dfs_in_dic_t_vs_p.items():
    #     true_pKi = df['True_pKi']
    #     predicted_pKi = df['Predicted_pKi']
    #     if 'conformations' in name:
    #         # Call the function for average predicted vs real pKi
    #         plot_avg_predicted_vs_real_pKi(df, name, plot_path)
    #         plot_predicted_vs_real_pKi(true_pKi, predicted_pKi, name, plot_path)
    #     else:
    #         # Call the original function for predicted vs real pKi
    #         plot_predicted_vs_real_pKi(true_pKi, predicted_pKi, name, plot_path)
    return

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main()
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    main()
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    main()