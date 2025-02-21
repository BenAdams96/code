from pathlib import Path
from global_files import public_variables
import shutil
import subprocess
import pandas as pd
from io import StringIO
import numpy as np
import os

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
    return dic

def get_all_targets(protein_dataset):
    """read out the original dataset csv file and get the targets + convert exp_mean to PKI
    out: pandas dataframe with two columns: ['mol_id','PKI']
    """
    df = pd.read_csv(protein_dataset)
    df['PKI'] = -np.log10(df['exp_mean [nM]'] * 1e-9)
    return df[['mol_id','PKI']]

def get_molecules_lists(MDsimulations_path):
    '''uses the MD_simulations folder and checks for every molecule the simulation whether it contains the .tpr and .xtc file
        to see if it contains a valid trajectory file (.tpr)
        output: lists of all the molecules, valid molecules, and invalid molecules in strings
        '''
    print('get molecules list, valid_mols, invalid_mols')
    molecules_list = []
    valid_mols = []
    invalid_mols = []
    for item in MDsimulations_path.iterdir(): #item is every folder '001' etc in the MD folder
        xtcfile = f"{item.name}_prod.xtc"  # trajectory file
        trajectory_file = item / xtcfile

        if item.is_dir():  # Check if the item is a directory
            molecules_list.append(item.name)
        if not trajectory_file.exists():
            invalid_mols.append(item.name)
        else:
            valid_mols.append(item.name)

    return molecules_list, valid_mols, invalid_mols #['001','002']

def main(base_path):
    
    return

if __name__ == "__main__":
    base_path = public_variables.base_path_
    main(base_path)

