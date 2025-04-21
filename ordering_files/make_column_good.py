# Standard library imports
import os
import re

# Third-party libraries
import numpy as np
import pandas as pd

# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Project-specific imports
from global_files import public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

from extract_ligand_conformations import trj_to_pdbfiles


def get_targets(dataset):
    """read out the original dataset csv file and get the targets + convert exp_mean to PKI
    out: pandas dataframe with two columns: ['mol_id','PKI']
    """
    df = pd.read_csv(dataset)
    df['PKI'] = -np.log10(df['exp_mean [nM]'] * 1e-9)
    return df[['mol_id','PKI']]
    
def make_column_good(df_path):
    print("Sorting full CSV...")
    full_df = pd.read_csv(df_path)

    # Convert 'conformations (ns)' to float for correct sorting
    full_df['conformations (ns)'] = full_df['conformations (ns)'].astype(float)

    # Now sort the full dataframe by 'mol_id' and 'conformations (ns)'
    full_df_sorted = full_df.sort_values(by=['mol_id', 'conformations (ns)']).reset_index(drop=True)

    # Write the sorted DataFrame back to the CSV file
    full_df_sorted.to_csv(df_path, index=False)  # Overwrite with sorted version
    return

def main():
    df_path = '/home/ben/Download/Afstuderen0/Afstuderen/dataframes/dataframes_GSK3_GETAWAY/initial_dataframe.csv'
    make_column_good(df_path)
    

if __name__ == "__main__":
    # Update public variables
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.GETAWAY, protein_=DatasetProtein.GSK3)

    # Call main
    main()
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.GETAWAY, protein_=DatasetProtein.GSK3)

    # Call main
    main()



# %%
