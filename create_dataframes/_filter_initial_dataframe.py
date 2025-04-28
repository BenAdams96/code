# Standard library imports
import os
import re

# Third-party libraries
import numpy as np
import pandas as pd
from pathlib import Path
# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Project-specific imports
from global_files import public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from global_files import dataframe_processing
from extract_ligand_conformations import trj_to_pdbfiles

# %%
#NOTE: this file does: get targets, count how many valid molecules and which,it creates the folder 'dataframes_WHIMJAK1' or equivellant
def main():
    
    full_df = pd.read_csv(pv.initial_dataframe_)

    # Convert 'conformations (ns)' to float for correct sorting
    full_df['conformations (ns)'] = full_df['conformations (ns)'].astype(float)

    # Sort the DataFrame first by 'mol_id' and then by 'conformations (ns)' in ascending order
    full_df_sorted = full_df.sort_values(by=['mol_id', 'conformations (ns)'], ascending=[True, True]).reset_index(drop=True)
    path = pv.dataframes_master_ / Path('initial_dataframe_not_filtered.csv')
    full_df_sorted.to_csv(pv.initial_dataframe_, index=False)
    # Drop constant columns (columns where all values are the same)
    nunique = full_df_sorted.nunique()
    constant_columns = nunique[nunique <= 1].index
    full_df_sorted.drop(columns=constant_columns, inplace=True)
    print(f"Dropped constant columns: {list(constant_columns)}")
    
    df_cleaned = dataframe_processing.remove_low_cv_and_corr_columns_single_df(full_df_sorted, cv_threshold=0.01, corr_threshold=0.99)
    
    df_cleaned.to_csv(pv.initial_dataframe_, index=False)  # Overwrite with sorted version
    # # Identify and print columns with variance lower than 1%
    # # Exclude non-numeric columns before checking variance
    # numeric_df = full_df_sorted.select_dtypes(include=[np.number])
    # low_variance_cols = numeric_df.var()[numeric_df.var() < 0.01].index
    # print(f"Columns with variance lower than 1%: {list(low_variance_cols)}")
    # print(len(list(low_variance_cols)))
    # full_df_sorted_low_variance_dropped = full_df_sorted.drop(columns=low_variance_cols, inplace=False)
    # full_df_sorted_low_variance_dropped.to_csv(output_file.parent / 'initial_dataframe_lv.csv', index=False)
    # Write the sorted DataFrame back to the CSV file
    # full_df_sorted.to_csv(output_file, index=False)  # Overwrite with sorted version

if __name__ == "__main__":
    # Update public variables
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)

    # # Call main
    # main()
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.GETAWAY, protein_=DatasetProtein.pparD)

    # Call main
    main()



# %%
