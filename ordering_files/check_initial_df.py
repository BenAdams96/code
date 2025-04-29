

from global_files import csv_to_dictionary,global_functions, public_variables as pv
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from pathlib import Path
import shutil
import os

import pandas as pd

def check_mol_id_repetitions(df):
    # Group the dataframe by 'mol_id' and count the occurrences
    mol_id_counts = df['mol_id'].value_counts()

    # Check if any mol_id is not repeated exactly 1001 times
    non_compliant_mol_ids = mol_id_counts[mol_id_counts != 1001]

    if non_compliant_mol_ids.empty:
        print("All mol_id values appear exactly 1001 times.")
    else:
        print("The following mol_id values are not repeated 1001 times:")
        print(non_compliant_mol_ids)

def check_missing_mol_ids(df):
    # Check for missing mol_id values in the range 1 to 731
    expected_mol_ids = set(range(1, 732))
    actual_mol_ids = set(df['mol_id'].unique())

    missing_mol_ids = expected_mol_ids - actual_mol_ids
    if missing_mol_ids:
        print(f"Missing mol_id values: {sorted(missing_mol_ids)}")
    else:
        print("All mol_id values from 1 to 731 are present.")

    # Get molecules list from global functions (assuming global_functions and pv are defined elsewhere)
    molecules_list, valid_mols, invalid_mols = global_functions.get_molecules_lists(pv.MDsimulations_path_)
    
    # Convert missing mol_ids to strings, padding with leading zeros to match invalid_mols format
    missing_mol_ids_str = {str(mol_id).zfill(3) for mol_id in missing_mol_ids}

    # Find the difference between the missing mol_ids and invalid mols
    missing_vs_invalid = missing_mol_ids_str - set(invalid_mols)
    
    if missing_vs_invalid:
        print(f"Missing mol_id values not in invalid_mols: {sorted(missing_vs_invalid)}")
    else:
        print("All missing mol_ids are in the invalid mols list.")
    
    # Print invalid mols for reference
    print(f"Invalid mol_ids: {invalid_mols}")

def main(path):
    df = pd.read_csv(path)
    check_mol_id_repetitions(df)
    check_missing_mol_ids(df)
    return

if __name__ == "__main__":
    path = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataframes/dataframes_CLK4_GETAWAY/initial_dataframe.csv')
    pv.update_config(protein_= DatasetProtein.CLK4)
    main(path)

    path = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataframes/dataframes_GSK3_WHIM/initial_dataframe.csv')
    pv.update_config(protein_= DatasetProtein.GSK3)
    main(path)
    # pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.GSK3)
    # main()
    # pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.pparD)
    # main()

    