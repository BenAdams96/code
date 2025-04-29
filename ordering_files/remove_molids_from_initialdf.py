

from global_files import csv_to_dictionary,global_functions, public_variables as pv
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from pathlib import Path
import shutil
import os
import pandas as pd
from pathlib import Path
import pandas as pd
from pathlib import Path

def remove_mol_ids_from_chunk(chunk, mol_ids_to_remove):
    # Remove rows where 'mol_id' is in the list of mol_ids to remove
    return chunk[~chunk['mol_id'].isin(mol_ids_to_remove)]

def main(path, mol_ids_to_remove, chunk_size=10000):
    # Create an empty list to store filtered chunks
    filtered_chunks = []
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        # Process the chunk and remove the specified mol_ids
        filtered_chunk = remove_mol_ids_from_chunk(chunk, mol_ids_to_remove)
        filtered_chunks.append(filtered_chunk)
    
    # Concatenate the filtered chunks into a single dataframe
    df_filtered = pd.concat(filtered_chunks, ignore_index=True)
    
    print(f"Filtered dataframe head:\n{df_filtered.head()}")
    return df_filtered

if __name__ == "__main__":
    # Example list of mol_ids to remove
    mol_ids_to_remove = [46, 97, 168, 275, 305, 159, 394, 194, 491, 204, 566] #for clk4
    mol_ids_to_remove = [283,341,368,560,575,730] #for gsk3
    
    # Update the protein configuration (assuming pv and DatasetProtein are defined correctly elsewhere)
    pv.update_config(protein_=DatasetProtein.GSK3)
    
    # File path
    path = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataframes/dataframes_GSK3_WHIM/initial_dataframe.csv')
    
    # Apply the filtering function
    df_filtered = main(path, mol_ids_to_remove)
    
    # Save the filtered dataframe to a new file
    df_filtered.to_csv('/home/ben/Download/Afstuderen0/Afstuderen/dataframes/dataframes_GSK3_WHIM/initial_dataframe2.csv', index=False)



    