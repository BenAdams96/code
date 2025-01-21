

from global_files import public_variables
from pathlib import Path
import pandas as pd
import shutil
import os



def main():

    for protein in public_variables.list_dataset_proteins_:
        print(protein)
        for descriptor in public_variables.list_Descriptors_:
            print(descriptor)
            for path in public_variables.get_paths(protein, descriptor):
                if path.exists():
                    file_path = path / 'conformations_10.csv'
                    if file_path.exists():
                        print(f"Processing {file_path}")
                        
                        # Read the CSV file
                        df = pd.read_csv(file_path)
                        
                        # Sort by 'conformations (ns)' and 'mol_id'
                        df = df.sort_values(by=['mol_id', 'conformations (ns)'])
                        
                        # Define output path and write the sorted DataFrame
                        output_file_path = path / 'conformations_10_c.csv'
                        df.to_csv(output_file_path, index=False)
    return

if __name__ == "__main__":
    main()

    