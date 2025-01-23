

from global_files import public_variables as pv
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from pathlib import Path
import pandas as pd
import shutil
import os



def main():

    for protein in DatasetProtein:
        print(type(protein))
        print(protein)
        for descriptor in Descriptor:
            print(descriptor)
            for path in pv.get_paths(protein_=protein, descriptor_=descriptor):
                print(path)
                if path.exists():
                    file_path = path / 'conformations_10.csv'
                    if file_path.exists():
                        print(f"Processing {file_path}")
                        
                        # Read the CSV file
                        df = pd.read_csv(file_path)
                        
                        # Sort by 'conformations (ns)' and 'mol_id'
                        df = df.sort_values(by=['mol_id', 'conformations (ns)'])
                        
                        # Define output path and write the sorted DataFrame
                        output_file_path = path / 'conformations_10c.csv'
                        df.to_csv(output_file_path, index=False)
                        # if output_file_path.exists():
                        #     output_file_path.unlink()
                        # else:
                        #     continue
    return

if __name__ == "__main__":
    main()

    