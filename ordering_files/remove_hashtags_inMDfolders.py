

from global_files import csv_to_dictionary, public_variables as pv
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from pathlib import Path
import shutil
import os

# Function to check if a file should be kept
def delete_hash(MD_path):

    # Loop over the molecules (using padded numbers)
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        combined_path = MD_path / padded_num
        print(combined_path)
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            for file in combined_path.iterdir():  # Iterate over files in the directory
                if file.name.startswith("#"):
                    file.unlink()  # Delete the file
                    print(f"Deleted: {file}")


    return

def main():
    delete_hash(pv.MDsimulations_path_)
    return

if __name__ == "__main__":
    pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.JAK1)
    main()
    pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.GSK3)
    main()
    pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.pparD)
    main()
    pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.CLK4)
    main()

    