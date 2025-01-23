

from global_files import csv_to_dictionary, public_variables as pv
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from pathlib import Path
import shutil
import os

# Function to check if a file should be kept
def should_keep(file_name):
    try:
        # Try converting the file name to a float
        value = float(file_name.rstrip("ns"))
        # Check if it's a whole number (1, 2, 3, etc.) or single-decimal (0.1, 0.2, etc.)
        return value.is_integer() or (value * 10).is_integer()
    except ValueError:
        # If conversion to float fails, ignore the file
        return False

def main():
    print(pv.ligand_conformations_path_)
    # Path to the directory containing the files
    directory = pv.ligand_conformations_path_

    # List all files in the directory
    folders = [file for file in Path(directory).iterdir()]

    # Filter files
    folders_to_keep = [file for file in folders if should_keep(file.name)]

    # Delete unwanted files
    for folder in folders:
        if folder not in folders_to_keep:
            print(f"Removing: {folder.name}")
            shutil.rmtree(folder)
    return

if __name__ == "__main__":
    pv.update_config(model_= Model_classic.RF, protein=DatasetProtein.GSK3)
    main()

    