

from global_files import public_variables
from pathlib import Path
import shutil
import os



def main():

    for protein in public_variables.list_dataset_proteins_:
        print(protein)
        for descriptor in public_variables.list_Descriptors_:
            print(descriptor)
            for path in public_variables.get_paths(protein, descriptor):
                if path.exists():
                    files = list(path.iterdir())
                    # Filter for files that start with 'clustering'
                    clustering_files = [f for f in files if f.name.startswith("clustering_t")]
                    # If there are clustering files, process them
                    if clustering_files:
                        # Define the new folder path
                        clustering_folder_path = path / "clustering_files"
                        
                        # Create the folder if it doesn't already exist
                        clustering_folder_path.mkdir(parents=True, exist_ok=True)
                        # Move each clustering file to the new folder
                        for file in clustering_files:
                            shutil.move(file, clustering_folder_path / file.name)
    return

if __name__ == "__main__":
    main()

    