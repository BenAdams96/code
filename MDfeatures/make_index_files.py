
from pathlib import Path
import shutil
import subprocess
import pandas as pd
from io import StringIO
# import Afstuderen0.Afstuderen.removed.randomForest_add_MD_features as randomForest_add_MD_features
from global_files import public_variables
# import public_functions

import os
import re
import MDAnalysis as mda
# import MDAnalysis.analysis.psa

import numpy as np
from pathlib import Path

def make_index_files(MD_path):
    ''' function'''

    # Change the directory to MD_path once before the loop
    
    os.chdir(MD_path)
    user_input = 'q'

    for padded_num in (f"{i:03}" for i in range(1, 869)): #not abstract yet, not necessary
        tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
        ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
        combined_path = MD_path / padded_num
        if combined_path.exists() and combined_path.is_dir():
            print(f"The directory exists: {combined_path}")
            os.chdir(combined_path)
            print(tpr_file)
            if tpr_file.exists():
                # Construct the command and its arguments as a list
                command = ["gmx", "make_ndx", "-f", str(tpr_file), "-o", str(ndx_file)]
                
                # Run the command using subprocess and provide inputs interactively
                try:
                    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                        # Provide the inputs to the command
                        proc.communicate(input='1 name ligand\nq\n')
                    print(f"Index file created: {ndx_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running the command: {e}")
                except FileNotFoundError as e:
                    print(f"Command not found: {e}")
            else:
                print(f'tpr not present in: {padded_num}')
                continue
    return

def main(MDsimulations_path = public_variables.MDsimulations_path_):

    make_index_files(MDsimulations_path) #make index files
    return

if __name__ == "__main__":
    main()