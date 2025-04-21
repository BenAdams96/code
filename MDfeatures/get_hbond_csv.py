
from pathlib import Path
import shutil
import subprocess
import pandas as pd
from io import StringIO
# import Afstuderen0.Afstuderen.removed.randomForest_add_MD_features as randomForest_add_MD_features
# import public_functions

import os
import re
import MDAnalysis as mda
# import MDAnalysis.analysis.psa

import numpy as np
from pathlib import Path

from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

def calculate_hbond_dataframe_trajectory(MD_path,write_out):
    '''Function to compute hydrogen bonds using gmx hbond.'''

    # Change the directory to MD_path once before the loop
    os.chdir(MD_path)

    # List to store results
    result_list = []

    # Iterate through the padded range of numbers
    for mol in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)):  # Adjust the range as needed pv.PROTEIN.dataset_length+1
        # Define file paths
        xtc_file = MD_path / mol / f'{mol}_prod.xtc'
        tpr_file = MD_path / mol / f'{mol}_prod.tpr'
        ndx_file = MD_path / mol / f'{mol}_index.ndx'
        hbnum_file = MD_path / f'{mol}' / 'hbnum.xvg'
        hbond_file = MD_path / f'{mol}' / f'{mol}_hbond_traj.xvg'
        combined_path = MD_path / mol
        
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(combined_path)
            if tpr_file.exists():
                try:
                    command = f"gmx hbond -f {xtc_file} -s {tpr_file} -n {ndx_file} -dist {hbond_file}"
                    user_input = '\n'.join(["2", '3'])  # Adjust according to your input needs
                    subprocess.run(command, shell=True, input=user_input, capture_output=True, text=True)

                    # Assuming read_out_hbnum returns a DataFrame and a Series
                    num_of_hbonds_df, within_distance = read_out_hbnum(hbnum_file)

                    # Append results to the result_list for each timestep
                    # Append results efficiently
                    result_list.extend([
                        {'mol_id': mol, 'picoseconds': index * 10, 'num_of_hbonds': num_of_hbonds_df.at[index]} 
                        # 'non_bonding within 0.35': within_distance.at[index]}
                        for index in num_of_hbonds_df.index
                    ])

                    print(f'Done appending for {mol}')
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running the command: {e}")
                except FileNotFoundError as e:
                    print(f"Command not found: {e}")
            else:
                print(f"{mol} not valid")
                continue

    # Create a DataFrame from the result_list after processing all molecules
    final_df = pd.DataFrame(result_list)
    if write_out:
        final_df.to_csv(pv.energyfolder_path_ / 'hbonds.csv', index=False)
    return final_df

# def calculate_hbond_dataframe_trajectory(MD_path):
#     """Function to compute hydrogen bonds using gmx hbond efficiently."""

#     MD_path = Path(MD_path)  # Ensure MD_path is a Path object
#     result_list = []  # Store results in a list for efficient DataFrame creation

#     for i in range(1, pv.PROTEIN.dataset_length):  # Adjust as needed
#         mol = f"{i:03}"  # Format molecule index (e.g., 001, 002, etc.)
#         mol_path = MD_path / mol

#         xtc_file = mol_path / f'{mol}_prod.xtc'
#         tpr_file = mol_path / f'{mol}_prod.tpr'
#         ndx_file = mol_path / f'{mol}_index.ndx'
#         hbnum_file = mol_path / 'hbnum.xvg'
#         hbond_file = mol_path / f'{mol}_hbond_traj.xvg'

#         # Skip if the molecule directory or required files do not exist
#         if not mol_path.is_dir() or not tpr_file.exists():
#             print(f"Skipping {mol}, missing directory or tpr file.")
#             continue

#         try:
#             # Run gmx hbond
#             command = [
#                 "gmx", "hbond", "-f", str(xtc_file), "-s", str(tpr_file), 
#                 "-n", str(ndx_file), "-dist", str(hbond_file)
#             ]
#             process = subprocess.run(command, input="2\n3\n", text=True, capture_output=True, check=True)

#             # Read and process hbnum data
#             num_of_hbonds_df, within_distance = read_out_hbnum(hbnum_file)

#             # Append results efficiently
#             result_list.extend([
#                 {'mol_id': mol, 'picoseconds': index * 10, 'num_of_hbonds': num_of_hbonds_df.at[index], 
#                  'within_distance': within_distance.at[index]}
#                 for index in num_of_hbonds_df.index
#             ])

#             print(f'Done processing {mol}')
        
#         except subprocess.CalledProcessError as e:
#             print(f"Error processing {mol}: {e}")
#         except FileNotFoundError as e:
#             print(f"Missing command or file for {mol}: {e}")

#     # Convert list to DataFrame at the end for efficiency
#     return pd.DataFrame(result_list)

def read_out_hbnum(xvg_file):
    df = pd.read_csv(xvg_file, sep="\s+", comment='@', header=None, skiprows=17)
    
    # Assuming the second column contains the values you're interested in
    # Extract the value at the row you're interested in; in this case, the value on the first row
    num_of_hbonds = df.iloc[:, 1] if not df.empty else None
    within_distance = df.iloc[:, 2] if not df.empty else None

    return num_of_hbonds, within_distance

def main(MDsimulations_path = pv.MDsimulations_path_):
    hbond_df = calculate_hbond_dataframe_trajectory(MD_path=MDsimulations_path,write_out=True) #1 #use this one i guess. make sure export is okay
    return hbond_df


if __name__ == "__main__":
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main(pv.MDsimulations_path_)