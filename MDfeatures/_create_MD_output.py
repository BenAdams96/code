
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

from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

from MDfeatures import get_hbond_csv, get_rms, get_gyrate, get_dipoles, get_psa, get_sasa_psa, prepare_energy_files_from_MD, get_entropy


def merge_csv_files_on_columns(energyfolder_path, csv_filenames):
    merged_df = None
    merge_columns = ["mol_id", "picoseconds"]
    
    for filename in csv_filenames:
        file_path = os.path.join(energyfolder_path, filename)
        if merged_df is not None:
            print(len(merged_df))
        df = pd.read_csv(file_path)
        if 'picoseconds' not in df.columns:
            # Merge only on 'mol_id' if 'picoseconds' is not present
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='mol_id', how='inner')
            # Fill the 'picoseconds' column by creating a sequence of values for each molecule
            # merged_df['picoseconds'] = merged_df.groupby('mol_id').cumcount()
        else:
            # Merge on both 'mol_id' and 'picoseconds' if 'picoseconds' is present
            if merged_df is None:
                print('none')
                merged_df = df
            else:
                print('else')
                # print(merged_df[:10])
                # print(df[:10])

                merged_df = pd.merge(merged_df, df, on=merge_columns, how='inner')
    return merged_df

def main(protein = pv.PROTEIN):
    pv.update_config(protein_= protein)
    energyfolder_path = pv.energyfolder_path_
    energyfolder_path.mkdir(parents=True, exist_ok=True)

    file_list = ['hbonds.csv', 'rmsd.csv', 'gyration.csv', 'epsilon.csv','total_dipole_moment.csv', 'sasa.csv', 'psa.csv', f'MD_features_{pv.PROTEIN}.csv']
    MD_output_df = merge_csv_files_on_columns(energyfolder_path, file_list)
    MD_output_df = dataframe_processing.change_MD_column_names(MD_output_df)
    MD_output_df.to_csv(energyfolder_path / 'MD_output.csv', index=False)
    return


if __name__ == "__main__":
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main(pv.PROTEIN)
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    main(pv.PROTEIN)
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    main(pv.PROTEIN)
    # MD_output_df = pd.read_csv(pv.MD_outputfile_)
    # MD_output_df = change_column_names(MD_output_df)
    # MD_output_df.to_csv(pv.energyfolder_path_ / 'MD_output.csv', index=False)
    




############################## OLD STUFF ###############################
    
# def make_index_file2(path, pdb_filename):
#     ''' create the 'xvg_files' folder to store the xvg files
#         run the gmx energy command for every molecule
#         !!! NEEDS TO BE RUN ON LINUX/BASH !!!
#     '''

#     # create the path for the xvg file to be stored
#     xvg_file = xvgfolder_path / f'{molecule}.xvg'

#     # If the file already exists, modify the filename to avoid overwriting
#     if xvg_file.exists():
#         # Generate a new folder for the xvg files with a suffix to avoid overwriting
#         i = 1
#         xvgfolder_path = xvgfolder_path / f'_{i}.xvg'
#         while xvgfolder_path.exists():
#             i += 1

#     # Construct the command with the variable file prefix and output directory
#     command = f"gmx energy -f {molecule}_prod.edr -o {xvg_file}"

#     # specify the user input which are the features
#     user_input = '\n'.join(features)

#     if edrfolder_path.exists() and edrfolder_path.is_dir():
#         os.chdir(edrfolder_path)

#     #run the command which creates the xvg files
#     subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)

# def calculate_hbond_dataframe_trajectory(MD_path):
#     '''Function to compute hydrogen bonds using gmx hbond.'''

#     # Change the directory to MD_path once before the loop
#     os.chdir(MD_path)

#     final_df = pd.DataFrame(columns=['mol_id', 'picoseconds', 'num_of_hbonds'])
#     result_list = []
#     # Iterate through the padded range of numbers
#     for padded_num in (f"{i:03}" for i in range(1, 869)):  # Adjust the range as needed
#         print(padded_num)
#         # Define file paths
#         combined_path = MD_path / padded_num
#         if combined_path.exists() and combined_path.is_dir():
#             os.chdir(MD_path / f'{padded_num}')
#             xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
#             tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
#             ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
#             hbnum_file = public_variables.MDsimulations_path_ / f'{padded_num}' / 'hbnum.xvg'
#             hbond_file = public_variables.MDsimulations_path_ / f'{padded_num}' / f'{padded_num}_hbond_traj.xvg'
#             if tpr_file.exists(): # and pdbfile_path.exists():
#                 #gmx hbond -f 001_prod.xtc -s 001_prod.tpr -n 001_index.ndx -dist 001_hbondtesting.xvg
#                 # Construct the command and its arguments as a list
#                 command = ["gmx", "hbond", "-f", str(xtc_file), "-s", str(tpr_file), "-n", str(ndx_file), "-dist", str(hbond_file)]
                
#                 # Define the input strings for group selections
#                 # Assume group 2 is the donor and group 3 is the acceptor
#                 input_str = '2\n3\n'  # Adjust these numbers as needed
#                 # Run the command using subprocess and provide inputs interactively
#                 try:
#                     print('try')
#                     with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
#                         # Provide the inputs to the command
#                         proc.communicate(input=input_str)  # Send the input strings
#                         num_of_hbonds_df = read_out_hbnum(hbnum_file)
#                         result_df = pd.DataFrame({
#                             'mol_id': padded_num,
#                             'picoseconds': num_of_hbonds_df.index*10,     # Original index
#                             'num_of_hbonds': num_of_hbonds_df.values  # Values from num_of_hbonds
#                         })
                        
#                         print('append to result list')
#                         result_list.append(result_df)
#                         print('done appending')
#                 except subprocess.CalledProcessError as e:
#                             print(f"Error occurred while running the command: {e}")
#                 except FileNotFoundError as e:
#                     print(f"Command not found: {e}")
#             else:
#                 print(f"{padded_num} not valid")
#                 continue
#     final_df = pd.concat(result_list, ignore_index=True)
#     final_df.to_csv(public_variables.energyfolder_path_ / 'hbonds.csv', index=False)
#     return final_df