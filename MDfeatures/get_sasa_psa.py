
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


def make_PSA_index_files(MD_path):
    os.chdir(MD_path)
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #not abstract yet, not necessary pv.PROTEIN.dataset_length
        print(padded_num)
        pdb_file = MD_path / padded_num / f'{padded_num}_prod.pdb'
        ndx_file = MD_path / padded_num / f'index_PSA_{padded_num}.ndx'
        combined_path = MD_path / padded_num
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / padded_num)
            if pdb_file.exists():
                # Construct the command and its arguments as a list
                command = ["gmx", "make_ndx", "-f", str(pdb_file), "-o", str(ndx_file)]
                
                # Run the command using subprocess and provide inputs interactively
                try:
                    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                        # Provide the inputs to the command
                        proc.communicate(input='r 1 & a N* O* S*\nq\n')
                    print(f"Index file created: {ndx_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running the command: {e}")
                except FileNotFoundError as e:
                    print(f"Command not found: {e}")
            else:
                print(f'tpr not present in: {padded_num}')
                continue
    return

def run_gmx_sasa(MD_path, output_path):
    ''' create the 'xvg_files' folder to store the xvg files
        run the gmx energy command for every molecule
        !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    output_path.mkdir(parents=True, exist_ok=True)
    # create the path for the xvg file to be stored
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        print(padded_num)
        combined_path = MD_path / padded_num
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            sasa_file = output_path / f'{padded_num}.xvg'
            xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
            tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
            ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
            # If the file already exists, modify the filename to avoid overwriting
            if tpr_file.exists():
            # Generate a new folder for the xvg files with a suffix to avoid overwriting
                command = f'gmx sasa -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {sasa_file}'
                user_input = '2'
                #run the command which creates the xvg files
                subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
            else:
                continue
    return

def run_gmx_psa_sasa(MD_path, output_path):
    ''' create the 'xvg_files' folder to store the xvg files
        run the gmx energy command for every molecule
        !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    output_path.mkdir(parents=True, exist_ok=True)
    # create the path for the xvg file to be stored
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        print(padded_num)
        combined_path = MD_path / padded_num
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            sasa_file = output_path / f'{padded_num}.xvg'
            xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
            tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
            ndx_file = MD_path / padded_num / f'index_PSA_{padded_num}.ndx'
            # If the file already exists, modify the filename to avoid overwriting
            if tpr_file.exists():
            # Generate a new folder for the xvg files with a suffix to avoid overwriting
                command = f'gmx sasa -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {sasa_file}'
                user_input = '6'
                #run the command which creates the xvg files
                subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
            else:
                continue
    return

def sasa_xvg_files_to_csvfiles(energyfolder_path, sasa_xvgfolder_path):
    ''' go over all the xvg files in chronical order and create one big csv file which 
    '''
    all_data = pd.DataFrame()
    
    for xvg_file in sorted(sasa_xvgfolder_path.glob('*.xvg')):   #go over the xvg files/molecules in chronical order
        print(xvg_file)
        column_names = ['picoseconds']
        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    if line.startswith('@ s'):
                        # Extract the legend name
                        column_name = line.split('"')[1] #get "Bond" "Enthalpy" etc
                        column_names.append(column_name)
                else:
                    data_lines.append(line)
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines) #this is the complete xvg file but without the first 30 lines/comments or so

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        #NOTE: xvg file stays of equal length (independent of timeinterval_snapshot) so variable data as well
        #NOTE: variable 'filtered_dataframe' only contains the dataline at the time intervals we are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots*1000) == 0]
        # Rename the columns with the extracted legends
        filtered_dataframe.columns = column_names
        # add a column with the mol id
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])
        
        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        print(all_data)
    # concatenate the dataframes under eachother and save them in a csv file
    #
    
    #MDfeatures_allmol_csvfile_ = generate_new_csv_filename(sasa_xvgfolder_path, public_variables.MDfeatures_allmol_csvfile)
    return all_data

def psa_xvg_files_to_csvfiles(energyfolder_path, sasa_xvgfolder_path): # for PSA
    ''' go over all the xvg files in chronical order and create one big csv file which 
    '''
    all_data = pd.DataFrame()
    
    for xvg_file in sorted(sasa_xvgfolder_path.glob('*.xvg')):   #go over the xvg files/molecules in chronical order
        print(xvg_file)
        column_names = ['picoseconds']
        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    if line.startswith('@ s'):
                        # Extract the legend name
                        column_name = "PSA"
                        column_names.append(column_name)
                else:
                    data_lines.append(line)
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines) #this is the complete xvg file but without the first 30 lines/comments or so

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        #NOTE: xvg file stays of equal length (independent of timeinterval_snapshot) so variable data as well
        #NOTE: variable 'filtered_dataframe' only contains the dataline at the time intervals we are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots*1000) == 0]
        # Rename the columns with the extracted legends
        filtered_dataframe.columns = column_names
        # add a column with the mol id
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])
        
        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        
    # concatenate the dataframes under eachother and save them in a csv file
    #
    
    #MDfeatures_allmol_csvfile_ = generate_new_csv_filename(sasa_xvgfolder_path, public_variables.MDfeatures_allmol_csvfile)
    return all_data

def main(MDsimulations_path = pv.MDsimulations_path_):
    energyfolder_path = pv.energyfolder_path_
    SASA_xvg_dir = pv.energyfolder_path_ / 'SASA_xvg'
    PSA_xvg_dir = pv.energyfolder_path_ / 'PSA_xvg'

    make_PSA_index_files(MDsimulations_path)

    run_gmx_sasa(MDsimulations_path, SASA_xvg_dir)
    sasa_df = sasa_xvg_files_to_csvfiles(energyfolder_path, SASA_xvg_dir)

    run_gmx_psa_sasa(MDsimulations_path, SASA_xvg_dir)
    psa_df = psa_xvg_files_to_csvfiles(energyfolder_path, SASA_xvg_dir)

    return sasa_df, psa_df


if __name__ == "__main__":
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main(pv.ML_MODEL)






# def calculate_sasa(MD_path, lig_conf_system_path):
#     '''Function to compute hydrogen bonds using gmx hbond.'''

#     # Change the directory to MD_path once before the loop
#     os.chdir(MD_path)

#     df = pd.DataFrame(columns=['mol_id', 'picoseconds', 'num_of_hbonds', 'average_hbond_distance'])

#     # Iterate through the padded range of numbers
#     for padded_num in (f"{i:03}" for i in range(1, 869)):  # Adjust the range as needed
#         print(padded_num)
#         # Define file paths
#         combined_path = MD_path / padded_num
#         if combined_path.exists() and combined_path.is_dir():
#             os.chdir(MD_path / f'{padded_num}')
#             tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
#             ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
#             hbnum_file = pv.MDsimulations_path_ / f'{padded_num}' / 'hbnum.xvg'
#             hbond_file = pv.MDsimulations_path_ / f'{padded_num}' / f'{padded_num}_hbond.xvg'
#             if tpr_file.exists(): # and pdbfile_path.exists():
#                 for ns in range(0,11):
                
                
#                     pdbfile_path = lig_conf_system_path / f'{ns}ns' / f'{padded_num}_{ns}ns.pdb'
#                     if pdbfile_path.exists():
                    
#                         # Construct the command and its arguments as a list
#                         command = ["gmx", "hbond", "-f", str(pdbfile_path), "-s", str(tpr_file), "-n", str(ndx_file), "-dist", str(hbond_file)]
                        
#                         # Define the input strings for group selections
#                         # Assume group 2 is the donor and group 3 is the acceptor
#                         input_str = '2\n3\n'  # Adjust these numbers as needed
                        
#                         # Run the command using subprocess and provide inputs interactively
#                         try:
#                             with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
#                                 # Provide the inputs to the command
#                                 proc.communicate(input=input_str)  # Send the input strings
#                             #print(f"Hydrogen bond distances calculated: {hbond_file}")
#                             num_of_hbonds = read_out_hbnum(hbnum_file)
#                             average_distance = read_out_hbond_xvgfile(hbond_file)
#                             # Delete the files after processing
                            
#                             os.remove(hbnum_file)
#                             os.remove(hbond_file)

#                             df.loc[len(df)] = [padded_num, ns*1000, num_of_hbonds, average_distance]

#                         except subprocess.CalledProcessError as e:
#                             print(f"Error occurred while running the command: {e}")
#                         except FileNotFoundError as e:
#                             print(f"Command not found: {e}")
#                     else:
#                         print(f"Files not found: {tpr_file}, {pdbfile_path}")
#                         continue
#             else:
#                 print(f"{padded_num} not valid")
#                 continue
#     df.to_csv(pv.energyfolder_path_ / 'hbonds.csv', index=False)
#     return