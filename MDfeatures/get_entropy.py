
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

def run_entropy(MD_path):

    ''' 
    Create the 'xvg_files' folder to store the xvg files.
    Run the gmx gyrate command for every molecule.
    !!! NEEDS TO BE RUN ON LINUX/BASH !!!
    '''
    #output_path.mkdir(parents=True, exist_ok=True)  # Create the folder for storing xvg files
    mol_ids = []
    schlitter_list = []
    quasiharmonic_list = []
    # Loop over the molecules (using padded numbers)
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        # print(padded_num)
        combined_path = MD_path / padded_num
        print(combined_path)
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
            tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
            eigenvalues_xvgfile = 'eigenvalues.xvg'
            eigenvalues_trrfile = 'eigenvalues.trr'
            entropy_output_txt = f'entropy_output_{pv.PROTEIN}.txt'
            # If the .tpr file exists, run the gmx gyrate command
            if tpr_file.exists():

                # Construct the GROMACS gyrate command
                command = f'gmx covar -f {xtc_file} -s {tpr_file} -o {eigenvalues_xvgfile} -v {eigenvalues_trrfile}'
                user_input = '2\n2'  # Select the appropriate group (typically group 1 is the molecule or complex)
                
                # Run the command to calculate the radius of gyration
                subprocess.run(command, shell=True, input=user_input, capture_output=True, text=True)
            else:
                continue
            
            command = f'gmx anaeig -v {eigenvalues_trrfile} -entropy -s {tpr_file} > {entropy_output_txt}'
            user_input = '2\n2'  # Select the appropriate group (typically group 1 is the molecule or complex)
                
            # Run the command to calculate the radius of gyration
            subprocess.run(command, shell=True, input=user_input, capture_output=True, text=True)
            with open(entropy_output_txt, "r") as file:
                text = file.read()

            # Extract entropy values using regex
            schlitter_entropy = re.search(r"Schlitter formula is ([\d.]+)", text)
            quasiharmonic_entropy = re.search(r"Quasiharmonic analysis is ([\d.]+)", text)

            # Convert to float and append to lists
            mol_ids.append(padded_num)
            schlitter_list.append(float(schlitter_entropy.group(1)) if schlitter_entropy else None)
            quasiharmonic_list.append(float(quasiharmonic_entropy.group(1)) if quasiharmonic_entropy else None)
        # Create DataFrame outside the loop
    df = pd.DataFrame({
        "mol_id": mol_ids,
        "Schlitter Entropy": schlitter_list,
        "Quasiharmonic Entropy": quasiharmonic_list
    })
    return df

def gyration_xvg_files_to_csvfiles(gyration_xvgfolder_path):
    ''' Go over all the xvg files in chronological order and create one big CSV file. 
        For .xvg files with multiple columns, only keep the first (picoseconds) and second columns.
    '''
    all_data = pd.DataFrame()  # Initialize an empty DataFrame
    
    for xvg_file in sorted(gyration_xvgfolder_path.glob('*.xvg')):   # Go over the xvg files in chronological order
        print(xvg_file)
        column_names = ['picoseconds', 'Gyration']  # Adjust column names for the first and second columns

        # Read the file and filter out lines starting with '#' or '@', while extracting legends
        with open(xvg_file, 'r') as file:
            lines = file.readlines()
            data_lines = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    continue  # Skip comment and metadata lines
                else:
                    data_lines.append(line)  # Collect data lines
        
        # Join the filtered lines into a single string
        data_str = ''.join(data_lines)  # This is the complete xvg file but without comments

        # Use StringIO to read the string as if it were a file
        data = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

        # Select only the first (picoseconds) and second columns (SASA)
        data = data[[0, 1]]

        # Filter the data to match the time intervals you are interested in
        filtered_dataframe = data #data[data[0] % (public_variables.timeinterval_snapshots * 1000) == 0]

        # Rename the columns
        filtered_dataframe.columns = column_names

        # Add a column with the molecule ID (from the filename)
        filtered_dataframe.insert(0, 'mol_id', xvg_file.name.rsplit('.', 1)[0])

        # Append the selected columns to the all_data DataFrame
        all_data = pd.concat([all_data, filtered_dataframe], ignore_index=True)
        #print(all_data)

    # Save the concatenated data as a CSV file
    
    return all_data

def main(MDsimulations_path):
    print(f'entropy for {pv.PROTEIN}')
    print(MDsimulations_path)
    df = run_entropy(MDsimulations_path)
    df.to_csv(pv.energyfolder_path_ / 'entropy.csv', index=False)
    return


if __name__ == "__main__":
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main()

    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    main(pv.MDsimulations_path_)

    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    main(pv.MDsimulations_path_)
