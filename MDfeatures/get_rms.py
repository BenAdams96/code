
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

def run_gmx_rms(MD_path, output_path):
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
            rmsd_file = output_path / f'{padded_num}.xvg'
            xtc_file = MD_path / padded_num / f'{padded_num}_prod.xtc'
            tpr_file = MD_path / padded_num / f'{padded_num}_prod.tpr'
            ndx_file = MD_path / padded_num / f'{padded_num}_index.ndx'
            # If the file already exists, modify the filename to avoid overwriting
            if tpr_file.exists():
            # Generate a new folder for the xvg files with a suffix to avoid overwriting
                command = f'gmx rms -f {xtc_file} -s {tpr_file} -n {ndx_file} -o {rmsd_file}'
                user_input = '2\n2\n'
                #run the command which creates the xvg files
                subprocess.run(command, shell=True, input=user_input,capture_output=True, text=True)
            else:
                continue
    return

def rms_xvg_files_to_csvfiles(rmsd_xvgfolder_path):
    ''' go over all the xvg files in chronical order and create one big csv file which 
    '''
    all_data = pd.DataFrame()
    
    for xvg_file in sorted(rmsd_xvgfolder_path.glob('*.xvg')):   #go over the xvg files/molecules in chronical order
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
                        column_name = 'rmsd'
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
    # all_data.to_csv(energyfolder_path / 'rms.csv', index=False)
    return all_data

def main(MDsimulations_path = pv.MDsimulations_path_):
    RMSD_xvg_dir = pv.energyfolder_path_ / 'RMSD_xvg'
    run_gmx_rms(MDsimulations_path, RMSD_xvg_dir)
    rmsd_df = rms_xvg_files_to_csvfiles(RMSD_xvg_dir)
    return rmsd_df


if __name__ == "__main__":
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main(pv.ML_MODEL)