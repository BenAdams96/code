
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

from MDfeatures import get_hbond_csv, get_rms, get_gyrate, get_dipoles, get_psa, get_sasa_psa, prepare_energy_files_from_MD

def make_index_files(MD_path):
    ''' function'''

    # Change the directory to MD_path once before the loop
    os.chdir(MD_path)
    user_input = 'q'

    for mol in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length)): #not abstract yet, not necessary
        tpr_file = MD_path / mol / f'{mol}_prod.tpr'
        ndx_file = MD_path / mol / f'{mol}_index.ndx'
        combined_path = MD_path / mol
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(combined_path) #change path to the path of the molecule
            if ndx_file.exists():
                os.remove(ndx_file)
            if tpr_file.exists():
                # Construct the command and its arguments as a list
                command = ["gmx", "make_ndx", "-f", str(tpr_file), "-o", str(ndx_file)]
                # Run the command using subprocess and provide inputs interactively
                try:
                    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
                        # Provide the inputs to the command
                        proc.communicate(input='name 1 ligand\nq\n')
                    print(f"Index file created: {ndx_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running the command: {e}")
                except FileNotFoundError as e:
                    print(f"Command not found: {e}")
            else:
                print(f'tpr not present in: {mol}')
                continue
    return

def merge_csv_files_on_columns(folder_path, csv_filenames):
    """
    Merge multiple CSV files into a single DataFrame based on specified columns.

    Parameters:
    - folder_path: str, the folder where the CSV files are located.
    - csv_filenames: list of str, the names of the CSV files to merge.
    - merge_columns: list of str, the columns to use as keys for the merge.

    Returns:
    - DataFrame: merged DataFrame containing data from all CSV files.
    """
    merged_df = None
    merge_columns = ["mol_id" , "picoseconds"]
    # Loop through the list of CSV file names
    for i, filename in enumerate(csv_filenames):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Perform the merge
        if merged_df is None:
            # For the first file, initialize the merged DataFrame
            merged_df = df
        else:
            # Merge on the specified columns
            merged_df = pd.merge(merged_df, df, on=merge_columns, how='inner')

    return merged_df





def read_out_hbond_xvgfile(xvg_file):
    df = pd.read_csv(xvg_file, sep="\s+", comment='@', header=None, skiprows=17)
        
    # Remove any remaining comment lines starting with '@'
    # Extract columns (assuming the first column is distance and the second column is frequency)

    distances = df[0]
    frequencies = df[1]
    
    # Calculate the weighted average distance
    total_weight = round(frequencies.sum())
    if total_weight == 0:
        return 0
    else:
        weighted_average_distance = (distances * frequencies).sum() / total_weight
    return weighted_average_distance





def concatenate_csv_files(folder_path, csv_filenames):
    """
    Concatenate multiple CSV files into a single DataFrame.

    Parameters:
    - folder_path: str, the folder where the CSV files are located.
    - csv_filenames: list of str, the names of the CSV files to concatenate.

    Returns:
    - DataFrame: concatenated DataFrame containing data from all CSV files.
    """
    # List to store individual DataFrames
    dataframes = []

    # Loop through the list of CSV file names
    for i, filename in enumerate(csv_filenames):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Drop the first two columns
        if i == 0:
            df = df.iloc[:, :]
        else:
            df = df.iloc[:, 2:]
        # Append the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames in the list horizontally (i.e., along columns)
    concatenated_df = pd.concat(dataframes, axis=1)
    
    return concatenated_df



def calculate_psa_from_trajectory(MD_path, output_file=None, interval_ns=1):
    # Load the topology and trajectory into an MDAnalysis Universe

    interval_ps = interval_ns * 1000

    sorted_files = sorted(MD_path.iterdir(), key=lambda x: int(x.stem))
    for molecule_path in sorted_files:
        xtc_files = list(molecule_path.glob("*.xtc"))
        if xtc_files: #check if valid MD simulation has run
            molecule_number = molecule_path.name

            topology_file = molecule_path / f'{molecule_number}_prod.tpr'
            trajectory_file = molecule_path / f'{molecule_number}_prod.xtc'
            
            if molecule_number == '021':
                l = []
                print('molecule')
                u = mda.Universe(topology_file, trajectory_file)
                # for res in u.residues:
                #     l.append(res.resname)
                polar_atoms = u.select_atoms('resid 1 and (name O* or name N*)')
                psa_results = []

                for ts in u.trajectory:
                    if ts.time % 1000 == 0:
                        print(mda.__version__)
                        # sasa_calculator = sasa.SASA(u, selection=f'resname {ligand_resname} and (name O or name N)')
                # print(polar_atoms)
                # for atom in u.select_atoms('resid 1'):
                #     print(atom.name)
                # print(l)
                # print(len(l))


                # # Initialize a list to store all frames and their corresponding times
                # frames = []
                # frame_times = []

                # # Iterate over all frames in the trajectory
                # for frame in u.trajectory:
                #     frames.append(frame)            # Store the frame (if needed)
                #     frame_times.append(frame.time)
                # print(frame_times)
    # # Convert the desired timestamps into MDAnalysis time units (usually picoseconds)
    # trajectory_times = np.array([ts for ts in u.trajectory.times])
    
    # # Initialize dictionary to store results
    # psa_results = {}

    # # Loop over desired timestamps
    # for timestamp in timestamps:
    #     # Find the frame closest to the desired timestamp
    #     closest_frame_idx = np.abs(trajectory_times - timestamp).argmin()
        
    #     # Set the trajectory to that frame
    #     u.trajectory[closest_frame_idx]
        
    #     # Calculate PSA
    #     psa_calculator = PSA.PSAnalysis(u)
    #     psa_calculator.run()

    #     # Store the PSA result
    #     psa_results[timestamp] = psa_calculator.results.psa
    #     print(f"PSA at {timestamp} ps: {psa_calculator.results.psa} Å²")

    # # Optionally save results to a file
    # if output_file:
    #     with open(output_file, 'w') as f:
    #         for timestamp, psa in psa_results.items():
    #             f.write(f"{timestamp}, {psa}\n")

    return #psa_results

import pandas as pd

def change_column_names(MD_output):
    """Rename specific columns in the given DataFrame."""
    
    column_mapping = {
        'Total': 'SASA',
        'num_of_hbonds': 'num of H-bonds',
        'within_distance': 'H-bonds within 0.35A',
        'Mtot': 'Total dipole moment',
        'Bond': 'Ligand Bond energy',
        'U-B': 'Urey-Bradley energy',
        'Proper dih.': 'Torsional energy',
        'Coul-SR:Other-Other': 'Coul-SR: Lig-Lig',
        'LJ-SR:Other-Other': 'LJ-SR: Lig-Lig',
        'Coul-14:Other-Other': 'Coul-14: Lig-Lig',
        'LJ-14:Other-Other': 'LJ-14: Lig-Lig',
        'Coul-SR:Other-SOL': 'Coul-SR: Lig-Sol',
        'Coul-SR:Other-SOL': 'Coul-SR: Lig-Sol',
        # Add more mappings as needed
    }

    MD_output = MD_output.rename(columns=column_mapping)
    return MD_output

def main(protein = pv.PROTEIN):
    pv.update_config(protein_= protein)
    MDsimulations_path = pv.MDsimulations_path_
    energyfolder_path = pv.energyfolder_path_
    # create ndx files
    # make_index_files(MDsimulations_path) #make index files

    #create hbond csv #1
    hbond_df = get_hbond_csv.calculate_hbond_dataframe_trajectory(MD_path=MDsimulations_path) #1 #use this one i guess. make sure export is okay
    hbond_df.to_csv(pv.energyfolder_path_ / 'hbonds.csv', index=False)

    # create RMSD #2
    # RMSD_xvg_dir = energyfolder_path / 'RMSD_xvg'
    # get_rms.run_gmx_rms(MDsimulations_path, RMSD_xvg_dir) #creates the files
    # data = get_rms.rms_xvg_files_to_csvfiles(RMSD_xvg_dir) #3
    # data.to_csv(energyfolder_path / 'rmsd.csv', index=False)

    # #gyration #3
    # gyration_xvg_dir = energyfolder_path / 'gyration_xvg'
    # get_gyrate.run_gmx_gyrate(MDsimulations_path, gyration_xvg_dir)
    # data = get_gyrate.gyration_xvg_files_to_csvfiles(gyration_xvg_dir) #3
    # data.to_csv(energyfolder_path / 'gyration.csv', index=False)

    # #total dipole moment & epsilon #4 & 5
    # TDM_xvg_dir = energyfolder_path / 'Total_dipole_moment_xvg'
    # epsilon_xvg_dir = energyfolder_path / 'epsilon_xvg'
    # get_dipoles.run_gmx_dipoles(MDsimulations_path, TDM_xvg_dir, epsilon_xvg_dir)
    # data = get_dipoles.Total_dipole_moment_xvg_files_to_csvfiles(energyfolder_path, TDM_xvg_dir)
    # data.to_csv(energyfolder_path / 'total_dipole_moment.csv', index=False)
    # data = get_dipoles.epsilon_xvg_files_to_csvfiles(energyfolder_path, epsilon_xvg_dir)
    # data.to_csv(energyfolder_path / 'epsilon.csv', index=False)


    # #SASA & PSA
    # SASA_xvg_dir = pv.energyfolder_path_ / 'SASA_xvg'
    # PSA_xvg_dir = pv.energyfolder_path_ / 'PSA_xvg'

    # get_sasa_psa.make_PSA_index_files(MDsimulations_path)

    # get_sasa_psa.run_gmx_sasa(MDsimulations_path, SASA_xvg_dir)
    # sasa_df = get_sasa_psa.sasa_xvg_files_to_csvfiles(energyfolder_path, SASA_xvg_dir)

    # get_sasa_psa.run_gmx_psa_sasa(MDsimulations_path, PSA_xvg_dir)
    # psa_df = get_sasa_psa.psa_xvg_files_to_csvfiles(energyfolder_path, PSA_xvg_dir)

    # psa_df.to_csv(energyfolder_path / 'psa.csv', index=False)
    # sasa_df.to_csv(energyfolder_path / 'sasa.csv', index=False)

    #energy files
    # prepare_energy_files_from_MD.main(MDsimulations_path)

    
    #need ligand + system
    # run_gmx_sasa(MDsimulations_path, outputdir) #NOTE: done
    # sasa_xvg_files_to_csvfiles(energyfolder_path, outputdir) #2

    #PSA #7 #all 4 lines
    # make_PSA_index_files(MDsimulations_path)
    # outputdir = public_variables.energyfolder_path_ / 'PSA'
    # run_gmx_psa_sasa(MDsimulations_path, outputdir)
    # psa_xvg_files_to_csvfiles(energyfolder_path, outputdir)









    #all can be removed i think:
    ############################
    # lig_conf_system_path = public_variables.base_path_ / 'ligand_conformations_system'
    
    # calculate_hbond_dataframe(MD_path=MDsimulations_path, lig_conf_system_path = lig_conf_system_path)
    # pdb_file = Path('/home/ben/Download/Afstuderen0/MDsimulations/001/hbond_distances_try_pdb869.xvg')
    # print(read_out_hbond_xvgfile(pdb_file))
    # pdb_file = Path('/home/ben/Download/Afstuderen0/MDsimulations/001/hbnum.xvg')
    # print(read_out_hbnum(pdb_file))

    # calculate_hbond_dataframe(MD_path=MDsimulations_path, lig_conf_system_path = lig_conf_system_path)
    ############################


    
    file_list = ['hbonds.csv', 'rmsd.csv', 'gyration.csv', 'epsilon.csv', 'total_dipole_moment.csv', 'sasa.csv', 'psa.csv', f'MD_features_{pv.PROTEIN}.csv']
    folder_path = energyfolder_path
    MD_output_df = merge_csv_files_on_columns(folder_path, file_list)
    MD_output_df = change_column_names(MD_output_df)
    MD_output_df.to_csv(energyfolder_path / 'MD_output.csv', index=False)
    return


if __name__ == "__main__":
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    MD_output_df = pd.read_csv(pv.MD_outputfile_)
    MD_output_df = change_column_names(MD_output_df)
    MD_output_df.to_csv(pv.energyfolder_path_ / 'MD_output.csv', index=False)
    # main(pv.PROTEIN)




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