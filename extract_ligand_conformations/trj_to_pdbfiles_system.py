#use the 'simulations' folder and count how many molecules there are
#then create the 'ligandconformations' folder with subfolders 0ns to 10ns.
#put in every subfolder, the frame corresponding to the timestep of every molecule where the simulation ran.

import os
import numpy as np
from pathlib import Path
import pathlib
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
import shutil
from global_files import public_variables

def get_molecules_lists(MDsimulations_path):
    '''uses the MD_simulations folder and checks for every molecule the simulation whether it contains the .tpr and .xtc file
        to see if it contains a valid trajectory file (.tpr)
        output: lists of all the molecules, valid molecules, and invalid molecules in strings
        '''
    molecules_list = []
    valid_mols = []
    invalid_mols = []
    print('hi')
    for item in MDsimulations_path.iterdir(): #item is every folder '001' etc in the MD folder
        
        tprfile = f"{item.name}_prod.tpr"
        xtcfile = f"{item.name}_prod.xtc"  # trajectory file
        
        trajectory_file = item / xtcfile

        if item.is_dir():  # Check if the item is a directory
            molecules_list.append(item.name)
        if not trajectory_file.exists():
            invalid_mols.append(item.name)
        else:
            valid_mols.append(item.name)

    return molecules_list, valid_mols, invalid_mols #['001','002']

#also creates the folder "ligandconformations/0ns" etc
def trj_to_pdb(valid_molecules_list, frames_to_extract,base_path, MDsimulations_path, output_foldername):
    ''' 
    
    '''
    print(MDsimulations_path)
    # Create directories once for all frames ()
    #TODO: this can be done in the forloop of forloop is we say x= 0 and then do it
    for f in frames_to_extract:
        ns_foldername = f/100
        if ns_foldername.is_integer():
            ns_foldername = int(ns_foldername)
        dir = Path(f'{output_foldername}/{ns_foldername}ns')
        final_path = base_path / dir
        final_path.mkdir(parents=True, exist_ok=True)
    print('second for loop')
    #go over all the valid molecules and get the right snapshot of the trajectory into the right folders of 'ligand_conformations_for_every_ns'
    for x in valid_molecules_list:
        print(x)
        trajectory_dir = Path(MDsimulations_path) / x
        #trajectory_dir = base_path / Path("traj_files") ###
        tprfile = f"{x}_prod.tpr"
        xtcfile = f"{x}_prod.xtc"  # trajectory file

        topology_file = trajectory_dir / tprfile
        trajectory_file = trajectory_dir / xtcfile
        u = mda.Universe(topology_file, trajectory_file)
        for frame in frames_to_extract:
            

            residue_1 = u.select_atoms("all")

            u.trajectory[frame]  # Go to the specific frame [frame] or frames_to_extract[idx]
            ns_foldername = frame/100 #will be 1 2 3 4 or 0.5 1 1.5 depending on the amount of snapshots
            if ns_foldername.is_integer():
                ns_foldername = int(ns_foldername)
            output_pdb_file = base_path / Path(f'{output_foldername}/{ns_foldername}ns') / f"{x}_{ns_foldername}ns.pdb" #output_foldername = 'ligand_conformations_
            with PDB.PDBWriter(output_pdb_file) as pdb:
                pdb.write(residue_1)
            
            #remove the lonepair lines from the pdb files?
            with open(output_pdb_file, 'r') as infile, open(str(output_pdb_file) + '_tmp', 'w') as tmpfile:
                for line in infile:
                    if ' LP' not in line:
                        tmpfile.write(line)

            #Replace the original file with the temporary file
            shutil.move(str(output_pdb_file) + '_tmp', output_pdb_file)
    return

def trj_to_pdb_single_frame(valid_molecules_list, frame,base_path, MDsimulations_path, output_foldername):
    ''' 
    
    '''
    print(MDsimulations_path)
    # Create directories once for all frames ()
    #TODO: this can be done in the forloop of forloop is we say x= 0 and then do it
    ns_foldername = frame/1000
    if ns_foldername.is_integer():
        ns_foldername = int(ns_foldername)
    dir = Path(f'{output_foldername}/{ns_foldername}ns')
    final_path = base_path / dir
    final_path.mkdir(parents=True, exist_ok=True)
    print('second for loop')
    #go over all the valid molecules and get the right snapshot of the trajectory into the right folders of 'ligand_conformations_for_every_ns'
    for x in valid_molecules_list:
        print(x)
        trajectory_dir = Path(MDsimulations_path) / x
        #trajectory_dir = base_path / Path("traj_files") ###
        tprfile = f"{x}_prod.tpr"
        xtcfile = f"{x}_prod.xtc"  # trajectory file

        topology_file = trajectory_dir / tprfile
        trajectory_file = trajectory_dir / xtcfile
        u = mda.Universe(topology_file, trajectory_file)

        residue_1 = u.select_atoms("all")

        u.trajectory[frame]  # Go to the specific frame [frame] or frames_to_extract[idx]
        ns_foldername = frame/1000 #will be 1 2 3 4 or 0.5 1 1.5 depending on the amount of snapshots
        if ns_foldername.is_integer():
            ns_foldername = int(ns_foldername)
        output_pdb_file = base_path / Path(f'{output_foldername}/{ns_foldername}ns') / f"{x}_{ns_foldername}ns.pdb" #output_foldername = 'ligand_conformations_
        with PDB.PDBWriter(output_pdb_file) as pdb:
            pdb.write(residue_1)
        
        #remove the lonepair lines from the pdb files?
        with open(output_pdb_file, 'r') as infile, open(str(output_pdb_file) + '_tmp', 'w') as tmpfile:
            for line in infile:
                if ' LP' not in line:
                    tmpfile.write(line)

        #Replace the original file with the temporary file
        shutil.move(str(output_pdb_file) + '_tmp', output_pdb_file)
    return

#NOTE: uses the MD simulations folder to create the folder 'ligand_conformations_for_every_snapshot'
def main(MD_folder,output_folder):
    base_path = public_variables.base_path_
    MDsimulations_path = public_variables.MDsimulations_path_
    
    output_dir = base_path / output_folder
    print('hi')
    # Create molecules list. so all molecules that are present in the folder 'simulations' get added to the list
    all_molecules_list, valid_mols, invalid_mols = get_molecules_lists(MDsimulations_path)
    print(all_molecules_list)
    print(valid_mols)
    print(invalid_mols)
    print('done get mols')
    #Define frames to extract
    frames_to_extract = list(range(0,10001,10))  # every 1ns extract frame for 10ns. so [0ns,1ns,2ns,...,10ns]
    print(frames_to_extract)
    print('done frames to extract')
    # Process trajectories
    trj_to_pdb_single_frame(valid_mols, 20,base_path, MDsimulations_path, output_folder)
    # trj_to_pdb(valid_mols, frames_to_extract,base_path, MDsimulations_path, output_folder)
    print(f"number of molecules: {len(all_molecules_list)}")
    print(f"number of molecules with succesful simulations: {len(valid_mols)}")
    print(f"Invalid molecules: {invalid_mols}")
    return

# Example usage
if __name__ == "__main__":
    print('hi')
    MDsimulations_folder = public_variables.MDsimulations_folder_ #the folder containing all the MD simulations {001,002..615}
    output_folder = 'ligand_conformations_system_with_lp2'
    #output_folder = public_variables.lig_conforms_foldername_

    main(MDsimulations_folder, output_folder)