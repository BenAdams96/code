#use the 'simulations' folder and count how many molecules there are
#then create the 'ligandconformations' folder with subfolders 0ns to 10ns.
#put in every subfolder, the frame corresponding to the timestep of every molecule where the simulation ran.

import os
import numpy as np
from pathlib import Path
import pathlib
import subprocess
import re
import MDAnalysis as mda
from MDAnalysis.coordinates import PDB
import shutil
from global_files import public_variables as pv
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


#NOTE: uses the MD simulations folder to create the folder 'ligand_conformations_for_every_snapshot'
def main():
    pdb_file_path = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataZ/ligand_conformation_files/ligand_conformations_GSK3/0.06ns/764_0.06ns.pdb')
    # pdb_file_path = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataZ/ligand_conformation_files/ligand_conformations_GSK3_notcentered/0.06ns/764_0.06ns.pdb')
    # pdb_file_path = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataZ/ligand_conformation_files/ligand_conformations_GSK3_notcentered/0.06ns/765_0.06ns.pdb')
    # pdb_file_path = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataZ/ligand_conformation_files/ligand_conformations_GSK3/0.06ns/765_0.06ns.pdb')
    
    mol = Chem.MolFromPDBFile(str(pdb_file_path), removeHs=False, sanitize=False)
                
    if mol is not None:
        print(mol)
        try:
            Chem.SanitizeMol(mol)
        except ValueError as e:
            print(f"Sanitization error: {e}")
            print(pdb_file_path)
    print('try')
    mol_descriptors = rdMolDescriptors.CalcGETAWAY(mol)
    print(mol_descriptors)
    return

# Example usage
if __name__ == "__main__":
    #
    main()

    # pv.update_config(model_=Model_classic, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # MDsimulations_path = pv.MDsimulations_path_ #the folder containing all the MD simulations {001,002..615}
    # frames = 1000
    # main(MDsimulations_path, pv.ligand_conformations_path_, frames)