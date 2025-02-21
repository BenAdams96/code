import os
from pathlib import Path

# file_name = "testingtesting.pdb"  # File name
# current_dir = os.getcwd()   # Get current working directory
# file_path = os.path.join(current_dir, file_name)  # Full path

# print(f"Looking for '{file_name}' in: {current_dir}")

# if os.path.exists(file_path):
#     os.remove(file_path)
#     print(f"✅ {file_name} deleted successfully.")
# else:
#     print(f"❌ File not found! Searching in subdirectories...")

#     # Search for the file in all subdirectories
#     for root, _, files in os.walk(current_dir):
#         if file_name in files:
#             file_path = os.path.join(root, file_name)
#             os.remove(file_path)
#             print(f"✅ Found and deleted: {file_path}")
#             break
#     else:
#         print(f"❌ {file_name} not found in any subdirectory.")


from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from  rdkit.Chem import rdFreeSASA


import MDAnalysis as mda
# from MDAnalysis.analysis import align, sasa
# from MDAnalysis.analysis import hydrogenbond
from pathlib import Path

import freesasa

# Load your PDB file

base_path_ = Path(__file__).resolve().parent  # 'Afstuderen' folder
pdb_file = str(base_path_ / "001_5ns.pdb")  # Convert to string
structure = freesasa.Structure(pdb_file)

result = freesasa.calc(structure)

# Print SASA result
print(f"SASA: {result.totalArea():.2f} Å²")

# # Example usage
# base_path_ = Path(__file__).resolve().parent  # 'Afstuderen' folder
# pdb_file = base_path_ / "1.pdb"  # Full path to the PDB file
# print(pdb_file)
# sasa, psa = calculate_sasa_psa(pdb_file)
# print(f"SASA: {sasa:.2f} Å²")
# print(f"TPSA: {psa:.2f} Å²")

