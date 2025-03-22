# Standard library imports
import os
import re
import ast

# Third-party libraries
import numpy as np
import pandas as pd
from pathlib import Path

# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Project-specific imports
from global_files import dataframe_processing, global_functions, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

def get_closest_rows(df, target=500):
    df['First Cluster Size'] = df['Cluster Sizes'].apply(lambda x: ast.literal_eval(x)[0])
    closest_rows = df.loc[df.groupby('mol_id')['First Cluster Size'].apply(lambda group: (group - target).abs().idxmin())]
    df = df.drop(columns=['First Cluster Size'])  # Optional: Remove the helper column
    return closest_rows

def create_dfs_dict(totaldf_path, to_keep=None, include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20']):
    print(f'create dfs in dict for {totaldf_path}')
    totaldf = pd.read_csv(totaldf_path)
    target_df = dataframe_processing.get_targets(pv.dataset_path_)
    # Check if conformations or picoseconds
    dfs_in_dict = {}
    # time_interval, time_col = (1, "nanoseconds (ns)") if "nanoseconds (ns)" in totaldf.columns else (1000, "picoseconds") if "picoseconds" in totaldf.columns else (None, None)
    always_keep = ['mol_id', 'PKI', 'conformations (ns)']

    # Drop PKI from totaldf if it exists (so only the PKI from target_df will be kept)
    if 'PKI' in totaldf.columns:
        totaldf.drop(columns=['PKI'], inplace=True)
        
    # Merge totaldf with target_df on 'mol_id'
    totaldf = pd.merge(totaldf, target_df, on='mol_id', how='left')
    
    # If the 'picoseconds' column exists, convert it to 'nanoseconds (ns)'
    if 'picoseconds' in totaldf.columns:
        totaldf['conformations (ns)'] = totaldf['picoseconds'] / 1000
        totaldf.drop(columns=['picoseconds'], inplace=True)
    
    # If to_keep is empty, keep all columns except 'mol_id', 'PKI', and 'conformations (ns)'
    if not to_keep:
        to_keep = [col for col in totaldf.columns if col not in always_keep]

    columns_to_keep = always_keep + to_keep

    # Reorganize columns: 'mol_id', 'pKi', and 'conformations (ns)' are first, then to_keep columns
    totaldf = totaldf[columns_to_keep]
    for x in include:
        if isinstance(x, int):
            filtered_df = totaldf[totaldf['conformations (ns)'] == x].copy()

            # Store the DataFrame in the dictionary
            dfs_in_dict[str(x) + 'ns'] = filtered_df
        elif isinstance(x, str) and x.startswith("c"):
            num_conformations = int(x[1:])
            total_time = totaldf['conformations (ns)'].max()
            
            target_conformations = [round(i * (total_time / num_conformations),1) for i in range(1, num_conformations + 1)]
            filtered_df = totaldf[totaldf['conformations (ns)'].isin(target_conformations)].copy()
            # Store the DataFrame in the dictionary
            dfs_in_dict[x] = filtered_df
        elif isinstance(x, str) and x.startswith("ti"):
            time_part, conformations_part = x.split('c')
            start_time, end_time = map(int, time_part[2:].split('_'))
            num_conformations = int(conformations_part)
            print(start_time)
            print(end_time)
            print(num_conformations)
            stepsize = (end_time - start_time) / num_conformations
            target_conformations = np.arange(start_time + stepsize, end_time+stepsize, stepsize).round(1)
            filtered_df = totaldf[totaldf['conformations (ns)'].isin(target_conformations)].copy()
            # Store the DataFrame in the dictionary
            dfs_in_dict[x] = filtered_df
        elif isinstance(x, str) and x.startswith("ta"):
            bins, conformations_part = x.split('c')
            num_conformations = int(conformations_part)
            total_time = totaldf['conformations (ns)'].max()
            bins = int(bins[2:])
            stepsize_outer = total_time/bins
            for i in np.arange(0,total_time,stepsize_outer):
                start_time = i
                end_time = i+stepsize_outer
                stepsize_inner = (end_time - start_time) / num_conformations
                target_conformations = np.arange(start_time+stepsize_inner, end_time+(stepsize_inner*0.1), stepsize_inner).round(2)
                filtered_df = totaldf[totaldf['conformations (ns)'].isin(target_conformations)].copy()
                # Round the start_time and end_time and check if they are effectively integers
                if start_time.is_integer():
                    start_time = int(start_time)
                if end_time.is_integer():  
                    end_time = int(end_time)
                dfs_in_dict[f't{start_time}_{end_time}c{num_conformations}'] = filtered_df
        elif isinstance(x, str) and x.startswith("CLt"):
            CLtarget, clusters, conformations = x.split('_')
            CLtarget = int(CLtarget[3:])
            clusters = int(clusters[2:])
            conformations = int(conformations[1:])
            cluster_information_path = pv.base_path_ / 'dataZ' /'clustering' / f'clustering_information_{pv.PROTEIN}.csv'
            cluster_information_df = pd.read_csv(cluster_information_path)
            for i in range(1,clusters+1):
                # Extract the first cluster size and find the closest match to CLtarget
                cluster_information_df['First Cluster Size'] = cluster_information_df['Cluster Sizes'].apply(lambda x: ast.literal_eval(x)[0])
                closest_rows_df = cluster_information_df.loc[
                    cluster_information_df.groupby('mol_id')['First Cluster Size'].apply(lambda group: (group - CLtarget).abs().idxmin())
                ]
                
                # Copy relevant columns and convert string representation of lists to actual lists
                # Copy relevant columns and convert string representation of lists to actual lists
                target_conformations = closest_rows_df[['mol_id', 'random conformations per cluster']].copy()
                
                # Convert string representation of lists to actual lists (if needed)
                target_conformations['random conformations per cluster'] = target_conformations['random conformations per cluster'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                
                # Extract the i-th list from each nested list and keep only the first 10 values
                target_conformations['random conformations per cluster'] = target_conformations['random conformations per cluster'].apply(
                    lambda lst: lst[i-1][:conformations] if isinstance(lst, list) and len(lst) > i-1 and isinstance(lst[i-1], list) else None
                )
                
                # Divide each value in the selected list by 100
                target_conformations['random conformations per cluster'] = target_conformations['random conformations per cluster'].apply(
                    lambda lst: [x / 100 for x in lst] if isinstance(lst, list) else None
                )
                
                # Explode the lists into separate rows (for each conformation)
                expanded_target_conformations = target_conformations.explode('random conformations per cluster')

                # Rename column to match totaldf and merge
                expanded_target_conformations.rename(columns={'random conformations per cluster': 'conformations (ns)'}, inplace=True)
                filtered_df = totaldf.merge(expanded_target_conformations, on=['mol_id', 'conformations (ns)'])
                filtered_df.sort_values(by=['mol_id', 'conformations (ns)'], inplace=True)
                #filtered_df.to_csv(save_path / f'CLtarget{CLtarget}_cluster{i}_c{conformations}.csv', index=False)
                dfs_in_dict[f'CLt{CLtarget}_cl{i}_c{conformations}'] = filtered_df
    return dfs_in_dict



# def create_full_dfs(folder_path_clustered_pdbs, molID_PKI_df, descriptors):
#     ''' Create the WHIM dataframes for every molecule for every timestep (xdirs)
#         First goes over the timesteps, then every molecule within that timestep
#         Output: total_df_conf_order - dataframe with the descriptors of the molecule for every timestep including mol_id and PKI
#     '''
#     if descriptors == 'WHIM':
#         num_columns = 114
#     elif descriptors == 'GETAWAY':
#         num_columns = 273
#     else:
#         raise ValueError("Error: Choose a valid descriptor")
    
#     if public_variables.dataset_protein_ == 'JAK1':
#         max_molecule_number = 615
#     elif public_variables.dataset_protein_ == 'GSK3':
#         max_molecule_number = 856

#     rows = []


#     # List all PDB files in the directory
#     pdb_files = [file for file in os.listdir(folder_path_clustered_pdbs) if file.endswith('.pdb')]
#     filtered_sorted_list = sorted([file for file in pdb_files if int(file.split('_')[0]) <= max_molecule_number], #TODO: shitty solution
#                         key=lambda x: int(x.split('_')[0]))
    
#     # print(filtered_sorted_list)
#     for pdb_file in filtered_sorted_list:
#         #print(pdb_file) #is 001_centroid.pdb
#         dir_path = folder_path_clustered_pdbs
#         pdb_file_path = os.path.join(dir_path, pdb_file)
#         print(pdb_file)
#         #print(pdb_file_path) #is 001_centroid.pdb
#         mol = Chem.MolFromPDBFile(pdb_file_path, removeHs=False, sanitize=False)
        
#         if mol is not None:
#             try:
#                 Chem.SanitizeMol(mol)
#             except ValueError as e:
#                 print(f"Sanitization error: {e}")
#                 print(pdb_file)
#         else:
#             print("Invalid molecule:")
#             print(pdb_file)
#             continue
        
#         # Calculate descriptors
#         if descriptors == 'WHIM':
#             mol_descriptors = rdMolDescriptors.CalcWHIM(mol)
#         elif descriptors == 'GETAWAY':
#             mol_descriptors = rdMolDescriptors.CalcGETAWAY(mol)
        
#         # index_to_insert = int(pdb_file[:3]) + int((float(dir_path.name.rstrip('ns')) / 10) * (len(sorted_folders) - 1) * len(all_molecules_list))
#         pki_value = molID_PKI_df.loc[molID_PKI_df['mol_id'] == int(pdb_file[:3]), 'PKI'].values[0]
#         centroid_timepoint = pdb_file.split('_')[1].split('.')[0]
#         conformation_value = int(centroid_timepoint)/1000
#         # Collect the row data
#         rows.append([int(pdb_file[:3]), pki_value, conformation_value] + mol_descriptors)

#     # Convert rows list to DataFrame
#     columns = ['mol_id', 'PKI', 'conformations (ns)'] + list(range(num_columns))
#     total_df_conf_order = pd.DataFrame(rows, columns=columns).dropna().reset_index(drop=True)
#     return total_df_conf_order

# def get_sorted_folders(base_path):
#     '''The folder with CSV files 'dataframes_JAK1_WHIM' is the input and these CSV files will be
#        put into a list of pandas DataFrames, also works with 0.5 1 1.5 formats.
#     '''
#     folders = [f for f in base_path.iterdir() if f.is_dir()]
#     sorted_folders = []
#     # Define the regex pattern to match filenames like '0ns.csv', '1ns.csv', '1.5ns.csv', etc.
#     pattern = re.compile(r'^\d+(\.\d+)?ns$')

#     for csv_file in sorted(base_path.glob('*'), key=lambda x: extract_number(x.name)):
#         if pattern.match(csv_file.name):  # Check if the file name matches the pattern
#             sorted_folders.append(csv_file)
#         else:
#             sorted_folders.insert(0,csv_file)
#     return sorted_folders

# def extract_number(filename):
#     # Use regular expression to extract numeric part (integer or float) before 'ns.csv'
#     match = re.search(r'(\d+(\.\d+)?)ns$', filename)
#     if match:
#         number_str = match.group(1)
#         # Convert to float first
#         number = float(number_str)
#         # If it's an integer, convert to int
#         if number.is_integer():
#             return int(number)
#         return number
#     else:
#         return float('inf')

# def remove_invalids_from_dfs(dic_with_dfs,invalids):
#     invalids = list(map(int, invalids))
#     print(invalids)
#     filtered_dic_with_dfs = {}
#     for name, df in dic_with_dfs.items():
#         filtered_df = df[~df['mol_id'].isin(invalids)]
#         filtered_dic_with_dfs[name] = filtered_df
#     return filtered_dic_with_dfs

# def save_dataframes(dic_with_dfs,base_path):
#     dir = public_variables.dfs_descriptors_only_path_
#     final_path = base_path / dir
#     final_path.mkdir(parents=True, exist_ok=True)
#     timeinterval = public_variables.timeinterval_snapshots
#     #TODO: use a dictionary.
#     for name, df in dic_with_dfs.items():
#         #print(f"name: {name}, i: {df.head(1)}")
#         df.to_csv(final_path / f'{name}.csv', index=False)

def get_filtered_and_sorted_folders(base_folder, target_filter=None, cluster_filter=None):
    # Get all folders in the base directory
    all_folders = [
        f for f in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, f))
    ]

    # Regex to extract target and cluster values
    folder_pattern = re.compile(r'clustering_target(\d+(?:\.\d+)?)%_cluster(\d+)')

    # Extract and parse target and cluster values
    folder_info = []
    for folder in all_folders:
        match = folder_pattern.search(folder)
        if match:
            target = float(match.group(1))
            cluster = int(match.group(2))
            folder_info.append((folder, target, cluster))

    # Filter based on target_filter and cluster_filter
    if target_filter is not None:
        folder_info = [
            (folder, target, cluster) for folder, target, cluster in folder_info 
            if target in target_filter
        ]
    if cluster_filter is not None:
        folder_info = [
            (folder, target, cluster) for folder, target, cluster in folder_info 
            if cluster in cluster_filter
        ]

    # Sort folders: target descending, cluster ascending
    folder_info.sort(key=lambda x: (-x[1], x[2]))

    # Return full paths of sorted folders
    return [os.path.join(base_folder, folder) for folder, _, _ in folder_info]

#NOTE: this file does: get targets, count how many valid molecules and which,it creates the folder 'dataframes_WHIMJAK1' or equivellant
def main(dfs_path, initial_df):
    base_path = pv.base_path_
    smiles_activity_dataset = pv.dataset_path_
    df_targets = global_functions.get_targets_series(smiles_activity_dataset) #df with columns: 'mol_id' and 'PKI value'. all molecules
    print(df_targets)
    clustering_information_path = pv.base_path_ / 'dataZ' / 'clustering' / f'clustering_information_{pv.PROTEIN}.csv'
    clustering_information_df = pd.read_csv(clustering_information_path)
    
    target_filter = [30,20,10]  # Only include targets 10 and 20
    cluster_filter = [1,2,3,4,5,6,7,8,9,10]

    #add like a filter which ones i want. yes
    cluster_folder =  dfs_path / 'clustering folder'
    cluster_folder.mkdir(parents=True, exist_ok=True)
    # sorted_folders = get_filtered_and_sorted_folders(cluster_folder, target_filter, cluster_filter)
    dfs_in_dict = dataframe_processing.create_dfs_dict(totaldf_path=initial_df, to_keep=None, include = ['CLt100_cl10_c10'])
    dataframe_processing.save_dict_with_dfs(dfs_in_dict, dfs_path)

    # print(sorted_folders)
    # for folder in sorted_folders:
    #     print(folder)
        #create the dataframes, which eventually will be placed in 'dataframes_JAK1_WHIM' and also add the targets to the dataframes.
        # 
        # public_variables.dfs_descriptors_only_path_.mkdir(parents=True, exist_ok=True)
        # df.to_csv(public_variables.dfs_descriptors_only_path_ / f'{Path(folder).name}.csv', index=False)
    # df_sorted_by_molid.to_csv(public_variables.dataframes_master_ / 'initial_dataframe_mol_id.csv', index=False)
    return

if __name__ == "__main__":
    pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.JAK1)
    main(pv.dfs_MD_only_path_, pv.initial_dataframe_)
    # base_path = pv.base_path_
    # dataset_csvfile_path = pv.dataset_path_ # 'JAK1dataset.csv'
    # # RDKIT_descriptors = pv.Descriptor_

    # main(base_path, dataset_csvfile_path, RDKIT_descriptors)

