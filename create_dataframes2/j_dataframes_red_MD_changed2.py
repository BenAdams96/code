# Project-specific imports
from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from create_dataframes2 import b_dataframes_reduced_fromt, i_dataframes_MD_only_changed_improvedq

from pathlib import Path
import shutil
import pandas as pd
import re
import os

def extract_k_and_scoring(filename):
    # Use regex to find the pattern 'k' followed by digits and the scoring method
    match = re.search(r'k(\d+)_(.+)\.pkl', filename)
    if match:
        k_value = int(match.group(1))
        scoring_metric = match.group(2)
        return k_value, scoring_metric
    else:
        return None, None
    
def get_molecules_lists_temp(parent_path):
    folder = pv.dfs_descriptors_only_path_
    csv_file = '0ns.csv'
    final_path = parent_path / folder / csv_file
    molecules_list = []
    invalid_mols = []
    df = pd.read_csv(final_path)
    mol_id_column = df['mol_id']

    valid_mol_list_str = list(map(str, mol_id_column))
    print(valid_mol_list_str)
    print(len(valid_mol_list_str))
    return  valid_mol_list_str

def extract_number(filename):
    return int(filename.split('ns.csv')[0])

def MD_features_implementation(savefolder_name, include,threshold, to_keep = ['SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','Coul-SR: Lig-Sol']):
    destination_folder = pv.dataframes_master_ / savefolder_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    reduced_dfs_in_dic = b_dataframes_reduced_fromt.main(threshold, include = include, write_out = False)
    MD_dfs_in_dic = i_dataframes_MD_only_changed_improvedq.main(savefolder_name=savefolder_name, to_keep=to_keep,include = include, write_out = False)

    print(reduced_dfs_in_dic['0ns'])
    print(MD_dfs_in_dic['0ns'])

    merged_dfs = {}

    for key in reduced_dfs_in_dic.keys():
        if key in MD_dfs_in_dic:
            merged_dfs[key] = pd.merge(
                reduced_dfs_in_dic[key], 
                MD_dfs_in_dic[key].drop(columns=['PKI'], errors='ignore'),
                on=['mol_id', 'conformations (ns)'],  # Adjust these keys based on your data
                how='inner'  # Use 'inner' if you only want matching rows
            )
        else:
            merged_dfs[key] = reduced_dfs_in_dic[key]  # Keep the original if no match
    b_dataframes_reduced_fromt.save_dataframes_to_csv(merged_dfs, save_path=destination_folder)
    # always_keep = ['mol_id', 'conformations (ns)']

    
    # # Combine always-keep columns with the ones in the to_keep list
    # for name, df in list(dfs_in_dic.items()):
    #     if name.startswith('conformations'):
    #         print(name)
    #         columns_to_keep = ['mol_id','picoseconds'] + [col for col in to_keep if col in df_MDfeatures.columns]
    #         df_MDfeatures = df_MDfeatures[columns_to_keep]
    #         merged_df = pd.merge(df, df_MDfeatures, left_on=['mol_id', 'conformations (ns)'], right_on=['mol_id', 'picoseconds'], how='inner')
    #         merged_df = merged_df.drop(columns='picoseconds')
    #         merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
    #         print(f'done with {name}')
    #     elif name.endswith('ns'):
    #         print(name)
    #         df_MDfeatures2 = df_MDfeatures[df_MDfeatures['picoseconds'] == int(name.rstrip('ns'))]
    #         columns_to_keep = ['mol_id','picoseconds'] + [col for col in to_keep if col in df_MDfeatures.columns]
    #         df_MDfeatures2 = df_MDfeatures2[columns_to_keep]
    #         merged_df = pd.merge(df, df_MDfeatures2, on='mol_id', how='inner')
    #         merged_df = merged_df.drop(columns='picoseconds')
    #         merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
    #         print(f'done with {name}')
    #     elif name.startswith('clustering'):
    #         print(name)
    #         merged_df = pd.merge(df, df_MDfeatures, left_on=['mol_id', 'conformations (ns)'], right_on=['mol_id', 'picoseconds'], how='inner')
    #         merged_df = merged_df.drop(columns='picoseconds')
    #         merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False) 
    #         print(f'done with {name}')
    #     else:
    #         df.to_csv(destination_folder / Path(name + '.csv'), index=False)
    #         continue
    # for name, df in list(dfs_in_dic.items()):
    #     if name.startswith('conformations'):
    #         print(name)
    #         columns_to_keep = always_keep + [col for col in to_keep if col in df_MDfeatures.columns]
    #         # Also keep 'conformations' if it exists
    #         if 'conformations (ns)' in df_MDfeatures.columns:
    #             columns_to_keep.append('conformations (ns)')
            
    #         # Filter the DataFrame to only keep the desired columns
    #         df_MDfeatures = df_MDfeatures[columns_to_keep]

    #         merged_df = pd.merge(df, df_MDfeatures, left_on=['mol_id', 'conformations (ns)'], right_on=['mol_id', 'conformations (ns)'], how='inner')
            
    #         merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
    #         print(f'done with {name}')
    #     elif name.endswith('ns'):
    #         print(name)
    #         columns_to_keep = always_keep + [col for col in to_keep if col in df_MDfeatures.columns]
    #         # Also keep 'conformations' if it exists
            
    #         # Filter the DataFrame to only keep the desired columns
    #         df_MDfeatures = df_MDfeatures[columns_to_keep]
    #         print(df)
    #         print(df_MDfeatures)
    #         df_MDfeatures2 = df_MDfeatures[df_MDfeatures['conformations (ns)'] == int(name.rstrip('ns'))]
    #         merged_df = pd.merge(df, df_MDfeatures2, on='mol_id', how='inner')
    #         # merged_df = merged_df.drop(columns='picoseconds')
            
    #         merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
    #         print(f'done with {name}')
    #     elif name.startswith('clustering'):
    #         print(name)
    #         merged_df = pd.merge(df, df_MDfeatures, left_on=['mol_id', 'conformations (ns)'], right_on=['mol_id', 'picoseconds'], how='inner')
    #         merged_df = merged_df.drop(columns='picoseconds')
    #         merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False) 
    #         print(f'done with {name}')
    #     else:
    #         df.to_csv(destination_folder / Path(name + '.csv'), index=False)
    #         continue
    return

def main(savefolder_name,include,threshold, to_keep):
    MD_features_implementation(savefolder_name,include,threshold, to_keep)

    return

if __name__ == "__main__":
    main()
