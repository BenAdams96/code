# Project-specific imports
from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
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

def MD_features_implementation(savefolder_name, dfs_MD_path,minus):
    dfs_descPCA_path = pv.dfs_DescPCA_path_.parent / 'desc_PCA20'

    destination_folder = pv.dataframes_master_ / savefolder_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    dfs_dictionary_desc = csv_to_dictionary.csvfiles_to_dic_include(dfs_descPCA_path,include_files=['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_10.csv'])#,'conformations_1000.csv','conformations_1000_molid.csv'])
    dfs_dictionary_MD = csv_to_dictionary.csvfiles_to_dic_include(pv.dataframes_master_ / dfs_MD_path,include_files=['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_10.csv'])#,'conformations_1000.csv','conformations_1000_molid.csv'])

    for name, df in list(dfs_dictionary_desc.items()):
        if name.startswith('conformations'):
            print(name)
            df_MD_PCA = dfs_dictionary_MD[name]
            merged_df = pd.merge(df, df_MD_PCA, on=['mol_id','PKI', 'conformations (ns)'], how='inner')
            merged_df = merged_df.drop(columns=[minus])
            merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')
        elif name.endswith('ns'):
            print(name)
            df_MD_PCA = dfs_dictionary_MD[name]

            merged_df = pd.merge(df, df_MD_PCA, on=['mol_id','PKI'], how='inner')
            merged_df = merged_df.drop(columns='picoseconds', errors='ignore')
            merged_df = merged_df.drop(columns=[minus])
            merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')






    # MD_outputfile = pv.MD_outputfile_ #csv file with all the succesfull molecules and their MD simulation features for every ns
    
    # df_MDfeatures = pd.read_csv(MD_outputfile)
    # df_MDfeatures['picoseconds'] = df_MDfeatures['picoseconds'] / 1000 #go from picoseconds to ns

    # # delete all csv files in the folder except for MD_output.csv
    # for file_name in os.listdir(destination_folder):
    #     if file_name.endswith('.csv') and file_name != 'MD_output.csv':
    #         file_path = os.path.join(destination_folder, file_name)
    #         os.remove(file_path)

    # #copy_redfolder_only_csv_files(reduced_dataframes_folder, destination_folder)
    # os.makedirs(destination_folder, exist_ok=True)
    # shutil.copy(MD_outputfile, destination_folder) #copy 'MD_output.csv' to
    # #NOTE: not sure if pv.inital_dataframe will work because its a full path
    # dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_exclude(reduced_dataframes_folder, exclude_files=['concat_ver.csv', 'concat_hor.csv','rdkit_min.csv','MD_output.csv', 'conformations_1000.csv', 'conformations_1000_molid.csv', 'conformations_1000_mol_id.csv', f'{pv.initial_dataframe_}.csv', 'initial_dataframe_mol_id.csv','stable_conformations.csv']) # , '0ns.csv', '1ns.csv', '2ns.csv', '3ns.csv', '4ns.csv', '5ns.csv', '6ns.csv', '7ns.csv', '8ns.csv', '9ns.csv', '10ns.csv'
    # sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys()))
    # print(sorted_keys_list)
    # dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic}

    # for name, df in list(dfs_in_dic.items()):
    #     if name.startswith('conformations'):
    #         print(name)
    #         merged_df = pd.merge(df, df_MDfeatures, left_on=['mol_id', 'conformations (ns)'], right_on=['mol_id', 'picoseconds'], how='inner')
    #         merged_df = merged_df.drop(columns='picoseconds')
    #         merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
    #         print(f'done with {name}')
    #     elif name.endswith('ns'):
    #         print(name)
    #         df_MDfeatures2 = df_MDfeatures[df_MDfeatures['picoseconds'] == int(name.rstrip('ns'))]
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
    return

def main(savefolder_name, dfs_MD_path,minus):
    MD_features_implementation(savefolder_name, dfs_MD_path,minus)

    return

if __name__ == "__main__":
    main()
