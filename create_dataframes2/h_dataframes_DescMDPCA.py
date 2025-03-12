# Project-specific imports
from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA

from pathlib import Path
import shutil
import pandas as pd
import re
import os

def standardize_dataframe(df):
    """Preprocess the dataframe by handling NaNs and standardizing."""
    # Handle NaNs: drop rows with NaNs or fill them

    df_cleaned = df.dropna()  # or df.fillna(df.mean())
    # print(df_cleaned)
    # Identify which non-feature columns to keep
    non_feature_columns = ['mol_id','PKI','conformations (ns)']
    existing_non_features = [col for col in non_feature_columns if col in df_cleaned.columns]

    # Drop non-numeric target columns if necessary
    features_df = df_cleaned.drop(columns=existing_non_features, axis=1, errors='ignore')

    # Standardize the dataframe
    scaler = StandardScaler()
    features_scaled_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)
    
    # Concatenate the non-feature columns back into the standardized dataframe
    standardized_df = pd.concat([df_cleaned[existing_non_features], features_scaled_df], axis=1)
    
    return standardized_df

def calculate_correlation_matrix(df):
    """Calculate the correlation matrix of a standardized dataframe."""
    df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
    return df.corr()


def correlation_matrix_single_csv(df):
    # Preprocess the dataframe: handle NaNs and standardize
    st_df = standardize_dataframe(df)
    
    # Calculate and visualize correlation matrix for the standardized dataframe
    correlation_matrix = calculate_correlation_matrix(st_df)
    return st_df, correlation_matrix

def compute_correlation_matrices_of_dictionary(dfs_dictionary, exclude_files: list=None):
    """
    Compute and visualize the correlation matrices for a list of dataframes.
    
    Args:
        dfs (list): List of dataframes to process.
        save_plot_folder (Path): Folder path where to save the plots.
    
    Returns:
        list of tuples: Each tuple contains (original_df, df_notargets, st_df, correlation_matrix).
    """
    print(f'correlation matrix of {dfs_dictionary.keys()}')

    standardized_dfs_dic = {}
    correlation_matrices_dic = {}
    
    for name, df in dfs_dictionary.items():
        print(f'correlation matrix of: {name}')
        
        st_df, correlation_matrix = correlation_matrix_single_csv(df)
        
        # visualize_matrix(correlation_matrix, dfs_path, name, title_suffix="Original")
        
        # Store the results for potential further use
        standardized_dfs_dic[name] = st_df
        correlation_matrices_dic[name] = correlation_matrix
    return standardized_dfs_dic, correlation_matrices_dic

def identify_columns_to_drop(correlation_matrix, st_df, variances, threshold):
    """Identify columns to drop based on correlation threshold and variance."""
    corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    columns_to_drop = set()
    processed_pairs = set()

    for i, j in zip(*corr_pairs):
        if i != j:  # Skip self-correlation
            pair = tuple(sorted((i, j)))  # Ensure consistent ordering
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                # Choose column to drop based on variance
                if variances[i] > variances[j]:
                    columns_to_drop.add(st_df.columns[j])
                else:
                    columns_to_drop.add(st_df.columns[i])
    
    return columns_to_drop

def identify_columns_to_drop_2_keep_lowest(correlation_matrix, df, variances, threshold):
    """Identify columns to drop based on correlation threshold and keeping the lowest indexed feature."""
    corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    columns_to_drop = set()
    processed_pairs = set()
    
    for i, j in zip(*corr_pairs):
        if i != j:  # Skip self-correlation
            pair = tuple(sorted((i, j)))  # Ensure consistent ordering
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                # Drop the column with the higher index
                if i < j:
                    columns_to_drop.add(df.columns[j])  # Drop column j (higher index)
                else:
                    columns_to_drop.add(df.columns[i])  # Drop column i (higher index)
    
    return columns_to_drop

def get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dfs_dictionary, threshold):
    """
    Reduce dataframes based on correlation and visualize the reduced matrices.
    
    Args:
        correlation_matrices_dic (dict): Dictionary of correlation matrices.
        dfs_dictionary (dict): Dictionary of dataframes corresponding to the correlation matrices.
        threshold (float): Correlation threshold for dropping features.
        save_plot_folder_reduced (Path): Folder path where to save the reduced matrix plots.
    
    Returns:
        dict: Dictionary of reduced dataframes.
    """
    reduced_dfs_dictionary = {}
    
    for key in correlation_matrices_dic.keys():
        # Calculate variances for the non-standardized dataframe
        
        
        # Identify non-feature columns to retain
        non_feature_columns = ['mol_id','PKI','conformations (ns)']
        existing_non_features = [col for col in non_feature_columns if col in dfs_dictionary[key].columns]
        
        # Drop only the features for correlation analysis
        features_df = dfs_dictionary[key].drop(columns=existing_non_features, axis=1)
        variances = features_df.var()

        # Identify columns to drop based on high correlation and variance
        columns_to_drop = identify_columns_to_drop_2_keep_lowest(correlation_matrices_dic[key], features_df, variances, threshold)
        
        # Create the reduced dataframe by including the retained non-feature columns
        reduced_df = pd.concat([dfs_dictionary[key][existing_non_features], features_df], axis=1)
        reduced_df = reduced_df.drop(columns=columns_to_drop, axis=1)

        reduced_dfs_dictionary[key] = reduced_df
    
    return reduced_dfs_dictionary

def save_dataframes_to_csv(dic_with_dfs,save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dic_with_dfs.items():
        print(f"save dataframe: {name}")
        df.to_csv(save_path / f'{name}.csv', index=False)

def save_reduced_dataframes(dfs, base_path):
    dir = pv.dfs_reduced_path_
    final_path = base_path / dir
    final_path.mkdir(parents=True, exist_ok=True)
    timeinterval = pv.timeinterval_snapshots

    for i, x in enumerate(np.arange(0, len(dfs) * timeinterval, timeinterval)):
        if x.is_integer():
            x = int(x)
        print(f"x: {x}, i: {i}")
        dfs[i].to_csv(final_path / f'{x}ns.csv', index=False)

def load_results(csv_path):
    df = pd.DataFrame()
    return df

def remove_constant_columns_from_dfs(dfs_dictionary):
    cleaned_dfs = {}
    
    for key, df in dfs_dictionary.items():
        # Identify constant columns
        constant_columns = df.columns[df.nunique() <= 1]
        
        if not constant_columns.empty:
            print(f"In '{key}', the following constant columns were removed: {', '.join(constant_columns)}")
        # Remove constant columns and keep only non-constant columns
        non_constant_columns = df.loc[:, df.nunique() > 1]
        cleaned_dfs[key] = non_constant_columns
    return cleaned_dfs

def PCA_for_dfs(dfs_dictionary, components):
    dfs_dictionary_pca = {}
    for key, df in dfs_dictionary.items():
        # Standardize the dataframe
        standardized_df = standardize_dataframe(df)  # Assuming this function returns the standardized df

        # Drop non-feature columns for PCA
        features_df = standardized_df.drop(columns=['mol_id', 'PKI', 'conformations (ns)'], errors='ignore')

        # Apply PCA
        pca = PCA(n_components=components)  # Use the specified number of components
        pca_result = pca.fit_transform(features_df)

        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_{i+1}' for i in range(pca_result.shape[1])])

        # Re-add the non-feature columns to the PCA dataframe at the start
        non_feature_columns = ['mol_id', 'PKI', 'conformations (ns)']
        existing_non_feature_df = standardized_df[standardized_df.columns.intersection(non_feature_columns)].reset_index(drop=True)

        # Insert non-feature columns at the start of the PCA DataFrame
        pca_df = pd.concat([existing_non_feature_df.reset_index(drop=True), pca_df], axis=1)

        # Store the PCA results in the new dictionary
        dfs_dictionary_pca[key] = pca_df

        # Optionally, save PCA results to CSV
        # pca_df.to_csv(dfs_dictionary_pca / f'{key}', index=False)

    return dfs_dictionary_pca

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

def MD_features_implementation(components):
    dfs_descriptors_only_path = pv.dfs_descriptors_only_path_
    dfs_MD_only_path = pv.dfs_MD_only_path_

    new_name = f"(DescMD)PCA_{components}"
    destination_folder = pv.dfs_DescMDPCA_path_.parent / new_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    dfs_dictionary_desc = csv_to_dictionary.csvfiles_to_dic_include(pv.dfs_descriptors_only_path_,include_files=['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_10.csv'])#,'conformations_1000.csv','conformations_1000_molid.csv'])
    newname_df_MDnew_only = pv.dataframes_master_ / 'MD_new only'
    dfs_dictionary_MD = csv_to_dictionary.csvfiles_to_dic_include(newname_df_MDnew_only,include_files=['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_10.csv'])#,'conformations_1000.csv','conformations_1000_molid.csv'])
    print(dfs_dictionary_MD)
    dfs_dictionary_pca = {}
    for name, df in list(dfs_dictionary_desc.items()):
        if name.startswith('conformations'):
            print(name)
            df_MD_PCA = dfs_dictionary_MD[name]
            
            merged_df = pd.merge(df, df_MD_PCA, on=['mol_id','PKI', 'conformations (ns)'], how='inner')
            merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')
        elif name.endswith('ns'):
            print(name)
            df_MD_PCA = dfs_dictionary_MD[name]

            merged_df = pd.merge(df, df_MD_PCA, on=['mol_id','PKI'], how='inner')
            merged_df = merged_df.drop(columns='picoseconds', errors='ignore')

            merged_df.to_csv(destination_folder / Path(name + '.csv'), index=False)
            print(f'done with {name}')
        # print(merged_df)
        dfs_dictionary_pca[name] = merged_df

    dfs_dictionary = remove_constant_columns_from_dfs(dfs_dictionary_pca)
    
    dfs_dictionary_pca = PCA_for_dfs(dfs_dictionary_pca, components)
    
    # Reduce the dataframes based on correlation
    # reduced_dfs_in_dic = get_reduced_features_for_dataframes_in_dic(correlation_matrices_dic, dfs_dictionary, threshold)
    #reduced dataframes including mol_ID and PKI. so for 0ns 1ns etc. 
    save_dataframes_to_csv(dfs_dictionary_pca, save_path=destination_folder)



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

def main(components):
    MD_features_implementation(components)

    return

if __name__ == "__main__":
    main()
