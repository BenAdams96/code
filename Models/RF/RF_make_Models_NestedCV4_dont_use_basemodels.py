from sklearn.model_selection import GridSearchCV, KFold

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor

from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np


import pandas as pd
import math
import re
import os

def save_dataframes_to_csv(dic_with_dfs,save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in dic_with_dfs.items():
        print(f"save dataframe: {name}")
        df.to_csv(save_path / f'{name}.csv', index=False)

def bin_pki_values(y, num_bins=5):
    """
    Bin the pKi values into discrete bins for stratified splitting.
    Parameters:
        y (array-like): Target variable (pKi values).
        num_bins (int): Number of bins to stratify.
    Returns:
        binned_y: Binned version of y.
    """
    bins = np.linspace(y.min(), y.max(), num_bins + 1)  # Define bin edges
    binned_y = pd.cut(y, bins, right=True, include_lowest=True, labels=False)  # Include both sides
    return bins, binned_y

def hyperparameter_tuning(model, X, y, param_grid, cv, scoring='r2'):
    """Performs hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search

def nested_cross_validation(name, df,outer_folds=10, inner_folds=5):
    """
    Perform nested cross-validation with stratified outer folds.
    Parameters:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        outer_folds (int): Number of outer folds.
        inner_folds (int): Number of inner folds.
    """
    print("Nested Cross Validation:", name)

    fold_assignments_df = pd.DataFrame({"mol_id": df["mol_id"]})

    target_column = 'PKI'
    X = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
    y = df[target_column]  # Target (pKi values)
    
    # Stratified outer loop
    bins, binned_y = bin_pki_values(y, num_bins=5)  # Bin pKi values for stratification, still a dataframe: all molecules with an index of which bin they are in
    unique, counts = np.unique(binned_y, return_counts=True)
    print("bins:", bins)
    print("counts in bin:", counts)
    # Group by mol_id (reduces dataframe to the amount of molecules)
    grouped_df = df.groupby('mol_id').first()  # One row per molecule, so group by mol_id

    #have 5 bins of equal lengths, and bin the molecule based on pki in the correct group
    #series with index: mol_id and as value the bin index (ranging from 0 to 4)
    grouped_binned_y = bin_pki_values(grouped_df[target_column], num_bins=5)[1]

    unique_mol_ids = grouped_df.index  # Index([1,2,3,4,6,...,615]) array of all mol_id
    # print(grouped_df)
    # print(grouped_binned_y)
    # print(all_mol_ids)

    fold_results = []
    custom_outer_splits = []
    all_true_pki_series = pd.Series(dtype=float)
    all_predicted_pki_series = pd.Series(dtype=float)
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,shuffle=True, random_state=10)


    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X=grouped_df, y=grouped_binned_y, groups=unique_mol_ids)):
        print(f'Outer fold: {outer_fold}')
        outer_col = f"outer_{outer_fold}"
        
        fold_assignments_df[outer_col] = None

        # map train_idx to the actual mol ids (so back to the range of 1 to 615 (for JAK1))
        train_mol_ids = grouped_df.iloc[train_idx].index  # Train molecule IDs
        test_mol_ids = grouped_df.iloc[test_idx].index  # Test molecule IDs

        # map the mol ids back to all instances of it, and then get the indexes
        # so when 10 conformations, 1 mol id gives the indexes of those 10
        # (same as train_idx and test_idx if there is only 1 conformation)
        train_idx_all = df[df['mol_id'].isin(train_mol_ids)].index  # All rows for train molecules
        test_idx_all = df[df['mol_id'].isin(test_mol_ids)].index  # All rows for test molecules

        # train_molid_all = df[df['mol_id'].isin(train_mol_ids)]['mol_id']  # All rows for train molecules (10 conf: list with all mol_ids 10 times the same)
        # test_molid_all = df[df['mol_id'].isin(test_mol_ids)]['mol_id']  # All rows for test molecules
        custom_outer_splits.append((train_idx_all, test_idx_all))
        
        #create
        fold_assignments_df.loc[fold_assignments_df["mol_id"].isin(train_mol_ids), outer_col] = "train"
        fold_assignments_df.loc[fold_assignments_df["mol_id"].isin(test_mol_ids), outer_col] = "test"

        #counts is how many molecules are in each bin for the test set (dont use it, just to check things)
        unique, counts_train = np.unique(binned_y.iloc[train_idx_all], return_counts=True)
        unique, counts_test = np.unique(binned_y.iloc[test_idx_all], return_counts=True)
        #counts_train=[ 24  29  49 176 232] counts_test=[ 4  2  6 20 25] (for JAK1)

        # Split data (use X because we dont want to incorperate the columns 'pki', 'mol_id', 'conformations (ns)')
        X_train, X_test = X.loc[train_idx_all], X.loc[test_idx_all]
        y_train, y_test = y.loc[train_idx_all], y.loc[test_idx_all]
        
        groups_train = df.loc[X_train.index, 'mol_id']  # Train set molecule IDs (series with index train_full_idxs and all molids duplicate times) 
        groups_test = df.loc[X_test.index, 'mol_id']  # Test set molecule IDs

        # Bin y_train for stratification in inner loop
        # divide binned_y (which binned every row/idx/molid into a bin) into the training part and testing part
        binned_y_train = binned_y.loc[X_train.index] #series with idx of all molecules/conformations and the bin it belongs to.
        binned_y_test = binned_y.loc[X_test.index]
        # so for 10 conformations, each conformation is a idx in this series

        best_model = None
        best_score = -np.inf
        

        # Inner loop: StratifiedGroupKFold
        custom_inner_splits = []
        true_pki_inner_idx_all = []
        predicted_pki_inner_idx_all = []
        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=10)

        # Group `X_train` by `mol_id` (X_train is different size dependent on how many conformations)
        # grouped_X_train is always the same size df since it groups based on mol_id
        grouped_X_train = X_train.groupby(groups_train).first()  # One row per molecule (groups_train because needs to be same size i guess)
        grouped_binned_y_train = binned_y_train.groupby(groups_train).first() #same as grouped_binned_y but now only the train molecules

        #group_X_train has as index mol_id, so grouped_X_train.index are all unique train molecule ids
        for inner_fold, (train_inner_idx, val_idx) in enumerate(inner_cv.split(grouped_X_train, grouped_binned_y_train, groups=grouped_X_train.index)):
            # print(f'  Inner fold: {inner_fold}')
            inner_col = f"outer_{outer_fold}_inner_{inner_fold}"
            # print(inner_col)
            # print(train_inner_idx) #NOTE: idx based on grouped_X_train which is reset
            # print(val_idx)

            # map train_inner_idx to the actual mol ids (so back to the range of 1 to 615 (for JAK1))
            # but now only in the range of the training part instead of the whole dataset
            train_inner_mol_ids = grouped_X_train.iloc[train_inner_idx].index
            val_mol_ids = grouped_X_train.iloc[val_idx].index
            #NOTE: the mol_ids are correct, checked multiple times

            # Expand molecule IDs to full conformations in `X_train`
            # map the mol ids back to all instances of it, and then get the indexes
            train_inner_idx_full = df[df['mol_id'].isin(train_inner_mol_ids)].index.intersection(X_train.index)
            val_idx_full = df[df['mol_id'].isin(val_mol_ids)].index.intersection(X_train.index) #same as val_idx if only 1 conformation
            #these are mapped back to the original df. these combined is the same as train_idx_full

            # ignore (was for testing)
            # train_inner_mol_ids_test = df.iloc[train_inner_idx_full]['mol_id']
            # val_mol_ids_test = df.iloc[val_idx_full]['mol_id']

            
            unique, counts = np.unique(binned_y.iloc[val_idx_full], return_counts=True)

            custom_inner_splits.append((train_inner_idx_full, val_idx_full))
            # Split inner train/validation data
            X_train_inner, X_val = X_train.loc[train_inner_idx_full], X_train.loc[val_idx_full]
            y_train_inner, y_val = y_train.loc[train_inner_idx_full], y_train.loc[val_idx_full]

            fold_assignments_df.loc[fold_assignments_df["mol_id"].isin(train_inner_mol_ids), inner_col] = "train"
            fold_assignments_df.loc[fold_assignments_df["mol_id"].isin(val_mol_ids), inner_col] = "val"
            fold_assignments_df[inner_col] = None

        #TODO: custom_inner_splits is set up for the whole df and not the X_train so wrong indexing?
        model = pv.ML_MODEL.model
        # grid_search = hyperparameter_tuning(model_instance, X, y, pv.ML_MODEL.hyperparameter_grid, cv=custom_inner_splits, scoring='r2')
        grid_search = hyperparameter_tuning(model, X, y, pv.hyperparameter_grid_RF, cv=custom_inner_splits, scoring='r2')
        df_results = pd.DataFrame(grid_search.cv_results_)
        print(df_results)
        for i in range(len(df_results)):
            params = df_results.loc[i, 'params']  # Extract the hyperparameter set
            print(f"Hyperparameters Set {i + 1}: {params}")
        best_model = grid_search.best_estimator_

        #build a "final" model for this train test split.
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        r2_score_ = r2_score(y_test, y_pred)
        tree = best_model.estimators_[0]
        tree_rules = export_text(tree, feature_names=list(X.columns))  # X.columns if using a DataFrame
        print(tree_rules)
        print(r2_score_)

        grid_search2 = hyperparameter_tuning(model, X, y, pv.hyperparameter_grid_RF, cv=custom_inner_splits, scoring='neg_root_mean_squared_error')
        df_results = pd.DataFrame(grid_search2.cv_results_)
        print(df_results)
        best_model = grid_search2.best_estimator_

        #build a "final" model for this train test split.
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        r2_score_ = r2_score(y_test, y_pred)
        print(r2_score_)
        print('stop')

    #     rf_model.model.random_state = 42
    #     rf_model.model.fit(X_train, y_train)
    #     y_pred = rf_model.predict(X_test)

    #     r2_score = rf_model.model.score(X_test, y_test)
    #     print(f'R2 score for outer fold {outer_fold}: {r2_score}')
    #     fold_results.append(r2_score) #append r2_score to the outer_fold scores
       
    #     all_true_pki_series = pd.concat([all_true_pki_series, y_test]).sort_index() #pd Series of all true pki values
    #     all_predicted_pki_series = pd.concat([all_predicted_pki_series, pd.Series(y_pred, index=y_test.index)]).sort_index()

    # #save the true and predicted values in a csv file for later analysis
    # mol_id_and_ns = df[['mol_id'] + (['conformations (ns)'] if 'conformations (ns)' in df else [])]
    # true_pred_pki_df = pd.DataFrame({
    #     'mol_id': mol_id_and_ns['mol_id'],
    #     'True_pKi': all_true_pki_series,
    #     'Predicted_pKi': all_predicted_pki_series
    # })

    # # Add 'conformations (ns)' only if it exists in mol_id_and_ns
    # if 'conformations (ns)' in mol_id_and_ns:
    #     true_pred_pki_df.insert(1, 'conformations (ns)', mol_id_and_ns['conformations (ns)'])

    # #make sure the sorting is nice, for better viewing
    # sort_columns = ['mol_id'] + (['conformations (ns)'] if 'conformations (ns)' in true_pred_pki_df else [])
    # true_pred_pki_df = true_pred_pki_df.sort_values(by=sort_columns).reset_index(drop=True)
    # save_path = dfs_path / 'ModelResults_RF' / 'true_vs_prediction' / f'{name}.csv'

    # # Ensure the parent directory exists
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # true_pred_pki_df.to_csv(save_path, index=False)

    # # Store the results for the fold
    # data = pd.Series(fold_results)
    # print(f'R2 score: {data.mean()}')
    # print(f'R2 std: {data.std()}')
    # fold_results = {
    #     "mol_id": name,
    #     "mean_test_score": data.mean(),
    #     "std_test_score": data.std(),
    #     "params": grid_search.best_params_,
    #     **{f"split{split_idx}_test_score": fold_results[split_idx] for split_idx in range(outer_folds)},  # Loop to add splits
    # }

    return #fold_results


def main(dfs_path = pv.dfs_descriptors_only_path_):

    #create folder for storing the models and results from them
    Modelresults_path = dfs_path / pv.Modelresults_folder_ #CHECK: if 'RF' is in public_variables or something else perhaps
    Modelresults_path.mkdir(parents=True, exist_ok=True)
    
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_exclude(dfs_path, exclude_files=['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_1000.csv','conformations_10.csv','conformations_1000_molid.csv','conformations_500.csv','conformations_200.csv','conformations_100.csv','conformations_50.csv','initial_dataframe.csv','initial_dataframes_best.csv','MD_output.csv','conformations_20.csv','minimized_conformations_10.csv','stable_conformations.csv'])
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_include(dfs_path, include_files=['1ns.csv','conformations_10c.csv'])
    # dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_include(dfs_path, include_files=['conformations_10c.csv'])

    print(dfs_in_dic.keys())

    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic_sorted = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(dfs_in_dic_sorted.keys())
    # first_four_keys = list(dfs_in_dic.keys())[-4:]
    # filtered_dict = {key: dfs_in_dic[key] for key in first_four_keys}
    # print(filtered_dict.keys())

    ModelResults_ = []

    outer_folds = 10
    inner_folds = 5
    scoring = 'r2'

    csv_filename = f'ModelResults_O{outer_folds}_I{inner_folds}_{scoring}_{pv.DESCRIPTOR}.csv'
    csv_filename_temp = f'results_Ko{outer_folds}_Ki{inner_folds}_{scoring}_{pv.DESCRIPTOR}_temp.csv'
    
    for name, df in dfs_in_dic_sorted.items():
        model = RandomForestRegressor(random_state=42)
        fold_results = nested_cross_validation(name, df, outer_folds, inner_folds)
        ModelResults_.append(fold_results)
        pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename_temp, index=False)

    pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename, index=False)
    return

if __name__ == "__main__":
    # main(public_variables.dataframes_master_ / '2D')
    # main(public_variables.dfs_descriptors_only_path_)
    # main(public_variables.dfs_descriptors_only_path_)
    # main(public_variables.dfs_PCA_path)
    # dataframes_master_ = public_variables.base_path_ / Path(f'dataframes_GSK3_WHIM')
    # # main(dataframes_master_ / 'reduced_t0.85')
    # # main(dataframes_master_ / 'reduced_t0.85_MD')
    # main(dataframes_master_ / 'MD only')

    # main(public_variables.dfs_reduced_path_)
    # main(public_variables.dfs_reduced_and_MD_path_)

    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    print(pv.dfs_descriptors_only_path_)
    main(pv.dfs_descriptors_only_path_)

    #NOTE: do in main, if i want to use dfs_descriptors_only etc or like all combinations

    