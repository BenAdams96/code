from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from plotting import A_true_vs_pred_plotting
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os

def save_fold_results(results, metric, ModelResults, Modelresults_path):
    # Process each metric separately
    # Append the results for this metric
    ModelResults[metric].append(results)
    
    # Write the updated results to a temporary CSV file
    csv_filename_temp = f'ModelResults_{metric}_{pv.DESCRIPTOR}_temp.csv'
    pd.DataFrame(ModelResults[metric]).to_csv(Modelresults_path / csv_filename_temp, index=False)
    # ModelResults_.append(fold_results)
    #     pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename_temp, index=False)
    csv_filename = f'ModelResults_{metric}_{pv.DESCRIPTOR}.csv'
    csv_filepath = Modelresults_path / csv_filename  # Ensure it's a Path object
    new_results_df = pd.DataFrame([results])
    if csv_filepath.exists():
        # Load existing results
        existing_results_df = pd.read_csv(csv_filepath)

        # Merge the new results into the existing DataFrame based on 'mol_id'
        updated_results_df = existing_results_df.set_index("mol_id") \
            .combine_first(new_results_df.set_index("mol_id")) \
            .reset_index()

        # Ensure new mol_id rows are added
        updated_results_df = pd.concat([updated_results_df, new_results_df]).drop_duplicates(subset=['mol_id'], keep='last')

        # Fix the order of mol_id
        sorted_mol_ids = csv_to_dictionary.get_sorted_columns(updated_results_df['mol_id'].tolist())
        updated_results_df = updated_results_df.set_index('mol_id').loc[sorted_mol_ids].reset_index()
    else:
        updated_results_df = new_results_df

    updated_results_df.to_csv(csv_filepath, index=False)
    return

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

def visualize_splits(custom_split, df, output_dir="splits_visualization"):
    """
    Visualizes the distribution of the PKI column for training and test sets 
    in the custom splits and saves histogram plots for each split.
    
    Parameters:
    - custom_split: list of tuples [(train_indices, test_indices), ...]
    - df: DataFrame containing the target column 'PKI'
    - output_dir: Directory to save the histogram plots (default: "splits_visualization")
    """
    import os
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, (train_indices, test_indices) in enumerate(custom_split):
        # Extract training and test targets
        train_targets = df.loc[train_indices, 'PKI']
        test_targets = df.loc[test_indices, 'PKI']

        # Plot the distributions
        plt.figure(figsize=(10, 6))
        plt.hist(train_targets, bins=20, alpha=0.7, label="Training Set", color="blue")
        plt.hist(test_targets, bins=20, alpha=0.7, label="Test Set", color="orange")
        plt.xlabel("PKI")
        plt.ylabel("Frequency")
        plt.title(f"Split {i+1}: PKI Distribution")
        plt.legend()
        
        # Save the plot
        plt.savefig(f"{output_dir}/split_{i+1}_pki_distribution.png")
        plt.close()

def Kfold_Cross_Validation_incl_grouped_RF(df, hyperparameter_grid, kfold_, scoring_):
    print("kfold new + grouped")
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}

    targets = df['PKI']
    unique_mol_ids = df['mol_id'].unique()
    
    kf = KFold(n_splits=kfold_, shuffle=True, random_state=1)
    splits = list(kf.split(unique_mol_ids))
        
    custom_splits = []
    for train_mol_indices, test_mol_indices in kf.split(unique_mol_ids):
        # Get actual molecule IDs for train and test sets
        train_mols = unique_mol_ids[train_mol_indices]
        test_mols = unique_mol_ids[test_mol_indices]
        
        # Map back to the full dataset indices (6000 rows)
        train_indices = df[df['mol_id'].isin(train_mols)].index
        test_indices = df[df['mol_id'].isin(test_mols)].index
        
        # Append as a tuple of arrays
        custom_splits.append((train_indices, test_indices))
    # print(custom_splits[3][0])
    # print(custom_splits[3][1])

    visualize_splits(custom_splits, df, output_dir=pv.base_path_)
    # print('custom splits')
    # print(custom_splits[1][0][0:40])
    # print(custom_splits[1][1][0:40])
    # print(targets[0:40])
    # Initial model and grid search outside the loop
    rf_model = RandomForestModel(n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt')
    rf_model.model.random_state = 42
    df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
    print(df)
    grid_search = rf_model.hyperparameter_tuning(df,targets,hyperparameter_grid,cv=custom_splits,scoring_=scoring_)
    
    df_results = pd.DataFrame(grid_search.cv_results_) #what results? of cv. so the 10 results and 8 rows (each row was a set of hyperparameters)
    
    #grid_search.best_index_ = the index that is best of the x amount of hyperparameter combinations
    result = df_results.loc[grid_search.best_index_, columns_[1:]] #1: because we dont want to include 'time'
    #df_results = dataframe with 'hyperparametercombinations' amount of rows, and each row has mean_test_score and split_test_scores etc
    #result = of the best combination of hyperparameters. mean_test_score, std, params, split 0 to 4
    #make it so that the scores are always positive
    result["mean_test_score"] = abs(result['mean_test_score']) 
    for i in range(kfold_):
        result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

    result_df_row = result.to_frame().T
    result_df_row['mol_id'] = '2D' #add a column of the '1ns' etc
    result_df_row = result_df_row[columns_] #change the order to the one specified above

    #make 1 big dataframe for all the '0ns' '1ns' 'rdkit_min' etc
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
    models['2D'] = rf_model
    return models, ModelResults_

def Kfold_Cross_Validation_incl_grouped_XGB(df, hyperparameter_grid, kfold_, scoring_):
    print("kfold new + grouped")
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}

    targets = df['PKI']
    unique_mol_ids = df['mol_id'].unique()
    
    kf = KFold(n_splits=kfold_, shuffle=True, random_state=1)
    splits = list(kf.split(unique_mol_ids))
    
    custom_splits = []
    for train_mol_indices, test_mol_indices in kf.split(unique_mol_ids):
        # Get actual molecule IDs for train and test sets
        train_mols = unique_mol_ids[train_mol_indices]
        test_mols = unique_mol_ids[test_mol_indices]
        
        # Map back to the full dataset indices (6000 rows)
        train_indices = df[df['mol_id'].isin(train_mols)].index
        test_indices = df[df['mol_id'].isin(test_mols)].index
        
        # Append as a tuple of arrays
        custom_splits.append((train_indices, test_indices))
    # print('custom splits')
    # print(custom_splits[1][0][0:40])
    # print(custom_splits[1][1][0:40])
    # print(targets[0:40])
    # Initial model and grid search outside the loop
    rf_model = XGBoostModel()
    rf_model.model.random_state = 42
    df = df.drop(columns=['mol_id','PKI','conformations (ns)'], axis=1, errors='ignore')
    
    #NOTE: careful: conformations (ns) is still in it
    grid_search = rf_model.hyperparameter_tuning(df,targets,hyperparameter_grid,cv=custom_splits,scoring_=scoring_)
    
    df_results = pd.DataFrame(grid_search.cv_results_) #what results? of cv. so the 10 results and 8 rows (each row was a set of hyperparameters)
    
    #grid_search.best_index_ = the index that is best of the x amount of hyperparameter combinations
    result = df_results.loc[grid_search.best_index_, columns_[1:]] #1: because we dont want to include 'time'
    #df_results = dataframe with 'hyperparametercombinations' amount of rows, and each row has mean_test_score and split_test_scores etc
    #result = of the best combination of hyperparameters. mean_test_score, std, params, split 0 to 4
    #make it so that the scores are always positive
    result["mean_test_score"] = abs(result['mean_test_score']) 
    for i in range(kfold_):
        result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

    result_df_row = result.to_frame().T
    result_df_row['mol_id'] = 'XGB' #add a column of the '1ns' etc
    result_df_row = result_df_row[columns_] #change the order to the one specified above

    #make 1 big dataframe for all the '0ns' '1ns' 'rdkit_min' etc
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
    models['XGB'] = rf_model
    return models, ModelResults_

def Kfold_Cross_Validation_incl_grouped_SVM(df, hyperparameter_grid, kfold_, scoring_):
    print("kfold new + grouped")
    
    columns_ = ["mol_id", "mean_test_score", "std_test_score", "params"]
    split_columns = [f'split{i}_test_score' for i in range(kfold_)]
    columns_ += split_columns
    
    ModelResults_ = pd.DataFrame(columns=columns_)
    models = {}


    targets = df['PKI']
    unique_mol_ids = df['mol_id'].unique()
    
    kf = KFold(n_splits=kfold_, shuffle=True, random_state=1)
    splits = list(kf.split(unique_mol_ids))
    
    custom_splits = []
    for train_mol_indices, test_mol_indices in kf.split(unique_mol_ids):
        # Get actual molecule IDs for train and test sets
        train_mols = unique_mol_ids[train_mol_indices]
        test_mols = unique_mol_ids[test_mol_indices]
        
        # Map back to the full dataset indices (6000 rows)
        train_indices = df[df['mol_id'].isin(train_mols)].index
        test_indices = df[df['mol_id'].isin(test_mols)].index
        
        # Append as a tuple of arrays
        custom_splits.append((train_indices, test_indices))
    # print('custom splits')
    # print(custom_splits[1][0][0:40])
    # print(custom_splits[1][1][0:40])
    # print(targets[0:40])
    # Initial model and grid search outside the loop
    # Initialize SVM model
    print(custom_splits)
    svm_model = SupportVectorMachineRegressor(C=1.0, kernel='rbf')  # Default values, customize as needed

    # Drop columns not used in features
    df_features = df.drop(columns=['mol_id', 'PKI', 'conformations (ns)'], axis=1, errors='ignore')

    # Perform grid search with the SVM model
    grid_search = svm_model.hyperparameter_tuning(df_features, targets, pv.HYPERPARAMETER_GRID, cv=custom_splits, scoring_=scoring_)

    # Collect results
    df_results = pd.DataFrame(grid_search.cv_results_)
    result = df_results.loc[grid_search.best_index_, columns_[1:]]
    
    # Make scores positive
    result["mean_test_score"] = abs(result['mean_test_score'])
    for i in range(kfold_):
        result[f'split{i}_test_score'] = abs(result[f'split{i}_test_score'])

    result_df_row = result.to_frame().T
    result_df_row['mol_id'] = 'SVM'
    result_df_row = result_df_row[columns_]

    # Concatenate results for all mol_id categories
    ModelResults_ = pd.concat([ModelResults_, result_df_row], ignore_index=True)
    models['SVM'] = svm_model
    
    return models, ModelResults_


def create_true_pred_dataframe(name, df, dfs_path, all_idx_ytrue_pki_series,all_idx_ypredicted_pki_series):

    #idx, mol_id, conformations (ns) of the whole df
    mol_id_and_ns = df[['mol_id'] + (['conformations (ns)'] if 'conformations (ns)' in df else [])]
    
    #idx, mol_id, y_true, y_pred
    df_ytrue_ypred_pki = pd.DataFrame({
        'mol_id': mol_id_and_ns['mol_id'],
        'True_pKi': all_idx_ytrue_pki_series,
        'Predicted_pKi': all_idx_ypredicted_pki_series
    })
    
    # Add 'conformations (ns)' only if it exists in mol_id_and_ns. why? no need to
    if 'conformations (ns)' in mol_id_and_ns:
        df_ytrue_ypred_pki.insert(1, 'conformations (ns)', mol_id_and_ns['conformations (ns)'])

    #make sure the sorting is nice, for better viewing. (all 10 conformations of mol_id 1 first, etc.)
    sort_columns = ['mol_id'] + (['conformations (ns)'] if 'conformations (ns)' in df_ytrue_ypred_pki else [])
    df_ytrue_ypred_pki = df_ytrue_ypred_pki.sort_values(by=sort_columns).reset_index(drop=True)

    # Ensure the parent directory exists, save the true and predicted values in a csv file for later analysis
    save_path = dfs_path / pv.true_predicted

    save_path.mkdir(parents=True, exist_ok=True)
    df_ytrue_ypred_pki.to_csv(save_path / f'{name}_true_predicted.csv', index=False)

    return df_ytrue_ypred_pki

def nested_cross_validation(name, df, dfs_path, outer_folds=10, inner_folds=5, scoring = 'r2'):
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
    print(pv.ML_MODEL)
    if pv.ML_MODEL.name == 'SVM':
        print('this is SVM!')
        feature_cols = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], errors='ignore').columns

        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print('df Scaled!')
    print(df)

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

    fold_results = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }

    fold_feature_importance = []
    custom_outer_splits = []
    all_best_params_outer = []
    all_predictions = []
    all_true_values = []
    all_idx_ytrue_pki_series = pd.Series(dtype=float)
    all_idx_ypredicted_pki_series = pd.Series(dtype=float)
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,shuffle=True, random_state=10)
    test_molid_ytrue_pki_series = pd.Series(dtype=float)

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
        model_instance = pv.ML_MODEL.model #create random forest model for example

        grid_search = hyperparameter_tuning(model_instance, X, y, pv.ML_MODEL.hyperparameter_grid, cv=custom_inner_splits, scoring=scoring)
        df_results = pd.DataFrame(grid_search.cv_results_)
        all_best_params_outer.append(grid_search.best_params_)
        # print(df_results)
        
        #build a "final" model for this train test split.
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
        print(pv.ML_MODEL.name)
        if pv.ML_MODEL.name != 'SVM':
            print('no svm')
            importance = best_model.feature_importances_

            # Store the feature importance for the current fold
            fold_feature_importance.append(importance)

        y_pred = pd.Series(best_model.predict(X_test), index=y_test.index, name='Predicted_pKi')

        # Get the mol_id for the current fold based on the test indices (get average prediction for each molecule)
        mol_id_for_fold = df.loc[test_idx_all, 'mol_id']
        y_test_average = y_test.groupby(mol_id_for_fold).mean()
        y_pred_average = y_pred.groupby(mol_id_for_fold).mean()
        r2_value = r2_score(y_test_average, y_pred_average)
        mse_value = mean_squared_error(y_test_average, y_pred_average)
        mae_value = mean_absolute_error(y_test_average, y_pred_average)
        print(f"Fold {outer_fold}; MSE: {mse_value:.4f}, MAE: {mae_value:.4f}, R² score: {r2_value:.4f}")
        
        fold_results['R2'].append(r2_value)
        fold_results['MSE'].append(mse_value)
        fold_results['MAE'].append(mae_value)

        all_predictions.extend(y_pred_average)
        all_true_values.extend(y_test_average)

        #get two series with idx and pki. every outerfold it adds the new test part.
        all_idx_ytrue_pki_series = pd.concat([all_idx_ytrue_pki_series, y_test]).sort_index() #pd Series of all true pki values
        all_idx_ypredicted_pki_series = pd.concat([all_idx_ypredicted_pki_series, pd.Series(y_pred, index=y_test.index)]).sort_index()
    # if pv.ML_MODEL.name != 'SVM':
    #     plot_feature_importance(X, fold_feature_importance, dfs_path, name)
    mean_scores = {}
    #get once the the mean scores using all the predictions
    r2_value = r2_score(all_true_values, all_predictions)
    mse_value = mean_squared_error(all_true_values, all_predictions)
    mae_value = mean_absolute_error(all_true_values, all_predictions)
    mean_scores['R2'] = r2_value
    mean_scores['MSE'] = mse_value
    mean_scores['MAE'] = mae_value

    # create_true_pred_dataframe(name, df, dfs_path, all_idx_ytrue_pki_series, all_idx_ypredicted_pki_series)
    df_ytrue_ypred_pki = create_true_pred_dataframe(name, df, dfs_path, all_idx_ytrue_pki_series,all_idx_ypredicted_pki_series)
    A_true_vs_pred_plotting.plot_avg_predicted_vs_real_pKi(df_ytrue_ypred_pki, name, dfs_path)
    # Store the results for the fold
    results_all = {}
    for metric, fold_list in fold_results.items():
        fold_array = np.array(fold_list, dtype=np.float64)    
        results = {
            "mol_id": name,
            "total_mean_score": mean_scores[metric],
            "mean_test_score": fold_array.mean(),
            "std_test_score": fold_array.std(),
            **{f"split{split_idx}_test_score": fold_array[split_idx] for split_idx in range(len(fold_array))},
            **{f"split{split_idx}_hyperparameter_set": all_best_params_outer[split_idx] for split_idx in range(len(all_best_params_outer))},  # Loop to add splits
        }
        results_all[metric] = results
    # print(results_all)
    return results_all

def main(dfs_2D_path):  ###set as default, but can always change it to something else.
    csv_file_path =  dfs_2D_path / f'2D_ECFP_{pv.PROTEIN}.csv'

    #create folder for storing the models and results from them
    Modelresults_path = pv.dfs_2D_path / pv.Modelresults_folder_
    Modelresults_path.mkdir(parents=True, exist_ok=True)
    print(Modelresults_path)
    df = pd.read_csv(csv_file_path)
    
    #remove the mol_id and PKI #NOTE: empty rows have already been removed beforehand, but still do it just to be sure!
    columns_to_drop = ['mol_id', 'PKI', "conformations (ns)"]

    # # parameter_grid = public_variables.parameter_grid_
    # parameter_grid = {
    #         'kfold_': [10],
    #         'scoring_': [('neg_root_mean_squared_error','RMSE')],
    #         } #kfolds (5, 10) and metrics (rmse, r2)
    # hyperparameter_grid_RF = {
    #         'n_estimators': [50,100,150],
    #         'max_depth': [8,15],
    #         'min_samples_split': [8,15],
    #         'min_samples_leaf': [8,15],
    #         'max_features': ['sqrt'] #'None' can lead to overfitting.
    #     }
    
    # # # hyperparameter_grid_XGboost = {
    # #     'n_estimators': [50, 100, 150],          # Number of trees (lower values for quicker training)
    # #     'max_depth': [3, 5, 7],             # Maximum depth of each tree (shallower trees to avoid overfitting)
    # #     'learning_rate': [0.01, 0.1],       # Learning rate (smaller values for more gradual training)
    # #     'subsample': [0.6, 0.8],            # Subsample ratio of the training instance (to prevent overfitting)
    # #     'colsample_bytree': [0.6, 0.8],     # Subsample ratio of columns when constructing each tree
    # #     'gamma': [0, 0.1],                  # Minimum loss reduction required to make a further partition on a leaf node
    # # }
    # hyperparameter_grid_XGB = {
    # 'n_estimators': [50, 100],                # Limit the number of boosting rounds
    # 'max_depth': [3, 5],                       # Shallower trees to prevent overfitting
    # 'learning_rate': [0.01, 0.05],             # Lower learning rate to control overfitting
    # 'subsample': [0.6, 0.8],                   # Limit subsampling to reduce overfitting risk
    # 'colsample_bytree': [0.6, 0.8],            # Limit feature subsampling
    # 'gamma': [0, 0.1],                         # Keep gamma small to avoid excessive pruning
    # }

    # hyperparameter_grid_SVM = {
    # 'kernel': ['rbf','linear'],        # Most commonly effective kernels
    # 'C': [0.1, 1,10],                      # Reasonable regularization strengths for regression
    # 'epsilon': [0.1],             # Standard values for error tolerance
    # 'gamma': ['scale', 0.1]             # Default and a specific small value for tuning influence
    # }

    # param_combinations = list(itertools.product(pv.ML_MODEL.hyperparameter_grid['kfold_'], parameter_grid['scoring_']))
    # print(param_combinations)
    # dic_models = {}
    outer_folds = 10
    inner_folds = 5
    scoring = 'r2'
    ModelResults = {'R2': [], 'MSE': [], 'MAE': []}

    fold_results = nested_cross_validation('2D', df, pv.dfs_2D_path, outer_folds, inner_folds, scoring)
    for metric, results in fold_results.items():
            save_fold_results(results, metric, ModelResults, Modelresults_path)
    # for kfold_value, scoring_value in param_combinations:
    #     print(f"kfold_: {kfold_value}, scoring_: {scoring_value[0]}")
    #     models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped_RF(df, hyperparameter_grid_RF, kfold_=kfold_value, scoring_=scoring_value[0])
    #     print('done with Kfold cross validation')
    #     csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_2D.csv'
    #     Modelresults_.to_csv(Modelresults_path_RF / csv_filename, index=False)


    #     models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped_XGB(df, hyperparameter_grid_XGB, kfold_=kfold_value, scoring_=scoring_value[0])
    #     print('done with Kfold cross validation')
    #     csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_2D.csv'
    #     Modelresults_.to_csv(Modelresults_path_XGB / csv_filename, index=False)


    #     models_dic, Modelresults_ = Kfold_Cross_Validation_incl_grouped_SVM(df, hyperparameter_grid_SVM, kfold_=kfold_value, scoring_=scoring_value[0])
    #     print('done with Kfold cross validation')
    #     csv_filename = f'results_K{kfold_value}_{scoring_value[1]}_2D.csv'
    #     Modelresults_.to_csv(Modelresults_path_SVM / csv_filename, index=False)
        
        #instantialize the dic of dictionaries to store all models eventually.
        #TODO: make it so that the dataframe is also stored with it. no. later perhaps
        
        # save_models(original_models,folder_for_results_path,kfold_value,scoring_value[1])
        # visualize_scores(results_df, kfold_=kfold_value, scoring_=scoring_value[1], save_plot_folder=folder_for_results_path)

        # modelnames = f'RF_Allmodels_k{kfold_value}_{scoring_value[1]}'
        # dic_models[modelnames] = {}
        # dic_models[modelnames]['original models'] = models_dic
        # visualize_scores_box_plot(Modelresults_, kfold_value,scoring_value[1], Modelresults_path)
        # print(models_dic)

        # save_models_reduced(red_models[0],folder_for_results_path,kfold_value,scoring_value[1])
        # visualize_scores_box_plot_reduced(red_models[1], kfold_value,scoring_value[1], folder_for_results_path)
    # print(dic_models)

    #print(dic_models['RF_Allmodels_k5_RMSE']['original models']['1.5ns']) to acces a model

    #TODO: general script that contains save models instead of in randomforest_read_in_models
    #save_originalmodels_hdf5(Modelresults_path, dic_models, dfs_stripped)
    # randomForest_read_in_models.save_model_dictionary(Modelresults_path,'original_models_dic.pkl',dic_models)
    # return

    return

if __name__ == "__main__":
    # for model in Model_classic:
    hpset = ['small', 'big']
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1, hyperparameter_set='small')
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4)
    # main(pv.dfs_2D_path)

    pv.update_config(model_=Model_classic.XGB, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    main(pv.dfs_2D_path)
    pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    main(pv.dfs_2D_path)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dfs_2D_path)


    # for set in hpset:
        # for model in Model_classic:
    # models = [Model_classic.RF, Model_classic.XGB]
    # for protein in DatasetProtein:
    #     print(protein)
    #     for model in models:
    #         pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=protein, hyperparameter_set='big')
    #         main(pv.dfs_2D_path)
    # pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=set)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD, hyperparameter_set=set)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4, hyperparameter_set=set)
    # main(pv.dfs_2D_path)













    # for protein in DatasetProtein:
    #     pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=protein, hyperparameter_set='big')
    #     main(pv.dfs_2D_path)
        # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=protein, hp_set='small')
        # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(pv.dfs_2D_path)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dfs_2D_path)

