from sklearn.model_selection import GridSearchCV, KFold

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from plotting import A_true_vs_pred_plotting
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np

import joblib

import pandas as pd
import math
import re
import os

def save_average_feature_importance_plot(average_feature_importance, feature_names, save_path, name):
    """
    Saves a plot of the average feature importance across all folds using SHAP.

    Parameters:
    - average_feature_importance: The mean of SHAP values across all folds (array of feature importances)
    - feature_names: List of feature names (same order as the columns in X)
    - save_path: Path where the plot should be saved
    """
    
    # If average_feature_importance is a dictionary, extract its values
    if isinstance(average_feature_importance, dict):
        average_feature_importance = np.mean(list(average_feature_importance.values()), axis=0)
    
    # Create a bar plot for the average feature importance
    plt.figure(figsize=(10, 6))
    feature_idx = np.argsort(average_feature_importance)  # Sort feature importances
    plt.barh(range(len(average_feature_importance)), average_feature_importance[feature_idx], align='center')
    plt.yticks(range(len(average_feature_importance)), np.array(feature_names)[feature_idx])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Features")
    plt.title("Average Feature Importance Across Folds")
    plt.gca().invert_yaxis()  # Invert y-axis to display most important features at the top
    
    # Save the plot as a file
    plt.savefig(save_path / f'shap_average_feature_importance_{name}')
    plt.close()

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
        sorted_mol_ids = dataframe_processing.sort_columns(updated_results_df['mol_id'].tolist())
        updated_results_df = updated_results_df.set_index('mol_id').loc[sorted_mol_ids].reset_index()
    else:
        updated_results_df = new_results_df

    updated_results_df.to_csv(csv_filepath, index=False)
    return


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
        verbose=3,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)

    return grid_search

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

def plot_feature_importance(X, fold_feature_importance, dfs_path, name):
    # Assuming 'best_model' is fitted after hyperparameter tuning

    # After all outer folds are completed
    print(fold_feature_importance)
    average_importance = np.mean(fold_feature_importance, axis=0)

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': X.columns,  # Replace with your actual feature names
        'Importance': average_importance
    }).sort_values(by='Importance', ascending=False)

    # Optionally, plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel("Average Feature Importance")
    plt.title("Average Feature Importance Across Folds")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.savefig(dfs_path / pv.Modelresults_folder_ / f'average_feature_importance_{name}.png')

    return

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
    
    if pv.ML_MODEL.name == 'SVM':
        df = dataframe_processing.standardize_dataframe(df)
        # feature_cols = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], errors='ignore').columns

        # scaler = StandardScaler()
        # df[feature_cols] = scaler.fit_transform(df[feature_cols])


    X = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
    y = df[target_column]  # Target (pKi values)
    # Stratified outer loop
    num_of_bins = 5
    bins, binned_y = bin_pki_values(y, num_bins=num_of_bins)  # Bin pKi values for stratification, still a dataframe: all molecules with an index of which bin they are in
    unique, counts = np.unique(binned_y, return_counts=True)
    print("bins:", bins)
    print("counts in bin:", counts)
    # Group by mol_id (reduces dataframe to the amount of molecules)
    grouped_df = df.groupby('mol_id').first()  # One row per molecule, so group by mol_id

    #have 5 bins of equal lengths, and bin the molecule based on pki in the correct group
    #series with index: mol_id and as value the bin index (ranging from 0 to 4)
    grouped_binned_y = bin_pki_values(grouped_df[target_column], num_bins=num_of_bins)[1]

    unique_mol_ids = grouped_df.index  # Index([1,2,3,4,6,...,615]) array of all mol_id
    # print(grouped_df)
    # print(grouped_binned_y)
    # print(all_mol_ids)
    
    fold_results = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }
    X_all_folds = []
    shap_values_per_fold = {}
    feature_importance_per_fold = {}
    r2_train_scores = []
    fold_feature_importance = []
    custom_outer_splits = []
    all_best_params_outer = []
    all_predictions = []
    all_true_values = []
    all_idx_ytrue_pki_series = pd.Series(dtype=float)
    all_idx_ypredicted_pki_series = pd.Series(dtype=float)
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,shuffle=True, random_state=10)
    test_molid_ytrue_pki_series = pd.Series(dtype=float)
    all_splits = list(outer_cv.split(X=grouped_df, y=grouped_binned_y, groups=unique_mol_ids))
    # print(all_splits)
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
        X_all_folds.append(X_test)
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
        all_splits = list(inner_cv.split(grouped_X_train, grouped_binned_y_train, groups=grouped_X_train.index))
        # print(all_splits)
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
        #back to outerfold
        model_instance = pv.ML_MODEL.model #create random forest model for example

        grid_search = hyperparameter_tuning(model_instance, X, y, pv.ML_MODEL.hyperparameter_grid, cv=custom_inner_splits, scoring=scoring)
        df_results = pd.DataFrame(grid_search.cv_results_)
        print(df_results)
        all_best_params_outer.append(grid_search.best_params_)
        # print(df_results)
        
        #build a "final" model for this train test split.
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
        print(pv.ML_MODEL.name)

        # shap_base_path = dfs_path / pv.Modelresults_folder_ / 'shap_info' / name / f'fold{outer_fold}' #create Modelresults folder
        # shap_base_path.mkdir(parents=True, exist_ok=True)
        # if pv.ML_MODEL.name != 'SVM':
        #     print('no svm')
        #     importance = best_model.feature_importances_
        #     # SHAP
        #     explainer = shap.TreeExplainer(best_model)
        #     shap_values = explainer(X_test)

        #     # Save everything
        #     joblib.dump(best_model, shap_base_path / f'model_fold{outer_fold}.pkl')
        #     joblib.dump(shap_values, shap_base_path / f'shap_values_fold{outer_fold}.pkl')
        #     joblib.dump(X_test, shap_base_path / f'X_test_fold{outer_fold}.pkl')

        # elif pv.ML_MODEL.name == 'SVM':
        #     print('svm')
        #     # Check if the SVM model is linear or non-linear
        #     if best_model.kernel == 'linear':
        #         print('Linear SVM')
        #         X_train_temp = X_train.set_index(df.loc[X_train.index, 'mol_id'])   
        #         X_test_temp = X_test.set_index(df.loc[X_test.index, 'mol_id'])

        #         X_train_mean = X_train_temp.groupby(X_train_temp.index).mean()  # Mean across features
        #         X_test_mean = X_test_temp.groupby(X_test_temp.index).mean()  # Mean across features

        #         explainer = shap.KernelExplainer(best_model.predict, X_train_mean)
        #         shap_values = explainer.shap_values(X_test_mean)
        #     else:
        #         print('Non-linear SVM')
        #         X_train_temp = X_train.set_index(df.loc[X_train.index, 'mol_id'])   
        #         X_test_temp = X_test.set_index(df.loc[X_test.index, 'mol_id'])            

        #         # SHAP for Non-linear SVM (e.g., RBF kernel): Use mean of X_train as the background data
        #         X_train_mean = X_train_temp.groupby(X_train_temp.index).mean()  # Mean across features
        #         X_test_mean = X_test_temp.groupby(X_test_temp.index).mean()  # Mean across features
        #         print(X_train_mean)

        #         print(X_test_mean)
        #         explainer = shap.KernelExplainer(best_model.predict, X_train_mean)
        #         print('start shap')
        #         shap_values = explainer.shap_values(X_test_mean)
        #         print('end shap')

        #         print(shap_values)
        #         print(shap_values)

        #     # Save everything
        #     joblib.dump(best_model, shap_base_path / f'model_fold{outer_fold}.pkl')
        #     joblib.dump(shap_values, shap_base_path / f'shap_values_fold{outer_fold}.pkl')
        #     joblib.dump(X_test, shap_base_path / f'X_test_fold{outer_fold}.pkl')
        mol_id_for_fold = df.loc[train_idx_all, 'mol_id']
        y_train_average = y_train.groupby(mol_id_for_fold).mean()
        y_pred_train = pd.Series(best_model.predict(X_train), index=y_train.index, name='Predicted_pKi')
        y_pred_train_average = y_pred_train.groupby(mol_id_for_fold).mean()
        r2_value_train = r2_score(y_train_average, y_pred_train_average)
        print(r2_value_train)
        r2_train_scores.append(r2_value_train)
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
    #     # average_feature_importance = np.mean(list(feature_importance_per_fold.values()), axis=0)
    #     # save_average_feature_importance_plot(average_feature_importance, X.columns, dfs_path, name)
    #     plot_feature_importance(X, fold_feature_importance, dfs_path, name)

        # X_all = np.concatenate(X_all_folds, axis=0)

    mean_scores = {}
    #get once the the mean scores using all the predictions
    r2_value = r2_score(all_true_values, all_predictions)
    mse_value = mean_squared_error(all_true_values, all_predictions)
    mae_value = mean_absolute_error(all_true_values, all_predictions)
    mean_scores['R2'] = r2_value
    mean_scores['MSE'] = mse_value
    mean_scores['MAE'] = mae_value

    #write out the r2 train
    # Define the output file path
    output_file = dfs_path / pv.Modelresults_folder_ / "R2_train_scores.csv"

    # Create DataFrame for the current run
    df_new = pd.DataFrame({
        'mol_id': [name],  # Store 'name' as a single-item list
        **{f'split{i+1}_train_score': [r2_train_scores[i]] for i in range(10)}  # Create 10 columns
    })

    # If the file exists, update or append the data
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)

        # Check if the molecule ID already exists
        if name in df_existing['mol_id'].values:
            # Update the existing row with new scores
            idx = df_existing.index[df_existing['mol_id'] == name].tolist()[0]  # Get index of existing row
            for i in range(10):
                df_existing.at[idx, f'split{i+1}_train_score'] = r2_train_scores[i]
        else:
            # Append new data if mol_id is not in the existing file
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # If file does not exist, create it with new data
        df_existing = df_new

    # Save back to CSV
    df_existing.to_csv(output_file, index=False)
    print(f"Updated R² train scores saved to {output_file}")

    df_ytrue_ypred_pki = create_true_pred_dataframe(name, df, dfs_path, all_idx_ytrue_pki_series, all_idx_ypredicted_pki_series)
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
    return results_all


def main(dfs_path = pv.dfs_descriptors_only_path_,  include_files = []):

    #create folder for storing the models and results from them
    Modelresults_path = dfs_path / pv.Modelresults_folder_ #create Modelresults folder
    Modelresults_path.mkdir(parents=True, exist_ok=True)
    
    if not include_files:
        include_files = [0]
    #,'3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv',
    dfs_in_dict = dataframe_processing.csvfiles_to_dict_include(dfs_path, include_files=include_files) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    print(dfs_in_dict)
    outer_folds = 10
    inner_folds = 5
    scoring = 'neg_mean_squared_error'
    ModelResults = {'R2': [], 'MSE': [], 'MAE': []}

    for name, df in dfs_in_dict.items():
        # Perform nested cross-validation for the current dataset
        fold_results = nested_cross_validation(name, df, dfs_path, outer_folds, inner_folds, scoring)
        for metric, results in fold_results.items():
            save_fold_results(results, metric, ModelResults, Modelresults_path)

    # for metric, results in ModelResults.items():
    #     csv_filename = f'ModelResults_{metric}_{pv.DESCRIPTOR}.csv'
    #     csv_filepath = Modelresults_path / csv_filename  # Ensure it's a Path object

    #     # Convert new results to a DataFrame
    #     new_results_df = pd.DataFrame(results)
    #     if csv_filepath.exists():
    #         # Load existing file
    #         existing_results_df = pd.read_csv(csv_filepath)

    #         # Update only matching 'mol_id' rows and keep others
    #         existing_results_df.update(new_results_df)

    #         # Add any new 'mol_id' rows that were not in the existing file
    #         updated_results_df = pd.concat([existing_results_df, new_results_df]).drop_duplicates(subset=['mol_id'], keep='last')
    #         # Restore the original order of mol_id from existing_results_df
    #         updated_results_df = existing_results_df[['mol_id']].merge(updated_results_df, on='mol_id', how='left')
    #     else:
    #         updated_results_df = new_results_df

    #     # Save back to CSV
    #     updated_results_df.to_csv(csv_filepath, index=False)        
    return

if __name__ == "__main__":
    include_files = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20','c50']
    include_files = [1]
    pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    main(pv.dfs_descriptors_only_path_,include_files = include_files)

    main(pv.dfs_descriptors_only_path_,include_files = include_files)
    for model in Model_classic:
        for descriptor in Descriptor:
            print(model)
            print(descriptor)
    
            pv.update_config(model_=model, descriptor_=descriptor, protein_=DatasetProtein.GSK3)
            main(pv.dfs_descriptors_only_path_,include_files = include_files)
            # main(pv.dfs_reduced_path_,include_files = include_files)
            # #add pca
            # main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
            # #add pca+MD
            # main(pv.dfs_MD_only_path_,include_files = include_files)
    

            # pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1, hyperparameter_set=x)
            # main(pv.dfs_descriptors_only_path_,include_files = include_files)
            # main(pv.dfs_reduced_path_,include_files = include_files)
            # main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
            # main(pv.dfs_MD_only_path_,include_files = include_files)
            # main(pv.dfs_dPCA_MD_path_,include_files = include_files)

        # pv.update_config(model_=Model_classic.XGB, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=x)
        # main(pv.dfs_MD_only_path_,include_files = include_files)

        # pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=x)
        # main(pv.dfs_descriptors_only_path_,include_files = include_files)
        # pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=x)
        # main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
        # pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=x)
        # main(pv.dfs_MD_only_path_,include_files = include_files)
    # main(pv.dfs_descriptors_only_path_)
    # for model in Model_classic:
    #     print(model)
    #     for protein in DatasetProtein:
    #         print(protein)
    #         pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=protein)
    #         for path in pv.dataframes_master_.iterdir():
    #             if path.is_dir() and not path.name.startswith('boxplots'):
    #                 print(path)
    #                 main(path)
    # main(pv.dataframes_master_ / '(DescMD)PCA_10')
    # main(pv.dataframes_master_ / '(DescMD)PCA_20')
    # main(pv.dataframes_master_ / 'DescPCA20 MDnewPCA')
    # main(pv.dataframes_master_ / 'DescPCA20 MDnewPCA minus PC1')
    # main(pv.dataframes_master_ / 'DescPCA20 MDnewPCA minus PCMD1')
    # main(pv.dataframes_master_ / 'MD_old only')
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dataframes_master_ / 'MD_new only4',include_files = ['0ns.csv','1ns.csv','3ns.csv','5ns.csv','7ns.csv','9ns.csv','c10.csv'])
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dataframes_master_ / 'MD_new only4',include_files = ['0ns.csv','1ns.csv','3ns.csv','5ns.csv','7ns.csv','9ns.csv','c10.csv'])
    # pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dataframes_master_ / 'MD_new only4',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dataframes_master_ / 'MD_new onlyall',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dataframes_master_ / 'MD_new only3',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dataframes_master_ / 'MD_new only4',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dataframes_master_ / 'MD_old only',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dfs_descriptors_only_path_,include_files = ['0ns.csv','1ns.csv','3ns.csv','5ns.csv','7ns.csv','9ns.csv','conformations_10.csv'])
    
    
    
    
    
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dataframes_master_ / 'MD_new onlyall',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / "desc_PCA15",include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / 'dPCA MD_new',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / 'red MD_new',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dfs_descriptors_only_path_,include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dfs_reduced_path_,include_files = ['0ns.csv','1ns.csv','c10.csv'])

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(pv.dataframes_master_ / 'MD_new onlyall',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / "desc_PCA15",include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / 'dPCA MD_new',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / 'red MD_new',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dfs_descriptors_only_path_,include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dfs_reduced_path_,include_files = ['0ns.csv','1ns.csv','c10.csv'])

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dataframes_master_ / 'MD_new onlyall',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / "desc_PCA15",include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / 'dPCA MD_new',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dataframes_master_ / 'red MD_new',include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dfs_descriptors_only_path_,include_files = ['0ns.csv','1ns.csv','c10.csv'])
    # main(pv.dfs_reduced_path_,include_files = ['0ns.csv','1ns.csv','c10.csv'])









    # main(pv.dataframes_master_ / 'MD_new only3',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)

    # main(pv.dataframes_master_ / 'dPCA MD2',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dataframes_master_ / 'MD_new only3',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dataframes_master_ / 'MD_new only4',include_files = ['0ns.csv','1ns.csv','c10.csv','conformations_10.csv'])
    # main(pv.dataframes_master_ / 'MDnewPCA')
    # main(pv.dataframes_master_ / 'red MD_old')
    # main(pv.dataframes_master_ / 'red MD_new')
    # main(pv.dataframes_master_ / 'MD_new only reduced')
    # main(pv.dataframes_master_ / 'red MD_new reduced')







    # for path in pv.dataframes_master_.iterdir():
    #     if path.is_dir() and not path.name.startswith('boxplots'):
    #         print(path)
    #         main(path, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','10ns.csv','conformations_10.csv'])



    # pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)

    # main(pv.dfs_descriptors_only_path_)
    # main(pv.dfs_reduced_path_)
    # main(pv.dfs_reduced_and_MD_path_)
    # main(pv.dfs_MD_only_path_)

    # main(pv.dfs_reduced_PCA_path_)
    # main(pv.dfs_reduced_MD_PCA_path_)
    # main(pv.dfs_reduced_and_MD_combined_path_)
    # main(pv.dfs_all_PCA)





    