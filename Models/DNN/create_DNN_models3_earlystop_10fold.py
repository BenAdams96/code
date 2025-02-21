from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import os
import torch
import json
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import random

from Models.DNN.FullyConnectedDNN_class2 import FullyConnectedDNN2
from Models.DNN.FullyConnectedDNN_class2_1 import FullyConnectedDNN3

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

def set_random_seed(seed):
    print('set seed')
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (CUDA)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoid non-deterministic algorithms
    torch.backends.cuda.matmul.allow_tf32 = False  # Ensure precise floating-point calculations
    torch.backends.cudnn.allow_tf32 = False

# Call this before model creation to ensure reproducibility
set_random_seed(42)  # Use any seed of your choice

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

def DNN_function(name, df, dfs_path):

    # Step 2: Split into features and target
    target_column = 'PKI'
    X = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
    y = df[target_column] #series

    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # # Step 4: Bin target values to handle imbalance
    bins = 5
    binned_y, bin_edges = pd.cut(y, bins=bins, labels=False, retbins=True, include_lowest=True)
    grouped_df = df.groupby('mol_id').first()

    dir_path = pv.base_path_ / 'code' / 'DNN' / 'train_val_loss' / 'test'
    dir_path.mkdir(parents=True, exist_ok=True)

    bin_counts = np.bincount(binned_y)  # Count samples per bin
    print(bin_counts)

    grouped_binned_y, bin_edges = pd.cut(grouped_df[target_column], bins=bins, labels=False, retbins=True, include_lowest=True)
    unique_mol_ids = grouped_df.index

    fold_results = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }
    # Initialize variables
    outer_folds = 10
    custom_outer_splits = []
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,shuffle=True, random_state=10)

    hyperparameter_grid = [
        {"learning_rate": 0.001, "hidden_layers": [64, 32, 16],"dropout_rate": 0.1}, #increase dropout rate to 0.2 as well because small dataset
        # {"learning_rate": 0.001, "hidden_layers": [64, 32],"dropout_rate": 0.1},
        # {"learning_rate": 0.001, "hidden_layers": [128, 64, 32],"dropout_rate": 0.1},
        # {"learning_rate": 0.001, "hidden_layers": [128, 64],"dropout_rate": 0.1},
        # {"learning_rate": 0.001, "hidden_layers": [64, 32, 16],"dropout_rate": 0.1},
        # {"learning_rate": 0.001, "hidden_layers": [64, 32],"dropout_rate": 0.1},
        # {"learning_rate": 0.001, "hidden_layers": [128, 64, 32],"dropout_rate": 0.1},
        # {"learning_rate": 0.001, "hidden_layers": [128, 64],"dropout_rate": 0.1},
        # {"learning_rate": 0.0005, "hidden_layers": [256, 128, 64],"dropout_rate": 0.1},
        # {"learning_rate": 0.0005, "hidden_layers": [256, 128],"dropout_rate": 0.1},
    ]
    fold_results = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }

    # Store losses and predictions across folds
    mol_ids = []
    all_train_outer_losses = []
    all_val_outer_losses = []
    all_best_params_outer = []
    all_predictions = []
    all_true_values = []

    all_outer_indices = list(outer_cv.split(X=grouped_df, y=grouped_binned_y, groups=unique_mol_ids))
    print(all_outer_indices[0][1])
    for outer_fold_number, (train_idx, test_idx) in enumerate(outer_cv.split(X=grouped_df, y=grouped_binned_y, groups=unique_mol_ids)):
        print('outerfold: ', outer_fold_number)
        validation_outer_fold = ((outer_fold_number+1) % outer_folds) #remainder of numerator and denominator, gives index of another outer_split
        validation_outer_idx = all_outer_indices[validation_outer_fold][1]

        # map train_idx to the actual mol ids (so back to the range of 1 to 615 (for JAK1))
        train_mol_ids = set(grouped_df.iloc[train_idx].index)  # Train molecule IDs
        validation_outer_mol_ids = set(grouped_df.iloc[validation_outer_idx].index)
        test_mol_ids = set(grouped_df.iloc[test_idx].index)  # Test molecule IDs
        train_mol_ids = train_mol_ids - validation_outer_mol_ids

        train_idx_all = df[df['mol_id'].isin(train_mol_ids)].index
        validation_outer_idx_all = df[df['mol_id'].isin(validation_outer_mol_ids)].index
        test_idx_all = df[df['mol_id'].isin(test_mol_ids)].index

        custom_outer_splits.append((train_idx_all, validation_outer_idx_all, test_idx_all))

        X_train, X_validation_outer ,X_test = X.loc[train_idx_all], X.loc[validation_outer_idx_all], X.loc[test_idx_all]
        y_train, y_validation_outer, y_test = y.loc[train_idx_all], y.loc[validation_outer_idx_all], y.loc[test_idx_all]
       
        train_all_idxs_molids = df.loc[X_train.index, 'mol_id']  # Train set molecule IDs (series with index train_full_idxs and all molids duplicate times) 
        validation_outer_all_idxs_molids = df.loc[X_validation_outer.index, 'mol_id']
        test_all_idxs_molids = df.loc[X_test.index, 'mol_id']  # Test set molecule IDs

        binned_y_train = binned_y.loc[X_train.index] #series with idx of all molecules/conformations and the bin it belongs to.
        binned_y_validation_outer = binned_y.loc[X_validation_outer.index]
        binned_y_test = binned_y.loc[X_test.index]

        inner_folds = 5
        custom_inner_splits = []
        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=10)

        grouped_X_train = X_train.groupby(train_all_idxs_molids).first()
        grouped_binned_y_train = binned_y_train.groupby(train_all_idxs_molids).first() #same as grouped_binned_y but now only the train molecules

        hyperparameter_losses = [[] for _ in hyperparameter_grid]
        
        # Dictionary to store training and validation loss for each hyperparameter set
        hyperparameter_loss_curves = {param_idx: {"train_losses": [], "val_losses": []} for param_idx in range(len(hyperparameter_grid))}
        for inner_fold, (train_inner_idx, validation_inner_idx) in enumerate(inner_cv.split(grouped_X_train, grouped_binned_y_train, groups=grouped_X_train.index)):
            inner_col = f"outer_{outer_fold_number}_inner_{inner_fold}"
            print(inner_col)
            train_inner_mol_ids = grouped_X_train.iloc[train_inner_idx].index
            val_mol_ids = grouped_X_train.iloc[validation_inner_idx].index

            train_inner_idx_full = df[df['mol_id'].isin(train_inner_mol_ids)].index.intersection(X_train.index)
            val_idx_full = df[df['mol_id'].isin(val_mol_ids)].index.intersection(X_train.index)

            custom_inner_splits.append((train_inner_idx_full, val_idx_full))

            X_train_inner, X_validation_inner = X_train.loc[train_inner_idx_full], X_train.loc[val_idx_full]
            y_train_inner, y_validation_inner = y_train.loc[train_inner_idx_full], y_train.loc[val_idx_full]
            
            
            train_inner_dataset = TensorDataset(torch.tensor(X_train_inner.values, dtype=torch.float32),
                                                torch.tensor(y_train_inner.values.reshape(-1, 1), dtype=torch.float32))
            val_inner_dataset = TensorDataset(torch.tensor(X_validation_inner.values, dtype=torch.float32),
                                        torch.tensor(y_validation_inner.values.reshape(-1, 1), dtype=torch.float32))
            train_inner_loader = DataLoader(train_inner_dataset, batch_size=64, shuffle=True)
            val_inner_loader = DataLoader(val_inner_dataset, batch_size=64, shuffle=False)
            # for inputs, targets in train_inner_loader:
            #     print(f"Inputs: {inputs.shape}")  # Shape of input tensor
            #     print(f"Targets: {targets.shape}")  # Shape of target tensor
            #     print(f"Inputs (first sample): {inputs[0]}")  # Inspect first input sample
            #     print(f"Targets (first sample): {targets[0]}")  # Inspect first target sample
                
            # Hyperparameter tuning
            best_params = None
            best_val_loss = float('inf')
            # set_random_seed(42)
            for param_idx, params in enumerate(hyperparameter_grid):

                model1 = FullyConnectedDNN2(input_size=X_train_inner.shape[1], hidden_layers=params["hidden_layers"],dropout_rate=params["dropout_rate"])
                optimizer1 = torch.optim.Adam(model1.parameters(), lr=params["learning_rate"])
                criterion1 = torch.nn.MSELoss()
                
                # Train with validation to tune hyperparameters
                train_inner_losses, val_inner_losses = model1.train_with_validation(
                    train_inner_loader, val_inner_loader, num_epochs=500, optimizer=optimizer1,
                    criterion=criterion1, device="cuda" if torch.cuda.is_available() else "cpu", patience=50
                )
            
                # Store validation loss for the current hyperparameter set
                hyperparameter_losses[param_idx].append(min(val_inner_losses))

                # Append losses for each inner fold (so we get all 5 curves)
                hyperparameter_loss_curves[param_idx]["train_losses"].append(train_inner_losses)
                hyperparameter_loss_curves[param_idx]["val_losses"].append(val_inner_losses)
        # After finishing all inner folds, compute the average validation loss for each hyperparameter set
        avg_hyperparameter_losses = [sum(losses) / len(losses) for losses in hyperparameter_losses]

        # Find the hyperparameter set with the lowest average validation loss
        best_hyperparameter_idx = min(range(len(avg_hyperparameter_losses)), key=avg_hyperparameter_losses.__getitem__)
        # Retrieve the loss curves for the best hyperparameter set
        best_train_losses = hyperparameter_loss_curves[best_hyperparameter_idx]["train_losses"]
        best_val_losses = hyperparameter_loss_curves[best_hyperparameter_idx]["val_losses"]
        best_params = hyperparameter_grid[best_hyperparameter_idx]
        # Prepare dictionary for storing
        loss_data = {
            "train_losses": best_train_losses,
            "val_losses": best_val_losses,
            "best_hyperparameters": best_params  # Include the best hyperparameters
        }

        # Save to JSON file
        save_path = dfs_path / pv.Inner_train_Val_losses
        save_path.mkdir(parents = True, exist_ok=True)
        with open(save_path / f"{name}_loss_curves_outer_{outer_fold_number}.json", "w") as f:
            json.dump(loss_data, f, indent=4)
        

        #Before Final Model Training
        set_random_seed(42)

        # Train final model on 80% training data with best hyperparameters
        train_outer_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                            torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32))
        validation_outer_dataset = TensorDataset(torch.tensor(X_validation_outer.values, dtype=torch.float32),
                                    torch.tensor(y_validation_outer.values.reshape(-1, 1), dtype=torch.float32))
        train_outer_loader = DataLoader(train_outer_dataset, batch_size=32, shuffle=True)
        val_outer_loader = DataLoader(validation_outer_dataset, batch_size=32, shuffle=False)
        
        epochs = 1000
        results = {}
        
        final_model = FullyConnectedDNN2(input_size=X_train.shape[1], hidden_layers=best_params["hidden_layers"],dropout_rate=best_params["dropout_rate"])

        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])

        criterion = torch.nn.MSELoss()

        train_outer_losses, val_outer_losses = final_model.train_with_validation(
                train_outer_loader, val_outer_loader, num_epochs=epochs, optimizer=optimizer,
                criterion=criterion, device="cuda" if torch.cuda.is_available() else "cpu", patience=100
            )

        # Store the losses and hyperparameters for this outer fold
        all_train_outer_losses.append(train_outer_losses)
        all_val_outer_losses.append(val_outer_losses)
        all_best_params_outer.append(best_params)
        # Evaluate on test set
        final_model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                    torch.tensor(y_test.values, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        fold_predictions = []
        fold_true_values = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = final_model(inputs)
                fold_predictions.extend(outputs.cpu().numpy().flatten())
                fold_true_values.extend(targets.numpy())
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_true_values.extend(targets.numpy())
            # # Assuming `df.loc[test_idx_all, 'mol_id']` provides the mol_ids for the fold
            mol_ids.extend(df.loc[test_idx_all, 'mol_id'].values)  # Extract mol_id corresponding to test set
            
        # Calculate metrics for this epoch
        r2_value = r2_score(fold_true_values, fold_predictions)
        mse_value = mean_squared_error(fold_true_values, fold_predictions)
        mae_value = mean_absolute_error(fold_true_values, fold_predictions)
        print(f"Fold {outer_fold_number}; MSE: {mse_value:.4f}, MAE: {mae_value:.4f}, Test R² score: {r2_value:.4f}")

        fold_results['R2'].append(r2_value)
        fold_results['MSE'].append(mse_value)
        fold_results['MAE'].append(mae_value)

    # Prepare the data to be saved in JSON format
    outer_fold_data = {
        "train_losses": all_train_outer_losses,
        "val_losses": all_val_outer_losses,
        "best_hyperparameters": all_best_params_outer
    }
    # Save to JSON file
    save_path = dfs_path / pv.Outer_train_Val_losses
    save_path.mkdir(parents = True, exist_ok=True)

    with open(save_path / f'{name}_loss_curves_all_outer_folds.json', 'w') as f:
        json.dump(outer_fold_data, f)

    results_all = {}
    for metric, fold_list in fold_results.items():
        fold_array = np.array(fold_list, dtype=np.float64)    
        results = {
            "mol_id": name,
            "mean_test_score": fold_array.mean(),
            "std_test_score": fold_array.std(),
            "params": best_params,
            **{f"split{split_idx}_test_score": fold_array[split_idx] for split_idx in range(len(fold_array))},  # Loop to add splits
        }
        results_all[metric] = results
    
    # Create DataFrame with mol_id, True_pKi, and Predicted_pKi
    results_df = pd.DataFrame({
        'mol_id': mol_ids,
        'True_pKi': all_true_values,
        'Predicted_pKi': all_predictions
    })
    results_df = results_df.sort_values(by='mol_id', ascending=True)
    print('done')
    print('done')
    # # Flatten the lists
    # all_predictions_flat = list(itertools.chain(*all_predictions))
    # all_true_values_flat = list(itertools.chain(*all_true_values))
    # mol_ids = df["mol_id"].tolist()
    # print(mol_ids)
    # results_df = pd.DataFrame({
    #     "mol_id": mol_ids,  # Molecule IDs
    #     "true_value": all_true_values_flat,  # True values for each conformation
    #     "prediction": all_predictions_flat   # Predictions for each conformation
    # })
    # averaged_results = results_df.groupby("mol_id").agg(
    #     avg_true_value=("true_value", "mean"),
    #     avg_prediction=("prediction", "mean")
    # ).reset_index()

    # # If you want the averaged true values and predictions as lists
    # avg_true_values = averaged_results["avg_true_value"].tolist()
    # avg_predictions = averaged_results["avg_prediction"].tolist()
    # mse_avg = mean_squared_error(avg_true_values, avg_predictions)
    # r2_avg = r2_score(avg_true_values, avg_predictions)
    # # Create the results dictionary with flattened values
    # results_dict = {
    #     "predictions": all_predictions_flat,
    #     "true_values": all_true_values_flat
    # }
    # # Save all_train_inner_losses
    # newfolder = dfs_path / pv.Modelresults_folder_ / 'train_loss_plots'
    
    # newfolder.mkdir(parents=True, exist_ok=True)
    # train_inner_losses_df = pd.DataFrame.from_dict(all_train_inner_losses, orient="index").T
    # train_inner_losses_df.to_csv(newfolder / f"train_inner_losses_{name}.csv", index=False)

    # # Save all_val_inner_losses
    # val_inner_losses_df = pd.DataFrame.from_dict(all_val_inner_losses, orient="index").T
    # val_inner_losses_df.to_csv(newfolder / f"val_inner_losses.csv_{name}", index=False)

    # # Save all_train_outer_losses
    # train_outer_losses_df = pd.DataFrame.from_dict(all_train_outer_losses, orient="index").T
    # train_outer_losses_df.to_csv(newfolder / f"train_outer_losses_{name}.csv", index=False)

    # # Save all_val_outer_losses
    # val_outer_losses_df = pd.DataFrame.from_dict(all_val_outer_losses, orient="index").T
    # val_outer_losses_df.to_csv(newfolder / f"val_outer_losses_{name}.csv", index=False)


    # results_df = pd.DataFrame(results_dict)
    # # Convert the dictionary into a DataFrame
    # anotherfolder = dfs_path / pv.Modelresults_folder_ / 'pred_true_plots'
    # anotherfolder.mkdir(parents=True, exist_ok=True)
    # results_df.to_csv(dfs_path / pv.Modelresults_folder_ / 'pred_true_plots' / f"model_pred_vs_true_{name}.csv", index=False)

    # # Print overall metrics
    # print(f"Average MSE: {np.mean(mse_scores):.4f}")
    # print(f"Average R²: {np.mean(r2_scores):.4f}")

    # # fold_results.append(r2_score)

    # data = pd.Series(fold_results)

    # fold_results = {
    #     "mol_id": name,
    #     "mean_test_score": pd.Series(r2_scores).mean(),
    #     "std_test_score": pd.Series(r2_scores).std(),
    #     "params": [1,2],
    #     **{f"split{split_idx}_test_score": r2_scores[split_idx] for split_idx in range(outer_folds)},  # Loop to add splits
    # }

    # create_train_inner_loss_plots(name, all_params, all_train_inner_lossesl, all_val_inner_lossesl, dfs_path)
    # create_train_loss_plots(name, all_params, all_train_outer_lossesl, all_val_outer_lossesl, dfs_path)
    # create_pred_true_plots(name, all_params, all_true_values, all_predictions, dfs_path)

    return fold_results


def create_train_inner_loss_plots(name, all_params, all_train_losses, all_val_losses, dfs_path):

    # Plot Training and Validation Loss Curves
    model_ = 'DNN'
    Modelresults_path = dfs_path / f'ModelResults_{model_}'
    path = Modelresults_path / 'train_loss_plots'
    path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
        plt.plot(train_loss, label=f"Fold {fold + 1} - Train Loss")
        plt.plot(val_loss, label=f"Fold {fold + 1} - Validation Loss", linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if name == '0ns':
        plt.title(f"minimized conformation - Training and Validation Loss Across Folds")
    else:
        plt.title(f"{name} - Training and Validation Loss Across Folds")
    plt.legend()
    plt.savefig(path / f'{name}_traininnerloss_all_folds.png')
    plt.close()

    for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
        print(all_params)
        print(fold)
        print(all_params[fold])
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label=f"Train Loss")
        plt.plot(val_loss, label=f"Validation Loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if name == '0ns':
            plt.title(f"minimized conformation -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
        else:
            plt.title(f"{name} -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
        plt.legend()
        plt.savefig(path / f'{name}_traininnerloss_fold_{fold + 1}.png')
        plt.close()
    return

def create_train_loss_plots(name, all_params, all_train_losses, all_val_losses, dfs_path):

    # Plot Training and Validation Loss Curves
    model_ = 'DNN'
    Modelresults_path = dfs_path / f'ModelResults_{model_}'
    path = Modelresults_path / 'train_loss_plots'
    path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
        plt.plot(train_loss, label=f"Fold {fold + 1} - Train Loss")
        plt.plot(val_loss, label=f"Fold {fold + 1} - Validation Loss", linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if name == '0ns':
        plt.title(f"minimized conformation - Training and Validation Loss Across Folds")
    else:
        plt.title(f"{name} - Training and Validation Loss Across Folds")
    plt.legend()
    plt.savefig(path / f'{name}_trainloss_all_folds.png')
    plt.close()

    for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
        print(all_params)
        print(fold)
        print(all_params[fold])
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label=f"Train Loss")
        plt.plot(val_loss, label=f"Validation Loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if name == '0ns':
            plt.title(f"minimized conformation -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
        else:
            plt.title(f"{name} -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
        plt.legend()
        plt.savefig(path / f'{name}_trainloss_fold_{fold + 1}.png')
        plt.close()
    return

def create_pred_true_plots(name, params, all_true_values, all_predictions, dfs_path):
    model_ = 'DNN'
    Modelresults_path = dfs_path / f'ModelResults_{model_}'
    path = Modelresults_path / 'pred_true_plots'
    path.mkdir(parents=True, exist_ok=True)
    # Plot Predicted vs True pKi Values
    plt.figure(figsize=(8, 8))
    plt.scatter(all_true_values, all_predictions, alpha=0.6, edgecolor='k')
    plt.plot([min(all_true_values), max(all_true_values)],
             [min(all_true_values), max(all_true_values)], color="red", linestyle="--", label="Ideal Fit")
    plt.xlabel("True pKi")
    plt.ylabel("Predicted pKi")
    if name == '0ns':
        plt.title(f"minimized conformation - Predicted vs True pKi Across All Folds")
    else:
        plt.title(f"{name} - Predicted vs True pKi Across All Folds")
    plt.legend()
    plt.savefig(path / f'{name}_pred_true.png')
    plt.close()
    return

def main(dfs_path = pv.dfs_descriptors_only_path_):
    print(dfs_path)
    model_ = 'DNN'
    Modelresults_path = dfs_path / f'ModelResults_{model_}' #CHECK: if 'RF' is in pv or something else perhaps
    Modelresults_path.mkdir(parents=True, exist_ok=True)

    # dfs_in_dic = csv_to_dictionary.csvfiles_to_dic(dfs_path, exclude_files=['conformations_1000.csv','conformations_1000_molid.csv','conformations_500.csv','conformations_200.csv','conformations_100.csv','conformations_50.csv','initial_dataframe.csv','initial_dataframes_best.csv','MD_output.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_include(dfs_path, include_files=['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_include(dfs_path, include_files=['0ns.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    # dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_include(dfs_path, include_files=['conformations_10c.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(dfs_in_dic.keys())
    first_four_keys = list(dfs_in_dic.keys())[0:11] #+ [list(dfs_in_dic.keys())[14]] + [list(dfs_in_dic.keys())[16]] + [list(dfs_in_dic.keys())[17]] #+ list(dfs_in_dic.keys())[16] + list(dfs_in_dic.keys())[17]
    print(first_four_keys)
    filtered_dict = {key: dfs_in_dic[key] for key in first_four_keys}
    print(filtered_dict.keys())
    ModelResults_ = []

    csv_filename = f'results_Ko10_Ki5_r2_{pv.DESCRIPTOR}.csv'
    csv_filename_temp = f'results_Ko10_Ki5_r2_{pv.DESCRIPTOR}_temp.csv' #if i break it early i still get some results
    
    for name, df in filtered_dict.items():
        print(name)
        fold_results = DNN_function(name, df, dfs_path)
        ModelResults_.append(fold_results)
        pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename_temp, index=False)
    
    pd.DataFrame(ModelResults_).to_csv(Modelresults_path / csv_filename, index=False)

    return

if __name__ == "__main__":
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)

    # main(pv.dfs_descriptors_only_path_)
    # main(pv.dfs_reduced_and_MD_path_)

    main(pv.dfs_reduced_path_)
    # main(pv.dfs_MD_only_path_)
    # for protein_ in pv.DatasetProtein:
    #     pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein)
    #     # Construct paths dynamically for each protein
    #     main(pv.dfs_descriptors_only_path_)
    #     main(pv.dfs_reduced_path_)
    #     main(pv.dfs_reduced_and_MD_path_)
    #     main(pv.dfs_MD_only_path_)

