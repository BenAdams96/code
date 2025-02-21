from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
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

from Models.LSTM.LSTM_class import LSTM


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

def get_averaged_prediction(model, loader, device):
    model.eval()  # Set model to evaluation mode
    predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # For each molecule, process each of its conformations (assuming 10 conformations)
            molecule_predictions = []
            for conformation in inputs:
                output = model(conformation.unsqueeze(0))  # Unsqueeze to make it a batch of 1
                molecule_predictions.append(output.item())
            
            # Average the predictions for the molecule
            averaged_prediction = sum(molecule_predictions) / len(molecule_predictions)
            
            # Store averaged predictions
            predictions.append(averaged_prediction)
            true_values.append(targets.item())  # Assuming each batch has 1 true value

    return predictions, true_values

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

def LSTM_function(name, df, dfs_path):


    # Step 2: Split into features and target
    target_column = 'PKI'
    if 'conformations (ns)' in df.columns:
        df = df.sort_values(by=['mol_id', 'conformations (ns)']).reset_index(drop=True)
    else:
        df = df.sort_values(by=['mol_id']).reset_index(drop=True)

    X = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], axis=1, errors='ignore')
    y = df[target_column] #series
    print(y.shape)

    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame

    # # Step 4: Bin target values to handle imbalance
    bins = 5
    binned_y, bin_edges = pd.cut(y, bins=bins, labels=False, retbins=True, include_lowest=True)
    grouped_df = df.groupby('mol_id').first()

    bin_counts = np.bincount(binned_y)  # Count samples per bin
    print(bin_counts)

    grouped_binned_y, bin_edges = pd.cut(grouped_df[target_column], bins=bins, labels=False, retbins=True, include_lowest=True)
    unique_mol_ids = grouped_df.index

    num_of_conformations = df.groupby('mol_id').size().iloc[0]

    # Initialize variables
    outer_folds = 10
    custom_outer_splits = []
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,shuffle=True, random_state=10)

    fold_results = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }

    hyperparameter_grid = [
        {"hidden_size": 32,  # Number of LSTM units per layer
                    "num_layers": 2,  # Number of stacked LSTM layers
                    "dropout": 0.2,  # Dropout to prevent overfitting
                    "learning_rate": 1e-3,  # Learning rate for Adam/SGD
                    "weight_decay": 1e-5,  # L2 regularization
                    },]
    # Store losses and predictions across folds
    mol_ids = []
    mol_ids_fold = []
    all_train_outer_losses = []
    all_val_outer_losses = []
    all_best_params_outer = []
    all_predictions = []
    all_true_values = []

    all_outer_indices = list(outer_cv.split(X=grouped_df, y=grouped_binned_y, groups=unique_mol_ids)) #mol indexes
    print(all_outer_indices[0][1])
    # print(all_outer_indices) #6 7 16 28 / 8 19 20 29 42
    # print(all_outer_indices)
    for outer_fold_number, (train_idx, test_idx) in enumerate(outer_cv.split(X=grouped_df, y=grouped_binned_y, groups=unique_mol_ids)):
        print('outerfold: ', outer_fold_number)
        validation_outer_fold = ((outer_fold_number+1) % outer_folds) #remainder of numerator and denominator, gives index of another outer_split
        validation_outer_idx = all_outer_indices[validation_outer_fold][1]

        # map train_idx to the actual mol ids (so back to the range of 1 to 615 (for JAK1))
        train_mol_ids = set(grouped_df.iloc[train_idx].index)  # Train molecule IDs
        validation_outer_mol_ids = set(grouped_df.iloc[validation_outer_idx].index)
        test_mol_ids = set(grouped_df.iloc[test_idx].index)  # Test molecule IDs
        train_mol_ids = train_mol_ids - validation_outer_mol_ids
        print(test_mol_ids)
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
        # Dictionary to store training and validation loss for each hyperparameter set
        # hyperparameter_loss_curves = {param_idx: {"train_losses": [], "val_losses": []} for param_idx in range(len(hyperparameter_grid))}
        for inner_fold, (train_inner_idx, validation_inner_idx) in enumerate(inner_cv.split(grouped_X_train, grouped_binned_y_train, groups=grouped_X_train.index)):
            inner_col = f"outer_{outer_fold_number}_inner_{inner_fold}"
            print(inner_col)

            train_inner_mol_ids = grouped_X_train.iloc[train_inner_idx].index
            val_mol_ids = grouped_X_train.iloc[validation_inner_idx].index
            print(val_mol_ids)

            train_inner_idx_full = df[df['mol_id'].isin(train_inner_mol_ids)].index.intersection(X_train.index)
            val_idx_full = df[df['mol_id'].isin(val_mol_ids)].index.intersection(X_train.index)

            custom_inner_splits.append((train_inner_idx_full, val_idx_full))

            X_train_inner, X_validation_inner = X_train.loc[train_inner_idx_full], X_train.loc[val_idx_full]
            y_train_inner, y_validation_inner = y_train.loc[train_inner_idx_full], y_train.loc[val_idx_full]
            print(X_validation_inner)
            X_train_inner_grouped = X_train_inner.values.reshape(-1, num_of_conformations, X_train_inner.shape[1])  # Shape: (num_molecules, num_of_conformations, num_features)
            y_train_inner_grouped = y_train_inner.values.reshape(-1, num_of_conformations)  # Shape: (num_molecules, 10)
            X_validation_inner_grouped = X_validation_inner.values.reshape(-1, num_of_conformations, X_validation_inner.shape[1])  # Shape: (num_molecules, 10, num_features)
            y_validation_inner_grouped = y_validation_inner.values.reshape(-1, num_of_conformations)

            train_inner_dataset = TensorDataset(torch.tensor(X_train_inner_grouped, dtype=torch.float32),
                                                torch.tensor(y_train_inner_grouped, dtype=torch.float32))
            val_inner_dataset = TensorDataset(torch.tensor(X_validation_inner_grouped, dtype=torch.float32),
                                   torch.tensor(y_validation_inner_grouped, dtype=torch.float32))
            train_inner_loader = DataLoader(train_inner_dataset, batch_size=128, shuffle=True) #128 molecules at once independent of c conformations
            val_inner_loader = DataLoader(val_inner_dataset, batch_size=128, shuffle=False)

            # Hyperparameter tuning
            best_params = None
            best_val_loss = float('inf')
            # set_random_seed(42)
            
            for param_idx, params in enumerate(hyperparameter_grid): #pv.Model_deep.hyperparameter_grid
                print(params)
                model = LSTM(
                    input_size=X_train_inner.shape[1], 
                    hidden_size=params["hidden_size"],  # Fix: Use "hidden_size" instead of "hidden_layers"
                    num_layers=params["num_layers"],  # Fix: Add "num_layers" parameter
                    output_size=1,  # Assuming pKi prediction is a single value
                    dropout=params["dropout"]  # Fix: Use "dropout" instead of "dropout_rate"
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
                criterion = torch.nn.MSELoss()

                train_inner_losses, val_inner_losses = model.train_with_early_stopping(
                    train_loader=train_inner_loader, 
                    val_loader=val_inner_loader, 
                    num_epochs=500,  # You define this
                    optimizer=optimizer,
                    criterion=criterion,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    patience=50  # Early stopping patience
                )

                # Store validation loss for the current hyperparameter set
                hyperparameter_losses[param_idx].append(min(val_inner_losses))

                # Append losses for each inner fold (so we get all 5 curves)
                hyperparameter_loss_curves[param_idx]["train_losses"].append(train_inner_losses)
                hyperparameter_loss_curves[param_idx]["val_losses"].append(val_inner_losses)
            print('done')
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
        save_path = dfs_path / pv.Inner_train_Val_losses / name
        save_path.mkdir(parents = True, exist_ok=True)
        with open(save_path / f"{name}_5loss_curves_of_best_inner_of_outer_{outer_fold_number}.json", "w") as f:
            json.dump(loss_data, f, indent=4)
    
        #Before Final Model Training
        set_random_seed(41) #does it change the behaviour of results? yes! good fold2 results
        # print(X_train_inner.shape[1])
        print(num_of_conformations)
        X_train_outer_grouped = X_train.values.reshape(-1, num_of_conformations, X_train_inner.shape[1])
        y_train_outer_grouped = y_train.values.reshape(-1, num_of_conformations)  # Shape: (num_molecules, 10)
        X_validation_outer_grouped = X_validation_outer.values.reshape(-1, num_of_conformations, X_validation_inner.shape[1])  # Shape: (num_molecules, 10, num_features)
        y_validation_outer_grouped = y_validation_outer.values.reshape(-1, num_of_conformations)  # Shape: (num_molecules, 10)
        # print(X_train_outer_grouped)
        # Train final model on 80% training data with best hyperparameters
        train_outer_dataset = TensorDataset(torch.tensor(X_train_outer_grouped, dtype=torch.float32),
                                            torch.tensor(y_train_outer_grouped, dtype=torch.float32))
        validation_outer_dataset = TensorDataset(torch.tensor(X_validation_outer_grouped, dtype=torch.float32),
                                    torch.tensor(y_validation_outer_grouped, dtype=torch.float32))
        train_outer_loader = DataLoader(train_outer_dataset, batch_size=64, shuffle=True)
        val_outer_loader = DataLoader(validation_outer_dataset, batch_size=64, shuffle=False)
        # print(train_outer_dataset)
        epochs = 2000
        results = {}
        set_random_seed(42) #setting random seed here at 42 is same results
        final_model = LSTM(
                input_size=X_train_inner.shape[1], 
                hidden_size=params["hidden_size"],  # Fix: Use "hidden_size" instead of "hidden_layers"
                num_layers=params["num_layers"],  # Fix: Add "num_layers" parameter
                output_size=1,  # Assuming pKi prediction is a single value
                dropout=params["dropout"]  # Fix: Use "dropout" instead of "dropout_rate"
                )
        optimizer = torch.optim.Adam(final_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        criterion = torch.nn.MSELoss()

        train_outer_losses, val_outer_losses = final_model.train_with_early_stopping(
            train_loader=train_outer_loader,
            val_loader=val_outer_loader,
            num_epochs=2000,  # You define this
            optimizer=optimizer,
            criterion=criterion,
            device="cuda" if torch.cuda.is_available() else "cpu",
            patience=150  # Early stopping patience
        )
        all_train_outer_losses.append(train_outer_losses)
        all_val_outer_losses.append(val_outer_losses)
        all_best_params_outer.append(best_params)

        final_model.eval()
        X_test_grouped = X_test.values.reshape(-1, num_of_conformations, X_test.shape[1])  # Shape: (num_molecules, num_of_conformations, num_features)
        y_test_grouped = y_test.values.reshape(-1, num_of_conformations).mean(axis=1, keepdims=True)
        test_dataset = TensorDataset(torch.tensor(X_test_grouped, dtype=torch.float32),
                                    torch.tensor(y_test_grouped, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # print(X_test_grouped)
        fold_predictions = []
        fold_true_values = []
        # set_random_seed(42) #does it change the behaviour of results? no
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = final_model(inputs)

                fold_predictions.extend(outputs.cpu().numpy().flatten())
                fold_true_values.extend(targets.numpy())
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_true_values.extend(targets.numpy())
            mol_ids.extend(df.loc[test_idx_all, 'mol_id'].unique())
        # Calculate metrics for this epoch
        r2_value = r2_score(fold_true_values, fold_predictions)
        mse_value = mean_squared_error(fold_true_values, fold_predictions)
        mae_value = mean_absolute_error(fold_true_values, fold_predictions)
        print(f"Fold {outer_fold_number}; MSE: {mse_value:.4f}, MAE: {mae_value:.4f}, Test R² score: {r2_value:.4f}")

        fold_results['R2'].append(r2_value)
        fold_results['MSE'].append(mse_value)
        fold_results['MAE'].append(mae_value)

    mean_scores = {}
    #get once the the mean scores using all the predictions
    r2_value = r2_score(all_true_values, all_predictions)
    mse_value = mean_squared_error(all_true_values, all_predictions)
    mae_value = mean_absolute_error(all_true_values, all_predictions)
    mean_scores['R2'] = r2_value
    mean_scores['MSE'] = mse_value
    mean_scores['MAE'] = mae_value

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
        json.dump(outer_fold_data, f, indent=4)

    results_all_metrics = {}
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
        results_all_metrics[metric] = results
    
    # Create DataFrame with mol_id, True_pKi, and Predicted_pKi
    df_true_predicted = pd.DataFrame({
        'mol_id': mol_ids,
        'True_pKi': all_true_values,
        'Predicted_pKi': all_predictions
    })
    df_true_predicted = df_true_predicted.sort_values(by='mol_id', ascending=True)
    save_path = dfs_path / pv.true_predicted
    save_path.mkdir(parents = True, exist_ok=True)
    df_true_predicted.to_csv(save_path / f'{name}_true_predicted.csv', index=False)
    print(f'done with {name}')
    
    return results_all_metrics


# def create_train_inner_loss_plots(name, all_params, all_train_losses, all_val_losses, dfs_path):

#     # Plot Training and Validation Loss Curves
#     model_ = 'DNN'
#     Modelresults_path = dfs_path / f'ModelResults_{model_}'
#     path = Modelresults_path / 'train_loss_plots'
#     path.mkdir(parents=True, exist_ok=True)
#     plt.figure(figsize=(10, 6))
#     for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
#         plt.plot(train_loss, label=f"Fold {fold + 1} - Train Loss")
#         plt.plot(val_loss, label=f"Fold {fold + 1} - Validation Loss", linestyle="--")

#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     if name == '0ns':
#         plt.title(f"minimized conformation - Training and Validation Loss Across Folds")
#     else:
#         plt.title(f"{name} - Training and Validation Loss Across Folds")
#     plt.legend()
#     plt.savefig(path / f'{name}_traininnerloss_all_folds.png')
#     plt.close()

#     for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
#         print(all_params)
#         print(fold)
#         print(all_params[fold])
#         plt.figure(figsize=(10, 6))
#         plt.plot(train_loss, label=f"Train Loss")
#         plt.plot(val_loss, label=f"Validation Loss", linestyle="--")
#         plt.xlabel("Epochs")
#         plt.ylabel("Loss")
#         if name == '0ns':
#             plt.title(f"minimized conformation -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
#         else:
#             plt.title(f"{name} -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
#         plt.legend()
#         plt.savefig(path / f'{name}_traininnerloss_fold_{fold + 1}.png')
#         plt.close()
#     return

# def create_train_loss_plots(name, all_params, all_train_losses, all_val_losses, dfs_path):

#     # Plot Training and Validation Loss Curves
#     model_ = 'DNN'
#     Modelresults_path = dfs_path / f'ModelResults_{model_}'
#     path = Modelresults_path / 'train_loss_plots'
#     path.mkdir(parents=True, exist_ok=True)
#     plt.figure(figsize=(10, 6))
#     for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
#         plt.plot(train_loss, label=f"Fold {fold + 1} - Train Loss")
#         plt.plot(val_loss, label=f"Fold {fold + 1} - Validation Loss", linestyle="--")

#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     if name == '0ns':
#         plt.title(f"minimized conformation - Training and Validation Loss Across Folds")
#     else:
#         plt.title(f"{name} - Training and Validation Loss Across Folds")
#     plt.legend()
#     plt.savefig(path / f'{name}_trainloss_all_folds.png')
#     plt.close()

#     for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
#         print(all_params)
#         print(fold)
#         print(all_params[fold])
#         plt.figure(figsize=(10, 6))
#         plt.plot(train_loss, label=f"Train Loss")
#         plt.plot(val_loss, label=f"Validation Loss", linestyle="--")
#         plt.xlabel("Epochs")
#         plt.ylabel("Loss")
#         if name == '0ns':
#             plt.title(f"minimized conformation -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
#         else:
#             plt.title(f"{name} -Fold {fold + 1} - Training and Validation Loss - {all_params[fold]}")
#         plt.legend()
#         plt.savefig(path / f'{name}_trainloss_fold_{fold + 1}.png')
#         plt.close()
#     return

# def create_pred_true_plots(name, params, all_true_values, all_predictions, dfs_path):
#     model_ = 'DNN'
#     Modelresults_path = dfs_path / f'ModelResults_{model_}'
#     path = Modelresults_path / 'pred_true_plots'
#     path.mkdir(parents=True, exist_ok=True)
#     # Plot Predicted vs True pKi Values
#     plt.figure(figsize=(8, 8))
#     plt.scatter(all_true_values, all_predictions, alpha=0.6, edgecolor='k')
#     plt.plot([min(all_true_values), max(all_true_values)],
#              [min(all_true_values), max(all_true_values)], color="red", linestyle="--", label="Ideal Fit")
#     plt.xlabel("True pKi")
#     plt.ylabel("Predicted pKi")
#     if name == '0ns':
#         plt.title(f"minimized conformation - Predicted vs True pKi Across All Folds")
#     else:
#         plt.title(f"{name} - Predicted vs True pKi Across All Folds")
#     plt.legend()
#     plt.savefig(path / f'{name}_pred_true.png')
#     plt.close()
#     return

def save_fold_results(results, metric, ModelResults, Modelresults_path):
    
    # Process each metric separately
    # Append the results for this metric
    ModelResults[metric].append(results)
    
    # Write the updated results to a temporary CSV file
    csv_filename_temp = f'ModelResults_{metric}_{pv.DESCRIPTOR}_temp.csv'
    pd.DataFrame(ModelResults[metric]).to_csv(Modelresults_path / csv_filename_temp, index=False)

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

def main(dfs_path = pv.dfs_descriptors_only_path_, include_files = []):
    print(dfs_path)
    
    pv.update_config(model_=Model_deep.LSTM)
    Modelresults_path = dfs_path / pv.Modelresults_folder_ #create Modelresults folder
    Modelresults_path.mkdir(parents=True, exist_ok=True)

    if not include_files:
        include_files = ['conformations_10.csv']
    
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_include(dfs_path, include_files=include_files) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    
    sorted_keys_list = csv_to_dictionary.get_sorted_columns(list(dfs_in_dic.keys())) #RDKIT first
    dfs_in_dic_sorted = {key: dfs_in_dic[key] for key in sorted_keys_list if key in dfs_in_dic} #order
    print(dfs_in_dic_sorted.keys())

    for name, df in dfs_in_dic_sorted.items():
        # Perform nested cross-validation for the current dataset
        results_all_metrics = LSTM_function(name, df, dfs_path)
        ModelResults = {'R2': [], 'MSE': [], 'MAE': []}
        for metric, results in results_all_metrics.items():
            save_fold_results(results, metric, ModelResults, Modelresults_path)
    return

if __name__ == "__main__":
    pv.update_config(model_=Model_deep.LSTM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)

    # main(pv.dfs_descriptors_only_path_)
    main(pv.dfs_reduced_path_)


