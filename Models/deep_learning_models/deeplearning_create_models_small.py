from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
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
from Models.DNN.FullyConnectedDNN_class_final import FullyConnectedDNN



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


def build_LSTM_model(train_loader, val_loader, input_size, nums_of_epochs, patience, params):
    model = LSTM(
        input_size=input_size, 
        hidden_size=params["hidden_size"],  # Fix: Use "hidden_size" instead of "hidden_layers"
        num_layers=params["num_layers"],  # Fix: Add "num_layers" parameter
        output_size=1,  # Assuming pKi prediction is a single value
        dropout=params["dropout"]  # Fix: Use "dropout" instead of "dropout_rate"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = torch.nn.MSELoss()
    train_inner_losses, val_inner_losses = model.train_with_early_stopping(
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=nums_of_epochs,  # You define this
        optimizer=optimizer,
        criterion=criterion,
        device="cuda" if torch.cuda.is_available() else "cpu",
        patience=patience  # Early stopping patience
    )
    return model, train_inner_losses, val_inner_losses

def build_DNN_model(train_loader, val_loader, input_size, num_of_conformations, nums_of_epochs, patience , params):
    model = FullyConnectedDNN(input_size=input_size, hidden_layers=params["hidden_layers"],dropout_rate=params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.MSELoss()
    
    # Train with validation to tune hyperparameters
    train_inner_losses, val_inner_losses = model.train_with_validation(
        train_loader, val_loader, num_epochs=nums_of_epochs, num_of_conformations=num_of_conformations, optimizer=optimizer,
        criterion=criterion, device="cuda" if torch.cuda.is_available() else "cpu", patience=patience
    )
    return model, train_inner_losses, val_inner_losses

def get_inner_dataloaders(train_inner_mol_ids, val_inner_mol_ids, X_train_outer, y_train_outer, num_of_conformations):
    
    train_inner_idx_full = X_train_outer[X_train_outer['mol_id'].isin(train_inner_mol_ids)].index
    val_inner_idx_full = X_train_outer[X_train_outer['mol_id'].isin(val_inner_mol_ids)].index
    # custom_inner_splits.append((train_inner_idx_full, val_idx_full))

    X_train_outer = X_train_outer.drop(columns=['PKI', 'mol_id', 'conformations (ns)'], errors='ignore')

    X_train_inner, X_validation_inner = X_train_outer.loc[train_inner_idx_full], X_train_outer.loc[val_inner_idx_full]
    y_train_inner, y_validation_inner = y_train_outer.loc[train_inner_idx_full], y_train_outer.loc[val_inner_idx_full]

    X_train_inner_grouped = X_train_inner.values.reshape(-1, num_of_conformations, X_train_inner.shape[1])  # Shape: (num_molecules, num_of_conformations, num_features)
    y_train_inner_grouped = y_train_inner.values.reshape(-1, num_of_conformations)  # Shape: (num_molecules, 10)
    X_validation_inner_grouped = X_validation_inner.values.reshape(-1, num_of_conformations, X_validation_inner.shape[1])  # Shape: (num_molecules, 10, num_features)
    y_validation_inner_grouped = y_validation_inner.values.reshape(-1, num_of_conformations)

    train_inner_dataset = TensorDataset(torch.tensor(X_train_inner_grouped, dtype=torch.float32),
                                        torch.tensor(y_train_inner_grouped, dtype=torch.float32))
    val_inner_dataset = TensorDataset(torch.tensor(X_validation_inner_grouped, dtype=torch.float32),
                            torch.tensor(y_validation_inner_grouped, dtype=torch.float32))
    train_inner_loader = DataLoader(train_inner_dataset, batch_size=128, shuffle=True) #64 molecules at once independent of c conformations
    val_inner_loader = DataLoader(val_inner_dataset, batch_size=128, shuffle=False)
    return train_inner_loader, val_inner_loader

def inner_fold_splitting(inner_folds, inner_fold_idx, train_outer_mol_ids, grouped_binned_y):
    # Select only bins for molecules in train_outer_mol_ids
    train_inner_bins = grouped_binned_y.loc[train_outer_mol_ids]

    # Convert train_outer_mol_ids to a NumPy array for proper indexing
    train_outer_mol_ids = np.array(train_outer_mol_ids)  
    train_outer_bin_values = train_inner_bins.values  # Binned labels for stratification

    # Stratified K-Fold Cross-Validation
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=10)
    all_inner_indices = list(inner_cv.split(X=np.arange(len(train_outer_mol_ids)), y=train_outer_bin_values))

    # Extract train and validation molecule IDs for the specified inner fold
    train_inner_idx, val_inner_idx = all_inner_indices[inner_fold_idx]

    train_inner_mol_ids = train_outer_mol_ids[train_inner_idx].tolist()
    val_inner_mol_ids = train_outer_mol_ids[val_inner_idx].tolist()
    return train_inner_mol_ids, val_inner_mol_ids

def random_outer_splitting(outer_fold_number, grouped_df, grouped_binned_y, unique_mol_ids):

    random_state_value = outer_fold_number  # Change the seed for each fold
    
    # Step 1: Split into 70% training and 30% combined validation+test
    val_train_mol_ids, test_mol_ids = train_test_split(unique_mol_ids, test_size=0.15, stratify=grouped_binned_y[unique_mol_ids], random_state=random_state_value)

    # Step 2: Split the 30% into 10% validation and 20% test
    train_outer_mol_ids, val_outer_mol_ids = train_test_split(val_train_mol_ids, test_size=1/17, stratify=grouped_binned_y[val_train_mol_ids], random_state=random_state_value)
    return train_outer_mol_ids, test_mol_ids, val_outer_mol_ids

def random_inner_splitting(inner_fold_number, grouped_binned_y, train_outer_mol_ids):

    random_state_value = inner_fold_number  # Change the seed for each fold

    # Step 1: Split into 65% training and 10% validation from 75% outer_train
    train_inner_mol_ids, val_inner_mol_ids = train_test_split(train_outer_mol_ids, test_size=3/17, stratify=grouped_binned_y[train_outer_mol_ids], random_state=random_state_value)

    return train_inner_mol_ids, val_inner_mol_ids


def kfold_outer_splitting(outer_fold_number, outer_folds, grouped_df,grouped_binned_y, unique_mol_ids):

    outer_cv = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=10)
    
    all_outer_indices = list(outer_cv.split(X=grouped_df, y=grouped_binned_y, groups=unique_mol_ids))
    all_outer_mol_ids = [
        (grouped_df.index[train_idx].tolist(), grouped_df.index[val_idx].tolist())
        for train_idx, val_idx in all_outer_indices
    ]
    
    train_mol_ids, test_mol_ids = all_outer_mol_ids[outer_fold_number]
    train_outer_mol_ids, val_outer_mol_ids = train_test_split(train_mol_ids, test_size=1/18, stratify=grouped_binned_y[train_mol_ids], random_state=10)
    # val_outer_mol_ids = all_outer_mol_ids[val_outer_fold][1]

    # print('test')
    # print(sorted(test_mol_ids))
    # print(len(sorted(test_mol_ids)))

    # print('val outer mol ids')

    # print(sorted(val_outer_mol_ids))
    # print(len(sorted(val_outer_mol_ids)))

    # print('train outer mol ids')

    # print(train_outer_mol_ids)
    # print(len(train_outer_mol_ids))

    # print('next')
    # print(sorted(train_outer_mol_ids), sorted(test_mol_ids), sorted(val_outer_mol_ids))
    return train_outer_mol_ids, test_mol_ids, val_outer_mol_ids

def deeplearning_function(name, df, dfs_path, random_splitting = False):
    if random_splitting == False:
        folding = '10folds'
    else:
        folding = 'random'

    # Step 2: Split into features and target
    target_column = 'PKI'

    # Step 1: Sort dataframe
    sort_cols = ['mol_id', 'conformations (ns)'] if 'conformations (ns)' in df.columns else ['mol_id']
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Step 2: Standardize the features and store them back in df
    feature_cols = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], errors='ignore').columns

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    y = df[target_column] #series
    
    # # Step 4: Bin target values to handle imbalance
    bins = 4
    binned_y, bin_edges = pd.cut(y, bins=bins, labels=False, retbins=True, include_lowest=True)
    grouped_df = df.groupby('mol_id').first()
    bin_counts = np.bincount(binned_y)  # Count samples per bin

    grouped_binned_y, bin_edges = pd.cut(grouped_df[target_column], bins=bins, labels=False, retbins=True, include_lowest=True)

    unique_mol_ids = list(grouped_df.index)

    num_of_conformations = df.groupby('mol_id').size().iloc[0]
    # Count the number of conformations per molecule
    mol_conformation_counts = df.groupby('mol_id').size()

    # Find molecules that do not have the expected number of conformations
    valid_mol_ids = mol_conformation_counts[mol_conformation_counts == num_of_conformations].index

    # Keep only molecules with the correct number of conformations
    df = df[df['mol_id'].isin(valid_mol_ids)].copy()
    fold_results = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }

    hyperparameter_grid = pv.ML_MODEL.get_hyperparameter_grid #48 long
    print(len(hyperparameter_grid))
    # hyperparameter_grid = [hyperparameter_grid[0]]
    # print(hyperparameter_grid)
    
    # Store losses and predictions across folds
    mol_ids = []
    all_train_outer_losses = []
    all_val_outer_losses = []
    all_best_params_outer = []
    all_predictions = []
    all_true_values = []

    outer_folds = 10
    for outer_fold_idx in range(outer_folds):
        if random_splitting:
            print('random splitting')
            train_outer_mol_ids, test_mol_ids, val_outer_mol_ids = random_outer_splitting(outer_fold_idx, grouped_df,grouped_binned_y, unique_mol_ids)
        else:
            print('kfold_outer splitting')
            train_outer_mol_ids, test_mol_ids, val_outer_mol_ids = kfold_outer_splitting(outer_fold_idx,outer_folds, grouped_df,grouped_binned_y, unique_mol_ids)
        print(len(sorted(train_outer_mol_ids)))
        # print(sorted(val_outer_mol_ids))

        # print(sorted(test_mol_ids))

        train_outer_idx_all = df[df['mol_id'].isin(train_outer_mol_ids)].index
        test_idx_all = df[df['mol_id'].isin(test_mol_ids)].index
        val_outer_idx_all = df[df['mol_id'].isin(val_outer_mol_ids)].index

        # custom_outer_splits.append((train_idx_all, validation_outer_idx_all, test_idx_all))
        X_train_outer, X_val_outer ,X_test = df.loc[train_outer_idx_all], df.loc[val_outer_idx_all], df.loc[test_idx_all]
        y_train_outer, y_val_outer, y_test = y.loc[train_outer_idx_all], y.loc[val_outer_idx_all], y.loc[test_idx_all]

        # mol_ids_to_data_outer()
        hyperparameter_losses = [[] for _ in hyperparameter_grid]
        hyperparameter_loss_curves = {param_idx: {"train_losses": [], "val_losses": []} for param_idx in range(len(hyperparameter_grid))}

        inner_folds = 2
        for inner_fold_idx in range(inner_folds):
            inner_col = f"outer_{outer_fold_idx}_inner_{inner_fold_idx}"
            print(inner_col)

            train_inner_mol_ids, val_inner_mol_ids = random_inner_splitting(inner_fold_idx, grouped_binned_y, train_outer_mol_ids)
            # bin_distribution = pd.value_counts(grouped_binned_y[train_inner_mol_ids], sort=True)
            # print(bin_distribution)
            # bin_distribution = pd.value_counts(grouped_binned_y[train_outer_mol_ids], sort=True)
            # print(bin_distribution)
            # bin_distribution = pd.value_counts(grouped_binned_y[val_inner_mol_ids], sort=True)
            # print(bin_distribution)
            # bin_distribution = pd.value_counts(grouped_binned_y[test_mol_ids], sort=True)
            # print(bin_distribution)
            train_inner_loader, val_inner_loader = get_inner_dataloaders(train_inner_mol_ids, val_inner_mol_ids, X_train_outer, y_train_outer, num_of_conformations)
            # Hyperparameter tuning
            best_params = None
            best_val_loss = float('inf')
            input_size = len(feature_cols)

            nums_of_epochs = 2000
            patience = 150
            for param_idx, hyperparameter_set in enumerate(hyperparameter_grid):

                if pv.ML_MODEL == Model_deep.LSTM:
                    model, train_inner_losses, val_inner_losses = build_LSTM_model(train_inner_loader, val_inner_loader, input_size, nums_of_epochs, patience, hyperparameter_set)
                elif pv.ML_MODEL == pv.Model_deep.DNN:
                    model, train_inner_losses, val_inner_losses = build_DNN_model(train_inner_loader, val_inner_loader, input_size, num_of_conformations, nums_of_epochs, patience, hyperparameter_set)

                # Store validation loss for the current hyperparameter set
                hyperparameter_losses[param_idx].append(np.mean(val_inner_losses[-50:]) if len(val_inner_losses) >= 50 else np.mean(val_inner_losses))

                # Append losses for each inner fold (so we get all 5 curves)
                hyperparameter_loss_curves[param_idx]["train_losses"].append(train_inner_losses)
                hyperparameter_loss_curves[param_idx]["val_losses"].append(val_inner_losses)

        
        # Compute the average validation loss over the last 50 epochs for each hyperparameter set
        avg_hyperparameter_losses = [sum(losses) / len(losses) for losses in hyperparameter_losses] #if i want to use the all losses
        print(avg_hyperparameter_losses)
        # avg_hyperparameter_losses = [
        #     sum(losses[-50:]) / len(losses[-50:]) if len(losses) >= 50 else sum(losses) / len(losses) 
        #     for losses in hyperparameter_losses
        # ]

        # Find the hyperparameter set with the lowest average validation loss
        best_hyperparameter_idx = min(range(len(avg_hyperparameter_losses)), key=avg_hyperparameter_losses.__getitem__)
        # Retrieve the loss curves for the best hyperparameter set
        best_train_losses = hyperparameter_loss_curves[best_hyperparameter_idx]["train_losses"]
        best_val_losses = hyperparameter_loss_curves[best_hyperparameter_idx]["val_losses"]
        best_params = hyperparameter_grid[best_hyperparameter_idx]
        print('best hyperparameter set: ', best_params)

        # Prepare dictionary for storing
        loss_data = {
            "train_losses": best_train_losses,
            "val_losses": best_val_losses,
            "best_hyperparameters": best_params  # Include the best hyperparameters
        }

        # Save to JSON file
        save_path = dfs_path / pv.Inner_train_Val_losses / name
        save_path.mkdir(parents = True, exist_ok=True)
        with open(save_path / f"{name}_{folding}_5loss_curves_of_best_inner_of_outer_{outer_fold_idx}.json", "w") as f:
            json.dump(loss_data, f, indent=4)
        
        feature_cols = df.drop(columns=[target_column, 'mol_id', 'conformations (ns)'], errors='ignore').columns
        X_train_outer = X_train_outer[feature_cols]
        X_val_outer = X_val_outer[feature_cols]
        X_test = X_test[feature_cols]
        X_train_outer_grouped = X_train_outer.values.reshape(-1, num_of_conformations, X_train_outer.shape[1])
        y_train_outer_grouped = y_train_outer.values.reshape(-1, num_of_conformations)  # Shape: (num_molecules, 10)
        X_val_outer_grouped = X_val_outer.values.reshape(-1, num_of_conformations, X_val_outer.shape[1])  # Shape: (num_molecules, 10, num_features)
        y_val_outer_grouped = y_val_outer.values.reshape(-1, num_of_conformations)  # Shape: (num_molecules, 10)
        
        # Train final model on 80% training data with best hyperparameters
        train_outer_dataset = TensorDataset(torch.tensor(X_train_outer_grouped, dtype=torch.float32),
                                            torch.tensor(y_train_outer_grouped, dtype=torch.float32))
        validation_outer_dataset = TensorDataset(torch.tensor(X_val_outer_grouped, dtype=torch.float32),
                                    torch.tensor(y_val_outer_grouped, dtype=torch.float32))
        train_outer_loader = DataLoader(train_outer_dataset, batch_size=64, shuffle=True)
        val_outer_loader = DataLoader(validation_outer_dataset, batch_size=64, shuffle=False)
        
        results = {}
        nums_of_epochs = 2000
        patience = 250
        set_random_seed(42)
        if pv.ML_MODEL == Model_deep.LSTM:
            final_model, train_outer_losses, val_outer_losses = build_LSTM_model(train_outer_loader, val_outer_loader, input_size, nums_of_epochs, patience, best_params)
        elif pv.ML_MODEL == Model_deep.DNN:
            final_model, train_outer_losses, val_outer_losses = build_DNN_model(train_outer_loader, val_outer_loader, input_size, num_of_conformations, nums_of_epochs, patience, best_params)

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
        print("cuda") if torch.cuda.is_available() else print("cpu")

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
        print(f"Fold {outer_fold_idx}; MSE: {mse_value:.4f}, MAE: {mae_value:.4f}, Test R² score: {r2_value:.4f}")

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

    with open(save_path / f'{name}_{folding}_loss_curves_all_outer_folds.json', 'w') as f:
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
    df_true_predicted.to_csv(save_path / f'{name}_{folding}_true_predicted.csv', index=False)
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

def save_fold_results(results, metric, ModelResults, Modelresults_path, random_splitting):
    if random_splitting == False:
        folding = '10folds'
    else:
        folding = 'random'
    # Process each metric separately
    # Append the results for this metric
    ModelResults[metric].append(results)
    
    # Write the updated results to a temporary CSV file
    csv_filename_temp = f'ModelResults_{folding}_{metric}_{pv.DESCRIPTOR}_temp.csv'
    pd.DataFrame(ModelResults[metric]).to_csv(Modelresults_path / csv_filename_temp, index=False)

    csv_filename = f'ModelResults_{folding}_{metric}_{pv.DESCRIPTOR}.csv'
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

def main(dfs_path = pv.dfs_descriptors_only_path_, random_splitting = False ,include_files = []):
    print(dfs_path)
    
    Modelresults_path = dfs_path / pv.Modelresults_folder_ #create Modelresults folder
    Modelresults_path.mkdir(parents=True, exist_ok=True)

    if not include_files:
        include_files = ['conformations_10.csv']
    
    dfs_in_dict = dataframe_processing.csvfiles_to_dict_include(dfs_path, include_files=include_files) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary

    ModelResults = {'R2': [], 'MSE': [], 'MAE': []}

    for name, df in dfs_in_dict.items():
        print(name)
        # Perform nested cross-validation for the current dataset
        results_all_metrics = deeplearning_function(name, df, dfs_path, random_splitting)
        for metric, results in results_all_metrics.items():
            save_fold_results(results, metric, ModelResults, Modelresults_path, random_splitting)
    return

if __name__ == "__main__":
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)

    # # # main(pv.dfs_descriptors_only_path_)
    # main(pv.dfs_descriptors_only_path_, random_splitting = True, include_files = ['0ns.csv','conformations_10.csv'])
    # pv.update_config(model_=Model_deep.LSTM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dfs_reduced_and_MD_path_, random_splitting = False, include_files = ['conformations_10.csv','conformations_50.csv','conformations_100.csv'])

    # main(pv.dfs_reduced_and_MD_path_, random_splitting = True, include_files = ['conformations_10.csv','conformations_50.csv','conformations_100.csv'])
    # main(pv.dfs_MD_only_path_, random_splitting = False, include_files = ['conformations_10.csv','conformations_50.csv','conformations_100.csv'])
    # main(pv.dfs_reduced_path_, random_splitting = True, include_files = ['conformations_50.csv','conformations_100.csv'])

    # for path in pv.get_paths():
    #     print(path)
    #     main(path, random_splitting = False, include_files = ['conformations_10.csv','conformations_50.csv','conformations_100.csv'])
    #     main(path, random_splitting = True, include_files = ['conformations_10.csv','conformations_50.csv','conformations_100.csv'])
    
    pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])

    # main(pv.dfs_reduced_PCA_path_, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])
    # main(pv.dfs_reduced_MD_PCA_path_, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])
    # main(pv.dfs_reduced_and_MD_combined_PCA_path_, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])
    # main(pv.dfs_all_PCA, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])
    
    # for protein in DatasetProtein:
    #     pv.update_config(protein_=protein)
    #     print(protein)
    #     if protein == DatasetProtein.JAK1:
    #         print('ja')
    #         continue
    #     if protein == DatasetProtein.GSK3:
    #         continue
    #         # main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['conformations_10.csv','conformations_20.csv'])
    #         # main(pv.dfs_reduced_path_, random_splitting = False, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv'])
    #     if protein == DatasetProtein.pparD:
    #         # main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv'])
    #         main(pv.dfs_reduced_path_, random_splitting = False, include_files = ['2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv'])

    #     pv.update_config(protein_=protein)
    #         # main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['conformations_10.csv'])
    # for protein in DatasetProtein:
    #     pv.update_config(protein_=protein)
    #     print(protein)
        # if protein == DatasetProtein.JAK1:
        #     main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['conformations_10.csv', 'conformations_20.csv','conformations_50.csv', 'conformations_100.csv', 'conformations_200.csv'])

        # if protein == DatasetProtein.GSK3:
        #     main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['conformations_10.csv', 'conformations_20.csv','conformations_50.csv', 'conformations_100.csv', 'conformations_200.csv'])
        #     main(pv.dfs_reduced_path_, random_splitting = False, include_files = ['conformations_10.csv', 'conformations_20.csv','conformations_50.csv', 'conformations_100.csv', 'conformations_200.csv'])

        #     # main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['conformations_10.csv','conformations_20.csv'])
        #     # main(pv.dfs_reduced_path_, random_splitting = False, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv'])
        # if protein == DatasetProtein.pparD:
        #     main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['conformations_10.csv', 'conformations_20.csv','conformations_50.csv', 'conformations_100.csv', 'conformations_200.csv'])
        #     main(pv.dfs_reduced_path_, random_splitting = False, include_files = ['conformations_10.csv', 'conformations_20.csv','conformations_50.csv', 'conformations_100.csv', 'conformations_200.csv'])

            # main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv'])







    #12 uur per ding
    # main(pv.dfs_descriptors_only_path_, random_splitting = True, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv','minimized_conformations_10.csv'])
    # main(pv.dfs_descriptors_only_path_, random_splitting = True, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv','minimized_conformations_10.csv'])
    # main(pv.dfs_reduced_path_, random_splitting = True, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv','minimized_conformations_10.csv'])
    # main(pv.dfs_reduced_path_, random_splitting = True, include_files = ['0ns.csv','2ns.csv','4ns.csv','6ns.csv','8ns.csv','10ns.csv','conformations_10.csv','conformations_20.csv','minimized_conformations_10.csv'])
    
    
    
    # main(pv.dfs_MD_only_path_, random_splitting = True, include_files = ['0ns.csv','5ns.csv','conformations_10.csv'])
    # main(pv.dfs_descriptors_only_path_, random_splitting = False, include_files = ['5ns.csv'])

    # for path in pv.get_paths():
    #     main(path, random_splitting = True, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])
    #     # main(path, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])
    # for path in pv.get_paths():
    #     main(path, random_splitting = False, include_files = ['0ns.csv','1ns.csv','5ns.csv','conformations_10.csv'])

    # main(pv.dfs_descriptors_only_path_, random_splitting = True, include_files = ['conformations_10.csv','conformations_100.csv'])


