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
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

from Models.DNN import FullyConnectedDNN_class2
torch.manual_seed(42)

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
    print(grouped_binned_y)

    fold_results = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }
    # Initialize variables
    custom_outer_splits = []
    all_idx_ytrue_pki_series = pd.Series(dtype=float)
    all_idx_ypredicted_pki_series = pd.Series(dtype=float)
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,shuffle=True, random_state=10)

    outer_folds = 5
    outer_kf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    r2_scores = []
    mse_scores = []

    hyperparameter_grid = [
        {"learning_rate": 0.001, "hidden_layers": [64, 32, 16],"dropout_rate": 0.1}, #increase dropout rate to 0.2 as well because small dataset
        {"learning_rate": 0.001, "hidden_layers": [64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [64, 32, 16],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64, 32],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [128, 64],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [256, 128, 64],"dropout_rate": 0.1},
        {"learning_rate": 0.001, "hidden_layers": [256, 128],"dropout_rate": 0.1},
    ]

    # Store losses and predictions across folds
    all_predictions = []
    all_true_values = []
    all_train_inner_losses = {}
    all_val_inner_losses = {}
    all_train_outer_losses = {}
    all_val_outer_losses = {}
    all_train_inner_lossesl = []
    all_val_inner_lossesl = []
    all_train_outer_lossesl = []
    all_val_outer_lossesl = []
    all_params = []
    fold_results = {"epoch_results": {}}
    # for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X_scaled, binned_y)):
    #     print(f"Outer Fold {fold + 1}")

    #     # Split into train and test sets
    #     X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    #     y_train, y_test = y[train_idx], y[test_idx]

    #     # Inner split: Further split training data into 70% training, 10% validation
    #     X_inner_train, X_val, y_inner_train, y_val = train_test_split(
    #         X_train, y_train, test_size=0.125, stratify=pd.cut(y_train.flatten(), bins=bins, labels=False),
    #         random_state=42
    #     )
    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X_scaled, binned_y)):
        print(f"Outer Fold {fold + 1}")

        # Split into train and test sets
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Further split training data into 70% training, 10% validation outer
        X_train_outer, X_val_outer, y_train_outer, y_val_outer = train_test_split(
            X_train, y_train, test_size=0.125, stratify=pd.cut(y_train.flatten(), bins=bins, labels=False),
            random_state=42
        ) #size 688 and 99

        #divide the 70% training into 56% training inner and 14% validation inner
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
            X_train_outer, y_train_outer, test_size=0.2, stratify=pd.cut(y_train_outer.flatten(), bins=bins, labels=False),
            random_state=42
        )

        # Convert to tensors for inner split
        train_inner_dataset = TensorDataset(torch.tensor(X_train_inner, dtype=torch.float32),
                                            torch.tensor(y_train_inner, dtype=torch.float32))
        val_inner_dataset = TensorDataset(torch.tensor(X_val_inner, dtype=torch.float32),
                                    torch.tensor(y_val_inner, dtype=torch.float32))
        train_inner_loader = DataLoader(train_inner_dataset, batch_size=32, shuffle=True)
        val_inner_loader = DataLoader(val_inner_dataset, batch_size=32, shuffle=False)

        # Hyperparameter tuning
        best_params = None
        best_val_loss = float('inf')
        for params in hyperparameter_grid:
            model = FullyConnectedDNN_class2.FullyConnectedDNN2(input_size=X_train_inner.shape[1], hidden_layers=params["hidden_layers"],dropout_rate=params["dropout_rate"])
            optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
            criterion = torch.nn.MSELoss()

            # Train with validation to tune hyperparameters
            train_inner_losses, val_inner_losses = model.train_with_validation(
                train_inner_loader, val_inner_loader, num_epochs=300, optimizer=optimizer,
                criterion=criterion, device="cuda" if torch.cuda.is_available() else "cpu", patience=25
            )

            # Track the best hyperparameters
            if val_inner_losses[-1] < best_val_loss:
                best_val_loss = val_inner_losses[-1]
                best_params = params
                best_train_inner_losses = train_inner_losses
                best_val_inner_losses = val_inner_losses

        print(f"Best params for fold {fold + 1}: {best_params}, Validation Loss: {best_val_loss:.4f}")

        # Train final model on 70% training data with best hyperparameters
        train_outer_dataset = TensorDataset(torch.tensor(X_train_outer, dtype=torch.float32),
                                            torch.tensor(y_train_outer, dtype=torch.float32))
        val_outer_dataset = TensorDataset(torch.tensor(X_val_outer, dtype=torch.float32),
                                    torch.tensor(y_val_outer, dtype=torch.float32))
        train_outer_loader = DataLoader(train_outer_dataset, batch_size=32, shuffle=True)
        val_outer_loader = DataLoader(val_outer_dataset, batch_size=32, shuffle=False)

        epochs = 1000
        results = {}
        
        final_model = FullyConnectedDNN_class2.FullyConnectedDNN2(input_size=X_train_outer.shape[1], hidden_layers=best_params["hidden_layers"],dropout_rate=best_params["dropout_rate"])
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
        criterion = torch.nn.MSELoss()

        train_outer_losses, val_outer_losses = final_model.train_with_validation(
                train_outer_loader, val_outer_loader, num_epochs=epochs, optimizer=optimizer,
                criterion=criterion, device="cuda" if torch.cuda.is_available() else "cpu", patience=100
            )

        # Evaluate on test set
        final_model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                    torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        predictions = []
        true_values = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = final_model(inputs)
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(targets.numpy())
        
        # Calculate metrics for this epoch
        mse = mean_squared_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        print(f"Epochs {epochs} - Test MSE: {mse:.4f}, Test R²: {r2:.4f}")
        fold_results["epoch_results"][epochs] = {"mse": mse, "r2": r2}
        # Append fold results
        all_train_inner_losses[fold] = train_inner_losses
        all_val_inner_losses[fold] = val_inner_losses
        all_train_outer_losses[fold] = train_outer_losses
        all_val_outer_losses[fold] = val_outer_losses

        all_train_inner_lossesl.append(train_inner_losses)
        all_val_inner_lossesl.append(val_inner_losses)
        all_train_outer_lossesl.append(train_outer_losses)
        all_val_outer_lossesl.append(val_outer_losses)

        all_predictions.extend(predictions)
        all_true_values.extend(true_values)
        # train_inner_losses.append(final_train_losses)
        # val_inner_losses.append(final_val_losses)
        all_params.append(best_params)
        mse_scores.append(mse)        
        r2_scores.append(r2)
    
    # Flatten the lists
    all_predictions_flat = list(itertools.chain(*all_predictions))
    all_true_values_flat = list(itertools.chain(*all_true_values))
    mol_ids = df["mol_id"].tolist()
    print(mol_ids)
    results_df = pd.DataFrame({
        "mol_id": mol_ids,  # Molecule IDs
        "true_value": all_true_values_flat,  # True values for each conformation
        "prediction": all_predictions_flat   # Predictions for each conformation
    })
    averaged_results = results_df.groupby("mol_id").agg(
        avg_true_value=("true_value", "mean"),
        avg_prediction=("prediction", "mean")
    ).reset_index()

    # If you want the averaged true values and predictions as lists
    avg_true_values = averaged_results["avg_true_value"].tolist()
    avg_predictions = averaged_results["avg_prediction"].tolist()
    mse_avg = mean_squared_error(avg_true_values, avg_predictions)
    r2_avg = r2_score(avg_true_values, avg_predictions)
    # Create the results dictionary with flattened values
    results_dict = {
        "predictions": all_predictions_flat,
        "true_values": all_true_values_flat
    }
    # Save all_train_inner_losses
    newfolder = dfs_path / pv.Modelresults_folder_ / 'train_loss_plots'
    
    newfolder.mkdir(parents=True, exist_ok=True)
    train_inner_losses_df = pd.DataFrame.from_dict(all_train_inner_losses, orient="index").T
    train_inner_losses_df.to_csv(newfolder / f"train_inner_losses_{name}.csv", index=False)

    # Save all_val_inner_losses
    val_inner_losses_df = pd.DataFrame.from_dict(all_val_inner_losses, orient="index").T
    val_inner_losses_df.to_csv(newfolder / f"val_inner_losses.csv_{name}", index=False)

    # Save all_train_outer_losses
    train_outer_losses_df = pd.DataFrame.from_dict(all_train_outer_losses, orient="index").T
    train_outer_losses_df.to_csv(newfolder / f"train_outer_losses_{name}.csv", index=False)

    # Save all_val_outer_losses
    val_outer_losses_df = pd.DataFrame.from_dict(all_val_outer_losses, orient="index").T
    val_outer_losses_df.to_csv(newfolder / f"val_outer_losses_{name}.csv", index=False)


    results_df = pd.DataFrame(results_dict)
    # Convert the dictionary into a DataFrame
    anotherfolder = dfs_path / pv.Modelresults_folder_ / 'pred_true_plots'
    anotherfolder.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(dfs_path / pv.Modelresults_folder_ / 'pred_true_plots' / f"model_pred_vs_true_{name}.csv", index=False)

    # Print overall metrics
    print(f"Average MSE: {np.mean(mse_scores):.4f}")
    print(f"Average R²: {np.mean(r2_scores):.4f}")

    # fold_results.append(r2_score)

    data = pd.Series(fold_results)

    fold_results = {
        "mol_id": name,
        "mean_test_score": pd.Series(r2_scores).mean(),
        "std_test_score": pd.Series(r2_scores).std(),
        "params": [1,2],
        **{f"split{split_idx}_test_score": r2_scores[split_idx] for split_idx in range(outer_folds)},  # Loop to add splits
    }

    # create_train_inner_loss_plots(name, all_params, all_train_inner_lossesl, all_val_inner_lossesl, dfs_path)
    # create_train_loss_plots(name, all_params, all_train_outer_lossesl, all_val_outer_lossesl, dfs_path)
    create_pred_true_plots(name, all_params, all_true_values, all_predictions, dfs_path)

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
    dfs_in_dic = csv_to_dictionary.csvfiles_to_dic_include(dfs_path, include_files=['conformations_10.csv']) #get all the created csvfiles from e.g. 'dataframes_JAK1_WHIM' into a dictionary
    
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

    main(pv.dfs_descriptors_only_path_)
    # main(pv.dfs_reduced_and_MD_path_)

    # main(pv.dfs_reduced_path_)
    # main(pv.dfs_MD_only_path_)
    # for protein_ in pv.DatasetProtein:
    #     pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein)
    #     # Construct paths dynamically for each protein
    #     main(pv.dfs_descriptors_only_path_)
    #     main(pv.dfs_reduced_path_)
    #     main(pv.dfs_reduced_and_MD_path_)
    #     main(pv.dfs_MD_only_path_)

