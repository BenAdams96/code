import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from global_files import public_variables
from collections import defaultdict
import math

from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

def display_dataframe_summary(dataframes_nested_dict):
    """
    Display a summary of the DataFrames collected in a nested dictionary format.
    
    Args:
    - dataframes_nested_dict (dict): Nested dictionary with filenames as keys and subfolder DataFrames as values.
    """
    for file_name, subfolder_dict in dataframes_nested_dict.items():
        print(f"\n{file_name}:")
        for subfolder_name, df in subfolder_dict.items():
            # Display summary of the DataFrame
            print(f"  {subfolder_name}:")
            # print(f"    - Shape: {df.shape}")  # Show shape of DataFrame
            # print(f"    - Columns: {df.columns.tolist()}")  # Show list of columns
            # print(f"    - Head:\n{df.head(2)}")  # Show the first two rows as a quick look
            # print("    ...")  # To indicate that there's more data

# # Define the path to the master folder
# master_folder = Path('/home/ben/Download/Afstuderen0/Afstuderen/dataframes_JAK1_WHIM_i1')

# # Collect DataFrames from CSV files into a nested dictionary
# dataframes_nested_dict = collect_dataframes_nested_dict(master_folder)

# # Visualize or further analyze the collected data
# visualize_collected_data_nested(dataframes_nested_dict)

def csvfile_to_df(csv_file):
    df = pd.read_csv(csv_file)
    return df



def dataframes_to_dic(dfs_path):
    """
    Collect DataFrames from CSV files with matching names across different subfolders within a master folder.
    
    Args:
    - master_folder (Path): Path to the master folder containing subfolders.
    
    Returns:
    - dataframes_nested_dict (dict): Nested dictionary where keys are filenames and values are dicts of DataFrames by subfolder names.
    """
    dataframes_nested_dict = defaultdict(lambda: defaultdict(pd.DataFrame))
    subfolders = [f for f in dfs_path.iterdir() if f.is_dir()]
    
    # Sort subfolders based on their order in the file system
    subfolders.sort()
    
    for subfolder in subfolders:
        subfolder_name = subfolder.name
        model_results_folder = subfolder / pv.Modelresults_folder_
        if model_results_folder.exists(): #checks if Modelresults folder exists, if not, it doesnt count
            for csv_file in model_results_folder.glob('*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    dataframes_nested_dict[csv_file.name][subfolder_name] = df
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
    #ordered_row_names = df['id'].tolist()
    
    return dataframes_nested_dict #, ordered_row_names

def prepare_data_for_boxplot(dataframes_nested_dict, subfolders_order):
    """
    Prepare data for a boxplot visualization by organizing split scores across different subfolders.
    
    Args:
    - dataframes_nested_dict (dict): Nested dictionary containing DataFrames by filename and subfolder name.
    
    Returns:
    - dict: Dictionary where keys are file names and values are lists of data for the boxplot.
    """
    plot_data_dict = {}
    
    for file_name, subfolder_dict in dataframes_nested_dict.items():
        # Sort the subfolder keys based on the order of subfolders in the master folder
        #sorted_subfolders = sorted(subfolder_dict.keys(), key=lambda x: x.lower())

        # Initialize the list to store data for plotting
        plot_data = []
        
        # Determine the number of rows in the DataFrame (assumed to be the same across subfolders)
        num_rows = list(subfolder_dict.values())[0].shape[0]
        split_columns = [col for col in list(subfolder_dict.values())[0].columns if col.startswith('split') and col.endswith('_test_score')]
        
        # Loop through each row and collect data for all subfolders
        for row_idx in range(num_rows):
            row_data = []  # Collect split scores for a single row across subfolders
            for subfolder_name in subfolders_order:
                df = subfolder_dict[subfolder_name]
                # Collect split scores for the current row
                
                
                split_scores = df.iloc[row_idx][split_columns].values
                row_data.append(split_scores)
            plot_data.extend(row_data)
        
        # Save collected data for each CSV file
        plot_data_dict[file_name] = plot_data
    
    return plot_data_dict

def plot_boxplots_compare_ns(data_for_plot, sorted_subfolders,ordered_row_names, save_plot_folder):
    """
    Plot separate boxplots for different K-fold and metric combinations with improved readability.
    
    Args:
    - data_for_plot (dict): Prepared data dictionary for plotting.
    - save_plot_folder (Path): Folder path to save the resulting boxplots.
    - box_width (float): Width of each individual box.
    - group_gap (float): Gap between boxes in the same group.
    - group_spacing (float): Spacing between different groups of boxes.
    """
    # Create a folder to save plots if it doesn't exist
    save_plot_folder = save_plot_folder / Path('boxplots_compare_ns')
    save_plot_folder.mkdir(parents=True, exist_ok=True)
    
    # Define colors and labels for different subfolders
    colors = {
        "descriptors only": 'lightblue',
        "reduced_t0.85": 'lightgreen',
        "reduced_t0.85_MD": 'lightcoral',
        "MD only": 'lightgray',  # New color for "MD only"
        "custom_dataframes": 'mediumorchid'
    }

    labels = {
        "descriptors only": 'Descriptors Only',
        "reduced_t0.85": 'Reduced t0.85',
        "reduced_t0.85_MD": 'Reduced t0.85 MD',
        "MD only": 'MD Only',  # Added label for "MD only"
        "custom_dataframes": 'x features'
    }
    
    colors = {key: colors[key] for key in sorted_subfolders if key in colors}
    labels = {key: labels[key] for key in sorted_subfolders if key in labels}

    box_width=1
    group_gap=0
    group_spacing=0

    # Loop through each filename and corresponding plot data
    for file_name, plot_data in data_for_plot.items():
        plt.figure(figsize=(14, 7))
        
        # Number of boxes per group (based on number of subfolders)
        num_subfolders = len(sorted_subfolders)
        num_rows = len(plot_data) // num_subfolders  # Number of 'ns' groups (and 'rdkit_min')
        
        positions = []
        box_data = []
        box_colors = []
        widths = []
        
        # Plot the boxes for each row
        for row_idx in range(num_rows):  # loop over the rows
            row_data = plot_data[row_idx*num_subfolders:(row_idx+1)*num_subfolders]
            # Calculate base position for the current group
            base_position = row_idx * ((num_subfolders * box_width) + group_spacing + ((num_subfolders-1) * group_gap))
            for idx, (subfolder_name, color) in enumerate(colors.items()):
                # Collect split scores for the current subfolder
                split_scores = row_data[idx]
                
                # Calculate position for the current box
                pos = base_position + idx * (box_width + group_gap)
                # Append data and position for the box plot
                positions.append(pos)
                box_data.append(split_scores)
                box_colors.append(color)
                widths.append(box_width)  # Append width for each box
            base_position = base_position + group_gap
        # Create a boxplot with specified positions, colors, and widths
        
        boxplot_dict = plt.boxplot(box_data, positions=positions, patch_artist=True, 
                                   widths=widths,  # Set box widths
                                   medianprops=dict(color='red'))
        # # Create a boxplot with specified positions and colors
        # boxplot_dict = plt.boxplot(box_data, positions=positions, patch_artist=True, 
        #                            medianprops=dict(color='red'))
        
        # Apply colors to each box
        for patch, color in zip(boxplot_dict['boxes'], box_colors):
            patch.set_facecolor(color)
        
        # Add a legend
        handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()]
        plt.legend(handles=handles, labels=labels.values(), loc='best')

        plt.title(f"Boxplot for {file_name}")
        plt.xlabel("Rows (Hyperparameter Sets) across Subfolders")
        plt.ylabel("Scores")
        # Adjust x-axis limits
        plt.xlim(-(box_width/2)-group_spacing, positions[-1] + group_spacing + (box_width/2))

        # Set xticks and labels
        xtick_labels = ordered_row_names
        xticks_positions = np.arange(
            (num_subfolders-1)*(0.5*box_width+0.5*group_gap),#((((num_subfolders) * box_width) + ((num_subfolders-1)*group_gap)),
            positions[-1],
            (num_subfolders * box_width) + ((num_subfolders-1) * group_gap) + group_spacing
        )
        plt.xticks(ticks=xticks_positions, labels=xtick_labels, rotation=45)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot
        plot_file_path = save_plot_folder / f'boxplot_{file_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        plt.tight_layout()
        plt.savefig(plot_file_path)
        plt.close()



def nested_data_dict(dfs_path, modelresults_dict, idlist_include_files: list = None):
    outer_dict = defaultdict(lambda: defaultdict(), modelresults_dict)
    
    model_results_folder = dfs_path / pv.Modelresults_folder_
    
    if model_results_folder.exists():
        for csv_file in model_results_folder.glob("*.csv"): #csv_file = results_k10_r2 etc. of 'descriptors only' for example. whole path
            if not csv_file.name.endswith("temp.csv") and "ModelResults_R2" in csv_file.name:
                print(csv_file)
                print('#########################################')
                try:
                        df = pd.read_csv(csv_file)
                        row_data_dict = modelresults_to_dict(df, idlist_include_files)
                        outer_dict[csv_file.name][dfs_path.name] = row_data_dict
                        
                        # Check if a CSV file with 'R2_train' exists in the same folder
                        r2_train_file = next((f for f in model_results_folder.glob("*.csv") if "R2_train" in f.name), None)
                        print(r2_train_file)
                        if r2_train_file:
                            df = pd.read_csv(r2_train_file)
                            row_data_dict = modelresults_to_dict(df, idlist_include_files)
                            outer_dict[csv_file.name][dfs_path.name+'_train'] = row_data_dict

                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
    return outer_dict

def modelresults_to_dict(modelresult_df, idlist_include_files: list = None):
    """
    """
    # If exclude_files is None, initialize it as an empty list
    if idlist_include_files is None:
        idlist_include_files = []
    
    # Filter the DataFrame to exclude rows based on the 'id' column if specified
    if idlist_include_files:
        modelresult_df = modelresult_df[modelresult_df['mol_id'].isin(idlist_include_files)]

    # Get the columns that start with 'split' and end with '_test_score'
    split_columns = [col for col in modelresult_df.columns if col.startswith('split') and col.endswith('_score')]

    # Convert the filtered DataFrame to a dictionary
    row_data_dict = modelresult_df.set_index('mol_id')[split_columns].T.to_dict('list')
    return row_data_dict

def boxplots_compare_individuals(master_folder, csv_filename, modelresults_dict):
    """
    """
    plt.figure(figsize=(14, 7))

    # Create a folder to save plots if it doesn't exist
    save_plot_folder = master_folder / Path('boxplots_compare_individual')
    save_plot_folder.mkdir(parents=True, exist_ok=True)
    
    # Define colors and labels for different subfolders
    colors = {
        pv.dfs_descriptors_only_path_.name: 'lightblue',
        pv.dfs_reduced_path_.name: 'lightgreen',
        pv.dfs_reduced_and_MD_path_.name: 'lightcoral',
        pv.dfs_MD_only_path_.name: 'lightgray',  # New color for "MD only"
        "PCA":"salmon",
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'mediumorchid'
    }

    labels = {
        pv.dfs_descriptors_only_path_.name: 'All RDkit 3D features',
        pv.dfs_reduced_path_.name: f'Reduced RDkit 3D features',
        pv.dfs_reduced_and_MD_path_.name: 'Reduced RDkit 3D features + MD features',
        pv.dfs_MD_only_path_.name: 'MD features only',  # Added label for "MD only"
        "PCA":"salmon",
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'x features'
    }

    filtered_colors = {key: colors[key] for key in modelresults_dict.keys() if key in colors}
    filtered_labels = {key: labels[key] for key in modelresults_dict.keys() if key in labels}

    # Define parameters
    box_width = 1
    group_gap = 0
    group_spacing = 1  # Increased spacing to separate different subgroups clearly
    border = 2

    
    positions = []
    box_data = []
    box_colors = []
    widths = []

    # Get the group with the most values directly
    group_with_most_values = max(modelresults_dict, key=lambda k: len(modelresults_dict[k]))

    # Get all subgroups from that group
    all_subgroups = list(modelresults_dict[group_with_most_values].keys())
    
    num_groups = len(filtered_colors)
    print(all_subgroups) #NOTE: correct order

    # Loop through each subgroup
    for subgroup_idx, subgroup in enumerate(all_subgroups):
        
        for group_idx, (group_name, color) in enumerate(filtered_colors.items()):
            # Check if the subgroup exists in the current group
            split_scores = modelresults_dict[group_name].get(subgroup, [])  # Use empty list if not found
            
            # Calculate the position for this boxplot
            pos = subgroup_idx * (num_groups * (box_width + group_gap) + group_spacing) + (group_idx * (box_width + group_gap))  # Adjust spacing
            
            positions.append(pos)
            box_data.append(split_scores)
            box_colors.append(color)
            widths.append(box_width)  # Width of each box
    # Create boxplots
    boxplot_dict = plt.boxplot(box_data, positions=positions, patch_artist=True, 
                                widths=widths,  # Set box widths
                                medianprops=dict(color='red'))
    
    # Apply colors to each box
    for patch, color in zip(boxplot_dict['boxes'], box_colors):
        patch.set_facecolor(color)
    
    # # Draw lines connecting individual scores within the same subgroup across different groups
    # for subgroup_idx in range(len(all_subgroups)):
    #     # Get scores for the current subgroup from all groups
    #     scores_per_group = []
    #     x_positions = []
        
    #     for group_idx in range(len(filtered_colors)):
    #         group_name = list(filtered_colors.keys())[group_idx]
    #         subgroup_name = all_subgroups[subgroup_idx]
    #         scores = modelresults_dict[group_name][subgroup_name]
    #         scores_per_group.append(scores)
    #         x_pos = positions[group_idx * len(all_subgroups) + subgroup_idx]
    #         x_positions.append(x_pos)

    #         # Check if the number of scores is consistent across groups
    #         if len(scores) != len(scores_per_group[0]):
    #             print(f"Warning: Inconsistent number of scores for subgroup '{subgroup_name}' in group '{group_name}'")

    #     # Draw lines connecting each pair of scores for this subgroup
    #     for i in range(len(scores_per_group[0])):  # Assuming scores_per_group has same length
    #         y_values = [scores_per_group[group_idx][i] for group_idx in range(len(filtered_colors))]
    #         plt.plot(x_positions, y_values, color='black', alpha=0.5, zorder=2)

    # Add a legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in filtered_colors.values()]
    plt.legend(handles=handles, labels=filtered_labels.values(), loc='best')

    plt.title(f"Boxplots for (Kfold=10 ; R-squared)")
    plt.xlabel("Subgroups")
    plt.ylabel("Scores")
    
    # Adjust x-axis limits
    plt.xlim(-(box_width/2) - border, positions[-1] + (box_width/2) + border)

    # Set xticks and labels
    tick_positions = [i * (num_groups * (box_width + group_gap)+ group_spacing) + (num_groups - 1) / 2 for i in range(len(all_subgroups))]
    plt.xticks(ticks=tick_positions, labels=all_subgroups, rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_file_path = save_plot_folder / f'{csv_filename}.png'
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()
    return


def boxplots_compare_groups(path, csv_filename, modelresults_dict):
    """
    """
    print(path)
    print(csv_filename)
    print(modelresults_dict)
    plt.figure()
    plt.figure(figsize=(12, 6))
    # Create a folder to save plots if it doesn't exist
    save_plot_folder = pv.dataframes_master_ / 'boxplots_compare_groups'
    save_plot_folder.mkdir(parents=True, exist_ok=True)
    print(save_plot_folder)
    # Define colors and labels for different subfolders
    colors = {
        pv.dfs_descriptors_only_path_.name: 'lightblue',
        pv.dfs_descriptors_only_path_.name+'_train': 'lightblue',
        pv.dfs_reduced_path_.name: 'lightgreen',
        pv.dfs_reduced_path_.name+'_train': 'lightgreen',
        pv.dfs_reduced_and_MD_path_.name: 'lightcoral',
        pv.dfs_reduced_and_MD_path_.name+'_train': 'lightcoral',

        pv.dfs_MD_only_path_.name: 'lightgray',  # New color for "MD only"

        "PCA": "salmon",
        "2D": 'goldenrod',
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'mediumorchid',

        # New keys with assigned colors
        "(DescMD)PCA_10": "darkorange",
        "(DescMD)PCA_20": "tomato",
        "desc_PCA10": "dodgerblue",
        "desc_PCA20": "deepskyblue",
        "DescPCA20 MDnewPCA": "darkviolet",
        "DescPCA20 MDnewPCA minus PC1": "orchid",
        "DescPCA20 MDnewPCA minus PCMD1": "plum",
        "MD_new only": "forestgreen",
        "MD_old only": 'lightgray',
        "MD_new only reduced": "mediumseagreen",
        "MD_old only reduced": "forestgreen",
        "MDnewPCA": "mediumseagreen",
        "red MD_new": "limegreen",
        "red MD_new reduced": "crimson",
        "red MD_old": 'lightcoral',
        "MD_new only3train": 'green',
        "MD_new onlytrain": 'red',
        "MD_new only3": 'green',
        "MD_new only": 'red',

        'MD_new onlyall': 'lightgray',
        'MD_new onlyall_train': 'lightgray',
        'desc_PCA15': 'tomato',
        'desc_PCA15_train': 'tomato',
        'dPCA MD_new': 'crimson',
        'dPCA MD_new_train': 'crimson',
        'red MD_new': 'forestgreen',
        'red MD_new_train': 'forestgreen',
    }

    labels = {
        "2D":"2D fingerprint",
        pv.dfs_descriptors_only_path_.name: 'RDkit Descriptors',
        pv.dfs_descriptors_only_path_.name+'_train': 'RDkit Descriptors train',

        pv.dfs_reduced_path_.name: f'RDkit Descriptors Red.',
        pv.dfs_reduced_path_.name+'_train': f'RDkit Descriptors Red. train',

        pv.dfs_reduced_and_MD_path_.name: 'RDkit Descriptors Red. + MD features',
        pv.dfs_MD_only_path_.name: 'MD features',  # Added label for "MD only"

        # pv.dfs_reduced_PCA_path_.name: 'PCA desc',
        # pv.dfs_reduced_MD_PCA_path_.name: f'PCA MD',
        # pv.dfs_reduced_and_MD_combined_PCA_path_.name: 'PCA desc + PCA MD',
        # pv.dfs_all_PCA.name: 'PCDA desc+MD',  # Added label for "MD only"
        # New keys with assigned colors
        "(DescMD)PCA_10": "(DescMD)PCA_10",
        "(DescMD)PCA_20": "(DescMD)PCA_20",
        "desc_PCA10": "desc_PCA10",
        "desc_PCA20": "desc_PCA20",
        "DescPCA20 MDnewPCA": "DescPCA20 MDnewPCA",
        "DescPCA20 MDnewPCA minus PC1": "DescPCA20 MDnewPCA minus PC1",
        "DescPCA20 MDnewPCA minus PCMD1": "DescPCA20 MDnewPCA minusPCMD1",
        "MD_new only": "MD_new only",
        "MD_old only": "MD_old only",
        "MD_new only reduced": "MD_new only reduced",
        "MD_old only reduced": "MD_old only reduced",
        "MDnewPCA": "MDnewPCA",
        "red MD_new": "red MD_new",
        "red MD_new reduced": "red MD_new reduced",
        "red MD_old": "red MD_old",
        "MD_new only3": 'MD_new only3',
        "MD_new only": 'MD_new only',
        "MD_new only3train": 'MD_new only3train',
        "MD_new onlytrain": 'MD_new onlytrain',



        'MD_new onlyall': 'MD_new onlyall',
        'MD_new onlyall_train': 'MD_new onlyall_train',
        'desc_PCA15': 'desc_PCA15',
        'desc_PCA15_train': 'desc_PCA15_train',
        'dPCA MD_new': 'dPCA MD_new',
        'dPCA MD_new_train': 'dPCA MD_new_train',
        'red MD_new': 'red MD_new',
        'red MD_new_train': 'red MD_new_train',

    }

    filtered_colors = {key: colors[key] for key in modelresults_dict.keys() if key in colors}
    filtered_labels = {key: labels[key] for key in modelresults_dict.keys() if key in labels}
    
    # Define parameters
    box_width = 3
    group_gap = 1
    group_spacing = 5  # Increased spacing to separate different subgroups clearly
    border = 2

    # Get the group with the most values directly
    group_with_most_values = max(modelresults_dict, key=lambda k: len(modelresults_dict[k]))

    # Get all subgroups from that group
    all_subgroups = list(modelresults_dict[group_with_most_values].keys())
    
    positions = []
    box_data = []
    box_colors = []
    widths = []
    base_position = 0

    min_value = float('inf')  # Set to positive infinity
    max_value = float('-inf')  # Set to negative infinity

    for group_idx, (group_name, color) in enumerate(filtered_colors.items()):
        num_rows = len(modelresults_dict[group_name].values())
        
        for row_idx, subgroup in enumerate(modelresults_dict[group_name].keys()):
            
            split_scores = modelresults_dict[group_name][subgroup]
            
             #using group_idx/num_rows will make it assume that all groups have same length!
            min_value = min(min_value, min(split_scores))
            max_value = max(max_value, max(split_scores))

            pos = base_position + row_idx * (box_width + group_gap)

            positions.append(pos)
            box_data.append(split_scores)
            box_colors.append(color)
            widths.append(box_width)
        base_position = base_position + ((num_rows * (box_width + group_gap) - group_gap) + group_spacing)
    
    # # Add individual data points
    # for pos, split_scores in zip(positions, box_data):
    #     plt.scatter([pos] * len(split_scores), split_scores, color='black', alpha=0.8, s=10, zorder=3)  # Smaller, darker dots

    # # Draw lines connecting individual scores within the same subgroup across different groups
    # for subgroup_idx in range(len(all_subgroups)):
    #     # Get scores for the current subgroup from all groups
    #     scores_per_group = []
    #     x_positions = []
        
    #     for group_idx in range(len(filtered_colors)):
    #         group_name = list(filtered_colors.keys())[group_idx]
    #         subgroup_name = all_subgroups[subgroup_idx]
    #         scores = modelresults_dict[group_name][subgroup_name]
    #         scores_per_group.append(scores)
    #         x_pos = positions[group_idx * len(all_subgroups) + subgroup_idx]
    #         x_positions.append(x_pos)

    #         # Check if the number of scores is consistent across groups
    #         if len(scores) != len(scores_per_group[0]):
    #             print(f"Warning: Inconsistent number of scores for subgroup '{subgroup_name}' in group '{group_name}'")

    #     # Draw lines connecting each pair of scores for this subgroup
    #     for i in range(len(scores_per_group[0])):  # Assuming scores_per_group has same length
    #         y_values = [scores_per_group[group_idx][i] for group_idx in range(len(filtered_colors))]
    #         plt.plot(x_positions, y_values, color='black', alpha=0.5, zorder=2)
    # print(all_subgroups)
    # for group_idx, group_name in enumerate(filtered_colors.keys()):
    #     print('group_idx')
    #     print(group_idx)

    #     y_values = []
    #     x_positions = []

    #     for subgroup_idx in range(len(all_subgroups)):
    #         subgroup_name = all_subgroups[subgroup_idx]
    #         scores = modelresults_dict[group_name][subgroup_name]
    #         y_values.append(scores)
    #         x_pos = positions[group_idx * len(all_subgroups) + subgroup_idx]
    #         x_positions.append(x_pos)

    #     # Draw lines connecting each pair of scores for this subgroup within all groups
    #     for i in range(len(y_values[0])):  # Assuming all y_values have the same length
    #         plt.plot(x_positions, [y[i] for y in y_values], color='black', alpha=0.5, zorder=2)
    boxplot_dict = plt.boxplot(box_data, positions=positions, patch_artist=True,
                                widths=widths,  # Set box widths
                                medianprops=dict(color='red'))
    
    # Apply colors to each box
    for patch, color in zip(boxplot_dict['boxes'], box_colors):
        patch.set_facecolor(color)
    

    # Round min_value down to the nearest number with 1 decimal place
    min_value = (math.floor(min_value * 10) / 10) - 0.1

    # Round max_value up to the nearest number with 1 decimal place
    max_value = (math.ceil(max_value * 10) / 10)+0.1
    # Add a legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in filtered_colors.values()]
    plt.legend(handles=handles, labels=filtered_labels.values(), loc='best')

    plt.title(f"{pv.ML_MODEL} Boxplot results for Kfold=10 using {pv.DESCRIPTOR} 3D descriptors") #
    plt.xlabel("conformations group")
    plt.ylabel("R²-score")
    
    # Adjust x-axis limits
    plt.xlim(-(box_width/2) - border, positions[-1] + (box_width/2) + border)
    # Set y-axis limits between 0.4 and 0.9
    plt.ylim(min_value, max_value)
    
    # Set xticks and labels
    xtick_labels = [
    'minimized' if label == '0ns' else '10 conf.' if label == 'conformations_10' else label
    for group in modelresults_dict.values() for label in group.keys()
    ]
    print(xtick_labels)
    plt.xticks(ticks=positions, labels=xtick_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_file_path = save_plot_folder / f'{pv.ML_MODEL}_{csv_filename}_{len(modelresults_dict)}_train.png'
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()
    return

 ####################################################################################################################

def main(dfs_paths = [pv.dfs_descriptors_only_path_]):

    master_folder = pv.dataframes_master_

    all_modelresults_dict = {}

    for dfs_path, list_to_include in dfs_paths:
        # print(dfs_entry)
        # Check if the entry is a tuple (path, list) or just a path
        # if isinstance(dfs_entry, tuple):
        #     dfs_path, idlist_to_exclude = dfs_entry  # Unpack the tuple
        #     # print(dfs_path)
        #     # print(f"Processing path {dfs_path.name} with CSV list to exclude: {idlist_to_exclude}")
        # else:
        #     dfs_path = dfs_entry  # Only a path is given, no CSV list
        #     idlist_to_exclude = None  # Set CSV list to None if not provided
        #     # print(f"Processing path {dfs_path.name} with no specific CSV list to exclude")

        # if not dfs_path.exists():
        #     # print(f"Error: The path '{dfs_path}' does not exist.")
        #     continue
        print(all_modelresults_dict)
        all_modelresults_dict = nested_data_dict(dfs_path, all_modelresults_dict, list_to_include)
        print(all_modelresults_dict)
        # print('test')
        # print(all_modelresults_dict_test)
    for csvfile_name, modelresults_dict in all_modelresults_dict.items(): #loop over k10_r2 etc.
        # boxplots_compare_individuals(master_folder, csvfile_name, modelresults_dict)
        boxplots_compare_groups(path=pv.dataframes_master_, csv_filename=csvfile_name, modelresults_dict=modelresults_dict)
    return

if __name__ == "__main__":

    dfs_paths = []
    
    exclude_stable2 = ['stable_conformations']
    exclude_stable = ['stable_conformations','conformations_10','conformations_20','minimized_conformations_10']
    include_files=['0ns','1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','conformations_50']
    include_files=['0ns','1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','c10']
    include_files=['0ns','1ns', 'c10']

    # dfs_paths.append((public_variables.dataframes_master_ / '2D', []))
    # dfs_paths.append((public_variables.dfs_descriptors_only_path_, exclude_files_clusters + exclude_stable))
    # dfs_paths.append((public_variables.dfs_reduced_path_, exclude_files_clusters+ exclude_stable))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, exclude_files_clusters+ exclude_stable))
    # dfs_paths.append((public_variables.dfs_MD_only_path_, exclude_files_clusters + exclude_stable))#['conformations_11','1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','conformations_100'])) #['2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','conformations_50','conformations_100']
    # to_exclude = exclude_files_clusters50 + exclude_files_clusters10+ exclude_files_clusters30+ exclude_files_clusters40 + list(set(exclude_files_other) - set([]))
    # to_exclude = exclude_files_clusters40 + exclude_files_clusters20+ exclude_files_clusters10+ exclude_files_clusters50 + exclude_files_other
    # dfs_paths.append((public_variables.dfs_descriptors_only_path_, to_exclude))
    # dfs_paths.append((public_variables.dfs_reduced_path_, to_exclude))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, to_exclude))
    # dfs_paths.append((public_variables.dfs_MD_only_path_, to_exclude))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, exclude_files_clusters40 + exclude_files_clusters20+ exclude_files_clusters50+ exclude_files_clusters30 + exclude_files_other))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, exclude_files_clusters40 + exclude_files_clusters10+ exclude_files_clusters50+ exclude_files_clusters30 + exclude_files_other))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, exclude_files_clusters40 + exclude_files_clusters20+ exclude_files_clusters50+ exclude_files_clusters30 + exclude_files_other))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, exclude_files_clusters30 + exclude_files_clusters20+ exclude_files_clusters50+ exclude_files_clusters40 + exclude_files_other))
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    dfs_paths.append((public_variables.dfs_descriptors_only_path_, include_files))
    dfs_paths.append((public_variables.dfs_reduced_path_, include_files))
    dfs_paths.append((pv.dataframes_master_ / 'desc_PCA15', include_files))
    dfs_paths.append((pv.dataframes_master_ / 'red MD_new', include_files))
    dfs_paths.append((pv.dataframes_master_ / 'dPCA MD_new', include_files))
    dfs_paths.append((pv.dataframes_master_ / 'MD_new onlyall', include_files))







    # 'MD_new onlyall': 'MD_new onlyall',
    # 'MD_new onlyall_train': 'MD_new onlyall_train',
    # 'desc_PCA15': 'desc_PCA15',
    # 'desc_PCA15_train': 'desc_PCA15_train',
    # 'dPCA MD_new': 'dPCA MD_new',
    # 'dPCA MD_new_train': 'dPCA MD_new_train',
    # 'red MD_new': 'red MD_new',
    # 'red MD_new_train': 'red MD_new_train',

    # include_files=['0ns','1ns','3ns','5ns','conformations_10']
    # for path in pv.get_paths():
    # dfs_paths.append((pv.dfs_descriptors_only_path_, include_files))
    # dfs_paths.append((pv.dfs_reduced_path_, include_files))
    # dfs_paths.append((pv.dataframes_master_ / 'red MD_old', include_files))
    # dfs_paths.append((pv.dataframes_master_ / 'MD_old only', include_files))

    # dfs_paths.append((pv.dataframes_master_ / 'MD_new only', include_files))

    # dfs_paths.append((pv.dataframes_master_ / 'red MD_new', include_files))
    # dfs_paths.append((pv.dataframes_master_ / 'red MD_new reduced', include_files))
    # dfs_paths.append((pv.dataframes_master_ / 'DescPCA20 MDnewPCA', include_files))
    # dfs_paths.append((pv.dataframes_master_ / '(DescMD)PCA_20', include_files))



        # dfs_paths.append((path, include_files))
        # dfs_paths.append((pv.dfs_reduced_PCA_path_, include_files))
        # dfs_paths.append((pv.dfs_reduced_MD_PCA_path_, include_files))
        # dfs_paths.append((pv.dfs_reduced_and_MD_combined_PCA_path_, include_files))
        # dfs_paths.append((pv.dfs_all_PCA, include_files))

    # pv.update_config(model_=Model_deep.LSTM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # include_files=['0ns','1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','minimized_conformations_10','conformations_50','conformations_100']
    # for path in pv.get_paths():
    #     dfs_paths.append((path, include_files))
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
        
    # dfs_paths.append((pv.dfs_descriptors_only_path_, include_files))
    # dfs_paths.append((pv.dfs_reduced_path_, include_files))
    # dfs_paths.append((pv.dfs_reduced_and_MD_path_, include_files))
    # dfs_paths.append((pv.dfs_MD_only_path_, include_files))
    
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, exclude_files_clusters40 + exclude_files_clusters20+ exclude_files_clusters40+ exclude_files_clusters30 + exclude_files_other))
    
    
    
    # dfs_paths.append((public_variables.dfs_reduced_path_, []))
    # # dfs_paths.append((public_variables.dfs_PCA_path, []))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, []))
    # dfs_paths.append((public_variables.dfs_MD_only_path_, []))#['conformations_11','1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','conformations_100'])) #['rdkit_min','0ns'] ['1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns']
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, ['minimized']))
    # dfs_paths.append((public_variables.dfs_MD_only_path_, ['minimized']))

    # dfs_paths.append((public_variables.dataframes_master_ / 'custom_dataframes',['rdkit_min','0ns']))
    # dfs_paths.append((public_variables.dataframes_master_ / 'reduced_t0.75',['rdkit_min','0ns']))
    # dfs_paths.append((public_variables.dataframes_master_ / 'descriptors only scaled mw',['rdkit_min','0ns']))
    print('dfs path wordt nu geprint')
    print(dfs_paths)
    main(dfs_paths)
