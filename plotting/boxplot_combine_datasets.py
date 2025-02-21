import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from collections import defaultdict
import math
from scipy.stats import gaussian_kde
from matplotlib.collections import PolyCollection

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



def nested_data_dict(dfs_path, modelresults_dict, idlist_exclude_files: list = None):
    outer_dict = defaultdict(lambda: defaultdict(), modelresults_dict)
    
    model_results_folder = dfs_path / pv.Modelresults_folder_
    
    if model_results_folder.exists():
        for csv_file in model_results_folder.glob("*.csv"): #csv_file = results_k10_r2 etc. of 'descriptors only' for example. whole path
            print(csv_file)
            print('#########################################')
            try:
                    df = pd.read_csv(csv_file)
                    row_data_dict = modelresults_to_dict(df, idlist_exclude_files)
                    outer_dict[csv_file.name][dfs_path.name] = row_data_dict
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    return outer_dict

def modelresults_to_dict(modelresult_df, idlist_exclude_files: list = None):
    """
    """
    # If exclude_files is None, initialize it as an empty list
    if idlist_exclude_files is None:
        idlist_exclude_files = []
    
    # Filter the DataFrame to exclude rows based on the 'id' column if specified
    if idlist_exclude_files:
        modelresult_df = modelresult_df[~modelresult_df['mol_id'].isin(idlist_exclude_files)]
    
    # Get the columns that start with 'split' and end with '_test_score'
    split_columns = [col for col in modelresult_df.columns if col.startswith('split')]

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


def boxplots_compare_groups(modelresults_dict, save_path, datasets): #master_folder, csv_filename, modelresults_dict): #
    """
    """
    plt.figure()
    plt.figure(figsize=(12, 6))
    # Create a folder to save plots if it doesn't exist
    # save_plot_folder = master_folder / Path('boxplots_compare_groups')
    save_plot_folder = save_path / Path('boxplots_compare_groups')
    save_plot_folder.mkdir(parents=True, exist_ok=True)
    
    # Define colors and labels for different subfolders
    colors = {
        pv.dfs_descriptors_only_path_.name: 'lightblue',
        pv.dfs_reduced_path_.name: 'lightgreen',
        pv.dfs_reduced_and_MD_path_.name: 'lightcoral',
        pv.dfs_MD_only_path_.name: 'lightgray',  # New color for "MD only"
        "PCA":"salmon",
        "2D":'goldenrod',
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'mediumorchid'
    }

    labels = {
        "2D":"2D fingerprint",
        pv.dfs_descriptors_only_path_.name: 'All RDkit 3D features',
        pv.dfs_reduced_path_.name: f'Reduced RDkit 3D features',
        pv.dfs_reduced_and_MD_path_.name: 'Reduced RDkit 3D features + MD features',
        pv.dfs_MD_only_path_.name: 'MD features only',  # Added label for "MD only"
        "PCA":"PCA",
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'x features'
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
        print(group_idx)
        print(group_name)
        print(color)
        num_rows = len(modelresults_dict[group_name].values())
        
        for row_idx, subgroup in enumerate(modelresults_dict[group_name].keys()):

            split_scores = modelresults_dict[group_name][subgroup]
            
            min_value = min(min_value, min(split_scores))
            max_value = max(max_value, max(split_scores))

            #using group_idx/num_rows will make it assume that all groups have same length!

            pos = base_position + row_idx * (box_width + group_gap)

            positions.append(pos)
            box_data.append(split_scores)
            box_colors.append(color)
            widths.append(box_width)
        base_position = base_position + ((num_rows * (box_width + group_gap) - group_gap) + group_spacing)

    boxplot_dict = plt.boxplot(box_data, positions=positions, patch_artist=True,
                                widths=widths,  # Set box widths
                                medianprops=dict(color='red'))
    
    # Apply colors to each box
    for patch, color in zip(boxplot_dict['boxes'], box_colors):
        patch.set_facecolor(color)
    
    # Round min_value down to the nearest number with 1 decimal place
    min_value = math.floor(min_value * 10) / 10

    # Round max_value up to the nearest number with 1 decimal place
    max_value = math.ceil(max_value * 10) / 10

    # Add a legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in filtered_colors.values()]
    plt.legend(handles=handles, labels=filtered_labels.values(), loc='best')

    plt.title(f"{pv.MLmodel_} Boxplot results for Kfold=10 using {pv.Descriptor_} 3D descriptors") #
    plt.xlabel("conformations group")
    plt.ylabel("R²-score")
    
    # Adjust x-axis limits
    plt.xlim(-(box_width/2) - border, positions[-1] + (box_width/2) + border)
    # Set y-axis limits between 0.4 and 0.9
    plt.ylim(min_value, max_value)
    # Set xticks and labels
    xtick_labels = [subgroup for group in modelresults_dict.values() for subgroup in group.keys()]

    plt.xticks(ticks=positions, labels=xtick_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_file_path = save_plot_folder / f'boxplot_ko10_ki5_r2_{pv.Descriptor_}_{datasets}_len{row_idx+1}_groups{group_idx+1}.png' #row_idx is number of subgroups-1
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()
    return

def boxplots_compare_groups(modelresults_dict, save_path, datasets): #master_folder, csv_filename, modelresults_dict): #
    """
    """
    plt.figure()
    plt.figure(figsize=(12, 6))
    # Create a folder to save plots if it doesn't exist
    # save_plot_folder = master_folder / Path('boxplots_compare_groups')
    save_plot_folder = save_path / Path('boxplots_compare_groups')
    save_plot_folder.mkdir(parents=True, exist_ok=True)
    
    # Define colors and labels for different subfolders
    colors = {
        pv.dfs_descriptors_only_path_.name: 'lightblue',
        pv.dfs_reduced_path_.name: 'lightgreen',
        pv.dfs_reduced_and_MD_path_.name: 'lightcoral',
        pv.dfs_MD_only_path_.name: 'lightgray',  # New color for "MD only"
        "PCA":"salmon",
        "2D":'goldenrod',
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'mediumorchid'
    }

    labels = {
        "2D":"2D fingerprint",
        pv.dfs_descriptors_only_path_.name: 'All RDkit 3D features',
        pv.dfs_reduced_path_.name: f'Reduced RDkit 3D features',
        pv.dfs_reduced_and_MD_path_.name: 'Reduced RDkit 3D features + MD features',
        pv.dfs_MD_only_path_.name: 'MD features only',  # Added label for "MD only"
        "PCA":"PCA",
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'x features'
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
        print(group_idx)
        print(group_name)
        print(color)
        num_rows = len(modelresults_dict[group_name].values())
        
        for row_idx, subgroup in enumerate(modelresults_dict[group_name].keys()):

            split_scores = modelresults_dict[group_name][subgroup]
            
            min_value = min(min_value, min(split_scores))
            max_value = max(max_value, max(split_scores))

            #using group_idx/num_rows will make it assume that all groups have same length!

            pos = base_position + row_idx * (box_width + group_gap)

            positions.append(pos)
            box_data.append(split_scores)
            box_colors.append(color)
            widths.append(box_width)
        base_position = base_position + ((num_rows * (box_width + group_gap) - group_gap) + group_spacing)

    boxplot_dict = plt.boxplot(box_data, positions=positions, patch_artist=True,
                                widths=widths,  # Set box widths
                                medianprops=dict(color='red'))
    
    # Apply colors to each box
    for patch, color in zip(boxplot_dict['boxes'], box_colors):
        patch.set_facecolor(color)
    
    # Round min_value down to the nearest number with 1 decimal place
    min_value = (math.floor(min_value * 10) / 10)

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
    xtick_labels = [subgroup for group in modelresults_dict.values() for subgroup in group.keys()]

    plt.xticks(ticks=positions, labels=xtick_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_file_path = save_plot_folder / f'boxplot_ko10_ki5_r2_{pv.DESCRIPTOR}_{datasets}_len{row_idx+1}_groups{group_idx+1}.png' #row_idx is number of subgroups-1
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()
    return


def density_plots_compare_groups(modelresults_dict, save_path, datasets): #master_folder, csv_filename, modelresults_dict): #
    """
    """
    plt.figure()
    plt.figure(figsize=(12, 6))
    # Create a folder to save plots if it doesn't exist
    # save_plot_folder = master_folder / Path('boxplots_compare_groups')
    save_plot_folder = save_path / Path('boxplots_compare_groups')
    save_plot_folder.mkdir(parents=True, exist_ok=True)
    
    # Define colors and labels for different subfolders
    colors = {
        pv.dfs_descriptors_only_path_.name: 'lightblue',
        pv.dfs_reduced_path_.name: 'lightgreen',
        pv.dfs_reduced_and_MD_path_.name: 'lightcoral',
        pv.dfs_MD_only_path_.name: 'lightgray',  # New color for "MD only"
        "PCA":"salmon",
        "2D":'goldenrod',
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'mediumorchid'
    }

    labels = {
        "2D":"2D fingerprint",
        pv.dfs_descriptors_only_path_.name: 'All RDkit 3D features',
        pv.dfs_reduced_path_.name: f'Reduced RDkit 3D features',
        pv.dfs_reduced_and_MD_path_.name: 'Reduced RDkit 3D features + MD features',
        pv.dfs_MD_only_path_.name: 'MD features only',  # Added label for "MD only"
        "PCA":"PCA",
        'descriptors only scaled mw': 'salmon',
        "reduced_t0.9": 'teal',
        "reduced_t0.75": 'goldenrod',
        "custom_dataframes": 'x features'
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
        print(group_idx)
        print(group_name)
        print(color)
        num_rows = len(modelresults_dict[group_name].values())
        
        for row_idx, subgroup in enumerate(modelresults_dict[group_name].keys()):

            split_scores = modelresults_dict[group_name][subgroup]
            
            min_value = min(min_value, min(split_scores))
            max_value = max(max_value, max(split_scores))

            #using group_idx/num_rows will make it assume that all groups have same length!

            pos = base_position + row_idx * (box_width + group_gap)

            positions.append(pos)
            box_data.append(split_scores)
            box_colors.append(color)
            widths.append(box_width)
        base_position = base_position + ((num_rows * (box_width + group_gap) - group_gap) + group_spacing)

    # boxplot_dict = plt.boxplot(box_data, positions=positions, patch_artist=True,
    #                             widths=widths,  # Set box widths
    #                             medianprops=dict(color='red'))
    
    # # Apply colors to each box
    # for patch, color in zip(boxplot_dict['boxes'], box_colors):
    #     patch.set_facecolor(color)
    # Create a single violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.figure(figsize=(12, 6))
    violin_parts = plt.violinplot(box_data, positions=positions, widths=widths, showmeans=False, showextrema=False, showmedians=True)

    # Customize the appearance
    # colors = ['green', 'pink', 'cyan']  # Colors for each violin
    
    # print(dir(violin_parts))
    # print(type(violin_parts['bodies']))
    # print(dir(violin_parts['bodies']))

    for i, pc in enumerate(violin_parts['bodies']):
        print(type(pc))
        if isinstance(pc, PolyCollection):
            print("true")
            pc.set_facecolor(box_colors[i])  # Set the facecolor to the desired color
        # pc.color(colors[i])  # Set fill color
        # pc.set_edgecolor('black')    # Set edge color
        # pc.set_alpha(0.7)            # Transparency
    # for i, pc in enumerate(violin_parts['bodies']):
    #     if isinstance(pc, Polygon):
    #         pc.set_face
    # Add medians (red dots)
    medians = [np.median(data) for data in box_data]
    for i, pos in enumerate(positions):
        ax.scatter(pos, medians[i], color='red', zorder=3, label='Median' if i == 0 else "")
    # Round min_value down to the nearest number with 1 decimal place
    min_value = math.floor(min_value * 10) / 10

    # Round max_value up to the nearest number with 1 decimal place
    max_value = math.ceil(max_value * 10) / 10

    # # Add a legend
    # handles = [plt.Line2D([0], [0], color=color, lw=4) for color in filtered_colors.values()]
    # plt.legend(handles=handles, labels=filtered_labels.values(), loc='best')

    plt.title(f"{pv.MLmodel_} Boxplot results for Kfold=10 using {pv.Descriptor_} 3D descriptors") #
    plt.xlabel("conformations group")
    plt.ylabel("R²-score")
    
    # Adjust x-axis limits
    plt.xlim(-(box_width/2) - border, positions[-1] + (box_width/2) + border)
    # Set y-axis limits between 0.4 and 0.9
    plt.ylim(min_value, max_value)
    # Set xticks and labels
    xtick_labels = [subgroup for group in modelresults_dict.values() for subgroup in group.keys()]

    plt.xticks(ticks=positions, labels=xtick_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_file_path = save_plot_folder / f'densityboxplot_ko10_ki5_r2_{pv.Descriptor_}_{datasets}_len{row_idx+1}_groups{group_idx+1}.png' #row_idx is number of subgroups-1
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()
    return

####################################################################################################################

def save_combined_df(dataframe, folder, datasets) -> Path:
    save_path = pv.base_path_ / "Afstuderencode" / "plots" / "Modelresults_Combined" / f"{pv.ML_MODEL}"  # = repositories on laptop

    save_path = pv.base_path_ / 'code' / 'plots' / "Modelresults_Combined" / f"{pv.ML_MODEL}"
    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    final_save_path = save_path / f"combined_df_{folder}_{datasets}.csv"
    dataframe.to_csv(final_save_path, index=False)
    #to_csv
    #the location
    return save_path

####################################################################################################################


def main(dfs_paths = [pv.dfs_descriptors_only_path_]):
    dfs_paths = []
    pv.update_config(model_=Model_deep.DNN)
    files_to_include = ['minimized_conformation','0ns','1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20', 'minimized_conformations_10']
    
    datasets_protein = ['JAK1', 'GSK3', 'pparD']
    descriptor_groups = ['descriptors only', f'reduced_t{pv.correlation_threshold_}',f'reduced_t{pv.correlation_threshold_}_MD', 'MD only']

    group_results_dic = {}

    for descriptor in descriptor_groups:
        print(descriptor)
        combined_df = None
        for protein in DatasetProtein:
            df_path = pv.base_path_ / 'dataframes' /  f'dataframes_{protein}_WHIM' / descriptor  /  f'ModelResults_{pv.ML_MODEL}' / 'results_Ko10_Ki5_r2_WHIM.csv'
            df  = pd.read_csv(df_path)
            # Filter only rows where mol_id is in mol_id_list
            df = df[df['mol_id'].isin(files_to_include)]
            
            
            # Filter only 'mol_id' and 'split*' columns
            df = df[['mol_id'] + [col for col in df.columns if col.startswith('split')]]
            # Reset index to preserve row order
            df = df.reset_index(drop=True)
            # Rename 'split*' columns to include the dataset name
            df = df.rename(columns={col: f"{col}_{protein}" for col in df.columns if col != 'mol_id'})

            combined_df = combined_df.merge(df, on='mol_id', how='inner') if combined_df is not None else df

            
        print(combined_df)
        save_csv_path = save_combined_df(combined_df, descriptor, datasets_protein)
        print(save_csv_path)
        group_result_dic = modelresults_to_dict(combined_df, idlist_exclude_files=[]) #{'minimized_conformation': [20 values], '0ns': [20 values], etc}

        group_results_dic[descriptor] = group_result_dic
    
    boxplots_compare_groups(group_results_dic, save_csv_path, datasets_protein)
    # density_plots_compare_groups(group_results_dic, save_csv_path, datasets_protein)

    # for csvfile_name, modelresults_dict in outer_dict.items(): #loop over k10_r2 etc.
    #     # boxplots_compare_individuals(master_folder, csvfile_name, modelresults_dict)
    #     # print('ja')
    #     boxplots_compare_groups(public_variables.base_path_ / 'Afstuderencode' / f'dataframes_{dataset}_{public_variables.Descriptor_}', csvfile_name, modelresults_dict)
        # all_modelresults_dict = nested_data_dict(public_variables.base_path_ / 'AFSTUDERENCODE' / f'dataframes_{dataset}_{public_variables.Descriptor_}' / folder, modelresults_dict)
        # print(all_modelresults_dict)

        # modelresults_dict[folder] = [combined_df]
    # for csvfile_name, modelresults_dict in all_modelresults_dict.items(): #loop over k10_r2 etc.
    #     # boxplots_compare_individuals(master_folder, csvfile_name, modelresults_dict)
    #     print('ja')
    #     boxplots_compare_groups(public_variables.base_path_ / 'AFSTUDERENCODE' / f'dataframes_{dataset}_{public_variables.Descriptor_}', csvfile_name, modelresults_dict)
        #place the combined_dfs in a dictionary. this then goes to boxplots_compare_groups
        
    # boxplots_compare_groups(modelresults_dict, save_path)
    return

if __name__ == "__main__":
    main()



    # print(public_variables.base_path_ / f'dataframes_{datasets[0]}_{public_variables.Descriptor_}' / folders[0] / f'Modelresults_{public_variables.MLmodel_}')
    #read this csv
    #extract the correct line that we get the 3 dataframes in one. pd.merge or pd.join
    #read out this pandas dataframe with the mol_id the same
    #which can then be plotted
    #keep dataframe since i want to keep the mol_id unique for plotting.
    #then quickly done for each
    #











    # exclude_files_clusters = [
    # 'clustering_target10.0%_cluster1', 'clustering_target10.0%_cluster2.csv',
    # 'clustering_target10.0%_cluster3', 'clustering_target10.0%_cluster4.csv',
    # 'clustering_target10.0%_cluster5', 'clustering_target10.0%_cluster6.csv',
    # 'clustering_target10.0%_cluster7', 'clustering_target10.0%_cluster8.csv',
    # 'clustering_target10.0%_cluster9', 'clustering_target10.0%_cluster10.csv',
    # 'clustering_target20.0%_cluster1.csv', 'clustering_target20.0%_cluster2.csv',
    # 'clustering_target20.0%_cluster3.csv', 'clustering_target20.0%_cluster4.csv',
    # 'clustering_target20.0%_cluster5.csv', 'clustering_target20.0%_cluster6.csv',
    # 'clustering_target20.0%_cluster7.csv', 'clustering_target20.0%_cluster8.csv',
    # 'clustering_target20.0%_cluster9.csv', 'clustering_target20.0%_cluster10.csv',
    # 'clustering_target30.0%_cluster1.csv', 'clustering_target30.0%_cluster2.csv',
    # 'clustering_target30.0%_cluster3.csv', 'clustering_target30.0%_cluster4.csv',
    # 'clustering_target30.0%_cluster5.csv', 'clustering_target30.0%_cluster6.csv',
    # 'clustering_target30.0%_cluster7.csv', 'clustering_target30.0%_cluster8.csv',
    # 'clustering_target30.0%_cluster9.csv', 'clustering_target30.0%_cluster10.csv',
    # 'clustering_target40.0%_cluster1.csv', 'clustering_target40.0%_cluster2.csv',
    # 'clustering_target40.0%_cluster3.csv', 'clustering_target40.0%_cluster4.csv',
    # 'clustering_target40.0%_cluster5.csv', 'clustering_target40.0%_cluster6.csv',
    # 'clustering_target40.0%_cluster7.csv', 'clustering_target40.0%_cluster8.csv',
    # 'clustering_target40.0%_cluster9.csv', 'clustering_target40.0%_cluster10.csv',
    # 'clustering_target50.0%_cluster1.csv', 'clustering_target50.0%_cluster2.csv',
    # 'clustering_target50.0%_cluster3.csv', 'clustering_target50.0%_cluster4.csv',
    # 'clustering_target50.0%_cluster5.csv', 'clustering_target50.0%_cluster6.csv',
    # 'clustering_target50.0%_cluster7.csv', 'clustering_target50.0%_cluster8.csv',
    # 'clustering_target50.0%_cluster9.csv', 'clustering_target50.0%_cluster10.csv'
    # ]
    # exclude_files_clusters10 = [
    # 'clustering_target10.0%_cluster1.csv', 'clustering_target10.0%_cluster2.csv',
    # 'clustering_target10.0%_cluster3.csv', 'clustering_target10.0%_cluster4.csv',
    # 'clustering_target10.0%_cluster5.csv', 'clustering_target10.0%_cluster6.csv',
    # 'clustering_target10.0%_cluster7.csv', 'clustering_target10.0%_cluster8.csv',
    # 'clustering_target10.0%_cluster9.csv', 'clustering_target10.0%_cluster10.csv']
    # exclude_files_clusters20 = [
    # 'clustering_target20.0%_cluster1.csv', 'clustering_target20.0%_cluster2.csv',
    # 'clustering_target20.0%_cluster3.csv', 'clustering_target20.0%_cluster4.csv',
    # 'clustering_target20.0%_cluster5.csv', 'clustering_target20.0%_cluster6.csv',
    # 'clustering_target20.0%_cluster7.csv', 'clustering_target20.0%_cluster8.csv',
    # 'clustering_target20.0%_cluster9.csv', 'clustering_target20.0%_cluster10.csv']
    # exclude_files_clusters30 = [
    # 'clustering_target30.0%_cluster1.csv', 'clustering_target30.0%_cluster2.csv',
    # 'clustering_target30.0%_cluster3.csv', 'clustering_target30.0%_cluster4.csv',
    # 'clustering_target30.0%_cluster5.csv', 'clustering_target30.0%_cluster6.csv',
    # 'clustering_target30.0%_cluster7.csv', 'clustering_target30.0%_cluster8.csv',
    # 'clustering_target30.0%_cluster9.csv', 'clustering_target30.0%_cluster10.csv']
    # exclude_files_clusters40 = [
    # 'clustering_target40.0%_cluster1.csv', 'clustering_target40.0%_cluster2.csv',
    # 'clustering_target40.0%_cluster3.csv', 'clustering_target40.0%_cluster4.csv',
    # 'clustering_target40.0%_cluster5.csv', 'clustering_target40.0%_cluster6.csv',
    # 'clustering_target40.0%_cluster7.csv', 'clustering_target40.0%_cluster8.csv',
    # 'clustering_target40.0%_cluster9.csv', 'clustering_target40.0%_cluster10.csv']
    # exclude_files_clusters50 = [
    # 'clustering_target50.0%_cluster1.csv', 'clustering_target50.0%_cluster2.csv',
    # 'clustering_target50.0%_cluster3.csv', 'clustering_target50.0%_cluster4.csv',
    # 'clustering_target50.0%_cluster5.csv', 'clustering_target50.0%_cluster6.csv',
    # 'clustering_target50.0%_cluster7.csv', 'clustering_target50.0%_cluster8.csv',
    # 'clustering_target50.0%_cluster9.csv', 'clustering_target50.0%_cluster10.csv'
    # ]
    # exclude_files_other = ['0ns.csv','1ns.csv','2ns.csv','3ns.csv','4ns.csv','5ns.csv','6ns.csv','7ns.csv','8ns.csv','9ns.csv','10ns.csv','conformations_1000.csv','conformations_1000_molid.csv','conformations_500.csv','conformations_200.csv','conformations_100.csv','conformations_50.csv','initial_dataframe.csv','initial_dataframes_best.csv','MD_output.csv','conformations_10.csv','conformations_20.csv','minimized_conformations_10.csv','stable_conformations.csv']
    always_exclude = [
    '10.0%_cluster6', '10.0%_cluster7', '10.0%_cluster8',
    '10.0%_cluster9', '10.0%_cluster10', '20.0%_cluster6',
    '20.0%_cluster7', '20.0%_cluster8', '20.0%_cluster9', '20.0%_cluster10',
    '30.0%_cluster6', '30.0%_cluster7', '30.0%_cluster8',
    '30.0%_cluster9', '30.0%_cluster10','40.0%_cluster6',
    '40.0%_cluster7', '40.0%_cluster8', '40.0%_cluster9', '40.0%_cluster10',
    '50.0%_cluster6', '50.0%_cluster7', '50.0%_cluster8',
    '50.0%_cluster9', '50.0%_cluster10'
    ]
    exclude_files_clusters = [
    '10.0%_cluster1', '10.0%_cluster2', '10.0%_cluster3', '10.0%_cluster4',
    '10.0%_cluster5', '10.0%_cluster6', '10.0%_cluster7', '10.0%_cluster8',
    '10.0%_cluster9', '10.0%_cluster10', '20.0%_cluster1', '20.0%_cluster2',
    '20.0%_cluster3', '20.0%_cluster4', '20.0%_cluster5', '20.0%_cluster6',
    '20.0%_cluster7', '20.0%_cluster8', '20.0%_cluster9', '20.0%_cluster10',
    '30.0%_cluster1', '30.0%_cluster2', '30.0%_cluster3', '30.0%_cluster4',
    '30.0%_cluster5', '30.0%_cluster6', '30.0%_cluster7', '30.0%_cluster8',
    '30.0%_cluster9', '30.0%_cluster10', '40.0%_cluster1', '40.0%_cluster2',
    '40.0%_cluster3', '40.0%_cluster4', '40.0%_cluster5', '40.0%_cluster6',
    '40.0%_cluster7', '40.0%_cluster8', '40.0%_cluster9', '40.0%_cluster10',
    '50.0%_cluster1', '50.0%_cluster2', '50.0%_cluster3', '50.0%_cluster4',
    '50.0%_cluster5', '50.0%_cluster6', '50.0%_cluster7', '50.0%_cluster8',
    '50.0%_cluster9', '50.0%_cluster10'
    ]
    exclude_files_clusters10 = [
        '10.0%_cluster1', '10.0%_cluster2', '10.0%_cluster3', '10.0%_cluster4',
        '10.0%_cluster5', '10.0%_cluster6', '10.0%_cluster7', '10.0%_cluster8',
        '10.0%_cluster9', '10.0%_cluster10'
    ]
    exclude_files_clusters20 = [
        '20.0%_cluster1', '20.0%_cluster2', '20.0%_cluster3', '20.0%_cluster4',
        '20.0%_cluster5', '20.0%_cluster6', '20.0%_cluster7', '20.0%_cluster8',
        '20.0%_cluster9', '20.0%_cluster10'
    ]
    exclude_files_clusters30 = [
        '30.0%_cluster1', '30.0%_cluster2', '30.0%_cluster3', '30.0%_cluster4',
        '30.0%_cluster5', '30.0%_cluster6', '30.0%_cluster7', '30.0%_cluster8',
        '30.0%_cluster9', '30.0%_cluster10'
    ]
    exclude_files_clusters40 = [
        '40.0%_cluster1', '40.0%_cluster2', '40.0%_cluster3', '40.0%_cluster4',
        '40.0%_cluster5', '40.0%_cluster6', '40.0%_cluster7', '40.0%_cluster8',
        '40.0%_cluster9', '40.0%_cluster10'
    ]
    exclude_files_clusters50 = [
        '50.0%_cluster1', '50.0%_cluster2', '50.0%_cluster3', '50.0%_cluster4',
        '50.0%_cluster5', '50.0%_cluster6', '50.0%_cluster7', '50.0%_cluster8',
        '50.0%_cluster9', '50.0%_cluster10'
    ]
    exclude_files_other = [
        '0ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns',
        'conformations_1000', 'conformations_1000_molid', 'conformations_500', 
        'conformations_200', 'conformations_100', 'conformations_50', 
        'initial_dataframe', 'initial_dataframes_best', 'MD_output', 
        'conformations_10', 'conformations_20', 'minimized_conformations_10', 
        'stable_conformations'
    ]


    
    # dpath = public_variables.base_path_ / Path('dataframes_JAK1_WHIM') 
    # ddpath = dpath / Path('descriptors only')
    # print(ddpath)
    # print(public_variables.dfs_descriptors_only_path_)
    # # drpath = dpath / Path('reduced_t0.65')
    # # drmpath = dpath / Path('reduced_t0.65_MD')
    # # dmpath = dpath / Path('MD only')
    # dfs_paths.append((ddpath, ['2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','conformations_50','conformations_100']))
    # # dfs_paths.append((drpath, ['2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','conformations_50','conformations_100']))
    # # dfs_paths.append((drmpath, ['minimized','conformations_10','conformations_20','conformations_50','conformations_100']))
    # # dfs_paths.append((dmpath, ['minimized','conformations_10','conformations_20','conformations_50','conformations_100']))
    exclude_stable2 = ['stable_conformations']
    exclude_stable = ['stable_conformations','conformations_10','conformations_20','minimized_conformations_10']
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




    # dfs_paths.append((public_variables.dfs_descriptors_only_path_, exclude_files_clusters + exclude_stable2))
    # dfs_paths.append((public_variables.dfs_reduced_path_, exclude_files_clusters+ exclude_stable2))
    # dfs_paths.append((public_variables.dfs_reduced_and_MD_path_, exclude_files_clusters+ exclude_stable2))
    # dfs_paths.append((public_variables.dfs_MD_only_path_, exclude_files_clusters+ exclude_stable2))
    
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
    # print('dfs path wordt nu geprint')
    # print(dfs_paths)
    
