import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from global_files import public_variables
from collections import defaultdict
import math
import re

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



def nested_data_dict(dfs_path1, dfs_path2_min, modelresults_dict, idlist_include_files: list = None):
    outer_dict = defaultdict(lambda: defaultdict(), modelresults_dict)
    
    model_results_folder1 = dfs_path1 / pv.Modelresults_folder_
    model_results_folder2_min = dfs_path2_min / pv.Modelresults_folder_

    if model_results_folder1.exists() and model_results_folder2_min:
        for csv_file in model_results_folder1.glob("*.csv"): #csv_file = results_k10_r2 etc. of 'descriptors only' for example. whole path
            csv_file2_min = model_results_folder2_min / csv_file.name
            if csv_file2_min.exists():
                print(model_results_folder2_min / csv_file.name)
                if not csv_file.name.endswith("temp.csv") and "R2" in csv_file.name and not "train" in csv_file.name:
                    print(csv_file)
                    print('#########################################')
                    # try:
                    df1 = pd.read_csv(csv_file)
                    df2 = pd.read_csv(csv_file2_min)
                    print(df1)
                    row_data_dict = modelresults_to_dict(df1, df2, idlist_include_files)
                    print(f'{dfs_path1.name}__{dfs_path2_min.name}')
                    outer_dict[csv_file.name][f'{dfs_path1.name}__{dfs_path2_min.name}'] = row_data_dict
                            
                    # except Exception as e:
                    #     print(f"Error reading {csv_file}: {e}")
    return outer_dict

def modelresults_to_dict(df1,df2_min, idlist_include_files: list = None):
    """
    """
    # If exclude_files is None, initialize it as an empty list
    if idlist_include_files is None:
        idlist_include_files = []
    
    # Filter the DataFrame to exclude rows based on the 'id' column if specified
    if idlist_include_files:
        df1 = df1[df1['mol_id'].isin(idlist_include_files)]
        df2_min = df2_min[df2_min['mol_id'].isin(idlist_include_files)]

    # Ensure both DataFrames have the same mol_id order
    df1 = df1.set_index('mol_id')
    df2_min = df2_min.set_index('mol_id')

    print(df1)
    print(df2_min)
    # Identify common test score columns
    test_score_columns = [col for col in df1.columns if 'split' in col and 'test_score' in col]
    common_columns = list(set(test_score_columns) & set(df2_min.columns))
    # Sort the values in each row for both DataFrames
    sorted_values_df1 = df1[common_columns].apply(lambda row: sorted(row, reverse=True), axis=1)
    sorted_values_df2 = df2_min[common_columns].apply(lambda row: sorted(row, reverse=True), axis=1)
    print(sorted_values_df1)
    print(sorted_values_df2)

    # Row-wise subtraction
    # Subtract the sorted values element-wise and create a DataFrame
    # Check if '2D' exists in df2_min and handle the subtraction accordingly
    if '2D' in sorted_values_df2.index:
        # Get the row with index '2D' from df2_min
        row_2D = sorted_values_df2.loc['2D']
        print(row_2D)
        # Create a DataFrame where the '2D' row is repeated across all rows of sorted_values_df1
        repeated_2D = pd.Series([row_2D] * len(sorted_values_df1), index=sorted_values_df1.index)
        print(repeated_2D)
        # Subtract the '2D' row from each row in sorted_values_df1 using combine
        result = sorted_values_df1.combine(
            repeated_2D, 
            lambda s1, s2: [a - b for a, b in zip(s1, s2)], 
            fill_value=0
        )
    else:
        # If '2D' doesn't exist, proceed with the usual element-wise subtraction
        result = sorted_values_df1.combine(
            sorted_values_df2, 
            lambda s1, s2: [a - b for a, b in zip(s1, s2)], 
            fill_value=0
        )
    # Convert the result into a dictionary
    result_dict = {}
    for index, row in result.items():  # Using .items() to iterate over Series
        result_dict[index] = row

    # Print the dictionary for debugging purposes
    print(result_dict)


    # Return the dictionary
    return result_dict

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
        pv.dfs_reduced_path_.name: 'lightgreen',
        pv.dfs_reduced_and_MD_path_.name: 'lightcoral',
        pv.dfs_MD_only_path_.name: 'lightgray',  # New color for "MD only"
        pv.dfs_dPCA_MD_path_.name: 'orchid',
        f'{pv.dfs_reduced_and_MD_path_.name}__{pv.dfs_2D_path.name}': 'goldenrod',
        f'{pv.dfs_reduced_and_MD_path_.name}__{pv.dfs_descriptors_only_path_.name}': 'lightcoral',
        f'{pv.dfs_MD_only_path_.name}__{pv.dfs_descriptors_only_path_.name}': 'lightgray',
        f'{pv.dfs_dPCA_MD_path_.name}__{pv.dfs_descriptors_only_path_.name}': 'orchid',
    }

    labels = {
        "2D":"2D fingerprint",
        pv.dfs_descriptors_only_path_.name: 'RDkit Descriptors',
        pv.dfs_reduced_path_.name: f'RDkit Descriptors Red.',
        pv.dfs_reduced_and_MD_path_.name: 'RDkit Descriptors Red. + MD features',
        pv.dfs_MD_only_path_.name: 'MD features',  # Added label for "MD only"
        pv.dfs_dPCA_MD_path_.name: 'desc PCA + MD',
        f'{pv.dfs_reduced_and_MD_path_.name}__{pv.dfs_2D_path.name}': 'difference red_MD - WHIM',
        f'{pv.dfs_reduced_and_MD_path_.name}__{pv.dfs_descriptors_only_path_.name}': 'difference red_MD - WHIM',
        f'{pv.dfs_MD_only_path_.name}__{pv.dfs_descriptors_only_path_.name}': 'difference MD - WHIM',
        f'{pv.dfs_dPCA_MD_path_.name}__{pv.dfs_descriptors_only_path_.name}': 'difference dPCA_MD - WHIM',
        
    }

    filtered_colors = {key: colors[key] for key in modelresults_dict.keys() if key in colors}
    filtered_labels = {key: labels[key] for key in modelresults_dict.keys() if key in labels}
    
    box_width = 3
    group_gap = 1
    group_spacing = 5  # Increased spacing to separate different subgroups clearly
    border = 2
    
    base_position = 0
    group_spacing = 3  # Space between big groups
    bar_width = 0.6  # Width of bars within a group
    positions = []
    means = []
    stds = []
    bar_colors = []
    xtick_labels = []
    
    for group_idx, (group_name, color) in enumerate(filtered_colors.items()):
        subgroups = list(modelresults_dict[group_name].keys())
        num_subgroups = len(subgroups)
        
        for row_idx, subgroup in enumerate(subgroups):
            values = modelresults_dict[group_name][subgroup]
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            pos = base_position + row_idx * (bar_width + 0.2)  # Small gap within group
            positions.append(pos)
            means.append(mean_value)
            stds.append(std_value)
            bar_colors.append(color)
            # Check if 'subgroup' starts with 'CLt' and contains an integer at the end
            if subgroup.startswith('CLt'):
                # Use regular expression to find the last integer after 'CLt'
                match = re.search(r'CLt(\d+)', subgroup)
                if match:
                    integer_value = match.group(1)
                    xtick_labels.append(f'{integer_value} clusters conf. ')
                else:
                    xtick_labels.append('10 clusters conf.')  # If no integer is found, use default name
            else:
                # For other cases, keep the original logic
                xtick_labels.append(
                    'minimized' if subgroup == '0ns' else 
                    '10 conf.' if subgroup == 'c10' else 
                    subgroup
                )
        
        base_position += num_subgroups * (bar_width + 0.2) + group_spacing  # Space between groups
    
    # Create the legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in filtered_colors.values()]
    plt.legend(handles=handles, labels=filtered_labels.values(), loc='best')

    # Create the bar plot
    plt.bar(positions, means, yerr=stds, capsize=5, color=bar_colors, width=bar_width, edgecolor='black')
    plt.ylim(-0.4, 0.4)
    # Set y-ticks at the desired intervals (e.g., 0.2, 0.3, 0.4, etc.)
    y_ticks = np.arange(-0.4, 0.4, 0.1)  # Adjust the range and interval as needed
    plt.yticks(y_ticks)

    # Add a thick line at y=0
    plt.axhline(y=0, color='black', linewidth=1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set the x-ticks
    plt.xticks(ticks=positions, labels=xtick_labels, rotation=45, ha='right')

    # Set axis labels and title
    plt.ylabel("R²-score Difference")
    plt.title(f"{pv.ML_MODEL} Barplot - Difference between groups")
    plt.xlabel("Conformations Group")

    # Save the plot
    plot_file_path = save_plot_folder / f'{pv.ML_MODEL}_DIFF_{len(modelresults_dict)}.png'
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()

    return

 ####################################################################################################################

def main(dfs_paths = [pv.dfs_descriptors_only_path_]):

    master_folder = pv.dataframes_master_

    all_modelresults_dict = {}

    for dfs_path1, dfs_path2_min, list_to_include in dfs_paths:
        print(dfs_path1)
        print(dfs_path2_min)

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
        all_modelresults_dict = nested_data_dict(dfs_path1, dfs_path2_min,all_modelresults_dict, list_to_include)
        print(all_modelresults_dict)
        print('test')
        print(all_modelresults_dict)
    for csvfile_name, modelresults_dict in all_modelresults_dict.items(): #loop over k10_r2 etc.
        # boxplots_compare_individuals(master_folder, csvfile_name, modelresults_dict)
        print(csvfile_name) #ModelResults_R2_WHIM.csv
        boxplots_compare_groups(path=pv.dataframes_master_, csv_filename=csvfile_name, modelresults_dict=modelresults_dict)
    return

if __name__ == "__main__":

    dfs_paths = []
    
    exclude_stable2 = ['stable_conformations']
    exclude_stable = ['stable_conformations','conformations_10','conformations_20','minimized_conformations_10']
    include_files=['0ns','1ns','2ns','3ns','4ns','5ns','6ns','7ns','8ns','9ns','10ns','conformations_10','conformations_20','conformations_50']
    include_files=['2D','0ns','1ns','2ns','c10','CLt50_cl10x_c10']

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
    for model in Model_classic:
        dfs_paths = []
        pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set='big')
        dfs_paths.append((pv.dfs_reduced_and_MD_path_, pv.dfs_2D_path, include_files))
        dfs_paths.append((pv.dfs_reduced_and_MD_path_, pv.dfs_descriptors_only_path_, include_files))
        dfs_paths.append((pv.dfs_MD_only_path_, pv.dfs_descriptors_only_path_, include_files))
        dfs_paths.append((pv.dfs_dPCA_MD_path_, pv.dfs_descriptors_only_path_, include_files))

        main(dfs_paths)

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
