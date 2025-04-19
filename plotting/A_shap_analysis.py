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
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def try_shap(dfs_path = pv.dfs_descriptors_only_path_,  include_files = []):
    # Specify the fold you're interested in
    outer_fold = 1  # Change to the desired fold number

    # Define the base folder path for this fold
    shap_base_path = dfs_path / pv.Modelresults_folder_ / 'shap_info' / 'c10' / f'fold{outer_fold}' #create Modelresults folder

    # Load the model, SHAP values, and X_test
    model_path = shap_base_path / f'model_fold{outer_fold}.pkl'
    shap_values_path = shap_base_path / f'shap_values_fold{outer_fold}.pkl'
    X_test_path = shap_base_path / f'X_test_fold{outer_fold}.pkl'

    # Load the files
    model = joblib.load(model_path)
    shap_values = joblib.load(shap_values_path)
    X_test = joblib.load(X_test_path)

    # Check if the files are loaded correctly
    print("Model loaded:", model)
    print("SHAP values loaded:", shap_values)
    print("X_test data loaded:", X_test.shape)

    # If the model is a tree-based model (RF, XGB), use TreeExplainer
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Plot the SHAP summary plot
        shap.summary_plot(shap_values, X_test, plot_type="bar")  # Bar plot for global feature importance

        # Optionally, you can also do a SHAP dependence plot for a specific feature (e.g., feature 0)
        shap.dependence_plot(0, shap_values, X_test)

    # Show the plots
    plt.show()
    return

def main():
    include_files = ['c10']
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    try_shap(pv.dfs_descriptors_only_path_,include_files = include_files)

if __name__ == "__main__":
    main()
