from sklearn.model_selection import GridSearchCV, KFold

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor

from global_files import csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np


from Models.Classic_models import create_classic_models_nested
from Models.DNN import create_DNN_models_10fold_final
import pandas as pd
import math
import re
import os

def main(dfs_path = pv.dfs_descriptors_only_path_):
     
    return

if __name__ == "__main__":
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)

    create_classic_models_nested.main(pv.dfs_descriptors_only_path_)
    create_DNN_models_10fold_final.main(pv.dfs_descriptors_only_path_)
    # main(pv.dfs_reduced_path_)
    # main(pv.dfs_reduced_and_MD_path_)
    # main(pv.dfs_MD_only_path_)


    