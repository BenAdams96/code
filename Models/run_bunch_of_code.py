from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

from global_files import dataframe_processing, csv_to_dictionary, public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from plotting import A_true_vs_pred_plotting
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import List
from typing import Dict
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os

from Models.D2 import create_2d_models_nested
from Models.Classic_models import create_classic_models_nested
from Models.deep_learning_models import deeplearning_create_models_small

def main():  ###set as default, but can always change it to something else.
    

    return

if __name__ == "__main__":
    #2D part
    # hpset = ['small', 'big']
    # for set in hpset:
    #     for model in Model_classic:
    #         pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1, hyperparameter_set=set)
    #         create_2d_models_nested.main(pv.dfs_2D_path)
    #         pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=set)
    #         create_2d_models_nested.main(pv.dfs_2D_path)
    #         pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD, hyperparameter_set=set)
    #         create_2d_models_nested.main(pv.dfs_2D_path)
    #         pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4, hyperparameter_set=set)
    #         create_2d_models_nested.main(pv.dfs_2D_path)


    #classic part test
    # include_files = [0,'c10','xCLt100_cl10_c10']
    # pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1, hyperparameter_set='big')
    # create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)

    # set = 'small'
    # for model in Model_classic:
    #     include_files = [0,'xCLt100_cl10_c10']
    #     pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1, hyperparameter_set=set)
    #     create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_MD_path_,include_files = include_files)
    #classic part
    set = 'big'

    
    # include_files = [0,2,4,6,8,10,'c10','xCLt50_cl10_c10'] #'ta10c10'
    # pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=set)
    # create_classic_models_nested.main(pv.dfs_descriptors_only_path_,include_files = include_files)
    # create_classic_models_nested.main(pv.dfs_reduced_path_,include_files = include_files)
    # create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
    # create_classic_models_nested.main(pv.dfs_MD_only_path_,include_files = include_files)
    # create_classic_models_nested.main(pv.dfs_dPCA_path_,include_files = include_files)
    # create_classic_models_nested.main(pv.dfs_dPCA_MD_path_,include_files = include_files)

    include_files = [0,2,4,6,8,10,'c10'] # 'ta10c10'
    pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD, hyperparameter_set=set)
    create_classic_models_nested.main(pv.dfs_descriptors_only_path_,include_files = include_files)
    create_classic_models_nested.main(pv.dfs_reduced_path_,include_files = include_files)
    create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
    create_classic_models_nested.main(pv.dfs_MD_only_path_,include_files = include_files)
    create_classic_models_nested.main(pv.dfs_dPCA_path_,include_files = include_files)
    create_classic_models_nested.main(pv.dfs_dPCA_MD_path_,include_files = include_files)

    # include_files = [0,1,2,3,'c10']
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4)
    # deeplearning_create_models_small.main(pv.dfs_descriptors_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_and_MD_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_MD_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_MD_path_,random_splitting = False,include_files = include_files)


    # include_files = [0,2,4,6,8,10,'c10','xCLt100_cl10_c10']
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # deeplearning_create_models_small.main(pv.dfs_descriptors_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_and_MD_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_MD_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_MD_path_,random_splitting = False,include_files = include_files)
    
    # include_files = [0,2,4,6,8,10,'c10','xCLt50_cl10_c10']
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # deeplearning_create_models_small.main(pv.dfs_descriptors_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_and_MD_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_MD_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_MD_path_,random_splitting = False,include_files = include_files)

    # include_files = [0,2,4,6,8,10,'c10']
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # deeplearning_create_models_small.main(pv.dfs_descriptors_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_reduced_and_MD_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_MD_only_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_path_,random_splitting = False,include_files = include_files)
    # deeplearning_create_models_small.main(pv.dfs_dPCA_MD_path_,random_splitting = False,include_files = include_files)

    

    # for model in Model_classic:
    #     print(model)
    #     include_files = ['ta10c10'] #'ta10c10'
    #     pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1, hyperparameter_set=set)
    #     create_classic_models_nested.main(pv.dfs_descriptors_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_MD_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_MD_path_,include_files = include_files)

    #     include_files = ['ta10c10'] #'ta10c10'
    #     pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3, hyperparameter_set=set)
    #     create_classic_models_nested.main(pv.dfs_descriptors_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_MD_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_MD_path_,include_files = include_files)

    #     include_files = ['ta10c10'] # 'ta10c10'
    #     pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD, hyperparameter_set=set)
    #     create_classic_models_nested.main(pv.dfs_descriptors_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_MD_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_MD_path_,include_files = include_files)

    #     include_files = ['ta4c10'] # 'ta4c10' 
    #     pv.update_config(model_=model, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4, hyperparameter_set=set)
    #     create_classic_models_nested.main(pv.dfs_descriptors_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_reduced_and_MD_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_MD_only_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_path_,include_files = include_files)
    #     create_classic_models_nested.main(pv.dfs_dPCA_MD_path_,include_files = include_files)

    
    # for model in Model_classic:
    hpset = ['small', 'big']
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1, hyperparameter_set='small')
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_deep.DNN, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4)
    # main(pv.dfs_2D_path)
    


    # include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20','c50','c100','ta10c10','CLt100_cl10_c10'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(include)

    # include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20','c50','c100','ta10c10','CLt50_cl10_c10'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(include)

    # include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20','c50','c100','ta10c10'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(include)

    # include = [0,1,2,3,4,'c10','c20','c50','c100','ta4c10'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4)
    # main(include)
    # for protein in DatasetProtein:
    #     pv.update_config(model_=Model_classic.SVM, descriptor_=Descriptor.WHIM, protein_=protein, hyperparameter_set='big')
    #     main(pv.dfs_2D_path)
        # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=protein, hp_set='small')
        # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main(pv.dfs_2D_path)
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(pv.dfs_2D_path)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(pv.dfs_2D_path)

